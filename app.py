"""VoxCheck web demo.   Run:  python app.py   ->  http://127.0.0.1:5000   (python app.py --port 5050 for another port)"""
import logging
import os
import re
import socket
import sys
import tempfile
import uuid

from flask import Flask, jsonify, render_template, request

from voxcheck import audio as A
from voxcheck.detector import Detector
from voxcheck.meta import supported

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("voxcheck.app")

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(ROOT, "dataset")
SAMPLES = os.path.join(ROOT, "static", "samples")
AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".webm", ".m4a", ".opus")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB
app.config["TEMPLATES_AUTO_RELOAD"] = True

try:
    detector = Detector()
    load_error = None
except Exception as e:  # noqa: BLE001
    detector, load_error = None, str(e)


def self_test():
    """Run one sample through the whole pipeline at startup, exactly like a user request
    (temp file -> decode -> model). Returns None if OK, else a readable reason."""
    if detector is None:
        return "model not loaded: " + str(load_error)
    sample = next((os.path.join(SAMPLES, f) for f in sorted(os.listdir(SAMPLES))
                   if f.lower().endswith(AUDIO_EXT)), None) if os.path.isdir(SAMPLES) else None
    if sample is None:
        return None
    fd, tmp = tempfile.mkstemp(suffix=os.path.splitext(sample)[1], prefix="vox_selftest_")
    os.close(fd)
    try:
        with open(sample, "rb") as src, open(tmp, "wb") as dst:
            dst.write(src.read())
        detector.analyze(tmp)
        return None
    except Exception as e:  # noqa: BLE001
        log.exception("self-test failed")
        return f"{type(e).__name__}: {e}"
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


health_error = self_test()


def _read_sentences():
    p = os.path.join(ROOT, "data", "sentences.txt")
    return [s.strip() for s in open(p, encoding="utf-8") if s.strip()] if os.path.exists(p) else []



class Upload:
    """Saves the uploaded file to a temp path and always deletes it afterwards."""

    def __enter__(self):
        f = request.files.get("audio")
        if f is None or not f.filename:
            raise ValueError("No audio received.")
        ext = os.path.splitext(f.filename)[1].lower()
        ext = ext if re.fullmatch(r"\.[a-z0-9]{1,5}", ext) else ".bin"
        fd, self.path = tempfile.mkstemp(suffix=ext, prefix="vox_")
        os.close(fd)
        f.save(self.path)
        if os.path.getsize(self.path) == 0:
            raise ValueError("Uploaded file is empty.")
        return self.path

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except (OSError, AttributeError):
            pass


@app.route("/")
def index():
    samples = sorted(f for f in os.listdir(SAMPLES) if f.lower().endswith(AUDIO_EXT)) if os.path.isdir(SAMPLES) else []
    return render_template("index.html", meta=detector.meta if detector else None,
                           sentences=_read_sentences(), samples=samples)


@app.route("/analyze", methods=["POST"])
def analyze():
    if detector is None:
        return jsonify({"error": "Model not loaded: " + load_error}), 503
    try:
        with Upload() as path:
            return jsonify(detector.analyze(path))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:  # noqa: BLE001
        app.logger.exception("analyze failed")
        return jsonify({"error": "Could not process this file."}), 500


@app.route("/collect", methods=["POST"])
def collect():
    """Save a labelled recording into dataset/ (the team's own data collection tool)."""
    label = request.form.get("label", "human")
    speaker = re.sub(r"[^\w-]", "", request.form.get("speaker", "").strip().lower().replace(" ", "-"))
    speaker = speaker.replace("__", "_")[:30]
    if label not in ("human", "ai"):
        return jsonify({"error": "Unknown label."}), 400
    if not speaker:
        return jsonify({"error": "Enter a speaker name first."}), 400
    try:
        with Upload() as path:
            y = A.load_audio(path)
            if len(A.clean(y)) < A.MIN_SPEECH_S * A.SR:
                raise ValueError("Not enough speech in this recording.")
        out_dir = os.path.join(DATASET, label)
        os.makedirs(out_dir, exist_ok=True)
        name = f"me-{speaker}__{uuid.uuid4().hex[:8]}.wav"
        A.save_wav(os.path.join(out_dir, name), y)
        n = sum(1 for f in os.listdir(out_dir) if f.startswith(f"me-{speaker}__"))
        return jsonify({"saved": f"dataset/{label}/{name}", "count_for_speaker": n})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/feedback", methods=["POST"])
def feedback():
    """'The answer was wrong' -> keep the analysed recording with the CORRECT label for the next training round."""
    label = request.form.get("label")
    if label not in ("human", "ai"):
        return jsonify({"error": "Unknown label."}), 400
    try:
        with Upload() as path:
            y = A.load_audio(path)
            if len(A.clean(y)) < A.MIN_SPEECH_S * A.SR:
                raise ValueError("Not enough speech in this recording.")
        out_dir = os.path.join(DATASET, label)
        os.makedirs(out_dir, exist_ok=True)
        name = f"me-feedback__feedback-{uuid.uuid4().hex[:8]}.wav"
        A.save_wav(os.path.join(out_dir, name), y)
        n = sum(1 for d in ("human", "ai") for f in os.listdir(os.path.join(DATASET, d)) if "__feedback-" in f)
        return jsonify({"saved": f"dataset/{label}/{name}", "total_feedback": n})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/stats")
def stats():
    out = {}
    for label in ("human", "ai"):
        d = os.path.join(DATASET, label)
        files = [f for f in os.listdir(d) if f.lower().endswith(AUDIO_EXT) and supported(f)] if os.path.isdir(d) else []
        groups = {os.path.splitext(f)[0].split("__")[0] for f in files}
        out[f"{label}_files"] = len(files)
        out["human_speakers" if label == "human" else "ai_voices"] = len(groups)
    return jsonify(out)


@app.route("/about")
def about():
    """Experiment table + data summary for the 'How it works' tab."""
    import csv
    res = os.path.join(ROOT, "results")
    exps = []
    p = os.path.join(res, "experiments.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            exps = list(csv.DictReader(f))
    data = {}
    p = os.path.join(ROOT, "data", "processed", "manifest.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
        for lab, name in (("0", "human"), ("1", "ai")):
            rs = [r for r in rows if r["label"] == lab]
            data[name] = {"files": len(rs), "groups": len({r["group"] for r in rs}),
                          "minutes": round(sum(float(r["clean_s"] or 0) for r in rs) / 60, 1)}
    import json
    bpath = os.path.join(res, "breakdown.json")
    breakdown = json.load(open(bpath)) if os.path.exists(bpath) else None
    ppath = os.path.join(ROOT, "models", "playback_meta.json")
    playback = json.load(open(ppath)) if os.path.exists(ppath) else None
    hpath = os.path.join(res, "holdout_comparison.json")
    holdout = json.load(open(hpath)) if os.path.exists(hpath) else None
    return jsonify({"experiments": exps, "data": data, "breakdown": breakdown, "playback": playback, "holdout": holdout,
                    "selected": detector.meta.get("id") if detector else None,
                    "roc": os.path.exists(os.path.join(res, "roc_best.png")),
                    "prep": os.path.exists(os.path.join(res, "preprocessing_example.png"))})


@app.route("/results/<name>")
def result_image(name):
    from flask import abort, send_from_directory
    if name not in ("roc_best.png", "preprocessing_example.png"):
        abort(404)
    return send_from_directory(os.path.join(ROOT, "results"), name, max_age=0)


@app.route("/health")
def health():
    """Quick check that this server can really analyse audio (used by the page on load)."""
    return jsonify({"ok": detector is not None and health_error is None, "pid": os.getpid(),
                    "model": detector.meta.get("approach") if detector else None,
                    "error": load_error if detector is None else health_error})


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File too large (max 25 MB)."}), 413


def port_owner(port):
    """PID that is listening on 127.0.0.1:<port>, or None if the port is free.
    On Windows several programs can share one port, and requests then go to the OLD one -
    so we refuse to start a second VoxCheck instead of silently sharing."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        s.bind(("127.0.0.1", port))
        return None
    except OSError:
        pass
    finally:
        s.close()
    try:
        import subprocess
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[1].endswith(f":{port}") and parts[3] == "LISTENING":
                return parts[4]
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


if __name__ == "__main__":
    import argparse
    import werkzeug.serving
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("VOX_PORT", 5000)))
    port = ap.parse_args().port

    owner = port_owner(port)
    if owner:
        print(f"\n !! Port {port} is already in use (process {owner}) - probably an older VoxCheck window.\n"
              f"    Close that window, or stop it with:   Stop-Process -Id {owner}\n"
              f"    or start this one on another port:    python app.py --port {port + 1}\n")
        sys.exit(1)
    # don't let another program bind the same port later and steal our requests
    werkzeug.serving.BaseWSGIServer.allow_reuse_address = False
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        _bind = werkzeug.serving.BaseWSGIServer.server_bind

        def _exclusive_bind(self):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            _bind(self)
        werkzeug.serving.BaseWSGIServer.server_bind = _exclusive_bind

    print(" * Model:", detector.meta["approach"] if detector else "NOT LOADED - " + str(load_error))
    print(" * Self-test:", "OK - audio decoding and model work" if health_error is None else "FAILED - " + health_error)
    print(f" * Open  http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
