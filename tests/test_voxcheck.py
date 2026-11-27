"""Smoke tests for the pipeline and the web app.   Run:  python -m pytest -q tests
(or without pytest:  python tests/test_voxcheck.py)"""
import io
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from voxcheck import audio as A  # noqa: E402


def _wav_bytes(y, sr=A.SR):
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _tone(seconds, sr=A.SR):
    t = np.arange(int(seconds * sr)) / sr
    # amplitude-modulated harmonics ~ crude voiced speech, so silence removal keeps it
    y = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((140, 280, 420, 560), 1))
    return (0.2 * y * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def test_shapes():
    w = A.prepare(_tone(5))
    assert w.shape[1] == int(A.WIN_S * A.SR) and len(w) == 4
    assert A.logmel(w[0]).shape == (A.N_MELS, 201)
    assert A.handcrafted(w[0]).shape == (99,)


def test_short_and_silent_rejected():
    for y in (np.zeros(A.SR * 3, np.float32), _tone(0.4)):
        try:
            A.prepare(y)
            raise AssertionError("should have raised")
        except ValueError:
            pass


def test_decode_resamples_any_rate():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.wav")
        open(p, "wb").write(_wav_bytes(_tone(2, 44100), 44100))
        y = A.load_audio(p)
        assert abs(len(y) - 2 * A.SR) < 200


def test_augment_keeps_length_roughly():
    y = _tone(3)
    z = A.augment(y, np.random.default_rng(1))
    assert abs(len(z) - len(y)) < 0.1 * A.SR and np.isfinite(z).all()


def test_meta_parsing():
    from voxcheck.meta import attack_of, group_of, lang_of
    cases = [  # file, label, group, lang, attack
        ("ls-1272__1272-128104-0000.flac", 0, "ls-1272", "en", "human"),
        ("fl-uz-s1591__4967621184651376579.wav", 0, "fl-uz-s1591", "uz", "human"),
        ("me-gulbara__a1b2c3d4.wav", 0, "me-gulbara", "team", "human"),
        ("edge-ru-ru-dmitry__ky-0003.mp3", 1, "edge-ru-ru-dmitry", "ky", "edge"),
        ("edge-en-us-guy__0001.mp3", 1, "edge-en-us-guy", "en", "edge"),
        ("fl-ru-s20__vcgender-9f3e.wav", 1, "fl-ru-s20", "ru", "vcgender"),
        ("fl-tr-s77__xtts-tr-0004.wav", 1, "fl-tr-s77", "tr", "xtts"),
        ("gtts-uz__uz-0001.mp3", 1, "gtts-uz", "uz", "gtts"),
        ("piper-ru-ru-irina-medium__ru-0001.wav", 1, "piper-ru-ru-irina-medium", "ru", "piper"),
        ("sapi-irina__0001.wav", 1, "sapi-irina", "ru", "sapi"),
    ]
    for f, lab, g, lg, att in cases:
        assert (group_of(f), lang_of(f), attack_of(f, lab)) == (g, lg, att), f


def test_replay_and_spec_augment():
    import torch
    from voxcheck.model import spec_augment
    z = A.replay(_tone(3), np.random.default_rng(0))
    assert len(z) == 3 * A.SR and np.isfinite(z).all()
    x = torch.ones(4, 1, A.N_MELS, 201)
    m = spec_augment(x)
    assert m.shape == x.shape and (m == 0).any() and (m == 1).any()


def test_app_endpoints():
    from app import app
    c = app.test_client()
    assert c.get("/").status_code == 200
    assert c.get("/stats").status_code == 200
    assert c.get("/about").status_code == 200
    assert c.get("/results/../app.py").status_code == 404
    r = c.post("/analyze", data={"audio": (io.BytesIO(_wav_bytes(_tone(4))), "t.wav")},
               content_type="multipart/form-data")
    assert r.status_code in (200, 503), r.get_json()
    if r.status_code == 200:
        d = r.get_json()
        assert 0 <= d["ai"] <= 100 and abs(d["ai"] + d["human"] - 100) < 0.2 and d["timeline"]
    bad = c.post("/analyze", data={"audio": (io.BytesIO(b"not audio"), "x.mp3")},
                 content_type="multipart/form-data")
    assert bad.status_code == 400
    silent = c.post("/analyze", data={"audio": (io.BytesIO(_wav_bytes(np.zeros(A.SR * 3, np.float32))), "s.wav")},
                    content_type="multipart/form-data")
    assert silent.status_code in (400, 503)
    assert c.post("/analyze").status_code in (400, 503)
    assert c.post("/collect", data={"speaker": "", "label": "human"}).status_code == 400
    assert c.post("/collect", data={"speaker": "x", "label": "robot"}).status_code == 400


def test_decode_falls_back_when_ffmpeg_breaks():
    """If ffmpeg is missing or broken, wav/mp3 must still decode (libsndfile fallback)."""
    orig = A.ffmpeg_bin
    A.ffmpeg_bin = lambda: os.path.join(ROOT, "no-such-ffmpeg.exe")
    try:
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "x.wav")
            open(p, "wb").write(_wav_bytes(_tone(2, 44100), 44100))
            assert abs(len(A.load_audio(p)) - 2 * A.SR) < 200
            mp3 = os.path.join(ROOT, "static", "samples", "ai-svetlana-neural.mp3")
            if os.path.exists(mp3):
                assert len(A.load_audio(mp3)) > A.SR
    finally:
        A.ffmpeg_bin = orig


def test_browser_wav_recording_decodes():
    """The page records 16-bit mono WAV at the sound card rate (usually 48 kHz)."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "recording.wav")
        open(p, "wb").write(_wav_bytes(_tone(3, 48000), 48000))
        assert len(A.prepare(A.load_audio(p))) >= 1


def test_health_and_port_guard():
    import socket
    from app import app, port_owner
    h = app.test_client().get("/health").get_json()
    assert set(h) >= {"ok", "pid", "model", "error"}
    s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen(1)
    try:
        assert port_owner(s.getsockname()[1])          # a busy port is detected
    finally:
        s.close()


def test_collect_needs_code_and_consent():
    """Saving a recording: refused without consent, refused with a wrong code, saved + logged when both are right."""
    import app as appmod
    c = appmod.app.test_client()
    wav = lambda: (io.BytesIO(_wav_bytes(_tone(3))), "r.wav")
    old_code, old_dataset = appmod.ACCESS_CODE, appmod.DATASET
    with tempfile.TemporaryDirectory() as td:
        appmod.ACCESS_CODE, appmod.DATASET = "K7Q2", td
        try:
            base = {"speaker": "tester", "label": "human"}
            r = c.post("/collect", data={**base, "code": "K7Q2", "audio": wav()}, content_type="multipart/form-data")
            assert r.status_code == 403 and r.get_json()["need"] == "consent"
            r = c.post("/collect", data={**base, "code": "nope", "consent": "yes", "audio": wav()}, content_type="multipart/form-data")
            assert r.status_code == 403 and r.get_json()["need"] == "code"
            r = c.post("/collect", data={**base, "code": "K7Q2", "consent": "yes", "audio": wav()}, content_type="multipart/form-data")
            assert r.status_code == 200, r.get_json()
            log = open(os.path.join(td, "consent_log.csv"), encoding="utf-8").read()
            assert "tester" in log and r.get_json()["saved"].split("/")[-1] in log
            assert c.get("/health").get_json()["code_required"] is True
        finally:
            appmod.ACCESS_CODE, appmod.DATASET = old_code, old_dataset


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
