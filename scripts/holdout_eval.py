"""Fair comparison of model versions on a HOLDOUT set that no model was trained on.

  python scripts/holdout_eval.py --build                      # create ../downloads_raw/holdout once
  python scripts/holdout_eval.py --models models models_backup_2227

Holdout (focus languages uz / ru / en):
  human : FLEURS dev utterances that were NOT imported into dataset/
  ai    : fresh Microsoft neural TTS phrases + voice-changer copies of the holdout humans
  every clip also as a simulated "played through a speaker" version
"""
import argparse
import asyncio
import glob
import json
import os
import random
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from voxcheck import audio as A  # noqa: E402

HOLD = os.environ.get("VOX_HOLDOUT", os.path.join(os.path.dirname(ROOT), "downloads_raw", "holdout"))
LANG_DIR = {"uz": "uz_uz", "ru": "ru_ru", "en": "en_us"}
VOICES = {"uz": ["uz-UZ-SardorNeural", "uz-UZ-MadinaNeural"], "ru": ["ru-RU-DmitryNeural", "ru-RU-SvetlanaNeural"],
          "en": ["en-US-AndrewNeural", "en-GB-SoniaNeural", "en-AU-NatashaNeural", "en-IN-PrabhatNeural"]}


def build(per_lang=30):
    import edge_tts
    import make_fakes as M
    rnd = random.Random(2026)
    rng = np.random.default_rng(2026)
    os.makedirs(HOLD, exist_ok=True)
    used = {os.path.basename(f).split("__")[1][:-4] for f in glob.glob(os.path.join(ROOT, "dataset", "human", "fl-*"))}
    for lang, code in LANG_DIR.items():
        pool = [f for f in glob.glob(os.path.join(os.path.dirname(ROOT), "downloads_raw", "fleurs", code, "dev", "*.wav"))
                if os.path.basename(f)[:-4] not in used]
        hum = rnd.sample(pool, min(per_lang, len(pool)))
        texts = [l.strip() for l in open(os.path.join(ROOT, "texts", f"tts_texts_{lang}_fleurs.txt"), encoding="utf-8")
                 if 30 < len(l.strip()) < 200]
        for i, f in enumerate(hum):
            y = A.load_audio(f)
            A.save_wav(os.path.join(HOLD, f"human_{lang}_{i:02d}.wav"), y)
            att = ["vcpitch", "vcgender", "vcclassic"][i % 3]
            A.save_wav(os.path.join(HOLD, f"ai-{att}_{lang}_{i:02d}.wav"), M.apply(y, att, rng))

        async def tts():
            for i in range(per_lang):
                out = os.path.join(HOLD, f"ai-tts_{lang}_{i:02d}.mp3")
                if os.path.exists(out):
                    continue
                for _ in range(3):
                    try:
                        await asyncio.wait_for(edge_tts.Communicate(rnd.choice(texts), rnd.choice(VOICES[lang])).save(out), 45)
                        break
                    except Exception:  # noqa: BLE001
                        await asyncio.sleep(2)
        asyncio.run(tts())
    # simulated loudspeaker versions of everything
    for f in sorted(glob.glob(os.path.join(HOLD, "*.*"))):
        if "_replay" in f:
            continue
        b, _ = os.path.splitext(os.path.basename(f))
        A.save_wav(os.path.join(HOLD, f"{b}_replay.wav"), A.replay(A.load_audio(f), rng))
    print(f"holdout built: {len(os.listdir(HOLD))} files in {HOLD}")


def evaluate(model_dir):
    """run in a subprocess so each model folder gets a fresh Detector"""
    code = f"""
import os, sys, glob, json
os.environ['VOX_MODELS'] = {model_dir!r}
sys.path.insert(0, {ROOT!r})
from voxcheck.detector import Detector
D = Detector()
out = {{}}
for f in sorted(glob.glob(os.path.join({HOLD!r}, '*.*'))):
    try:
        out[os.path.basename(f)] = D.analyze(f)['ai'] / 100
    except Exception as e:
        out[os.path.basename(f)] = None
print('JSON' + json.dumps({{'approach': D.meta['approach'], 'scores': out}}))
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", errors="ignore")
    line = next((l for l in r.stdout.splitlines() if l.startswith("JSON")), None)
    if not line:
        raise RuntimeError(r.stderr[-800:])
    return json.loads(line[4:])


def summarize(res):
    from sklearn.metrics import roc_curve
    sc = {k: v for k, v in res["scores"].items() if v is not None}

    def eer(keys):
        y = np.array([0 if k.startswith("human") else 1 for k in keys])
        p = np.array([sc[k] for k in keys])
        if len(set(y)) < 2:
            return float("nan")
        fpr, tpr, _ = roc_curve(y, p)
        i = np.nanargmin(np.abs(1 - tpr - fpr))
        return float((fpr[i] + 1 - tpr[i]) / 2)

    def acc(keys):
        return float(np.mean([(sc[k] >= .5) == (not k.startswith("human")) for k in keys])) if keys else float("nan")
    allk = list(sc)
    clean = [k for k in allk if "_replay" not in k]
    rep = [k for k in allk if "_replay" in k]
    out = {"approach": res["approach"], "EER all": eer(allk), "EER clean": eer(clean), "EER speaker-replay": eer(rep),
           "acc all": acc(allk)}
    for lang in LANG_DIR:
        ks = [k for k in clean if f"_{lang}_" in k]
        out[f"acc {lang}"] = acc(ks)
    for g in ("human", "ai-tts", "ai-vc", "ai-xtts"):
        ks = [k for k in clean if k.startswith(g)]
        out[f"acc {g}"] = acc(ks)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--models", nargs="*", default=[])
    a = ap.parse_args()
    if a.build:
        build()
    rows = []
    for m in a.models:
        r = summarize(evaluate(os.path.abspath(m)))
        r["folder"] = m
        rows.append(r)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, ensure_ascii=False))
    if rows:
        rows = [{k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()} for r in rows]   # NaN -> null
        json.dump(rows, open(os.path.join(ROOT, "results", "holdout_comparison.json"), "w"), indent=1)
