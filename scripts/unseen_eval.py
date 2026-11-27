"""Test on voices and speakers the model has NEVER seen (harder than holdout_eval.py).

  python scripts/unseen_eval.py --build      # create ../downloads_raw/unseen once (needs internet)
  python scripts/unseen_eval.py              # score the current models/ -> results/unseen_voices.md

Human : FLEURS *test* split (uz / ru / en). Training used only the train and dev splits, and FLEURS splits
        have no speakers in common, so these are new people.
AI    : Meta MMS-TTS (VITS, facebook/mms-tts-*): an engine that is not in training, in all three languages.
        Piper voices that are not in training (ru: irina, ruslan; en: amy, ryan, lessac, kristin).
Every clip is also scored as a simulated "phone speaker -> laptop mic" copy (same simulator as training,
so that part is only a channel-robustness check, not an unseen channel).
"""
import argparse
import io
import json
import os
import random
import sys
import tarfile
import urllib.request
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from voxcheck import audio as A  # noqa: E402

OUT = os.environ.get("VOX_UNSEEN", os.path.join(os.path.dirname(ROOT), "downloads_raw", "unseen"))
FLEURS = {"uz": "uz_uz", "ru": "ru_ru", "en": "en_us"}
MMS = {"uz": "facebook/mms-tts-uzb-script_cyrillic", "ru": "facebook/mms-tts-rus", "en": "facebook/mms-tts-eng"}
PIPER_UNSEEN = {"ru": ["ru_RU-irina-medium", "ru_RU-ruslan-medium"],
                "en": ["en_US-amy-medium", "en_US-ryan-medium", "en_US-lessac-medium", "en_US-kristin-medium"]}
PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
PIPER_DIR = os.path.join(os.path.dirname(ROOT), "downloads_raw", "piper")


def fleurs_test(lang, n):
    """Stream the FLEURS test archive and stop after n usable clips (no full 300-500 MB download)."""
    url = f"https://huggingface.co/datasets/google/fleurs/resolve/main/data/{FLEURS[lang]}/audio/test.tar.gz"
    got = len([f for f in os.listdir(OUT) if f.startswith(f"human_{lang}_")])
    if got >= n:
        return
    with urllib.request.urlopen(url, timeout=60) as r, tarfile.open(fileobj=r, mode="r|gz") as tar:
        for m in tar:
            if got >= n:
                break
            if not m.isfile() or not m.name.endswith(".wav"):
                continue
            data = tar.extractfile(m).read()
            import soundfile as sf
            y, sr = sf.read(io.BytesIO(data), dtype="float32")
            if not 3 <= len(y) / sr <= 20:
                continue
            A.save_wav(os.path.join(OUT, f"human_{lang}_{got:02d}.wav"), y if sr == A.SR else
                       __import__("librosa").resample(y, orig_sr=sr, target_sr=A.SR))
            got += 1
    print(f"  human {lang}: {got}")


def texts(lang, rnd, n):
    p = os.path.join(ROOT, "texts", f"tts_texts_{lang}_fleurs.txt")
    pool = [l.strip() for l in open(p, encoding="utf-8") if 40 < len(l.strip()) < 220]
    return rnd.sample(pool, n)


_UZ_CYR = [("o'", "ў"), ("g'", "ғ"), ("sh", "ш"), ("ch", "ч"), ("yo", "ё"), ("yu", "ю"), ("ya", "я"), ("ye", "е"),
           ("a", "а"), ("b", "б"), ("d", "д"), ("e", "е"), ("f", "ф"), ("g", "г"), ("h", "ҳ"), ("i", "и"), ("j", "ж"),
           ("k", "к"), ("l", "л"), ("m", "м"), ("n", "н"), ("o", "о"), ("p", "п"), ("q", "қ"), ("r", "р"), ("s", "с"),
           ("t", "т"), ("u", "у"), ("v", "в"), ("x", "х"), ("y", "й"), ("z", "з"), ("c", "с"), ("w", "в"), ("'", "ъ")]


def uz_to_cyrillic(t):
    """Latin Uzbek -> Cyrillic (the official MMS Uzbek voice only reads Cyrillic). Word-initial e -> э."""
    import re
    t = re.sub(r"[‘’ʻʼ`]", "'", t.lower())
    t = re.sub(r"e", "э", t)
    for a, b in _UZ_CYR:
        t = t.replace(a, b)
    return t


def mms(lang, n, rnd):
    import torch
    from transformers import AutoTokenizer, VitsModel
    if len([f for f in os.listdir(OUT) if f.startswith(f"ai-mms_{lang}_")]) >= n:
        return
    tok, model = AutoTokenizer.from_pretrained(MMS[lang]), VitsModel.from_pretrained(MMS[lang]).eval()
    torch.manual_seed(0)
    for i, t in enumerate(texts(lang, rnd, n)):
        t = uz_to_cyrillic(t) if lang == "uz" else t
        with torch.no_grad():
            y = model(**tok(t, return_tensors="pt")).waveform[0].numpy()
        sr = model.config.sampling_rate
        y = __import__("librosa").resample(y, orig_sr=sr, target_sr=A.SR) if sr != A.SR else y
        A.save_wav(os.path.join(OUT, f"ai-mms_{lang}_{i:02d}.wav"), y / (np.abs(y).max() + 1e-9) * 0.8)
    print(f"  mms {lang}: {n}")


def piper(lang, per_voice, rnd):
    from piper import PiperVoice
    for v in PIPER_UNSEEN.get(lang, []):
        mp = os.path.join(PIPER_DIR, v + ".onnx")
        if not os.path.exists(mp):
            loc, name, q = v.split("-")
            sub = f"{loc.split('_')[0]}/{loc}/{name}/{q}/{v}"
            for ext in (".onnx", ".onnx.json"):
                urllib.request.urlretrieve(PIPER_BASE + sub + ext, mp.replace(".onnx", ext))
        voice = PiperVoice.load(mp)
        for i, t in enumerate(texts(lang, rnd, per_voice)):
            path = os.path.join(OUT, f"ai-piper_{lang}_{v.split('-')[1]}{i:02d}.wav")
            if not os.path.exists(path):
                with wave.open(path, "wb") as wf:
                    voice.synthesize_wav(t, wf)
        print(f"  piper {v}: {per_voice}")


def build(n_human=40, n_mms=30, n_piper=30):
    os.makedirs(OUT, exist_ok=True)
    rnd = random.Random(7)
    for lang in FLEURS:
        fleurs_test(lang, n_human)
        mms(lang, n_mms, rnd)
        if lang in PIPER_UNSEEN:
            piper(lang, n_piper // len(PIPER_UNSEEN[lang]), rnd)
    rng = np.random.default_rng(7)
    from scipy.signal import fftconvolve
    for f in sorted(os.listdir(OUT)):
        if "_mic" in f:
            continue
        y = A.phone_speaker(A.load_audio(os.path.join(OUT, f)), rng)
        y = A.laptop_mic(fftconvolve(y, A._room_ir(rng))[: len(y)].astype(np.float32), rng)
        A.save_wav(os.path.join(OUT, os.path.splitext(f)[0] + "_mic.wav"), y)
    print(f"built {len(os.listdir(OUT))} files in {OUT}")


def eer(y, p):
    from sklearn.metrics import roc_curve
    if len(set(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    i = np.nanargmin(np.abs(1 - tpr - fpr))
    return float((fpr[i] + 1 - tpr[i]) / 2)


def evaluate():
    from voxcheck.detector import Detector
    D = Detector()
    rows = []
    for f in sorted(os.listdir(OUT)):
        try:
            p = D.analyze(os.path.join(OUT, f))["ai"] / 100
        except ValueError:
            continue
        src, lang = f.split("_")[0], f.split("_")[1]
        rows.append(dict(file=f, src=src, lang=lang, mic="_mic" in f, y=int(src != "human"), p=p))
    json.dump(rows, open(os.path.join(ROOT, "results", "unseen_voices.json"), "w"), indent=1)

    def acc(rs):
        return f"{np.mean([(r['p'] >= .5) == r['y'] for r in rs]):.1%} ({len(rs)})" if rs else "–"
    md = ["# Unseen voices and speakers", "",
          "Made by `scripts/unseen_eval.py`. None of these voices or speakers is in the training data: "
          "FLEURS **test** speakers (training used train + dev), Meta **MMS-TTS** (engine not in training), "
          "and **Piper voices not in training**. *mic* = simulated phone speaker -> laptop mic copy.", "",
          f"Model: {D.meta['approach']}", "",
          "| Source | uz | ru | en | all | all, mic copy |", "|---|---|---|---|---|---|"]
    for src, name in (("human", "Human (FLEURS test speakers), correct"), ("ai-mms", "MMS-TTS (new engine), caught"),
                      ("ai-piper", "Piper, new voices, caught")):
        cells = [acc([r for r in rows if r["src"] == src and r["lang"] == lg and not r["mic"]]) for lg in FLEURS]
        md.append(f"| {name} | " + " | ".join(cells) + f" | {acc([r for r in rows if r['src'] == src and not r['mic']])}"
                  f" | {acc([r for r in rows if r['src'] == src and r['mic']])} |")
    for mic in (False, True):
        rs = [r for r in rows if r["mic"] == mic]
        md.append(f"\nEER ({'mic copies' if mic else 'clean'}): **{eer([r['y'] for r in rs], [r['p'] for r in rs]):.3f}**")
    open(os.path.join(ROOT, "results", "unseen_voices.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    evaluate()
