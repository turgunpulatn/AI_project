"""Rebuild everything end to end.

  python run_pipeline.py              # all steps (needs the downloads described in README)
  python run_pipeline.py --skip-data  # only prepare_data + train + playback detector (reuse dataset/ as it is)

Every step is re-runnable: generators skip files that already exist.
TRAIN_STEPS are exactly the settings of the current model in models/ (trained 2026-11-26, uz / en / ru):
~20 min data preparation + ~60 min training + ~6 min playback detector on an RTX 4060.
"""
import os
import subprocess
import sys

PY = sys.executable
DL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "downloads_raw")

DATA_STEPS = [
    # human speech
    [PY, "scripts/import_librispeech.py", "--archive", os.path.join(DL, "dev-clean.tar.gz"), "--per-speaker", "25"],
    [PY, "scripts/import_fleurs.py", "--root", os.path.join(DL, "fleurs"), "--per-lang", "350"],
    # AI speech: 4 TTS engines + voice cloning
    [PY, "scripts/generate_ai.py", "--engine", "sapi", "--per-voice", "20"],
    [PY, "scripts/generate_ai.py", "--engine", "edge", "--per-lang", "150"],
    [PY, "scripts/generate_ai.py", "--engine", "gtts", "--per-lang", "60", "--langs", "uz", "ru", "en"],
    [PY, "scripts/generate_ai.py", "--engine", "piper", "--per-voice", "45"],
    # more voice identities (ru / uz had only 2 native voices each); the unseen-voice test voices stay out
    [PY, "scripts/generate_ai.py", "--engine", "edge-extra", "--per-voice", "25", "--langs", "ru", "uz"],
    [PY, "scripts/generate_ai.py", "--engine", "piper", "--per-voice", "30", "--langs", "en", "--voices",
     "en_GB-cori-medium", "en_GB-jenny_dioco-medium", "en_GB-northern_english_male-medium", "en_US-joe-medium",
     "en_US-john-medium", "en_US-norman-medium", "en_US-hfc_female-medium", "en_US-sam-medium"],
    [PY, "scripts/generate_ai.py", "--engine", "xtts", "--per-lang", "40"],
    # AI: voice changers / vocoder made from real voices
    [PY, "scripts/make_fakes.py", "--fraction", "0.3"],
]
TRAIN_STEPS = [
    [PY, "scripts/prepare_data.py", "--aug", "2", "--max-windows", "2", "--boost", "3", "--workers", "10"],
    [PY, "train.py", "--epochs", "12"],
    [PY, "train_playback.py", "--epochs", "12", "--workers", "10"],
]

if __name__ == "__main__":
    steps = TRAIN_STEPS if "--skip-data" in sys.argv else DATA_STEPS + TRAIN_STEPS
    for cmd in steps:
        print("\n>>>", " ".join(os.path.basename(c) if i == 0 else c for i, c in enumerate(cmd)), flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"!! step failed (exit {r.returncode}); fix it and re-run - finished steps are skipped")
            sys.exit(r.returncode)
