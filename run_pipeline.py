"""Rebuild everything end to end.

  python run_pipeline.py              # all steps (needs the downloads described in README)
  python run_pipeline.py --skip-data  # only prepare_data + train (reuse dataset/ as it is)

Every step is re-runnable: generators skip files that already exist.
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
    [PY, "scripts/generate_ai.py", "--engine", "gtts", "--per-lang", "60",
     "--langs", "ru", "tr", "en", "ar", "hi", "zh", "de", "es", "fr"],
    [PY, "scripts/generate_ai.py", "--engine", "piper", "--per-voice", "45"],
    [PY, "scripts/generate_ai.py", "--engine", "xtts", "--per-lang", "40"],
    # AI: voice changers / vocoder made from real voices
    [PY, "scripts/make_fakes.py", "--fraction", "0.3"],
]
TRAIN_STEPS = [
    [PY, "scripts/prepare_data.py", "--aug", "2", "--max-windows", "3"],
    [PY, "train.py", "--epochs", "15"],
]

if __name__ == "__main__":
    steps = TRAIN_STEPS if "--skip-data" in sys.argv else DATA_STEPS + TRAIN_STEPS
    for cmd in steps:
        print("\n>>>", " ".join(os.path.basename(c) if i == 0 else c for i, c in enumerate(cmd)), flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"!! step failed (exit {r.returncode}); fix it and re-run - finished steps are skipped")
            sys.exit(r.returncode)
