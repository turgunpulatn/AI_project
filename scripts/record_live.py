"""Record LIVE speech from the laptop microphone in one go (for the playback detector and the human class).

  python scripts/record_live.py --name pulat --minutes 5

Read the sentences shown on screen (or anything else) out loud, normally, at different distances.
The recording is cut into ~4 s pieces -> dataset/human/me-<name>__live-<session>-<n>.wav
It uses the SAME microphone path as scripts/record_replay.py, so "live" and "replayed" differ only in
whether a loudspeaker was involved.
"""
import argparse
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from voxcheck import audio as A  # noqa: E402
import record_replay as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="speaker name, e.g. pulat")
    ap.add_argument("--minutes", type=float, default=5)
    a = ap.parse_args()
    name = "".join(ch for ch in a.name.lower() if ch.isalnum() or ch == "-")[:30] or "team"
    mics = R.list_mics()
    if not mics:
        sys.exit("no microphone found")
    sents = [l.strip() for l in open(os.path.join(ROOT, "data", "sentences.txt"), encoding="utf-8") if l.strip()]
    for extra in ("tts_texts_uz_fleurs.txt", "tts_texts_ru_fleurs.txt", "tts_texts_en_fleurs.txt"):
        p = os.path.join(ROOT, "data", extra)
        if os.path.exists(p):
            sents += [l.strip() for l in open(p, encoding="utf-8") if 20 < len(l.strip()) < 140][:60]
    random.shuffle(sents)

    rec = R.open_live_recorder(mics[0][1])
    if rec is None:
        sys.exit("microphone opened silent several times - check that it is not muted, then run again")
    p, th, buf, _ = rec
    buf.clear()
    total = a.minutes * 60
    t0 = time.time()
    print(f"\n  RECORDING {a.minutes:g} min - read each sentence out loud (any language: uz / ru / en).")
    print("  Move closer/further from the laptop now and then. Press Ctrl+C to stop early.\n")
    try:
        i = 0
        while time.time() - t0 < total:
            left = int(total - (time.time() - t0))
            print(f"  [{left // 60}:{left % 60:02d} left]  {sents[i % len(sents)]}", flush=True)
            i += 1
            time.sleep(6)
    except KeyboardInterrupt:
        print("\n  stopped early")
    p.terminate()
    th.join(5)
    y = np.frombuffer(b"".join(buf), dtype=np.float32).copy()
    print(f"\n  recorded {len(y) / A.SR:.0f} s, level {20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9):.0f} dBFS")

    # cut into ~4 s pieces of actual speech
    session = time.strftime("%H%M")
    out_dir = os.path.join(ROOT, "dataset", "human")
    seg, n = int(4 * A.SR), 0
    for s in range(0, len(y) - seg + 1, seg):
        piece = y[s:s + seg]
        speech = A.clean(piece)
        if len(speech) < 1.5 * A.SR or np.max(np.abs(piece)) < 0.01:
            continue
        A.save_wav(os.path.join(out_dir, f"me-{name}__live-{session}-{n:03d}.wav"), piece)
        n += 1
    print(f"  saved {n} live clips -> dataset/human/me-{name}__live-{session}-*.wav")


if __name__ == "__main__":
    main()
