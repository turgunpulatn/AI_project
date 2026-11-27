"""Import real human speech from LibriSpeech (openslr.org/12, CC BY 4.0) into dataset/human.

  python scripts/import_librispeech.py --archive ../downloads_raw/dev-clean.tar.gz --per-speaker 25

* keeps at most N utterances per speaker (balanced speakers)
* only utterances of 2-15 s
* files are named  ls-<speaker>__<utt>.flac  ->  group = speaker
* writes the transcripts of OTHER utterances to texts/tts_texts_en.txt so the AI class
  can be generated from the same kind of text (content does not leak the label).
"""
import argparse
import os
import random
import shutil
import tarfile
from collections import defaultdict

import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--extract-to", default=os.path.join(os.path.dirname(ROOT), "downloads_raw", "librispeech"))
    ap.add_argument("--per-speaker", type=int, default=25)
    a = ap.parse_args()
    rnd = random.Random(42)

    if not os.path.isdir(os.path.join(a.extract_to, "LibriSpeech")):
        print("extracting ...")
        with tarfile.open(a.archive) as t:
            t.extractall(a.extract_to, filter="data")
    base = os.path.join(a.extract_to, "LibriSpeech")
    split = next(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))

    utts = defaultdict(list)          # speaker -> [(path, text)]
    for dirpath, _, files in os.walk(os.path.join(base, split)):
        for f in files:
            if f.endswith(".trans.txt"):
                for line in open(os.path.join(dirpath, f), encoding="utf-8"):
                    uid, text = line.strip().split(" ", 1)
                    utts[uid.split("-")[0]].append((os.path.join(dirpath, uid + ".flac"), text))

    out = os.path.join(ROOT, "dataset", "human")
    os.makedirs(out, exist_ok=True)
    copied, leftover_texts = 0, []
    for spk, items in sorted(utts.items()):
        rnd.shuffle(items)
        n = 0
        for path, text in items:
            dur = sf.info(path).duration
            if n < a.per_speaker and 2.0 <= dur <= 15.0:
                uid = os.path.splitext(os.path.basename(path))[0]
                shutil.copy(path, os.path.join(out, f"ls-{spk}__{uid}.flac"))
                n += 1
            elif 4 <= len(text.split()) <= 30:
                leftover_texts.append(text.capitalize() + ".")
        copied += n
    rnd.shuffle(leftover_texts)
    with open(os.path.join(ROOT, "texts", "tts_texts_en.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(leftover_texts))
    print(f"copied {copied} utterances from {len(utts)} speakers -> dataset/human")
    print(f"wrote {len(leftover_texts)} unused transcripts -> texts/tts_texts_en.txt")


if __name__ == "__main__":
    main()
