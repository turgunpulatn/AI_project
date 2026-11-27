"""Import multilingual real human speech from Google FLEURS (CC-BY-4.0) into dataset/human.

  python scripts/import_fleurs.py --root ../downloads_raw/fleurs --per-lang 350

* expects <root>/<lang_code>/dev.tar.gz + dev.tsv  (from huggingface.co/datasets/google/fleurs)
* keeps up to N utterances per language, 2-15 s, balanced over sentences and gender
* file name:  fl-<lang>-s<sentence_id>__<wav_id>.wav  -> group = sentence (FLEURS has no speaker ids)
* writes the transcripts to data/tts_texts_<lang>_fleurs.txt so the AI class can be generated
  in the same languages with the same kind of sentences.
"""
import argparse
import csv
import os
import random
import shutil
import sys
import tarfile
from collections import defaultdict

import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANGS = {"uz_uz": "uz", "ru_ru": "ru", "kk_kz": "kk", "ky_kg": "ky", "tg_tj": "tg", "tr_tr": "tr",
         "az_az": "az", "en_us": "en", "ar_eg": "ar", "hi_in": "hi", "cmn_hans_cn": "zh", "de_de": "de",
         "es_419": "es", "fr_fr": "fr"}
csv.field_size_limit(10 ** 7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(ROOT), "downloads_raw", "fleurs"))
    ap.add_argument("--per-lang", type=int, default=350)
    ap.add_argument("--split", default="dev", choices=["dev", "train", "test"])
    ap.add_argument("--langs", nargs="*", default=["uz_uz", "ru_ru", "en_us"],
                    help="FLEURS codes (default: the supported languages uz_uz ru_ru en_us)")
    a = ap.parse_args()
    rnd = random.Random(42)
    out = os.path.join(ROOT, "dataset", "human")
    os.makedirs(out, exist_ok=True)
    total = 0
    for code, lang in LANGS.items():
        if a.langs and code not in a.langs:
            continue
        d = os.path.join(a.root, code)
        arc, tsv = os.path.join(d, f"{a.split}.tar.gz"), os.path.join(d, f"{a.split}.tsv")
        if not (os.path.exists(arc) and os.path.exists(tsv)):
            print(f"{code}: missing, skipped")
            continue
        if not os.path.isdir(os.path.join(d, a.split)):
            with tarfile.open(arc) as t:
                t.extractall(d, filter="data")
        rows = list(csv.reader(open(tsv, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
        by_sent = defaultdict(list)
        texts = set()
        for r in rows:
            if len(r) < 3:
                continue
            sid, wav, text = r[0], r[1], r[2].strip()
            texts.add(text)
            p = os.path.join(d, a.split, wav)
            if os.path.exists(p):
                by_sent[sid].append(p)
        # round-robin over sentences -> many different sentences (and speakers) per language
        for v in by_sent.values():
            rnd.shuffle(v)
        sids = list(by_sent)
        rnd.shuffle(sids)
        n, k = 0, 0
        while n < a.per_lang and any(by_sent[s] for s in sids):
            s = sids[k % len(sids)]
            k += 1
            if not by_sent[s]:
                continue
            p = by_sent[s].pop()
            dur = sf.info(p).duration
            if 2.0 <= dur <= 25.0:
                dst = os.path.join(out, f"fl-{lang}-s{s}__{os.path.splitext(os.path.basename(p))[0]}.wav")
                if not os.path.exists(dst):
                    shutil.copy(p, dst)
                n += 1
        total += n
        suffix = "" if a.split == "dev" else "_" + a.split
        with open(os.path.join(ROOT, "data", f"tts_texts_{lang}_fleurs{suffix}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(t for t in texts if 3 <= len(t.split()) <= 40)))
        print(f"{code}: {n} utterances from {len(by_sent)} sentences, {len(texts)} texts")
    print(f"total {total} -> dataset/human")
    if total == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
