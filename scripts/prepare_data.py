"""Cleaning + labelling + formatting pipeline.

dataset/human/*, dataset/ai/*   (any audio format)
   -> decode 16 kHz mono -> remove silence -> loudness normalise
   -> K augmented copies per file (replay through speaker / room / phone / codec / noise)   [aug_id > 0]
   -> 2 s windows -> log-mel (for CNN) + hand-crafted features (for classical ML)
   -> dataset/processed/mel.npy (float16, written incrementally - never all in RAM)
      dataset/processed/dataset.npz (labels, groups, language, attack type, hand-crafted features)
      dataset/processed/manifest.csv, results/data_report.md

Label  : folder (human=0, ai=1).  Group / language / attack: from the file name, see voxcheck/meta.py.

Usage:  python scripts/prepare_data.py --aug 2 --max-windows 3
"""
import argparse
import csv
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from voxcheck import audio as A  # noqa: E402
from voxcheck.meta import ATTACK_NAMES, attack_of, group_of, lang_of, supported  # noqa: E402

EXTS = (".wav", ".mp3", ".flac", ".ogg", ".webm", ".m4a", ".opus")
LABELS = {"human": 0, "ai": 1}
PROC = os.path.join(ROOT, "dataset", "processed")


def process(job):
    path, label, file_id, n_aug, max_win = job
    rng = np.random.default_rng(file_id)
    out = {"mel": [], "hc": [], "wav": [], "aug": [], "info": None, "err": None}
    try:
        raw = A.load_audio(path)
        y = A.clean(raw)
        out["info"] = dict(raw_s=len(raw) / A.SR, clean_s=len(y) / A.SR,
                           raw_rms_db=20 * np.log10(np.sqrt(np.mean(raw ** 2)) + 1e-9),
                           clean_rms_db=20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9) if len(y) else -99)
        if len(y) < A.MIN_SPEECH_S * A.SR:
            out["err"] = "less than 1 s of speech after cleaning"
            return path, label, file_id, out
        versions = [y] + [A.clean(A.augment(y, rng)) for _ in range(n_aug)]
        for aug_id, v in enumerate(versions):
            if len(v) < A.SR // 2:
                continue
            ws = A.windows(v)
            if max_win and len(ws) > max_win:  # evenly spaced subset keeps long files from dominating
                ws = ws[np.linspace(0, len(ws) - 1, max_win).round().astype(int)]
            for w in ws:
                out["mel"].append(A.logmel(w).astype(np.float16))
                out["hc"].append(A.handcrafted(w))
                out["wav"].append(w.astype(np.float16))
                out["aug"].append(aug_id)
    except Exception as e:  # noqa: BLE001
        out["err"] = str(e)[:120]
    return path, label, file_id, out


def make_figure(path, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa.display
    raw = A.load_audio(path)
    y = A.clean(raw)
    aug = A.clean(A.replay(y, np.random.default_rng(0)))
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    for a, sig, title in zip(ax, [raw, y, aug], ["1. raw input", "2. after silence removal + loudness normalisation",
                                                 "3. augmented copy: played through a speaker in a room"]):
        m = librosa.power_to_db(librosa.feature.melspectrogram(y=sig, sr=A.SR, n_mels=A.N_MELS), ref=np.max)
        librosa.display.specshow(m, sr=A.SR, x_axis="time", y_axis="mel", ax=a)
        a.set_title(f"{title}  ({len(sig) / A.SR:.1f} s)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", type=int, default=2, help="augmented copies per file")
    ap.add_argument("--max-windows", type=int, default=3, help="max 2 s windows per recording version")
    ap.add_argument("--boost", type=int, default=3,
                    help="x more augmented copies for REAL microphone data: replays, team recordings, user corrections "
                         "(rare but closest to the demo)")
    ap.add_argument("--no-wavlm", action="store_true", help="skip WavLM feature extraction")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()

    jobs, metas, fid, skipped_lang = [], [], 0, 0
    for folder, label in LABELS.items():
        d = os.path.join(ROOT, "dataset", folder)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if f.lower().endswith(EXTS) and os.path.getsize(os.path.join(d, f)) > 1000:
                if not supported(f):           # only Uzbek / English / Russian (+ team mic recordings)
                    skipped_lang += 1
                    continue
                # real microphone data (speaker replays, team recordings, "the answer was wrong" corrections)
                # is rare but exactly what the live demo sees -> more augmented copies
                rare = "__replay-" in f or f.startswith("me-")
                jobs.append((os.path.join(d, f), label, fid, a.aug * (a.boost if rare else 1), a.max_windows))
                metas.append((group_of(f), lang_of(f), attack_of(f, label)))
                fid += 1
    if not jobs:
        sys.exit("No audio found in dataset/human or dataset/ai")
    print(f"skipped {skipped_lang} files in unsupported languages (only uz / en / ru are used)")
    print(f"{len(jobs)} files, {a.aug} augmented copies each, <= {a.max_windows} windows/version, {a.workers} workers")

    os.makedirs(PROC, exist_ok=True)
    raw_path = os.path.join(PROC, "mel.raw")
    use_wavlm = not a.no_wavlm
    if use_wavlm:
        try:
            from voxcheck import ssl
            ssl.load()
            print(f"WavLM features on {ssl._device}")
        except Exception as e:  # noqa: BLE001
            print("WavLM unavailable, skipped:", str(e)[:100])
            use_wavlm = False
    emb_path, wav_buf = os.path.join(PROC, "wavlm.raw"), []
    femb = open(emb_path, "wb") if use_wavlm else None

    def flush(force=False):
        nonlocal wav_buf
        if use_wavlm and wav_buf and (force or len(wav_buf) >= 960):
            femb.write(ssl.embed(np.stack(wav_buf).astype(np.float32)).tobytes())
            wav_buf = []
    n_frames = A.logmel(np.zeros(int(A.WIN_S * A.SR), np.float32)).shape
    hc, y, groups, langs, attacks, files, augs, manifest = [], [], [], [], [], [], [], []
    def bounded_map(ex, fn, items, block=400):
        """ex.map() submits EVERYTHING at once; if the consumer (WavLM on GPU) is slower than the
        workers, finished results pile up in RAM. Submitting in blocks keeps memory bounded."""
        for s in range(0, len(items), block):
            yield from ex.map(fn, items[s:s + block], chunksize=4)

    with open(raw_path, "wb") as fraw, ProcessPoolExecutor(a.workers) as ex:
        for i, (path, label, file_id, out) in enumerate(bounded_map(ex, process, jobs), 1):
            g, lang, att = metas[file_id]
            info = out["info"] or {}
            manifest.append(dict(file=os.path.relpath(path, ROOT), label=label, group=g, lang=lang, attack=att,
                                 windows=len(out["mel"]),
                                 status="ok" if not out["err"] else "skipped: " + out["err"],
                                 **{k: round(float(v), 2) for k, v in info.items()}))
            n = len(out["mel"])
            if n:
                fraw.write(np.stack(out["mel"]).astype(np.float16).tobytes())
                if use_wavlm:
                    wav_buf += out["wav"]
                    flush()
                hc += out["hc"]; augs += out["aug"]
                y += [label] * n; groups += [g] * n; langs += [lang] * n; attacks += [att] * n; files += [file_id] * n
            if i % 500 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)

    flush(force=True)
    if femb:
        femb.close()
    # raw float16 stream -> proper .npy (memory-mapped, so it is never fully in RAM here)
    N = len(y)
    if use_wavlm:
        e = np.memmap(emb_path, dtype=np.float16, mode="r")
        assert e.size == N * ssl.DIM, (e.size, N)
        np.save(os.path.join(PROC, "wavlm.npy"), np.asarray(e).reshape(N, ssl.DIM))
        del e
        os.remove(emb_path)
    elif os.path.exists(os.path.join(PROC, "wavlm.npy")):
        os.remove(os.path.join(PROC, "wavlm.npy"))
    src = np.memmap(raw_path, dtype=np.float16, mode="r", shape=(N, *n_frames))
    dst = np.lib.format.open_memmap(os.path.join(PROC, "mel.npy"), mode="w+", dtype=np.float16, shape=(N, *n_frames))
    for s in range(0, N, 5000):
        dst[s:s + 5000] = src[s:s + 5000]
    dst.flush()
    del src, dst
    os.remove(raw_path)
    np.savez_compressed(os.path.join(PROC, "dataset.npz"),
                        hc=np.stack(hc), y=np.array(y, np.int64), group=np.array(groups), lang=np.array(langs),
                        attack=np.array(attacks), file_id=np.array(files), aug=np.array(augs))
    with open(os.path.join(PROC, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        keys = ["file", "label", "group", "lang", "attack", "windows", "status", "raw_s", "clean_s",
                "raw_rms_db", "clean_rms_db"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(manifest)

    # ---- data report (before / after, per class, language and attack type) ----
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    ok = [m for m in manifest if m["status"] == "ok"]
    L = ["# Data report", "", "| Class | Files | Skipped | Groups | Raw audio (min) | After cleaning (min) | Windows (incl. aug) |",
         "|---|---|---|---|---|---|---|"]
    for name, lab in LABELS.items():
        ms = [m for m in manifest if m["label"] == lab]
        oks = [m for m in ms if m["status"] == "ok"]
        L.append(f"| {name} | {len(ms)} | {len(ms) - len(oks)} | {len({m['group'] for m in oks})} | "
                 f"{sum(m.get('raw_s', 0) for m in ms) / 60:.1f} | {sum(m['clean_s'] for m in oks) / 60:.1f} | "
                 f"{sum(m['windows'] for m in oks)} |")
    L += ["", "## Per language (files)", "", "| Language | Human | AI |", "|---|---|---|"]
    lc = Counter((m["lang"], m["label"]) for m in ok)
    for lang in sorted({m["lang"] for m in ok}):
        L.append(f"| {lang} | {lc[(lang, 0)]} | {lc[(lang, 1)]} |")
    L += ["", "## AI files per attack type", "", "| Attack | Files |", "|---|---|"]
    for att, n in Counter(m["attack"] for m in ok if m["label"] == 1).most_common():
        L.append(f"| {ATTACK_NAMES.get(att, att)} | {n} |")
    if ok:
        L += ["", f"Mean loudness before: {np.mean([m['raw_rms_db'] for m in ok]):.1f} dBFS "
                  f"(std {np.std([m['raw_rms_db'] for m in ok]):.1f}) -> after: "
                  f"{np.mean([m['clean_rms_db'] for m in ok]):.1f} dBFS (std {np.std([m['clean_rms_db'] for m in ok]):.1f})",
              f"Silence removed: {sum(m['raw_s'] - m['clean_s'] for m in ok) / 60:.1f} min", "",
              "![before/after](preprocessing_example.png)"]
        try:
            make_figure(os.path.join(ROOT, ok[0]["file"]), os.path.join(ROOT, "results", "preprocessing_example.png"))
        except Exception as e:  # noqa: BLE001
            print("figure skipped:", e)
    open(os.path.join(ROOT, "results", "data_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:6]))
    print(f"\nSaved {N} windows -> dataset/processed/mel.npy + dataset.npz")


if __name__ == "__main__":
    main()
