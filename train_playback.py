"""PLAYBACK detector: is this voice coming LIVE from a person's mouth, or is it being PLAYED through a
loudspeaker (phone, laptop, speaker)? For anti-fraud this is a warning sign on its own: an AI voice
(or a recording of a real person) played into a microphone is a replay attack.

Positives (playback = 1):
  * REAL replays recorded on this laptop (scripts/record_replay.py): dataset/*/*__replay-*.wav
  * simulated loudspeaker replays (voxcheck.audio.replay) of a sample of all other recordings
Negatives (live = 0):
  * team recordings made LIVE into the laptop microphone (Collect data tab): me-*  (most important)
  * the same sample of recordings WITHOUT a loudspeaker: original, room echo only, codecs, noise, phone line
Every example (both classes) also gets a random codec/noise treatment, so the detector cannot cheat on
the browser's Opus codec or on the recording level.

Models compared (grouped 5-fold CV, groups = source recording / speaker):
  P1 VoxCNN on log-mel (own, from scratch)      P2 MLP on WavLM features (reused frozen extractor)
Reported separately on the REAL subset (real replays vs live mic recordings), which is what matters.

  python train_playback.py
Output: models/playback.pt, results/playback.md. A model that is worse than the one in use (or above the EER
limit) is saved as models/playback_candidate.pt + results/playback_candidate.md instead, never replacing it.
"""
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from voxcheck import audio as A  # noqa: E402

EXTS = (".wav", ".mp3", ".flac", ".ogg", ".webm", ".m4a")


def codec_noise(y, rng):
    """random codec + noise, applied to BOTH classes (no room/speaker simulation here)."""
    import subprocess
    import tempfile
    codec = [None, ("libopus", "ogg", ["16k", "24k", "48k"]), ("libmp3lame", "mp3", ["64k", "128k"]),
             ("aac", "m4a", ["64k"])][rng.integers(4)]
    if codec:
        with tempfile.TemporaryDirectory() as td:
            src, dst = os.path.join(td, "i.wav"), os.path.join(td, "o." + codec[1])
            A.save_wav(src, y)
            subprocess.run([A.ffmpeg_bin(), "-v", "error", "-nostdin", "-y", "-i", src, "-c:a", codec[0],
                            "-b:a", str(rng.choice(codec[2])), dst], check=True, capture_output=True)
            y = A.load_audio(dst)
    snr = rng.uniform(20, 45)
    n = rng.standard_normal(len(y)).astype(np.float32)
    n *= np.sqrt((np.mean(y ** 2) + 1e-9) / 10 ** (snr / 10) / (np.mean(n ** 2) + 1e-9))
    return np.clip((y + n) * rng.uniform(0.5, 1.5), -1, 1).astype(np.float32)


def is_mic_feedback(path):
    """Feedback clips ('the answer was wrong') can be a microphone recording OR an uploaded file.
    Only mic clips are real 'played through a speaker' examples; an uploaded TTS file was never played.
    New clips carry the source in the name (feedback-mic-* / feedback-file-*). Older untagged clips are
    judged by their noise floor: a microphone always has hiss, an uploaded file has digital silence."""
    name = os.path.basename(path)
    if "__feedback-" not in name:
        return True                                   # not feedback: a normal team mic recording
    if "__feedback-mic-" in name:
        return True
    if "__feedback-file-" in name:
        return False
    y = A.load_audio(path)
    n = len(y) // 320 * 320
    if n == 0:
        return False
    rms = np.sqrt(np.mean(y[:n].reshape(-1, 320) ** 2, axis=1)) + 1e-10
    return 20 * np.log10(np.percentile(rms, 5) / rms.max()) > -70


def make_examples(job):
    """-> list of (window, label, kind) for one source file."""
    path, mode, seed = job
    rng = np.random.default_rng(seed)
    out = []
    try:
        y = A.clean(A.load_audio(path))
        if len(y) < A.SR:
            return path, out
        y = y[: 8 * A.SR]
        if mode == "real_replay":
            versions = [(y, 1, "real_replay")] + [(codec_noise(y, rng), 1, "real_replay") for _ in range(3)]
        elif mode == "live":
            versions = [(y, 0, "live_mic")] + [(codec_noise(y, rng), 0, "live_mic") for _ in range(3)]
        else:  # sample of the big dataset: one direct version + one simulated loudspeaker version
            from scipy.signal import fftconvolve
            direct = y if rng.random() < 0.5 else fftconvolve(y, A._room_ir(rng))[: len(y)].astype(np.float32)
            if rng.random() < 0.5:              # live voice into a laptop mic (hiss + AGC) is still LIVE
                direct = A.laptop_mic(direct, rng)
            if rng.random() < 0.6:              # played from a phone (most common attack) ...
                played = A.phone_speaker(y, rng)
                if rng.random() < 0.7:
                    played = fftconvolve(played, A._room_ir(rng))[: len(played)].astype(np.float32)
            else:                               # ... or from a laptop / bluetooth speaker
                played = A.replay(y, rng)
            if rng.random() < 0.8:
                played = A.laptop_mic(played, rng)
            versions = [(codec_noise(direct, rng), 0, "direct"), (codec_noise(played, rng), 1, "sim_replay")]
        for v, lab, kind in versions:
            v = A.clean(v)
            if len(v) < A.SR:
                continue
            ws = A.windows(v)
            ws = ws[np.linspace(0, len(ws) - 1, min(3, len(ws))).round().astype(int)]
            out += [(w.astype(np.float16), lab, kind) for w in ws]
    except Exception as e:  # noqa: BLE001
        print("skip", os.path.basename(path), str(e)[:80])
    return path, out


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import argparse
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    import train as T
    from voxcheck import ssl

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=2500, help="recordings sampled from the big dataset for simulation")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--force", action="store_true", help="replace models/playback.pt even if the new model is worse")
    ap.add_argument("--max-real-eer", type=float, default=0.20,
                    help="only enable the detector in the app if its EER on REAL recordings is below this")
    a = ap.parse_args()
    rnd = random.Random(0)
    files = {lab: [os.path.join(ROOT, "dataset", lab, f) for f in os.listdir(os.path.join(ROOT, "dataset", lab))
                   if f.lower().endswith(EXTS)] for lab in ("human", "ai")}
    allf = files["human"] + files["ai"]
    # real device -> microphone recordings: scripted laptop-speaker replays + AI voices the team played from a
    # phone / laptop into the app's microphone (dataset/ai/me-*, except digital voice-changer copies)
    from voxcheck.meta import MANIPULATIONS
    mic_ai = [f for f in files["ai"] if os.path.basename(f).startswith("me-")
              and os.path.basename(f).split("__")[-1].split("-")[0] not in (MANIPULATIONS - {"feedback"})
              and is_mic_feedback(f)]
    real = [f for f in allf if "__replay-" in f] + mic_ai
    # live voices: team mic recordings, plus "It was a human" feedback recorded with the microphone in the app
    # (tagged feedback-mic-*; untagged / uploaded feedback files are not used, they may never have been spoken live)
    live = [f for f in files["human"] if os.path.basename(f).startswith(("me-", "human_")) and "__replay-" not in f
            and ("__feedback-" not in f or "__feedback-mic-" in f)]
    rest = [f for f in allf if f not in set(real) and f not in set(live)]
    sample = rnd.sample(rest, min(a.sample, len(rest)))
    print(f"real replays {len(real)} | live mic recordings {len(live)} | sampled for simulation {len(sample)}")
    if len(live) < 20:
        sys.exit("Need at least 20 live microphone recordings (Collect data tab) - record more first.")
    jobs = [(f, "real_replay", i) for i, f in enumerate(real)] + [(f, "live", 10_000 + i) for i, f in enumerate(live)] \
        + [(f, "sim", 20_000 + i) for i, f in enumerate(sample)]

    W, Y, K, G = [], [], [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (path, ex_list) in enumerate(ex.map(make_examples, jobs, chunksize=4), 1):
            name = os.path.basename(path)
            g = name.split("__")[0] if not name.startswith("me-") else "live:" + name  # each live take = own group
            for w, lab, kind in ex_list:
                W.append(w); Y.append(lab); K.append(kind); G.append(g)
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    W = np.stack(W); Y = np.array(Y); K = np.array(K); G = np.array(G)
    print(f"windows: {len(Y)}  playback={int(Y.sum())}  live/direct={int((Y == 0).sum())}  "
          f"{ {k: int((K == k).sum()) for k in np.unique(K)} }")

    mel = np.stack([A.logmel(w.astype(np.float32)).astype(np.float16) for w in W])
    emb = ssl.embed(W.astype(np.float32))
    mel_src, emb_src = T.to_source(mel), T.to_source(emb)

    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1).split(W, Y, G))
    real_mask = np.isin(K, ["real_replay", "live_mic"])
    res = {}
    for name, kind in [("P1 VoxCNN (own) on log-mel", "cnn"), ("P2 WavLM features + own MLP", "mlp")]:
        oof = np.zeros(len(Y))
        for tr, te in folds:
            if kind == "cnn":
                m = T.train_cnn(mel_src, Y[tr], {"channels": (16, 32, 64, 128), "specaug": True}, a.epochs, rows=tr)
                oof[te] = T.predict_cnn(m, mel_src, rows=te)
            else:
                m = T.train_mlp(emb_src, Y[tr], a.epochs, rows=tr)
                oof[te] = T.predict_mlp(m, emb_src, rows=te)
        r = dict(acc=float(((oof >= .5) == Y).mean()), auc=float(roc_auc_score(Y, oof)), eer=T.eer(Y, oof))
        if real_mask.any() and len(set(Y[real_mask])) == 2:
            yr, pr = Y[real_mask], oof[real_mask]
            r.update(real_acc=float(((pr >= .5) == yr).mean()), real_auc=float(roc_auc_score(yr, pr)),
                     real_eer=T.eer(yr, pr), real_live_ok=float((pr[yr == 0] < .5).mean()),
                     real_replay_caught=float((pr[yr == 1] >= .5).mean()))
        phone = (Y == 1) & np.char.startswith(G.astype(str), "live:me-")     # AI played from a phone into the app's mic
        if phone.any():
            per_clip = {}
            for g_, p_ in zip(G[phone], oof[phone]):
                per_clip.setdefault(g_, []).append(p_)
            r.update(phone_mic_caught=float(np.mean([np.mean(v) >= .5 for v in per_clip.values()])),
                     phone_mic_clips=len(per_clip))
        res[name] = (kind, r)
        print(name, {k: round(v, 3) for k, v in r.items()}, flush=True)

    # choose by EER on the REAL subset (what the demo faces), tie -> own CNN
    best_name = min(res, key=lambda n: (round(res[n][1].get("real_eer", res[n][1]["eer"]), 3), res[n][0] != "cnn"))
    kind = res[best_name][0]
    print("selected:", best_name)
    real_eer = res[best_name][1].get("real_eer", 1.0)
    enabled = real_eer < a.max_real_eer
    # never throw away a working detector for a worse one: the app keeps the current playback.pt unless the new
    # model is enabled AND at least as good on real recordings (or --force)
    cur_path = os.path.join(ROOT, "models", "playback_meta.json")
    cur = json.load(open(cur_path)) if os.path.exists(cur_path) else {}
    cur_ok = bool(cur.get("enabled")) and os.path.exists(os.path.join(ROOT, "models", "playback.pt"))
    replace = enabled and (a.force or not cur_ok or real_eer <= cur.get("real_eer", 1.0))
    out_name = "playback.pt" if replace else "playback_candidate.pt"
    tag = "" if replace else "_candidate"
    why = (f"above the {a.max_real_eer:.2f} limit" if not enabled else
           f"worse than the model in use ({cur.get('real_eer', 1.0):.3f})")
    if replace:
        print(f"real-recording EER {real_eer:.3f} -> ENABLED in app (models/playback.pt)")
    else:
        print(f"real-recording EER {real_eer:.3f} is {why} -> saved as models/playback_candidate.pt; "
              f"{'the current models/playback.pt stays in use' if cur_ok else 'no playback warning in the app'}")
    if replace and os.path.exists(os.path.join(ROOT, "models", "playback_candidate.pt")):
        os.remove(os.path.join(ROOT, "models", "playback_candidate.pt"))
    rows = np.arange(len(Y))
    if kind == "cnn":
        m = T.train_cnn(mel_src, Y, {"channels": (16, 32, 64, 128), "specaug": True}, a.epochs, rows=rows)
        torch.save({"kind": "cnn", "channels": [16, 32, 64, 128], "state_dict": m.cpu().state_dict()},
                   os.path.join(ROOT, "models", out_name))
    else:
        m = T.train_mlp(emb_src, Y, a.epochs, rows=rows)
        torch.save({"kind": "mlp", "dim": ssl.DIM, "state_dict": m.cpu().state_dict()},
                   os.path.join(ROOT, "models", out_name))
    meta = {"approach": best_name, "trained_at": time.strftime("%Y-%m-%d %H:%M"), "enabled": bool(enabled),
            "real_eer": real_eer, "threshold_real_eer": a.max_real_eer,
            "n_real_replays": len(real), "n_live": len(live), "results": {n: r for n, (_, r) in res.items()}}
    json.dump(meta, open(os.path.join(ROOT, "models", f"playback{tag}_meta.json"), "w"), indent=2)

    md = ["# Playback (replay) detector" + (" - candidate, NOT used by the app" if tag else ""), "",
          f"Data: {len(real)} real replays recorded through a speaker into this laptop's mic ({len(mic_ai)} of them AI "
          f"voices played from a phone/laptop into the app), {len(live)} live microphone recordings, "
          f"{len(sample)} recordings with simulated phone / laptop-speaker replay. 5-fold CV grouped by source.", "",
          "| Model | Acc (all) | EER (all) | **EER (real only)** | Live voice accepted | Real replay caught | Phone→mic AI caught |",
          "|---|---|---|---|---|---|---|"]
    for n, (_, r) in res.items():
        f = lambda k: f"{r[k]:.1%}" if k in r else "–"
        md.append(f"| {n}{' ✔' if n == best_name else ''} | {r['acc']:.1%} | {r['eer']:.3f} | "
                  f"{r.get('real_eer', float('nan')):.3f} | {f('real_live_ok')} | {f('real_replay_caught')} | "
                  f"{f('phone_mic_caught')} ({r.get('phone_mic_clips', 0)} clips) |")
    md += ["", f"Selected: **{best_name}**. EER on real recordings {real_eer:.3f}: "
           + ("below the 0.20 threshold, so the detector is **enabled** in the app." if replace else
              why + ", so it is **not used** (kept as models/playback_candidate.pt; the app keeps "
              "models/playback.pt if there is one). More live and replay recordings are needed.")]
    open(os.path.join(ROOT, "results", f"playback{tag}.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
