"""REAL replay data: play recordings through the laptop speakers and record them with the laptop
microphone at the same time - exactly the "AI voice played into the mic" attack, on the demo hardware.

BOTH classes are replayed (human recordings stay label human, AI stays label ai), so the model learns
"AI vs human" and NOT "replayed vs not replayed".

  python scripts/record_replay.py --n 300            # ~35 min, laptop plays audio out loud
  python scripts/record_replay.py --list-devices

Output: dataset/<label>/<source_group>__replay-<id>.wav  (same group as the source -> same CV fold)
Tips: set Windows volume to ~50-70 %, keep the room normally quiet (some background noise is fine),
vary the setup between sessions (headphones off, phone speaker, different distance) for more variety.
"""
import argparse
import hashlib
import os
import random
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from voxcheck import audio as A  # noqa: E402
from voxcheck.meta import FOCUS_LANGS, attack_of, group_of, lang_of  # noqa: E402

EXTS = (".wav", ".mp3", ".flac", ".ogg", ".webm", ".m4a")


def ff(name):
    return os.path.join(os.path.dirname(A.ffmpeg_bin()), name + (".exe" if os.name == "nt" else ""))


def list_mics():
    out = subprocess.run([A.ffmpeg_bin(), "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                         capture_output=True, text=True, encoding="utf-8", errors="ignore").stderr.splitlines()
    mics, last = [], None
    for line in out:
        if "(audio)" in line:
            last = line.split('"')[1]
        elif last and "Alternative name" in line:
            mics.append((last, line.split('"')[1]))
            last = None
    return mics


def pick_sources(n, rnd):
    """Half human, half AI; spread over languages and attack types; skip existing replays."""
    picks = []
    for label in ("human", "ai"):
        d = os.path.join(ROOT, "dataset", label)
        files = [f for f in os.listdir(d) if f.lower().endswith(EXTS) and "__replay-" not in f
                 and os.path.getsize(os.path.join(d, f)) > 1000]
        buckets = {}
        for f in files:
            key = (lang_of(f), attack_of(f, label))
            buckets.setdefault(key, []).append(f)
        for v in buckets.values():
            rnd.shuffle(v)
        keys = list(buckets)
        keys += [k for k in keys if k[0] in FOCUS_LANGS] * 2      # uz / en / ru get 3x more turns
        rnd.shuffle(keys)
        k, chosen = 0, []
        while len(chosen) < n // 2 and any(buckets.values()):
            b = buckets[keys[k % len(keys)]]
            k += 1
            if b:
                chosen.append(b.pop())
        picks += [(label, os.path.join(d, f)) for f in chosen]
    rnd.shuffle(picks)
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--mic", help="dshow device (alternative name); default = first microphone")
    ap.add_argument("--max-seconds", type=float, default=7.0)
    ap.add_argument("--batch", type=int, default=30, help="clips per continuous recording")
    ap.add_argument("--list-devices", action="store_true")
    a = ap.parse_args()
    mics = list_mics()
    if a.list_devices or not mics:
        print("\n".join(f"{n}  ->  {alt}" for n, alt in mics) or "no microphone found")
        return
    mic = a.mic or mics[0][1]
    print(f"microphone: {next((n for n, alt in mics if alt == mic), mic)}")
    rnd = random.Random(int(time.time()))
    picks = pick_sources(a.n, rnd)
    print(f"replaying {len(picks)} recordings ({sum(l == 'human' for l, _ in picks)} human / "
          f"{sum(l == 'ai' for l, _ in picks)} AI) - the laptop will play audio out loud")
    done, failed_batches = 0, 0
    with tempfile.TemporaryDirectory() as td:
        for b in range(0, len(picks), a.batch):
            batch = picks[b:b + a.batch]
            got = record_batch(batch, mic, td, a.max_seconds)
            if got is None:
                failed_batches += 1
                if failed_batches >= 3:
                    sys.exit("microphone keeps returning silence: check volume / mute / privacy settings.")
                continue
            for (label, src), seg in got:
                h = hashlib.md5(f"{src}{time.time()}".encode()).hexdigest()[:8]
                A.save_wav(os.path.join(ROOT, "dataset", label, f"{group_of(src)}__replay-{h}.wav"), seg)
                done += 1
            print(f"  {min(b + a.batch, len(picks))}/{len(picks)}  saved {done}", flush=True)
    print(f"done: {done} real replays saved")


def open_live_recorder(mic, tries=6):
    """Open the microphone ONCE as a continuous stream. The Realtek driver sometimes opens 'dead'
    (digital silence), so check the first second and reopen if needed."""
    import threading
    for _ in range(tries):
        p = subprocess.Popen([A.ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-f", "dshow",
                              "-audio_buffer_size", "50", "-i", f"audio={mic}", "-ac", "1", "-ar", str(A.SR),
                              "-f", "f32le", "-"], stdout=subprocess.PIPE)
        buf, t0 = [], [None]

        def reader():
            while True:
                chunk = p.stdout.read(3200)
                if not chunk:
                    break
                if t0[0] is None:
                    t0[0] = time.perf_counter()
                buf.append(chunk)

        th = threading.Thread(target=reader, daemon=True)
        th.start()
        time.sleep(1.5)
        x = np.frombuffer(b"".join(buf), dtype=np.float32)
        if x.size and np.max(np.abs(x[len(x) // 3:])) > 0.004:     # alive (room noise is above this)
            return p, th, buf, t0
        p.kill()
        th.join(2)
        time.sleep(0.5)
    return None


def record_batch(batch, mic, td, max_s):
    """Play a batch of clips back-to-back into ONE continuous recording, then cut each clip out
    (position found by cross-correlation with the original)."""
    from scipy.signal import correlate
    rec = open_live_recorder(mic)
    if rec is None:
        print("  microphone opened silent several times - batch skipped")
        return None
    p, th, buf, t0 = rec
    clips = []
    for label, src in batch:
        y = A.clean(A.load_audio(src))
        if len(y) < A.SR:
            continue
        y = y[: int(max_s * A.SR)] * 0.9 / (np.max(np.abs(y)) + 1e-9)
        play = os.path.join(td, "play.wav")
        A.save_wav(play, y)
        t_start = time.perf_counter()
        subprocess.run([ff("ffplay"), "-nodisp", "-autoexit", "-loglevel", "quiet", play])
        clips.append((label, src, y, t_start))
        time.sleep(0.4)
    time.sleep(1.0)
    p.terminate()
    th.join(5)
    r = np.frombuffer(b"".join(buf), dtype=np.float32).copy()
    out = []
    if not clips:
        return out
    # the driver delivers audio with an unknown start-up delay -> locate the FIRST clip by searching the
    # whole recording, then apply the same offset to the clock-based position of every other clip
    y0, ts0 = clips[0][2], clips[0][3]
    c0 = np.abs(correlate(r[: int((ts0 - t0[0] + 8) * A.SR) + len(y0)], y0, mode="valid", method="fft"))
    offset = int(np.argmax(c0)) - int((ts0 - t0[0]) * A.SR)
    for label, src, y, t_start in clips:
        guess = int((t_start - t0[0]) * A.SR) + offset
        lo, hi = max(0, guess - int(0.3 * A.SR)), min(len(r), guess + len(y) + int(1.2 * A.SR))
        region = r[lo:hi]
        if len(region) < len(y):
            continue
        c = correlate(region, y, mode="valid", method="fft")
        start = lo + int(np.argmax(np.abs(c)))
        seg = r[start:start + len(y)]
        lvl = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9)
        if lvl < -52 or np.max(np.abs(seg)) < 0.01:                  # dead / not captured
            continue
        out.append(((label, src), seg))
    print(f"  batch: {len(out)}/{len(clips)} clips captured "
          f"(recording level {20 * np.log10(np.sqrt(np.mean(r ** 2)) + 1e-9):.0f} dBFS)", flush=True)
    return out


if __name__ == "__main__":
    main()
