"""Create MANIPULATED voices (label = ai) from real human recordings: the "changed voice" attacks
used in phone fraud, made by the team with signal processing.

  python scripts/make_fakes.py --fraction 0.3

Attacks (one random attack per selected recording):
  vcpitch    pitch shift +-3..8 semitones, formants move with it      (typical voice-changer app)
  vcgender   formant shift only, pitch kept                           (male <-> female "gender" changer)
  vcclassic  resample trick: pitch + speed change together            (chipmunk / deep voice)
  vcrobot    phase-zeroed FFT resynthesis + slight pitch               (robot effect)
  vocoder    mel spectrogram -> Griffin-Lim resynthesis               (vocoder artifacts like neural TTS/VC)

Output: dataset/ai/<source_group>__<attack>-<id>.wav  (same group as the source -> same CV fold)
"""
import argparse
import os
import random
import subprocess
import sys
import tempfile
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from voxcheck import audio as A  # noqa: E402
from voxcheck.meta import group_of  # noqa: E402

ATTACKS = [("vcpitch", 0.30), ("vcgender", 0.25), ("vcclassic", 0.15), ("vocoder", 0.20), ("vcrobot", 0.10)]
EXTS = (".wav", ".mp3", ".flac", ".ogg", ".webm", ".m4a")


def ffmpeg_filter(y, af):
    with tempfile.TemporaryDirectory() as td:
        src, dst = os.path.join(td, "i.wav"), os.path.join(td, "o.wav")
        A.save_wav(src, y)
        subprocess.run([A.ffmpeg_bin(), "-v", "error", "-nostdin", "-y", "-i", src, "-af", af,
                        "-ar", str(A.SR), "-ac", "1", dst], check=True, capture_output=True)
        return A.load_audio(dst)


def apply(y, attack, rng):
    sr = A.SR
    if attack == "vcpitch":
        st = rng.choice([-1, 1]) * rng.uniform(3, 8)
        return ffmpeg_filter(y, f"rubberband=pitch={2 ** (st / 12):.4f}:formant=shifted")
    if attack == "vcgender":
        f = rng.choice([rng.uniform(0.78, 0.9), rng.uniform(1.12, 1.3)])
        return ffmpeg_filter(y, f"asetrate={int(sr * f)},aresample={sr},atempo={1 / f:.4f},"
                                f"rubberband=pitch={1 / f:.4f}:formant=preserved")
    if attack == "vcclassic":
        f = rng.choice([rng.uniform(0.75, 0.9), rng.uniform(1.1, 1.3)])
        return ffmpeg_filter(y, f"asetrate={int(sr * f)},aresample={sr}")
    if attack == "vcrobot":
        st = rng.uniform(-3, 3)
        return ffmpeg_filter(y, f"rubberband=pitch={2 ** (st / 12):.4f},"
                                "afftfilt=real='hypot(re,im)':imag='0':win_size=512")
    if attack == "vocoder":
        import librosa
        m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
        return librosa.feature.inverse.mel_to_audio(m, sr=sr, n_fft=1024, hop_length=256, n_iter=32).astype(np.float32)
    raise ValueError(attack)


def job(args):
    path, attack, seed, out_dir = args
    rng = np.random.default_rng(seed)
    try:
        y = A.load_audio(path)
        z = apply(y, attack, rng)
        z = z / (np.max(np.abs(z)) + 1e-9) * 0.8
        name = f"{group_of(path)}__{attack}-{seed:08x}.wav"
        A.save_wav(os.path.join(out_dir, name), z)
        return attack, None
    except Exception as e:  # noqa: BLE001
        return attack, f"{os.path.basename(path)}: {str(e)[:100]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.3, help="share of human recordings to fake")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    hum_dir, out_dir = os.path.join(ROOT, "dataset", "human"), os.path.join(ROOT, "dataset", "ai")
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(hum_dir) if f.lower().endswith(EXTS))
    # Re-runnable: a source that already has a fake is never faked again, and new sources are added only
    # until `fraction` of all human files is faked (before, every run picked a NEW random 30% once the
    # file list changed, so fakes kept piling up as the team added recordings).
    seeds_by_group = {}
    for f in os.listdir(out_dir):
        stem = os.path.splitext(f)[0]
        if "__" in stem and stem.split("__")[1].split("-")[0] in dict(ATTACKS):
            g, rest = stem.split("__", 1)
            try:
                seeds_by_group.setdefault(g, set()).add(int(rest.split("-")[1], 16))
            except (IndexError, ValueError):
                pass

    def stable_seed(f):
        return zlib.crc32(f.encode())

    def already_faked(f):
        seeds = seeds_by_group.get(group_of(f), ())
        legacy = int.from_bytes(f.encode()[-8:].ljust(8, b"\0"), "little") % (2 ** 32)   # old naming: legacy ^ index
        return stable_seed(f) in seeds or any((s ^ legacy) < len(files) for s in seeds)

    faked = {f for f in files if already_faked(f)}
    n_fakes = sum(len(s) for s in seeds_by_group.values())
    todo = max(0, int(len(files) * a.fraction) - n_fakes)
    fresh = sorted((f for f in files if f not in faked), key=stable_seed)[:todo]   # stable, not list-dependent
    rnd = random.Random(123)
    names, weights = zip(*ATTACKS)
    jobs = [(os.path.join(hum_dir, f), rnd.choices(names, weights)[0], stable_seed(f), out_dir) for f in fresh]
    print(f"{len(files)} human files, {n_fakes} fakes exist, faking {len(jobs)} more "
          f"(target {a.fraction:.0%}), {a.workers} workers")
    stats, errs = {}, []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (att, err) in enumerate(ex.map(job, jobs, chunksize=8), 1):
            stats[att] = stats.get(att, 0) + (err is None)
            if err:
                errs.append(err)
            if i % 300 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  {stats}")
    if errs:
        print(f"{len(errs)} errors, e.g. {errs[:3]}")


if __name__ == "__main__":
    main()
