"""Shared audio pipeline used by BOTH training and the web app.

Keeping one implementation guarantees the model sees exactly the same
preprocessing at train time and at demo time.

Pipeline:  decode (ffmpeg) -> 16 kHz mono -> trim/remove silence
           -> loudness normalise -> 2 s windows (1 s hop)
           -> log-mel spectrogram (CNN)  |  hand-crafted features (classical ML)
"""
import glob
import os
import shutil
import subprocess
import tempfile

import librosa
import numpy as np

SR = 16000
WIN_S = 2.0          # window length fed to the model
HOP_S = 1.0          # step between windows
N_MELS = 64
N_FFT = 512
HOP_LEN = 160        # 10 ms -> 201 frames per 2 s window
MIN_SPEECH_S = 1.0
MAX_AUDIO_S = 60.0

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ffmpeg_bin():
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    for local in [os.path.join(_ROOT, "tools", "ffmpeg", exe)] + sorted(glob.glob(os.path.join(_ROOT, "ffmpeg-*", "bin", exe))):
        if os.path.exists(local):
            return local
    found = shutil.which("ffmpeg")
    if not found:
        raise RuntimeError("ffmpeg not found. Put ffmpeg.exe in tools/ffmpeg/ or install it on PATH.")
    return found


import logging

log = logging.getLogger("voxcheck.audio")


def _load_ffmpeg(path, sr, max_s):
    cmd = [ffmpeg_bin(), "-v", "error", "-nostdin", "-i", path,
           "-t", str(max_s), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exit {proc.returncode}: "
                           f"{proc.stderr.decode('utf-8', 'replace').strip()[-300:] or 'no message'}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _load_soundfile(path, sr, max_s):
    """Fallback without ffmpeg: wav / flac / ogg / mp3 through libsndfile, resampled with librosa."""
    import soundfile as sf
    with sf.SoundFile(path) as f:
        y = f.read(frames=int(max_s * f.samplerate), dtype="float32", always_2d=True)
        file_sr = f.samplerate
    y = y.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32)


def load_audio(path, sr=SR, max_s=MAX_AUDIO_S):
    """Decode any audio file to float32 mono at `sr`.
    ffmpeg first (same decoder as in training; handles webm/m4a/opus too), libsndfile as a fallback,
    so one broken ffmpeg call can never take the whole demo down. Real errors are logged, the user
    gets a short message."""
    errors = []
    for name, fn in (("ffmpeg", _load_ffmpeg), ("soundfile", _load_soundfile)):
        try:
            y = fn(path, sr, max_s)
            if y.size:
                if errors:
                    log.warning("decoded %s with %s after: %s", os.path.basename(path), name, "; ".join(errors))
                return y
            errors.append(f"{name}: no samples")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {str(e)[:300]}")
    log.error("could not decode %s (%d bytes): %s", path,
              os.path.getsize(path) if os.path.exists(path) else -1, " | ".join(errors))
    if all("no samples" in e for e in errors):
        raise ValueError("Audio file is empty.")
    raise ValueError("Could not read this file as audio. Please upload wav, mp3, ogg, m4a or webm.")


def save_wav(path, y, sr=SR):
    import soundfile as sf
    sf.write(path, np.clip(y, -1, 1), sr, subtype="PCM_16")


def clean(y, top_db=35, target_dbfs=-20.0):
    """Remove leading/trailing/internal silence, remove DC, normalise loudness.
    Returns an EMPTY array for (near-)digital silence, so callers can reject it."""
    y = y - np.mean(y)
    if y.size == 0 or np.max(np.abs(y)) < 1e-3:  # below -60 dBFS peak: nothing to analyse
        return np.zeros(0, np.float32)
    intervals = librosa.effects.split(y, top_db=top_db, frame_length=1024, hop_length=256)
    if len(intervals):
        pad = int(0.05 * SR)
        parts = [y[max(0, s - pad):min(len(y), e + pad)] for s, e in intervals]
        y = np.concatenate(parts)
    rms = np.sqrt(np.mean(y ** 2)) + 1e-9
    y = y * (10 ** (target_dbfs / 20) / rms)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def windows(y, win_s=WIN_S, hop_s=HOP_S):
    """Split into fixed windows; short clips are repeat-padded to one window."""
    win, hop = int(win_s * SR), int(hop_s * SR)
    if len(y) < win:
        y = np.tile(y, int(np.ceil(win / max(len(y), 1))))[:win]
    starts = list(range(0, len(y) - win + 1, hop))
    return np.stack([y[s:s + win] for s in starts])


def logmel(w):
    m = librosa.feature.melspectrogram(y=w, sr=SR, n_fft=N_FFT, hop_length=HOP_LEN,
                                       n_mels=N_MELS, fmin=20, fmax=SR // 2)
    m = librosa.power_to_db(m, ref=1.0, top_db=80.0)
    return ((m - m.mean()) / (m.std() + 1e-6)).astype(np.float32)


def handcrafted(w):
    """99 summary features for classical models (MFCC, deltas, spectral shape)."""
    S = np.abs(librosa.stft(w, n_fft=N_FFT, hop_length=HOP_LEN))
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(
        librosa.feature.melspectrogram(S=S ** 2, sr=SR, n_fft=N_FFT, n_mels=N_MELS)), n_mfcc=20)
    d1 = librosa.feature.delta(mfcc)
    feats = [mfcc.mean(1), mfcc.std(1), d1.mean(1), d1.std(1)]
    for f in (librosa.feature.spectral_centroid(S=S, sr=SR, n_fft=N_FFT),
              librosa.feature.spectral_bandwidth(S=S, sr=SR, n_fft=N_FFT),
              librosa.feature.spectral_rolloff(S=S, sr=SR, n_fft=N_FFT),
              librosa.feature.spectral_flatness(S=S, n_fft=N_FFT),
              librosa.feature.zero_crossing_rate(w, frame_length=N_FFT, hop_length=HOP_LEN),
              librosa.feature.rms(S=S, frame_length=N_FFT)):
        feats.append([f.mean(), f.std()])
    feats.append(librosa.feature.spectral_contrast(S=S, sr=SR, n_fft=N_FFT).mean(1))
    return np.concatenate([np.ravel(f) for f in feats]).astype(np.float32)


def prepare(y_raw):
    """raw audio -> cleaned windows. Raises ValueError on unusable input."""
    y = clean(y_raw)
    if len(y) < MIN_SPEECH_S * SR:
        raise ValueError("Not enough speech detected (need at least 1 second of voice).")
    return windows(y)


# ---------------- augmentation (training only) ----------------
_CODECS = [None,
           ("libmp3lame", "mp3", ["32k", "64k", "128k"]),
           ("libopus", "ogg", ["12k", "24k", "48k"]),
           ("aac", "m4a", ["48k", "96k"])]


def _room_ir(rng, sr=SR):
    """Synthetic room impulse response: direct sound + exponentially decaying diffuse tail."""
    rt60 = rng.uniform(0.15, 0.8)
    n = int(sr * rt60)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir[: int(0.002 * sr)] = 0
    ir[0] = 1.0 / rng.uniform(0.3, 1.0)        # direct-to-reverb ratio
    return (ir / np.sqrt(np.sum(ir ** 2))).astype(np.float32)


def replay(y, rng):
    """Simulate playing audio through a small loudspeaker into a microphone in a room:
    speaker band-pass + resonance, soft clipping, room reverb, mic high-pass."""
    from scipy.signal import butter, fftconvolve, sosfilt
    lo, hi = rng.uniform(150, 450), rng.uniform(4000, 7500)
    y = sosfilt(butter(2, [lo, hi], btype="band", fs=SR, output="sos"), y)
    f0 = rng.uniform(600, 3000)                 # cheap-speaker resonance bump
    y = y + rng.uniform(0.2, 0.8) * sosfilt(butter(2, [f0 * 0.8, f0 * 1.2], btype="band", fs=SR, output="sos"), y)
    y = np.tanh(rng.uniform(1, 3) * y / (np.max(np.abs(y)) + 1e-9))      # speaker distortion
    y = fftconvolve(y, _room_ir(rng))[: len(y)]
    y = sosfilt(butter(1, 80, btype="high", fs=SR, output="sos"), y)
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.7).astype(np.float32)


def _peq(y, f0, gain_db, q, sr=SR):
    """Peaking-EQ biquad (RBJ cookbook): boost/cut `gain_db` around f0."""
    from scipy.signal import lfilter
    a_ = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * a_, -2 * np.cos(w0), 1 - alpha * a_]
    a = [1 + alpha / a_, -2 * np.cos(w0), 1 - alpha / a_]
    return lfilter(b, a, y)


def phone_speaker(y, rng):
    """Simulate a PHONE / tablet loudspeaker (the usual way an AI voice reaches a laptop mic):
    tiny driver -> almost no energy below 300-700 Hz, peaky 1-4 kHz resonances, early roll-off, distortion.
    Measured on our real phone->laptop-mic recordings: the 100-300 Hz band is ~10 dB weaker than in
    live speech and the energy peaks at 1-3 kHz (see results/mic_channel.md)."""
    from scipy.signal import butter, sosfilt
    y = sosfilt(butter(int(rng.choice([2, 4])), rng.uniform(200, 600), btype="high", fs=SR, output="sos"), y)
    y = sosfilt(butter(4, rng.uniform(5000, 7800), btype="low", fs=SR, output="sos"), y)
    for _ in range(int(rng.integers(1, 4))):
        y = _peq(y, rng.uniform(800, 4500), rng.uniform(3, 10), rng.uniform(1, 4))
    y = np.tanh(rng.uniform(1, 4) * y / (np.max(np.abs(y)) + 1e-9))
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.7).astype(np.float32)


def laptop_mic(y, rng):
    """Simulate a built-in laptop MICROPHONE as heard through the browser:
    high-pass, a presence tilt, the mic's own hiss (high noise floor) and automatic gain control
    that pumps the noise up between words. Applied to BOTH classes, so 'sounds like a laptop mic'
    can never mean 'human' (that was the shortcut behind 'AI recorded with the mic -> Human')."""
    from scipy.signal import butter, sosfilt
    y = sosfilt(butter(2, rng.uniform(40, 150), btype="high", fs=SR, output="sos"), y)
    if rng.random() < 0.6:
        y = _peq(y, rng.uniform(2000, 6000), rng.uniform(-4, 4), 1.0)
    # self-noise, 12-30 dB below speech: white hiss + low-frequency rumble (fan, desk, handling)
    hiss = rng.standard_normal(len(y))
    rumble = sosfilt(butter(2, rng.uniform(100, 300), btype="low", fs=SR, output="sos"), rng.standard_normal(len(y)))
    hiss = hiss / np.std(hiss) + rng.uniform(0.5, 2.0) * rumble / (np.std(rumble) + 1e-9)
    snr = rng.uniform(12, 30)
    hiss *= np.sqrt((np.mean(y ** 2) + 1e-9) / 10 ** (snr / 10) / (np.mean(hiss ** 2) + 1e-9))
    y = y + hiss
    # AGC: slow gain that lifts quiet parts (and the hiss in them) towards the speech level
    frame = int(0.02 * SR)
    n = int(np.ceil(len(y) / frame))
    env = np.sqrt(np.mean(np.pad(y, (0, n * frame - len(y))).reshape(n, frame) ** 2, axis=1)) + 1e-6
    k = int(rng.integers(5, 25))                                   # 100-500 ms smoothing
    env = np.convolve(env, np.ones(k) / k, mode="same")
    gain = np.clip((np.max(env) / env) ** rng.uniform(0.2, 0.6), 1.0, 10 ** (20 / 20))
    gain = np.interp(np.arange(len(y)), np.arange(n) * frame + frame / 2, gain)
    y = y * gain
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.7).astype(np.float32)


def augment(y, rng):
    """Simulate real-world channels so the model can't cheat on 'clean vs noisy':
    laptop-speaker or PHONE-speaker replay / room reverb, laptop microphone (hiss + AGC),
    band-limit (phone call), codec round-trip, noise and gain.
    Applied identically to BOTH classes."""
    from scipy.signal import fftconvolve
    r = rng.random()
    played = r < 0.5
    if r < 0.2:
        y = replay(y, rng)                     # played through a laptop / bluetooth speaker and re-recorded
    elif r < 0.5:
        y = phone_speaker(y, rng)              # played from a phone held near the laptop
        if rng.random() < 0.7:
            y = fftconvolve(y, _room_ir(rng))[: len(y)].astype(np.float32)
    elif r < 0.65:
        y = fftconvolve(y, _room_ir(rng))[: len(y)].astype(np.float32)   # room only
    if rng.random() < (0.8 if played else 0.4):
        y = laptop_mic(y, rng)                 # captured by a built-in laptop microphone
    hp = int(rng.choice([0, 80, 150, 300]))
    lp = int(rng.choice([0, 0, 3400, 5000, 7000]))
    codec = _CODECS[rng.integers(len(_CODECS))]
    if rng.random() < 0.15:                    # narrow-band phone call: 8 kHz + low bitrate
        lp, codec = 3400, ("libmp3lame", "mp3", ["16k", "24k"])
    filters = []
    if hp:
        filters.append(f"highpass=f={hp}")
    if lp:
        filters.append(f"lowpass=f={lp}")
    if filters or codec:
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "in.wav")
            save_wav(src, y)
            ext = codec[1] if codec else "wav"
            dst = os.path.join(td, "out." + ext)
            cmd = [ffmpeg_bin(), "-v", "error", "-nostdin", "-y", "-i", src]
            if filters:
                cmd += ["-af", ",".join(filters)]
            if codec:
                cmd += ["-c:a", codec[0], "-b:a", str(rng.choice(codec[2]))]
            subprocess.run(cmd + [dst], check=True, capture_output=True)
            y = load_audio(dst)
    snr = rng.uniform(10, 40)
    noise = rng.standard_normal(len(y)).astype(np.float32)
    if rng.random() < 0.5:  # pinkish noise
        noise = np.cumsum(noise)
        noise -= np.convolve(noise, np.ones(400) / 400, mode="same")
    sig_p = np.mean(y ** 2) + 1e-9
    noise *= np.sqrt(sig_p / (10 ** (snr / 10)) / (np.mean(noise ** 2) + 1e-9))
    y = (y + noise) * rng.uniform(0.5, 1.5)
    return np.clip(y, -1, 1).astype(np.float32)
