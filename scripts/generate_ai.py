"""Create the AI-voice half of the dataset ourselves, with several different TTS / cloning engines
in Uzbek, Russian and English (architecture diversity = better generalisation to unseen fakes).

Engines
  sapi   Windows built-in voices (offline; concatenative/old)
  edge   Microsoft neural voices, uz / ru / en                (pip install edge-tts, internet)
  gtts   Google Translate TTS                             (pip install gTTS, internet)
  piper  Piper / VITS neural TTS, runs offline            (pip install piper-tts; voices from HF rhasspy/piper-voices)
  xtts   XTTS-v2 zero-shot VOICE CLONING of real speakers (pip install coqui-tts; Coqui CPML licence)

Text: texts/sentences.txt + texts/tts_texts_<lang>*.txt (FLEURS / LibriSpeech transcripts), so AI and human
say the same kind of sentences in the same languages. Languages without a native voice (ky, tg) are read by
related-language voices (kk, ru) - like a scammer using the closest available voice.

File names: see voxcheck/meta.py.   Usage:
  python scripts/generate_ai.py --engine edge  --per-lang 150
  python scripts/generate_ai.py --engine gtts  --per-lang 50
  python scripts/generate_ai.py --engine piper --per-voice 50
  python scripts/generate_ai.py --engine xtts  --per-lang 40
  python scripts/generate_ai.py --engine edge-extra --per-voice 25 --langs ru uz
  python scripts/generate_ai.py --engine piper --per-voice 30 --voices en_US-joe-medium ...
"""
import argparse
import asyncio
import glob
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from voxcheck.meta import LANGS  # noqa: E402

OUT = os.path.join(ROOT, "dataset", "ai")
LANG_ORDER = ["uz", "ru", "en"]   # supported languages (older runs used 14)
EDGE_LOCALES = {"uz": ["uz-UZ"], "ru": ["ru-RU"], "kk": ["kk-KZ"], "ky": ["kk-KZ", "ru-RU"], "tg": ["ru-RU", "kk-KZ"],
                "tr": ["tr-TR"], "az": ["az-AZ"], "en": ["en-"], "ar": ["ar-"], "hi": ["hi-IN"], "zh": ["zh-CN"],
                "de": ["de-DE"], "es": ["es-"], "fr": ["fr-FR"]}
XTTS_LANG = {"uz": "tr", "az": "tr", "kk": "ru", "ky": "ru", "tg": "ru", "ru": "ru", "tr": "tr", "en": "en",
             "ar": "ar", "hi": "hi", "zh": "zh-cn", "de": "de", "es": "es", "fr": "fr"}
PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
PIPER_DIR = os.path.join(os.path.dirname(ROOT), "downloads_raw", "piper")


_UZ = re.compile(r"\b(iltimos|bugun|ertaga|kecha|juda|uchun|biz|bu|va|edi|men|sizni|qancha|yana|bilan)\b|[og]'(?!clock)[a-z]",
                 re.IGNORECASE)


def _lang_simple(s):
    """language of a line in the mixed reading script texts/sentences.txt (ru / uz / en)"""
    if any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in s):
        return "ru"
    return "uz" if _UZ.search(s) else "en"


def texts(lang):
    """Sentence pool for a language code."""
    pool = []
    p = os.path.join(ROOT, "texts", "sentences.txt")
    if os.path.exists(p) and lang in ("en", "ru", "uz"):
        pool += [l.strip() for l in open(p, encoding="utf-8") if l.strip() and _lang_simple(l) == lang]
    for p in sorted(glob.glob(os.path.join(ROOT, "texts", f"tts_texts_{lang}*.txt"))):
        if os.path.basename(p)[len(f"tts_texts_{lang}"):][:1] in ("", ".", "_"):
            pool += [l.strip() for l in open(p, encoding="utf-8") if 10 <= len(l.strip()) <= 260]
    return list(dict.fromkeys(pool))


def _exists(path):
    return os.path.exists(path) and os.path.getsize(path) > 1000


# ------------------------------------------------------------------ SAPI
PS_SCRIPT = r"""
param($voice, $listFile, $outDir, $prefix)
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice($voice)
$i = 0
Get-Content -Encoding UTF8 $listFile | ForEach-Object {
  $i++
  $f = Join-Path $outDir ("{0}__{1:D4}.wav" -f $prefix, $i)
  if (-not (Test-Path $f)) {
    $s.Rate = Get-Random -Minimum -2 -Maximum 3
    $s.SetOutputToWaveFile($f)
    $s.Speak($_)
  }
}
$s.SetOutputToNull()
"""


def sapi(per_voice, rnd):
    voices = subprocess.run(["powershell", "-NoProfile", "-Command",
                             "Add-Type -AssemblyName System.Speech;"
                             "(New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices()"
                             "|%{$_.VoiceInfo.Name + '|' + $_.VoiceInfo.Culture}"],
                            capture_output=True, text=True).stdout.split("\n")
    with tempfile.TemporaryDirectory() as td:
        script = os.path.join(td, "tts.ps1")
        open(script, "w", encoding="utf-8").write(PS_SCRIPT)
        for line in filter(None, (v.strip() for v in voices)):
            name, culture = line.split("|")
            pool = texts(culture.split("-")[0]) or texts("en")
            lines = rnd.sample(pool, min(per_voice, len(pool)))
            lst = os.path.join(td, "s.txt")
            open(lst, "w", encoding="utf-8-sig").write("\n".join(lines))
            prefix = "sapi-" + name.split()[1].lower()
            subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script,
                            name, lst, OUT, prefix], check=True)
            print(f"  {prefix}: {len(lines)} files")


# ------------------------------------------------------------------ EDGE
async def edge(per_lang, rnd, langs, concurrency=4):
    import edge_tts
    allv = await edge_tts.list_voices()
    sem = asyncio.Semaphore(concurrency)
    stats = {"ok": 0, "skip": 0, "fail": 0}

    async def one(voice, path, text):
        if _exists(path):
            stats["skip"] += 1
            return
        async with sem:
            for attempt in range(4):
                try:
                    await asyncio.wait_for(   # a hung connection must not block the whole run
                        edge_tts.Communicate(text, voice, rate=f"{rnd.randint(-15, 15):+d}%",
                                             pitch=f"{rnd.randint(-8, 8):+d}Hz").save(path), timeout=45)
                    if _exists(path):
                        stats["ok"] += 1
                        return
                except Exception:  # noqa: BLE001
                    pass
                await asyncio.sleep(1 + attempt * 2)
            stats["fail"] += 1
            if os.path.exists(path):
                os.remove(path)

    jobs = []
    for lang in langs:
        voices = [v["ShortName"] for v in allv if any(v["Locale"].startswith(l) for l in EDGE_LOCALES[lang])]
        pool = texts(lang)
        if not voices or not pool:
            print(f"  {lang}: no voices or texts, skipped")
            continue
        rnd.shuffle(voices)
        voices = voices[:12]                       # cap voices per language (English has ~50)
        for i in range(per_lang):
            v = voices[i % len(voices)]
            group = "edge-" + v.lower().replace("neural", "")
            path = os.path.join(OUT, f"{group}__{lang}-{i:04d}.mp3")
            jobs.append(one(v, path, pool[rnd.randrange(len(pool))]))
        print(f"  {lang}: {per_lang} files over {len(voices)} voices")
    for k in range(0, len(jobs), 60):
        await asyncio.gather(*jobs[k:k + 60])
        print(f"  {min(k + 60, len(jobs))}/{len(jobs)}  {stats}", flush=True)


# ------------------------------------------------------------------ EDGE, extra voice identities
# Russian and Uzbek have only 2 native Microsoft voices each, so the model could learn "these 2 voices"
# instead of "AI speech". These add many new voice identities: multilingual voices from other locales
# reading Russian, and Turkish / Azerbaijani / Kazakh voices reading Uzbek or Russian (accented AI speech).
EXTRA_VOICES = {
    "ru": ["en-AU-WilliamMultilingualNeural", "en-US-AndrewMultilingualNeural", "en-US-AvaMultilingualNeural",
           "en-US-BrianMultilingualNeural", "en-US-EmmaMultilingualNeural", "fr-FR-VivienneMultilingualNeural",
           "fr-FR-RemyMultilingualNeural", "it-IT-GiuseppeMultilingualNeural", "ko-KR-HyunsuMultilingualNeural",
           "pt-BR-ThalitaMultilingualNeural", "kk-KZ-AigulNeural", "kk-KZ-DauletNeural"],
    "uz": ["tr-TR-EmelNeural", "tr-TR-AhmetNeural", "az-AZ-BabekNeural", "az-AZ-BanuNeural",
           "kk-KZ-AigulNeural", "kk-KZ-DauletNeural", "it-IT-GiuseppeMultilingualNeural",
           "fr-FR-VivienneMultilingualNeural", "en-US-AvaMultilingualNeural", "pt-BR-ThalitaMultilingualNeural"],
}


async def edge_extra(per_voice, rnd, langs, concurrency=4):
    import edge_tts
    sem = asyncio.Semaphore(concurrency)
    stats = {"ok": 0, "skip": 0, "fail": 0}

    async def one(voice, path, text):
        if _exists(path):
            stats["skip"] += 1
            return
        async with sem:
            for attempt in range(4):
                try:
                    await asyncio.wait_for(edge_tts.Communicate(
                        text, voice, rate=f"{rnd.randint(-15, 15):+d}%", pitch=f"{rnd.randint(-8, 8):+d}Hz").save(path), 45)
                    if _exists(path):
                        stats["ok"] += 1
                        return
                except Exception:  # noqa: BLE001
                    pass
                await asyncio.sleep(1 + attempt * 2)
            stats["fail"] += 1
            if os.path.exists(path):
                os.remove(path)

    jobs = []
    for lang in langs:
        pool = texts(lang)
        for v in EXTRA_VOICES.get(lang, []):
            group = "edge-" + v.lower().replace("neural", "")
            jobs += [one(v, os.path.join(OUT, f"{group}__{lang}-x{i:03d}.mp3"), pool[rnd.randrange(len(pool))])
                     for i in range(per_voice)]
    for k in range(0, len(jobs), 60):
        await asyncio.gather(*jobs[k:k + 60])
        print(f"  {min(k + 60, len(jobs))}/{len(jobs)}  {stats}", flush=True)


# ------------------------------------------------------------------ gTTS
def gtts(per_lang, rnd, langs):
    import time
    from gtts import gTTS
    from gtts.lang import tts_langs
    supported = tts_langs()
    code = {"zh": "zh-CN"}
    for lang in langs:
        c = code.get(lang, lang)
        pool = texts(lang)
        if c not in supported or not pool:
            print(f"  gtts {lang}: not supported, skipped")
            continue
        ok = 0
        for i in range(per_lang):
            path = os.path.join(OUT, f"gtts-{lang}__{lang}-{i:04d}.mp3")
            if _exists(path):
                ok += 1
                continue
            for attempt in range(3):
                try:
                    gTTS(pool[rnd.randrange(len(pool))], lang=c, slow=rnd.random() < 0.15).save(path)
                    ok += 1
                    break
                except Exception:  # noqa: BLE001
                    time.sleep(2 + 3 * attempt)
        print(f"  gtts {lang}: {ok} files", flush=True)


# ------------------------------------------------------------------ PIPER
# Piper voices kept OUT of training: scripts/unseen_eval.py tests on them
PIPER_HELDOUT = {"ru_RU-irina-medium", "ru_RU-ruslan-medium", "en_US-amy-medium", "en_US-ryan-medium",
                 "en_US-lessac-medium", "en_US-kristin-medium"}


def piper_voices(langs, per_lang_voices, only=None):
    """Download (once) up to N medium/high-quality Piper voices per language (or exactly `only`)."""
    os.makedirs(PIPER_DIR, exist_ok=True)
    idx_path = os.path.join(PIPER_DIR, "voices.json")
    if not os.path.exists(idx_path):
        urllib.request.urlretrieve(PIPER_BASE + "voices.json", idx_path)
    idx = json.load(open(idx_path, encoding="utf-8"))
    chosen = {}
    for lang in langs:
        cands = sorted(k for k, v in idx.items() if v["language"]["family"] == lang
                       and v["quality"] in ("medium", "high") and k not in PIPER_HELDOUT
                       and (only is None or k in only))
        picked = []
        for key in cands[:per_lang_voices]:
            files = [f for f in idx[key]["files"] if f.endswith((".onnx", ".onnx.json"))]
            for f in files:
                dst = os.path.join(PIPER_DIR, os.path.basename(f))
                if not os.path.exists(dst):
                    urllib.request.urlretrieve(PIPER_BASE + f, dst)
            picked.append(os.path.join(PIPER_DIR, os.path.basename(next(f for f in files if f.endswith(".onnx")))))
        if picked:
            chosen[lang] = picked
    return chosen


def piper(per_voice, rnd, langs, per_lang_voices=2, only=None):
    import wave
    from piper import PiperVoice
    for lang, models in piper_voices(langs, per_lang_voices if not only else 99, only).items():
        pool = texts(lang)
        for mp in models:
            voice = PiperVoice.load(mp)
            name = os.path.basename(mp).split(".")[0].lower().replace("_", "-")   # e.g. ru-ru-irina-medium
            group = "piper-" + name
            ok = 0
            for i in range(per_voice):
                path = os.path.join(OUT, f"{group}__{lang}-{i:04d}.wav")
                if not _exists(path):
                    try:
                        with wave.open(path, "wb") as wf:
                            voice.synthesize_wav(pool[rnd.randrange(len(pool))], wf)
                    except Exception as e:  # noqa: BLE001
                        print("   piper error", name, str(e)[:80])
                        continue
                ok += 1
            print(f"  {group}: {ok} files", flush=True)


# ------------------------------------------------------------------ XTTS (voice cloning)
def xtts(per_lang, rnd, langs):
    """Clone real FLEURS / LibriSpeech speakers: reference = one human clip, text = another sentence.
    Saved under the SOURCE speaker's group, so clone and original share a CV fold."""
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    import importlib.machinery
    import importlib.util
    import types

    import torch
    import torchaudio
    # coqui-tts insists on `torchcodec` for audio I/O, which on Windows needs FFmpeg shared DLLs.
    # It only reads reference audio through torchaudio.load -> we register a stub module and
    # route loading through our own ffmpeg-based loader instead.
    if importlib.util.find_spec("torchcodec") is None:
        stub = types.ModuleType("torchcodec")
        stub.__spec__ = importlib.machinery.ModuleSpec("torchcodec", None)
        sys.modules["torchcodec"] = stub
    from voxcheck.audio import load_audio as _ff_load

    def _load(path, *args, **kwargs):
        return torch.from_numpy(_ff_load(str(path), sr=24000))[None], 24000

    torchaudio.load = _load
    from TTS.api import TTS
    # the stub must not be visible to `transformers` (it would try to use it for audio)
    if isinstance(sys.modules.get("torchcodec"), types.ModuleType) and sys.modules["torchcodec"].__spec__.origin is None:
        del sys.modules["torchcodec"]
        try:
            import transformers.utils.import_utils as tiu
            if hasattr(tiu, "_torchcodec_available"):
                tiu._torchcodec_available = False
            if hasattr(tiu.is_torchcodec_available, "cache_clear"):
                tiu.is_torchcodec_available.cache_clear()
        except ImportError:
            pass
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dev)
    hum = os.path.join(ROOT, "dataset", "human")
    for lang in langs:
        refs = sorted(glob.glob(os.path.join(hum, f"fl-{lang}-*.wav")))
        if lang == "en":
            refs += sorted(glob.glob(os.path.join(hum, "ls-*.flac")))[::10]
        pool = texts(lang)
        if not refs or not pool:
            print(f"  xtts {lang}: no reference speakers/texts, skipped")
            continue
        ok = 0
        for i in range(per_lang):
            ref = refs[rnd.randrange(len(refs))]
            group = os.path.splitext(os.path.basename(ref))[0].split("__")[0]
            path = os.path.join(OUT, f"{group}__xtts-{lang}-{i:04d}.wav")
            if _exists(path):
                ok += 1
                continue
            try:
                tts.tts_to_file(text=pool[rnd.randrange(len(pool))][:250], speaker_wav=ref,
                                language=XTTS_LANG[lang], file_path=path)
                ok += 1
            except Exception as e:  # noqa: BLE001
                print("   xtts error", lang, str(e)[:100])
        print(f"  xtts {lang}: {ok} files (device={dev})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["sapi", "edge", "edge-extra", "gtts", "piper", "xtts"], default="edge")
    ap.add_argument("--per-lang", type=int, default=150)
    ap.add_argument("--per-voice", type=int, default=40)
    ap.add_argument("--langs", nargs="*", default=LANG_ORDER)
    ap.add_argument("--voices", nargs="*", help="piper: exactly these voices, e.g. en_US-joe-medium")
    a = ap.parse_args()
    assert set(a.langs) <= LANGS, f"unknown language in {a.langs}"
    os.makedirs(OUT, exist_ok=True)
    rnd = random.Random(7)
    print(f"Generating AI speech with '{a.engine}' -> {OUT}")
    if a.engine == "sapi":
        sapi(a.per_voice, rnd)
    elif a.engine == "edge":
        asyncio.run(edge(a.per_lang, rnd, a.langs))
    elif a.engine == "edge-extra":
        asyncio.run(edge_extra(a.per_voice, rnd, a.langs))
    elif a.engine == "gtts":
        gtts(a.per_lang, rnd, a.langs)
    elif a.engine == "piper":
        piper(a.per_voice, rnd, a.langs, only=set(a.voices) if a.voices else None)
    else:
        xtts(a.per_lang, rnd, a.langs)
