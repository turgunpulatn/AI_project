"""File-name conventions -> group (for splitting), language and attack type.

  <group>__<rest>.<ext>
  group  = speaker / TTS voice / FLEURS sentence; never split across train and test
  rest   = "<lang>-<n>"          for TTS speaking a language different from the voice's own
         | "<attack>-<id>"       for manipulated copies of a human recording (same group as the source,
                                 so the original and its fake always land in the same fold)
Examples
  ls-1272__1272-128104-0000.flac          human, en (LibriSpeech speaker 1272)
  fl-uz-s1591__4967621184651376579.wav    human, uz (FLEURS sentence 1591)
  me-gulbara__a1b2c3d4.wav                human, team recording
  edge-ru-ru-dmitry__ky-0003.mp3          AI (edge TTS), Kyrgyz text read by a Russian voice
  fl-ru-s20__vcgender-9f3e.wav            AI (voice changer: formant/gender shift of a real Russian voice)
  fl-tr-s77__xtts-tr-0004.wav             AI (XTTS voice clone of that Turkish speaker)
"""
import os

# every language code that can appear in a file name (the dataset folder still holds older 14-language files)
KNOWN_LANGS = {"uz", "ru", "kk", "ky", "tg", "tr", "az", "en", "ar", "hi", "zh", "de", "es", "fr"}
# languages VoxCheck supports: only these are used for training and evaluation (others are skipped in prepare_data)
LANGS = {"uz", "en", "ru"}
FOCUS_LANGS = LANGS


def supported(fname):
    """True if a file is in a supported language (team / unknown-language recordings are always kept)."""
    lg = lang_of(fname)
    return lg not in KNOWN_LANGS or lg in LANGS
MANIPULATIONS = {"vcpitch", "vcgender", "vcclassic", "vcrobot", "vocoder", "xtts", "replay", "feedback"}
ATTACK_NAMES = {
    "human": "Real human", "edge": "Neural TTS (Microsoft)", "sapi": "Classic TTS (Windows SAPI)",
    "gtts": "Neural TTS (Google)", "piper": "Neural TTS (Piper / VITS)", "xtts": "Voice cloning (XTTS-v2)",
    "vcpitch": "Voice changer: pitch shift", "vcgender": "Voice changer: formant / gender",
    "vcclassic": "Voice changer: speed+pitch", "vcrobot": "Voice changer: robot", "vocoder": "Vocoder resynthesis",
    "replay": "AI played through a real speaker", "feedback": "User-reported mistakes",
    "other": "Other AI",
}


def _parts(fname):
    stem = os.path.splitext(os.path.basename(fname))[0]
    return stem.split("__", 1) if "__" in stem else [stem]


def group_of(fname):
    return _parts(fname)[0]


def attack_of(fname, label):
    if label in (0, "human"):
        return "human"
    p = _parts(fname)
    if len(p) > 1 and p[1].split("-")[0] in MANIPULATIONS:
        return p[1].split("-")[0]
    head = p[0].split("-")[0]
    return head if head in ATTACK_NAMES else "other"


def lang_of(fname):
    p = _parts(fname)
    if len(p) > 1:
        tok = p[1].split("-")
        if tok[0] in KNOWN_LANGS:
            return tok[0]
        if tok[0] == "xtts" and len(tok) > 1 and tok[1] in KNOWN_LANGS:
            return tok[1]
    g = p[0]
    t = g.split("-")
    if g.startswith("fl-") and len(t) > 1:
        return t[1]
    if g.startswith("ls-"):
        return "en"
    if g.startswith(("edge-", "gtts-", "piper-")) and len(t) > 1:
        return t[1] if t[1] in KNOWN_LANGS else "other"
    if g.startswith("sapi-"):
        return "ru" if "irina" in g else "en"
    if g.startswith("me-") or g.startswith("human_") or g.startswith("ai_"):
        return "team"
    return "other"
