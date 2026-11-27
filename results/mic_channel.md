# Challenge: AI voice recorded through the laptop microphone was called "Human"

## Symptom
Clean AI files were caught 18/18 (100%), but **10 of 14 AI voices played into the laptop microphone** were called
*Human* (AI score 9–42%). The playback detector also missed them (0–33%).

## Diagnosis: long-term spectrum, share of energy per band (dB)

| Band (Hz) | <100 | 100–200 | 200–300 | 300–500 | 0.5–1k | 1–2k | 2–3k | 3–4k | 4–5k | 5–6k | 6–7k | 7–8k | Noise floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **AI played into the mic (real)** | -17.6 | **-24.9** | -16.7 | -14.1 | -6.0 | **-7.0** | -11.5 | -16.0 | -18.9 | -20.9 | -19.2 | -20.1 | **-21 dB** |
| Live human into the mic (real) | -17.1 | -16.5 | -12.1 | -12.0 | -11.5 | -12.2 | -16.0 | -15.0 | -18.3 | -22.2 | -22.0 | -25.4 | -24 dB |
| Old simulation (laptop speaker) | -41.1 | -15.0 | -8.6 | -5.1 | -6.0 | -12.3 | -18.0 | -19.7 | -20.7 | -22.4 | -28.7 | -43.2 | (none) |
| Training human (FLEURS uz) | -19.9 | -11.4 | -6.6 | -6.5 | -7.0 | -16.4 | -24.7 | -23.5 | -26.2 | -27.3 | -24.4 | -26.9 | -42 dB |
| Training AI (clean TTS) | -27.2 | -7.1 | -5.7 | -6.6 | -8.9 | -15.7 | -22.7 | -23.6 | -24.5 | -23.9 | -23.3 | -26.3 | -135 dB |
| **New simulation (phone speaker + laptop mic)** | -24.2 | -19.1 | -11.8 | -6.0 | -5.6 | -10.0 | -13.8 | -14.2 | -16.4 | -18.8 | -22.6 | -27.5 | -12…-30 dB |

What this shows:
1. The AI was played from a **phone speaker**: almost no energy at 100–300 Hz, with the peak at 1–3 kHz. Our old "replay"
   augmentation modelled a **laptop** speaker, which keeps the bass, so the model had never seen this channel.
2. The laptop mic adds **hiss and automatic gain**, giving a noise floor around -21 dB. In training, a noisy, natural-sounding recording almost
   always meant *human* (FLEURS), while AI was digitally clean (-135 dB floor). The model learned the shortcut
   "sounds like a real microphone → human".

## Fix
* `voxcheck/audio.py`: new `phone_speaker()` (steep high-pass at 200–600 Hz, 1–3 resonance peaks between 0.8 and 4.5 kHz, distortion) and
  `laptop_mic()` (high-pass, hiss + low-frequency rumble at 12–30 dB SNR, AGC that lifts the noise between words).
  `augment()` now applies a speaker replay to 50% of copies (30% phone, 20% laptop speaker) and the laptop-mic channel to
  80% of replayed copies and 40% of the rest. This is applied to **both classes**, so "sounds like a mic" no longer means "human".
* Real microphone recordings (`me-*`: team voices, AI played into the app, "answer was wrong" corrections) get 3× more
  augmented copies.
* Playback detector: trained with simulated phone-speaker replays too, and the AI voices played into the app count as real replays.
  Live voices also go through the laptop-mic simulation, so the mic itself doesn't count as playback.
* Only Uzbek, English and Russian are used (other languages are skipped in `prepare_data.py`).

Results are in the next section.

## Results after retraining (2026-11-26, 21:20)

| Test set | Before | After |
|---|---|---|
| Clean AI files (test_audio/, never trained on) | 18/18 caught | 18/18 caught |
| AI played into the mic, recordings **used in training** (optimistic) | 4/14 caught | 13/14 caught |
| **New mic recordings never seen in training**: AI | – | 1/4 caught (2/4 with the playback warning) |
| **New mic recordings never seen in training**: human | – | 7/7 called human, but 3/7 get a false playback warning |
| Hard holdout (336 clips), EER | 1.4% | 0.4% |
| Cross-validation, "AI played through a real speaker" (out-of-fold) | – | 27.8% caught |

Honest reading: the phone/mic simulation fixed the channel shortcut for the recording setup we trained on, and clean-file
accuracy didn't suffer. But AI played into the mic **from a setup the model hasn't heard** is still mostly missed. The sample is
small (4 AI clips), but the out-of-fold 27.8% points the same way. What's needed is many more real phone→mic recordings
(different phones, distances, rooms), plus a decision threshold calibrated on mic recordings.

Bug found and fixed in this run: `train_playback.py` skipped the AI recordings saved with the "wrong answer" button
(`__feedback-` is in the manipulation list), so only 3 of the 18 real phone→mic AI clips were used as real replays.
After the fix (18 clips used), retraining only the playback detector gave:

| Playback detector | Phone→mic AI flagged | Live voice accepted (real) |
|---|---|---|
| Before the fix | 8/18 = 44.4% (15 of the 18 were unseen) | 87.4% (out-of-fold) |
| **After the fix** | **85.7% out-of-fold** (21 clips) | 86.1% (out-of-fold) |

In the app, a mic recording with playback ≥ 50% is shown as **Suspicious — played from a device**, so most phone→mic AI
is now flagged even when the main model says "human".
