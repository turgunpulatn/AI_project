# VoxCheck — anti-fraud AI voice detector (Uzbek, English, Russian)

VoxCheck tells whether a voice recording is a **real person** or **fake**: text-to-speech, a cloned voice,
or a voice changed with a voice-changer app. Phone scammers use all three ("mom, I lost my phone, send money",
"this is your bank's security service..."). VoxCheck works in **Uzbek, English and Russian**.

Pipeline: **own multilingual dataset → cleaning + attack simulation → 10 compared models → own CNN → web app**

## Quick start (run the demo)

```powershell
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000, then record from the mic, upload a file, or try a sample.
On start the app runs a self-test (one sample through decoding and the model) and prints `Self-test: OK`.
If port 5000 is already taken, for example by an older VoxCheck window, the app refuses to start and prints the process ID to stop.
Use `python app.py --port 5050` to choose another port. `http://127.0.0.1:5000/health` shows whether the server can analyse audio.
The page records the microphone as plain WAV, so recordings never depend on a browser codec. If ffmpeg fails, the server decodes with libsndfile instead.
The app loads the saved model from `models/` and never retrains during the demo.
It runs on CPU, so no GPU is needed for the demo.
**ffmpeg** is not in the repo (too large). Install it once with `winget install Gyan.FFmpeg`, or put `ffmpeg.exe` (and `ffplay.exe`) into `tools/ffmpeg/`.
Without ffmpeg the app still works for microphone recordings and wav / mp3 / ogg / flac uploads, but not for webm / m4a.

**Not in the repo:** `dataset/` (about 18.7k audio files plus processed features; it also holds team voice recordings, which are personal data).
The trained models in `models/` are included, so the demo runs right after cloning. To retrain, get `dataset/` from the team's shared drive.

Tests: `python tests/test_voxcheck.py` checks decoding, silence and short-clip rejection, augmentation, file-name metadata and all web endpoints.

## Working as a team

**Code:** clone the repo, work on your own branch, open a pull request:
```powershell
git clone https://github.com/turgunpulatn/AI_project.git
cd AI_project
pip install -r requirements.txt
git checkout -b <your-name>/<what-you-change>
```

**Recording voices from your own laptop or phone (no install needed).** The project owner runs:
```powershell
# once: put cloudflared.exe from https://github.com/cloudflare/cloudflared/releases into tools\ (no admin)
#       or: winget install --id Cloudflare.cloudflared
powershell -ExecutionPolicy Bypass -File scripts\share.ps1
```
This prints a temporary `https://….trycloudflare.com` link and a team **access code**. No domain or account is needed.
Open the link, go to **Collect data**, enter the code, tick the consent box, record and save.
Recordings are saved in `dataset/` on the owner's laptop, and each one adds a line to `dataset/consent_log.csv`.
The link works only while the owner's window is open. HTTPS is required because browsers only allow the microphone on `https://` (or `localhost`).

**What to record (most useful for the model):** your own voice with label *human*, and AI voices played **from your phone** into the
microphone with label *ai*, in Uzbek, English and Russian. Use one speaker name per person.

**Training** runs on the owner's GPU (`python run_pipeline.py --skip-data`, about 1 hour). Collected recordings are used in the next run.

## Results

**Languages: Uzbek, English, Russian only.** Other languages are skipped in `prepare_data.py`.

Dataset used for training: **18,711 recordings** (9,647 human, 9,064 AI) in uz / en / ru, plus team microphone recordings.
Evaluation uses 5-fold cross-validation **grouped by speaker / TTS voice / sentence**, with metrics per recording.
Test folds include the phone-speaker and laptop-mic copies, so the test is **harder** than in the earlier 14-language run
(its table is kept in `results/experiments_14lang.md`). Full table with params and speed: `results/experiments.md`.

| # | Approach | Accuracy | ROC-AUC | EER ↓ |
|---|---|---|---|---|
| E1 | Majority baseline | 0.516 | 0.500 | 0.500 |
| E2 | Logistic regression (99 acoustic features) | 0.634 | 0.678 | 0.391 |
| E3 | Random forest (first prototype) | 0.637 | 0.737 | 0.352 |
| E4 | SVM (RBF) | 0.634 | 0.706 | 0.364 |
| E5 | Random forest + augmentation | 0.752 | 0.850 | 0.246 |
| E6 | VoxCNN (own), no augmentation | 0.686 | 0.787 | 0.305 |
| E7 | VoxCNN small + augmentation | 0.853 | 0.931 | 0.148 |
| E8 | VoxCNN + augmentation | 0.881 | 0.950 | 0.122 |
| E10 | VoxCNN-wide + augmentation (own, from scratch, log-mel) | 0.902 | 0.963 | 0.100 |
| E11 | WavLM-Base+ features (pretrained, frozen) + own MLP | 0.907 | 0.966 | 0.096 |
| E13 | VoxCNN-wide on a **linear-frequency** spectrogram (LFCC-style front-end) | 0.890 | 0.958 | 0.112 |
| E9 | Ensemble RF + VoxCNN-wide | 0.899 | 0.963 | 0.098 |
| **E12** | **Ensemble VoxCNN-wide + WavLM classifier (selected)** | **0.927** | **0.980** | **0.071** |
| E14 | Ensemble VoxCNN-wide + linear-spectrogram CNN + WavLM classifier | 0.927 | 0.981 | 0.068 |

**Why E12:** our own CNN and the WavLM classifier make different mistakes, so averaging them cuts the error by about 25%
compared with either one alone. E14 adds the linear-spectrogram CNN and is only 0.002 better on uz/en/ru (0.064 vs 0.066),
which is within our "equal" margin of 0.005, so the smaller and faster E12 wins. The linear front-end on its own (E13) was worse
than log-mel (E10), so the high-frequency detail it keeps did not pay off on this data.
The trade-off of E12 is the reused, frozen WavLM extractor (94M parameters). Analysis takes ~0.1–0.3 s per recording.

**Data version matters more than the model.** Before this run, Russian and Uzbek AI speech came from only 2 native Edge voices each.
We added 26 more voice identities (multilingual Edge voices reading ru / uz text, Turkish / Azerbaijani / Kazakh voices reading
Uzbek, and 8 more English Piper voices: +773 AI files). Same model (E12), same CV protocol:

| | before (17,917 files) | **after (18,711 files)** |
|---|---|---|
| CV EER, E12 | 0.077 | **0.071** |
| Unseen voices, clean EER | 2.6% | **1.5%** |
| Unseen voices, phone→mic EER | 17.9% | **13.8%** |

**Holdout** (`scripts/holdout_eval.py`): 336 clips never seen in training by either model, scored on the same clips.
The clips are new, but the 8 Microsoft TTS voices also appear in training, and the "speaker replay" copies use the same
simulator as the training augmentation. So this shows the model is not just memorizing files. It does **not** show the model
works on new voices or real devices. The real-microphone test below is the harder test.

**Unseen voices and speakers** (`scripts/unseen_eval.py`, full table in `results/unseen_voices.md`). Nothing here is in training:
new people from the FLEURS **test** split (training used train + dev), **Meta MMS-TTS**, a TTS engine we never trained on, and
**6 Piper voices** we never trained on. Each clip was also scored as a simulated "phone speaker → laptop mic" copy.

| | clean | phone→mic copy |
|---|---|---|
| New human speakers called human (uz / ru / en, 40 each) | **100%** | 98.3% |
| MMS-TTS caught (uz / ru / en 100%) | **100%** | 68.9% |
| New Piper voices caught (ru 60.0%, en 96.4%) | **77.6%** | 70.7% |
| EER | **1.5%** | 13.8% |

So the model generalises to a new engine on clean audio, but about 30% of new voices played through a phone are still missed,
and the two unseen Russian Piper voices are the weakest case.
More real phone→mic recordings of **many different** TTS voices are the most useful data to add.
They are unseen FLEURS speakers in uz / ru / en, TTS, voice changers, and simulated speaker replay of everything.

| | before the mic fix | after the mic fix | **current (more voices)** |
|---|---|---|---|
| Accuracy | 98.2% | 99.4% | **99.4%** |
| EER, all | 1.4% | 0.4% | **1.0%** |
| EER, simulated speaker replay | 2.4% | 0.4% | **0.4%** |
| uz / ru / en accuracy | 100 / 100 / 98.9% | 100 / 100 / 98.9% | **100 / 100 / 98.9%** |

On this easy holdout both models get 2 of 336 clips wrong; the small EER change is a ranking difference
on 336 clips and is within noise. The unseen-voice test above is the more informative one.

**Real microphone (AI played from a phone into the laptop):** see `results/mic_channel.md`. Clean files are still caught 18/18.
AI recordings captured through the mic went from 4/14 to 13/14 caught, but these recordings were used in training.
On 11 new mic recordings the model had **never seen**, 7/7 humans were called human, but only 1 of 4 AI voices was caught (2/4 with the playback warning, before the playback fix below).
**AI played into the mic remains the main open problem**, and more real phone→mic recordings are the next step.

**What mattered most** (same CV folds, EER, lower is better):
* Augmentation: our CNN went from 0.305 without augmentation to 0.122 with it (E6 → E8).
* Spectrogram CNN instead of hand-crafted features: 0.246 → 0.100 (E5 → E10).
* Self-supervised WavLM features plus the ensemble with our CNN: 0.096 → 0.071 (E11 → E12).
* More TTS voice identities for ru / uz: unseen-voice EER 2.6% → 1.5% clean, 17.9% → 13.8% phone→mic.
* Phone-speaker and laptop-mic augmentation: hard-holdout EER 1.4% → 0.4%. Fixing the data of the playback detector
  raised AI played from a phone into the mic from 44% to 86% flagged (see `results/mic_channel.md`).

### Playback detector ("is the voice coming from a speaker?")
A second model (`train_playback.py`, results in `results/playback.md`). It was trained on 148 real replays recorded into this laptop's microphone,
including **18 AI voices played from a phone into the app**, 62 live recordings of a team member, and 2,500 simulated phone and laptop-speaker replays.
Our own VoxCNN (P1) beat the WavLM version (P2): **EER 12.5% on real recordings**, and **85.7% of the phone→mic AI clips are flagged** (out-of-fold).
Before a data bug was fixed, it flagged only 44.4% of them. It accepts 86% of live voices.
In the app, a **microphone** recording that sounds played-back is shown as "Suspicious — voice played from a device".
Only a clear playback score (≥ 70%) overrides a human verdict. Between 50% and 70% the verdict stays *human* with a softer
"maybe played from a device" note, because live voices in a new room often score near 50% (the detector learned "live" from
few speakers). Pressing **It was a human** after a microphone recording saves the clip as a live voice for the next
`train_playback.py` run.
**Limitation:** the phone→mic clips come from a few phones in one room, so more phones, distances and rooms are the next step.
Also, 55 of the live clips are one team member. Each take is its own CV group, so that voice and room appear in both
train and test folds, and the 12.5% EER is likely optimistic. Live recordings from more people will fix this.
Only feedback clips recorded with the **microphone** count as real speaker replays. Uploaded files were never played
through a speaker, so the app now tags feedback as `feedback-mic-*` or `feedback-file-*`.

**Before retraining the playback detector, record more replays.** The model in `models/` was trained on 148 real replays.
Only 37 replay recordings plus 22 phone clips are left in `dataset/`; the others were removed in the uz/en/ru cleanup.
A test retrain on that data on 2026-11-28 reached only 23.4% EER on real recordings. That is above the 20% limit, so
`train_playback.py` would switch the warning off in the app. Run `scripts/record_replay.py` and record phone→mic clips first.

### Known limitations of the evaluation
* FLEURS has no speaker IDs, so human FLEURS clips are grouped by **sentence**. The same FLEURS speaker can be in both
  train and test folds, which makes CV numbers a little optimistic for the human class.
* Voice-changer and vocoder fakes are labelled *AI*. For anti-fraud that is the right label (the voice is not
  what the caller really sounds like), but they are signal-processing fakes, not neural TTS.

## Data (what we collected / generated)

Counts are the recordings used to train the current model (from `dataset/processed/manifest.csv`, Uzbek / English / Russian).

| Class | Source | How | Files |
|---|---|---|---|
| Human | **Google FLEURS** (CC-BY-4.0), uz / ru / en | `scripts/import_fleurs.py`: balanced by sentence, 2–25 s | 8,560 |
| Human | **LibriSpeech dev-clean** (CC-BY-4.0), 40 speakers | `scripts/import_librispeech.py` | 1,002 |
| Human | **Team microphone recordings** and real speaker replays of human voices | app → *Collect data*; `scripts/record_replay.py` | 74 |
| AI | Microsoft Edge neural TTS, 62 voices (incl. multilingual and tr / az / kk voices reading ru / uz) | `generate_ai.py --engine edge`, `--engine edge-extra` | 3,225 |
| AI | Piper (VITS) offline neural TTS, 12 voices (6 more are kept out for the unseen test) | `--engine piper` | 840 |
| AI | Google TTS | `--engine gtts` | 500 |
| AI | Windows SAPI | `--engine sapi` | 60 |
| AI | **XTTS-v2 voice cloning** of real FLEURS / LibriSpeech speakers | `--engine xtts` | 873 |
| AI | **Voice changers**: pitch shift 1,034, formant/gender 853, speed+pitch 553, robot 359 | `scripts/make_fakes.py` (ffmpeg rubberband) | 2,799 |
| AI | Vocoder resynthesis (Griffin-Lim) | `scripts/make_fakes.py` | 719 |
| AI | **AI played from a phone or speaker into the laptop mic** (team recordings, scripted replays, "wrong answer" corrections) | app, `scripts/record_replay.py` | 34 |
| | **Total** | | **9,647 human (24.6 h after cleaning) / 9,064 AI (19.7 h)** |

TTS text comes from FLEURS / LibriSpeech transcripts in the same language, so both classes say the same kind of sentences.
A fake made from a real recording keeps the source's group, so the original and its fake are always in the same fold.

## Reproduce

```powershell
# downloads (outside the repo, ../downloads_raw/):
#   LibriSpeech dev-clean.tar.gz  https://www.openslr.org/12
#   FLEURS data/<lang>/audio/dev.tar.gz + data/<lang>/dev.tsv  https://huggingface.co/datasets/google/fleurs
python run_pipeline.py              # import + generate + prepare + train (re-runnable, skips existing files)
python scripts/record_live.py --name <you> --minutes 5   # live mic speech (for the playback detector)
python scripts/record_replay.py --n 300                  # real speaker replays (laptop plays audio!)
python train_playback.py                                 # playback detector
python scripts/holdout_eval.py --build --models models   # holdout check
python scripts/unseen_eval.py --build                     # unseen voices / speakers check
python run_pipeline.py --skip-data  # only prepare_data + train
```
Training on an RTX 4060 takes ~70 min for all 10 experiments. Install the CUDA build of PyTorch:
`pip install torch --index-url https://download.pytorch.org/whl/cu126`.

## Project structure

```
VoxCheck/
├── app.py               web app (Check a voice · Collect data · How it works)
├── train.py             10 experiments, model selection, final model
├── train_playback.py    second model: is the voice played from a device?
├── run_pipeline.py      one command that rebuilds data + models
├── requirements.txt
├── voxcheck/            core code used by both training and the app
├── scripts/             data collection, evaluation, share.ps1
├── templates/ static/   web page and demo samples
├── models/              trained models loaded by the app
├── results/             experiment tables, figures, reports (for the slides)
├── texts/               sentences to read aloud and TTS texts
├── test_audio/          extra test clips never used in training
├── prototype/           first prototype (random forest on MFCC)
├── tests/               automated tests
├── dataset/             all audio + processed features   (local only, not in git)
└── tools/               ffmpeg + cloudflared              (local only, not in git)
```

| Path | What |
|---|---|
| `voxcheck/audio.py` | shared preprocessing (same in training and app); augmentation incl. **phone / laptop speaker replay, laptop mic, room reverb**, phone line, codecs |
| `voxcheck/model.py` | **VoxCNN**: strided stem + 4 conv blocks + mean/max pooling, written from scratch |
| `voxcheck/meta.py` | file-name conventions → group, language, attack type |
| `voxcheck/detector.py` | loads the saved models once, scores 2 s windows, returns probability + timeline |
| `scripts/import_*.py`, `scripts/generate_ai.py`, `scripts/make_fakes.py` | data collection / generation |
| `scripts/prepare_data.py` | decode → silence removal → loudness norm → augmentation → windows → log-mel, linear spectrogram, WavLM and 99 features (streamed to disk) |
| `scripts/holdout_eval.py`, `scripts/unseen_eval.py` | holdout test and the unseen voices / speakers test |
| `scripts/share.ps1` | temporary public HTTPS link so classmates can record (see *Working as a team*) |
| `results/` | `experiments.md`, `mic_channel.md`, `playback.md`, `data_report.md`, `unseen_voices.md`, ROC + preprocessing figures, `training_log.txt` (log of the run behind the current model) |

## Reused (disclosed)

* **Data:** FLEURS (Conneau et al., 2022, CC-BY-4.0); LibriSpeech (Panayotov et al., 2015, CC-BY-4.0).
* **Tools used only to *generate* fakes:** Microsoft Edge TTS (edge-tts), Google TTS (gTTS), Piper voices (rhasspy/piper-voices),
  Coqui XTTS-v2 (CPML licence, non-commercial), Windows SAPI, ffmpeg rubberband.
* **Libraries:** PyTorch, scikit-learn, librosa, scipy, Flask, ffmpeg.
* **WavLM-Base+** (Microsoft, MIT licence) is used as a frozen, pretrained feature extractor in E11/E12. Only the classifier on top is trained by us.
  VoxCNN (E10) is trained from random initialisation on our data. No pretrained deepfake detector is used.

## Known limitations / future work

* **AI played into the microphone from an unfamiliar phone or room is often missed by the main model.** On 4 new phone→mic AI clips
  it caught only 1 (the playback detector flags most of them as "played from a device"). More real phone→mic recordings from many
  phones and rooms, plus a decision threshold calibrated on microphone recordings, are the next steps (`results/mic_channel.md`).
* The playback detector learned "live voice" mostly from one team member (62 recordings), so other people sometimes get a false
  "played from a device" warning. Live recordings from classmates (`scripts/share.ps1`) fix this.
* Unseen fake generators, such as ElevenLabs or new cloning models, may be harder. More engines in `dataset/ai` improve generalisation.
* Speed+pitch voice changers and Piper/VITS voices are the weakest categories. A linear-frequency (LFCC-style) front-end
  was tried (E13) and did not beat log-mel. Fine-tuning WavLM end-to-end (instead of keeping it frozen) is the next model experiment.
* FLEURS has no speaker IDs (only sentence IDs), so a FLEURS speaker can appear in both train and test folds. LibriSpeech is split by speaker.
* Real phone calls (8 kHz) are simulated, not recorded.
