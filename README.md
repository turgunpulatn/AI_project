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

**Not in the repo:** `dataset/` (about 18k audio files plus processed features; it also holds team voice recordings, which are personal data).
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

Dataset used for training: **17,917 recordings** (9,640 human, 8,277 AI) in uz / en / ru, plus team microphone recordings.
Evaluation uses 5-fold cross-validation **grouped by speaker / TTS voice / sentence**, with metrics per recording.
Test folds include the new phone-speaker and laptop-mic copies, which makes the test **harder** than in the earlier 14-language run,
so the CV numbers below are not directly comparable to that run (its table is kept in `results/experiments_14lang.md`).

| # | Approach | Accuracy | ROC-AUC | EER ↓ |
|---|---|---|---|---|
| E1 | Majority baseline | 0.539 | 0.500 | 0.500 |
| E2 | Logistic regression (99 acoustic features) | 0.623 | 0.668 | 0.395 |
| E3 | Random forest (first prototype) | 0.623 | 0.731 | 0.357 |
| E4 | SVM (RBF) | 0.621 | 0.694 | 0.377 |
| E5 | Random forest + augmentation | 0.750 | 0.845 | 0.249 |
| E6 | VoxCNN (own), no augmentation | 0.684 | 0.785 | 0.302 |
| E7 | VoxCNN small + augmentation | 0.847 | 0.925 | 0.156 |
| E8 | VoxCNN + augmentation | 0.877 | 0.947 | 0.126 |
| E10 | VoxCNN-wide + augmentation (own, from scratch) | 0.894 | 0.959 | 0.112 |
| E11 | WavLM-Base+ features (pretrained, frozen) + own MLP | 0.903 | 0.963 | 0.103 |
| E9 | Ensemble RF + VoxCNN-wide | 0.891 | 0.960 | 0.107 |
| **E12** | **Ensemble VoxCNN-wide + WavLM classifier (selected)** | **0.921** | **0.977** | **0.077** |

**Why E12:** it has the lowest EER (0.077). Our own CNN and the WavLM classifier make different mistakes, so averaging them
cuts the error by about 25% compared with either one alone. The trade-off is the reused, frozen WavLM extractor (94M parameters).
Analysis takes ~0.1–0.3 s per recording.

**Hard holdout** (`scripts/holdout_eval.py`): 336 clips never seen in training by either model, scored on the same clips.
They are unseen FLEURS speakers in uz / ru / en, TTS, voice changers, and simulated speaker replay of everything.

| | before the mic fix | **after the mic fix (current)** |
|---|---|---|
| Accuracy | 98.2% | **99.4%** |
| EER, all | 1.4% | **0.4%** |
| EER, simulated speaker replay | 2.4% | **0.4%** |
| uz / ru / en accuracy | 100 / 100 / 98.9% | 100 / 100 / 98.9% |

**Real microphone (AI played from a phone into the laptop):** see `results/mic_channel.md`. Clean files are still caught 18/18.
AI recordings captured through the mic went from 4/14 to 13/14 caught, but these recordings were used in training.
On 11 new mic recordings the model had **never seen**, 7/7 humans were called human, but only 1 of 4 AI voices was caught (2/4 with the playback warning, before the playback fix below).
**AI played into the mic remains the main open problem**, and more real phone→mic recordings are the next step.

**What mattered most** (same CV folds, EER, lower is better):
* Augmentation: our CNN went from 0.302 without augmentation to 0.126 with it (E6 → E8).
* Spectrogram CNN instead of hand-crafted features: 0.249 → 0.112 (E5 → E10).
* Self-supervised WavLM features plus the ensemble with our CNN: 0.103 → 0.077 (E11 → E12).
* Phone-speaker and laptop-mic augmentation: hard-holdout EER 1.4% → 0.4%. Fixing the data of the playback detector
  raised AI played from a phone into the mic from 44% to 86% flagged (see `results/mic_channel.md`).

### Playback detector ("is the voice coming from a speaker?")
A second model (`train_playback.py`, results in `results/playback.md`). It was trained on 148 real replays recorded into this laptop's microphone,
including **18 AI voices played from a phone into the app**, 62 live recordings of a team member, and 2,500 simulated phone and laptop-speaker replays.
Our own VoxCNN (P1) beat the WavLM version (P2): **EER 12.5% on real recordings**, and **85.7% of the phone→mic AI clips are flagged** (out-of-fold).
Before a data bug was fixed, it flagged only 44.4% of them. It accepts 86% of live voices.
In the app, a **microphone** recording that sounds played-back is shown as "Suspicious — voice played from a device".
**Limitation:** the phone→mic clips come from a few phones in one room, so more phones, distances and rooms are the next step.

## Data (what we collected / generated)

Counts are the recordings used to train the current model (from `dataset/processed/manifest.csv`, Uzbek / English / Russian).

| Class | Source | How | Files |
|---|---|---|---|
| Human | **Google FLEURS** (CC-BY-4.0), uz / ru / en | `scripts/import_fleurs.py`: balanced by sentence, 2–25 s | 8,560 |
| Human | **LibriSpeech dev-clean** (CC-BY-4.0), 40 speakers | `scripts/import_librispeech.py` | 997 |
| Human | **Team microphone recordings** and real speaker replays of human voices | app → *Collect data*; `scripts/record_replay.py` | 83 |
| AI | Microsoft Edge neural TTS, 51 voices | `generate_ai.py --engine edge` | 2,692 |
| AI | Piper (VITS) offline neural TTS, 4 voices | `--engine piper` | 600 |
| AI | Google TTS | `--engine gtts` | 500 |
| AI | Windows SAPI | `--engine sapi` | 60 |
| AI | **XTTS-v2 voice cloning** of real FLEURS / LibriSpeech speakers | `--engine xtts` | 873 |
| AI | **Voice changers**: pitch shift 1,034, formant/gender 853, speed+pitch 553, robot 359 | `scripts/make_fakes.py` (ffmpeg rubberband) | 2,799 |
| AI | Vocoder resynthesis (Griffin-Lim) | `scripts/make_fakes.py` | 719 |
| AI | **AI played from a phone or speaker into the laptop mic** (team recordings, scripted replays, "wrong answer" corrections) | app, `scripts/record_replay.py` | 34 |
| | **Total** | | **9,640 human (24.6 h) / 8,277 AI (18.0 h)** |

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
| `scripts/prepare_data.py` | decode → silence removal → loudness norm → augmentation → windows → log-mel + 99 features (streamed to disk) |
| `scripts/share.ps1` | temporary public HTTPS link so classmates can record (see *Working as a team*) |
| `results/` | `experiments.md`, `mic_channel.md`, `playback.md`, `data_report.md`, ROC + preprocessing figures, `training_log.txt` |

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
* Speed+pitch voice changers and Piper/VITS voices are the weakest categories. A linear-frequency or LFCC front-end that keeps
  high-frequency detail is the next experiment to try.
* FLEURS has no speaker IDs (only sentence IDs), so a FLEURS speaker can appear in both train and test folds. LibriSpeech is split by speaker.
* Real phone calls (8 kHz) are simulated, not recorded.
