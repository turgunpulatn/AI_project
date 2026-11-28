# Experiments

5-fold cross-validation grouped by speaker / TTS voice; metrics per recording (clean + degraded versions). EER = equal error rate (lower is better), the standard anti-spoofing metric.

| # | Approach | What changed | Accuracy | F1 (AI) | ROC-AUC | EER | **EER uz/en/ru** | EER (top-half windows) | Params | ms / 2 s window | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | Baseline (majority class) | - | 0.516 | 0.000 | 0.500 | 0.500 ± 0.000 | 0.500 | 0.500 | - | 0.01 |  |
| E2 | Logistic regression | 99 hand-crafted acoustic features | 0.634 | 0.712 | 0.678 | 0.391 ± 0.025 | 0.388 | 0.394 | - | 0.17 |  |
| E3 | Random forest (old demo) | same features, 300 trees | 0.637 | 0.727 | 0.737 | 0.352 ± 0.010 | 0.350 | 0.356 | - | 28.79 |  |
| E4 | SVM (RBF) | same features | 0.634 | 0.710 | 0.706 | 0.364 ± 0.021 | 0.363 | 0.368 | - | 0.44 |  |
| E5 | Random forest + augmentation | + replay/codec/noise augmented copies | 0.752 | 0.740 | 0.850 | 0.246 ± 0.013 | 0.243 | 0.251 | - | 28.74 |  |
| E6 | VoxCNN (own) - no augmentation | log-mel, 4 conv blocks | 0.686 | 0.730 | 0.787 | 0.305 ± 0.012 | 0.302 | 0.306 | 313k | 0.97 |  |
| E7 | VoxCNN (own) - small + augmentation | 4x fewer channels, + aug + SpecAugment | 0.853 | 0.842 | 0.931 | 0.148 ± 0.014 | 0.144 | 0.162 | 83k | 0.85 |  |
| E8 | VoxCNN (own) + augmentation | + replay/codec/noise aug + SpecAugment | 0.881 | 0.872 | 0.950 | 0.122 ± 0.014 | 0.117 | 0.131 | 313k | 0.95 |  |
| E10 | VoxCNN-wide (own) + augmentation | 2x channels (more capacity) | 0.902 | 0.895 | 0.963 | 0.100 ± 0.010 | 0.096 | 0.107 | 1215k | 0.88 |  |
| E11 | WavLM-Base+ (pretrained, frozen) + own MLP | self-supervised speech features (reused) + our classifier | 0.907 | 0.901 | 0.966 | 0.096 ± 0.014 | 0.091 | 0.101 | 94M frozen + 1196k | 25.0 |  |
| E13 | VoxCNN-wide (own) on linear-frequency spectrogram + aug | linear bands keep high-frequency TTS/vocoder artefacts (LFCC front-end) | 0.890 | 0.882 | 0.958 | 0.112 ± 0.013 | 0.108 | 0.120 | 1215k | 2.38 |  |
| E9 | Ensemble E5 + E10 | average of best classical + best CNN | 0.899 | 0.892 | 0.963 | 0.098 ± 0.011 | 0.094 | 0.103 | - + 1215k | 29.62 |  |
| E12 | Ensemble E10 + E11 | average of our CNN + WavLM classifier | 0.927 | 0.922 | 0.980 | 0.071 ± 0.013 | 0.066 | 0.080 | 1215k + 94M frozen + 1196k | 25.88 | ✔ |
| E14 | Ensemble E10 + E13 + E11 | average of mel CNN + linear-spectrogram CNN + WavLM classifier | 0.927 | 0.922 | 0.981 | 0.068 ± 0.012 | 0.064 | 0.078 | 1215k + 1215k + 94M frozen + 1196k | 28.26 |  |

**Selected: E12 – Ensemble E10 + E11** — uz/en/ru EER 0.066 is within 0.005 of the best (0.064) and the model is faster/smaller (AUC 0.980).
Rule: lowest EER on the focus languages (uz/en/ru); models within 0.005 of the best count as equal and the fastest one wins. Focus languages are oversampled x1 in training.

Confusion matrix of the selected model (out-of-fold, rows = true, cols = predicted):

| | pred Human | pred AI |
|---|---|---|
| true Human | 27999 | 1270 |
| true AI | 2856 | 24579 |

![ROC](roc_best.png)

## Selected model: detection per attack type

| Type | Recordings | Correct |
|---|---|---|
| Real human | 29269 | 95.7% |
| Neural TTS (Microsoft) | 9675 | 95.2% |
| User-reported mistakes | 147 | 36.1% |
| Neural TTS (Google) | 1500 | 82.3% |
| Other AI | 34 | 79.4% |
| Neural TTS (Piper / VITS) | 2520 | 85.6% |
| AI played through a real speaker | 126 | 30.2% |
| Classic TTS (Windows SAPI) | 180 | 90.0% |
| Voice changer: speed+pitch | 1675 | 71.8% |
| Voice changer: formant / gender | 2579 | 88.4% |
| Voice changer: pitch shift | 3118 | 95.9% |
| Voice changer: robot | 1089 | 98.7% |
| Vocoder resynthesis | 2173 | 92.0% |
| Voice cloning (XTTS-v2) | 2619 | 82.3% |

## Selected model: accuracy per language

| Language | Recordings | Human correct | AI caught |
|---|---|---|---|
| en | 23025 | 93.5% | 91.4% |
| ru | 17356 | 97.7% | 85.3% |
| team | 830 | 54.1% | 64.0% |
| uz | 15493 | 98.9% | 93.3% |
