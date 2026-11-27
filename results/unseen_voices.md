# Unseen voices and speakers

Made by `scripts/unseen_eval.py`. None of these voices or speakers is in the training data: FLEURS **test** speakers (training used train + dev), Meta **MMS-TTS** (engine not in training), and **Piper voices not in training**. *mic* = simulated phone speaker -> laptop mic copy.

Model: Ensemble E10 + E11

| Source | uz | ru | en | all | all, mic copy |
|---|---|---|---|---|---|
| Human (FLEURS test speakers), correct | 100.0% (40) | 100.0% (40) | 100.0% (40) | 100.0% (120) | 95.0% (120) |
| MMS-TTS (new engine), caught | 100.0% (30) | 100.0% (30) | 90.0% (30) | 96.7% (90) | 61.1% (90) |
| Piper, new voices, caught | – | 66.7% (30) | 82.1% (28) | 74.1% (58) | 58.6% (58) |

EER (clean): **0.026**

EER (mic copies): **0.179**
