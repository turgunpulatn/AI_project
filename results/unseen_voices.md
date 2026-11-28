# Unseen voices and speakers

Made by `scripts/unseen_eval.py`. None of these voices or speakers is in the training data: FLEURS **test** speakers (training used train + dev), Meta **MMS-TTS** (engine not in training), and **Piper voices not in training**. *mic* = simulated phone speaker -> laptop mic copy.

Model: Ensemble E10 + E11

| Source | uz | ru | en | all | all, mic copy |
|---|---|---|---|---|---|
| Human (FLEURS test speakers), correct | 100.0% (40) | 100.0% (40) | 100.0% (40) | 100.0% (120) | 98.3% (120) |
| MMS-TTS (new engine), caught | 100.0% (30) | 100.0% (30) | 100.0% (30) | 100.0% (90) | 68.9% (90) |
| Piper, new voices, caught | – | 60.0% (30) | 96.4% (28) | 77.6% (58) | 70.7% (58) |

EER (clean): **0.015**

EER (mic copies): **0.138**
