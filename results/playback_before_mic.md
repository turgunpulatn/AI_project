# Playback (replay) detector

Data: 130 real replays recorded through this laptop's speaker+mic, 62 live microphone recordings, 2500 recordings with simulated loudspeaker replay. 5-fold CV grouped by source.

| Model | Acc (all) | EER (all) | **EER (real only)** | Live voice accepted | Real replay caught |
|---|---|---|---|---|---|
| P1 VoxCNN (own) on log-mel ✔ | 95.5% | 0.045 | 0.098 | 90.9% | 89.6% |
| P2 WavLM features + own MLP | 92.2% | 0.079 | 0.118 | 84.2% | 91.2% |

Selected: **P1 VoxCNN (own) on log-mel**. EER on real recordings 0.098: below the 0.20 threshold, so the detector is **enabled** in the app.
