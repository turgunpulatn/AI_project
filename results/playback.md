# Playback (replay) detector

Data: 148 real replays recorded through a speaker into this laptop's mic (18 of them AI voices played from a phone/laptop into the app), 62 live microphone recordings, 2500 recordings with simulated phone / laptop-speaker replay. 5-fold CV grouped by source.

| Model | Acc (all) | EER (all) | **EER (real only)** | Live voice accepted | Real replay caught | Phone→mic AI caught |
|---|---|---|---|---|---|---|
| P1 VoxCNN (own) on log-mel ✔ | 92.7% | 0.073 | 0.125 | 86.1% | 87.9% | 85.7% (21 clips) |
| P2 WavLM features + own MLP | 89.7% | 0.104 | 0.156 | 78.6% | 88.5% | 81.0% (21 clips) |

Selected: **P1 VoxCNN (own) on log-mel**. EER on real recordings 0.125: below the 0.20 threshold, so the detector is **enabled** in the app.
