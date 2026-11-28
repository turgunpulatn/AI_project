# Data report

| Class | Files | Skipped | Groups | Raw audio (min) | After cleaning (min) | Windows (incl. aug) |
|---|---|---|---|---|---|---|
| human | 9647 | 0 | 4885 | 1711.8 | 1477.3 | 57776 |
| ai | 9064 | 3 | 2800 | 1338.2 | 1184.7 | 52670 |

## Per language (files)

| Language | Human | AI |
|---|---|---|
| en | 3852 | 3807 |
| ru | 2857 | 2915 |
| team | 74 | 52 |
| uz | 2864 | 2287 |

## AI files per attack type

| Attack | Files |
|---|---|
| Neural TTS (Microsoft) | 3225 |
| Voice changer: pitch shift | 1034 |
| Voice cloning (XTTS-v2) | 873 |
| Voice changer: formant / gender | 853 |
| Neural TTS (Piper / VITS) | 840 |
| Vocoder resynthesis | 719 |
| Voice changer: speed+pitch | 553 |
| Neural TTS (Google) | 500 |
| Voice changer: robot | 359 |
| Classic TTS (Windows SAPI) | 60 |
| User-reported mistakes | 21 |
| AI played through a real speaker | 18 |
| Other AI | 6 |

Mean loudness before: -27.4 dBFS (std 10.5) -> after: -20.0 dBFS (std 0.2)
Silence removed: 387.9 min

![before/after](preprocessing_example.png)
