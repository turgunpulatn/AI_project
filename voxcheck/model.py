"""VoxCNN: a small convolutional network written and trained from scratch.

Input : log-mel spectrogram  (batch, 1, 64 mel bands, 201 frames = 2 s)
Output: one logit  ->  sigmoid = probability the voice is AI-generated
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class VoxCNN(nn.Module):
    def __init__(self, channels=(16, 32, 64, 128), dropout=0.3):
        super().__init__()
        # strided stem halves time & frequency first -> ~4x fewer FLOPs on CPU
        self.stem = nn.Sequential(nn.Conv2d(1, channels[0], 3, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))
        layers, c_in = [], channels[0]
        for c in channels:
            layers.append(ConvBlock(c_in, c))
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(2 * c_in, 64), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):                      # x: (B, 1, F, T)
        h = self.features(self.stem(x)).mean(dim=2)       # pool over frequency -> (B, C, T')
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)  # temporal mean + max pooling
        return self.head(h).squeeze(1)


class VoxMLP(nn.Module):
    """Small classifier trained by the team on top of frozen WavLM features (E11).
    Normalisation statistics are stored inside the model so the app needs nothing else."""

    def __init__(self, dim, hidden=256, dropout=0.3):
        super().__init__()
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sd", torch.ones(dim))
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):                      # x: (B, dim)
        return self.net((x - self.mu) / self.sd).squeeze(1)


def spec_augment(x, max_f=8, max_t=20):
    """SpecAugment-style masking (training only): one random frequency band and one time span
    per example, fully vectorised (no Python loop -> fast on GPU)."""
    B, _, F, T = x.shape
    dev = x.device
    f = torch.randint(0, max_f + 1, (B, 1), device=dev)
    t = torch.randint(0, max_t + 1, (B, 1), device=dev)
    f0 = (torch.rand(B, 1, device=dev) * (F - f + 1)).long()
    t0 = (torch.rand(B, 1, device=dev) * (T - t + 1)).long()
    fr, tr = torch.arange(F, device=dev)[None], torch.arange(T, device=dev)[None]
    fm = (fr >= f0) & (fr < f0 + f)              # (B, F)
    tm = (tr >= t0) & (tr < t0 + t)              # (B, T)
    return x.masked_fill(fm[:, None, :, None] | tm[:, None, None, :], 0.0)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
