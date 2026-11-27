"""Loads the saved model(s) (never retrains) and scores an audio file.

models/model_meta.json -> "file" is one file or a comma-separated list (ensemble).
  *_wavlm.pt -> frozen WavLM-Base+ features + our MLP
  *.pt       -> VoxCNN on log-mel spectrograms
  *.joblib   -> scikit-learn model on hand-crafted features
Window probabilities are averaged over models, then aggregated per recording with the rule chosen
in training ("mean" or "top" = mean of the most AI-like half of the windows).
"""
import json
import os
import time

import numpy as np

from . import audio as A

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.environ.get("VOX_MODELS", os.path.join(ROOT, "models"))


def aggregate(p_win, agg):
    if agg == "top" and len(p_win) > 1:
        k = int(np.ceil(len(p_win) / 2))
        return float(np.mean(np.sort(p_win)[-k:]))
    return float(np.mean(p_win))


class Detector:
    def __init__(self):
        meta_path = os.path.join(MODELS, "model_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("No trained model. Run: python scripts/prepare_data.py && python train.py")
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.agg = self.meta.get("agg", "mean")
        self.members = []                       # list of (kind, model)
        for name in self.meta["file"].split(","):
            path = os.path.join(MODELS, name.strip())
            if path.endswith(".pt"):
                import torch
                torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
                ck = torch.load(path, map_location="cpu")
                if ck.get("kind") == "mlp":
                    from .model import VoxMLP
                    from . import ssl
                    net = VoxMLP(ck["dim"])
                    net.load_state_dict(ck["state_dict"])
                    ssl.load()                  # WavLM (GPU if available)
                    self.members.append(("mlp", net.eval()))
                else:
                    from .model import VoxCNN
                    net = VoxCNN(channels=tuple(ck["channels"]))
                    net.load_state_dict(ck["state_dict"])
                    self.members.append(("cnn", net.eval()))
            else:
                import joblib
                self.members.append(("sk", joblib.load(path)))
        # warm-up: librosa/numba/torch compile on first call (~7 s) -> do it at startup, not during the demo
        self.playback = self._load_playback()
        rng = np.random.default_rng(0)
        w = A.windows((rng.standard_normal(2 * A.SR) * 0.1).astype(np.float32))
        f = {}
        self._score(w, f)
        self._playback_score(w, f)

    @staticmethod
    def _load_playback():
        """Optional second model: is the voice played through a loudspeaker? (train_playback.py)"""
        path = os.path.join(MODELS, "playback.pt")
        if not os.path.exists(path):
            return None
        import torch
        ck = torch.load(path, map_location="cpu")
        if ck["kind"] == "mlp":
            from .model import VoxMLP
            from . import ssl
            net = VoxMLP(ck["dim"])
            ssl.load()
        else:
            from .model import VoxCNN
            net = VoxCNN(channels=tuple(ck["channels"]))
        net.load_state_dict(ck["state_dict"])
        return ck["kind"], net.eval()

    @staticmethod
    def _feat(kind, wins, f):
        """features computed once per request and shared by all models"""
        import torch
        if kind not in f:
            if kind == "cnn":
                f[kind] = torch.from_numpy(np.stack([A.logmel(w) for w in wins])).unsqueeze(1)
            elif kind == "mlp":
                from . import ssl
                f[kind] = torch.from_numpy(ssl.embed(wins).astype(np.float32))
            else:
                f[kind] = np.stack([A.handcrafted(w) for w in wins])
        return f[kind]

    def _score(self, wins, f):
        scores = []
        for kind, m in self.members:
            x = self._feat(kind, wins, f)
            if kind in ("cnn", "mlp"):
                import torch
                with torch.inference_mode():
                    scores.append(torch.sigmoid(m(x)).numpy())
            else:
                scores.append(m.predict_proba(x)[:, 1])
        return np.mean(scores, axis=0)

    def _playback_score(self, wins, f):
        if self.playback is None:
            return None
        import torch
        kind, m = self.playback
        with torch.inference_mode():
            return torch.sigmoid(m(self._feat(kind, wins, f))).numpy()

    def analyze(self, path):
        t0 = time.perf_counter()
        raw = A.load_audio(path)
        wins = A.prepare(raw)                     # raises ValueError for silence / too short
        feats = {}
        p_win = self._score(wins, feats)
        ai = aggregate(p_win, self.agg)
        margin = abs(ai - 0.5) * 2
        pb = self._playback_score(wins, feats)
        return {
            "playback": None if pb is None else round(float(np.mean(pb)) * 100, 1),
            "prediction": "AI-generated" if ai >= 0.5 else "Human",
            "ai": round(ai * 100, 1),
            "human": round((1 - ai) * 100, 1),
            "confidence": "high" if margin > 0.6 else "medium" if margin > 0.3 else "low",
            "timeline": [round(float(p) * 100, 1) for p in p_win],   # AI % per 2 s window, 1 s step
            "duration_s": round(len(raw) / A.SR, 1),
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "model": self.meta["approach"],
        }
