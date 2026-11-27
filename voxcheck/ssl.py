"""Self-supervised speech features from WavLM-Base+ (Microsoft, MIT licence) - a REUSED, pretrained,
FROZEN feature extractor. Only the small classifier on top is trained by the team (see train.py, E11).

Per 2 s window: mean and standard deviation over time of hidden layers 4, 8 and 12
-> 6 x 768 = 4608 numbers. Lower/middle layers keep acoustic detail (vocoder / replay artefacts),
the top layer keeps more phonetic information.
"""
import numpy as np

MODEL_ID = "microsoft/wavlm-base-plus"
LAYERS = (4, 8, 12)
DIM = 2 * len(LAYERS) * 768

_model = None
_device = None


def available():
    import importlib.util
    return importlib.util.find_spec("transformers") is not None


def load(device=None):
    global _model, _device
    if _model is None:
        import torch
        from transformers import WavLMModel
        _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _model = WavLMModel.from_pretrained(MODEL_ID).to(_device).eval()
        if _device == "cuda":
            _model = _model.half()
    return _model


def embed(windows, batch=48):
    """windows: (N, 32000) float32 at 16 kHz -> (N, DIM) float16"""
    import torch
    m = load()
    out = []
    for i in range(0, len(windows), batch):
        x = torch.from_numpy(np.ascontiguousarray(windows[i:i + batch], dtype=np.float32))
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-5)
        x = x.to(_device, dtype=next(m.parameters()).dtype)
        with torch.inference_mode():
            hs = m(x, output_hidden_states=True).hidden_states
        feats = []
        for layer in LAYERS:
            h = hs[layer].float()
            feats += [h.mean(1), h.std(1)]
        out.append(torch.cat(feats, 1).cpu().numpy().astype(np.float16))
    return np.concatenate(out) if out else np.zeros((0, DIM), np.float16)
