"""Experiments + model selection + final training.

Every approach is evaluated with the SAME grouped cross-validation folds
(a speaker / TTS voice is never in train and test at once) and the SAME metrics,
computed per recording (window probabilities averaged, exactly like the web app).

Outputs
  results/experiments.csv / experiments.md   comparison table (for the slides)
  results/roc_best.png                        ROC of the selected model
  models/voxcheck_model.pt|.joblib            final model, retrained on all data
  models/model_meta.json                      what was selected and why

Usage:  python train.py              (all experiments)
        python train.py --epochs 8   (faster)
"""
import argparse
import csv
import json
import os
import time

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from voxcheck.model import VoxCNN, spec_augment, count_params
from voxcheck.sk_models import FastSVM

ROOT = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))


# ------------------------------------------------------------------ metrics
def eer(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    two = len(set(y)) == 2
    return dict(acc=accuracy_score(y, pred), f1=f1_score(y, pred, zero_division=0),
                auc=roc_auc_score(y, p) if two else np.nan, eer=eer(y, p) if two else np.nan)


def per_recording(p_win, rec_ids, y_win, agg="mean"):
    """window probabilities -> one score per recording (file x aug version).
    agg="mean": plain average;  agg="top": average of the most AI-like half of the windows
    (a few clearly fake seconds are not diluted by the rest - what a fraud detector wants)."""
    _, inv = np.unique(rec_ids, return_inverse=True)
    n = np.bincount(inv)
    y = (np.bincount(inv, weights=y_win) / n).round().astype(int)
    if agg == "mean":
        return y, np.bincount(inv, weights=p_win) / n
    order = np.lexsort((-p_win, inv))                    # group by recording, highest p first
    inv_s, p_s = inv[order], p_win[order]
    start = np.r_[0, np.cumsum(n)[:-1]]
    rank = np.arange(len(p_s)) - start[inv_s]
    keep = rank < np.ceil(n[inv_s] / 2)
    return y, np.bincount(inv_s[keep], weights=p_s[keep]) / np.bincount(inv_s[keep])


# ------------------------------------------------------------------ models
def sk_model(name):
    if name == "majority":
        return DummyClassifier(strategy="prior")
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"))
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=0)
    if name == "svm":
        return FastSVM(C=3.0)
    raise ValueError(name)


BS = 128 if DEV == "cuda" else 64
AMP = DEV == "cuda"


torch.backends.cudnn.benchmark = DEV == "cuda"


def to_source(mel):
    """Put the whole float16 spectrogram array on the GPU once if it fits (fast batch sampling);
    otherwise keep it as a CPU tensor."""
    t = torch.from_numpy(np.ascontiguousarray(mel))
    if DEV == "cuda":
        free, _ = torch.cuda.mem_get_info()
        if t.numel() * 2 < free * 0.6:
            return t.to(DEV)
    return t


def train_cnn(X, y, cfg, epochs, seed=0, rows=None):
    """X: numpy array or tensor from to_source(); rows: which rows of X to train on (y is already rows-aligned)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not torch.is_tensor(X):
        X = to_source(X)
    rows_t = torch.as_tensor(np.arange(len(y)) if rows is None else rows, device=X.device)
    model = VoxCNN(channels=cfg["channels"], dropout=0.3).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=1e-3)
    steps = epochs * int(np.ceil(len(y) / BS))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.get("lr", 1e-3), total_steps=max(steps, 1))
    pos_w = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1), dtype=torch.float32, device=DEV)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    yt = torch.from_numpy(y.astype(np.float32)).to(DEV)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(yt), device=DEV)
        for i in range(0, len(perm), BS):
            idx = perm[i:i + BS]
            xb = X[rows_t[idx.to(rows_t.device)]].to(DEV, non_blocking=True).float().unsqueeze(1)
            if cfg.get("specaug"):
                xb = spec_augment(xb)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
                loss = loss_fn(model(xb).float(), yt[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    return model.eval()


@torch.no_grad()
def predict_cnn(model, X, rows=None):
    out = []
    rows = np.arange(len(X)) if rows is None else np.asarray(rows)
    for i in range(0, len(rows), 1024):
        r = rows[i:i + 1024]
        if torch.is_tensor(X):
            xb = X[torch.as_tensor(r, device=X.device)].to(DEV).float().unsqueeze(1)
        else:
            xb = torch.from_numpy(np.ascontiguousarray(X[r])).to(DEV).float().unsqueeze(1)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
            out.append(torch.sigmoid(model(xb).float()).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def train_mlp(E, y, epochs, rows, seed=0):
    """E: tensor (N, dim) float16 (GPU if it fits); rows: training rows."""
    from voxcheck.model import VoxMLP
    torch.manual_seed(seed)
    rows_t = torch.as_tensor(rows, device=E.device)
    sub = E[rows_t[torch.randperm(len(rows_t), device=E.device)[:20000]]].float()
    model = VoxMLP(E.shape[1]).to(DEV)
    model.mu.copy_(sub.mean(0).to(DEV)); model.sd.copy_(sub.std(0).clamp_min(1e-3).to(DEV))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    bs = 256
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * int(np.ceil(len(rows) / bs)))
    pos_w = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1), dtype=torch.float32, device=DEV)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    yt = torch.from_numpy(y.astype(np.float32)).to(DEV)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(yt), device=DEV)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb = E[rows_t[idx.to(rows_t.device)]].to(DEV).float()
            xb = xb + 0.05 * torch.randn_like(xb) * model.sd          # light feature noise
            loss = loss_fn(model(xb), yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return model.eval()


@torch.no_grad()
def predict_mlp(model, E, rows):
    out = []
    for i in range(0, len(rows), 4096):
        r = torch.as_tensor(rows[i:i + 4096], device=E.device)
        out.append(torch.sigmoid(model(E[r].to(DEV).float())).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


EXPERIMENTS = [
    # id, name, kind, features, train on augmented copies?, config, what changed
    ("E1", "Baseline (majority class)", "sk", "majority", False, {}, "-"),
    ("E2", "Logistic regression", "sk", "logreg", False, {}, "99 hand-crafted acoustic features"),
    ("E3", "Random forest (old demo)", "sk", "rf", False, {}, "same features, 300 trees"),
    ("E4", "SVM (RBF)", "sk", "svm", False, {}, "same features"),
    ("E5", "Random forest + augmentation", "sk", "rf", True, {}, "+ replay/codec/noise augmented copies"),
    ("E6", "VoxCNN (own) - no augmentation", "cnn", "mel", False,
     {"channels": (16, 32, 64, 128), "specaug": False}, "log-mel, 4 conv blocks"),
    ("E7", "VoxCNN (own) - small + augmentation", "cnn", "mel", True,
     {"channels": (8, 16, 32, 64), "specaug": True}, "4x fewer channels, + aug + SpecAugment"),
    ("E8", "VoxCNN (own) + augmentation", "cnn", "mel", True,
     {"channels": (16, 32, 64, 128), "specaug": True}, "+ replay/codec/noise aug + SpecAugment"),
    ("E10", "VoxCNN-wide (own) + augmentation", "cnn", "mel", True,
     {"channels": (32, 64, 128, 256), "specaug": True}, "2x channels (more capacity)"),
    ("E11", "WavLM-Base+ (pretrained, frozen) + own MLP", "mlp", "wavlm", True, {},
     "self-supervised speech features (reused) + our classifier"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only", nargs="*", help="run only these experiment ids, e.g. E2 E8")
    ap.add_argument("--tol", type=float, default=0.005, help="EER tolerance for preferring a faster model")
    ap.add_argument("--focus-weight", type=int, default=1,
                    help="oversample the focus languages (uz, en, ru) this many times in every training fold")
    ap.add_argument("--reselect", action="store_true",
                    help="skip cross-validation; reuse results/cv_cache.joblib and only redo selection + final model")
    a = ap.parse_args()

    proc = os.path.join(ROOT, "data", "processed")
    d = np.load(os.path.join(proc, "dataset.npz"))
    hc, y, group, file_, aug = d["hc"], d["y"], d["group"], d["file_id"], d["aug"]
    lang, attack = d["lang"], d["attack"]
    mel = np.load(os.path.join(proc, "mel.npy"), mmap_mode="r")   # memory-mapped: no extra RAM copy
    mel_src = to_source(mel) if not a.reselect else mel            # whole array on the GPU if it fits
    print(f"spectrograms on: {mel_src.device if torch.is_tensor(mel_src) else 'cpu (memory-mapped)'}")
    rec = file_.astype(np.int64) * 100 + aug
    emb_src = None
    if os.path.exists(os.path.join(proc, "wavlm.npy")) and not a.reselect:
        emb_src = to_source(np.load(os.path.join(proc, "wavlm.npy")))
        print(f"WavLM features {tuple(emb_src.shape)} on: {emb_src.device}")
    if a.reselect:
        c = joblib.load(os.path.join(ROOT, "results", "cv_cache.joblib"))
        return finish(c["rows"], c["oof"], c["k"], a, data=(mel, hc, y, aug, file_, rec, lang, attack))
    print(f"windows={len(y)}  files={len(np.unique(file_))}  human={int((y == 0).sum())}  ai={int((y == 1).sum())}"
          f"  device={DEV}")

    g_per_class = [len(np.unique(group[y == c])) for c in (0, 1)]
    split_groups = group
    if min(g_per_class) < 2:
        print("!! fewer than 2 speakers/voices in a class -> falling back to per-file splitting "
              "(scores will be optimistic; collect more speakers)")
        split_groups = file_.astype(str)
        g_per_class = [len(np.unique(file_[y == c])) for c in (0, 1)]
    k = max(2, min(a.folds, min(g_per_class)))
    folds = list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42).split(hc, y, split_groups))
    print(f"{k}-fold grouped cross-validation\n")
    from voxcheck.meta import FOCUS_LANGS
    uniq_rec, first_idx = np.unique(rec, return_index=True)
    rec_lang = lang[first_idx]

    def focus_eer(fold_preds):
        """mean EER over folds, using only recordings in the focus languages (uz / en / ru)"""
        out = []
        for p_, (_, te_) in zip(fold_preds, folds):
            yr_, pr_ = per_recording(p_, rec[te_], y[te_])
            lg = rec_lang[np.searchsorted(uniq_rec, np.unique(rec[te_]))]
            m_ = np.isin(lg, list(FOCUS_LANGS))
            if m_.any() and len(set(yr_[m_])) == 2:
                out.append(metrics(yr_[m_], pr_[m_])["eer"])
        return float(np.mean(out)) if out else float("nan")

    focus_rows = np.isin(lang, list(FOCUS_LANGS))

    rows, oof, fold_p = [], {}, {}
    for eid, name, kind, feat, use_aug, cfg, changed in EXPERIMENTS:
        if a.only and eid not in a.only:
            continue
        if kind == "mlp" and emb_src is None:
            print(f"{eid} skipped (no data/processed/wavlm.npy)")
            continue
        t0 = time.time()
        fold_m, ys, ps, fps, rs = [], [], [], [], []
        for tr, te in folds:
            if not use_aug:
                tr = tr[aug[tr] == 0]
            if a.focus_weight > 1:
                tr = np.concatenate([tr] + [tr[focus_rows[tr]]] * (a.focus_weight - 1))
            if kind == "sk":
                m = sk_model(feat).fit(hc[tr], y[tr])
                p = m.predict_proba(hc[te])[:, 1]
            elif kind == "mlp":
                m = train_mlp(emb_src, y[tr], a.epochs, rows=tr)
                p = predict_mlp(m, emb_src, rows=te)
            else:
                m = train_cnn(mel_src, y[tr], cfg, a.epochs, rows=tr)
                p = predict_cnn(m, mel_src, rows=te)
            fps.append(p)
            yr, pr = per_recording(p, rec[te], y[te])
            fold_m.append(metrics(yr, pr))
            ys.append(yr); ps.append(pr); rs.append(np.unique(rec[te]))
        fold_p[eid] = fps
        eer_top = float(np.nanmean([metrics(*per_recording(p_, rec[te_], y[te_], "top"))["eer"]
                                    for p_, (_, te_) in zip(fps, folds)]))
        res = {k_: float(np.nanmean([f[k_] for f in fold_m])) for k_ in fold_m[0]}
        res_std = {k_: float(np.nanstd([f[k_] for f in fold_m])) for k_ in fold_m[0]}
        eer_focus = focus_eer(fps)
        oof[eid] = (np.concatenate(ys), np.concatenate(ps), np.concatenate(rs))

        # inference speed + size on one window
        if kind == "sk":
            ms_list = []
            for _ in range(5):
                t = time.perf_counter(); m.predict_proba(hc[:1]); ms_list.append((time.perf_counter() - t) * 1000)
            ms = float(np.median(ms_list)); params = "-"
        elif kind == "mlp":
            ms, params = 25.0, f"94M frozen + {sum(p_.numel() for p_ in m.parameters()) / 1e3:.0f}k"  # WavLM on GPU dominates
        else:
            ms_list = []
            with torch.no_grad():
                xb = torch.from_numpy(mel[:1]).float().unsqueeze(1).to(DEV)
                for _ in range(5):
                    t = time.perf_counter(); m(xb); ms_list.append((time.perf_counter() - t) * 1000)
            ms = float(np.median(ms_list)); params = f"{count_params(m) / 1e3:.0f}k"
        row = dict(id=eid, approach=name, changed=changed, acc=res["acc"], f1=res["f1"], auc=res["auc"],
                   eer=res["eer"], eer_std=res_std["eer"], params=params, ms_per_window=round(ms, 2),
                   train_s=round(time.time() - t0, 1), kind=kind, feat=feat, use_aug=use_aug, cfg=cfg,
                   eer_top=eer_top, eer_focus=eer_focus)
        rows.append(row)
        print(f"{eid} {name:38s} acc={res['acc']:.3f} f1={res['f1']:.3f} auc={res['auc']:.3f} "
              f"EER={res['eer']:.3f}±{res_std['eer']:.3f}  EER(uz/en/ru)={eer_focus:.3f}  "
              f"EER(top-half)={eer_top:.3f}  ({row['train_s']}s)")

    # ---------------- ensembles = mean probability of two models (no extra training)
    best_of = lambda kind: min((r for r in rows if r["kind"] == kind and r["feat"] != "majority"
                                and not np.isnan(r["eer"])), key=lambda r: r["eer"], default=None)

    def ensemble(eid, r1, r2, changed):
        fold_m, ys, ps, rs, fps = [], [], [], [], []
        for (tr, te), p1, p2 in zip(folds, fold_p[r1["id"]], fold_p[r2["id"]]):
            fps.append((p1 + p2) / 2)
            yr, pr = per_recording(fps[-1], rec[te], y[te])
            fold_m.append(metrics(yr, pr)); ys.append(yr); ps.append(pr); rs.append(np.unique(rec[te]))
        fold_p[eid] = fps
        res = {k_: float(np.nanmean([f[k_] for f in fold_m])) for k_ in fold_m[0]}
        eer_top = float(np.nanmean([metrics(*per_recording(p_, rec[te_], y[te_], "top"))["eer"]
                                    for p_, (_, te_) in zip(fps, folds)]))
        oof[eid] = (np.concatenate(ys), np.concatenate(ps), np.concatenate(rs))
        eer_focus = focus_eer(fps)
        rows.append(dict(id=eid, approach=f"Ensemble {r1['id']} + {r2['id']}", changed=changed, eer_focus=eer_focus,
                         acc=res["acc"], f1=res["f1"], auc=res["auc"], eer=res["eer"],
                         eer_std=float(np.nanstd([f["eer"] for f in fold_m])), eer_top=eer_top,
                         params=f"{r1['params']} + {r2['params']}",
                         ms_per_window=round(r1["ms_per_window"] + r2["ms_per_window"], 2),
                         train_s=0, kind="ens", feat="ens", use_aug=True, cfg={}, members=[r1, r2]))
        r = rows[-1]
        print(f"{eid} {r['approach']:38s} acc={r['acc']:.3f} f1={r['f1']:.3f} auc={r['auc']:.3f} "
              f"EER={r['eer']:.3f}  EER(uz/en/ru)={eer_focus:.3f}  EER(top-half)={eer_top:.3f}")

    sk_r, cnn_r, mlp_r = best_of("sk"), best_of("cnn"), best_of("mlp")
    if sk_r and cnn_r:
        ensemble("E9", sk_r, cnn_r, "average of best classical + best CNN")
    if cnn_r and mlp_r:
        ensemble("E12", cnn_r, mlp_r, "average of our CNN + WavLM classifier")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    joblib.dump({"rows": rows, "oof": oof, "k": k}, os.path.join(ROOT, "results", "cv_cache.joblib"))
    finish(rows, oof, k, a, data=(mel, hc, y, aug, file_, rec, lang, attack), emb=emb_src, mel_dev=mel_src)


def _sel_eer(r):
    """selection metric: EER on the focus languages (uz / en / ru) when available, else overall EER"""
    v = r.get("eer_focus")
    return r["eer"] if v is None or np.isnan(v) else v


def select(rows, tol):
    """Lowest mean EER on the focus languages wins; but if a faster/smaller model is within `tol` of it
    (statistically the same), prefer the faster one (Occam's razor)."""
    cand = [r for r in rows if r["feat"] != "majority" and not np.isnan(r["eer"])]
    min_eer = min(_sel_eer(r) for r in cand)
    near = [r for r in cand if _sel_eer(r) <= min_eer + tol]
    return min(near, key=lambda r: (r["ms_per_window"], _sel_eer(r))), min_eer


def breakdown(yb, pb, rb, rec, lang, attack):
    """Accuracy of the selected model per attack type and per language (out-of-fold recordings).
    Also written to results/breakdown.json for the web app."""
    from voxcheck.meta import ATTACK_NAMES
    uniq, first = np.unique(rec, return_index=True)
    pos = np.searchsorted(uniq, rb)
    r_att, r_lang = attack[first][pos], lang[first][pos]
    pred = (pb >= 0.5).astype(int)
    out = {"attack": [], "lang": []}
    md = ["", "## Selected model: detection per attack type", "",
          "| Type | Recordings | Correct |", "|---|---|---|"]
    for att in sorted(set(r_att), key=lambda x: (x != "human", x)):
        m = r_att == att
        acc = float((pred[m] == yb[m]).mean())
        out["attack"].append({"key": str(att), "name": ATTACK_NAMES.get(att, att), "n": int(m.sum()), "acc": acc})
        md.append(f"| {ATTACK_NAMES.get(att, att)} | {int(m.sum())} | {acc:.1%} |")
    md += ["", "## Selected model: accuracy per language", "",
           "| Language | Recordings | Human correct | AI caught |", "|---|---|---|---|"]
    for lg in sorted(set(r_lang)):
        m = r_lang == lg
        h, s_ = m & (yb == 0), m & (yb == 1)
        hc_ = float((pred[h] == 0).mean()) if h.any() else None
        ac_ = float((pred[s_] == 1).mean()) if s_.any() else None
        out["lang"].append({"key": str(lg), "n": int(m.sum()), "human_acc": hc_, "ai_acc": ac_})
        f = lambda v: "–" if v is None else f"{v:.1%}"
        md.append(f"| {lg} | {int(m.sum())} | {f(hc_)} | {f(ac_)} |")
    json.dump(out, open(os.path.join(ROOT, "results", "breakdown.json"), "w"), indent=1)
    return md


def finish(rows, oof, k, a, data, emb=None, mel_dev=None):
    mel, hc, y, aug, file_, rec, lang, attack = data
    best, min_eer = select(rows, a.tol)
    why = (f"lowest EER on uz/en/ru = {_sel_eer(best):.3f}" if _sel_eer(best) == min_eer else
           f"uz/en/ru EER {_sel_eer(best):.3f} is within {a.tol} of the best ({min_eer:.3f}) and the model is faster/smaller")
    print(f"\nSelected: {best['id']} {best['approach']}  ({why})")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "experiments.csv"), "w", newline="", encoding="utf-8") as f:
        keys = ["id", "approach", "changed", "acc", "f1", "auc", "eer", "eer_std", "eer_focus", "eer_top", "params",
                "ms_per_window", "train_s"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    yb, pb, rb = oof[best["id"]]
    cm = confusion_matrix(yb, (pb >= 0.5).astype(int), labels=[0, 1])
    md = ["# Experiments", "", f"{k}-fold cross-validation grouped by speaker / TTS voice; metrics per recording "
          "(clean + degraded versions). EER = equal error rate (lower is better), the standard anti-spoofing metric.",
          "", "| # | Approach | What changed | Accuracy | F1 (AI) | ROC-AUC | EER | **EER uz/en/ru** | EER (top-half windows) | Params | ms / 2 s window | Selected |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sel = "✔" if r is best else ""
        md.append(f"| {r['id']} | {r['approach']} | {r['changed']} | {r['acc']:.3f} | {r['f1']:.3f} | "
                  f"{r['auc']:.3f} | {r['eer']:.3f} ± {r['eer_std']:.3f} | {r.get('eer_focus', float('nan')):.3f} | {r.get('eer_top', float('nan')):.3f} | {r['params']} | {r['ms_per_window']} | {sel} |")
    md += ["", f"**Selected: {best['id']} – {best['approach']}** — {why} (AUC {best['auc']:.3f}).",
           f"Rule: lowest EER on the focus languages (uz/en/ru); models within {a.tol} of the best count as equal and "
           "the fastest one wins. Focus languages are oversampled x" + str(getattr(a, "focus_weight", 1)) + " in training.",
           "", "Confusion matrix of the selected model (out-of-fold, rows = true, cols = predicted):", "",
           "| | pred Human | pred AI |", "|---|---|---|",
           f"| true Human | {cm[0, 0]} | {cm[0, 1]} |", f"| true AI | {cm[1, 0]} | {cm[1, 1]} |", "",
           "![ROC](roc_best.png)"]
    md += breakdown(yb, pb, rb, rec, lang, attack)
    open(os.path.join(ROOT, "results", "experiments.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5))
    for r in rows:
        if r["feat"] == "majority":
            continue
        fpr, tpr, _ = roc_curve(*oof[r["id"]][:2])
        plt.plot(fpr, tpr, lw=2.5 if r is best else 1, label=f"{r['id']} AUC={r['auc']:.2f}")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate (human called AI)"); plt.ylabel("True positive rate (AI caught)")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "results", "roc_best.png"), dpi=110)

    # ---------------- final model on ALL data
    print("Retraining selected model on all data ...")
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    for old in os.listdir(os.path.join(ROOT, "models")):
        if old.startswith("voxcheck_model"):
            os.remove(os.path.join(ROOT, "models", old))

    def fit_save(r, suffix=""):
        idx = np.arange(len(y)) if r["use_aug"] else np.where(aug == 0)[0]
        if r["kind"] == "sk":
            m = sk_model(r["feat"]).fit(hc[idx], y[idx])
            path = os.path.join(ROOT, "models", f"voxcheck_model{suffix}.joblib")
            joblib.dump(m, path)
        elif r["kind"] == "mlp":
            from voxcheck import ssl
            src = emb if emb is not None else to_source(np.load(os.path.join(ROOT, "data", "processed", "wavlm.npy")))
            m = train_mlp(src, y[idx], a.epochs, rows=idx)
            path = os.path.join(ROOT, "models", f"voxcheck_model{suffix}_wavlm.pt")
            torch.save({"kind": "mlp", "dim": ssl.DIM, "ssl": ssl.MODEL_ID, "layers": list(ssl.LAYERS),
                        "state_dict": m.cpu().state_dict()}, path)
        else:
            src = mel_dev if mel_dev is not None else mel
            m = train_cnn(src, y[idx], r["cfg"], a.epochs, rows=idx)
            path = os.path.join(ROOT, "models", f"voxcheck_model{suffix}.pt")
            torch.save({"state_dict": m.cpu().state_dict(), "channels": list(r["cfg"]["channels"])}, path)
        print(f"Saved -> {path}")
        return os.path.basename(path), int(len(idx))

    if best["kind"] == "ens":
        files = [fit_save(m_, f"_{i}")[0] for i, m_ in enumerate(best["members"])]
        path, n_idx = ",".join(files), len(y)
    else:
        path, n_idx = fit_save(best)
    meta = {k_: best[k_] for k_ in ("id", "approach", "kind", "feat", "acc", "f1", "auc", "eer")}
    agg = "top" if best.get("eer_top", 1) < best["eer"] else "mean"
    print(f"window aggregation: {agg}  (EER mean={best['eer']:.3f}, top-half={best.get('eer_top', float('nan')):.3f})")
    meta.update(file=path, trained_on_windows=n_idx, folds=k, agg=agg, eer_top=best.get("eer_top"),
                eer_focus=best.get("eer_focus"),
                n_files=int(len(np.unique(file_))), trained_at=time.strftime("%Y-%m-%d %H:%M"))
    json.dump(meta, open(os.path.join(ROOT, "models", "model_meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
