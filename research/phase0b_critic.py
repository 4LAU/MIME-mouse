"""Phase 0b: geometric-vocabulary critic vs RF-OOB shared-space diagnostic.

Resolves a confound left open by Phase 0 (research/phase0_critic.py). Phase 0
fed a small transformer critic raw per-step (dx,dy) and found: OOF AUC 0.632
(RF is 0.757), weak synthetic-only Spearman(critic,RF)=0.196, and the critic
was nearly blind (|corr| 0.002-0.031) to the RF's actual top tell (mean_jerk,
angular_velocity_std, curvature_mean, curvature_std) while keying on
LOW-RF-importance features (max_velocity, max_acceleration, std_velocity,
std_acceleration).

The open question: did the critic ignore jerk/curvature/angular-velocity
because a transformer reading raw (dx,dy) cannot COMPUTE those higher-order
finite-difference-of-a-finite-difference quantities from only 4000 samples of
noisy data, or because a learned critic fundamentally can't see the RF's
tell? Phase 0b removes the confound by handing the critic the RF's own
per-step geometric vocabulary (speed, acceleration, jerk, curvature, angular
velocity) as extra input channels, computed via the exact finite-difference
recipe features.py uses (see features.py extract_features, lines ~52-134),
under a constant dt=1/125 assumption (the assembled data has no raw
timestamps, only dx/dy, so dt cannot be reconstructed exactly; this is an
approximation the task explicitly allows).

Reuses research/phase0_critic_data.npz (built by Phase 0) verbatim -- no
path regeneration, no model loading, no GPU generation. Only:
  - dxdy (4000,256,2) padded per-step deltas, pad_mask (4000,256) bool
  - y (4000,) labels (0=human,1=synth), same 2000/2000 order as Phase 0
  - feats18 (4000,18) RF input features, rf_oob_scores (4000,) RF-OOB score
  - feature_names (18,) matching features.FEATURE_NAMES

Safety:
  - Never reads data/human_eval_features.npy.
  - Never loads/touches training/candi_polar_flow_best.pt (not needed here).
  - torch.cuda.set_per_process_memory_fraction(0.80) if cuda used.

Usage:
    CPU smoke (tiny, fast, no CV):
        .venv/Scripts/python.exe research/phase0b_critic.py --smoke

    Full run (5-fold CV):
        .venv/Scripts/python.exe research/phase0b_critic.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from experiments._common import get_device
from features import FEATURE_NAMES

REPO_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = REPO_ROOT / "research"

DT = 1.0 / 125.0
EPS = 1e-6
MAX_LEN = 256

D_MODEL = 192
N_LAYERS = 4
N_HEAD = 4
D_FF = 384
DROPOUT = 0.15
LR = 3e-4
MAX_EPOCHS = 40
N_FOLDS = 5
PATIENCE = 6

CHANNEL_NAMES = ["dx", "dy", "speed", "acc", "jerk", "curvature", "angular_velocity"]
N_CHANNELS = len(CHANNEL_NAMES)


# ---------------------------------------------------------------------------
# Per-step geometric channel construction (mirrors features.py exactly,
# under the constant dt=1/125 assumption, since raw timestamps are not
# available from the saved (dx,dy)-only cache).
# ---------------------------------------------------------------------------
def compute_channels_for_path(dx: np.ndarray, dy: np.ndarray, dt: float = DT,
                               eps: float = EPS):
    """dx, dy: 1D float64 arrays of VALID (unpadded) per-step deltas for one
    path, length L. Returns (L, N_CHANNELS) float64 array aligned 1:1 to the
    dx/dy steps, and the count of non-finite entries clamped away.

    Shortening from successive np.diff calls (acc: L-1, jerk: L-2,
    curvature/omega: L-1) is repaired by LEFT edge-padding (repeating the
    first computable value) back up to length L, so every channel lines up
    on the same step index as dx/dy and the pre-existing pad_mask (which
    only marks true end-of-sequence padding) stays valid unmodified -- no
    separate per-channel mask is needed.
    """
    L = len(dx)
    vx = dx / dt
    vy = dy / dt
    speed = np.sqrt(dx ** 2 + dy ** 2 + eps ** 2) / dt  # length L, matches features.py's ds/dt

    if L >= 2:
        acc = np.diff(speed) / dt  # length L-1, matches features.py's scalar acc = diff(speed)/dt2
    else:
        acc = np.zeros(0)
    if len(acc) >= 2:
        jerk = np.diff(acc) / dt  # length L-2, matches features.py's jerk = diff(acc)/dt2[:-1]
    else:
        jerk = np.zeros(0)

    if L >= 2:
        ax = np.diff(vx) / dt
        ay = np.diff(vy) / dt
        speed_mid = np.maximum(speed[:-1], eps)
        cross = np.abs(vx[:-1] * ay - vy[:-1] * ax)
        curvature = np.clip(cross / (speed_mid ** 3), 0, 1e6)  # length L-1, matches features.py
    else:
        curvature = np.zeros(0)

    angles = np.arctan2(dy, dx)
    if L >= 2:
        angle_diff = np.diff(angles)
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        omega = np.clip(angle_diff / dt, -1e6, 1e6)  # length L-1, matches features.py's omega
    else:
        omega = np.zeros(0)

    def left_pad_edge(arr, target_len):
        if len(arr) == target_len:
            return arr
        n_pad = target_len - len(arr)
        if len(arr) == 0:
            return np.zeros(target_len)
        return np.concatenate([np.full(n_pad, arr[0]), arr])

    acc_p = left_pad_edge(acc, L)
    jerk_p = left_pad_edge(jerk, L)
    curvature_p = left_pad_edge(curvature, L)
    omega_p = left_pad_edge(omega, L)

    stacked = np.stack([dx, dy, speed, acc_p, jerk_p, curvature_p, omega_p], axis=1)
    n_nonfinite = int(np.sum(~np.isfinite(stacked)))
    if n_nonfinite:
        stacked = np.nan_to_num(stacked, nan=0.0, posinf=1e6, neginf=-1e6)
    return stacked.astype(np.float64), n_nonfinite


def build_all_channels(dxdy: np.ndarray, pad_mask: np.ndarray):
    n = dxdy.shape[0]
    channels = np.zeros((n, MAX_LEN, N_CHANNELS), dtype=np.float64)
    lengths = pad_mask.sum(axis=1)
    total_nonfinite = 0
    total_valid_elems = 0
    for i in range(n):
        L = int(lengths[i])
        if L == 0:
            continue
        dx = dxdy[i, :L, 0].astype(np.float64)
        dy = dxdy[i, :L, 1].astype(np.float64)
        stacked, n_nf = compute_channels_for_path(dx, dy)
        channels[i, :L, :] = stacked
        total_nonfinite += n_nf
        total_valid_elems += L * N_CHANNELS
    return channels, total_nonfinite, total_valid_elems


def robust_standardize(channels: np.ndarray, pad_mask: np.ndarray, eps: float = 1e-6):
    """Per-channel median/IQR standardization over VALID steps only, same
    scale applied across both classes (scale computed pooling all paths),
    followed by a signed-log compression: z' = sign(z) * log1p(|z|).

    Curvature and jerk in particular are heavy (power-law-like) tailed --
    a single sharp direction reversal or resample artifact produces a
    curvature/jerk value orders of magnitude above the bulk of the
    distribution (observed: standardized curvature up to ~2.6e6, jerk raw
    up to ~3.5e8). Feeding that directly into the transformer overflowed
    attention/softmax and produced NaN losses within the first epoch on the
    full run. A hard clip would map all such outliers to one value and
    destroy their relative ordering (which may itself carry signal, since
    tail behavior is plausibly part of what the RF's std_jerk/curvature_std
    features are keying on). log1p compression keeps the transform
    monotonic and near-linear for typical in-distribution steps (log1p(z)
    approx z for small z) while bounding extreme tails to a small numeric
    range (log1p(2.6e6) approx 14.8), which resolved the NaNs."""
    n, L, C = channels.shape
    standardized = channels.copy()
    scales = np.zeros((C, 2), dtype=np.float64)  # (median, iqr) per channel
    for c in range(C):
        vals = channels[:, :, c][pad_mask]
        med = float(np.median(vals))
        q75, q25 = np.percentile(vals, [75, 25])
        iqr = max(float(q75 - q25), eps)
        scales[c] = [med, iqr]
        z = (channels[:, :, c] - med) / iqr
        standardized[:, :, c] = np.sign(z) * np.log1p(np.abs(z))
    standardized = standardized * pad_mask[:, :, None]
    return standardized, scales


# ---------------------------------------------------------------------------
# Critic model (mirrors Phase 0's PathCritic, generalized to N_CHANNELS
# input channels and the slightly larger capacity spec'd for Phase 0b).
# ---------------------------------------------------------------------------
class GeoPathCritic(nn.Module):
    def __init__(self, n_channels=N_CHANNELS, d_model=D_MODEL, n_layers=N_LAYERS,
                 n_head=N_HEAD, d_ff=D_FF, dropout=DROPOUT, max_len=MAX_LEN):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, pad_mask):
        # x: (B, L, C), pad_mask: (B, L) bool, True=valid
        B, L, _ = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.input_proj(x) + self.pos_embed(pos)
        h = self.encoder(h, src_key_padding_mask=~pad_mask)
        mask_f = pad_mask.unsqueeze(-1).float()
        pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        logit = self.head(pooled).squeeze(-1)
        return logit


# ---------------------------------------------------------------------------
# Training / CV (structurally identical to Phase 0's run_cv/train_one_fold).
# ---------------------------------------------------------------------------
def train_one_fold(X_tr, mask_tr, y_tr, X_va, mask_va, y_va, device,
                    max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=0,
                    verbose_prefix=""):
    torch.manual_seed(seed)
    model = GeoPathCritic().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    Xtr = torch.from_numpy(X_tr).float().to(device)
    Mtr = torch.from_numpy(mask_tr).to(device)
    Ytr = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    Xva = torch.from_numpy(X_va).float().to(device)
    Mva = torch.from_numpy(mask_va).to(device)

    n_tr = len(Xtr)
    batch_size = 128
    best_auc = -1.0
    best_logits_va = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        total_loss = 0.0
        for b0 in range(0, n_tr, batch_size):
            idx = perm[b0:b0 + batch_size]
            opt.zero_grad()
            logits = model(Xtr[idx], Mtr[idx])
            loss = bce(logits, Ytr[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        total_loss /= n_tr

        model.eval()
        with torch.no_grad():
            logits_va = model(Xva, Mva).cpu().numpy()
        auc_va = roc_auc_score(y_va, logits_va)
        if auc_va > best_auc:
            best_auc = auc_va
            best_logits_va = logits_va.copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if verbose_prefix and (epoch % 5 == 0 or epoch == max_epochs - 1):
            print(f"{verbose_prefix} epoch={epoch} loss={total_loss:.4f} "
                  f"val_auc={auc_va:.4f} best={best_auc:.4f}", flush=True)
        if epochs_no_improve >= patience:
            break

    return model, best_auc, best_logits_va


def run_cv(X, pad_mask, y, device, n_folds=N_FOLDS, seed=0, verbose=True):
    n = len(y)
    oof_logits = np.zeros(n, dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_aucs = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n), y)):
        t0 = time.perf_counter()
        _, fold_auc, logits_va = train_one_fold(
            X[tr_idx], pad_mask[tr_idx], y[tr_idx],
            X[va_idx], pad_mask[va_idx], y[va_idx],
            device, seed=seed + fold,
            verbose_prefix=f"[fold {fold}]" if verbose else "")
        oof_logits[va_idx] = logits_va
        fold_aucs.append(fold_auc)
        print(f"[phase0b] fold {fold}: val_auc={fold_auc:.4f} "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)
    oof_probs = 1.0 / (1.0 + np.exp(-oof_logits))
    oof_auc = roc_auc_score(y, oof_probs)
    return oof_probs, oof_auc, fold_aucs


def train_full_model(X, pad_mask, y, device, epochs=15, seed=0):
    torch.manual_seed(seed)
    model = GeoPathCritic().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(X).float().to(device)
    M = torch.from_numpy(pad_mask).to(device)
    Y = torch.from_numpy(y.astype(np.float32)).to(device)
    n = len(Xt)
    batch_size = 128
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for b0 in range(0, n, batch_size):
            idx = perm[b0:b0 + batch_size]
            opt.zero_grad()
            logits = model(Xt[idx], M[idx])
            loss = bce(logits, Y[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        total_loss /= n
        print(f"[phase0b] full-fit epoch={epoch} loss={total_loss:.4f}", flush=True)
    return model


# ---------------------------------------------------------------------------
# CPU smoke test: tiny subset, confirm finite channels + one forward/backward.
# ---------------------------------------------------------------------------
def run_smoke():
    print("[phase0b][smoke] loading npz...", flush=True)
    d = np.load(RESEARCH_DIR / "phase0_critic_data.npz", allow_pickle=True)
    dxdy = d["dxdy"]
    pad_mask = d["pad_mask"]
    y = d["y"]
    print(f"[phase0b][smoke] npz keys={d.files}", flush=True)
    for k in d.files:
        print(f"  {k}: shape={d[k].shape} dtype={d[k].dtype}", flush=True)

    n_human_smoke = 25
    n_synth_smoke = 25
    human_idx = np.arange(n_human_smoke)
    synth_idx = np.arange(2000, 2000 + n_synth_smoke)
    idx = np.concatenate([human_idx, synth_idx])

    sub_dxdy = dxdy[idx]
    sub_mask = pad_mask[idx]
    sub_y = y[idx]

    channels, n_nonfinite, n_valid_elems = build_all_channels(sub_dxdy, sub_mask)
    print(f"[phase0b][smoke] channels shape={channels.shape} "
          f"n_nonfinite_clamped={n_nonfinite}/{n_valid_elems}", flush=True)
    assert np.all(np.isfinite(channels)), "smoke: non-finite values survived clamping"

    std_channels, scales = robust_standardize(channels, sub_mask)
    print(f"[phase0b][smoke] standardized, scales (median,iqr) per channel:", flush=True)
    for name, (med, iqr) in zip(CHANNEL_NAMES, scales):
        print(f"    {name:16s} median={med:.6g} iqr={iqr:.6g}", flush=True)
    assert np.all(np.isfinite(std_channels)), "smoke: non-finite after standardization"

    device = torch.device("cpu")
    model = GeoPathCritic().to(device)
    X = torch.from_numpy(std_channels).float().to(device)
    M = torch.from_numpy(sub_mask).to(device)
    Y = torch.from_numpy(sub_y.astype(np.float32)).to(device)

    logits = model(X, M)
    loss = nn.BCEWithLogitsLoss()(logits, Y)
    loss.backward()

    grad_norm = 0.0
    n_params_with_grad = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += float(p.grad.norm().item() ** 2)
            n_params_with_grad += 1
    grad_norm = grad_norm ** 0.5
    print(f"[phase0b][smoke] forward+backward OK: loss={loss.item():.4f} "
          f"grad_norm={grad_norm:.4f} params_with_grad={n_params_with_grad}", flush=True)
    assert grad_norm > 0.0, "smoke: zero gradient norm, grads not flowing"
    assert np.isfinite(loss.item()), "smoke: non-finite loss"
    print("[phase0b][smoke] PASSED", flush=True)


# ---------------------------------------------------------------------------
# Full pipeline.
# ---------------------------------------------------------------------------
def run_full(device: torch.device):
    t_start = time.perf_counter()

    print("[phase0b] loading research/phase0_critic_data.npz...", flush=True)
    d = np.load(RESEARCH_DIR / "phase0_critic_data.npz", allow_pickle=True)
    print(f"[phase0b] npz keys={d.files}", flush=True)
    for k in d.files:
        print(f"  {k}: shape={d[k].shape} dtype={d[k].dtype}", flush=True)

    dxdy = d["dxdy"]
    pad_mask = d["pad_mask"]
    y = d["y"]
    feats18 = d["feats18"]
    rf_oob_scores = d["rf_oob_scores"]
    feature_names_saved = [str(s) for s in d["feature_names"]]
    assert feature_names_saved == FEATURE_NAMES, "feature name order mismatch vs features.py"

    n = len(y)
    print(f"[phase0b] N paths={n} (expect 4000: 2000 human + 2000 synth)", flush=True)

    print("[phase0b] building geometric channels from saved dxdy (no path "
          "regeneration)...", flush=True)
    t0 = time.perf_counter()
    channels, n_nonfinite, n_valid_elems = build_all_channels(dxdy, pad_mask)
    print(f"[phase0b] channels built in {time.perf_counter()-t0:.1f}s "
          f"shape={channels.shape} n_nonfinite_clamped={n_nonfinite}/"
          f"{n_valid_elems} ({100.0*n_nonfinite/max(n_valid_elems,1):.4f}%)",
          flush=True)

    std_channels, scales = robust_standardize(channels, pad_mask)
    print("[phase0b] per-channel robust (median/IQR) standardization scales:", flush=True)
    for name, (med, iqr) in zip(CHANNEL_NAMES, scales):
        print(f"    {name:16s} median={med:.6g} iqr={iqr:.6g}", flush=True)
    assert np.all(np.isfinite(std_channels)), "non-finite values after standardization, STOP"

    # --- RF: reuse saved OOB scores for all correlations; refit ONLY to
    # recover feature_importances_ (not saved in the npz), then sanity-check
    # the refit's OOB decision function matches the saved array (same
    # recipe/seed/data => should be deterministic / near-identical).
    print("[phase0b] refitting RF (same recipe as Phase 0) solely to recover "
          "feature_importances_; using the SAVED rf_oob_scores for all "
          "correlations below, per task spec...", flush=True)
    clf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1,
                                  random_state=42)
    clf.fit(feats18, y)
    refit_oob = clf.oob_decision_function_[:, 1]
    refit_auc = roc_auc_score(y, refit_oob)
    saved_auc = roc_auc_score(y, rf_oob_scores)
    max_abs_diff = float(np.max(np.abs(refit_oob - rf_oob_scores)))
    print(f"[phase0b] RF refit sanity: refit_oob_auc={refit_auc:.4f} "
          f"saved_oob_auc={saved_auc:.4f} max_abs_diff(refit_vs_saved)="
          f"{max_abs_diff:.6f}", flush=True)

    # --- critic 5-fold CV ---
    print(f"[phase0b] running {N_FOLDS}-fold critic CV on {n} paths, "
          f"{N_CHANNELS} channels ({CHANNEL_NAMES})...", flush=True)
    oof_probs, oof_auc, fold_aucs = run_cv(std_channels, pad_mask, y, device,
                                            n_folds=N_FOLDS, seed=0)
    print(f"[phase0b] critic OOF AUC: {oof_auc:.4f} (fold aucs: "
          f"{[f'{a:.4f}' for a in fold_aucs]})", flush=True)

    # --- shared-space Spearman correlations (using SAVED rf_oob_scores) ---
    is_synth = (y == 1)
    is_human = (y == 0)
    rho_synth, p_synth = spearmanr(oof_probs[is_synth], rf_oob_scores[is_synth])
    rho_human, p_human = spearmanr(oof_probs[is_human], rf_oob_scores[is_human])
    rho_pooled, p_pooled = spearmanr(oof_probs, rf_oob_scores)
    print(f"[phase0b] Spearman(critic_oof, rf_oob) SYNTH-only: rho={rho_synth:.4f} "
          f"p={p_synth:.3e}", flush=True)
    print(f"[phase0b] Spearman(critic_oof, rf_oob) HUMAN-only: rho={rho_human:.4f} "
          f"p={p_human:.3e}", flush=True)
    print(f"[phase0b] Spearman(critic_oof, rf_oob) POOLED:     rho={rho_pooled:.4f} "
          f"p={p_pooled:.3e}", flush=True)

    # --- feature-alignment table ---
    importances = clf.feature_importances_
    feat_spearman = []
    for i, name in enumerate(FEATURE_NAMES):
        rho, _ = spearmanr(oof_probs, feats18[:, i])
        feat_spearman.append(abs(rho))
    order = np.argsort(importances)[::-1]
    table_rows = []
    print("\n=== PHASE 0b FEATURE ALIGNMENT TABLE (sorted by RF importance) ===")
    print(f"{'feature':24s} {'rf_importance':>14s} {'|spearman(critic,feat)|':>24s}")
    for idx in order:
        name = FEATURE_NAMES[idx]
        imp = float(importances[idx])
        rho_abs = float(feat_spearman[idx])
        table_rows.append({"feature": name, "rf_importance": imp,
                            "abs_spearman_critic": rho_abs})
        print(f"{name:24s} {imp:14.4f} {rho_abs:24.4f}")

    # --- train full-data critic + save checkpoint ---
    print("[phase0b] training full-data critic for checkpoint save...", flush=True)
    full_model = train_full_model(std_channels, pad_mask, y, device, epochs=15)
    ckpt_path = RESEARCH_DIR / "phase0b_critic.pt"
    torch.save({
        "model_state_dict": full_model.state_dict(),
        "channel_names": CHANNEL_NAMES,
        "channel_scales": scales.tolist(),
        "max_len": MAX_LEN,
        "d_model": D_MODEL, "n_layers": N_LAYERS, "n_head": N_HEAD,
        "d_ff": D_FF, "dropout": DROPOUT,
    }, ckpt_path)
    print(f"[phase0b] saved critic checkpoint to {ckpt_path}", flush=True)

    results = {
        "n_paths_per_class": int(n // 2),
        "n_channels": N_CHANNELS,
        "channel_names": CHANNEL_NAMES,
        "n_nonfinite_clamped": int(n_nonfinite),
        "n_valid_elems": int(n_valid_elems),
        "pct_nonfinite_clamped": 100.0 * n_nonfinite / max(n_valid_elems, 1),
        "rf_refit_sanity": {
            "refit_oob_auc": float(refit_auc),
            "saved_oob_auc": float(saved_auc),
            "max_abs_diff_refit_vs_saved": max_abs_diff,
        },
        "critic_oof_auc": float(oof_auc),
        "critic_fold_aucs": [float(a) for a in fold_aucs],
        "spearman_synth_only": {"rho": float(rho_synth), "p": float(p_synth)},
        "spearman_human_only": {"rho": float(rho_human), "p": float(p_human)},
        "spearman_pooled": {"rho": float(rho_pooled), "p": float(p_pooled)},
        "feature_alignment_table": table_rows,
        "elapsed_sec": time.perf_counter() - t_start,
        # Phase 0 numbers, copied verbatim for side-by-side comparison.
        "phase0_reference": {
            "critic_oof_auc": 0.632,
            "rf_oob_auc": 0.757,
            "spearman_synth_only": 0.196,
            "spearman_human_only": 0.248,
            "spearman_pooled": 0.311,
        },
    }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                     help="Tiny CPU smoke test: channel build + one forward/backward.")
    ap.add_argument("--device", type=str, default=None,
                     help="cuda/cpu; default auto-detect via get_device()")
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"[phase0b] device={device}", flush=True)

    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.80)
        print("[phase0b] WDDM spill guard: set_per_process_memory_fraction(0.80)",
              flush=True)

    results = run_full(device)

    out_json = RESEARCH_DIR / "phase0b_critic_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[phase0b] wrote results to {out_json}", flush=True)

    print("\n=== PHASE 0b SUMMARY (vs PHASE 0) ===")
    print(f"{'metric':40s} {'phase0':>10s} {'phase0b':>10s}")
    print(f"{'critic OOF AUC':40s} {0.632:10.4f} {results['critic_oof_auc']:10.4f}")
    print(f"{'RF-OOB AUC (unchanged detector)':40s} {0.757:10.4f} "
          f"{results['rf_refit_sanity']['saved_oob_auc']:10.4f}")
    print(f"{'Spearman synth-only':40s} {0.196:10.4f} "
          f"{results['spearman_synth_only']['rho']:10.4f}")
    print(f"{'Spearman human-only':40s} {0.248:10.4f} "
          f"{results['spearman_human_only']['rho']:10.4f}")
    print(f"{'Spearman pooled':40s} {0.311:10.4f} "
          f"{results['spearman_pooled']['rho']:10.4f}")
    print(f"\nnon-finite steps clamped: {results['n_nonfinite_clamped']} / "
          f"{results['n_valid_elems']} ({results['pct_nonfinite_clamped']:.4f}%)")


if __name__ == "__main__":
    main()
