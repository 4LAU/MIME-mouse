"""Phase 0: whole-path critic vs RF-OOB shared-space diagnostic (go/no-go).

Tests whether a small transformer critic that reads a raw (dx,dy) sequence
latches onto the same structure the real RF detector cares about, or a
mirage the RF ignores. No adversarial loop -- just: (1) can a critic tell
human from synthetic at all (5-fold CV AUC), (2) does its per-path score
rank-correlate with the RF's per-path OOB score, especially among synthetic
paths, (3) does it weigh the same 18 features the RF weighs most.

Reuses, verbatim in spirit:
  - research/phase_a_baseline.py: model load, sample_guided_flow,
    decode_polar, build_trajectory, make_specs (the exact 0.752 generation
    config: steps=200, guide=0.15, cfg=0, no rounding for this run).
  - features.py: resample_trajectory, extract_features, extract_feature_matrix.
  - training/train_events_polar_grpo.py build_val_human_features: the honest
    validation-human reconstruction recipe (seed-42 eval-index exclusion,
    seed-20260709 draw, SIR feature-match screening). Reimplemented here
    WITHOUT the human_eval_features.npy assert (that file is PROTECTED and
    is never opened by this script).

Safety:
  - Never reads data/human_eval_features.npy.
  - Never writes to training/candi_polar_flow_best.pt (load-only, eval()).
  - torch.cuda.set_per_process_memory_fraction(0.80) WDDM spill guard.

Usage:
    CPU smoke (tiny, fast):
        .venv/Scripts/python.exe research/phase0_critic.py --smoke

    Full GPU run:
        .venv/Scripts/python.exe research/phase0_critic.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from experiments._common import DurationModel, get_device
from features import FEATURE_NAMES, extract_features, extract_feature_matrix, resample_trajectory

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_a_baseline import (  # noqa: E402
    CKPT_NAME, CFG_SCALE, DUR_STD, GUIDE, N_SAMPLE_STEPS,
    build_trajectory, decode_polar, make_specs, sample_guided_flow,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
TRAIN_DIR = REPO_ROOT / "training"

EVAL_HUMANS_SEED = 42
EVAL_HUMANS_N = 2000
VAL_SEED = 20260709
VAL_N = 2000

MAX_LEN = 256
D_MODEL = 128
N_LAYERS = 3
N_HEAD = 4
D_FF = 256
DROPOUT = 0.15
LR = 3e-4
MAX_EPOCHS = 40
N_FOLDS = 5

RF_SANITY_LOW = 0.70
RF_SANITY_HIGH = 0.82


# ---------------------------------------------------------------------------
# Human validation-path reconstruction (raw paths, not just features).
# Mirrors training/train_events_polar_grpo.py build_val_human_features
# exactly, EXCEPT it never opens data/human_eval_features.npy -- that file
# is used ONLY there as a post-hoc sanity assert on the eval_idx draw, and
# is skipped here since PROTECTED. The seed-42 draw itself is identical, so
# the excluded index SET is identical regardless of whether the assert runs.
# ---------------------------------------------------------------------------
def reconstruct_human_val_paths(n_val: int, val_seed: int, verbose: bool = True):
    """NOTE: n_val must be VAL_N (2000), the exact recipe parameter -- the
    draw-candidate pool size is min(n_val + 1000, len(remaining)), so a
    smaller n_val does NOT give a prefix of the n_val=2000 draw, it gives a
    disjoint random draw entirely. Callers wanting fewer rows (e.g. --smoke)
    must slice the returned 2000 down themselves, never pass a smaller
    n_val here."""
    offsets = np.load(TRAIN_DIR / "full_pool_offsets.npy")
    flat = np.load(TRAIN_DIR / "pool_flat_i16.npy", mmap_mode="r")
    t_arr = np.load(TRAIN_DIR / "pool_t_rel_f32.npy", mmap_mode="r")
    n_pool = len(offsets) - 1

    def pool_traj(idx: int):
        s, e = int(offsets[idx]), int(offsets[idx + 1])
        xy = flat[s:e].astype(np.float64)
        ts = t_arr[s:e].astype(np.float64)
        return [(float(xy[j, 0]), float(xy[j, 1]), float(ts[j])) for j in range(len(xy))]

    eval_idx = np.random.default_rng(EVAL_HUMANS_SEED).choice(
        n_pool, size=EVAL_HUMANS_N, replace=False)
    mask = np.ones(n_pool, dtype=bool)
    mask[eval_idx] = False
    remaining = np.flatnonzero(mask)

    draw = np.random.default_rng(val_seed).choice(
        remaining, size=min(n_val + 1000, len(remaining)), replace=False)

    sir_features = np.load(DATA_DIR / "human_ref_features_sir.npy")
    sir_keys = {np.round(row, 6).tobytes() for row in sir_features}

    resampled_paths = []
    feat_rows = []
    n_sir_hits = 0
    n_none = 0
    for idx in draw:
        traj = pool_traj(int(idx))
        resampled = resample_trajectory(traj)
        f = extract_features(resampled)
        if f is None or not np.all(np.isfinite(f)):
            n_none += 1
            continue
        if np.round(f, 6).tobytes() in sir_keys:
            n_sir_hits += 1
            continue
        resampled_paths.append(resampled)
        feat_rows.append(f)
        if len(resampled_paths) >= n_val:
            break

    if len(resampled_paths) < n_val:
        raise RuntimeError(f"could only build {len(resampled_paths)}/{n_val} "
                            "validation human rows")

    feats = np.asarray(feat_rows, dtype=np.float64)
    if verbose:
        print(f"[phase0] reconstructed {len(resampled_paths)} human val paths "
              f"(n_none={n_none} n_sir_hits={n_sir_hits})", flush=True)
    return resampled_paths, feats


def verify_against_cache(feats: np.ndarray, n_check: int, atol: float = 1e-5):
    """Correctness anchor, ORDER-INVARIANT (set-membership at 6dp), not
    positional allclose.

    Diagnostic history (2026-07-19): a positional row-for-row allclose
    against data/human_val_features_grpo.npy FAILED (max_abs_diff ~1.5e8)
    on first attempt. Investigation (scratch reimplementation of
    build_val_human_features's exact logic, independent of this script,
    matched this script's output bit-for-bit) ruled out a logic bug in the
    reconstruction. A full n=2000 run showed the reconstructed feature-row
    SET and the cached file's row SET have zero-discrepancy 2000/2000
    overlap at 6-decimal rounding -- i.e. every single human trajectory and
    its feature vector is exactly right; only the ENUMERATION ORDER differs
    from whatever produced the cache (git history shows the cache file's
    mtime predates the commit that added build_val_human_features by ~6h,
    consistent with the cache having been built by an uncommitted working
    copy of the script, possibly with a different loop/batching order, e.g.
    parallel workers, before the identical logic was committed). Since
    order carries no statistical meaning for an i.i.d. validation sample
    fed into RF/critic training, set-membership is the correct invariant to
    check, not position.
    """
    cached = np.load(DATA_DIR / "human_val_features_grpo.npy")
    n = min(n_check, len(feats))

    cached_keys = {tuple(np.round(row, 6)) for row in cached}
    mine_keys = [tuple(np.round(row, 6)) for row in feats[:n]]
    n_match = sum(1 for k in mine_keys if k in cached_keys)
    match_frac = n_match / max(n, 1)

    # Informational only: positional agreement, expected to be low/partial
    # given the known order difference documented above.
    n_pos = min(n, len(cached))
    positional_frac = float(np.mean(np.all(
        np.isclose(feats[:n_pos], cached[:n_pos], atol=atol, rtol=1e-5), axis=1
    ))) if n_pos > 0 else 0.0

    print(f"[phase0] correctness anchor (order-invariant, set-membership): "
          f"{n_match}/{n} reconstructed rows found in "
          f"human_val_features_grpo.npy ({match_frac:.4%}); positional "
          f"row-for-row agreement (informational only): {positional_frac:.2%}",
          flush=True)

    if match_frac < 1.0:
        raise AssertionError(
            f"only {n_match}/{n} reconstructed human feature rows match a "
            "row in data/human_val_features_grpo.npy (set-membership) -- "
            "reconstruction is wrong, STOP")
    return match_frac, positional_frac


# ---------------------------------------------------------------------------
# Synthetic path generation (phase_a_baseline's exact 0.752-config path).
# ---------------------------------------------------------------------------
def load_model_on_device(ckpt_name: str, device: torch.device):
    """Same as phase_a_baseline.load_model but honors an explicit device
    instead of always picking get_device()'s (GPU-preferring) choice --
    needed so --smoke can force CPU without touching the GPU at all."""
    ckpt_path = TRAIN_DIR / ckpt_name
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    from models.candi import CANDIModel
    model = CANDIModel(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    data_scale = ckpt["data_scale"]
    polar = ckpt.get("polar", False)
    pred_type = ckpt.get("pred_type", "x0")
    assert polar, "expected polar checkpoint"
    assert pred_type == "flow", "this script's decode path assumes flow pred_type"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[phase0] loaded {ckpt_path} device={device} params={n_params:,} "
          f"epoch={ckpt.get('epoch')} data_scale={data_scale}", flush=True)
    return model, data_scale, device, ckpt["config"]["max_seq_len"]


def generate_synth_paths(n: int, seed: int, ckpt_name: str = CKPT_NAME,
                          device: torch.device | None = None):
    if device is None:
        device = get_device()
    model, data_scale, dev, max_seq_len_cfg = load_model_on_device(ckpt_name, device)
    model.max_seq_len_cfg = max_seq_len_cfg
    duration_model = DurationModel(TRAIN_DIR, std_mult=DUR_STD)
    specs = make_specs(n, seed)

    results = [None] * len(specs)
    pending = []
    for idx, (sx, sy, ex, ey) in enumerate(specs):
        dx = ex - sx
        dy = ey - sy
        total_dist = math.hypot(dx, dy)
        if total_dist < 1.0:
            results[idx] = [(sx, sy, 0.0), (ex, ey, 0.008)]
            continue
        log_dist = math.log(total_dist)
        angle = math.atan2(dy, dx)
        duration = duration_model.sample(log_dist)
        log_dur = math.log(duration)
        seq_len = max(5, min(int(round(duration * 125.0)), model.max_seq_len_cfg))
        pending.append({
            "idx": idx, "seq_len": seq_len, "angle": angle,
            "cond": [log_dist, log_dur, math.cos(angle), math.sin(angle)],
            "total_dist": total_dist, "dx": dx, "dy": dy,
            "sx": sx, "sy": sy, "ex": ex, "ey": ey,
        })

    groups: dict = {}
    for item in pending:
        groups.setdefault(item["seq_len"], []).append(item)

    EVAL_BATCH = 128
    for seq_len, items in groups.items():
        for c0 in range(0, len(items), EVAL_BATCH):
            chunk = items[c0:c0 + EVAL_BATCH]
            cond = torch.tensor([it["cond"] for it in chunk],
                                 dtype=torch.float32, device=dev)
            tcos = np.array([math.cos(it["angle"]) for it in chunk])
            tsin = np.array([math.sin(it["angle"]) for it in chunk])
            with torch.no_grad():
                raw, stall = sample_guided_flow(
                    model, data_scale, dev, cond, seq_len, tcos, tsin,
                    n_steps=N_SAMPLE_STEPS, cfg_scale=CFG_SCALE, guide=GUIDE,
                )
            raw_all = raw.cpu().numpy()
            stall_all = stall.cpu().numpy()
            for b, it in enumerate(chunk):
                cum_x, cum_y = decode_polar(raw_all[b], stall_all[b], data_scale)
                results[it["idx"]] = build_trajectory(
                    cum_x, cum_y, stall_all[b], seq_len,
                    it["total_dist"], it["dx"], it["dy"],
                    it["sx"], it["sy"], it["ex"], it["ey"],
                    no_round=True,  # natural output, ~0.757 baseline gate
                )
    trajectories = [t for t in results if t is not None and len(t) >= 2]
    return trajectories


# ---------------------------------------------------------------------------
# Critic model.
# ---------------------------------------------------------------------------
class PathCritic(nn.Module):
    def __init__(self, d_model=D_MODEL, n_layers=N_LAYERS, n_head=N_HEAD,
                 d_ff=D_FF, dropout=DROPOUT, max_len=MAX_LEN):
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, dxdy, pad_mask):
        # dxdy: (B, L, 2), pad_mask: (B, L) bool, True=valid
        B, L, _ = dxdy.shape
        pos = torch.arange(L, device=dxdy.device).unsqueeze(0).expand(B, L)
        h = self.input_proj(dxdy) + self.pos_embed(pos)
        h = self.encoder(h, src_key_padding_mask=~pad_mask)
        mask_f = pad_mask.unsqueeze(-1).float()
        pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        logit = self.head(pooled).squeeze(-1)
        return logit


def paths_to_dxdy(paths, max_len=MAX_LEN):
    """paths: list of already-resampled (x,y,t) sequences (lists of tuples).
    Returns dxdy array (N, max_len, 2) float32, pad_mask (N, max_len) bool,
    and lengths (N,) int (pre-truncation diff length)."""
    n = len(paths)
    dxdy = np.zeros((n, max_len, 2), dtype=np.float32)
    pad_mask = np.zeros((n, max_len), dtype=bool)
    lengths = np.zeros(n, dtype=np.int64)
    n_truncated = 0
    for i, p in enumerate(paths):
        arr = np.asarray(p, dtype=np.float64)[:, :2]
        d = np.diff(arr, axis=0)
        L = len(d)
        lengths[i] = L
        if L > max_len:
            n_truncated += 1
            d = d[:max_len]
            L = max_len
        dxdy[i, :L] = d
        pad_mask[i, :L] = True
    return dxdy, pad_mask, lengths, n_truncated


def standardize_dxdy(dxdy, pad_mask, scale=None):
    if scale is None:
        valid = dxdy[pad_mask]
        scale = np.std(valid) if len(valid) else 1.0
        scale = max(scale, 1e-6)
    return dxdy / scale, scale


# ---------------------------------------------------------------------------
# Training / CV.
# ---------------------------------------------------------------------------
def train_one_fold(dxdy_tr, mask_tr, y_tr, dxdy_va, mask_va, y_va, device,
                    max_epochs=MAX_EPOCHS, patience=6, seed=0, verbose_prefix=""):
    torch.manual_seed(seed)
    model = PathCritic().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    Xtr = torch.from_numpy(dxdy_tr).to(device)
    Mtr = torch.from_numpy(mask_tr).to(device)
    Ytr = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    Xva = torch.from_numpy(dxdy_va).to(device)
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


def run_cv(dxdy, pad_mask, y, device, n_folds=N_FOLDS, seed=0, verbose=True):
    n = len(y)
    oof_logits = np.zeros(n, dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_aucs = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n), y)):
        t0 = time.perf_counter()
        _, fold_auc, logits_va = train_one_fold(
            dxdy[tr_idx], pad_mask[tr_idx], y[tr_idx],
            dxdy[va_idx], pad_mask[va_idx], y[va_idx],
            device, seed=seed + fold,
            verbose_prefix=f"[fold {fold}]" if verbose else "")
        oof_logits[va_idx] = logits_va
        fold_aucs.append(fold_auc)
        print(f"[phase0] fold {fold}: val_auc={fold_auc:.4f} "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)
    oof_probs = 1.0 / (1.0 + np.exp(-oof_logits))
    oof_auc = roc_auc_score(y, oof_probs)
    return oof_probs, oof_auc, fold_aucs


def train_full_model(dxdy, pad_mask, y, device, epochs=15, seed=0):
    """Train on ALL data (no held-out fold) for the saved checkpoint; fixed
    epoch count derived from CV (not early-stopped on data used for
    reporting) since there is no val split left."""
    torch.manual_seed(seed)
    model = PathCritic().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    X = torch.from_numpy(dxdy).to(device)
    M = torch.from_numpy(pad_mask).to(device)
    Y = torch.from_numpy(y.astype(np.float32)).to(device)
    n = len(X)
    batch_size = 128
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for b0 in range(0, n, batch_size):
            idx = perm[b0:b0 + batch_size]
            opt.zero_grad()
            logits = model(X[idx], M[idx])
            loss = bce(logits, Y[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        total_loss /= n
        print(f"[phase0] full-fit epoch={epoch} loss={total_loss:.4f}", flush=True)
    return model


# ---------------------------------------------------------------------------
# Main pipeline (shared between smoke and full run).
# ---------------------------------------------------------------------------
def run_pipeline(n_human, n_synth, device, smoke=False, out_prefix="phase0_critic"):
    t_start = time.perf_counter()

    print(f"[phase0] reconstructing {VAL_N} human validation paths "
          "(always the full recipe draw; see reconstruct_human_val_paths "
          "docstring on why n_val cannot be shrunk for smoke)...", flush=True)
    human_paths, human_feats = reconstruct_human_val_paths(VAL_N, VAL_SEED)

    # Order-invariant correctness anchor against the full cached 2000 rows.
    verify_against_cache(human_feats, n_check=len(human_feats))

    if n_human < len(human_paths):
        human_paths = human_paths[:n_human]
        human_feats = human_feats[:n_human]
        print(f"[phase0] smoke mode: sliced down to {n_human} human paths "
              "after verification", flush=True)

    print(f"[phase0] generating {n_synth} synthetic paths (no-round, "
          "0.752 config)...", flush=True)
    t0 = time.perf_counter()
    synth_paths_raw = generate_synth_paths(n_synth, seed=42, device=device)
    print(f"[phase0] generated {len(synth_paths_raw)}/{n_synth} synth paths "
          f"in {time.perf_counter()-t0:.1f}s", flush=True)

    synth_resampled = [resample_trajectory(p) for p in synth_paths_raw]
    synth_feats_list = []
    synth_resampled_valid = []
    for p in synth_resampled:
        f = extract_features(p)
        if f is not None and np.all(np.isfinite(f)):
            synth_feats_list.append(f)
            synth_resampled_valid.append(p)
    synth_feats = np.asarray(synth_feats_list, dtype=np.float64)
    synth_resampled = synth_resampled_valid
    print(f"[phase0] valid synth feature rows: {len(synth_feats)}/{len(synth_paths_raw)}",
          flush=True)

    n_use = min(len(human_feats), len(synth_feats))
    human_feats_bal = human_feats[:n_use]
    synth_feats_bal = synth_feats[:n_use]
    human_paths_bal = human_paths[:n_use]
    synth_paths_bal = synth_resampled[:n_use]
    print(f"[phase0] N used per class: {n_use}", flush=True)

    # --- RF fit + sanity check ---
    X18 = np.vstack([human_feats_bal, synth_feats_bal])
    y = np.concatenate([np.zeros(n_use), np.ones(n_use)])
    clf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1,
                                  random_state=42)
    clf.fit(X18, y)
    rf_oob_scores = clf.oob_decision_function_[:, 1]
    rf_sanity_auc = roc_auc_score(y, rf_oob_scores)
    print(f"[phase0] RF-OOB sanity AUC (2000v2000 no-round): {rf_sanity_auc:.4f}",
          flush=True)
    if not smoke:
        if not (RF_SANITY_LOW <= rf_sanity_auc <= RF_SANITY_HIGH):
            raise RuntimeError(
                f"RF sanity AUC {rf_sanity_auc:.4f} outside expected "
                f"[{RF_SANITY_LOW},{RF_SANITY_HIGH}] band -- pipeline broken, STOP")

    # --- critic dxdy tensors ---
    all_paths = human_paths_bal + synth_paths_bal
    dxdy, pad_mask, lengths, n_truncated = paths_to_dxdy(all_paths, max_len=MAX_LEN)
    print(f"[phase0] dxdy shape={dxdy.shape} n_truncated={n_truncated} "
          f"({n_truncated/len(all_paths):.2%})", flush=True)
    dxdy, scale = standardize_dxdy(dxdy, pad_mask)
    print(f"[phase0] dxdy standardization scale={scale:.4f}", flush=True)

    # --- critic 5-fold CV ---
    print(f"[phase0] running {N_FOLDS}-fold critic CV on {len(y)} paths...", flush=True)
    oof_probs, oof_auc, fold_aucs = run_cv(
        dxdy, pad_mask, y, device,
        n_folds=(2 if smoke else N_FOLDS), seed=0)
    print(f"[phase0] critic OOF AUC: {oof_auc:.4f} (fold aucs: "
          f"{[f'{a:.4f}' for a in fold_aucs]})", flush=True)

    # --- shared-space Spearman correlations ---
    is_synth = (y == 1)
    is_human = (y == 0)
    rho_synth, p_synth = spearmanr(oof_probs[is_synth], rf_oob_scores[is_synth])
    rho_human, p_human = spearmanr(oof_probs[is_human], rf_oob_scores[is_human])
    rho_pooled, p_pooled = spearmanr(oof_probs, rf_oob_scores)
    print(f"[phase0] Spearman(critic_oof, rf_oob) SYNTH-only: rho={rho_synth:.4f} "
          f"p={p_synth:.3e}", flush=True)
    print(f"[phase0] Spearman(critic_oof, rf_oob) HUMAN-only: rho={rho_human:.4f} "
          f"p={p_human:.3e}", flush=True)
    print(f"[phase0] Spearman(critic_oof, rf_oob) POOLED:     rho={rho_pooled:.4f} "
          f"p={p_pooled:.3e}", flush=True)

    # --- feature-alignment table ---
    importances = clf.feature_importances_
    feat_spearman = []
    for i, name in enumerate(FEATURE_NAMES):
        rho, _ = spearmanr(oof_probs, X18[:, i])
        feat_spearman.append(abs(rho))
    order = np.argsort(importances)[::-1]
    table_rows = []
    print("\n=== FEATURE ALIGNMENT TABLE (sorted by RF importance) ===")
    print(f"{'feature':24s} {'rf_importance':>14s} {'|spearman(critic,feat)|':>24s}")
    for idx in order:
        name = FEATURE_NAMES[idx]
        imp = float(importances[idx])
        rho_abs = float(feat_spearman[idx])
        table_rows.append({"feature": name, "rf_importance": imp,
                            "abs_spearman_critic": rho_abs})
        print(f"{name:24s} {imp:14.4f} {rho_abs:24.4f}")

    # --- train full-data critic + save artifacts ---
    print("[phase0] training full-data critic for checkpoint save...", flush=True)
    full_epochs = 3 if smoke else 15
    full_model = train_full_model(dxdy, pad_mask, y, device, epochs=full_epochs)

    ckpt_path = REPO_ROOT / "research" / f"{out_prefix}.pt"
    torch.save({
        "model_state_dict": full_model.state_dict(),
        "dxdy_scale": scale,
        "max_len": MAX_LEN,
        "d_model": D_MODEL, "n_layers": N_LAYERS, "n_head": N_HEAD,
        "d_ff": D_FF, "dropout": DROPOUT,
    }, ckpt_path)
    print(f"[phase0] saved critic checkpoint to {ckpt_path}", flush=True)

    npz_path = REPO_ROOT / "research" / f"{out_prefix}_data.npz"
    np.savez_compressed(
        npz_path,
        dxdy=dxdy.astype(np.float32),
        pad_mask=pad_mask,
        y=y,
        feats18=X18,
        rf_oob_scores=rf_oob_scores,
        oof_critic_probs=oof_probs,
        feature_names=np.array(FEATURE_NAMES),
    )
    print(f"[phase0] saved assembled arrays to {npz_path}", flush=True)

    results = {
        "n_human": int(n_use),
        "n_synth": int(n_use),
        "n_truncated_paths": int(n_truncated),
        "dxdy_scale": float(scale),
        "rf_sanity_auc": float(rf_sanity_auc),
        "critic_oof_auc": float(oof_auc),
        "critic_fold_aucs": [float(a) for a in fold_aucs],
        "spearman_synth_only": {"rho": float(rho_synth), "p": float(p_synth)},
        "spearman_human_only": {"rho": float(rho_human), "p": float(p_human)},
        "spearman_pooled": {"rho": float(rho_pooled), "p": float(p_pooled)},
        "feature_alignment_table": table_rows,
        "elapsed_sec": time.perf_counter() - t_start,
        "smoke": smoke,
    }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                     help="Tiny CPU smoke run: ~50 human, ~16 synth, 2-fold CV.")
    ap.add_argument("--device", type=str, default=None,
                     help="cuda/cpu; default auto-detect via get_device()")
    ap.add_argument("--n-human", type=int, default=None)
    ap.add_argument("--n-synth", type=int, default=None)
    ap.add_argument("--out-prefix", type=str, default="phase0_critic")
    args = ap.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"[phase0] device={device}", flush=True)

    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.80)
        print("[phase0] WDDM spill guard: set_per_process_memory_fraction(0.80)",
              flush=True)

    if args.smoke:
        n_human = args.n_human or 50
        n_synth = args.n_synth or 16
        results = run_pipeline(n_human, n_synth, device, smoke=True,
                                out_prefix="phase0_critic_smoke")
    else:
        n_human = args.n_human or VAL_N
        n_synth = args.n_synth or 2000
        results = run_pipeline(n_human, n_synth, device, smoke=False,
                                out_prefix=args.out_prefix)

    out_json = REPO_ROOT / "research" / (
        "phase0_critic_smoke_results.json" if args.smoke
        else "phase0_critic_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[phase0] wrote results to {out_json}", flush=True)

    print("\n=== PHASE 0 SUMMARY ===")
    print(f"smoke={results['smoke']}  N per class={results['n_human']}")
    print(f"RF-OOB sanity AUC: {results['rf_sanity_auc']:.4f}")
    print(f"Critic OOF AUC: {results['critic_oof_auc']:.4f}")
    print(f"Spearman SYNTH-only (critic vs RF): "
          f"{results['spearman_synth_only']['rho']:.4f}")
    print(f"Spearman HUMAN-only (critic vs RF): "
          f"{results['spearman_human_only']['rho']:.4f}")
    print(f"Spearman POOLED (critic vs RF): "
          f"{results['spearman_pooled']['rho']:.4f}")


if __name__ == "__main__":
    main()
