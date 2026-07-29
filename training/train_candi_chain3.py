"""Burst 3 of DIFFUSION_PILOT_V2.md Phase 1: widen the retargeted loss from
curvature-only to the full smooth (differentiable) detector feature set.

Burst 2 (training/train_candi_chain2.py, training/candi_polar_flow_chain2.pt)
moved detector-space curvature variety 0.57 -> 0.75 at N=2000 (first-ever
gradient movement of a whole-path stat here), but natural AUC went UP to
0.795 (baseline 0.757): the RF detector shifted its reliance to
path_efficiency / mean_jerk / mean_acceleration once curvature stopped being
the cheapest tell -- curvature-only matching trades one unrealism for
another (whack-a-mole). Burst 3 widens the loss to every feature that has a
smooth differentiable form, computed on the SAME rotate+perp-corrected
cartesian decode chain2 already builds, so there is nowhere cheap to hide:

  mean_velocity, std_velocity, mean_acceleration, std_acceleration,
  mean_jerk, std_jerk, path_efficiency, curvature_mean, curvature_std,
  angular_velocity_mean, angular_velocity_std.

All formulas match features.py's extract_features exactly (same dt=1/125
grid, same clamps), computed on the corrected cartesian path instead of
features.py's resampled real-mouse points. NOT covered (no smooth
differentiable form -- hard-count / argmax-based, noted as uncovered, never
attempted here): max_velocity, velocity_skewness, max_acceleration,
max_deviation, num_direction_changes, movement_duration,
time_to_peak_velocity.

Detector-space transform per feature follows research/cond_realization_probe
.py's to_detector_space column spec (the only full-feature-vector smooth
transform definition in the repo), reused here as the recipe for what
"detector space" means per feature, both for the training loss and for the
post-hoc scoring table:
  mean_velocity, std_velocity, std_acceleration, std_jerk,
  angular_velocity_mean, angular_velocity_std: log1p(clip(x, 0, None))
  mean_acceleration: x / 1e4        mean_jerk: x / 1e6
  path_efficiency: identity
  curvature_mean, curvature_std: log1p(clip(x, 0, None) * 1e3)

For each covered feature the loss compares the ACROSS-BATCH mean and std of
the per-path detector-space value to a FIXED scalar human target (mean and
std), precomputed offline by training/compute_human_curv_targets.py from a
5000-sample training-pool draw at native lengths
(training/human_feature_targets.json) -- never data/human_val_features_grpo
.npy (post-hoc scoring only) or data/human_eval_features.npy (never touched
anywhere in this file). Every one of the 22 sub-terms (11 features x
{mean-of-stat, std-of-stat}) is z-scored by that feature's human target std,
so a feature with tight human dispersion does not get swamped by ones with
wide raw units:

    loss = sum_f [ ((batch_mean_f - target_mean_f) / target_std_f) ** 2
                 + ((batch_std_f  - target_std_f)  / target_std_f) ** 2 ]

Everything else (rotate+perp differentiable decode, seq-len-bucketed cond
sampling, rolling-buffer honest logging, grad-norm lambda calibration, WDDM
VRAM guard, backward-flow-before-curv-graph ordering, per-step gradient
checkpointing, hard 90-minute wall clock, lr 1e-5) is an unchanged carry-
forward from burst 2 / burst 1 (training/train_candi_chain.py,
training/train_candi_chain2.py).

NEW vs burst 2: named checkpoints training/chain3_step{N}.pt saved every
--named-ckpt-every optimizer steps (default 50), NEVER overwritten, so a
mid-burst variety peak (burst 2's peak eased back before the final save and
was lost) can always be recovered and scored later.

FRESH START: copies training/candi_polar_flow_best.pt to
training/candi_polar_flow_chain3.pt (NOT chain2.pt -- that is a separate,
already-scored lineage). Never writes to best.pt.

CPU smoke test: `python -m training.train_candi_chain3 --smoke`
(fabricated tensors, no file I/O, no GPU; verifies finite loss / nonzero
grad for the SUM, and additionally verifies each of the 11 feature terms
individually contributes a nonzero finite gradient in isolation -- the
jerk term is a third derivative of position and the most fragile).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from experiments._common import DurationModel
from models.candi import CANDIModel
from training.train_candi import CANDIDataset
from training.train_candi_chain import (
    md5_file, differentiable_generate, sample_flow_anchor_batch,
    total_grad_norm, read_last_temp, save_checkpoint,
)
from training.train_candi_chain2 import sample_cond_batch, ensure_chain_copy

TRAIN_DIR = Path("training")
DATA_DIR = Path("data")
SRC_CKPT_NAME = "candi_polar_flow_best.pt"
CHAIN_CKPT_NAME = "candi_polar_flow_chain3.pt"
EXPECTED_SRC_MD5 = "91326a29750789f3167055324ef377c5"
WATCHDOG_LOG = Path("research/gpu_temp_phase1.log")
TARGETS_PATH = TRAIN_DIR / "human_feature_targets.json"

VRAM_FRACTION = 0.80
N_SAMPLE_STEPS = 200
GUIDE = 0.15
PERP_SCALE = 0.85
HZ = 125.0
ROLL_BUFFER_MAXLEN = 512
ROLL_BUFFER_MIN = 256
NAMED_CKPT_EVERY = 50
_EPS = 1e-6

# --------------------------------------------------------------------------
# Covered features + detector-space transforms (torch, differentiable)
# research/cond_realization_probe.py's to_detector_space column spec.
# --------------------------------------------------------------------------

COVERED_FEATURES = [
    "mean_velocity",
    "std_velocity",
    "mean_acceleration",
    "std_acceleration",
    "mean_jerk",
    "std_jerk",
    "path_efficiency",
    "curvature_mean",
    "curvature_std",
    "angular_velocity_mean",
    "angular_velocity_std",
]

UNCOVERED_FEATURES = [
    "max_velocity", "velocity_skewness", "max_acceleration", "max_deviation",
    "num_direction_changes", "movement_duration", "time_to_peak_velocity",
]


def _det_lg(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.clamp(x, min=0.0))


DET_TRANSFORMS = {
    "mean_velocity": _det_lg,
    "std_velocity": _det_lg,
    "mean_acceleration": lambda x: x / 1e4,
    "std_acceleration": _det_lg,
    "mean_jerk": lambda x: x / 1e6,
    "std_jerk": _det_lg,
    "path_efficiency": lambda x: x,
    "curvature_mean": lambda x: torch.log1p(torch.clamp(x, min=0.0) * 1e3),
    "curvature_std": lambda x: torch.log1p(torch.clamp(x, min=0.0) * 1e3),
    "angular_velocity_mean": _det_lg,
    "angular_velocity_std": _det_lg,
}


def load_targets(path: Path) -> dict:
    targets = json.loads(path.read_text())
    for name in COVERED_FEATURES:
        for stat in ("mean", "std"):
            key = f"target_{stat}_{name}_det"
            assert key in targets, f"missing {key} in {path}"
    return targets


# --------------------------------------------------------------------------
# Differentiable decode: rotate+perp cartesian path (unchanged from chain2)
# then EVERY covered feature computed on that path with features.py's exact
# grid conventions.
# --------------------------------------------------------------------------

def _safe_hypot(x: torch.Tensor, y: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """torch.hypot(0, 0)'s gradient is 0/0 = NaN at that point; padded
    positions hit this every step. sqrt(x^2+y^2+eps^2) is smooth everywhere
    and numerically identical to hypot away from the origin."""
    return torch.sqrt(x * x + y * y + eps * eps)


def _safe_atan2(y: torch.Tensor, x: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """torch.atan2(0, 0)'s gradient is also 0/0 = NaN (same trap as hypot).
    Padded steps have dxp=dyp=0 exactly (frozen cumsum after real length),
    so a real padded batch hits this every step when computing per-segment
    heading angles for angular velocity. Nudging x by a tiny eps strictly
    away from zero keeps atan2 (and its gradient) finite everywhere; the
    bias is ~1e-6 vs pixel-scale displacements (O(1) or larger), negligible
    on every real (non-padded) segment."""
    return torch.atan2(y, x + eps)


def _masked_mean_std(values: torch.Tensor, valid: torch.Tensor, eps: float = _EPS):
    """values, valid: (B, L). Returns per-path (B,) mean and std over the
    valid entries, matching features.py's np.mean/np.std(ddof=0) exactly."""
    v = valid.float()
    n = v.sum(dim=1).clamp(min=1.0)
    vals = values * v
    mean = vals.sum(dim=1) / n
    sqsum = (values ** 2 * v).sum(dim=1)
    var = sqsum / n - mean ** 2
    std = torch.sqrt(var.clamp(min=eps))
    return mean, std


def decode_and_features(dp_final, pad_mask, tgt_angle, total_dist, spd_s, dh_s,
                         perp_scale=PERP_SCALE, hz=HZ, eps=_EPS):
    """dp -> speed/heading -> cumsum cartesian decode -> rotate-to-target +
    perp-scale correction (research/phase_a_baseline.py's build_trajectory,
    CORRECT="rotate", PERP_SCALE=0.85 -- identical to chain2's
    decode_and_curvature) -> every COVERED_FEATURES stat on the corrected
    cartesian path, using features.py's exact grid conventions (dt=1/125
    throughout, no resampling needed since the decode is already on that
    grid).

    dp_final: (B, T, 2) polar x0_hat, gradients intact.
    pad_mask: (B, T) bool, True = real / False = padded.
    tgt_angle, total_dist: (B,) -- the cond's target heading angle and the
      physical point-to-point distance (data/human_distances.npy units).

    Returns (raw_stats: dict[name -> (B,) tensor], path_valid: (B,) bool).
    """
    B, T = pad_mask.shape
    mask = pad_mask.float()
    speed = torch.clamp(dp_final[..., 0] / spd_s, min=0.0) * mask
    dh = dp_final[..., 1] / dh_s * mask
    heading = torch.cumsum(dh, dim=1)
    vx = speed * torch.cos(heading)
    vy = speed * torch.sin(heading)
    cum_x = torch.cumsum(vx, dim=1)
    cum_y = torch.cumsum(vy, dim=1)

    # rotate-to-target (target direction is always unit-normalized: tgt_mag == 1)
    raw_mag = _safe_hypot(cum_x[:, -1], cum_y[:, -1], eps)
    raw_ang = torch.atan2(cum_y[:, -1], cum_x[:, -1])
    rot = tgt_angle - raw_ang
    scale = 1.0 / raw_mag.clamp(min=eps)
    cos_r, sin_r = torch.cos(rot), torch.sin(rot)
    rx = (cum_x * cos_r.unsqueeze(1) - cum_y * sin_r.unsqueeze(1)) * scale.unsqueeze(1)
    ry = (cum_x * sin_r.unsqueeze(1) + cum_y * cos_r.unsqueeze(1)) * scale.unsqueeze(1)
    degenerate = (raw_mag <= eps).unsqueeze(1)
    rx = torch.where(degenerate, cum_x, rx)
    ry = torch.where(degenerate, cum_y, ry)

    # perp-scale
    dx_n = torch.cos(tgt_angle).unsqueeze(1)
    dy_n = torch.sin(tgt_angle).unsqueeze(1)
    par = rx * dx_n + ry * dy_n
    perp_x = (rx - par * dx_n) * perp_scale
    perp_y = (ry - par * dy_n) * perp_scale
    fx = par * dx_n + perp_x
    fy = par * dy_n + perp_y

    pos_x = fx * total_dist.unsqueeze(1)
    pos_y = fy * total_dist.unsqueeze(1)

    zeros = torch.zeros(B, 1, device=pos_x.device, dtype=pos_x.dtype)
    x_full = torch.cat([zeros, pos_x], dim=1)
    y_full = torch.cat([zeros, pos_y], dim=1)

    dt = 1.0 / hz

    # --- grid 0 (length T): per-segment displacement / velocity / speed ---
    dxp = x_full[:, 1:] - x_full[:, :-1]
    dyp = y_full[:, 1:] - y_full[:, :-1]
    vx2 = dxp / dt
    vy2 = dyp / dt
    ds = _safe_hypot(dxp, dyp, eps)
    speed2 = ds / dt

    mask0 = torch.cat([torch.ones(B, 1, device=mask.device), mask], dim=1)
    mask1 = mask0[:, 1:] * mask0[:, :-1]           # validity of grid 0 (length T)

    # --- grid 1 (length T-1): scalar accel (features.py's acc = d(speed)/dt,
    # NOT the vector-acceleration magnitude), vector accel components (for
    # curvature only), per-segment heading angle diff (for angular velocity) ---
    step_valid = mask1[:, 1:] * mask1[:, :-1]      # validity of grid 1 (length T-1)

    acc = (speed2[:, 1:] - speed2[:, :-1]) / dt
    ax = (vx2[:, 1:] - vx2[:, :-1]) / dt
    ay = (vy2[:, 1:] - vy2[:, :-1]) / dt

    angles = _safe_atan2(vy2, vx2, eps)             # grid 0, length T
    angle_diff = torch.remainder(angles[:, 1:] - angles[:, :-1] + math.pi,
                                  2.0 * math.pi) - math.pi   # grid 1, length T-1
    omega = torch.clamp(angle_diff / dt, -1e6, 1e6)

    speed_mid = speed2[:, :-1].clamp(min=eps)
    cross = (vx2[:, :-1] * ay - vy2[:, :-1] * ax).abs()
    curvature = torch.clamp(cross / speed_mid.pow(3), 0.0, 1e6)

    # --- grid 2 (length T-2): jerk = d(acc)/dt ---
    jerk_valid = step_valid[:, 1:] * step_valid[:, :-1]
    jerk = (acc[:, 1:] - acc[:, :-1]) / dt

    n_valid_path = step_valid.sum(dim=1)
    path_valid = n_valid_path >= 3   # matches chain2's curvature-grid floor (~= features.py's len(pts)>=5)

    mean_velocity, std_velocity = _masked_mean_std(speed2, mask1.bool(), eps)
    mean_acceleration, std_acceleration = _masked_mean_std(acc, step_valid.bool(), eps)
    mean_jerk, std_jerk = _masked_mean_std(jerk, jerk_valid.bool(), eps)
    angular_velocity_mean, _ = _masked_mean_std(omega.abs(), step_valid.bool(), eps)
    _, angular_velocity_std = _masked_mean_std(omega, step_valid.bool(), eps)
    curvature_mean, curvature_std = _masked_mean_std(curvature, step_valid.bool(), eps)

    d_straight = _safe_hypot(x_full[:, -1] - x_full[:, 0], y_full[:, -1] - y_full[:, 0], eps)
    d_traveled = (ds * mask1).sum(dim=1)
    path_efficiency = d_straight / d_traveled.clamp(min=eps)

    raw_stats = {
        "mean_velocity": mean_velocity, "std_velocity": std_velocity,
        "mean_acceleration": mean_acceleration, "std_acceleration": std_acceleration,
        "mean_jerk": mean_jerk, "std_jerk": std_jerk,
        "path_efficiency": path_efficiency,
        "curvature_mean": curvature_mean, "curvature_std": curvature_std,
        "angular_velocity_mean": angular_velocity_mean, "angular_velocity_std": angular_velocity_std,
    }
    return raw_stats, path_valid


def to_detector_space(raw_stats: dict) -> dict:
    return {name: DET_TRANSFORMS[name](val) for name, val in raw_stats.items()}


def multi_feature_target_loss(raw_stats, path_valid, targets, eps=_EPS):
    """Retarget spec item 1 (burst 3): sum over all COVERED_FEATURES of the
    z-scored squared error between this batch's across-batch mean/std (in
    detector space) and the fixed human target mean/std, normalized by the
    feature's own human target std so no feature dominates by scale."""
    if path_valid.sum() < 2:
        z = raw_stats["curvature_mean"].sum() * 0.0
        return z, {name: {"batch_mean": float("nan"), "batch_std": float("nan"),
                           "variety_ratio": float("nan")} for name in COVERED_FEATURES}

    det_stats = to_detector_space(raw_stats)
    total = None
    stats = {}
    for name in COVERED_FEATURES:
        v = det_stats[name][path_valid]
        b_mean = v.mean()
        b_std = v.std(unbiased=False).clamp(min=eps)
        t_mean = targets[f"target_mean_{name}_det"]
        t_std = max(targets[f"target_std_{name}_det"], eps)
        term = ((b_mean - t_mean) / t_std) ** 2 + ((b_std - t_std) / t_std) ** 2
        total = term if total is None else total + term
        stats[name] = {
            "batch_mean": float(b_mean.detach()),
            "batch_std": float(b_std.detach()),
            "variety_ratio": float(b_std.detach()) / t_std,
        }
    return total, stats


# --------------------------------------------------------------------------
# Rolling buffer for honest logging (all covered features)
# --------------------------------------------------------------------------

class RollingVariety:
    """Rolling buffer (>= ROLL_BUFFER_MIN entries) of recent per-path
    detector-space stats for every covered feature, used ONLY for logging --
    gradients always use the current step's batch. Fixes the small-batch
    across-batch std reading too high from heavy-tail under-sampling
    (burst 2 finding: +0.13 at batch 64 vs the N=2000 gate)."""

    def __init__(self, targets, maxlen=ROLL_BUFFER_MAXLEN):
        self.bufs = {name: deque(maxlen=maxlen) for name in COVERED_FEATURES}
        self.targets = targets

    def push(self, raw_stats, path_valid):
        det_stats = to_detector_space(raw_stats)
        valid_np = path_valid.detach().cpu().numpy()
        for name in COVERED_FEATURES:
            vals = det_stats[name].detach().cpu().numpy()[valid_np]
            self.bufs[name].extend(vals.tolist())

    def ready(self):
        return len(self.bufs["curvature_mean"]) >= ROLL_BUFFER_MIN

    def read(self):
        out = {}
        for name in COVERED_FEATURES:
            buf = self.bufs[name]
            if not buf:
                out[name] = (float("nan"), 0)
                continue
            t_std = max(self.targets[f"target_std_{name}_det"], _EPS)
            out[name] = (float(np.std(buf)) / t_std, len(buf))
        return out


# --------------------------------------------------------------------------
# CPU smoke test (fabricated tensors, no file I/O, no GPU)
# --------------------------------------------------------------------------

def smoke_test():
    print("[smoke3] CPU smoke test: K=4, batch=4, 3 steps, fabricated cond/pad_mask", flush=True)
    device = torch.device("cpu")
    torch.manual_seed(0)
    config = dict(d_model=32, n_heads=2, n_layers=2, d_ff=64, max_seq_len=16,
                  cond_dim=4, n_diffusion_steps=100, cond_dropout=0.1, dropout=0.0)
    model = CANDIModel(**config).to(device)
    data_scale = np.array([13.95, 2.33], dtype=np.float32)
    spd_s, dh_s = float(data_scale[0]), float(data_scale[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    targets = json.loads(TARGETS_PATH.read_text()) if TARGETS_PATH.exists() else None
    if targets is None:
        # fabricated targets matching the shape of the real file, in case it
        # hasn't been generated yet in this environment
        targets = {}
        for name in COVERED_FEATURES:
            targets[f"target_mean_{name}_det"] = 1.0
            targets[f"target_std_{name}_det"] = 1.0
    roller = RollingVariety(targets, maxlen=64)

    batch = 4
    seq_len_chain = 10
    n_steps = 4
    k = 4

    dxdy_b = torch.randn(batch, seq_len_chain, 2, device=device)
    pad_b = torch.ones(batch, seq_len_chain, dtype=torch.bool, device=device)
    stall_b = torch.zeros(batch, seq_len_chain, device=device)
    cond_b = torch.randn(batch, 4, device=device)

    for step in range(3):
        model.train()
        B = dxdy_b.shape[0]
        t_cont = torch.rand(B, device=device)
        t_int = (t_cont * (config["n_diffusion_steps"] - 1)).long()
        dxdy_noisy, noise, velocity = model.q_flow(dxdy_b, t_cont)
        stall_masked, disc_mask = model.q_discrete(stall_b, t_int)
        t_for_model = t_cont * (config["n_diffusion_steps"] - 1)
        dxdy_pred, stall_logit = model(dxdy_noisy, stall_masked, disc_mask.float(), t_for_model, cond_b, pad_b)
        pad_f = pad_b.float().unsqueeze(-1)
        cont_loss = ((dxdy_pred - velocity) ** 2 * pad_f).sum() / pad_f.sum().clamp(1)
        bce = nn.BCEWithLogitsLoss(reduction="none")
        disc_loss_raw = bce(stall_logit, stall_b)
        disc_weight = disc_mask.float() * pad_b.float()
        disc_loss = (disc_loss_raw * disc_weight).sum() / disc_weight.sum().clamp(1)
        flow_loss = cont_loss + 1.0 * disc_loss

        model.eval()
        cond = torch.randn(batch, 4, device=device)
        tgt_angle = torch.atan2(cond[:, 3], cond[:, 2])
        total_dist = torch.rand(batch, device=device) * 200.0 + 20.0
        lens = [10, 10, 7, 5]
        pad_mask = torch.zeros(batch, seq_len_chain, dtype=torch.bool, device=device)
        for i, L in enumerate(lens):
            pad_mask[i, :L] = True

        dp_final = differentiable_generate(
            model, cond, tgt_angle, seq_len_chain, spd_s, dh_s,
            k=k, n_steps=n_steps, guide=GUIDE, use_ckpt=False, device=device,
            pad_mask=pad_mask,
        )
        raw_stats, path_valid = decode_and_features(
            dp_final, pad_mask, tgt_angle, total_dist, spd_s, dh_s)
        curv_loss, curv_stats = multi_feature_target_loss(raw_stats, path_valid, targets)
        roller.push(raw_stats, path_valid)

        total = flow_loss + 0.5 * curv_loss
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        gn = total_grad_norm(model)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        assert math.isfinite(float(flow_loss.item())), "flow loss not finite"
        assert math.isfinite(float(curv_loss.item())), "curv loss not finite"
        assert gn > 0.0, "grad norm is zero"
        print(f"[smoke3] step={step} flow_loss={flow_loss.item():.4f} "
              f"curv_loss={curv_loss.item():.4f} grad_norm={gn:.4e} "
              f"path_valid={int(path_valid.sum())}/{batch}", flush=True)
        for name in COVERED_FEATURES:
            s = curv_stats[name]
            print(f"    {name:24s} batch_mean={s['batch_mean']:.4f} "
                  f"batch_std={s['batch_std']:.4f} variety_ratio={s['variety_ratio']:.4f}", flush=True)

    vr = roller.read()
    print(f"[smoke3] rolling buffer reads:", flush=True)
    for name in COVERED_FEATURES:
        ratio, n = vr[name]
        print(f"    {name:24s} n={n} variety_ratio={ratio:.4f}", flush=True)

    # --- per-feature isolated gradient check: each of the 11 terms must
    # individually contribute a nonzero, finite gradient. Jerk (third
    # derivative of position) is the most fragile chain. ---
    print("[smoke3] per-feature isolated gradient check...", flush=True)
    model2 = CANDIModel(**config).to(device)
    model2.load_state_dict(model.state_dict())
    cond = torch.randn(batch, 4, device=device)
    tgt_angle = torch.atan2(cond[:, 3], cond[:, 2])
    total_dist = torch.rand(batch, device=device) * 200.0 + 20.0
    pad_mask = torch.zeros(batch, seq_len_chain, dtype=torch.bool, device=device)
    for i, L in enumerate(lens):
        pad_mask[i, :L] = True

    for name in COVERED_FEATURES:
        model2.zero_grad(set_to_none=True)
        model2.eval()
        dp_final2 = differentiable_generate(
            model2, cond, tgt_angle, seq_len_chain, spd_s, dh_s,
            k=k, n_steps=n_steps, guide=GUIDE, use_ckpt=False, device=device,
            pad_mask=pad_mask,
        )
        raw_stats2, path_valid2 = decode_and_features(
            dp_final2, pad_mask, tgt_angle, total_dist, spd_s, dh_s)
        det_stats2 = to_detector_space(raw_stats2)
        v = det_stats2[name][path_valid2]
        t_mean = targets[f"target_mean_{name}_det"]
        t_std = max(targets[f"target_std_{name}_det"], _EPS)
        term = ((v.mean() - t_mean) / t_std) ** 2 + ((v.std(unbiased=False).clamp(min=_EPS) - t_std) / t_std) ** 2
        term.backward()
        gn2 = total_grad_norm(model2)
        assert math.isfinite(float(term.item())), f"{name}: loss term not finite"
        assert math.isfinite(gn2), f"{name}: grad norm not finite"
        assert gn2 > 0.0, f"{name}: grad norm is zero -- no gradient reaches the model"
        print(f"    {name:24s} term={term.item():.6f} grad_norm={gn2:.6e} OK", flush=True)

    tmp_path = Path("training") / "_smoke_chain3_test.pt"
    save_checkpoint(tmp_path, model, optimizer, config, data_scale, data_scale, 3,
                     argparse.Namespace(smoke=True))
    reloaded = torch.load(tmp_path, map_location="cpu", weights_only=False)
    model3 = CANDIModel(**reloaded["config"])
    model3.load_state_dict(reloaded["model_state_dict"])
    tmp_path.unlink()
    print("[smoke3] checkpoint save/load roundtrip OK", flush=True)
    print("[smoke3] SMOKE TEST PASSED", flush=True)


# --------------------------------------------------------------------------
# Main training loop
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data-dir", default="training")
    ap.add_argument("--src-ckpt", default=SRC_CKPT_NAME)
    ap.add_argument("--chain-ckpt", default=CHAIN_CKPT_NAME)
    ap.add_argument("--load-from", default=None)
    ap.add_argument("--targets", default=str(TARGETS_PATH))
    ap.add_argument("--k", type=int, default=N_SAMPLE_STEPS)
    ap.add_argument("--n-steps", type=int, default=N_SAMPLE_STEPS)
    ap.add_argument("--sample-batch", type=int, default=64,
                     help="Batch size for the seq-len-bucketed feature-loss batch")
    ap.add_argument("--pool-mult", type=int, default=6,
                     help="Oversample multiplier for the seq-len bucketing window")
    ap.add_argument("--batch-size", type=int, default=128,
                     help="Batch size for the real-data flow-matching anchor minibatch")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--curv-weight", type=float, default=0.0,
                     help="Lambda for the feature-moment loss term; 0.0 = auto-calibrate "
                          "from the measured grad-norm ratio at step 0")
    ap.add_argument("--disc-weight", type=float, default=1.0)
    ap.add_argument("--guide", type=float, default=GUIDE)
    ap.add_argument("--perp-scale", type=float, default=PERP_SCALE)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--total-steps", type=int, default=1000)
    ap.add_argument("--reset-schedule", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=90.0)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--named-ckpt-every", type=int, default=NAMED_CKPT_EVERY,
                     help="Save a NEVER-overwritten training/chain3_step{N}.pt every N steps")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--warmup-batches", type=int, default=None,
                     help="No-grad batches to prefill the rolling buffer before "
                          "step 0 (default: enough to reach ROLL_BUFFER_MIN)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--watchdog-log", default=str(WATCHDOG_LOG))
    args = ap.parse_args()

    if args.smoke:
        smoke_test()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION, device=0)
        total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        print(f"[chain3] VRAM cap: {VRAM_FRACTION:.2f} x {total_mb:.0f}MB = "
              f"{VRAM_FRACTION * total_mb:.0f}MB (spill-to-shared-memory guard)", flush=True)

    data_dir = Path(args.data_dir)
    src_path = data_dir / args.src_ckpt
    chain_path = data_dir / args.chain_ckpt
    chain_latest_path = chain_path.with_stem(chain_path.stem + "_latest")

    md5_before = md5_file(src_path)
    print(f"[chain3] source MD5 before: {md5_before} (expected {EXPECTED_SRC_MD5})", flush=True)
    assert md5_before == EXPECTED_SRC_MD5, "source checkpoint MD5 does not match expected -- STOP"

    ensure_chain_copy(src_path, chain_path)

    load_path = Path(args.load_from) if args.load_from else chain_path
    ckpt = torch.load(load_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    data_scale = ckpt["data_scale"]
    data_std = ckpt.get("data_std", data_scale)
    assert ckpt.get("polar", False), "expected polar checkpoint"
    assert ckpt.get("pred_type", "x0") == "flow", "expected flow pred_type checkpoint"

    model = CANDIModel(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[chain3] loaded {load_path} device={device} params={n_params:,}", flush=True)

    targets = load_targets(Path(args.targets))
    print(f"[chain3] human feature targets ({len(COVERED_FEATURES)} covered "
          f"features, uncovered: {UNCOVERED_FEATURES}):", flush=True)
    for name in COVERED_FEATURES:
        print(f"    {name:24s} mean={targets[f'target_mean_{name}_det']:.4f} "
              f"std={targets[f'target_std_{name}_det']:.4f}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if "optimizer_state_dict" in ckpt and args.load_from:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("[chain3] resumed optimizer state (AdamW momentum) from checkpoint", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[chain3] WARNING: could not resume optimizer state: {exc}", flush=True)

    start_step = 0
    if args.load_from and not args.reset_schedule:
        start_step = int(ckpt.get("global_step", 0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_steps)
    for _ in range(start_step):
        scheduler.step()
    print(f"[chain3] schedule start_step={start_step} reset_schedule={args.reset_schedule} "
          f"total_steps={args.total_steps}", flush=True)

    # --- data pools ---
    print("[chain3] loading training data pools...", flush=True)
    dxdy = np.load(data_dir / "zimt_dxdy.npy", mmap_mode="r")
    stall = np.load(data_dir / "zimt_stall.npy", mmap_mode="r")
    lengths = np.load(data_dir / "zimt_lengths.npy")
    conditions = np.load(data_dir / "zimt_conditions.npy")
    spd_all = np.load(data_dir / "zimt_polar_spd.npy", mmap_mode="r")
    dh_all = np.load(data_dir / "zimt_polar_dh.npy", mmap_mode="r")
    N = len(lengths)

    n_val = min(N // 10, 30000)
    perm = np.random.default_rng(42).permutation(N)
    tr_idx = perm[n_val:]

    max_seq_len = config["max_seq_len"]
    flow_dataset = CANDIDataset(
        dxdy, stall, lengths, conditions, max_seq_len, data_scale, polar=True,
        spd=spd_all, dh=dh_all,
    )

    distances = np.load(DATA_DIR / "human_distances.npy")
    duration_model = DurationModel(str(data_dir), std_mult=0.7)
    print(f"[chain3] cond pool: {len(distances):,} human_distances.npy distances "
          f"+ DurationModel (never restricted to a length subset)", flush=True)

    rng = np.random.default_rng(args.seed)
    spd_s, dh_s = float(data_scale[0]), float(data_scale[1])

    # --- rolling-buffer warm-up ---
    roller = RollingVariety(targets)
    n_warmup = args.warmup_batches
    if n_warmup is None:
        n_warmup = math.ceil(ROLL_BUFFER_MIN / args.sample_batch)
    print(f"[chain3] warming up rolling buffer with {n_warmup} no-grad batches "
          f"(sample_batch={args.sample_batch})...", flush=True)
    model.eval()
    t_warm0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_warmup):
            cond_np, angle_np, total_dist_np, pad_mask_np, bucket_T = sample_cond_batch(
                distances, duration_model, args.sample_batch, args.pool_mult, max_seq_len, rng)
            cond = torch.from_numpy(cond_np).to(device)
            tgt_angle = torch.from_numpy(angle_np).to(device)
            total_dist = torch.from_numpy(total_dist_np).to(device)
            pad_mask = torch.from_numpy(pad_mask_np).to(device)
            dp_final = differentiable_generate(
                model, cond, tgt_angle, bucket_T, spd_s, dh_s,
                k=args.k, n_steps=args.n_steps, guide=args.guide, use_ckpt=False,
                device=device, pad_mask=pad_mask,
            )
            raw_stats, path_valid = decode_and_features(
                dp_final, pad_mask, tgt_angle, total_dist, spd_s, dh_s, args.perp_scale)
            roller.push(raw_stats, path_valid)
    vr0 = roller.read()
    print(f"[chain3] === STEP-0 PRE-TRAINING ROLLING VARIETY READ "
          f"(warmup {time.perf_counter()-t_warm0:.1f}s) ===", flush=True)
    for name in COVERED_FEATURES:
        ratio, n = vr0[name]
        flag = " <-- expected ~0.55 for curvature; if >=0.8 the retarget failed, STOP" \
            if name == "curvature_mean" else ""
        print(f"    {name:24s} n={n} variety_ratio={ratio:.4f}{flag}", flush=True)

    t_burst_start = time.perf_counter()
    max_seconds = args.max_minutes * 60.0
    global_step = start_step
    steps_done_this_run = 0
    bce = nn.BCEWithLogitsLoss(reduction="none")

    print(f"[chain3] === starting training loop: k={args.k} n_steps={args.n_steps} "
          f"sample_batch={args.sample_batch} pool_mult={args.pool_mult} "
          f"batch_size={args.batch_size} lr={args.lr} "
          f"curv_weight={'auto' if args.curv_weight == 0.0 else args.curv_weight} "
          f"max_minutes={args.max_minutes} ===", flush=True)

    lambda_chosen = args.curv_weight
    while True:
        elapsed_burst = time.perf_counter() - t_burst_start
        if elapsed_burst >= max_seconds:
            print(f"[chain3] hard 90-minute wall clock reached ({elapsed_burst/60:.1f} min), "
                  f"stopping cleanly", flush=True)
            break
        if args.max_steps is not None and steps_done_this_run >= args.max_steps:
            print(f"[chain3] reached --max-steps={args.max_steps}, stopping", flush=True)
            break

        t_step0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # --- (a) real-data flow-matching + stall anchor loss ---
        model.train()
        optimizer.zero_grad(set_to_none=True)
        dxdy_b, stall_b, pad_b, cond_b = sample_flow_anchor_batch(
            flow_dataset, tr_idx, args.batch_size, rng, device)
        B = dxdy_b.shape[0]
        t_cont = torch.rand(B, device=device)
        t_int = (t_cont * (config["n_diffusion_steps"] - 1)).long()
        dxdy_noisy, noise, velocity = model.q_flow(dxdy_b, t_cont)
        stall_masked, disc_mask = model.q_discrete(stall_b, t_int)
        t_for_model = t_cont * (config["n_diffusion_steps"] - 1)
        dxdy_pred, stall_logit = model(dxdy_noisy, stall_masked, disc_mask.float(), t_for_model, cond_b, pad_b)
        pad_f = pad_b.float().unsqueeze(-1)
        cont_loss = ((dxdy_pred - velocity) ** 2 * pad_f).sum() / pad_f.sum().clamp(1)
        disc_loss_raw = bce(stall_logit, stall_b)
        disc_w = disc_mask.float() * pad_b.float()
        disc_loss = (disc_loss_raw * disc_w).sum() / disc_w.sum().clamp(1)
        flow_loss = cont_loss + args.disc_weight * disc_loss
        flow_loss.backward()  # frees flow graph before the curv graph is built
        flow_grad_norm = total_grad_norm(model)
        if global_step == start_step:
            flow_grads = {name: (p.grad.detach().clone() if p.grad is not None else None)
                          for name, p in model.named_parameters()}
            optimizer.zero_grad(set_to_none=True)

        # --- (b) widened feature-moment loss on a seq-len-bucketed sampled batch ---
        model.eval()
        cond_np, angle_np, total_dist_np, pad_mask_np, bucket_T = sample_cond_batch(
            distances, duration_model, args.sample_batch, args.pool_mult, max_seq_len, rng)
        cond = torch.from_numpy(cond_np).to(device)
        tgt_angle = torch.from_numpy(angle_np).to(device)
        total_dist = torch.from_numpy(total_dist_np).to(device)
        pad_mask = torch.from_numpy(pad_mask_np).to(device)

        dp_final = differentiable_generate(
            model, cond, tgt_angle, bucket_T, spd_s, dh_s,
            k=args.k, n_steps=args.n_steps, guide=args.guide, use_ckpt=True,
            device=device, pad_mask=pad_mask,
        )
        raw_stats, path_valid = decode_and_features(
            dp_final, pad_mask, tgt_angle, total_dist, spd_s, dh_s, args.perp_scale)
        curv_loss, curv_stats = multi_feature_target_loss(raw_stats, path_valid, targets)
        if not torch.isfinite(curv_loss):
            print(f"[chain3] WARNING: non-finite curv_loss at step {global_step}, "
                  f"skipping this step's feature-moment term", flush=True)
            curv_loss = torch.zeros((), device=device)

        roller.push(raw_stats, path_valid)

        curv_grad_norm = None
        if global_step == start_step:
            curv_loss.backward()
            curv_grad_norm = total_grad_norm(model)
            if lambda_chosen == 0.0:
                lambda_chosen = flow_grad_norm / max(curv_grad_norm, 1e-12)
                print(f"[chain3] AUTO-CALIBRATED lambda={lambda_chosen:.6e} "
                      f"(flow_grad_norm={flow_grad_norm:.6e} / curv_grad_norm={curv_grad_norm:.6e})",
                      flush=True)
            else:
                print(f"[chain3] measured flow_grad_norm={flow_grad_norm:.6e} "
                      f"curv_grad_norm={curv_grad_norm:.6e} at raw scale; "
                      f"using CLI lambda={lambda_chosen:.6e} "
                      f"(would-be auto lambda = {flow_grad_norm / max(curv_grad_norm, 1e-12):.6e})",
                      flush=True)
            for name, p in model.named_parameters():
                fg = flow_grads[name]
                cg = p.grad
                if fg is None and cg is None:
                    continue
                combined = (fg if fg is not None else torch.zeros_like(p)) \
                    + lambda_chosen * (cg if cg is not None else torch.zeros_like(p))
                p.grad = combined
        else:
            (lambda_chosen * curv_loss).backward()

        grad_norm_step = total_grad_norm(model)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        global_step += 1
        steps_done_this_run += 1

        if device.type == "cuda":
            torch.cuda.synchronize()
        step_elapsed = time.perf_counter() - t_step0
        peak_vram_mb = (torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                        if device.type == "cuda" else 0.0)

        vr_roll = roller.read()
        temp = read_last_temp(Path(args.watchdog_log))
        lr_now = scheduler.get_last_lr()[0]
        gn_msg = ""
        if curv_grad_norm is not None:
            gn_msg = (f" | CALIBRATION flow_grad_norm={flow_grad_norm:.4e} "
                      f"curv_grad_norm={curv_grad_norm:.4e} lambda={lambda_chosen:.4e}")
        # condensed per-step line: curvature + path_efficiency + mean_jerk (the
        # burst-2 whack-a-mole trio) always shown; full table every 10 steps.
        cm_ratio, _ = vr_roll["curvature_mean"]
        cs_ratio, _ = vr_roll["curvature_std"]
        pe_ratio, _ = vr_roll["path_efficiency"]
        mj_ratio, n_buf = vr_roll["mean_jerk"]
        print(f"[chain3] step={global_step:5d} flow_loss={flow_loss.item():.4f} "
              f"curv_loss={curv_loss.item():.4f} bucket_T={bucket_T} "
              f"vr_roll(n={n_buf}): curv_mean={cm_ratio:.4f} curv_std={cs_ratio:.4f} "
              f"path_eff={pe_ratio:.4f} mean_jerk={mj_ratio:.4f} "
              f"grad_norm={grad_norm_step:.4e} lr={lr_now:.2e} "
              f"peak_vram={peak_vram_mb:.0f}MB step_s={step_elapsed:.2f} "
              f"burst_min={elapsed_burst/60:.2f} temp={temp}{gn_msg}", flush=True)
        if global_step % 10 == 0:
            print(f"[chain3] step={global_step:5d} FULL rolling variety table:", flush=True)
            for name in COVERED_FEATURES:
                ratio, n = vr_roll[name]
                print(f"    {name:24s} n={n} variety_ratio={ratio:.4f}", flush=True)

        if global_step % args.ckpt_every == 0:
            save_checkpoint(chain_latest_path, model, optimizer, config, data_scale,
                             data_std, global_step, args)
            print(f"[chain3] saved {chain_latest_path} at step {global_step}", flush=True)

        if global_step % args.named_ckpt_every == 0:
            named_path = data_dir / f"chain3_step{global_step}.pt"
            save_checkpoint(named_path, model, optimizer, config, data_scale,
                             data_std, global_step, args)
            print(f"[chain3] saved NAMED checkpoint {named_path} (never overwritten)", flush=True)

    save_checkpoint(chain_latest_path, model, optimizer, config, data_scale,
                     data_std, global_step, args)
    save_checkpoint(chain_path, model, optimizer, config, data_scale,
                     data_std, global_step, args)
    print(f"[chain3] final save: {chain_path} and {chain_latest_path} at step {global_step}", flush=True)

    md5_after = md5_file(src_path)
    print(f"[chain3] source MD5 after: {md5_after}", flush=True)
    if md5_after != md5_before:
        print("[chain3] *** WARNING: source checkpoint MD5 CHANGED -- should never happen ***",
              flush=True)
    else:
        print("[chain3] source checkpoint MD5 unchanged, confirmed untouched.", flush=True)

    print(f"[chain3] DONE. steps_done_this_run={steps_done_this_run} "
          f"global_step={global_step} wall_clock_min={(time.perf_counter()-t_burst_start)/60:.1f}",
          flush=True)


if __name__ == "__main__":
    main()
