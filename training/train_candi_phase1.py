"""Phase 1 coupling rig: fine-tune CANDI to FOOL the frozen Phase 0b
geometric-vocabulary critic (research/phase0b_critic.pt), anchored by the
standard flow-matching loss so the generator does not drift/forget.

This is a coupling SMOKE TEST, not the full adversarial build: it proves
(a) the whole chain -- full 200-step differentiable sampler -> rotate +
perp-0.85 decode -> torch finite-diff geometric channels -> frozen
median/IQR standardization + signed-log -> frozen critic -- is numerically
identical to the space the critic was actually trained in, and (b)
gradient flows end to end into every generator parameter tensor. It does
NOT decide whether fooling the critic actually helps the real RF detector
at N=2000; that is a separate scoring step this script never runs.

Generated paths are native 125Hz (research/phase_a_baseline.py:decode_polar
integrates at implicit dt=1/125), so features.py's resample_trajectory is
an IDENTITY on generated paths -- no differentiable resampling is needed
anywhere in this file.

Reused verbatim (read, matched formula-for-formula, not reimplemented from
scratch):
  - research/phase0b_critic.py: GeoPathCritic class, CHANNEL_NAMES, the
    per-step channel recipe (compute_channels_for_path/build_all_channels)
    and the median/IQR + signed-log standardization (robust_standardize).
    The torch mirrors here (compute_channels_for_path_torch,
    build_channels_batch_torch, robust_standardize_torch) reproduce that
    numpy code element-for-element so the frozen critic sees an identical
    input space -- verified by --equiv-check below.
  - training/train_candi_chain.py: differentiable_generate (the proven
    full-chain K=200 per-step-checkpointed sampler), sample_flow_anchor_batch,
    total_grad_norm, read_last_temp, save_checkpoint, md5_file.
  - training/train_candi_chain2.py: sample_cond_batch (seq-len-bucketed
    cond sampling from human_distances.npy + DurationModel) and the
    rotate-to-target + perp-0.85 decode math (decode_to_steps below is a
    verbatim copy of decode_and_curvature's prefix, stopping at the
    per-step cartesian deltas instead of collapsing to curvature -- chain2
    itself is left untouched).

Safety:
  - Never reads data/human_eval_features.npy anywhere in this file.
  - Copies training/candi_polar_flow_best.pt to a NEW checkpoint name
    (training/candi_polar_flow_phase1.pt) at startup, only if that copy
    does not already exist (resume path); never writes to best.pt. MD5 of
    best.pt is asserted before the run and re-checked after.
  - torch.cuda.set_per_process_memory_fraction(0.80) before any CUDA
    tensor allocation (WDDM spill guard).
  - Hard --max-minutes wall clock enforced in-process; checkpoints every
    --ckpt-every steps.

Usage:
    Build/refresh the frozen channel-standardization stats (idempotent,
    auto-runs on first use if the file is missing):
        .venv/Scripts/python.exe -m training.train_candi_phase1 --build-stats

    Numeric equivalence check (decisive correctness anchor -- run this
    FIRST; if it fails, STOP, the trainer is measuring a different space
    than the critic was trained on):
        .venv/Scripts/python.exe -m training.train_candi_phase1 --equiv-check

    CPU smoke test (tiny fabricated generator, REAL frozen critic + REAL
    frozen stats, one full fool+anchor step, checks grad reaches every
    generator param tensor):
        .venv/Scripts/python.exe -m training.train_candi_phase1 --smoke

    One (or --max-steps N) short GPU training step(s):
        .venv/Scripts/python.exe -m training.train_candi_phase1 \\
            --max-steps 1 --max-minutes 5 --sample-batch 64 --k 200
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from experiments._common import DurationModel
from models.candi import CANDIModel
from research.phase0b_critic import GeoPathCritic
from training.train_candi import CANDIDataset
from training.train_candi_chain import (
    md5_file, differentiable_generate, sample_flow_anchor_batch,
    total_grad_norm, read_last_temp, save_checkpoint,
)
from training.train_candi_chain2 import sample_cond_batch

TRAIN_DIR = Path("training")
DATA_DIR = Path("data")
RESEARCH_DIR = Path("research")

SRC_CKPT_NAME = "candi_polar_flow_best.pt"
PHASE1_CKPT_NAME = "candi_polar_flow_phase1.pt"
EXPECTED_SRC_MD5 = "91326a29750789f3167055324ef377c5"
WATCHDOG_LOG = RESEARCH_DIR / "gpu_temp_phase1.log"
CRITIC_CKPT_PATH = RESEARCH_DIR / "phase0b_critic.pt"
STATS_PATH = RESEARCH_DIR / "phase1_channel_stats.npz"
CRITIC_DATA_PATH = RESEARCH_DIR / "phase0_critic_data.npz"

VRAM_FRACTION = 0.80
N_SAMPLE_STEPS = 200
GUIDE = 0.15
PERP_SCALE = 0.85
HZ = 125.0
DT = 1.0 / HZ
_EPS = 1e-6

CHANNEL_NAMES = ["dx", "dy", "speed", "acc", "jerk", "curvature", "angular_velocity"]
N_CHANNELS = len(CHANNEL_NAMES)


def ensure_phase1_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        print(f"[phase1] {dst} already exists, NOT overwriting with source "
              f"(resume path). Delete it manually to start a fresh run.",
              flush=True)
        return
    import shutil
    shutil.copy2(src, dst)
    print(f"[phase1] copied {src} -> {dst}", flush=True)


# --------------------------------------------------------------------------
# Frozen channel-standardization stats (median, IQR per channel), computed
# ONCE from research/phase0_critic_data.npz using research/phase0b_critic.py's
# own build_all_channels + robust_standardize -- never recomputed from the
# generator's batches during training.
# --------------------------------------------------------------------------

def build_and_save_channel_stats(out_path: Path = STATS_PATH,
                                  data_path: Path = CRITIC_DATA_PATH) -> np.ndarray:
    from research.phase0b_critic import build_all_channels, robust_standardize
    print(f"[phase1] building frozen channel stats from {data_path}...", flush=True)
    d = np.load(data_path, allow_pickle=True)
    dxdy, pad_mask = d["dxdy"], d["pad_mask"]
    channels, n_nf, n_valid = build_all_channels(dxdy, pad_mask)
    _, scales = robust_standardize(channels, pad_mask)
    np.savez(out_path, channel_names=np.array(CHANNEL_NAMES), scales=scales)
    print(f"[phase1] saved frozen channel stats to {out_path}", flush=True)
    for name, (med, iqr) in zip(CHANNEL_NAMES, scales):
        print(f"    {name:16s} median={med:.6g} iqr={iqr:.6g}", flush=True)
    return scales


def load_channel_stats(path: Path = STATS_PATH) -> np.ndarray:
    if not path.exists():
        return build_and_save_channel_stats(path)
    d = np.load(path, allow_pickle=True)
    names = [str(s) for s in d["channel_names"]]
    assert names == CHANNEL_NAMES, "channel name order mismatch vs frozen stats file"
    return d["scales"]


# --------------------------------------------------------------------------
# Differentiable torch channel builder -- mirrors research/phase0b_critic.py
# compute_channels_for_path / build_all_channels / robust_standardize
# element-for-element (verified in run_equivalence_check below).
# --------------------------------------------------------------------------

def _left_pad_edge_torch(arr: torch.Tensor, target_len: int) -> torch.Tensor:
    n = arr.shape[0]
    if n == target_len:
        return arr
    if n == 0:
        return arr.new_zeros(target_len)
    n_pad = target_len - n
    pad = arr[0:1].repeat(n_pad)
    return torch.cat([pad, arr])


def compute_channels_for_path_torch(dx: torch.Tensor, dy: torch.Tensor,
                                     dt: float = DT, eps: float = _EPS) -> torch.Tensor:
    """dx, dy: 1-D tensors, the TRUE (already-sliced, unpadded) per-step
    deltas for one path, length L. Returns (L, N_CHANNELS), gradients
    intact. Verbatim torch mirror of
    research/phase0b_critic.py:compute_channels_for_path -- same formulas,
    same left-edge-pad repair of the shortened diff series."""
    L = dx.shape[0]
    vx = dx / dt
    vy = dy / dt
    speed = torch.sqrt(dx ** 2 + dy ** 2 + eps ** 2) / dt

    if L >= 2:
        acc = torch.diff(speed) / dt
    else:
        acc = speed.new_zeros(0)
    if acc.shape[0] >= 2:
        jerk = torch.diff(acc) / dt
    else:
        jerk = speed.new_zeros(0)

    if L >= 2:
        ax = torch.diff(vx) / dt
        ay = torch.diff(vy) / dt
        speed_mid = torch.clamp(speed[:-1], min=eps)
        cross = (vx[:-1] * ay - vy[:-1] * ax).abs()
        curvature = torch.clamp(cross / speed_mid.pow(3), 0.0, 1e6)
    else:
        curvature = speed.new_zeros(0)

    angles = torch.atan2(dy, dx)
    if L >= 2:
        angle_diff = torch.diff(angles)
        angle_diff = torch.remainder(angle_diff + math.pi, 2 * math.pi) - math.pi
        omega = torch.clamp(angle_diff / dt, -1e6, 1e6)
    else:
        omega = speed.new_zeros(0)

    acc_p = _left_pad_edge_torch(acc, L)
    jerk_p = _left_pad_edge_torch(jerk, L)
    curvature_p = _left_pad_edge_torch(curvature, L)
    omega_p = _left_pad_edge_torch(omega, L)

    stacked = torch.stack([dx, dy, speed, acc_p, jerk_p, curvature_p, omega_p], dim=1)
    stacked = torch.nan_to_num(stacked, nan=0.0, posinf=1e6, neginf=-1e6)
    return stacked


def build_channels_batch_torch(dxp: torch.Tensor, dyp: torch.Tensor,
                                pad_mask: torch.Tensor, dt: float = DT,
                                eps: float = _EPS) -> torch.Tensor:
    """dxp, dyp: (B, T) per-step deltas. pad_mask: (B, T) bool, True=real,
    contiguous from the left (standard right-padding). Loops over the
    batch to honor each path's own true length (data-dependent, dynamic
    control flow -- fine under PyTorch's eager autograd), matching
    research/phase0b_critic.py:build_all_channels's per-path python loop
    exactly, while keeping every per-path op differentiable.

    Computed internally in float64: the numpy reference
    (research/phase0b_critic.py) is float64 throughout, and the curvature
    channel cubes speed_mid, which amplifies float32 rounding by orders of
    magnitude on this heavy-tailed quantity (measured: max abs standardized
    diff 1.2e-4 on CPU / 1.2e-3 on GPU in float32, vs 2e-15 -- machine
    epsilon -- in float64; every other channel stays ~1e-6 regardless).
    Upcasting here removes that fp32-vs-fp64 gap at negligible cost (O(T)
    per path); gen_channels_from_sample casts back to float32 only at the
    very end, right before the frozen (float32) critic."""
    dxp = dxp.double()
    dyp = dyp.double()
    B, T = dxp.shape
    lengths = pad_mask.sum(dim=1)
    rows = []
    for i in range(B):
        L = int(lengths[i].item())
        if L == 0:
            rows.append(dxp.new_zeros(T, N_CHANNELS))
            continue
        ch = compute_channels_for_path_torch(dxp[i, :L], dyp[i, :L], dt, eps)
        if L < T:
            ch = torch.cat([ch, dxp.new_zeros(T - L, N_CHANNELS)], dim=0)
        rows.append(ch)
    return torch.stack(rows, dim=0)


def robust_standardize_torch(channels: torch.Tensor, pad_mask: torch.Tensor,
                              scales: np.ndarray) -> torch.Tensor:
    """Applies FROZEN (median, iqr) per-channel stats (never recomputed
    from the current batch) then the signed-log compression, mirroring
    research/phase0b_critic.py:robust_standardize's per-element formula:
    z = (x - median) / iqr; out = sign(z) * log1p(|z|)."""
    med = torch.as_tensor(scales[:, 0], dtype=channels.dtype, device=channels.device)
    iqr = torch.as_tensor(scales[:, 1], dtype=channels.dtype, device=channels.device)
    z = (channels - med) / iqr
    standardized = torch.sign(z) * torch.log1p(torch.abs(z))
    standardized = standardized * pad_mask.unsqueeze(-1).float()
    return standardized


# --------------------------------------------------------------------------
# Decode: full-chain differentiable sampler output -> rotate + perp-0.85
# corrected per-step cartesian (dx, dy). Verbatim copy of
# training/train_candi_chain2.py:decode_and_curvature's prefix (through the
# per-step diff), returning the steps themselves instead of collapsing to
# curvature -- chain2.py itself is left untouched, imported only for
# sample_cond_batch.
# --------------------------------------------------------------------------

def _safe_hypot(x: torch.Tensor, y: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """sqrt(x^2+y^2+eps^2): smooth everywhere, unlike torch.hypot whose
    gradient is 0/0 = NaN at the origin -- padded positions hit exactly
    (0, 0) every step."""
    return torch.sqrt(x * x + y * y + eps * eps)


def decode_to_steps(dp_final: torch.Tensor, pad_mask: torch.Tensor,
                     tgt_angle: torch.Tensor, total_dist: torch.Tensor,
                     spd_s: float, dh_s: float, perp_scale: float = PERP_SCALE,
                     eps: float = _EPS):
    """dp_final: (B, T, 2) polar x0_hat, gradients intact. Returns dxp, dyp:
    (B, T) per-step cartesian deltas after the rotate-to-target +
    perp-scale transform (research/phase_a_baseline.py:build_trajectory
    with CORRECT="rotate", PERP_SCALE=0.85, continuous / no integer
    rounding -- rounding is not differentiable and is a generation-time-only
    quantization step, not part of the model's learned mapping)."""
    B, T = pad_mask.shape
    mask = pad_mask.float()
    speed = torch.clamp(dp_final[..., 0] / spd_s, min=0.0) * mask
    dh = dp_final[..., 1] / dh_s * mask
    heading = torch.cumsum(dh, dim=1)
    vx = speed * torch.cos(heading)
    vy = speed * torch.sin(heading)
    cum_x = torch.cumsum(vx, dim=1)
    cum_y = torch.cumsum(vy, dim=1)

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
    dxp = x_full[:, 1:] - x_full[:, :-1]
    dyp = y_full[:, 1:] - y_full[:, :-1]
    return dxp, dyp


def gen_channels_from_sample(dp_final: torch.Tensor, pad_mask: torch.Tensor,
                              tgt_angle: torch.Tensor, total_dist: torch.Tensor,
                              spd_s: float, dh_s: float, scales: np.ndarray,
                              perp_scale: float = PERP_SCALE) -> torch.Tensor:
    """Full differentiable path: sampled polar x0_hat -> cartesian steps ->
    geometric channels -> frozen standardization. Output is ready to feed
    straight into the frozen critic."""
    dxp, dyp = decode_to_steps(dp_final, pad_mask, tgt_angle, total_dist,
                                spd_s, dh_s, perp_scale)
    channels = build_channels_batch_torch(dxp, dyp, pad_mask)
    standardized = robust_standardize_torch(channels, pad_mask, scales)
    return standardized.float()  # frozen critic is float32; upcast was internal-only


# --------------------------------------------------------------------------
# Frozen critic loader
# --------------------------------------------------------------------------

def load_frozen_critic(device, path: Path = CRITIC_CKPT_PATH) -> GeoPathCritic:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    critic = GeoPathCritic(
        n_channels=N_CHANNELS, d_model=ckpt["d_model"], n_layers=ckpt["n_layers"],
        n_head=ckpt["n_head"], d_ff=ckpt["d_ff"], dropout=ckpt["dropout"],
        max_len=ckpt["max_len"],
    ).to(device)
    critic.load_state_dict(ckpt["model_state_dict"])
    critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)
    print(f"[phase1] loaded FROZEN critic from {path} "
          f"(d_model={ckpt['d_model']} n_layers={ckpt['n_layers']} "
          f"n_head={ckpt['n_head']} d_ff={ckpt['d_ff']}), all params frozen", flush=True)
    return critic


# --------------------------------------------------------------------------
# Validation 1: numeric equivalence (decisive correctness anchor)
# --------------------------------------------------------------------------

def run_equivalence_check(device, n_sample: int = 64, seed: int = 0, tol: float = 2e-4):
    from research.phase0b_critic import build_all_channels

    print(f"[phase1][equiv] loading {CRITIC_DATA_PATH}...", flush=True)
    d = np.load(CRITIC_DATA_PATH, allow_pickle=True)
    dxdy, pad_mask, y = d["dxdy"], d["pad_mask"], d["y"]
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n_sample, n), replace=False)
    sub_dxdy, sub_mask = dxdy[idx], pad_mask[idx]
    print(f"[phase1][equiv] sampled {len(idx)} paths "
          f"({int((y[idx] == 0).sum())} human, {int((y[idx] == 1).sum())} synth)", flush=True)

    scales = load_channel_stats()

    # --- numpy reference path (research/phase0b_critic.py, verbatim) ---
    channels_np, n_nf, n_valid = build_all_channels(sub_dxdy, sub_mask)
    med, iqr = scales[:, 0], scales[:, 1]
    z_np = (channels_np - med) / iqr
    std_np = np.sign(z_np) * np.log1p(np.abs(z_np))
    std_np = std_np * sub_mask[:, :, None]

    # --- torch path (this file) ---
    dxdy_t = torch.from_numpy(sub_dxdy.astype(np.float32)).to(device)
    mask_t = torch.from_numpy(sub_mask).to(device)
    channels_t = build_channels_batch_torch(dxdy_t[..., 0], dxdy_t[..., 1], mask_t)
    std_t = robust_standardize_torch(channels_t, mask_t, scales).float()

    max_diff_channels = float(np.max(np.abs(std_t.detach().cpu().numpy() - std_np)))
    print(f"[phase1][equiv] max abs diff, standardized channels "
          f"(torch builder vs numpy builder, frozen stats): {max_diff_channels:.6e}", flush=True)

    critic = load_frozen_critic(device)
    with torch.no_grad():
        logits_t = critic(std_t.float(), mask_t)
        logits_np_in = torch.from_numpy(std_np.astype(np.float32)).to(device)
        logits_np = critic(logits_np_in, mask_t)
    diff_logits = (logits_t - logits_np).abs().cpu().numpy()
    max_diff_logits = float(np.max(diff_logits))
    print(f"[phase1][equiv] max abs diff, critic logits "
          f"(torch-channel-path vs numpy-channel-path): {max_diff_logits:.6e}", flush=True)
    print(f"[phase1][equiv] logits_t sample: {logits_t[:5].cpu().numpy()}", flush=True)
    print(f"[phase1][equiv] logits_np sample: {logits_np[:5].cpu().numpy()}", flush=True)

    ok = max_diff_channels <= tol and max_diff_logits <= tol
    if not ok:
        print(f"[phase1][equiv] *** FAILED: max diffs exceed tolerance {tol:.1e} -- "
              f"the trainer is measuring a different space than the critic was "
              f"trained on. STOP, do not proceed to training. ***", flush=True)
    else:
        print(f"[phase1][equiv] PASSED (tolerance {tol:.1e})", flush=True)
    return max_diff_channels, max_diff_logits, ok


# --------------------------------------------------------------------------
# Validation 2: CPU smoke -- tiny fabricated generator, REAL frozen critic +
# REAL frozen stats, one fool+anchor step, checks grad reaches every
# generator parameter tensor.
# --------------------------------------------------------------------------

def smoke_test():
    print("[phase1][smoke] CPU smoke test: tiny generator, REAL frozen critic, "
          "K=4, batch=4, fool+anchor loss", flush=True)
    device = torch.device("cpu")
    torch.manual_seed(0)
    config = dict(d_model=32, n_heads=2, n_layers=2, d_ff=64, max_seq_len=16,
                  cond_dim=4, n_diffusion_steps=100, cond_dropout=0.1, dropout=0.0)
    model = CANDIModel(**config).to(device)
    data_scale = np.array([13.95, 2.33], dtype=np.float32)
    spd_s, dh_s = float(data_scale[0]), float(data_scale[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    n_param_tensors = sum(1 for _ in model.parameters())
    print(f"[phase1][smoke] generator has {n_param_tensors} parameter tensors", flush=True)

    critic = load_frozen_critic(device)
    scales = load_channel_stats()
    anchor_weight = 0.5

    batch = 4
    seq_len_chain = 8
    n_steps = 4
    k = 4

    dxdy_b = torch.randn(batch, seq_len_chain, 2, device=device)
    pad_b = torch.ones(batch, seq_len_chain, dtype=torch.bool, device=device)
    stall_b = torch.zeros(batch, seq_len_chain, device=device)
    cond_b = torch.randn(batch, 4, device=device)

    n_with_grad = n_total = 0
    for step in range(2):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        B = dxdy_b.shape[0]
        t_cont = torch.rand(B, device=device)
        t_int = (t_cont * (config["n_diffusion_steps"] - 1)).long()
        dxdy_noisy, noise, velocity = model.q_flow(dxdy_b, t_cont)
        stall_masked, disc_mask = model.q_discrete(stall_b, t_int)
        t_for_model = t_cont * (config["n_diffusion_steps"] - 1)
        dxdy_pred, stall_logit = model(dxdy_noisy, stall_masked, disc_mask.float(), t_for_model, cond_b, pad_b)
        pad_f = pad_b.float().unsqueeze(-1)
        cont_loss = ((dxdy_pred - velocity) ** 2 * pad_f).sum() / pad_f.sum().clamp(1)
        disc_loss_raw = nn.BCEWithLogitsLoss(reduction="none")(stall_logit, stall_b)
        disc_w = disc_mask.float() * pad_b.float()
        disc_loss = (disc_loss_raw * disc_w).sum() / disc_w.sum().clamp(1)
        flow_loss = cont_loss + 1.0 * disc_loss
        (anchor_weight * flow_loss).backward()  # frees flow graph before the fool/critic graph

        model.eval()
        cond = torch.randn(batch, 4, device=device)
        tgt_angle = torch.atan2(cond[:, 3], cond[:, 2])
        total_dist = torch.rand(batch, device=device) * 200.0 + 20.0
        lens = [8, 8, 6, 5]
        pad_mask = torch.zeros(batch, seq_len_chain, dtype=torch.bool, device=device)
        for i, L in enumerate(lens):
            pad_mask[i, :L] = True

        dp_final = differentiable_generate(
            model, cond, tgt_angle, seq_len_chain, spd_s, dh_s,
            k=k, n_steps=n_steps, guide=GUIDE, use_ckpt=False, device=device,
            pad_mask=pad_mask,
        )
        gen_std = gen_channels_from_sample(dp_final, pad_mask, tgt_angle, total_dist,
                                            spd_s, dh_s, scales)
        logits = critic(gen_std, pad_mask)
        target = torch.zeros_like(logits)  # label 0 = human: push critic to call gen "human"
        fool_loss = nn.BCEWithLogitsLoss()(logits, target)
        assert torch.isfinite(fool_loss), "fool_loss not finite"
        fool_loss.backward()

        gn = total_grad_norm(model)
        n_with_grad = 0
        n_total = 0
        for p in model.parameters():
            n_total += 1
            if p.grad is not None and torch.isfinite(p.grad).all():
                n_with_grad += 1
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        assert math.isfinite(float(flow_loss.item())), "anchor (flow) loss not finite"
        assert gn > 0.0, "grad norm is zero"
        print(f"[phase1][smoke] step={step} fool_loss={fool_loss.item():.4f} "
              f"flow_loss={flow_loss.item():.4f} logit_mean={logits.mean().item():.4f} "
              f"grad_norm={gn:.4e} params_with_finite_grad={n_with_grad}/{n_total}",
              flush=True)

    assert n_with_grad == n_total, (
        f"STOP: only {n_with_grad}/{n_total} generator parameter tensors received "
        f"finite gradient -- gradient path is broken somewhere")

    tmp_path = Path("training") / "_smoke_phase1_test.pt"
    save_checkpoint(tmp_path, model, optimizer, config, data_scale, data_scale, 2,
                     argparse.Namespace(smoke=True))
    reloaded = torch.load(tmp_path, map_location="cpu", weights_only=False)
    model2 = CANDIModel(**reloaded["config"])
    model2.load_state_dict(reloaded["model_state_dict"])
    tmp_path.unlink()
    print("[phase1][smoke] checkpoint save/load roundtrip OK", flush=True)
    print(f"[phase1][smoke] SMOKE TEST PASSED: {n_with_grad}/{n_total} param tensors "
          f"received finite gradient", flush=True)


# --------------------------------------------------------------------------
# Main training loop
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Run CPU smoke test and exit")
    ap.add_argument("--equiv-check", action="store_true",
                     help="Run the numeric equivalence check and exit")
    ap.add_argument("--build-stats", action="store_true",
                     help="(Re)build the frozen channel-standardization stats and exit")
    ap.add_argument("--data-dir", default="training")
    ap.add_argument("--src-ckpt", default=SRC_CKPT_NAME)
    ap.add_argument("--phase1-ckpt", default=PHASE1_CKPT_NAME)
    ap.add_argument("--load-from", default=None)
    ap.add_argument("--critic-ckpt", default=str(CRITIC_CKPT_PATH))
    ap.add_argument("--stats-path", default=str(STATS_PATH))
    ap.add_argument("--k", type=int, default=N_SAMPLE_STEPS)
    ap.add_argument("--n-steps", type=int, default=N_SAMPLE_STEPS)
    ap.add_argument("--sample-batch", type=int, default=64,
                     help="Batch size for the seq-len-bucketed generated batch fed to the critic")
    ap.add_argument("--pool-mult", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=128,
                     help="Batch size for the real-data flow-matching anchor minibatch")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--anchor-weight", type=float, default=1.0,
                     help="Lambda on the flow-matching anchor term: "
                          "loss = fool_loss + anchor_weight * flow_matching_anchor")
    ap.add_argument("--disc-weight", type=float, default=1.0)
    ap.add_argument("--guide", type=float, default=GUIDE)
    ap.add_argument("--perp-scale", type=float, default=PERP_SCALE)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--total-steps", type=int, default=1000)
    ap.add_argument("--reset-schedule", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=90.0)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--watchdog-log", default=str(WATCHDOG_LOG))
    args = ap.parse_args()

    if args.smoke:
        smoke_test()
        return

    if args.build_stats:
        build_and_save_channel_stats(Path(args.stats_path))
        return

    if args.equiv_check:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        run_equivalence_check(device)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION, device=0)
        total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        print(f"[phase1] VRAM cap: {VRAM_FRACTION:.2f} x {total_mb:.0f}MB = "
              f"{VRAM_FRACTION * total_mb:.0f}MB (spill-to-shared-memory guard)", flush=True)

    data_dir = Path(args.data_dir)
    src_path = data_dir / args.src_ckpt
    phase1_path = data_dir / args.phase1_ckpt
    phase1_latest_path = phase1_path.with_stem(phase1_path.stem + "_latest")

    md5_before = md5_file(src_path)
    print(f"[phase1] source MD5 before: {md5_before} (expected {EXPECTED_SRC_MD5})", flush=True)
    assert md5_before == EXPECTED_SRC_MD5, "source checkpoint MD5 does not match expected -- STOP"

    ensure_phase1_copy(src_path, phase1_path)

    load_path = Path(args.load_from) if args.load_from else phase1_path
    ckpt = torch.load(load_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    data_scale = ckpt["data_scale"]
    data_std = ckpt.get("data_std", data_scale)
    assert ckpt.get("polar", False), "expected polar checkpoint"
    assert ckpt.get("pred_type", "x0") == "flow", "expected flow pred_type checkpoint"

    model = CANDIModel(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[phase1] loaded {load_path} device={device} params={n_params:,}", flush=True)

    critic = load_frozen_critic(device, Path(args.critic_ckpt))
    scales = load_channel_stats(Path(args.stats_path))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if "optimizer_state_dict" in ckpt and args.load_from:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("[phase1] resumed optimizer state (AdamW momentum) from checkpoint", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[phase1] WARNING: could not resume optimizer state: {exc}", flush=True)

    start_step = 0
    if args.load_from and not args.reset_schedule:
        start_step = int(ckpt.get("global_step", 0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_steps)
    for _ in range(start_step):
        scheduler.step()
    print(f"[phase1] schedule start_step={start_step} reset_schedule={args.reset_schedule} "
          f"total_steps={args.total_steps}", flush=True)

    print("[phase1] loading training data pools...", flush=True)
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
    print(f"[phase1] cond pool: {len(distances):,} human_distances.npy distances "
          f"+ DurationModel", flush=True)

    rng = np.random.default_rng(args.seed)
    spd_s, dh_s = float(data_scale[0]), float(data_scale[1])

    t_burst_start = time.perf_counter()
    max_seconds = args.max_minutes * 60.0
    global_step = start_step
    steps_done_this_run = 0
    bce = nn.BCEWithLogitsLoss(reduction="none")

    print(f"[phase1] === starting training loop: k={args.k} n_steps={args.n_steps} "
          f"sample_batch={args.sample_batch} pool_mult={args.pool_mult} "
          f"batch_size={args.batch_size} lr={args.lr} anchor_weight={args.anchor_weight} "
          f"max_minutes={args.max_minutes} ===", flush=True)

    while True:
        elapsed_burst = time.perf_counter() - t_burst_start
        if elapsed_burst >= max_seconds:
            print(f"[phase1] hard wall clock reached ({elapsed_burst/60:.1f} min), "
                  f"stopping cleanly", flush=True)
            break
        if args.max_steps is not None and steps_done_this_run >= args.max_steps:
            print(f"[phase1] reached --max-steps={args.max_steps}, stopping", flush=True)
            break

        t_step0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # --- (a) flow-matching anchor: forward+backward first, frees the
        # graph before the fool-loss's full checkpointed chain is built
        # (same VRAM-ordering trick as train_candi_chain.py/chain2.py).
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
        (args.anchor_weight * flow_loss).backward()
        anchor_grad_norm = total_grad_norm(model)

        # --- (b) fool loss: full-chain differentiable generation -> torch
        # channels -> frozen standardization -> frozen critic.
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
        gen_std = gen_channels_from_sample(dp_final, pad_mask, tgt_angle, total_dist,
                                            spd_s, dh_s, scales, args.perp_scale)
        logits = critic(gen_std, pad_mask)
        target = torch.zeros_like(logits)  # label 0 = human
        fool_loss = nn.BCEWithLogitsLoss()(logits, target)
        if not torch.isfinite(fool_loss):
            print(f"[phase1] WARNING: non-finite fool_loss at step {global_step}, "
                  f"skipping this step's fool term", flush=True)
            fool_loss = torch.zeros((), device=device)
        else:
            fool_loss.backward()

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

        temp = read_last_temp(Path(args.watchdog_log))
        lr_now = scheduler.get_last_lr()[0]
        print(f"[phase1] step={global_step:5d} fool_loss={fool_loss.item():.4f} "
              f"flow_loss={flow_loss.item():.4f} logit_mean={logits.mean().item():.4f} "
              f"logit_std={logits.std().item():.4f} bucket_T={bucket_T} "
              f"anchor_grad_norm={anchor_grad_norm:.4e} grad_norm={grad_norm_step:.4e} "
              f"lr={lr_now:.2e} peak_vram={peak_vram_mb:.0f}MB step_s={step_elapsed:.2f} "
              f"burst_min={elapsed_burst/60:.2f} temp={temp}", flush=True)

        if global_step % args.ckpt_every == 0:
            save_checkpoint(phase1_latest_path, model, optimizer, config, data_scale,
                             data_std, global_step, args)
            print(f"[phase1] saved {phase1_latest_path} at step {global_step}", flush=True)

    save_checkpoint(phase1_latest_path, model, optimizer, config, data_scale,
                     data_std, global_step, args)
    save_checkpoint(phase1_path, model, optimizer, config, data_scale,
                     data_std, global_step, args)
    print(f"[phase1] final save: {phase1_path} and {phase1_latest_path} at step {global_step}",
          flush=True)

    md5_after = md5_file(src_path)
    print(f"[phase1] source MD5 after: {md5_after}", flush=True)
    if md5_after != md5_before:
        print("[phase1] *** WARNING: source checkpoint MD5 CHANGED -- should never happen ***",
              flush=True)
    else:
        print("[phase1] source checkpoint MD5 unchanged, confirmed untouched.", flush=True)

    print(f"[phase1] DONE. steps_done_this_run={steps_done_this_run} "
          f"global_step={global_step} wall_clock_min={(time.perf_counter()-t_burst_start)/60:.1f}",
          flush=True)


if __name__ == "__main__":
    main()
