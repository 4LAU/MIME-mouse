"""Burst 2 of DIFFUSION_PILOT_V2.md Phase 1: retargeted curvature loss.

Burst 1 (training/train_candi_chain.py, training/candi_polar_flow_chain.pt)
trained cleanly but scored flat at N=2000 (natural AUC 0.754, detector-space
variety ratios 0.571/0.606, statistically unchanged from the pre-pilot
baseline). A decomposition diagnostic found two mis-aims, both fixed here:

  1. The loss compared the synthetic batch to a same-step HUMAN BATCH drawn
     from the training pool, both at a FIXED seq_len=192 restricted to the
     >=192-length subset of the pool. That subset's implied distances ran
     2.2x the scoring median, so the model was fine-tuned on an off-manifold
     slice of conditions it will never actually see at eval time, and against
     a moving per-batch human target instead of a stable population target.
     Fix: conds are now drawn every step from the SAME recipe
     research/phase_a_baseline.py's scoring uses (data/human_distances.npy
     distances + experiments._common.DurationModel), bucketed by their
     implied seq_len for efficient padded batching (see sample_cond_batch),
     never fixed at one length. The human side of the comparison is now four
     FIXED SCALARS (mean/std of detector-space curvature_mean and
     curvature_std) precomputed ONCE offline by
     training/compute_human_curv_targets.py from a large training-pool
     sample at native lengths (training/human_curv_targets.json) -- never
     data/human_val_features_grpo.npy (post-hoc scoring only) or
     data/human_eval_features.npy (never touched anywhere in this file).

  2. The loss read the model's raw x0_hat polar output BEFORE the
     rotate-to-target + perp-0.85 post-processing scoring actually applies
     (research/phase_a_baseline.py's build_trajectory), so it never saw the
     compression that transform adds. Fix: decode_and_curvature applies that
     same differentiable rotate + perp transform to the cartesian decode
     before computing curvature, using features.py's exact cross-product
     curvature formula (not curvature_loss.py's atan2-diff mirror, which was
     an acceptable proxy for burst 1's per-step-friendly loss but is not the
     formula the RF detector's feature space actually measures).

Honest logging (retarget spec item 4): the batch-64 across-batch std reads
~+0.13 too high vs the N=2000 gate (heavy-tail under-sampling). Gradients
still use the current (batch-size) draw, but every step's per-path detector
stats are also pushed into a rolling buffer (>=256 entries, prefilled by a
no-grad warm-up pass before step 0) and the LOGGED variety ratio comes from
that buffer, not the raw per-step batch.

Carryforwards from burst 1 (all unchanged in spirit): VRAM fraction cap,
backward-flow-before-curv-graph memory ordering, per-step gradient
checkpointing through the full 200-step sampler chain, hard 90-minute
wall clock enforced in-process, checkpoint every --ckpt-every steps, lr
1e-5 default, one-time grad-norm lambda calibration, external GPU temp
watchdog log read for display only (the real enforcement is the separate
research/gpu_watchdog.py process).

FRESH START: copies training/candi_polar_flow_best.pt to
training/candi_polar_flow_chain2.pt (NOT burst 1's chain.pt -- that drifted
without validation gain and is not resumed here). Never writes to best.pt.

CPU smoke test: `python -m training.train_candi_chain2 --smoke`
(fabricated tensors, no file I/O, no GPU, verifies finite loss / nonzero
grad / checkpoint save-load roundtrip for the new decode_and_curvature +
curvature_target_loss path).
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

TRAIN_DIR = Path("training")
DATA_DIR = Path("data")
SRC_CKPT_NAME = "candi_polar_flow_best.pt"
CHAIN_CKPT_NAME = "candi_polar_flow_chain2.pt"
EXPECTED_SRC_MD5 = "91326a29750789f3167055324ef377c5"
WATCHDOG_LOG = Path("research/gpu_temp_phase1.log")
TARGETS_PATH = TRAIN_DIR / "human_curv_targets.json"

VRAM_FRACTION = 0.80
N_SAMPLE_STEPS = 200
GUIDE = 0.15
PERP_SCALE = 0.85
HZ = 125.0
ROLL_BUFFER_MAXLEN = 512
ROLL_BUFFER_MIN = 256
_EPS = 1e-6


def ensure_chain_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        print(f"[chain2] {dst} already exists, NOT overwriting with source "
              f"(resume path). Delete it manually to start a fresh chain.",
              flush=True)
        return
    import shutil
    shutil.copy2(src, dst)
    print(f"[chain2] copied {src} -> {dst}", flush=True)


def load_targets(path: Path) -> dict:
    targets = json.loads(path.read_text())
    for key in ("target_mean_curvature_mean_det", "target_std_curvature_mean_det",
                "target_mean_curvature_std_det", "target_std_curvature_std_det"):
        assert key in targets, f"missing {key} in {path}"
    return targets


# --------------------------------------------------------------------------
# Retarget spec item 2: seq-len-bucketed cond sampling from the full pool
# --------------------------------------------------------------------------

def sample_cond_batch(distances, duration_model, batch_size, pool_mult, max_seq_len, rng):
    """Draws a seq-len-bucketed batch of generation conds the same way
    research/phase_a_baseline.py's make_specs + generate_paths spec loop
    does: distance ~ data/human_distances.npy, angle ~ uniform(0, 2*pi),
    duration ~ DurationModel(log_dist). Over-samples a pool of
    pool_mult*batch_size candidates, sorts by implied seq_len, and takes one
    random contiguous window of size batch_size -- buckets similar-length
    conds together for low-padding-waste batching while a fresh random
    window each step still covers the whole length distribution over many
    steps (never fixed at one length, never restricted to any length
    subset -- burst 1's two mistakes).

    Returns cond (batch_size, 4) float32, angle (batch_size,) float32,
    total_dist (batch_size,) float32, pad_mask (batch_size, bucket_T) bool,
    bucket_T int.
    """
    pool_n = pool_mult * batch_size
    dist = rng.choice(distances, size=pool_n)
    angle = rng.uniform(0.0, 2.0 * math.pi, size=pool_n)
    log_dist = np.log(dist)
    seq_len = np.empty(pool_n, dtype=np.int64)
    log_dur = np.empty(pool_n, dtype=np.float64)
    for i in range(pool_n):
        dur = duration_model.sample(float(log_dist[i]))
        log_dur[i] = math.log(dur)
        seq_len[i] = max(5, min(int(round(dur * HZ)), max_seq_len))

    order = np.argsort(seq_len, kind="stable")
    start = int(rng.integers(0, pool_n - batch_size + 1))
    sel = order[start:start + batch_size]

    cond = np.stack(
        [log_dist[sel], log_dur[sel], np.cos(angle[sel]), np.sin(angle[sel])],
        axis=1,
    ).astype(np.float32)
    lens = seq_len[sel]
    total_dist = dist[sel].astype(np.float32)
    bucket_T = int(lens.max())
    pad_mask = np.zeros((batch_size, bucket_T), dtype=np.bool_)
    for i, L in enumerate(lens):
        pad_mask[i, :L] = True
    return cond, angle[sel].astype(np.float32), total_dist, pad_mask, bucket_T


# --------------------------------------------------------------------------
# Retarget spec item 3: differentiable rotate + perp decode, features.py
# cross-product curvature, detector-space transform
# --------------------------------------------------------------------------

def to_detector_space(col: torch.Tensor) -> torch.Tensor:
    """log1p(clip(x, 0, None) * 1e3) -- research/phase_a_baseline.py's
    to_detector_space_curv, the space the RF detector / checkpoint's
    feat_mu/feat_sd actually operate in for the two curvature columns."""
    return torch.log1p(torch.clamp(col, min=0.0) * 1e3)


def _safe_hypot(x: torch.Tensor, y: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """torch.hypot(0, 0) is exactly 0 in the forward pass but its gradient
    (x / hypot(x, y)) is 0/0 = NaN at that point -- padded positions have
    exactly-zero displacement, so a real padded batch hits this every step.
    Multiplying the eventual loss by a mask does not fix it: 0 * NaN is
    still NaN, and it poisons the whole backward pass. sqrt(x^2+y^2+eps^2)
    is smooth everywhere (gradient 0 at the origin, not NaN) and numerically
    identical to hypot away from the origin."""
    return torch.sqrt(x * x + y * y + eps * eps)


def decode_and_curvature(dp_final, pad_mask, tgt_angle, total_dist, spd_s, dh_s,
                          perp_scale=PERP_SCALE, hz=HZ, eps=_EPS):
    """dp -> speed/heading -> cumsum cartesian decode (unchanged from burst
    1), THEN the differentiable rotate-to-target + perp-scale transform
    (same formulas as research/phase_a_baseline.py's build_trajectory with
    CORRECT="rotate", PERP_SCALE=0.85 -- what burst 1's loss never saw,
    since it read x0 before decode), THEN features.py's cross-product
    curvature formula on the corrected cartesian path.

    dp_final: (B, T, 2) polar x0_hat, gradients intact.
    pad_mask: (B, T) bool, True = real / False = padded.
    tgt_angle, total_dist: (B,) -- the cond's target heading angle and the
      physical point-to-point distance (data/human_distances.npy units).

    Returns (mean_det, std_det, path_valid), each (B,): per-path
    curvature_mean / curvature_std in detector space, and a validity mask
    (>= 3 curvature samples, matching features.py's len(pts) >= 5 floor).
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

    # rotate-to-target: build_trajectory's CORRECT="rotate" branch. The
    # target direction is always unit-normalized (target_dx/total_dist etc.
    # has magnitude 1 by construction), so tgt_mag == 1 and scale == 1/raw_mag.
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

    # perp-scale: build_trajectory's PERP_SCALE branch (target unit vector
    # is (cos(tgt_angle), sin(tgt_angle)) since target_dx/dy are already
    # unit-normalized).
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
    dxp = x_full[:, 1:] - x_full[:, :-1]
    dyp = y_full[:, 1:] - y_full[:, :-1]
    vx2 = dxp / dt
    vy2 = dyp / dt
    speed2 = _safe_hypot(dxp, dyp, eps) / dt
    ax = (vx2[:, 1:] - vx2[:, :-1]) / dt
    ay = (vy2[:, 1:] - vy2[:, :-1]) / dt
    speed_mid = speed2[:, :-1].clamp(min=eps)
    cross = (vx2[:, :-1] * ay - vy2[:, :-1] * ax).abs()
    curvature = torch.clamp(cross / speed_mid.pow(3), 0.0, 1e6)

    mask0 = torch.cat([torch.ones(B, 1, device=mask.device), mask], dim=1)
    mask1 = mask0[:, 1:] * mask0[:, :-1]
    step_valid = mask1[:, 1:] * mask1[:, :-1]

    curvature = curvature * step_valid
    n_valid = step_valid.sum(dim=1)
    path_mean = curvature.sum(dim=1) / n_valid.clamp(min=1.0)
    path_sqsum = (curvature ** 2).sum(dim=1)
    path_var = path_sqsum / n_valid.clamp(min=1.0) - path_mean ** 2
    path_std = torch.sqrt(path_var.clamp(min=eps))
    path_valid = n_valid >= 3

    return to_detector_space(path_mean), to_detector_space(path_std), path_valid


def curvature_target_loss(mean_det, std_det, path_valid, targets, eps=_EPS):
    """Retarget spec item 3: penalize the ACROSS-BATCH std (primary) and
    mean of the per-path detector-space curvature stats against the four
    FIXED scalar targets (training/human_curv_targets.json), instead of
    burst 1's per-step human comparison batch."""
    if path_valid.sum() < 2:
        z = mean_det.sum() * 0.0
        return z, {"batch_mean_of_mean": float("nan"), "batch_std_of_mean": float("nan"),
                    "batch_mean_of_std": float("nan"), "batch_std_of_std": float("nan")}

    m = mean_det[path_valid]
    s = std_det[path_valid]

    def _rel_sq(a, b):
        return ((a - b) / (abs(b) + eps)) ** 2

    m_mean, m_std = m.mean(), m.std(unbiased=False).clamp(min=eps)
    s_mean, s_std = s.mean(), s.std(unbiased=False).clamp(min=eps)

    loss = (
        _rel_sq(m_mean, targets["target_mean_curvature_mean_det"])
        + _rel_sq(m_std, targets["target_std_curvature_mean_det"])
        + _rel_sq(s_mean, targets["target_mean_curvature_std_det"])
        + _rel_sq(s_std, targets["target_std_curvature_std_det"])
    )
    stats = {
        "batch_mean_of_mean": float(m_mean.detach()),
        "batch_std_of_mean": float(m_std.detach()),
        "batch_mean_of_std": float(s_mean.detach()),
        "batch_std_of_std": float(s_std.detach()),
        "variety_ratio_mean_batch": float(m_std.detach()) / max(targets["target_std_curvature_mean_det"], eps),
        "variety_ratio_std_batch": float(s_std.detach()) / max(targets["target_std_curvature_std_det"], eps),
    }
    return loss, stats


# --------------------------------------------------------------------------
# Retarget spec item 4: rolling buffer for honest logging
# --------------------------------------------------------------------------

class RollingVariety:
    """Rolling buffer of recent per-path detector-space curvature stats
    (>= ROLL_BUFFER_MIN entries), used ONLY for logging -- gradients always
    use the current step's batch. Fixes the batch-64 across-batch std
    reading ~+0.13 too high from heavy-tail under-sampling."""

    def __init__(self, targets, maxlen=ROLL_BUFFER_MAXLEN):
        self.buf_mean = deque(maxlen=maxlen)
        self.buf_std = deque(maxlen=maxlen)
        self.targets = targets

    def push(self, mean_det, std_det, path_valid):
        m = mean_det[path_valid].detach().cpu().numpy()
        s = std_det[path_valid].detach().cpu().numpy()
        self.buf_mean.extend(m.tolist())
        self.buf_std.extend(s.tolist())

    def ready(self):
        return len(self.buf_mean) >= ROLL_BUFFER_MIN

    def read(self):
        if not self.buf_mean:
            return float("nan"), float("nan"), 0
        vr_mean = float(np.std(self.buf_mean)) / max(self.targets["target_std_curvature_mean_det"], _EPS)
        vr_std = float(np.std(self.buf_std)) / max(self.targets["target_std_curvature_std_det"], _EPS)
        return vr_mean, vr_std, len(self.buf_mean)


# --------------------------------------------------------------------------
# CPU smoke test (fabricated tensors, no file I/O, no GPU)
# --------------------------------------------------------------------------

def smoke_test():
    print("[smoke2] CPU smoke test: K=4, batch=4, 3 steps, fabricated cond/pad_mask", flush=True)
    device = torch.device("cpu")
    torch.manual_seed(0)
    config = dict(d_model=32, n_heads=2, n_layers=2, d_ff=64, max_seq_len=16,
                  cond_dim=4, n_diffusion_steps=100, cond_dropout=0.1, dropout=0.0)
    model = CANDIModel(**config).to(device)
    data_scale = np.array([13.95, 2.33], dtype=np.float32)
    spd_s, dh_s = float(data_scale[0]), float(data_scale[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    targets = {
        "target_mean_curvature_mean_det": 10.5,
        "target_std_curvature_mean_det": 3.3,
        "target_mean_curvature_std_det": 11.8,
        "target_std_curvature_std_det": 3.9,
    }
    roller = RollingVariety(targets, maxlen=64)

    batch = 4
    seq_len_chain = 8
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
        # fabricated bucketed cond batch: two "lengths" via pad_mask, no file I/O
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
        mean_det, std_det, path_valid = decode_and_curvature(
            dp_final, pad_mask, tgt_angle, total_dist, spd_s, dh_s)
        curv_loss, curv_stats = curvature_target_loss(mean_det, std_det, path_valid, targets)
        roller.push(mean_det, std_det, path_valid)

        total = flow_loss + 0.5 * curv_loss
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        gn = total_grad_norm(model)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        assert math.isfinite(float(flow_loss.item())), "flow loss not finite"
        assert math.isfinite(float(curv_loss.item())), "curv loss not finite"
        assert gn > 0.0, "grad norm is zero"
        print(f"[smoke2] step={step} flow_loss={flow_loss.item():.4f} "
              f"curv_loss={curv_loss.item():.4f} grad_norm={gn:.4e} "
              f"path_valid={int(path_valid.sum())}/{batch} stats={curv_stats}", flush=True)

    vr_mean, vr_std, n = roller.read()
    print(f"[smoke2] rolling buffer n={n} vr_mean={vr_mean:.4f} vr_std={vr_std:.4f}", flush=True)

    tmp_path = Path("training") / "_smoke_chain2_test.pt"
    save_checkpoint(tmp_path, model, optimizer, config, data_scale, data_scale, 3,
                     argparse.Namespace(smoke=True))
    reloaded = torch.load(tmp_path, map_location="cpu", weights_only=False)
    model2 = CANDIModel(**reloaded["config"])
    model2.load_state_dict(reloaded["model_state_dict"])
    tmp_path.unlink()
    print("[smoke2] checkpoint save/load roundtrip OK", flush=True)
    print("[smoke2] SMOKE TEST PASSED", flush=True)


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
                     help="Batch size for the seq-len-bucketed curvature batch")
    ap.add_argument("--pool-mult", type=int, default=6,
                     help="Oversample multiplier for the seq-len bucketing window")
    ap.add_argument("--batch-size", type=int, default=128,
                     help="Batch size for the real-data flow-matching anchor minibatch")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--curv-weight", type=float, default=0.0,
                     help="Lambda for the curvature loss term; 0.0 = auto-calibrate "
                          "from the measured grad-norm ratio at step 0")
    ap.add_argument("--disc-weight", type=float, default=1.0)
    ap.add_argument("--guide", type=float, default=GUIDE)
    ap.add_argument("--perp-scale", type=float, default=PERP_SCALE)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--total-steps", type=int, default=1000)
    ap.add_argument("--reset-schedule", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=90.0)
    ap.add_argument("--ckpt-every", type=int, default=25)
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
        print(f"[chain2] VRAM cap: {VRAM_FRACTION:.2f} x {total_mb:.0f}MB = "
              f"{VRAM_FRACTION * total_mb:.0f}MB (spill-to-shared-memory guard)", flush=True)

    data_dir = Path(args.data_dir)
    src_path = data_dir / args.src_ckpt
    chain_path = data_dir / args.chain_ckpt
    chain_latest_path = chain_path.with_stem(chain_path.stem + "_latest")

    md5_before = md5_file(src_path)
    print(f"[chain2] source MD5 before: {md5_before} (expected {EXPECTED_SRC_MD5})", flush=True)
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
    print(f"[chain2] loaded {load_path} device={device} params={n_params:,}", flush=True)

    targets = load_targets(Path(args.targets))
    print(f"[chain2] human curvature targets: {targets}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if "optimizer_state_dict" in ckpt and args.load_from:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("[chain2] resumed optimizer state (AdamW momentum) from checkpoint", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[chain2] WARNING: could not resume optimizer state: {exc}", flush=True)

    start_step = 0
    if args.load_from and not args.reset_schedule:
        start_step = int(ckpt.get("global_step", 0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_steps)
    for _ in range(start_step):
        scheduler.step()
    print(f"[chain2] schedule start_step={start_step} reset_schedule={args.reset_schedule} "
          f"total_steps={args.total_steps}", flush=True)

    # --- data pools ---
    print("[chain2] loading training data pools...", flush=True)
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

    # Retarget spec item 2: cond sampling from the full-pool-equivalent
    # recipe (human_distances.npy + DurationModel), NOT restricted to a
    # length subset of the training pool.
    distances = np.load(DATA_DIR / "human_distances.npy")
    duration_model = DurationModel(str(data_dir), std_mult=0.7)
    print(f"[chain2] cond pool: {len(distances):,} human_distances.npy distances "
          f"+ DurationModel (never restricted to a length subset)", flush=True)

    rng = np.random.default_rng(args.seed)
    spd_s, dh_s = float(data_scale[0]), float(data_scale[1])

    # --- rolling-buffer warm-up (retarget spec item 4) ---
    roller = RollingVariety(targets)
    n_warmup = args.warmup_batches
    if n_warmup is None:
        n_warmup = math.ceil(ROLL_BUFFER_MIN / args.sample_batch)
    print(f"[chain2] warming up rolling buffer with {n_warmup} no-grad batches "
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
            mean_det, std_det, path_valid = decode_and_curvature(
                dp_final, pad_mask, tgt_angle, total_dist, spd_s, dh_s, args.perp_scale)
            roller.push(mean_det, std_det, path_valid)
    vr_mean0, vr_std0, n_buf = roller.read()
    print(f"[chain2] === STEP-0 PRE-TRAINING ROLLING VARIETY READ (n={n_buf}, "
          f"warmup {time.perf_counter()-t_warm0:.1f}s): "
          f"vr_mean={vr_mean0:.4f} vr_std={vr_std0:.4f} === "
          f"(expected ~0.55; if >=0.8 the retarget failed -- STOP)", flush=True)

    t_burst_start = time.perf_counter()
    max_seconds = args.max_minutes * 60.0
    global_step = start_step
    steps_done_this_run = 0
    bce = nn.BCEWithLogitsLoss(reduction="none")

    print(f"[chain2] === starting training loop: k={args.k} n_steps={args.n_steps} "
          f"sample_batch={args.sample_batch} pool_mult={args.pool_mult} "
          f"batch_size={args.batch_size} lr={args.lr} "
          f"curv_weight={'auto' if args.curv_weight == 0.0 else args.curv_weight} "
          f"max_minutes={args.max_minutes} ===", flush=True)

    lambda_chosen = args.curv_weight
    while True:
        elapsed_burst = time.perf_counter() - t_burst_start
        if elapsed_burst >= max_seconds:
            print(f"[chain2] hard 90-minute wall clock reached ({elapsed_burst/60:.1f} min), "
                  f"stopping cleanly", flush=True)
            break
        if args.max_steps is not None and steps_done_this_run >= args.max_steps:
            print(f"[chain2] reached --max-steps={args.max_steps}, stopping", flush=True)
            break

        t_step0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # --- (a) real-data flow-matching + stall anchor loss (unchanged carryforward) ---
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

        # --- (b) retargeted curvature loss on a seq-len-bucketed sampled batch ---
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
        mean_det, std_det, path_valid = decode_and_curvature(
            dp_final, pad_mask, tgt_angle, total_dist, spd_s, dh_s, args.perp_scale)
        curv_loss, curv_stats = curvature_target_loss(mean_det, std_det, path_valid, targets)
        if not torch.isfinite(curv_loss):
            print(f"[chain2] WARNING: non-finite curv_loss at step {global_step}, "
                  f"skipping this step's curvature term", flush=True)
            curv_loss = torch.zeros((), device=device)

        roller.push(mean_det, std_det, path_valid)

        curv_grad_norm = None
        if global_step == start_step:
            curv_loss.backward()
            curv_grad_norm = total_grad_norm(model)
            if lambda_chosen == 0.0:
                lambda_chosen = flow_grad_norm / max(curv_grad_norm, 1e-12)
                print(f"[chain2] AUTO-CALIBRATED lambda={lambda_chosen:.6e} "
                      f"(flow_grad_norm={flow_grad_norm:.6e} / curv_grad_norm={curv_grad_norm:.6e})",
                      flush=True)
            else:
                print(f"[chain2] measured flow_grad_norm={flow_grad_norm:.6e} "
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

        vr_mean_roll, vr_std_roll, n_buf = roller.read()
        temp = read_last_temp(Path(args.watchdog_log))
        lr_now = scheduler.get_last_lr()[0]
        gn_msg = ""
        if curv_grad_norm is not None:
            gn_msg = (f" | CALIBRATION flow_grad_norm={flow_grad_norm:.4e} "
                      f"curv_grad_norm={curv_grad_norm:.4e} lambda={lambda_chosen:.4e}")
        print(f"[chain2] step={global_step:5d} flow_loss={flow_loss.item():.4f} "
              f"curv_loss={curv_loss.item():.4f} bucket_T={bucket_T} "
              f"vr_mean_batch={curv_stats.get('variety_ratio_mean_batch', float('nan')):.4f} "
              f"vr_std_batch={curv_stats.get('variety_ratio_std_batch', float('nan')):.4f} "
              f"vr_mean_roll(n={n_buf})={vr_mean_roll:.4f} vr_std_roll={vr_std_roll:.4f} "
              f"grad_norm={grad_norm_step:.4e} lr={lr_now:.2e} "
              f"peak_vram={peak_vram_mb:.0f}MB step_s={step_elapsed:.2f} "
              f"burst_min={elapsed_burst/60:.2f} temp={temp}{gn_msg}", flush=True)

        if global_step % args.ckpt_every == 0:
            save_checkpoint(chain_latest_path, model, optimizer, config, data_scale,
                             data_std, global_step, args)
            print(f"[chain2] saved {chain_latest_path} at step {global_step}", flush=True)

    save_checkpoint(chain_latest_path, model, optimizer, config, data_scale,
                     data_std, global_step, args)
    save_checkpoint(chain_path, model, optimizer, config, data_scale,
                     data_std, global_step, args)
    print(f"[chain2] final save: {chain_path} and {chain_latest_path} at step {global_step}", flush=True)

    md5_after = md5_file(src_path)
    print(f"[chain2] source MD5 after: {md5_after}", flush=True)
    if md5_after != md5_before:
        print("[chain2] *** WARNING: source checkpoint MD5 CHANGED -- should never happen ***",
              flush=True)
    else:
        print("[chain2] source checkpoint MD5 unchanged, confirmed untouched.", flush=True)

    print(f"[chain2] DONE. steps_done_this_run={steps_done_this_run} "
          f"global_step={global_step} wall_clock_min={(time.perf_counter()-t_burst_start)/60:.1f}",
          flush=True)


if __name__ == "__main__":
    main()
