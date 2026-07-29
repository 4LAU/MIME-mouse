"""Phase 1 of DIFFUSION_PILOT_V2.md: fine-tune CANDI against the FULL
differentiable 200-step sampler chain, not the one-step shortcut that sank
version 1 and not a short tail. research/phase1_feasibility_probe.py
(research/phase1_results.csv) already proved the full chain (K=200) is
affordable with per-step gradient checkpointing: batch 128 = 3.3GB VRAM,
52.3 s/step; batch 16 = 484MB, 6.6 s/step; gradient reaches all 99
parameter tensors.

Each optimizer step combines two terms:
  (a) the standard flow-matching + stall (discrete) loss on a real-data
      minibatch, computed exactly the way training/train_candi.py computes
      it (cont_loss + disc_weight * disc_loss). This anchors the model and
      prevents drift.
  (b) lambda * curvature-moment loss (training/curvature_loss.py) computed
      on a batch of FULLY SAMPLED trajectories. Both the sampling
      conditions and the human comparison target come from the real
      training pool (training/zimt_conditions.npy, zimt_polar_spd.npy,
      zimt_polar_dh.npy) -- never from data/human_val_features_grpo.npy
      (post-hoc scoring only) or data/human_eval_features.npy (never
      touched anywhere in this file). The comparison window is fixed at
      --seq-len-chain steps and restricted to real trajectories at least
      that long, so both sides of the curvature comparison are pad-free.

The synthetic side runs the full 200-step guided-flow sampler with
per-step gradient checkpointing, adapted directly from
research/phase1_feasibility_probe.py's tail_step / prefix_step (pure-
tensor guide block; the two known differentiability blockers from that
probe -- .item() detaching raw_mag/raw_ang, and the discrete reveal state
needing to be frozen rather than updated inside the differentiable window
-- are fixed the same way). At the default K=200 there is no no-grad
prefix at all: the whole chain runs under checkpointing and the discrete
stall state is frozen at its initial (fully-masked) value for the entire
generation, exactly like the probe's K=200 cell.

curvature_moment_loss operates on the polar (speed, delta_heading)
representation directly (see training/curvature_loss.py) -- it does not
need the cartesian decode/rounding that scoring scripts use, so the
per-step tail_step's `dp` (the model's running x0_hat estimate) is passed
straight into it. This mirrors exactly how train_candi.py's v1 curv-loss
term already uses x0_hat.

Hard constraints enforced here:
  - Never reads/loads data/human_eval_features.npy or
    data/human_val_features_grpo.npy anywhere in this file. Scoring is a
    separate step (research/phase_a_baseline.py --ckpt ...).
  - Never imports experiments/candi.py.
  - Never writes to training/candi_polar_flow_best.pt. Copies it once to
    training/candi_polar_flow_chain.pt at startup (if that copy does not
    already exist) and only ever writes to the copy / its _latest sibling.
  - torch.cuda.set_per_process_memory_fraction(0.80) before any CUDA
    tensor is allocated (probe finding: without this, Windows WDDM
    silently spills into shared memory instead of raising OOM).
  - Hard 90-minute wall clock enforced in-process (checked every step,
    independent of the external per-minute GPU temperature watchdog).
  - Checkpoints every --ckpt-every optimizer steps.

CPU smoke test: `python -m training.train_candi_chain --smoke`
(K=4, batch=2, 3 steps, verifies finite loss / nonzero grad / checkpoint
save-load roundtrip, touches no GPU, no real checkpoint files).
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.candi import CANDIModel
from training.curvature_loss import curvature_moment_loss, _path_curvature_moments
from training.train_candi import CANDIDataset

TRAIN_DIR = Path("training")
SRC_CKPT_NAME = "candi_polar_flow_best.pt"
CHAIN_CKPT_NAME = "candi_polar_flow_chain.pt"
EXPECTED_SRC_MD5 = "91326a29750789f3167055324ef377c5"
WATCHDOG_LOG = Path("research/gpu_temp_phase1.log")

VRAM_FRACTION = 0.80  # hard constraint: prevents WDDM silent spill to shared memory
N_SAMPLE_STEPS = 200
GUIDE = 0.15  # matches the published 0.752 generation config


# --------------------------------------------------------------------------
# MD5 / checkpoint plumbing
# --------------------------------------------------------------------------

def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_chain_copy(src: Path, dst: Path) -> None:
    """Copy the protected source checkpoint to the working chain checkpoint,
    ONLY if the chain checkpoint does not already exist (so a resumed run
    doesn't clobber prior burst progress with the original weights again)."""
    if dst.exists():
        print(f"[chain] {dst} already exists, NOT overwriting with source "
              f"(resume path). Delete it manually to start a fresh chain.",
              flush=True)
        return
    shutil.copy2(src, dst)
    print(f"[chain] copied {src} -> {dst}", flush=True)


# --------------------------------------------------------------------------
# Differentiable sampler (adapted from research/phase1_feasibility_probe.py)
# --------------------------------------------------------------------------

def prefix_step(model, xt, stall_s, mflag, cond, tgt_angle, i, n_steps, spd_s, dh_s, guide,
                 pad_mask=None):
    """No-grad sampler step, used only for the steps before the
    differentiable window when --k < 200. Identical to the probe's
    prefix_step. Vectorized guide block (no .item() calls) -- safe here
    since this branch is always wrapped in torch.no_grad() by the caller.

    pad_mask (B, T) bool, True=real/False=padded, optional (default None =
    no padding, identical to the original burst-1 behavior): passed to the
    model as a key-padding mask, and folded into the guide block's `keep`
    so padded positions never influence the raw_mag/raw_ang endpoint used
    for guidance (added for burst 2's variable-length bucketed batches)."""
    dt = 1.0 / n_steps
    t_cont = 1.0 - i * dt
    B = xt.shape[0]
    dev = xt.device
    t_scaled = torch.full((B,), t_cont * (model.n_steps - 1), device=dev)
    v_pred, sl = model(xt, stall_s, mflag, t_scaled, cond, pad_mask)
    dp = xt - t_cont * v_pred

    frac = 1.0 - t_cont
    if frac > 0.3:
        conf = torch.abs(sl)
        thresh = max(0.5, 3.0 * (1.0 - frac))
        reveal = (conf > thresh) & (mflag > 0.5)
        stall_s = torch.where(reveal, (torch.sigmoid(sl) > 0.5).float(), stall_s)
        mflag = torch.where(reveal, torch.zeros_like(mflag), mflag)

    if frac > 0.3 and guide > 0:
        spd = torch.clamp(dp[..., 0] / spd_s, min=0)
        dh = dp[..., 1] / dh_s
        active_stall = (stall_s > 0.5) & (mflag < 0.5)
        keep = (~active_stall).float()
        if pad_mask is not None:
            keep = keep * pad_mask.float()
        heading = torch.cumsum(dh * keep, dim=1)
        cx = torch.cumsum(spd * keep * torch.cos(heading), dim=1)
        cy = torch.cumsum(spd * keep * torch.sin(heading), dim=1)
        raw_mag = torch.hypot(cx[:, -1], cy[:, -1])
        raw_ang = torch.atan2(cy[:, -1], cx[:, -1])
        rot = (tgt_angle - raw_ang) * guide * frac
        rot = torch.where(raw_mag > 1e-6, rot, torch.zeros_like(rot))
        dp = dp.clone()
        dp[:, 0, 1] = dp[:, 0, 1] + rot * dh_s

    if t_cont > 1e-6:
        v_guided = (xt - dp) / t_cont
    else:
        v_guided = v_pred
    xt = xt - dt * v_guided
    return xt, stall_s, mflag


def tail_step(xt, *, model, stall_frozen, mflag_frozen, cond, tgt_angle,
              i, n_steps, spd_s, dh_s, guide, pad_mask=None):
    """Differentiable sampler step. Verbatim from
    research/phase1_feasibility_probe.py: stall_frozen / mflag_frozen are
    constants for the whole differentiable window (discrete state frozen,
    no reveal update happens inside it at all); raw_mag / raw_ang stay
    tensors end to end (no .item() detach) with the near-zero-magnitude
    guard done via torch.where instead of a python `if` on a converted
    float, so gradient reaches the model through the guidance term too.

    pad_mask (B, T) bool, optional (default None = no padding): see
    prefix_step's docstring, same treatment here."""
    dt = 1.0 / n_steps
    t_cont = 1.0 - i * dt
    B = xt.shape[0]
    dev = xt.device
    t_scaled = torch.full((B,), t_cont * (model.n_steps - 1), device=dev)
    v_pred, sl = model(xt, stall_frozen, mflag_frozen, t_scaled, cond, pad_mask)
    dp = xt - t_cont * v_pred

    frac = 1.0 - t_cont
    if frac > 0.3 and guide > 0:
        spd = torch.clamp(dp[..., 0] / spd_s, min=0)
        dh = dp[..., 1] / dh_s
        active_stall = (stall_frozen > 0.5) & (mflag_frozen < 0.5)
        keep = (~active_stall).float()
        if pad_mask is not None:
            keep = keep * pad_mask.float()
        heading = torch.cumsum(dh * keep, dim=1)
        cx = torch.cumsum(spd * keep * torch.cos(heading), dim=1)
        cy = torch.cumsum(spd * keep * torch.sin(heading), dim=1)
        raw_mag = torch.hypot(cx[:, -1], cy[:, -1])
        raw_ang = torch.atan2(cy[:, -1], cx[:, -1])
        rot = (tgt_angle - raw_ang) * guide * frac
        rot = torch.where(raw_mag > 1e-6, rot, torch.zeros_like(rot))
        dp = dp.clone()
        dp[:, 0, 1] = dp[:, 0, 1] + rot * dh_s

    if t_cont > 1e-6:
        v_guided = (xt - dp) / t_cont
    else:
        v_guided = v_pred
    xt = xt - dt * v_guided
    return xt, dp


def differentiable_generate(model, cond, tgt_angle, seq_len, spd_s, dh_s,
                             k, n_steps, guide, use_ckpt, device, pad_mask=None):
    """Runs n_steps of guided-flow generation, the last k of them
    differentiably (per-step gradient checkpointed if use_ckpt). Returns
    dp_final: the polar (speed, delta_heading) x0_hat estimate from the
    final differentiable step, gradients intact back to model parameters.
    At the default k == n_steps there is no no-grad prefix at all.

    pad_mask (B, T) bool, optional (default None = no padding, identical to
    burst 1's behavior): threaded through every sampler step as the model's
    key-padding mask and folded into the guide block's keep mask (see
    prefix_step/tail_step docstrings). Used by burst 2's variable-length
    seq-len-bucketed batches; a fixed seq_len batch (burst 1) passes None."""
    B = cond.shape[0]
    xt = torch.randn(B, seq_len, 2, device=device)
    stall_s = torch.full((B, seq_len), model.STALL_MASK, device=device)
    mflag = torch.ones(B, seq_len, device=device)

    n_prefix = n_steps - k
    if n_prefix > 0:
        with torch.no_grad():
            for i in range(n_prefix):
                xt, stall_s, mflag = prefix_step(
                    model, xt, stall_s, mflag, cond, tgt_angle, i, n_steps,
                    spd_s, dh_s, guide, pad_mask,
                )

    stall_frozen = stall_s.detach()
    mflag_frozen = mflag.detach()
    xt = xt.detach().requires_grad_(True)

    dp_final = None
    for i in range(n_prefix, n_steps):
        step_fn = functools.partial(
            tail_step, model=model, stall_frozen=stall_frozen,
            mflag_frozen=mflag_frozen, cond=cond, tgt_angle=tgt_angle,
            i=i, n_steps=n_steps, spd_s=spd_s, dh_s=dh_s, guide=guide,
            pad_mask=pad_mask,
        )
        if use_ckpt:
            xt, dp_final = checkpoint(step_fn, xt, use_reentrant=False)
        else:
            xt, dp_final = step_fn(xt)
    return dp_final


# --------------------------------------------------------------------------
# Real-data batch sampling (training-distribution only; no human eval files)
# --------------------------------------------------------------------------

def sample_flow_anchor_batch(dataset: CANDIDataset, pool_idx: np.ndarray,
                              batch_size: int, rng: np.random.Generator, device):
    """Draws a random real-data minibatch the same shape train_candi.py's
    DataLoader would, without the DataLoader/worker machinery (simpler,
    avoids Windows multiprocessing overhead for a per-step draw)."""
    choice = rng.choice(pool_idx, size=batch_size, replace=len(pool_idx) < batch_size)
    items = [dataset[int(i)] for i in choice]
    dxdy_b = torch.stack([it[0] for it in items]).to(device)
    stall_b = torch.stack([it[1] for it in items]).to(device)
    pad_b = torch.stack([it[2] for it in items]).to(device)
    cond_b = torch.stack([it[3] for it in items]).to(device)
    return dxdy_b, stall_b, pad_b, cond_b


def sample_curv_batch(cond_all, spd_all, dh_all, eligible_idx, seq_len_chain,
                       batch_size, data_scale, rng: np.random.Generator, device):
    """Draws a paired (cond, human_polar) batch from real training
    trajectories at least seq_len_chain steps long, truncated to exactly
    seq_len_chain steps (no padding on either side of the curvature
    comparison -- avoids the mask mismatch a variable-length draw would
    create). cond supplies both the sampler's conditioning vector and the
    guidance target angle; human_polar is the training-distribution
    curvature target (hard constraint 5: never human_val_features_grpo.npy
    or human_eval_features.npy)."""
    choice = rng.choice(eligible_idx, size=batch_size,
                         replace=len(eligible_idx) < batch_size)
    cond_np = cond_all[choice].astype(np.float32)
    cond = torch.from_numpy(cond_np).to(device)
    tgt_angle = torch.atan2(cond[:, 3], cond[:, 2])

    spd = np.asarray(spd_all[choice, :seq_len_chain], dtype=np.float32) * float(data_scale[0])
    dh = np.asarray(dh_all[choice, :seq_len_chain], dtype=np.float32) * float(data_scale[1])
    human_polar = torch.from_numpy(np.stack([spd, dh], axis=-1)).to(device)
    pad_mask = torch.ones(batch_size, seq_len_chain, dtype=torch.bool, device=device)
    return cond, tgt_angle, human_polar, pad_mask


# --------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------

def detector_space(col: np.ndarray) -> np.ndarray:
    """research/phase_a_baseline.py:to_detector_space_curv -- the transform
    the RF detector / checkpoint's feature space actually operates in for
    the two curvature columns."""
    return np.log1p(np.clip(col, 0.0, None) * 1e3)


def variety_ratios_detector_space(dp_final, human_polar, pad_mask, spd_s, dh_s):
    """Per-path curvature_mean / curvature_std detector-space variety
    ratios (synthetic std / human std), the same statistic
    research/phase0_collapse_curve.py and research/phase_a_baseline.py
    report, computed here on the in-training sampled batch instead of a
    full post-hoc generation run."""
    with torch.no_grad():
        s_mean, s_std, s_valid = _path_curvature_moments(dp_final, pad_mask, spd_s, dh_s)
        h_mean, h_std, h_valid = _path_curvature_moments(human_polar, pad_mask, spd_s, dh_s)
        s_mean_np = s_mean[s_valid].cpu().numpy()
        s_std_np = s_std[s_valid].cpu().numpy()
        h_mean_np = h_mean[h_valid].cpu().numpy()
        h_std_np = h_std[h_valid].cpu().numpy()
        if len(s_mean_np) < 2 or len(h_mean_np) < 2:
            return float("nan"), float("nan")
        vr_mean = float(np.std(detector_space(s_mean_np)) / max(np.std(detector_space(h_mean_np)), 1e-12))
        vr_std = float(np.std(detector_space(s_std_np)) / max(np.std(detector_space(h_std_np)), 1e-12))
        return vr_mean, vr_std


def total_grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm().item()) ** 2
    return math.sqrt(total)


def read_last_temp(log_path: Path):
    if not log_path.exists():
        return None
    try:
        with open(log_path, "r") as fh:
            lines = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
        if not lines:
            return None
        return lines[-1].split(",")[-1]
    except Exception:
        return None


# --------------------------------------------------------------------------
# Checkpoint save
# --------------------------------------------------------------------------

def save_checkpoint(path: Path, model, optimizer, config, data_scale, data_std,
                     global_step, args):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "data_scale": data_scale,
        "data_std": data_std,
        "polar": True,
        "pred_type": "flow",
        "global_step": global_step,
        "chain_args": vars(args),
    }
    torch.save(ckpt, path)


# --------------------------------------------------------------------------
# CPU smoke test
# --------------------------------------------------------------------------

def smoke_test():
    print("[smoke] CPU smoke test: K=4, batch=2, 3 steps", flush=True)
    device = torch.device("cpu")
    torch.manual_seed(0)
    config = dict(d_model=32, n_heads=2, n_layers=2, d_ff=64, max_seq_len=16,
                  cond_dim=4, n_diffusion_steps=100, cond_dropout=0.1, dropout=0.0)
    model = CANDIModel(**config).to(device)
    data_scale = np.array([13.95, 2.33], dtype=np.float32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    seq_len_chain = 8
    n_steps = 4
    k = 4
    batch = 2

    rng = np.random.default_rng(0)
    spd_s, dh_s = float(data_scale[0]), float(data_scale[1])

    # --- fake "real" flow-anchor minibatch ---
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
        dp_final = differentiable_generate(
            model, cond, tgt_angle, seq_len_chain, spd_s, dh_s,
            k=k, n_steps=n_steps, guide=GUIDE, use_ckpt=False, device=device,
        )
        human_polar = torch.randn(batch, seq_len_chain, 2, device=device)
        pad_mask = torch.ones(batch, seq_len_chain, dtype=torch.bool, device=device)
        curv_loss, stats = curvature_moment_loss(dp_final, human_polar, pad_mask, spd_s, dh_s)

        total = flow_loss + 0.5 * curv_loss
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        gn = total_grad_norm(model)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        assert math.isfinite(float(flow_loss.item())), "flow loss not finite"
        assert math.isfinite(float(curv_loss.item())), "curv loss not finite"
        assert gn > 0.0, "grad norm is zero"
        print(f"[smoke] step={step} flow_loss={flow_loss.item():.4f} "
              f"curv_loss={curv_loss.item():.4f} grad_norm={gn:.4e}", flush=True)

    # checkpoint save/load roundtrip
    tmp_path = Path("training") / "_smoke_chain_test.pt"
    save_checkpoint(tmp_path, model, optimizer, config, data_scale, data_scale, 3,
                     argparse.Namespace(smoke=True))
    reloaded_config = torch.load(tmp_path, map_location="cpu", weights_only=False)
    model2 = CANDIModel(**reloaded_config["config"])
    model2.load_state_dict(reloaded_config["model_state_dict"])
    tmp_path.unlink()
    print("[smoke] checkpoint save/load roundtrip OK", flush=True)
    print("[smoke] SMOKE TEST PASSED", flush=True)


# --------------------------------------------------------------------------
# Main training loop
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Run CPU smoke test and exit")
    ap.add_argument("--data-dir", default="training")
    ap.add_argument("--src-ckpt", default=SRC_CKPT_NAME)
    ap.add_argument("--chain-ckpt", default=CHAIN_CKPT_NAME)
    ap.add_argument("--load-from", default=None,
                     help="Resume from this checkpoint instead of the fresh "
                          "chain copy (e.g. the _chain_latest.pt from a prior burst)")
    ap.add_argument("--k", type=int, default=N_SAMPLE_STEPS,
                     help="Number of last sampler steps that carry gradient "
                          "(default 200 = full chain, no no-grad prefix)")
    ap.add_argument("--n-steps", type=int, default=N_SAMPLE_STEPS)
    ap.add_argument("--seq-len-chain", type=int, default=192,
                     help="Fixed generation length for the differentiable "
                          "curvature-comparison batch")
    ap.add_argument("--sample-batch", type=int, default=64,
                     help="Batch size for the fully-sampled curvature batch")
    ap.add_argument("--batch-size", type=int, default=128,
                     help="Batch size for the real-data flow-matching anchor minibatch")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--curv-weight", type=float, default=0.0,
                     help="Lambda for the curvature loss term. 0.0 (default) "
                          "means: auto-calibrate from the measured grad-norm "
                          "ratio at step 0 (flow_grad_norm / curv_grad_norm), "
                          "print it, and use it for every step. A positive "
                          "value is used as-is (grad norms are still measured "
                          "and logged once at step 0 for visibility).")
    ap.add_argument("--disc-weight", type=float, default=1.0)
    ap.add_argument("--guide", type=float, default=GUIDE)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--total-steps", type=int, default=1000,
                     help="Cosine schedule horizon in optimizer steps (spans "
                          "multiple future bursts; a single 90-min burst at "
                          "batch 64 is expected to complete ~150-250 steps)")
    ap.add_argument("--reset-schedule", action="store_true",
                     help="Start the cosine schedule from step 0 on resume "
                          "instead of fast-forwarding to the checkpoint's saved step")
    ap.add_argument("--max-minutes", type=float, default=90.0,
                     help="Hard wall-clock cap enforced in-process")
    ap.add_argument("--ckpt-every", type=int, default=25,
                     help="Save the _chain_latest.pt checkpoint every N optimizer steps")
    ap.add_argument("--max-steps", type=int, default=None,
                     help="Optional cap on optimizer steps (for the sanity pilot)")
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
        print(f"[chain] VRAM cap: {VRAM_FRACTION:.2f} x {total_mb:.0f}MB = "
              f"{VRAM_FRACTION * total_mb:.0f}MB (spill-to-shared-memory guard)", flush=True)

    data_dir = Path(args.data_dir)
    src_path = data_dir / args.src_ckpt
    chain_path = data_dir / args.chain_ckpt
    chain_latest_path = chain_path.with_stem(chain_path.stem + "_latest")

    md5_before = md5_file(src_path)
    print(f"[chain] source MD5 before: {md5_before} (expected {EXPECTED_SRC_MD5})", flush=True)
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
    print(f"[chain] loaded {load_path} device={device} params={n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if "optimizer_state_dict" in ckpt and args.load_from:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("[chain] resumed optimizer state (AdamW momentum) from checkpoint", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[chain] WARNING: could not resume optimizer state: {exc}", flush=True)

    start_step = 0
    if args.load_from and not args.reset_schedule:
        start_step = int(ckpt.get("global_step", 0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_steps)
    for _ in range(start_step):
        scheduler.step()
    print(f"[chain] schedule start_step={start_step} reset_schedule={args.reset_schedule} "
          f"total_steps={args.total_steps}", flush=True)

    # --- data pools (training distribution only) ---
    print("[chain] loading training data pools...", flush=True)
    dxdy = np.load(data_dir / "zimt_dxdy.npy", mmap_mode="r")
    stall = np.load(data_dir / "zimt_stall.npy", mmap_mode="r")
    lengths = np.load(data_dir / "zimt_lengths.npy")
    conditions = np.load(data_dir / "zimt_conditions.npy")
    spd_all = np.load(data_dir / "zimt_polar_spd.npy", mmap_mode="r")
    dh_all = np.load(data_dir / "zimt_polar_dh.npy", mmap_mode="r")
    N = len(lengths)

    n_val = min(N // 10, 30000)
    perm = np.random.default_rng(42).permutation(N)  # identical split convention to train_candi.py
    tr_idx = perm[n_val:]

    max_seq_len = config["max_seq_len"]
    flow_dataset = CANDIDataset(
        dxdy, stall, lengths, conditions, max_seq_len, data_scale, polar=True,
        spd=spd_all, dh=dh_all,
    )

    eligible_mask = lengths[tr_idx] >= args.seq_len_chain
    eligible_idx = tr_idx[eligible_mask]
    print(f"[chain] eligible curvature-batch pool: {len(eligible_idx):,} / "
          f"{len(tr_idx):,} train trajectories have length >= {args.seq_len_chain}", flush=True)
    assert len(eligible_idx) >= args.sample_batch, "not enough long trajectories for --sample-batch"

    rng = np.random.default_rng(args.seed)
    spd_s, dh_s = float(data_scale[0]), float(data_scale[1])

    t_burst_start = time.perf_counter()
    max_seconds = args.max_minutes * 60.0
    global_step = start_step
    steps_done_this_run = 0
    bce = nn.BCEWithLogitsLoss(reduction="none")

    print(f"[chain] === starting training loop: k={args.k} n_steps={args.n_steps} "
          f"sample_batch={args.sample_batch} batch_size={args.batch_size} lr={args.lr} "
          f"curv_weight={'auto' if args.curv_weight == 0.0 else args.curv_weight} "
          f"max_minutes={args.max_minutes} ===", flush=True)

    lambda_chosen = args.curv_weight
    while True:
        elapsed_burst = time.perf_counter() - t_burst_start
        if elapsed_burst >= max_seconds:
            print(f"[chain] hard 90-minute wall clock reached ({elapsed_burst/60:.1f} min), "
                  f"stopping cleanly", flush=True)
            break
        if args.max_steps is not None and steps_done_this_run >= args.max_steps:
            print(f"[chain] reached --max-steps={args.max_steps}, stopping", flush=True)
            break

        t_step0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # --- (a) real-data flow-matching + stall anchor loss ---
        # Forward AND backward this term fully before building the curvature
        # graph (below), so the two graphs are never simultaneously resident
        # in VRAM. Both graphs together (flow: batch 128 seq<=256; curv: full
        # 200-step checkpointed chain) exceeded the 80%-VRAM cap when the
        # first version of this script computed both forwards before either
        # backward -- see EXPERIMENTS.md-style note in the pilot report.
        # backward() accumulates into .grad by default (no zero_grad in
        # between), so a plain flow_loss.backward() followed later by
        # (lambda * curv_loss).backward() gives the exact same combined
        # gradient as (flow_loss + lambda*curv_loss).backward() would, at a
        # fraction of the peak memory.
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
        flow_loss.backward()  # frees the flow graph immediately; .grad = d(flow)/dtheta
        flow_grad_norm = total_grad_norm(model)
        if global_step == start_step:
            flow_grads = {name: (p.grad.detach().clone() if p.grad is not None else None)
                          for name, p in model.named_parameters()}
            optimizer.zero_grad(set_to_none=True)  # isolate curv's raw grad for calibration

        # --- (b) curvature-moment loss on a fully-sampled batch ---
        # Built fresh now that the flow graph above has already been freed.
        model.eval()
        cond, tgt_angle, human_polar, pad_mask = sample_curv_batch(
            conditions, spd_all, dh_all, eligible_idx, args.seq_len_chain,
            args.sample_batch, data_scale, rng, device,
        )
        dp_final = differentiable_generate(
            model, cond, tgt_angle, args.seq_len_chain, spd_s, dh_s,
            k=args.k, n_steps=args.n_steps, guide=args.guide, use_ckpt=True, device=device,
        )
        curv_loss, curv_stats = curvature_moment_loss(dp_final, human_polar, pad_mask, spd_s, dh_s)
        if not torch.isfinite(curv_loss):
            print(f"[chain] WARNING: non-finite curv_loss at step {global_step}, "
                  f"skipping this step's curvature term", flush=True)
            curv_loss = torch.zeros((), device=device)

        curv_grad_norm = None
        if global_step == start_step:
            # One-time grad-norm calibration (hard requirement: measure once
            # at pilot start, not every step -- an extra isolated backward
            # pass every step would blow the time budget).
            curv_loss.backward()  # unweighted; .grad currently holds ONLY d(curv)/dtheta
            curv_grad_norm = total_grad_norm(model)
            if lambda_chosen == 0.0:
                lambda_chosen = flow_grad_norm / max(curv_grad_norm, 1e-12)
                print(f"[chain] AUTO-CALIBRATED lambda={lambda_chosen:.6e} "
                      f"(flow_grad_norm={flow_grad_norm:.6e} / curv_grad_norm={curv_grad_norm:.6e})",
                      flush=True)
            else:
                print(f"[chain] measured flow_grad_norm={flow_grad_norm:.6e} "
                      f"curv_grad_norm={curv_grad_norm:.6e} at raw (unweighted) scale; "
                      f"using CLI-specified lambda={lambda_chosen:.6e} "
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
            (lambda_chosen * curv_loss).backward()  # accumulates lambda*d(curv)/dtheta onto existing .grad

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

        vr_mean, vr_std = variety_ratios_detector_space(
            dp_final.detach(), human_polar, pad_mask, spd_s, dh_s)

        temp = read_last_temp(Path(args.watchdog_log))
        lr_now = scheduler.get_last_lr()[0]
        gn_msg = ""
        if curv_grad_norm is not None:
            gn_msg = (f" | CALIBRATION flow_grad_norm={flow_grad_norm:.4e} "
                      f"curv_grad_norm={curv_grad_norm:.4e} lambda={lambda_chosen:.4e}")
        print(f"[chain] step={global_step:5d} flow_loss={flow_loss.item():.4f} "
              f"curv_loss={curv_loss.item():.4f} vr_mean(det)={vr_mean:.4f} "
              f"vr_std(det)={vr_std:.4f} grad_norm={grad_norm_step:.4e} lr={lr_now:.2e} "
              f"peak_vram={peak_vram_mb:.0f}MB step_s={step_elapsed:.2f} "
              f"burst_min={elapsed_burst/60:.2f} temp={temp}{gn_msg}", flush=True)

        if global_step % args.ckpt_every == 0:
            save_checkpoint(chain_latest_path, model, optimizer, config, data_scale,
                             data_std, global_step, args)
            print(f"[chain] saved {chain_latest_path} at step {global_step}", flush=True)

    # final save, both the rolling latest and the chain checkpoint itself
    save_checkpoint(chain_latest_path, model, optimizer, config, data_scale,
                     data_std, global_step, args)
    save_checkpoint(chain_path, model, optimizer, config, data_scale,
                     data_std, global_step, args)
    print(f"[chain] final save: {chain_path} and {chain_latest_path} at step {global_step}", flush=True)

    md5_after = md5_file(src_path)
    print(f"[chain] source MD5 after: {md5_after}", flush=True)
    if md5_after != md5_before:
        print("[chain] *** WARNING: source checkpoint MD5 CHANGED -- should never happen ***",
              flush=True)
    else:
        print("[chain] source checkpoint MD5 unchanged, confirmed untouched.", flush=True)

    print(f"[chain] DONE. steps_done_this_run={steps_done_this_run} "
          f"global_step={global_step} wall_clock_min={(time.perf_counter()-t_burst_start)/60:.1f}",
          flush=True)


if __name__ == "__main__":
    main()
