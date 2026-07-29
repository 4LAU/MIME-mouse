"""Burst 3b of DIFFUSION_PILOT_V2.md Phase 1: bug-fix rerun of burst 3.

Burst 3 (training/train_candi_chain3.py, training/candi_polar_flow_chain3.pt)
widened the retargeted loss from curvature-only to an 11-feature moment loss
computed on the SAME rotate+perp differentiable cartesian decode chain2
built. It trained cleanly but scoring BACKFIRED: natural AUC went to
0.92-0.94 (baseline 0.757, burst 2's curvature-only number 0.795). Root
cause, confirmed by inspecting training/human_feature_targets.json: two of
the eleven covered features, mean_acceleration and mean_jerk, have a
near-zero human across-batch std in detector space (0.0166 and 0.0114,
versus 0.23-3.9 for every other covered feature). Burst 3's loss z-scored
every feature's mean-term AND std-term by dividing by that SAME feature's
own human target std:

    term_f = ((batch_mean_f - target_mean_f) / target_std_f) ** 2
           + ((batch_std_f  - target_std_f)  / target_std_f) ** 2

For mean_acceleration and mean_jerk this divides by ~0.011-0.017, an
effective per-term weight (1 / target_std^2) of ~3650x and ~7650x --
roughly 2000-4500x the median covered-feature weight (~1.7). Those two
terms dominated the sum, and the cheapest way for the model to minimize
them was to collapse per-path mean_acceleration and mean_jerk to nearly
identical values across the whole batch: the scored variety ratio for both
features fell to 0.01-0.02 (from the 20-45x OVERdispersed starting point),
a worse detector tell than the curvature deficit burst 2 was fixing in the
first place.

THE FIX (this file, minimal diff from chain3):

  1. WEIGHT CAPPING (spec item 1). Every covered feature's effective loss
     weight is 1 / max(target_std_f, eps) ** 2 (the natural consequence of
     the z-score-by-human-std normalization). compute_effective_weights()
     computes this RAW weight for every feature, takes the MEDIAN across
     all 11, and clips every feature's weight into
     [median / sqrt(10), median * sqrt(10)] -- a single global, principled
     rule (not a hand-picked per-feature override) that guarantees the
     largest and smallest effective weights differ by at most 10x, and each
     is within ~3.16x of the median. The clipped weight is converted back to
     an effective denominator (1/sqrt(weight)) and used in place of the raw
     target_std_f everywhere the loss and the honest-read logging divide by
     "the feature's own std". Printed in full (raw std, raw weight, capped
     weight, capped denom) at startup and in the CPU smoke test, with an
     assertion that max/min <= 10 (+ tiny float slack).

  2. mean_acceleration / mean_jerk special case (spec item 2). These are
     covered by the SAME global rule in (1), not a bespoke exception: their
     weight is capped down from ~3650x/7650x raw to at most
     median * sqrt(10) (~5.3), identical ceiling to any other feature that
     happened to land above the cap (e.g. path_efficiency, which was
     ~11.4x median raw and gets the same treatment). We chose "floor as in
     (1)" over dropping their std-term outright: their MEAN is also
     near-zero (~-0.0008 / -0.0006), so the mean-term needs the same
     denominator floor as the std-term, and treating both features
     uniformly with the other nine avoids introducing another hand-picked
     special case into a loss that already backfired once from ad hoc
     per-feature reasoning.

     On top of the weight cap, an EXPLICIT anti-collapse guard applies to
     ALL 11 covered features (not just these two): whenever a feature's
     HONEST variety ratio (batch_std / the REAL, uncapped human target
     std -- never the capped denom, so the guard cannot be defeated by the
     same capping that tames the gradient) drops below
     --anti-collapse-floor (default 0.3), a one-sided hinge penalty
     guard_mult * (relu(floor * target_std_f - batch_std_f) / denom_eff_f) ** 2
     is added to the loss for that feature. It is exactly zero whenever the
     batch is not collapsing, so it never perturbs normal training, and
     activates as an emergency brake if the batch ever undershoots 30% of
     human variety on any covered feature -- structurally the same failure
     burst 3 hit, now guarded against explicitly rather than hoped away by
     the weight cap alone.

  3. Everything else (covered feature set, detector-space transforms,
     variable-length seq-len-bucketed cond sampling, differentiable
     rotate+perp cartesian decode, features.py-matched formulas, named
     checkpoints every --named-ckpt-every steps as chain3b_step{N}.pt never
     overwritten) is an unchanged carry-forward from burst 3 / burst 2 /
     burst 1.

LOGGING FIX (the blind spot that let burst 3's collapse go unnoticed until
post-hoc N=2000 scoring): burst 3's only per-feature honest read was the
ROLLING BUFFER (RollingVariety, unchanged and kept here), which mixes
per-path samples pushed at many different points along the optimization
trajectory -- early (healthy) samples linger in the buffer for
ROLL_BUFFER_MAXLEN=512 entries and smear over a live collapse in progress.
IN ADDITION to that rolling log, every --probe-every steps (default: same
cadence as --named-ckpt-every, 50) this file runs a FROZEN-WEIGHT probe:
freezes the current weights, samples --probe-batch (default 256) paths
under torch.no_grad(), and computes the per-feature detector-space variety
ratio (batch_std / real target_std) on that SINGLE snapshot only. This is
the number that actually predicts N=2000 scoring (no smearing across
weight versions), printed as its own table, saved to
training/chain3b_probe_history.json, and used at the end of the burst to
rank named checkpoints (closest to ratio 1.0 across all 11 covered
features, in log space) for the top-3 pick the scoring sweep uses.

FRESH START: copies training/candi_polar_flow_best.pt to
training/candi_polar_flow_chain3b.pt (never chain3.pt, which is the
collapsed burst-3 lineage -- not resumed). Never writes to best.pt.

CPU smoke test: `python -m training.train_candi_chain3b --smoke`
(fabricated tensors, no file I/O, no GPU; verifies: effective weights are
printed and max/min <= 10; every one of the 11 feature terms individually
contributes a nonzero finite gradient; the anti-collapse guard fires with a
nonzero finite gradient when a feature's batch_std is fabricated near zero,
and does NOT fire when batch_std is fabricated at a healthy value; frozen
probe ratio computation is finite and NaN-free on a fabricated batch).
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
from training.train_candi_chain3 import decode_and_features, to_detector_space

TRAIN_DIR = Path("training")
DATA_DIR = Path("data")
SRC_CKPT_NAME = "candi_polar_flow_best.pt"
CHAIN_CKPT_NAME = "candi_polar_flow_chain3b.pt"
EXPECTED_SRC_MD5 = "91326a29750789f3167055324ef377c5"
WATCHDOG_LOG = Path("research/gpu_temp_phase1.log")
TARGETS_PATH = TRAIN_DIR / "human_feature_targets.json"
PROBE_HISTORY_PATH = TRAIN_DIR / "chain3b_probe_history.json"

VRAM_FRACTION = 0.80
N_SAMPLE_STEPS = 200
GUIDE = 0.15
PERP_SCALE = 0.85
HZ = 125.0
ROLL_BUFFER_MAXLEN = 512
ROLL_BUFFER_MIN = 256
NAMED_CKPT_EVERY = 50
WEIGHT_CAP_RATIO = 10.0
ANTI_COLLAPSE_FLOOR = 0.3
GUARD_MULT = 5.0
PROBE_BATCH = 256
PROBE_POOL_MULT = 6
_EPS = 1e-6

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


def load_targets(path: Path) -> dict:
    targets = json.loads(path.read_text())
    for name in COVERED_FEATURES:
        for stat in ("mean", "std"):
            key = f"target_{stat}_{name}_det"
            assert key in targets, f"missing {key} in {path}"
    return targets


# --------------------------------------------------------------------------
# THE FIX, item 1: median-based weight capping, principled and auditable.
# --------------------------------------------------------------------------

def compute_effective_weights(targets: dict, cap_ratio: float = WEIGHT_CAP_RATIO,
                               eps: float = _EPS, print_table: bool = True,
                               tag: str = "chain3b") -> dict:
    """Returns dict(name -> dict(raw_std, raw_weight, eff_weight, eff_denom,
    capped)). raw_weight = 1 / raw_std ** 2 is the weight burst 3's loss
    implicitly applied (z-score by the feature's own human std, both terms
    divided by the same std). eff_weight clips raw_weight into
    [median / sqrt(cap_ratio), median * sqrt(cap_ratio)], so
    max(eff_weight)/min(eff_weight) <= cap_ratio and every eff_weight is
    within sqrt(cap_ratio) (~3.16x for the default cap_ratio=10) of the
    median -- satisfies both the "10x of median" and "10x largest/smallest"
    framings of the requirement simultaneously."""
    raw_std = {name: max(targets[f"target_std_{name}_det"], eps) for name in COVERED_FEATURES}
    raw_weight = {name: 1.0 / (s * s) for name, s in raw_std.items()}
    median_w = float(np.median(list(raw_weight.values())))
    half_width = math.sqrt(cap_ratio)
    lo, hi = median_w / half_width, median_w * half_width

    out = {}
    for name in COVERED_FEATURES:
        w_raw = raw_weight[name]
        w_eff = float(np.clip(w_raw, lo, hi))
        out[name] = {
            "raw_std": raw_std[name],
            "raw_weight": w_raw,
            "eff_weight": w_eff,
            "eff_denom": 1.0 / math.sqrt(w_eff),
            "capped": abs(w_eff - w_raw) > 1e-9 * max(w_raw, 1.0),
        }

    max_w = max(v["eff_weight"] for v in out.values())
    min_w = min(v["eff_weight"] for v in out.values())
    ratio = max_w / min_w
    assert ratio <= cap_ratio + 1e-6, (
        f"weight cap failed: max/min effective weight = {ratio:.4f} > {cap_ratio}")

    if print_table:
        print(f"[{tag}] === effective per-feature weight audit "
              f"(median_raw_weight={median_w:.4f}, cap window "
              f"[{lo:.4f}, {hi:.4f}], cap_ratio={cap_ratio}) ===", flush=True)
        for name in COVERED_FEATURES:
            v = out[name]
            flag = " CAPPED" if v["capped"] else ""
            print(f"    {name:24s} raw_std={v['raw_std']:.6f} "
                  f"raw_weight={v['raw_weight']:10.4f} "
                  f"eff_weight={v['eff_weight']:10.4f} "
                  f"eff_denom={v['eff_denom']:.6f}{flag}", flush=True)
        print(f"[{tag}] max/min effective weight = {ratio:.4f} "
              f"(<= {cap_ratio} required); max/median = {max_w/median_w:.4f}; "
              f"min/median = {min_w/median_w:.4f}", flush=True)
    return out


# --------------------------------------------------------------------------
# THE FIX, item 2: loss with capped denominators + anti-collapse guard.
# --------------------------------------------------------------------------

def multi_feature_target_loss(raw_stats, path_valid, targets, eff_weights,
                               anti_collapse_floor=ANTI_COLLAPSE_FLOOR,
                               guard_mult=GUARD_MULT, eps=_EPS):
    """Same structure as chain3's multi_feature_target_loss (z-scored
    across-batch mean/std vs fixed human targets), with two changes:

    (1) normalizes by eff_weights[name]["eff_denom"] (the median-capped
        denominator) instead of the raw target_std_f -- this is THE FIX for
        the amplification that collapsed mean_acceleration/mean_jerk.
    (2) adds a one-sided anti-collapse hinge, guard_mult *
        (relu(anti_collapse_floor * REAL_target_std - batch_std) /
        eff_denom) ** 2, for any feature whose HONEST variety ratio (batch
        std over the REAL, uncapped target std) drops below
        anti_collapse_floor. Zero whenever ratio >= floor; a real, always-on
        safety net against a repeat of burst 3's collapse, independent of
        whether the weight cap alone would have been enough.

    Returns (total_loss, stats) where stats[name] carries batch_mean,
    batch_std, variety_ratio (honest, vs the REAL target std -- never the
    capped denom, so this number is never inflated by the cap), and
    guard_fired (bool)."""
    if path_valid.sum() < 2:
        z = raw_stats["curvature_mean"].sum() * 0.0
        return z, {name: {"batch_mean": float("nan"), "batch_std": float("nan"),
                           "variety_ratio": float("nan"), "guard_fired": False}
                    for name in COVERED_FEATURES}

    det_stats = to_detector_space(raw_stats)
    total = None
    stats = {}
    for name in COVERED_FEATURES:
        v = det_stats[name][path_valid]
        b_mean = v.mean()
        b_std = v.std(unbiased=False).clamp(min=eps)
        t_mean = targets[f"target_mean_{name}_det"]
        t_std_real = max(targets[f"target_std_{name}_det"], eps)  # honest, uncapped
        denom_eff = eff_weights[name]["eff_denom"]

        term = ((b_mean - t_mean) / denom_eff) ** 2 + ((b_std - t_std_real) / denom_eff) ** 2

        ratio_honest = float(b_std.detach()) / t_std_real
        guard_fired = ratio_honest < anti_collapse_floor
        if guard_fired:
            deficit = torch.clamp(anti_collapse_floor * t_std_real - b_std, min=0.0)
            term = term + guard_mult * (deficit / denom_eff) ** 2

        total = term if total is None else total + term
        stats[name] = {
            "batch_mean": float(b_mean.detach()),
            "batch_std": float(b_std.detach()),
            "variety_ratio": ratio_honest,
            "guard_fired": bool(guard_fired),
        }
    return total, stats


# --------------------------------------------------------------------------
# Rolling buffer for honest logging (unchanged carry-forward from chain3)
# --------------------------------------------------------------------------

class RollingVariety:
    """Rolling buffer (>= ROLL_BUFFER_MIN entries) of recent per-path
    detector-space stats for every covered feature, used ONLY for logging --
    gradients always use the current step's batch. Kept unchanged from
    chain3; see the module docstring's LOGGING FIX section for why this
    alone is not enough and the frozen probe below is the honest read."""

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
# LOGGING FIX: frozen-weight variety probe (the honest, non-smeared read)
# --------------------------------------------------------------------------

def frozen_variety_probe(model, distances, duration_model, targets, max_seq_len,
                          spd_s, dh_s, args, rng, device,
                          batch_size=PROBE_BATCH, pool_mult=PROBE_POOL_MULT):
    """Freezes the current weights (no_grad), samples ~batch_size paths in
    ONE shot via the same seq-len-bucketed cond recipe training uses, and
    computes the per-feature detector-space variety ratio (batch_std / real
    target_std) on that single snapshot. Unlike RollingVariety this never
    mixes samples from different points in training -- it is the number
    that actually predicts N=2000 scoring, per the burst-3 postmortem."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        cond_np, angle_np, total_dist_np, pad_mask_np, bucket_T = sample_cond_batch(
            distances, duration_model, batch_size, pool_mult, max_seq_len, rng)
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
        det_stats = to_detector_space(raw_stats)
        valid_np = path_valid.cpu().numpy()
        n_valid = int(valid_np.sum())
        ratios = {}
        for name in COVERED_FEATURES:
            vals = det_stats[name].cpu().numpy()[valid_np]
            t_std = max(targets[f"target_std_{name}_det"], _EPS)
            ratios[name] = float(np.std(vals)) / t_std if len(vals) >= 2 else float("nan")
    if was_training:
        model.train()
    return ratios, n_valid, bucket_T


def print_probe_table(tag, step, ratios, n_valid, bucket_T):
    print(f"[{tag}] === FROZEN-WEIGHT VARIETY PROBE step={step} "
          f"n_valid={n_valid} bucket_T={bucket_T} ===", flush=True)
    for name in COVERED_FEATURES:
        r = ratios[name]
        flag = ""
        if name in ("mean_acceleration", "mean_jerk"):
            flag = " <-- collapse-prone feature (burst 3 root cause)"
        elif r == r and r < ANTI_COLLAPSE_FLOOR:  # r == r excludes NaN
            flag = " <-- BELOW anti-collapse floor"
        print(f"    {name:24s} ratio={r:.4f}{flag}", flush=True)


def rank_checkpoints_by_probe(probe_history: dict):
    """Ranks named-checkpoint steps by how close ALL 11 covered-feature
    ratios are to 1.0 in log space (sum of squared log-ratio); lower is
    better. Skips NaN ratios. Returns a list of (score, step) sorted
    ascending."""
    scored = []
    for step, ratios in probe_history.items():
        vals = [r for r in ratios.values() if r == r and r > 0]  # finite, positive
        if len(vals) < len(COVERED_FEATURES):
            continue
        score = sum(math.log(v) ** 2 for v in vals)
        scored.append((score, int(step)))
    scored.sort()
    return scored


# --------------------------------------------------------------------------
# CPU smoke test (fabricated tensors, no file I/O, no GPU)
# --------------------------------------------------------------------------

def smoke_test():
    print("[smoke3b] CPU smoke test: K=4, batch=4, 3 steps, fabricated cond/pad_mask", flush=True)
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
        # fabricated targets matching the real file's SHAPE OF THE PROBLEM:
        # two features with a much tighter human std than the rest, so the
        # weight-cap audit below has something real to cap.
        targets = {}
        for name in COVERED_FEATURES:
            targets[f"target_mean_{name}_det"] = 1.0
            targets[f"target_std_{name}_det"] = 1.0
        targets["target_mean_mean_acceleration_det"] = -0.001
        targets["target_std_mean_acceleration_det"] = 0.017
        targets["target_mean_mean_jerk_det"] = -0.001
        targets["target_std_mean_jerk_det"] = 0.011

    eff_weights = compute_effective_weights(targets, tag="smoke3b")
    max_w = max(v["eff_weight"] for v in eff_weights.values())
    min_w = min(v["eff_weight"] for v in eff_weights.values())
    assert max_w / min_w <= WEIGHT_CAP_RATIO + 1e-6, "weight cap assertion failed in smoke test"
    print(f"[smoke3b] weight cap OK: max/min = {max_w/min_w:.4f} <= {WEIGHT_CAP_RATIO}", flush=True)

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
        curv_loss, curv_stats = multi_feature_target_loss(raw_stats, path_valid, targets, eff_weights)
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
        print(f"[smoke3b] step={step} flow_loss={flow_loss.item():.4f} "
              f"curv_loss={curv_loss.item():.4f} grad_norm={gn:.4e} "
              f"path_valid={int(path_valid.sum())}/{batch}", flush=True)
        for name in COVERED_FEATURES:
            s = curv_stats[name]
            print(f"    {name:24s} batch_mean={s['batch_mean']:.4f} "
                  f"batch_std={s['batch_std']:.4f} variety_ratio={s['variety_ratio']:.4f} "
                  f"guard_fired={s['guard_fired']}", flush=True)

    vr = roller.read()
    print(f"[smoke3b] rolling buffer reads:", flush=True)
    for name in COVERED_FEATURES:
        ratio, n = vr[name]
        print(f"    {name:24s} n={n} variety_ratio={ratio:.4f}", flush=True)

    # --- per-feature isolated gradient check: each of the 11 terms must
    # individually contribute a nonzero, finite gradient. ---
    print("[smoke3b] per-feature isolated gradient check...", flush=True)
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
        denom_eff = eff_weights[name]["eff_denom"]
        term = ((v.mean() - t_mean) / denom_eff) ** 2 + \
               ((v.std(unbiased=False).clamp(min=_EPS) - max(targets[f"target_std_{name}_det"], _EPS)) / denom_eff) ** 2
        term.backward()
        gn2 = total_grad_norm(model2)
        assert math.isfinite(float(term.item())), f"{name}: loss term not finite"
        assert math.isfinite(gn2), f"{name}: grad norm not finite"
        assert gn2 > 0.0, f"{name}: grad norm is zero -- no gradient reaches the model"
        print(f"    {name:24s} term={term.item():.6f} grad_norm={gn2:.6e} OK", flush=True)

    # --- anti-collapse guard behavior check ---
    # multi_feature_target_loss expects RAW (pre-transform) values -- it
    # calls to_detector_space internally -- so these fabrications must be
    # built in raw space and inverted through DET_TRANSFORMS["mean_jerk"]
    # (linear, x/1e6, trivially invertible) to land on a chosen det-space
    # mean/std, not fed as already-transformed det values.
    print("[smoke3b] anti-collapse guard check (fabricated batch_std)...", flush=True)
    fab_valid = torch.ones(batch, dtype=torch.bool)
    name = "mean_jerk"
    t_mean_real = targets[f"target_mean_{name}_det"]
    t_std_real = max(targets[f"target_std_{name}_det"], _EPS)
    raw_mean = t_mean_real * 1e6   # inverse of x/1e6
    raw_std_unit = t_std_real * 1e6

    # healthy case: det-space batch_std ~ 0.43x target -> guard should NOT fire
    healthy_vals = torch.full((batch,), raw_mean)
    healthy_vals[0] += raw_std_unit  # inject nonzero spread so .std() > 0
    fab_stats_healthy = {n2: (healthy_vals.clone() if n2 == name else torch.ones(batch))
                         for n2 in COVERED_FEATURES}
    _, stats_healthy = multi_feature_target_loss(fab_stats_healthy, fab_valid, targets, eff_weights)
    ratio_h = stats_healthy[name]["variety_ratio"]
    assert not stats_healthy[name]["guard_fired"], f"guard fired on a healthy batch_std (ratio={ratio_h:.4f})"
    print(f"    healthy case: variety_ratio={ratio_h:.4f} "
          f"guard_fired={stats_healthy[name]['guard_fired']} (expected False) OK", flush=True)

    # collapsed case: det-space batch_std ~0.01x target (burst 3's actual
    # observed collapse level), not EXACTLY zero -- .std()'s gradient has a
    # genuine sqrt'(0)=infinity singularity at exact equality (a property of
    # torch.std() itself, not this fix), so a near-collapse spread is both
    # more realistic and keeps the gradient check meaningful.
    collapsed_vals = torch.full((batch,), raw_mean)
    collapsed_vals[0] += 0.01 * raw_std_unit
    fab_stats_collapsed = {n2: (collapsed_vals.clone() if n2 == name else torch.ones(batch))
                           for n2 in COVERED_FEATURES}
    fab_stats_collapsed[name].requires_grad_(True)
    loss_collapsed, stats_collapsed = multi_feature_target_loss(
        fab_stats_collapsed, fab_valid, targets, eff_weights)
    ratio_c = stats_collapsed[name]["variety_ratio"]
    assert stats_collapsed[name]["guard_fired"], f"guard did NOT fire on a collapsed batch_std (ratio={ratio_c:.4f})"
    loss_collapsed.backward()
    assert fab_stats_collapsed[name].grad is not None, "guard produced no gradient"
    gnorm = float(fab_stats_collapsed[name].grad.detach().norm().item())
    assert math.isfinite(gnorm) and gnorm > 0.0, "guard gradient not finite/nonzero"
    print(f"    collapsed case: variety_ratio={ratio_c:.4f} "
          f"guard_fired={stats_collapsed[name]['guard_fired']} "
          f"(expected True), guard grad_norm={gnorm:.6e} OK", flush=True)

    # --- frozen probe ratio computation smoke (no file I/O: reuse the last
    # fabricated dp_final/pad_mask instead of sample_cond_batch) ---
    print("[smoke3b] frozen probe ratio computation smoke...", flush=True)
    with torch.no_grad():
        raw_stats3, path_valid3 = decode_and_features(
            dp_final2, pad_mask, tgt_angle, total_dist, spd_s, dh_s)
        det_stats3 = to_detector_space(raw_stats3)
        valid_np3 = path_valid3.cpu().numpy()
        for fname in COVERED_FEATURES:
            vals = det_stats3[fname].cpu().numpy()[valid_np3]
            t_std = max(targets[f"target_std_{fname}_det"], _EPS)
            ratio = float(np.std(vals)) / t_std if len(vals) >= 2 else float("nan")
            assert ratio == ratio, f"{fname}: frozen-probe ratio is NaN unexpectedly"
    print("[smoke3b] frozen probe ratios finite OK", flush=True)

    tmp_path = Path("training") / "_smoke_chain3b_test.pt"
    save_checkpoint(tmp_path, model, optimizer, config, data_scale, data_scale, 3,
                     argparse.Namespace(smoke=True))
    reloaded = torch.load(tmp_path, map_location="cpu", weights_only=False)
    model3 = CANDIModel(**reloaded["config"])
    model3.load_state_dict(reloaded["model_state_dict"])
    tmp_path.unlink()
    print("[smoke3b] checkpoint save/load roundtrip OK", flush=True)
    print("[smoke3b] SMOKE TEST PASSED", flush=True)


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
                     help="Save a NEVER-overwritten training/chain3b_step{N}.pt every N steps")
    ap.add_argument("--weight-cap-ratio", type=float, default=WEIGHT_CAP_RATIO,
                     help="Max allowed ratio between the largest and smallest "
                          "effective per-feature loss weight (THE FIX, item 1)")
    ap.add_argument("--anti-collapse-floor", type=float, default=ANTI_COLLAPSE_FLOOR,
                     help="Honest variety ratio below which the anti-collapse "
                          "guard activates for a covered feature (item 2)")
    ap.add_argument("--guard-mult", type=float, default=GUARD_MULT,
                     help="Weight multiplier applied to the anti-collapse hinge term")
    ap.add_argument("--probe-every", type=int, default=None,
                     help="Run the frozen-weight variety probe every N steps "
                          "(default: same as --named-ckpt-every)")
    ap.add_argument("--probe-batch", type=int, default=PROBE_BATCH)
    ap.add_argument("--probe-pool-mult", type=int, default=PROBE_POOL_MULT)
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

    if args.probe_every is None:
        args.probe_every = args.named_ckpt_every

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION, device=0)
        total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        print(f"[chain3b] VRAM cap: {VRAM_FRACTION:.2f} x {total_mb:.0f}MB = "
              f"{VRAM_FRACTION * total_mb:.0f}MB (spill-to-shared-memory guard)", flush=True)

    data_dir = Path(args.data_dir)
    src_path = data_dir / args.src_ckpt
    chain_path = data_dir / args.chain_ckpt
    chain_latest_path = chain_path.with_stem(chain_path.stem + "_latest")

    md5_before = md5_file(src_path)
    print(f"[chain3b] source MD5 before: {md5_before} (expected {EXPECTED_SRC_MD5})", flush=True)
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
    print(f"[chain3b] loaded {load_path} device={device} params={n_params:,}", flush=True)

    targets = load_targets(Path(args.targets))
    print(f"[chain3b] human feature targets ({len(COVERED_FEATURES)} covered "
          f"features, uncovered: {UNCOVERED_FEATURES}):", flush=True)
    for name in COVERED_FEATURES:
        print(f"    {name:24s} mean={targets[f'target_mean_{name}_det']:.4f} "
              f"std={targets[f'target_std_{name}_det']:.4f}", flush=True)

    eff_weights = compute_effective_weights(targets, cap_ratio=args.weight_cap_ratio, tag="chain3b")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if "optimizer_state_dict" in ckpt and args.load_from:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("[chain3b] resumed optimizer state (AdamW momentum) from checkpoint", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[chain3b] WARNING: could not resume optimizer state: {exc}", flush=True)

    start_step = 0
    if args.load_from and not args.reset_schedule:
        start_step = int(ckpt.get("global_step", 0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_steps)
    for _ in range(start_step):
        scheduler.step()
    print(f"[chain3b] schedule start_step={start_step} reset_schedule={args.reset_schedule} "
          f"total_steps={args.total_steps}", flush=True)

    # --- data pools ---
    print("[chain3b] loading training data pools...", flush=True)
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
    print(f"[chain3b] cond pool: {len(distances):,} human_distances.npy distances "
          f"+ DurationModel (never restricted to a length subset)", flush=True)

    rng = np.random.default_rng(args.seed)
    spd_s, dh_s = float(data_scale[0]), float(data_scale[1])

    # --- rolling-buffer warm-up ---
    roller = RollingVariety(targets)
    n_warmup = args.warmup_batches
    if n_warmup is None:
        n_warmup = math.ceil(ROLL_BUFFER_MIN / args.sample_batch)
    print(f"[chain3b] warming up rolling buffer with {n_warmup} no-grad batches "
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
    print(f"[chain3b] === STEP-0 PRE-TRAINING ROLLING VARIETY READ "
          f"(warmup {time.perf_counter()-t_warm0:.1f}s) ===", flush=True)
    for name in COVERED_FEATURES:
        ratio, n = vr0[name]
        flag = " <-- expected ~0.55 for curvature; if >=0.8 the retarget failed, STOP" \
            if name == "curvature_mean" else ""
        print(f"    {name:24s} n={n} variety_ratio={ratio:.4f}{flag}", flush=True)

    # --- STEP-0 FROZEN-WEIGHT PROBE (the honest, non-smeared read; pilot bar) ---
    probe_rng = np.random.default_rng(args.seed + 1)
    probe_history: dict = {}
    ratios0, n_valid0, bucket_T0 = frozen_variety_probe(
        model, distances, duration_model, targets, max_seq_len, spd_s, dh_s,
        args, probe_rng, device, batch_size=args.probe_batch, pool_mult=args.probe_pool_mult)
    print_probe_table("chain3b", start_step, ratios0, n_valid0, bucket_T0)
    probe_history[start_step] = ratios0
    bad = [n for n in ("mean_acceleration", "mean_jerk")
           if ratios0[n] == ratios0[n] and ratios0[n] < 0.05]
    if bad:
        print(f"[chain3b] WARNING: step-0 frozen probe already shows near-zero "
              f"variety for {bad} -- check the source checkpoint before proceeding", flush=True)

    t_burst_start = time.perf_counter()
    max_seconds = args.max_minutes * 60.0
    global_step = start_step
    steps_done_this_run = 0
    bce = nn.BCEWithLogitsLoss(reduction="none")

    print(f"[chain3b] === starting training loop: k={args.k} n_steps={args.n_steps} "
          f"sample_batch={args.sample_batch} pool_mult={args.pool_mult} "
          f"batch_size={args.batch_size} lr={args.lr} "
          f"curv_weight={'auto' if args.curv_weight == 0.0 else args.curv_weight} "
          f"weight_cap_ratio={args.weight_cap_ratio} "
          f"anti_collapse_floor={args.anti_collapse_floor} guard_mult={args.guard_mult} "
          f"probe_every={args.probe_every} max_minutes={args.max_minutes} ===", flush=True)

    lambda_chosen = args.curv_weight
    while True:
        elapsed_burst = time.perf_counter() - t_burst_start
        if elapsed_burst >= max_seconds:
            print(f"[chain3b] hard 90-minute wall clock reached ({elapsed_burst/60:.1f} min), "
                  f"stopping cleanly", flush=True)
            break
        if args.max_steps is not None and steps_done_this_run >= args.max_steps:
            print(f"[chain3b] reached --max-steps={args.max_steps}, stopping", flush=True)
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

        # --- (b) widened, weight-capped feature-moment loss on a seq-len-bucketed batch ---
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
        curv_loss, curv_stats = multi_feature_target_loss(
            raw_stats, path_valid, targets, eff_weights,
            anti_collapse_floor=args.anti_collapse_floor, guard_mult=args.guard_mult)
        if not torch.isfinite(curv_loss):
            print(f"[chain3b] WARNING: non-finite curv_loss at step {global_step}, "
                  f"skipping this step's feature-moment term", flush=True)
            curv_loss = torch.zeros((), device=device)

        roller.push(raw_stats, path_valid)

        curv_grad_norm = None
        if global_step == start_step:
            curv_loss.backward()
            curv_grad_norm = total_grad_norm(model)
            if lambda_chosen == 0.0:
                lambda_chosen = flow_grad_norm / max(curv_grad_norm, 1e-12)
                print(f"[chain3b] AUTO-CALIBRATED lambda={lambda_chosen:.6e} "
                      f"(flow_grad_norm={flow_grad_norm:.6e} / curv_grad_norm={curv_grad_norm:.6e})",
                      flush=True)
            else:
                print(f"[chain3b] measured flow_grad_norm={flow_grad_norm:.6e} "
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
        guards_fired = [n for n in COVERED_FEATURES if curv_stats[n]["guard_fired"]]
        guard_msg = f" | GUARD_FIRED={guards_fired}" if guards_fired else ""
        cm_ratio, _ = vr_roll["curvature_mean"]
        cs_ratio, _ = vr_roll["curvature_std"]
        pe_ratio, _ = vr_roll["path_efficiency"]
        mj_ratio, n_buf = vr_roll["mean_jerk"]
        ma_ratio, _ = vr_roll["mean_acceleration"]
        print(f"[chain3b] step={global_step:5d} flow_loss={flow_loss.item():.4f} "
              f"curv_loss={curv_loss.item():.4f} bucket_T={bucket_T} "
              f"vr_roll(n={n_buf}): curv_mean={cm_ratio:.4f} curv_std={cs_ratio:.4f} "
              f"path_eff={pe_ratio:.4f} mean_jerk={mj_ratio:.4f} mean_accel={ma_ratio:.4f} "
              f"grad_norm={grad_norm_step:.4e} lr={lr_now:.2e} "
              f"peak_vram={peak_vram_mb:.0f}MB step_s={step_elapsed:.2f} "
              f"burst_min={elapsed_burst/60:.2f} temp={temp}{gn_msg}{guard_msg}", flush=True)
        if global_step % 10 == 0:
            print(f"[chain3b] step={global_step:5d} FULL rolling variety table:", flush=True)
            for name in COVERED_FEATURES:
                ratio, n = vr_roll[name]
                print(f"    {name:24s} n={n} variety_ratio={ratio:.4f}", flush=True)

        if global_step % args.ckpt_every == 0:
            save_checkpoint(chain_latest_path, model, optimizer, config, data_scale,
                             data_std, global_step, args)
            print(f"[chain3b] saved {chain_latest_path} at step {global_step}", flush=True)

        if global_step % args.named_ckpt_every == 0:
            named_path = data_dir / f"chain3b_step{global_step}.pt"
            save_checkpoint(named_path, model, optimizer, config, data_scale,
                             data_std, global_step, args)
            print(f"[chain3b] saved NAMED checkpoint {named_path} (never overwritten)", flush=True)

        if global_step % args.probe_every == 0:
            ratios, n_valid, b_T = frozen_variety_probe(
                model, distances, duration_model, targets, max_seq_len, spd_s, dh_s,
                args, probe_rng, device, batch_size=args.probe_batch, pool_mult=args.probe_pool_mult)
            print_probe_table("chain3b", global_step, ratios, n_valid, b_T)
            probe_history[global_step] = ratios
            PROBE_HISTORY_PATH.write_text(json.dumps(probe_history, indent=2))

    save_checkpoint(chain_latest_path, model, optimizer, config, data_scale,
                     data_std, global_step, args)
    save_checkpoint(chain_path, model, optimizer, config, data_scale,
                     data_std, global_step, args)
    print(f"[chain3b] final save: {chain_path} and {chain_latest_path} at step {global_step}", flush=True)

    # final probe if the last step wasn't already a probe step, so the burst
    # always ends with a fresh honest read of the checkpoint being saved
    if global_step not in probe_history:
        ratios, n_valid, b_T = frozen_variety_probe(
            model, distances, duration_model, targets, max_seq_len, spd_s, dh_s,
            args, probe_rng, device, batch_size=args.probe_batch, pool_mult=args.probe_pool_mult)
        print_probe_table("chain3b", global_step, ratios, n_valid, b_T)
        probe_history[global_step] = ratios
        PROBE_HISTORY_PATH.write_text(json.dumps(probe_history, indent=2))

    ranked = rank_checkpoints_by_probe(probe_history)
    print(f"[chain3b] === checkpoint ranking by frozen-probe distance-from-1.0 "
          f"(lower = closer to human variety across all 11 covered features) ===", flush=True)
    for score, step in ranked[:5]:
        print(f"    step={step:5d} score={score:.4f}", flush=True)
    if ranked:
        top3 = [s for _, s in ranked[:3]]
        print(f"[chain3b] RECOMMENDED top-3 named checkpoints for N=2000 scoring: "
              f"{[f'chain3b_step{s}.pt' for s in top3]}", flush=True)

    md5_after = md5_file(src_path)
    print(f"[chain3b] source MD5 after: {md5_after}", flush=True)
    if md5_after != md5_before:
        print("[chain3b] *** WARNING: source checkpoint MD5 CHANGED -- should never happen ***",
              flush=True)
    else:
        print("[chain3b] source checkpoint MD5 unchanged, confirmed untouched.", flush=True)

    print(f"[chain3b] DONE. steps_done_this_run={steps_done_this_run} "
          f"global_step={global_step} wall_clock_min={(time.perf_counter()-t_burst_start)/60:.1f}",
          flush=True)


if __name__ == "__main__":
    main()
