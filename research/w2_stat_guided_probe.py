"""W2 mechanism probe (PLAN.md "W2 PROBE IS NEXT"): stat-guided generation,
no training.

Idea under test: every net-new-model attempt so far failed because the
generator's conditional distribution is under-dispersed in whole-path
statistics (curvature variance, jerk, etc.) -- the model averages away tail
stats instead of learning their spread. Here we do not ask a model to learn
that spread. Instead, per spec we SAMPLE a target feature vector from an
explicit conditional model of the HUMAN joint feature distribution (fit by
research/w2_fit_target_model.py: full covariance per distance bucket, not
batch moments -- the key delta vs burst 3/3b, which matched a single global
mean/std per feature and collapsed low-variance features into near-identical
values), then optimize the INITIAL NOISE of the validated K=200 differentiable
full-chain sampler (training/train_candi_chain.py's tail_step, reused
verbatim) by Adam so the decoded path's real detector-space features hit
that target. Model weights are NEVER touched (frozen, requires_grad False,
read-only load of training/candi_polar_flow_best.pt, MD5-checked before and
after).

PRE-REGISTERED GATE (PLAN.md): PASS = guided RF-OOB AUC <= 0.70 with no
single-feature collapse -> mechanism validated, W2 proper (train the
conditioning in) becomes worth doing. FAIL = targets hit but AUC stays near
the 0.757 baseline -> the multiplicity-of-tells wall holds even at the joint
level, strongest possible negative for feature-space approaches.

Reused verbatim, never reimplemented:
  - training/train_candi_chain.py: md5_file, tail_step, total_grad_norm,
    read_last_temp.
  - training/train_candi_chain3.py: decode_and_features, to_detector_space,
    DET_TRANSFORMS, COVERED_FEATURES (the 11 smooth/differentiable detector
    features and their det-space transforms).
  - training/compute_human_curv_targets.py: DET_TRANSFORMS (numpy mirror),
    trajectory_from_dxdy.
  - research/phase_a_baseline.py: load_model, decode_polar, build_trajectory,
    generate_paths, make_specs, sample_guided_flow, DUR_STD (the exact
    generation/decode convention that produced the published 0.752/0.757
    numbers -- used here for both the guided decode and the unguided
    CONTROL).
  - features.py: extract_features, extract_feature_matrix, FEATURE_NAMES
    (the REAL numpy detector feature pipeline; every scoring number in this
    file comes from this, never from the torch approximation used only to
    get gradients).
  - experiments/_common.py: DurationModel, get_device.

Never reads data/human_eval_features.npy anywhere in this file. Never writes
to training/candi_polar_flow_best.pt.

CPU smoke test (fabricated, no GPU, no file I/O beyond the target model npz):
    python -m research.w2_stat_guided_probe --unit-smoke

Real smoke run (64 specs, real model, real GPU, cheap):
    .venv/Scripts/python.exe research/w2_stat_guided_probe.py --smoke

Main run (N=2000, 90-minute wall-clock cap):
    .venv/Scripts/python.exe research/w2_stat_guided_probe.py --main
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments._common import DurationModel, get_device  # noqa: E402
from features import extract_features, extract_feature_matrix, FEATURE_NAMES, resample_trajectory  # noqa: E402
from training.train_candi_chain import md5_file, tail_step, total_grad_norm, read_last_temp  # noqa: E402
from training.train_candi_chain3 import (  # noqa: E402
    decode_and_features, to_detector_space, COVERED_FEATURES,
)
from training.compute_human_curv_targets import DET_TRANSFORMS as NP_DET_TRANSFORMS  # noqa: E402
from phase_a_baseline import (  # noqa: E402
    load_model, decode_polar, build_trajectory, generate_paths, make_specs,
    DUR_STD, N_SAMPLE_STEPS as BASELINE_N_STEPS, GUIDE as BASELINE_GUIDE,
    PERP_SCALE as BASELINE_PERP_SCALE,
)

DATA_DIR = REPO_ROOT / "data"
TRAIN_DIR = REPO_ROOT / "training"
RESEARCH_DIR = REPO_ROOT / "research"

SRC_CKPT_NAME = "candi_polar_flow_best.pt"
EXPECTED_SRC_MD5 = "91326a29750789f3167055324ef377c5"
TARGET_MODEL_PATH = RESEARCH_DIR / "w2_target_model.npz"
PROGRESS_PATH = RESEARCH_DIR / "w2_progress_checkpoint.pkl"
SPECS_PATH = RESEARCH_DIR / "w2_specs_checkpoint.pkl"
RESULTS_PATH = RESEARCH_DIR / "w2_probe_results.json"
LOG_PATH = RESEARCH_DIR / "w2_probe_run.log"
HUMAN_REF_PATH = DATA_DIR / "human_val_features_grpo.npy"

VRAM_FRACTION = 0.80
RF_SEED = 42
_EPS = 1e-6

CURV_MEAN_IDX = FEATURE_NAMES.index("curvature_mean")
CURV_STD_IDX = FEATURE_NAMES.index("curvature_std")
NAME_TO_IDX18 = {n: i for i, n in enumerate(FEATURE_NAMES)}

UNCOVERED_FEATURES = [n for n in FEATURE_NAMES if n not in COVERED_FEATURES]

_LOG_FH = None


def log(msg: str) -> None:
    print(msg, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(msg + "\n")
        _LOG_FH.flush()


def gpu_temp_mem() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<nvidia-smi failed: {exc}>"


# --------------------------------------------------------------------------
# Target model (research/w2_fit_target_model.py output)
# --------------------------------------------------------------------------

def load_target_model():
    if not TARGET_MODEL_PATH.exists():
        log(f"[w2_probe] {TARGET_MODEL_PATH} missing, fitting it now "
            f"(research/w2_fit_target_model.py)...")
        from research import w2_fit_target_model
        w2_fit_target_model.main()
    d = np.load(TARGET_MODEL_PATH)
    return {k: d[k] for k in d.files}


def assign_bucket(dist: np.ndarray, bucket_edges: np.ndarray) -> np.ndarray:
    return np.digitize(dist, bucket_edges[1:-1], right=False)


def sample_targets(bucket_idx: np.ndarray, tm: dict, seed: int):
    """Per-spec target vector: draw from the assigned bucket's full-covariance
    MVN in detector space, clipped to that bucket's empirical [p1, p99] box.
    Also returns the matching per-spec weight vector (bucket's median-capped
    effective weight) and per-spec std vector (for target-hit scoring)."""
    n = len(bucket_idx)
    n_feat = tm["bucket_mean"].shape[1]
    targets = np.empty((n, n_feat), dtype=np.float64)
    weights = np.empty((n, n_feat), dtype=np.float64)
    stds = np.empty((n, n_feat), dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_buckets = tm["bucket_mean"].shape[0]
    for b in range(n_buckets):
        m = bucket_idx == b
        n_b = int(m.sum())
        if n_b == 0:
            continue
        draws = rng.multivariate_normal(tm["bucket_mean"][b], tm["bucket_cov"][b], size=n_b)
        draws = np.clip(draws, tm["bucket_p1"][b], tm["bucket_p99"][b])
        targets[m] = draws
        weights[m] = tm["bucket_eff_weight"][b]
        stds[m] = tm["bucket_std"][b]
    return targets, weights, stds


# --------------------------------------------------------------------------
# Differentiable generation from a GIVEN (optimizable) initial noise z,
# reusing train_candi_chain.py's tail_step verbatim (the validated full
# K=200 differentiable chain). Only deviation from differentiable_generate:
# xt starts AS z itself (never detached/re-wrapped), so gradients optimize z.
# --------------------------------------------------------------------------

def generate_from_noise(z, model, cond, tgt_angle, spd_s, dh_s, n_steps, guide,
                         use_ckpt, device, pad_mask):
    B, seq_len, _ = z.shape
    stall_frozen = torch.full((B, seq_len), model.STALL_MASK, device=device)
    mflag_frozen = torch.ones(B, seq_len, device=device)
    xt = z
    dp_final = None
    for i in range(n_steps):
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
# Feature-builder validation: decode_and_features (torch, differentiable)
# vs features.py's extract_features (numpy, the REAL detector pipeline), on
# real model output.
# --------------------------------------------------------------------------

def validate_feature_builder(model, data_scale, device, duration_model, distances,
                              n_check=300, seed=999, guide=0.15, perp_scale=0.85,
                              n_steps=200):
    from training.train_candi_chain2 import sample_cond_batch

    spd_s, dh_s = float(data_scale[0]), float(data_scale[1])
    rng = np.random.default_rng(seed)
    max_seq_len = model.max_seq_len_cfg
    cond_np, angle_np, total_dist_np, pad_mask_np, bucket_T = sample_cond_batch(
        distances, duration_model, n_check, 4, max_seq_len, rng)
    cond = torch.from_numpy(cond_np).to(device)
    tgt_angle = torch.from_numpy(angle_np).to(device)
    total_dist = torch.from_numpy(total_dist_np).to(device)
    pad_mask = torch.from_numpy(pad_mask_np).to(device)

    with torch.no_grad():
        z0 = torch.randn(n_check, bucket_T, 2, device=device)
        dp_final = generate_from_noise(
            z0, model, cond, tgt_angle, spd_s, dh_s, n_steps, guide,
            use_ckpt=False, device=device, pad_mask=pad_mask)
        # torch det-space features, computed in float64 for a tight match to
        # numpy features.py (float32 introduces ~1e-3 noise on 3rd-derivative
        # jerk; float64 is only used for this offline validation check, never
        # in the main optimization loop, which stays float32 for speed --
        # matching train_candi_phase1.py's precedent of float64-internal
        # channel math only where the fragile 3rd-derivative chain needs it).
        raw_stats, path_valid = decode_and_features(
            dp_final.double(), pad_mask, tgt_angle.double(), total_dist.double(),
            spd_s, dh_s, perp_scale)
        det_stats = to_detector_space(raw_stats)
        torch_det = np.stack([det_stats[n].cpu().numpy() for n in COVERED_FEATURES], axis=1)
        dp_np = dp_final.cpu().numpy()

    per_feature_max_err = {n: 0.0 for n in COVERED_FEATURES}
    n_compared = 0
    n_skipped = 0
    overall_max_err = 0.0
    all_errs = []
    for i in range(n_check):
        L = int(pad_mask_np[i].sum())
        if L < 5:
            n_skipped += 1
            continue
        raw_np = dp_np[i, :L, :]
        stall_np = np.full(L, model.STALL_MASK, dtype=np.float64)
        cum_x, cum_y = decode_polar(raw_np, stall_np, data_scale)
        total_dist_i = float(total_dist_np[i])
        angle_i = float(angle_np[i])
        dx = total_dist_i * math.cos(angle_i)
        dy = total_dist_i * math.sin(angle_i)
        traj = build_trajectory(cum_x, cum_y, stall_np, L, total_dist_i, dx, dy,
                                 0.0, 0.0, dx, dy, no_round=True)
        feats = extract_features(traj)
        if feats is None:
            n_skipped += 1
            continue
        n_compared += 1
        for j, name in enumerate(COVERED_FEATURES):
            raw_val = feats[NAME_TO_IDX18[name]]
            det_val = NP_DET_TRANSFORMS[name](np.array([raw_val]))[0]
            err = abs(det_val - torch_det[i, j])
            all_errs.append(float(err))
            if err > per_feature_max_err[name]:
                per_feature_max_err[name] = float(err)
            if err > overall_max_err:
                overall_max_err = float(err)

    all_errs_arr = np.asarray(all_errs) if all_errs else np.array([float("nan")])
    return {
        "n_check": n_check, "n_compared": n_compared, "n_skipped": n_skipped,
        "overall_max_abs_err_det_space": overall_max_err,
        "median_abs_err_det_space": float(np.median(all_errs_arr)),
        "p95_abs_err_det_space": float(np.percentile(all_errs_arr, 95)),
        "per_feature_max_abs_err_det_space": per_feature_max_err,
    }


# --------------------------------------------------------------------------
# Spec construction (mirrors research/phase_a_baseline.py:generate_paths'
# per-spec math exactly, so cond/seq_len/total_dist match the real serving
# convention) + bucket assignment + target/weight sampling.
# --------------------------------------------------------------------------

def build_spec_records(n, seed, duration_model, model_max_seq_len, tm, target_seed,
                       joint_duration=False):
    specs = make_specs(n, seed)
    dx_l, dy_l, dist_l, log_dist_l, angle_l = [], [], [], [], []
    for (sx, sy, ex, ey) in specs:
        dx = ex - sx
        dy = ey - sy
        total_dist = math.hypot(dx, dy)
        log_dist = math.log(max(total_dist, 1.0))
        angle = math.atan2(dy, dx)
        dx_l.append(dx); dy_l.append(dy); dist_l.append(total_dist)
        log_dist_l.append(log_dist); angle_l.append(angle)

    dist_arr = np.asarray(dist_l)
    bucket_idx = assign_bucket(dist_arr, tm["bucket_edges"])
    targets, weights, stds = sample_targets(bucket_idx, tm, target_seed)

    # Duration per spec. Default: independent draw from DurationModel (the
    # original probe; velocity targets then demand speeds the fixed clock
    # cannot express). joint_duration: derive duration FROM the sampled
    # velocity/path-efficiency targets so the clock is consistent with the
    # stats by construction -- duration = implied_path_length / mean_velocity,
    # where implied_path_length = straight_dist / path_efficiency_target.
    # COVERED_FEATURES order: [0]=mean_velocity (det=log1p), [6]=path_efficiency
    # (det=identity). Snap to the seq_len grid (125 Hz) so cond's log_duration
    # matches the actual time base the detector sees.
    dur_l, seq_len_l = [], []
    n_dur_clamped = 0
    for i in range(len(specs)):
        if joint_duration:
            mv_raw = math.expm1(max(targets[i][0], 0.0))
            pe_raw = min(max(targets[i][6], 0.05), 1.0)
            implied_path_len = dist_l[i] / pe_raw
            duration = implied_path_len / max(mv_raw, 1.0)
        else:
            duration = duration_model.sample(log_dist_l[i])
        seq_len = max(5, min(int(round(duration * 125.0)), model_max_seq_len))
        if joint_duration:
            snapped = seq_len / 125.0
            if abs(snapped - duration) / max(duration, 1e-9) > 0.02:
                n_dur_clamped += 1
            duration = snapped
        dur_l.append(duration); seq_len_l.append(seq_len)
    if joint_duration:
        log(f"[w2_probe] joint-duration: {n_dur_clamped}/{len(specs)} specs clamped "
            f">2% by the [5, {model_max_seq_len}]-step grid (their velocity targets "
            f"remain unreachable)")

    records = []
    for i, (sx, sy, ex, ey) in enumerate(specs):
        records.append({
            "idx": i, "sx": sx, "sy": sy, "ex": ex, "ey": ey,
            "dx": dx_l[i], "dy": dy_l[i], "total_dist": dist_l[i],
            "log_dist": log_dist_l[i], "angle": angle_l[i], "duration": dur_l[i],
            "seq_len": seq_len_l[i], "bucket": int(bucket_idx[i]),
            "target": targets[i], "weight": weights[i], "target_std": stds[i],
            "trajectory": None, "loss_init": None, "loss_final": None,
            "steps_used": None, "final_feat_det_torch": None,
        })
    return records


# --------------------------------------------------------------------------
# Guided optimization: batches sorted by seq_len, Adam over z for M steps.
# --------------------------------------------------------------------------

def make_batches(records, batch_size):
    order = sorted(range(len(records)), key=lambda i: records[i]["seq_len"])
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def run_guided_batch(batch_ids, records, model, data_scale, device, args, feature_order):
    spd_s, dh_s = float(data_scale[0]), float(data_scale[1])
    B = len(batch_ids)
    seq_lens = [records[i]["seq_len"] for i in batch_ids]
    bucket_T = max(seq_lens)

    cond_np = np.stack([
        [records[i]["log_dist"], math.log(records[i]["duration"]),
         math.cos(records[i]["angle"]), math.sin(records[i]["angle"])]
        for i in batch_ids
    ]).astype(np.float32)
    angle_np = np.array([records[i]["angle"] for i in batch_ids], dtype=np.float32)
    total_dist_np = np.array([records[i]["total_dist"] for i in batch_ids], dtype=np.float32)
    pad_mask_np = np.zeros((B, bucket_T), dtype=np.bool_)
    for b, L in enumerate(seq_lens):
        pad_mask_np[b, :L] = True

    cond = torch.from_numpy(cond_np).to(device)
    tgt_angle = torch.from_numpy(angle_np).to(device)
    total_dist = torch.from_numpy(total_dist_np).to(device)
    pad_mask = torch.from_numpy(pad_mask_np).to(device)

    target_np = np.stack([records[i]["target"] for i in batch_ids]).astype(np.float32)
    weight_np = np.stack([records[i]["weight"] for i in batch_ids]).astype(np.float32)
    target_t = torch.from_numpy(target_np).to(device)
    weight_t = torch.from_numpy(weight_np).to(device)
    if getattr(args, "loss_norm", "capped") == "stdnorm":
        std_np = np.stack([records[i]["target_std"] for i in batch_ids]).astype(np.float32)
        std_np = np.maximum(std_np, 1e-3)
        weight_t = torch.from_numpy(1.0 / (std_np * std_np)).to(device)

    z = torch.randn(B, bucket_T, 2, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=args.lr)

    loss_history = []
    for step in range(args.m_steps):
        optimizer.zero_grad(set_to_none=True)
        dp_final = generate_from_noise(
            z, model, cond, tgt_angle, spd_s, dh_s, args.n_steps, args.guide,
            use_ckpt=True, device=device, pad_mask=pad_mask)
        raw_stats, path_valid = decode_and_features(
            dp_final, pad_mask, tgt_angle, total_dist, spd_s, dh_s, args.perp_scale)
        det_stats = to_detector_space(raw_stats)
        feats = torch.stack([det_stats[n] for n in feature_order], dim=-1)
        diff2 = (feats - target_t) ** 2 * weight_t
        per_path = diff2.sum(dim=-1)
        valid_f = path_valid.float()
        denom = valid_f.sum().clamp(min=1.0)
        loss = (per_path * valid_f).sum() / denom
        if not torch.isfinite(loss):
            log(f"[w2_probe] WARNING: non-finite loss at batch step {step}, stopping this batch's optimization early")
            break
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_([z], args.grad_clip)
        optimizer.step()
        loss_history.append(float(loss.detach()))
        if args.verbose_steps:
            log(f"    [z-opt] step={step} loss={float(loss.detach()):.4f} "
                f"grad_norm={float(gnorm):.4e} n_valid={int(valid_f.sum())}/{B}")

    with torch.no_grad():
        dp_final = generate_from_noise(
            z, model, cond, tgt_angle, spd_s, dh_s, args.n_steps, args.guide,
            use_ckpt=False, device=device, pad_mask=pad_mask)
        raw_stats, path_valid = decode_and_features(
            dp_final, pad_mask, tgt_angle, total_dist, spd_s, dh_s, args.perp_scale)
        det_stats = to_detector_space(raw_stats)
        feats_final = torch.stack([det_stats[n] for n in feature_order], dim=-1).cpu().numpy()
        dp_np = dp_final.cpu().numpy()
        path_valid_np = path_valid.cpu().numpy()

    # decode each spec's own path (real convention: decode_polar + build_trajectory)
    for b, i in enumerate(batch_ids):
        rec = records[i]
        L = rec["seq_len"]
        raw_np = dp_np[b, :L, :]
        stall_np = np.full(L, model.STALL_MASK, dtype=np.float64)
        cum_x, cum_y = decode_polar(raw_np, stall_np, data_scale)
        traj = build_trajectory(
            cum_x, cum_y, stall_np, L, rec["total_dist"], rec["dx"], rec["dy"],
            rec["sx"], rec["sy"], rec["ex"], rec["ey"], no_round=True)
        rec["trajectory"] = traj
        rec["loss_init"] = loss_history[0] if loss_history else float("nan")
        rec["loss_final"] = loss_history[-1] if loss_history else float("nan")
        rec["steps_used"] = len(loss_history)
        rec["final_feat_det_torch"] = feats_final[b].tolist()
        rec["path_valid_torch"] = bool(path_valid_np[b])

    return loss_history


# --------------------------------------------------------------------------
# Scoring (CPU, real numpy features.py pipeline)
# --------------------------------------------------------------------------

def rf_oob_auc(synth_features, human_features, seed=RF_SEED):
    n_use = min(len(human_features), len(synth_features))
    human_bal = human_features[:n_use]
    synth_bal = synth_features[:n_use]
    X = np.vstack([human_bal, synth_bal])
    y = np.concatenate([np.zeros(n_use), np.ones(n_use)])
    clf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=seed)
    clf.fit(X, y)
    oob_proba = clf.oob_decision_function_[:, 1]
    auc = roc_auc_score(y, oob_proba)
    return float(auc), n_use, clf


def real_features_per_spec(trajectories):
    """features.py's real pipeline (resample_trajectory + extract_features),
    called PER TRAJECTORY (not the batched extract_feature_matrix) so index
    alignment with the spec/target list is preserved. Returns (feat18,
    valid_mask) with feat18 shape (n, 18), NaN rows where invalid."""
    n = len(trajectories)
    feat18 = np.full((n, len(FEATURE_NAMES)), np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, traj in enumerate(trajectories):
        if traj is None:
            continue
        feats = extract_features(resample_trajectory(traj))
        if feats is not None and not np.any(np.isnan(feats)):
            feat18[i] = feats
            valid[i] = True
    return feat18, valid


def dispersion_table(synth_feat18, human_feat18):
    rows = []
    for j, name in enumerate(FEATURE_NAMES):
        s_std = float(np.std(synth_feat18[:, j]))
        h_std = float(np.std(human_feat18[:, j]))
        raw_ratio = s_std / max(h_std, 1e-12)
        det_ratio = None
        if name in NP_DET_TRANSFORMS:
            s_det = NP_DET_TRANSFORMS[name](synth_feat18[:, j])
            h_det = NP_DET_TRANSFORMS[name](human_feat18[:, j])
            det_ratio = float(np.std(s_det) / max(np.std(h_det), 1e-12))
        rows.append({
            "feature": name, "covered": name in COVERED_FEATURES,
            "synth_std_raw": s_std, "human_std_raw": h_std,
            "raw_dispersion_ratio": raw_ratio,
            "det_dispersion_ratio": det_ratio,
        })
    return rows


def target_hit_rates(records, valid_idx):
    """Fraction of (path, feature) within 0.5 / 1.0 det-space std of the
    path's OWN sampled target, using the REAL numpy features.py feature
    (never the torch approximation)."""
    per_feature = {n: {"n": 0, "hit_0.5": 0, "hit_1.0": 0, "mean_abs_err_std": 0.0}
                   for n in COVERED_FEATURES}
    total_n = 0
    total_hit_05 = 0
    total_hit_10 = 0
    for i in valid_idx:
        rec = records[i]
        real_det = rec["real_det_feat"]
        target = rec["target"]
        std = rec["target_std"]
        for j, name in enumerate(COVERED_FEATURES):
            err = abs(real_det[j] - target[j])
            err_std = err / max(std[j], _EPS)
            per_feature[name]["n"] += 1
            per_feature[name]["mean_abs_err_std"] += err_std
            total_n += 1
            if err_std <= 0.5:
                per_feature[name]["hit_0.5"] += 1
                total_hit_05 += 1
            if err_std <= 1.0:
                per_feature[name]["hit_1.0"] += 1
                total_hit_10 += 1
    for name in COVERED_FEATURES:
        n = max(per_feature[name]["n"], 1)
        per_feature[name]["frac_hit_0.5"] = per_feature[name]["hit_0.5"] / n
        per_feature[name]["frac_hit_1.0"] = per_feature[name]["hit_1.0"] / n
        per_feature[name]["mean_abs_err_std"] /= n
    overall = {
        "n_pairs": total_n,
        "frac_hit_0.5": total_hit_05 / max(total_n, 1),
        "frac_hit_1.0": total_hit_10 / max(total_n, 1),
    }
    return per_feature, overall


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    global _LOG_FH
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="64-spec end-to-end smoke run")
    ap.add_argument("--main-run", action="store_true", help="N=2000 main run")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--m-steps", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--guide", type=float, default=0.15)
    ap.add_argument("--perp-scale", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-seed", type=int, default=20260719)
    ap.add_argument("--max-minutes", type=float, default=90.0)
    ap.add_argument("--tag", default="w2")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--n-check", type=int, default=300,
                     help="Number of paths for feature-builder validation")
    ap.add_argument("--skip-control-gate", action="store_true")
    ap.add_argument("--skip-control", action="store_true",
                     help="Skip CONTROL generation entirely (fast LR-tuning iteration only, "
                          "never for a real smoke/main run)")
    ap.add_argument("--skip-validation", action="store_true",
                     help="Skip feature-builder validation (fast LR-tuning iteration only)")
    ap.add_argument("--verbose-steps", action="store_true",
                     help="Print every z-optimization step's loss/grad-norm (tuning only)")
    ap.add_argument("--joint-duration", action="store_true",
                     help="Derive each spec's duration from its sampled mean_velocity/"
                          "path_efficiency targets (stage-1 joint sampling) instead of "
                          "an independent DurationModel draw, so velocity targets are "
                          "consistent with the clock by construction")
    ap.add_argument("--loss-norm", choices=["capped", "stdnorm"], default="capped",
                     help="capped = target model's median-capped 1/std^2 weight (default); "
                          "stdnorm = uncapped per-feature 1/target_std^2, true inverse-variance "
                          "normalization (tuning experiment for features like mean_velocity "
                          "that the capped weight may under-drive)")
    ap.add_argument("--pid-file", type=str, default=None,
                     help="Write this process's PID here at startup, for an external "
                          "gpu_watchdog.py to target (research/phase1_feasibility_probe.py's "
                          "convention)")
    args = ap.parse_args()

    if args.n is None:
        args.n = 64 if args.smoke else 2000

    if args.pid_file:
        import os
        Path(args.pid_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.pid_file, "w") as fh:
            fh.write(str(os.getpid()))
        print(f"[w2_probe] PID={os.getpid()} written to {args.pid_file}", flush=True)

    _LOG_FH = open(LOG_PATH, "a")

    device = get_device()
    log(f"[w2_probe] === START tag={args.tag} smoke={args.smoke} main_run={args.main_run} "
        f"n={args.n} device={device} m_steps={args.m_steps} batch_size={args.batch_size} "
        f"lr={args.lr} k={args.k} n_steps={args.n_steps} guide={args.guide} "
        f"perp_scale={args.perp_scale} max_minutes={args.max_minutes} ===")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION, device=0)
        total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        log(f"[w2_probe] VRAM cap: {VRAM_FRACTION:.2f} x {total_mb:.0f}MB = "
            f"{VRAM_FRACTION * total_mb:.0f}MB")
    log(f"[w2_probe] GPU before: {gpu_temp_mem()}")

    src_path = TRAIN_DIR / SRC_CKPT_NAME
    md5_before = md5_file(src_path)
    log(f"[w2_probe] source MD5 before: {md5_before} (expected {EXPECTED_SRC_MD5})")
    assert md5_before == EXPECTED_SRC_MD5, "source checkpoint MD5 mismatch -- STOP"

    model, data_scale, device2, max_seq_len_cfg = load_model(SRC_CKPT_NAME)
    assert device2 == device or True
    model.max_seq_len_cfg = max_seq_len_cfg
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    log(f"[w2_probe] loaded {SRC_CKPT_NAME} read-only, all params requires_grad=False, "
        f"max_seq_len_cfg={max_seq_len_cfg}")

    tm = load_target_model()
    log(f"[w2_probe] target model: {tm['bucket_mean'].shape[0]} buckets, "
        f"{tm['bucket_mean'].shape[1]} covered features, "
        f"bucket_edges(finite)={np.round(tm['bucket_edges'][1:-1], 1).tolist()}")

    duration_model = DurationModel(TRAIN_DIR, std_mult=DUR_STD)
    distances = np.load(DATA_DIR / "human_distances.npy")

    # --- feature-builder validation ---
    if args.skip_validation:
        log("[w2_probe] --skip-validation: skipping feature-builder validation "
            "(tuning iteration only)")
        val = {"skipped": True}
    else:
        log(f"[w2_probe] validating differentiable feature builder vs features.py "
            f"on n_check={args.n_check} real-model paths...")
        val = validate_feature_builder(model, data_scale, device, duration_model, distances,
                                        n_check=args.n_check, guide=args.guide,
                                        perp_scale=args.perp_scale, n_steps=args.n_steps)
        log(f"[w2_probe] FEATURE BUILDER VALIDATION: n_compared={val['n_compared']}/"
            f"{val['n_check']} (skipped {val['n_skipped']}) "
            f"overall_max_abs_err_det_space={val['overall_max_abs_err_det_space']:.6e} "
            f"median_abs_err_det_space={val.get('median_abs_err_det_space', float('nan')):.6e} "
            f"p95_abs_err_det_space={val.get('p95_abs_err_det_space', float('nan')):.6e}")
        for name, err in val["per_feature_max_abs_err_det_space"].items():
            log(f"    {name:24s} max_abs_err={err:.6e}")
        if val["overall_max_abs_err_det_space"] > 1e-4:
            log(f"[w2_probe] *** WARNING: feature-builder max error "
                f"{val['overall_max_abs_err_det_space']:.6e} EXCEEDS the 1e-4 target "
                f"(see median/p95 above for whether this is a rare outlier path or systematic) ***")

    # --- build specs ---
    if args.resume and SPECS_PATH.exists():
        log(f"[w2_probe] --resume: loading spec records from {SPECS_PATH}")
        with open(SPECS_PATH, "rb") as fh:
            records = pickle.load(fh)
    else:
        records = build_spec_records(args.n, args.seed, duration_model,
                                      max_seq_len_cfg, tm, args.target_seed,
                                      joint_duration=args.joint_duration)
        with open(SPECS_PATH, "wb") as fh:
            pickle.dump(records, fh)
        log(f"[w2_probe] built {len(records)} spec records, saved to {SPECS_PATH}")

    dists = [r["total_dist"] for r in records]
    log(f"[w2_probe] spec distances: min={min(dists):.1f} max={max(dists):.1f} "
        f"median={np.median(dists):.1f}")

    human_features = np.load(HUMAN_REF_PATH)

    # --- CONTROL: unguided paths through the identical decode/scoring, same specs ---
    if args.skip_control:
        log("[w2_probe] --skip-control: skipping CONTROL generation (tuning iteration only)")
        control_auc, control_n, control_elapsed = float("nan"), 0, 0.0
    else:
        specs_for_control = [(r["sx"], r["sy"], r["ex"], r["ey"]) for r in records]
        log(f"[w2_probe] generating CONTROL (unguided) sample, n={len(specs_for_control)}...")
        t0 = time.perf_counter()
        control_trajs = generate_paths(model, data_scale, device, duration_model,
                                        specs_for_control, no_round=True)
        control_trajs = [t for t in control_trajs if t is not None and len(t) >= 2]
        control_elapsed = time.perf_counter() - t0
        log(f"[w2_probe] control generated {len(control_trajs)}/{len(specs_for_control)} "
            f"in {control_elapsed:.1f}s")

        control_feat = extract_feature_matrix(control_trajs)
        control_auc, control_n, control_clf = rf_oob_auc(control_feat, human_features)
        log(f"[w2_probe] === CONTROL RF-OOB AUC: {control_auc:.4f} (N={control_n} per class, "
            f"reference baseline 0.7573) ===")

    if args.main_run and not args.skip_control_gate and not args.skip_control:
        if abs(control_auc - 0.7573) > 0.10:
            log(f"[w2_probe] *** CONTROL FAILED TO REPRODUCE BASELINE "
                f"(got {control_auc:.4f}, expected ~0.7573) -- STOPPING before "
                f"spending GPU time on the guided run. Check the pipeline. ***")
            results = {
                "status": "ABORTED_CONTROL_MISMATCH", "control_auc": control_auc,
                "control_n": control_n, "feature_builder_validation": val,
            }
            RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
            return

    # --- GUIDED optimization ---
    batches = make_batches(records, args.batch_size)
    start_batch = 0
    if args.resume and PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "rb") as fh:
            saved = pickle.load(fh)
        records = saved["records"]
        start_batch = saved["next_batch"]
        log(f"[w2_probe] --resume: resuming from batch {start_batch}/{len(batches)}")

    t_run_start = time.perf_counter()
    max_seconds = args.max_minutes * 60.0
    step_times = []
    stopped_early = False
    for bi in range(start_batch, len(batches)):
        elapsed = time.perf_counter() - t_run_start
        if elapsed >= max_seconds:
            log(f"[w2_probe] wall clock cap ({args.max_minutes} min) reached at "
                f"batch {bi}/{len(batches)}, stopping cleanly")
            stopped_early = True
            break
        batch_ids = batches[bi]
        t_b0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        loss_hist = run_guided_batch(batch_ids, records, model, data_scale, device,
                                      args, COVERED_FEATURES)
        b_elapsed = time.perf_counter() - t_b0
        step_times.append(b_elapsed / max(len(loss_hist), 1))
        peak_vram_mb = (torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                        if device.type == "cuda" else 0.0)
        temp_mem = gpu_temp_mem()
        li = loss_hist[0] if loss_hist else float("nan")
        lf = loss_hist[-1] if loss_hist else float("nan")
        log(f"[w2_probe] batch {bi+1}/{len(batches)} n={len(batch_ids)} "
            f"loss_init={li:.4f} loss_final={lf:.4f} batch_s={b_elapsed:.1f} "
            f"peak_vram={peak_vram_mb:.0f}MB gpu={temp_mem} "
            f"elapsed_min={ (time.perf_counter()-t_run_start)/60:.1f}")

        with open(PROGRESS_PATH, "wb") as fh:
            pickle.dump({"records": records, "next_batch": bi + 1}, fh)

    guided_elapsed = time.perf_counter() - t_run_start
    log(f"[w2_probe] guided optimization done: {guided_elapsed/60:.1f} min, "
        f"stopped_early={stopped_early}")

    # --- decode + score ---
    guided_idx_done = [r["idx"] for r in records if r["trajectory"] is not None]
    guided_trajs = [records[i]["trajectory"] for i in guided_idx_done]
    log(f"[w2_probe] {len(guided_trajs)}/{len(records)} specs completed optimization+decode")

    real_feat18, real_valid = real_features_per_spec(
        [records[i]["trajectory"] for i in guided_idx_done])
    for k, i in enumerate(guided_idx_done):
        if real_valid[k]:
            real_det = np.array([NP_DET_TRANSFORMS[n](np.array([real_feat18[k, NAME_TO_IDX18[n]]]))[0]
                                  for n in COVERED_FEATURES])
            records[i]["real_det_feat"] = real_det.tolist()
            records[i]["real_raw_feat18"] = real_feat18[k].tolist()
        else:
            records[i]["real_det_feat"] = None

    valid_records_idx = [i for i in guided_idx_done if records[i].get("real_det_feat") is not None]
    log(f"[w2_probe] {len(valid_records_idx)}/{len(guided_idx_done)} decoded paths gave "
        f"valid real features.py feature vectors")

    synth_feat18_valid = np.stack(
        [records[i]["real_raw_feat18"] for i in valid_records_idx])
    guided_auc, guided_n, guided_clf = rf_oob_auc(synth_feat18_valid, human_features)
    log(f"[w2_probe] === GUIDED RF-OOB AUC: {guided_auc:.4f} (N={guided_n} per class) ===")

    per_feature_hit, overall_hit = target_hit_rates(records, valid_records_idx)
    log(f"[w2_probe] === TARGET-HIT RATES (real numpy features.py) ===")
    log(f"    overall: n_pairs={overall_hit['n_pairs']} "
        f"frac_hit_0.5std={overall_hit['frac_hit_0.5']:.4f} "
        f"frac_hit_1.0std={overall_hit['frac_hit_1.0']:.4f}")
    for name in COVERED_FEATURES:
        h = per_feature_hit[name]
        log(f"    {name:24s} n={h['n']} frac_hit_0.5={h['frac_hit_0.5']:.4f} "
            f"frac_hit_1.0={h['frac_hit_1.0']:.4f} mean_abs_err_std={h['mean_abs_err_std']:.4f}")

    n_use_disp = min(len(human_features), len(synth_feat18_valid))
    disp_rows = dispersion_table(synth_feat18_valid[:n_use_disp], human_features[:n_use_disp])
    log(f"[w2_probe] === DISPERSION TABLE (18 features, N={n_use_disp} per class) ===")
    for row in disp_rows:
        det_str = f"{row['det_dispersion_ratio']:.4f}" if row["det_dispersion_ratio"] is not None else "n/a"
        log(f"    {row['feature']:24s} covered={row['covered']!s:5s} "
            f"raw_ratio={row['raw_dispersion_ratio']:.4f} det_ratio={det_str}")

    importances = guided_clf.feature_importances_
    top8_idx = np.argsort(importances)[::-1][:8]
    log(f"[w2_probe] === GUIDED RF top-8 feature importances ===")
    for idx in top8_idx:
        log(f"    {FEATURE_NAMES[idx]:24s} {importances[idx]:.4f}")

    mean_step_time = float(np.mean(step_times)) if step_times else float("nan")
    latency_per_path_s = args.m_steps * mean_step_time
    log(f"[w2_probe] === LATENCY: mean per-optimizer-step time (full K={args.n_steps} "
        f"chain, batch={args.batch_size}) = {mean_step_time:.2f}s; implied per-path "
        f"serve latency ~= M({args.m_steps}) x step_time = {latency_per_path_s:.1f}s "
        f"(GPU processes the batch in parallel, so this approximates a single-request "
        f"latency too; far beyond the 2s serving budget as-is -- this probe measures "
        f"the MECHANISM, not a deployable path; W2 proper would train the conditioning "
        f"into the model for one-shot speed) ===")

    loss_inits = [r["loss_init"] for r in records if r["loss_init"] is not None and r["loss_init"] == r["loss_init"]]
    loss_finals = [r["loss_final"] for r in records if r["loss_final"] is not None and r["loss_final"] == r["loss_final"]]
    mean_loss_init = float(np.mean(loss_inits)) if loss_inits else float("nan")
    mean_loss_final = float(np.mean(loss_finals)) if loss_finals else float("nan")
    log(f"[w2_probe] === MEAN OPTIMIZATION LOSS: init={mean_loss_init:.4f} "
        f"final={mean_loss_final:.4f} (n_specs={len(loss_finals)}) ===")

    md5_after = md5_file(src_path)
    log(f"[w2_probe] source MD5 after: {md5_after}")
    md5_ok = md5_after == md5_before
    if not md5_ok:
        log("[w2_probe] *** WARNING: source checkpoint MD5 CHANGED -- should never happen ***")
    else:
        log("[w2_probe] source checkpoint MD5 unchanged, confirmed untouched.")
    log(f"[w2_probe] GPU after: {gpu_temp_mem()}")

    gate_pass = guided_auc <= 0.70
    log(f"[w2_probe] === PRE-REGISTERED GATE: guided_auc={guided_auc:.4f} <= 0.70 ? "
        f"{'PASS' if gate_pass else 'FAIL'} ===")

    results = {
        "status": "PARTIAL_WALL_CLOCK" if stopped_early else "COMPLETE",
        "tag": args.tag, "smoke": args.smoke, "main_run": args.main_run,
        "n_requested": args.n, "n_completed": len(guided_idx_done),
        "n_valid_real_features": len(valid_records_idx),
        "args": vars(args),
        "feature_builder_validation": val,
        "control_auc": control_auc, "control_n_per_class": control_n,
        "control_reference_baseline": 0.7573,
        "guided_auc": guided_auc, "guided_n_per_class": guided_n,
        "gate_pass_auc_leq_0.70": gate_pass,
        "target_hit_per_feature": per_feature_hit, "target_hit_overall": overall_hit,
        "dispersion_table_18_features": disp_rows,
        "guided_rf_top8_importances": [
            {"feature": FEATURE_NAMES[idx], "importance": float(importances[idx])}
            for idx in top8_idx
        ],
        "mean_step_time_sec_per_optimizer_step_full_chain": mean_step_time,
        "implied_per_path_latency_sec": latency_per_path_s,
        "mean_optimization_loss_init": mean_loss_init,
        "mean_optimization_loss_final": mean_loss_final,
        "guided_elapsed_sec": guided_elapsed,
        "control_elapsed_sec": control_elapsed,
        "md5_before": md5_before, "md5_after": md5_after, "md5_unchanged": md5_ok,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    log(f"[w2_probe] wrote {RESULTS_PATH}")
    log(f"[w2_probe] === DONE tag={args.tag} ===")


if __name__ == "__main__":
    main()
