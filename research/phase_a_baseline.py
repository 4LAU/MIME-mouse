"""Phase A: re-baseline candi_polar_flow_best.pt with integer-pixel rounding.

Deliberately does NOT import experiments/candi.py, because that module loads
data/human_eval_features.npy unconditionally at import time (for its
N_CANDIDATES>1 rescoring path), which is a protected file under the GPU-burst
eval safety rules. Instead this script duplicates the slice of
experiments/candi.py needed to reproduce its generate_paths() output bit-for-
bit for a given config: model load, guided polar decode, rotate/perp
correction + integer rounding (_build_trajectory), and the batched-by-seq_len
sampling loop.

Generation config: EXPERIMENTS.md's 2026-07-01 entry ("Post-processing
re-sweep at n=2000") records the exact knobs that produced the published
CANDI polar-flow number of 0.752 (which DIFFUSION_PILOT.md's gate references
directly): skew=0, perp=0.85, guide=0.15, correct=rotate, steps=200, CFG=0.
That is the "true starting point" this script re-measures with rounding now
added -- experiments/candi.py's raw module-level env-var defaults
(steps=50, CFG=2.0, correct=additive, no guide/perp) were NEVER the eval
convention used to produce any published CANDI number and give a much worse,
untuned baseline (confirmed: a first attempt at this script using those
defaults measured 0.96 AUC, wildly worse than 0.752, which is what surfaced
this distinction -- see the burst report).

Scoring reuses evaluate.py's spec-generation convention (center 960,540,
distances from data/human_distances.npy, uniform angle, seed 42) and
features.py's extract_feature_matrix, then trains the identical RF-OOB
detector evaluate.py uses -- but against data/human_val_features_grpo.npy,
never data/human_eval_features.npy.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from experiments._common import DurationModel, get_device
from features import FEATURE_NAMES, extract_feature_matrix

DATA_DIR = Path("data")
TRAIN_DIR = Path("training")
HZ = 125.0
CKPT_NAME = "candi_polar_flow_best.pt"
N_SYNTH = 2000
SEED = 42

# The published-0.752 generation config (EXPERIMENTS.md 2026-07-01), NOT
# experiments/candi.py's module-level env-var defaults.
N_SAMPLE_STEPS = 200
CFG_SCALE = 0.0
CORRECT = "rotate"
PERP_SCALE = 0.85
PERP_HP = 1.0          # untouched knob, stays at its default (no-op)
GUIDE = 0.15
DUR_STD = 0.7          # CANDI_DUR_STD default, not swept in that entry
EVAL_BATCH = 128        # CANDI_EVAL_BATCH default

CURV_MEAN_IDX = FEATURE_NAMES.index("curvature_mean")
CURV_STD_IDX = FEATURE_NAMES.index("curvature_std")

# DIFFUSION_PILOT_V2.md burst 3: the 11 "covered" (smooth, differentiable)
# features training/train_candi_chain3.py fits, and their detector-space
# transform per research/cond_realization_probe.py's to_detector_space
# column spec (the only full-feature-vector smooth transform definition in
# the repo, reused here as the scoring-side recipe for "detector space").
COVERED_FEATURES = [
    "mean_velocity", "std_velocity", "mean_acceleration", "std_acceleration",
    "mean_jerk", "std_jerk", "path_efficiency", "curvature_mean",
    "curvature_std", "angular_velocity_mean", "angular_velocity_std",
]


def _lg(x):
    return np.log1p(np.clip(x, 0.0, None))


COVERED_DET_TRANSFORMS = {
    "mean_velocity": _lg,
    "std_velocity": _lg,
    "mean_acceleration": lambda x: x / 1e4,
    "std_acceleration": _lg,
    "mean_jerk": lambda x: x / 1e6,
    "std_jerk": _lg,
    "path_efficiency": lambda x: x,
    "curvature_mean": lambda x: np.log1p(np.clip(x, 0.0, None) * 1e3),
    "curvature_std": lambda x: np.log1p(np.clip(x, 0.0, None) * 1e3),
    "angular_velocity_mean": _lg,
    "angular_velocity_std": _lg,
}


def covered_feature_variety_table(synth_bal, human_bal):
    """Per-feature detector-space std ratio (synth/human) for every burst-3
    covered feature, printed as a table for the post-hoc scoring sweep."""
    rows = []
    for name in COVERED_FEATURES:
        idx = FEATURE_NAMES.index(name)
        transform = COVERED_DET_TRANSFORMS[name]
        s_det = transform(synth_bal[:, idx])
        h_det = transform(human_bal[:, idx])
        s_std = float(np.std(s_det))
        h_std = float(np.std(h_det))
        ratio = s_std / max(h_std, 1e-12)
        rows.append((name, s_std, h_std, ratio))
    return rows


def print_covered_feature_table(rows):
    print("\n--- burst 3 covered-feature detector-space variety table "
          "(synth std / human std, vs data/human_val_features_grpo.npy) ---")
    for name, s_std, h_std, ratio in rows:
        print(f"  {name:24s} synth_std={s_std:8.4f}  human_std={h_std:8.4f}  "
              f"ratio={ratio:.4f}")


def to_detector_space_curv(col):
    """research/cond_realization_probe.py:to_detector_space's transform for
    the curvature_mean / curvature_std columns only (lines 156-157 there):
    log1p(clip(raw, 0, None) * 1e3). Raw curvature units are heavy-tailed
    (human_std ~5900 for curvature_mean), so raw-space std ratios are
    meaningless; this matches the space the RF detector and the checkpoint's
    feat_mu/feat_sd actually operate in."""
    return np.log1p(np.clip(col, 0.0, None) * 1e3)


def load_model(ckpt_name=CKPT_NAME):
    device = get_device()
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
    print(f"[phase_a] loaded {ckpt_path} device={device} params={n_params:,} "
          f"epoch={ckpt.get('epoch')} data_scale={data_scale}", flush=True)
    return model, data_scale, device, ckpt["config"]["max_seq_len"]


def decode_polar(raw_np, stall_np, data_scale):
    """experiments/candi.py's _decode_polar with every perturbation knob at
    its default-off value (skew/sharpen/jitter/etc. were never part of the
    0.752 config): just un-scale speed/dh, zero on stall, integrate heading,
    integrate position."""
    spd_scale, dh_scale = float(data_scale[0]), float(data_scale[1])
    speed = np.maximum(raw_np[:, 0] / spd_scale, 0.0)
    dheading = raw_np[:, 1] / dh_scale
    speed[stall_np > 0.5] = 0.0
    dheading[stall_np > 0.5] = 0.0
    heading = np.cumsum(dheading)
    cum_x = np.cumsum(speed * np.cos(heading))
    cum_y = np.cumsum(speed * np.sin(heading))
    return cum_x, cum_y


def build_trajectory(cum_x, cum_y, stall_np, seq_len, total_dist, dx, dy,
                      start_x, start_y, end_x, end_y, no_round=False):
    """experiments/candi.py's _build_trajectory with CORRECT="rotate" and
    PERP_SCALE=0.85 (PERP_HP stays 1.0, a no-op) -- the 0.752 config -- plus
    the integer-pixel rounding this pilot added."""
    target_dx = dx / total_dist if total_dist > 0 else 0.0
    target_dy = dy / total_dist if total_dist > 0 else 0.0

    if CORRECT == "rotate":
        raw_mag = math.hypot(cum_x[-1], cum_y[-1])
        if raw_mag > 1e-8:
            tgt_mag = math.hypot(target_dx, target_dy)
            scale = tgt_mag / raw_mag
            raw_ang = math.atan2(cum_y[-1], cum_x[-1])
            tgt_ang = math.atan2(target_dy, target_dx)
            rot = tgt_ang - raw_ang
            cos_r, sin_r = math.cos(rot), math.sin(rot)
            rx = (cum_x * cos_r - cum_y * sin_r) * scale
            ry = (cum_x * sin_r + cum_y * cos_r) * scale
            cum_x, cum_y = rx, ry
    else:
        err_x = target_dx - cum_x[-1]
        err_y = target_dy - cum_y[-1]
        if err_x * err_x + err_y * err_y > 1e-8:
            moving = stall_np < 0.5
            if moving.sum() > 0:
                magnitudes = np.sqrt(np.diff(np.concatenate([[0], cum_x])) ** 2 +
                                      np.diff(np.concatenate([[0], cum_y])) ** 2)
                mag_moving = magnitudes * moving
                total_mag = mag_moving.sum()
                weights = (mag_moving / total_mag if total_mag > 1e-8
                           else moving.astype(np.float64) / moving.sum())
                cum_x = cum_x + err_x * np.cumsum(weights)
                cum_y = cum_y + err_y * np.cumsum(weights)
            else:
                frac = np.linspace(0, 1, seq_len)
                cum_x = cum_x + err_x * frac
                cum_y = cum_y + err_y * frac

    if PERP_SCALE != 1.0 or PERP_HP != 1.0:
        tgt_mag = math.hypot(target_dx, target_dy)
        if tgt_mag > 1e-8:
            dx_n = target_dx / tgt_mag
            dy_n = target_dy / tgt_mag
            par = cum_x * dx_n + cum_y * dy_n
            perp_x = cum_x - par * dx_n
            perp_y = cum_y - par * dy_n
            perp_x = PERP_SCALE * perp_x
            perp_y = PERP_SCALE * perp_y
            cum_x = par * dx_n + perp_x
            cum_y = par * dy_n + perp_y

    out_x = cum_x * total_dist + start_x
    out_y = cum_y * total_dist + start_y

    # Integer-pixel rounding (this pilot's change): real mouse coordinates
    # are whole pixels; quantize only at generation time. --no-round skips
    # this to test whether the staircase/zigzag texture it introduces is
    # what the RF detector fingerprints.
    if not no_round:
        out_x = np.round(out_x)
        out_y = np.round(out_y)

    dt = 1.0 / HZ
    result = [(start_x, start_y, 0.0)]
    for i in range(seq_len):
        result.append((float(out_x[i]), float(out_y[i]), (i + 1) * dt))
    result[-1] = (end_x, end_y, result[-1][2])
    return result


def sample_guided_flow(model, data_scale, device, cond, seq_len,
                        target_cos, target_sin, n_steps, cfg_scale, guide,
                        z_scale=1.0):
    """Verbatim port of experiments/candi.py's _sample_guided_flow -- the
    branch generate_paths() actually takes for this checkpoint, since
    pred_type=="flow" and GUIDE>0 and POLAR dispatches here (NOT
    _sample_guided_polar, which is only for non-flow pred types). This is
    the 0.752 config: guide=0.15, steps=200, cfg=0. No protected-data
    dependency."""
    B = cond.shape[0]
    dev = device
    spd_s = float(data_scale[0])
    dh_s = float(data_scale[1])
    tgt_angles = np.atleast_1d(np.arctan2(target_sin, target_cos)).astype(np.float64)

    xt = torch.randn(B, seq_len, 2, device=dev) * z_scale
    stall_s = torch.full((B, seq_len), model.STALL_MASK, device=dev)
    mflag = torch.ones(B, seq_len, device=dev)

    dt = 1.0 / n_steps

    sl = None
    for i in range(n_steps):
        t_cont = 1.0 - i * dt
        t_scaled = torch.full((B,), t_cont * (model.n_steps - 1), device=dev)
        v_pred, sl = model(xt, stall_s, mflag, t_scaled, cond)

        if cfg_scale > 0:
            v_u, sl_u = model(xt, stall_s, mflag, t_scaled, torch.zeros_like(cond))
            v_pred = v_u + cfg_scale * (v_pred - v_u)
            sl = sl_u + cfg_scale * (sl - sl_u)

        dp = xt - t_cont * v_pred

        frac = 1.0 - t_cont
        if frac > 0.3:
            conf = torch.abs(sl)
            thresh = max(0.5, 3.0 * (1.0 - frac))
            reveal = (conf > thresh) & (mflag > 0.5)
            stall_s = torch.where(reveal, (torch.sigmoid(sl) > 0.5).float(), stall_s)
            mflag = torch.where(reveal, torch.zeros_like(mflag), mflag)

        if frac > 0.3 and guide > 0:
            dp = dp.clone()
            for b in range(B):
                spd = torch.clamp(dp[b, :, 0] / spd_s, min=0)
                dh = dp[b, :, 1] / dh_s
                active_stall = (stall_s[b] > 0.5) & (mflag[b] < 0.5)
                spd_eff = spd * (~active_stall).float()
                dh_eff = dh * (~active_stall).float()
                heading = torch.cumsum(dh_eff, dim=0)
                cx = torch.cumsum(spd_eff * torch.cos(heading), dim=0)
                cy = torch.cumsum(spd_eff * torch.sin(heading), dim=0)
                raw_mag = math.hypot(cx[-1].item(), cy[-1].item())
                if raw_mag > 1e-6:
                    raw_ang = math.atan2(cy[-1].item(), cx[-1].item())
                    tgt_ang = float(tgt_angles[b]) if tgt_angles.size > 1 else float(tgt_angles[0])
                    rot = (tgt_ang - raw_ang) * guide * frac
                    dp[b, 0, 1] += rot * dh_s

        if t_cont > 1e-6:
            v_guided = (xt - dp) / t_cont
        else:
            v_guided = v_pred
        xt = xt - dt * v_guided

    sp = torch.sigmoid(sl)
    final_stall = torch.where(mflag > 0.5, (sp > 0.5).float(), stall_s)
    out = xt.clone()
    out[final_stall > 0.5] = 0.0
    return out, final_stall


def generate_paths(model, data_scale, device, duration_model, specs, no_round=False,
                   z_scale=1.0):
    """Mirrors experiments/candi.py's generate_paths() GUIDE>0 branch (the
    0.752 config: candidates=1, feat_guide=0, guide=0.15): group by seq_len,
    batch through the guided polar sampler, decode, build_trajectory."""
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
        seq_len = max(5, min(int(round(duration * HZ)), model.max_seq_len_cfg))
        pending.append({
            "idx": idx, "seq_len": seq_len, "angle": angle,
            "cond": [log_dist, log_dur, math.cos(angle), math.sin(angle)],
            "total_dist": total_dist, "dx": dx, "dy": dy,
            "sx": sx, "sy": sy, "ex": ex, "ey": ey,
        })

    groups: dict = {}
    for item in pending:
        groups.setdefault(item["seq_len"], []).append(item)

    for seq_len, items in groups.items():
        for c0 in range(0, len(items), EVAL_BATCH):
            chunk = items[c0:c0 + EVAL_BATCH]
            cond = torch.tensor([it["cond"] for it in chunk],
                                 dtype=torch.float32, device=device)
            tcos = np.array([math.cos(it["angle"]) for it in chunk])
            tsin = np.array([math.sin(it["angle"]) for it in chunk])
            with torch.no_grad():
                raw, stall = sample_guided_flow(
                    model, data_scale, device, cond, seq_len, tcos, tsin,
                    n_steps=N_SAMPLE_STEPS, cfg_scale=CFG_SCALE, guide=GUIDE,
                    z_scale=z_scale,
                )
            raw_all = raw.cpu().numpy()
            stall_all = stall.cpu().numpy()
            for b, it in enumerate(chunk):
                cum_x, cum_y = decode_polar(raw_all[b], stall_all[b], data_scale)
                results[it["idx"]] = build_trajectory(
                    cum_x, cum_y, stall_all[b], seq_len,
                    it["total_dist"], it["dx"], it["dy"],
                    it["sx"], it["sy"], it["ex"], it["ey"],
                    no_round=no_round,
                )
    return results


def make_specs(n, seed):
    """Mirrors evaluate.py's generate_synthetic_trajectories spec loop
    exactly (center 960,540, distances from human_distances.npy, uniform
    angle) -- but only the spec math, never touching any human *features*
    file."""
    distances = np.load(DATA_DIR / "human_distances.npy")
    rng = np.random.default_rng(seed)
    center_x, center_y = 960.0, 540.0
    specs = []
    for _ in range(n):
        dist = float(rng.choice(distances))
        angle = float(rng.uniform(0, 2 * np.pi))
        end_x = center_x + dist * np.cos(angle)
        end_y = center_y + dist * np.sin(angle)
        specs.append((center_x, center_y, end_x, end_y))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_SYNTH,
                    help="Number of synthetic trajectories (default 2000; use "
                         "a small value for a CPU smoke test only, never as a "
                         "decision number)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-round", action="store_true",
                     help="Skip the two np.round() calls in build_trajectory. "
                          "This flag is what reproduces the published 0.752; "
                          "the rounded default reads 0.9520 (n=2000, "
                          "research/w3_candi_control_results.json). Rounding a "
                          "path the model produced in continuous displacement "
                          "space leaves a staircase the RF detector reads.")
    ap.add_argument("--save-features", type=str, default=None,
                     help="Path to np.save the synthetic feature matrix "
                          "(pre-balancing, all valid rows) so reruns aren't "
                          "needed for follow-up analysis.")
    ap.add_argument("--save-trajectories", type=str, default=None,
                     help="Path to pickle the raw trajectory list (list of "
                          "(x,y,t) tuples) BEFORE feature extraction, so "
                          "quantization schemes can be tested offline "
                          "without regenerating.")
    ap.add_argument("--ckpt", type=str, default=CKPT_NAME,
                     help="Checkpoint filename inside training/ (default "
                          f"{CKPT_NAME}). Must be a polar flow checkpoint.")
    args = ap.parse_args()

    model, data_scale, device, max_seq_len_cfg = load_model(args.ckpt)
    # generate_paths() needs the checkpoint's max_seq_len the same way
    # experiments/candi.py reads _cfg["max_seq_len"]; stash it on the model.
    model.max_seq_len_cfg = max_seq_len_cfg

    duration_model = DurationModel(TRAIN_DIR, std_mult=DUR_STD)

    specs = make_specs(args.n, args.seed)

    t0 = time.perf_counter()
    trajectories = generate_paths(model, data_scale, device, duration_model,
                                   specs, no_round=args.no_round)
    trajectories = [t for t in trajectories if t is not None and len(t) >= 2]
    gen_elapsed = time.perf_counter() - t0
    print(f"[phase_a] generated {len(trajectories)}/{args.n} trajectories "
          f"in {gen_elapsed:.1f}s (no_round={args.no_round})", flush=True)

    if args.save_trajectories:
        import pickle
        traj_path = Path(args.save_trajectories)
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        with open(traj_path, "wb") as fh:
            pickle.dump(trajectories, fh)
        print(f"[phase_a] pickled {len(trajectories)} raw trajectories to "
              f"{traj_path}", flush=True)

    synth_features = extract_feature_matrix(trajectories)
    valid_ratio = len(synth_features) / max(args.n, 1)
    print(f"[phase_a] valid feature vectors: {len(synth_features)}/{args.n} "
          f"({valid_ratio:.0%})", flush=True)

    if args.save_features:
        save_path = Path(args.save_features)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, synth_features)
        print(f"[phase_a] saved synthetic feature matrix to {save_path} "
              f"shape={synth_features.shape}", flush=True)

    human_features = np.load(DATA_DIR / "human_val_features_grpo.npy")
    print(f"[phase_a] human_val_features_grpo.npy shape={human_features.shape}", flush=True)

    n_use = min(len(human_features), len(synth_features))
    human_bal = human_features[:n_use]
    synth_bal = synth_features[:n_use]
    print(f"[phase_a] N used for scoring: {n_use} per class", flush=True)

    X = np.vstack([human_bal, synth_bal])
    y = np.concatenate([np.zeros(n_use), np.ones(n_use)])

    clf = RandomForestClassifier(
        n_estimators=100, oob_score=True, n_jobs=-1, random_state=SEED,
    )
    clf.fit(X, y)
    oob_proba = clf.oob_decision_function_[:, 1]
    auc_rf_oob = roc_auc_score(y, oob_proba)

    # Raw-unit ratios (kept for continuity with the first run; curvature raw
    # units are heavy-tailed and this ratio is not meaningful on its own).
    curv_mean_synth_std = float(np.std(synth_bal[:, CURV_MEAN_IDX]))
    curv_mean_human_std = float(np.std(human_bal[:, CURV_MEAN_IDX]))
    curv_std_synth_std = float(np.std(synth_bal[:, CURV_STD_IDX]))
    curv_std_human_std = float(np.std(human_bal[:, CURV_STD_IDX]))

    variety_ratio_mean = curv_mean_synth_std / max(curv_mean_human_std, 1e-12)
    variety_ratio_std = curv_std_synth_std / max(curv_std_human_std, 1e-12)

    # Detector-space ratios: log1p(clip(raw,0,None)*1e3) on the two curvature
    # columns, matching the space the RF/checkpoint actually see.
    synth_curv_mean_det = to_detector_space_curv(synth_bal[:, CURV_MEAN_IDX])
    human_curv_mean_det = to_detector_space_curv(human_bal[:, CURV_MEAN_IDX])
    synth_curv_std_det = to_detector_space_curv(synth_bal[:, CURV_STD_IDX])
    human_curv_std_det = to_detector_space_curv(human_bal[:, CURV_STD_IDX])

    det_synth_curv_mean_std = float(np.std(synth_curv_mean_det))
    det_human_curv_mean_std = float(np.std(human_curv_mean_det))
    det_synth_curv_std_std = float(np.std(synth_curv_std_det))
    det_human_curv_std_std = float(np.std(human_curv_std_det))

    det_variety_ratio_mean = det_synth_curv_mean_std / max(det_human_curv_mean_std, 1e-12)
    det_variety_ratio_std = det_synth_curv_std_std / max(det_human_curv_std_std, 1e-12)

    covered_rows = covered_feature_variety_table(synth_bal, human_bal)

    importances = clf.feature_importances_
    top8_idx = np.argsort(importances)[::-1][:8]

    print("\n=== PHASE A RESULTS ===")
    print(f"N synthetic generated: {len(trajectories)} (requested {args.n})")
    print(f"N used for scoring (per class): {n_use}")
    print(f"Generation wall time: {gen_elapsed:.1f}s")
    print(f"no_round: {args.no_round}")
    print(f"val_auc (RF OOB vs human_val_features_grpo.npy): {auc_rf_oob:.4f}")
    print(f"--- raw-unit variety ratios (heavy-tailed, continuity only) ---")
    print(f"curvature_mean: synth_std={curv_mean_synth_std:.6f} "
          f"human_std={curv_mean_human_std:.6f} ratio={variety_ratio_mean:.4f}")
    print(f"curvature_std:  synth_std={curv_std_synth_std:.6f} "
          f"human_std={curv_std_human_std:.6f} ratio={variety_ratio_std:.4f}")
    print(f"--- detector-space variety ratios (log1p(clip(raw,0,None)*1e3)) ---")
    print(f"curvature_mean: synth_std={det_synth_curv_mean_std:.6f} "
          f"human_std={det_human_curv_mean_std:.6f} ratio={det_variety_ratio_mean:.4f}")
    print(f"curvature_std:  synth_std={det_synth_curv_std_std:.6f} "
          f"human_std={det_human_curv_std_std:.6f} ratio={det_variety_ratio_std:.4f}")
    print(f"--- top 8 RF feature importances ---")
    for idx in top8_idx:
        print(f"{FEATURE_NAMES[idx]}: {importances[idx]:.4f}")
    print_covered_feature_table(covered_rows)


if __name__ == "__main__":
    main()
