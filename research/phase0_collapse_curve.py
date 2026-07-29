"""Phase 0 of DIFFUSION_PILOT_V2.md: a no-training diagnostic that measures
WHERE along CANDI's 200-step generation process curvature variety collapses.

Deliberately does NOT import experiments/candi.py at module level (that module
loads data/human_eval_features.npy, a protected held-out file, unconditionally
at import time). This script is adapted directly from research/phase_a_baseline.py,
which already solved that problem by reimplementing the exact 0.752 generation
config (checkpoint training/candi_polar_flow_best.pt, N_SAMPLE_STEPS=200,
CFG_SCALE=0.0, CORRECT="rotate", PERP_SCALE=0.85, GUIDE=0.15, seed 42) without
the protected import. All human reference stats come from
data/human_val_features_grpo.npy (N=2000), never data/human_eval_features.npy.

No training happens here. No checkpoint is written. training/candi_polar_flow_best.pt
is only read (torch.load) and its MD5 is expected to be identical before/after.

Mechanism: during sampling, at each step i the sampler already computes
    dp = xt - t_cont * v_pred
which is the model's one-step ("x0_hat") projection of the finished path given
its current v_pred (this is the exact convention in
research/phase_a_baseline.py's port of experiments/candi.py's
_sample_guided_flow, verbatim in the loop body). We snapshot this dp (AFTER
that step's reveal/guide adjustments, i.e. the same dp that actually drives
the next xt update) at selected values of "steps remaining" = n_steps - i,
decode each snapshot through the identical decode path used for the final
output (decode_polar + build_trajectory, no rounding), and measure
detector-space curvature variety ratio against human_val_features_grpo.npy.

Steps-remaining = 0 is NOT a dp snapshot -- it is the actual final sampler
output (out/final_stall after the loop finishes), decoded exactly as
phase_a_baseline.py does.

Caveat on the discrete stall state: at a mid-generation snapshot, some
positions may still be "masked" (mflag > 0.5, not yet hard-revealed by the
sampler's confidence-gated reveal mechanism). For those positions we use the
CURRENT sigmoid(sl) > 0.5 threshold, exactly mirroring how the final output
resolves any positions that were never confidently revealed
(final_stall = where(mflag>0.5, sigmoid(sl)>0.5, stall_s)). This is applied
at every snapshot, not just the end, so snapshot decoding is identical in
form to final decoding, just evaluated mid-loop.

Configs:
  (a) baseline       -- guide=0.15, perp_scale=0.85, correct=rotate, stall_reveal=True
  (b) guidance_off    -- guide=0.0,  perp_scale=1.0,  correct=rotate, stall_reveal=True
  (c) no_stall_reveal -- guide=0.15, perp_scale=0.85, correct=rotate, stall_reveal=False
      (the confidence-gated mid-generation reveal/freeze block is a clean,
      separable "if" block in the sampler -- skipping it leaves mflag at 1.0
      throughout and lets the FINAL sigmoid(sl)>0.5 decide every position's
      stall state, instead of freezing early high-confidence positions. This
      does not change any other semantics, so config (c) is run as specified.)

CORRECT is always "rotate" -- this is the endpoint-matching post-hoc
correction (without it trajectories would not reach their target endpoints at
all, making feature comparisons meaningless). "guidance" (the in-loop
steering block gated on `guide`) and "perp narrowing" (the post-hoc
perpendicular-deviation scale) are the two knobs config (b) turns off; they
are functionally distinct from CORRECT and from each other.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments._common import DurationModel, get_device
from features import FEATURE_NAMES, extract_feature_matrix

DATA_DIR = Path("data")
TRAIN_DIR = Path("training")
HZ = 125.0
CKPT_NAME = "candi_polar_flow_best.pt"
SEED = 42

N_SAMPLE_STEPS = 200
CFG_SCALE = 0.0
DUR_STD = 0.7
EVAL_BATCH = 128

SNAPSHOT_STEPS_REMAINING = [200, 100, 50, 25, 16, 10, 5, 2, 0]

CURV_MEAN_IDX = FEATURE_NAMES.index("curvature_mean")
CURV_STD_IDX = FEATURE_NAMES.index("curvature_std")
STD_JERK_IDX = FEATURE_NAMES.index("std_jerk")
PATH_EFF_IDX = FEATURE_NAMES.index("path_efficiency")

CONFIGS = {
    "baseline": dict(guide=0.15, perp_scale=0.85, correct="rotate", stall_reveal=True),
    "guidance_off": dict(guide=0.0, perp_scale=1.0, correct="rotate", stall_reveal=True),
    "no_stall_reveal": dict(guide=0.15, perp_scale=0.85, correct="rotate", stall_reveal=False),
}


def to_detector_space_col(col, name):
    """research/cond_realization_probe.py:to_detector_space, restricted to the
    four columns this diagnostic reports. curvature_mean/curvature_std use
    log1p(clip(raw,0,None)*1e3) (lines 156-157 there); std_jerk uses
    log1p(clip(raw,0,None)) (line 153); path_efficiency is untransformed
    (line 154, identity)."""
    if name in ("curvature_mean", "curvature_std"):
        return np.log1p(np.clip(col, 0.0, None) * 1e3)
    if name == "std_jerk":
        return np.log1p(np.clip(col, 0.0, None))
    if name == "path_efficiency":
        return col
    raise ValueError(name)


def load_model(ckpt_name=CKPT_NAME, force_cpu=False):
    device = torch.device("cpu") if force_cpu else get_device()
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


def decode_polar(raw_np, stall_np, data_scale):
    """Identical to phase_a_baseline.py:decode_polar."""
    spd_scale, dh_scale = float(data_scale[0]), float(data_scale[1])
    speed = np.maximum(raw_np[:, 0] / spd_scale, 0.0)
    dheading = raw_np[:, 1] / dh_scale
    speed[stall_np > 0.5] = 0.0
    dheading[stall_np > 0.5] = 0.0
    heading = np.cumsum(dheading)
    cum_x = np.cumsum(speed * np.cos(heading))
    cum_y = np.cumsum(speed * np.sin(heading))
    return cum_x, cum_y


def build_trajectory(cum_x, cum_y, seq_len, total_dist, dx, dy,
                      start_x, start_y, end_x, end_y, correct, perp_scale):
    """Identical to phase_a_baseline.py:build_trajectory, but with CORRECT and
    PERP_SCALE as parameters (needed here since this script runs multiple
    configs in one process) and no_round hardcoded True: Phase 0's hard
    constraint is no integer rounding anywhere, matching the split gate."""
    target_dx = dx / total_dist if total_dist > 0 else 0.0
    target_dy = dy / total_dist if total_dist > 0 else 0.0

    if correct == "rotate":
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
        raise NotImplementedError("only correct='rotate' is used in Phase 0")

    if perp_scale != 1.0:
        tgt_mag = math.hypot(target_dx, target_dy)
        if tgt_mag > 1e-8:
            dx_n = target_dx / tgt_mag
            dy_n = target_dy / tgt_mag
            par = cum_x * dx_n + cum_y * dy_n
            perp_x = cum_x - par * dx_n
            perp_y = cum_y - par * dy_n
            perp_x = perp_scale * perp_x
            perp_y = perp_scale * perp_y
            cum_x = par * dx_n + perp_x
            cum_y = par * dy_n + perp_y

    out_x = cum_x * total_dist + start_x
    out_y = cum_y * total_dist + start_y
    # No rounding anywhere (Phase 0 hard constraint).

    dt = 1.0 / HZ
    result = [(start_x, start_y, 0.0)]
    for i in range(seq_len):
        result.append((float(out_x[i]), float(out_y[i]), (i + 1) * dt))
    result[-1] = (end_x, end_y, result[-1][2])
    return result


def sample_guided_flow_snapshots(model, data_scale, device, cond, seq_len,
                                  target_cos, target_sin, n_steps, cfg_scale,
                                  guide, stall_reveal, snapshot_steps_remaining):
    """Verbatim port of phase_a_baseline.py:sample_guided_flow (itself a port
    of experiments/candi.py's _sample_guided_flow), extended to:
      (1) optionally skip the confidence-gated reveal/freeze block
          (stall_reveal=False -> config (c)),
      (2) record dp = xt - t_cont * v_pred (post reveal/guide adjustment, the
          same dp that drives the next xt update) at the requested
          steps-remaining checkpoints, together with the matching
          "current revealed state" stall snapshot.
    Returns (out, final_stall, snapshots) where snapshots maps
    steps_remaining -> (dp_np [B,seq_len,2], stall_np [B,seq_len])."""
    B = cond.shape[0]
    dev = device
    spd_s = float(data_scale[0])
    dh_s = float(data_scale[1])
    tgt_angles = np.atleast_1d(np.arctan2(target_sin, target_cos)).astype(np.float64)

    xt = torch.randn(B, seq_len, 2, device=dev)
    stall_s = torch.full((B, seq_len), model.STALL_MASK, device=dev)
    mflag = torch.ones(B, seq_len, device=dev)

    dt = 1.0 / n_steps
    want = {n_steps - sr for sr in snapshot_steps_remaining if sr > 0}
    snapshots = {}

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
        if stall_reveal and frac > 0.3:
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

        if i in want:
            snap_stall = torch.where(mflag > 0.5, (torch.sigmoid(sl) > 0.5).float(), stall_s)
            snapshots[n_steps - i] = (dp.detach().cpu().numpy().copy(),
                                       snap_stall.detach().cpu().numpy().copy())

        if t_cont > 1e-6:
            v_guided = (xt - dp) / t_cont
        else:
            v_guided = v_pred
        xt = xt - dt * v_guided

    sp = torch.sigmoid(sl)
    final_stall = torch.where(mflag > 0.5, (sp > 0.5).float(), stall_s)
    out = xt.clone()
    out[final_stall > 0.5] = 0.0
    return (out.cpu().numpy(), final_stall.cpu().numpy(), snapshots)


def generate_paths_snapshots(model, data_scale, device, duration_model, specs,
                              guide, perp_scale, correct, stall_reveal,
                              snapshot_steps_remaining):
    """Mirrors phase_a_baseline.py:generate_paths, extended to also decode and
    collect the dp snapshots at every requested steps-remaining checkpoint.
    Returns dict: steps_remaining -> list[len(specs)] of trajectories
    (None for degenerate specs, matching final-output handling)."""
    n = len(specs)
    results_by_sr = {sr: [None] * n for sr in snapshot_steps_remaining}
    pending = []
    for idx, (sx, sy, ex, ey) in enumerate(specs):
        dx = ex - sx
        dy = ey - sy
        total_dist = math.hypot(dx, dy)
        if total_dist < 1.0:
            degenerate = [(sx, sy, 0.0), (ex, ey, 0.008)]
            for sr in snapshot_steps_remaining:
                results_by_sr[sr][idx] = degenerate
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
                final_raw, final_stall, snaps = sample_guided_flow_snapshots(
                    model, data_scale, device, cond, seq_len, tcos, tsin,
                    n_steps=N_SAMPLE_STEPS, cfg_scale=CFG_SCALE, guide=guide,
                    stall_reveal=stall_reveal,
                    snapshot_steps_remaining=snapshot_steps_remaining,
                )
            # steps_remaining = 0: actual final output, decoded exactly like phase_a.
            for b, it in enumerate(chunk):
                cum_x, cum_y = decode_polar(final_raw[b], final_stall[b], data_scale)
                results_by_sr[0][it["idx"]] = build_trajectory(
                    cum_x, cum_y, seq_len, it["total_dist"], it["dx"], it["dy"],
                    it["sx"], it["sy"], it["ex"], it["ey"],
                    correct=correct, perp_scale=perp_scale,
                )
            # dp snapshots at every other requested steps_remaining.
            for sr, (dp_np, stall_np) in snaps.items():
                for b, it in enumerate(chunk):
                    cum_x, cum_y = decode_polar(dp_np[b], stall_np[b], data_scale)
                    results_by_sr[sr][it["idx"]] = build_trajectory(
                        cum_x, cum_y, seq_len, it["total_dist"], it["dx"], it["dy"],
                        it["sx"], it["sy"], it["ex"], it["ey"],
                        correct=correct, perp_scale=perp_scale,
                    )
    return results_by_sr


def make_specs(n, seed):
    """Identical to phase_a_baseline.py:make_specs."""
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
    ap.add_argument("--n", type=int, default=1000,
                     help="Number of synthetic trajectories per config "
                          "(default 1000; use a small value for a CPU smoke "
                          "test only, never as a decision number)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--cpu", action="store_true", help="Force CPU (smoke test)")
    ap.add_argument("--snapshots", type=str, default=None,
                     help="Comma-separated override of steps-remaining "
                          "checkpoints, for smoke tests")
    ap.add_argument("--configs", type=str, default=None,
                     help="Comma-separated subset of config names to run, "
                          "for smoke tests")
    ap.add_argument("--out-csv", type=str, default="research/phase0_results.csv")
    ap.add_argument("--out-features-dir", type=str, default="research/phase0_features")
    args = ap.parse_args()

    snapshot_steps_remaining = SNAPSHOT_STEPS_REMAINING
    if args.snapshots:
        snapshot_steps_remaining = [int(x) for x in args.snapshots.split(",")]

    configs = CONFIGS
    if args.configs:
        keep = set(args.configs.split(","))
        configs = {k: v for k, v in CONFIGS.items() if k in keep}

    model, data_scale, device, max_seq_len_cfg = load_model(force_cpu=args.cpu)
    model.max_seq_len_cfg = max_seq_len_cfg
    duration_model = DurationModel(TRAIN_DIR, std_mult=DUR_STD)
    specs = make_specs(args.n, args.seed)

    human_features = np.load(DATA_DIR / "human_val_features_grpo.npy")
    print(f"[phase0] human_val_features_grpo.npy shape={human_features.shape}", flush=True)
    human_det = {
        name: to_detector_space_col(human_features[:, idx], name)
        for name, idx in [("curvature_mean", CURV_MEAN_IDX),
                           ("curvature_std", CURV_STD_IDX),
                           ("std_jerk", STD_JERK_IDX),
                           ("path_efficiency", PATH_EFF_IDX)]
    }
    human_std = {name: float(np.std(col)) for name, col in human_det.items()}
    print(f"[phase0] human detector-space stds: {human_std}", flush=True)

    out_features_dir = Path(args.out_features_dir)
    out_features_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    t_all0 = time.perf_counter()
    for cfg_name, cfg in configs.items():
        print(f"[phase0] === config={cfg_name} params={cfg} ===", flush=True)
        t0 = time.perf_counter()
        results_by_sr = generate_paths_snapshots(
            model, data_scale, device, duration_model, specs,
            guide=cfg["guide"], perp_scale=cfg["perp_scale"],
            correct=cfg["correct"], stall_reveal=cfg["stall_reveal"],
            snapshot_steps_remaining=snapshot_steps_remaining,
        )
        elapsed = time.perf_counter() - t0
        print(f"[phase0] config={cfg_name} generation+snapshots done in {elapsed:.1f}s", flush=True)

        for sr in snapshot_steps_remaining:
            trajs = [t for t in results_by_sr[sr] if t is not None and len(t) >= 2]
            n_paths = len(trajs)
            feats = extract_feature_matrix(trajs)
            n_valid = len(feats)
            has_nan = bool(np.any(~np.isfinite(feats))) if feats.size else False

            feat_path = out_features_dir / f"{cfg_name}_sr{sr}.npy"
            np.save(feat_path, feats)

            ratios = {}
            for name, idx in [("curvature_mean", CURV_MEAN_IDX),
                               ("curvature_std", CURV_STD_IDX),
                               ("std_jerk", STD_JERK_IDX),
                               ("path_efficiency", PATH_EFF_IDX)]:
                if n_valid == 0:
                    ratios[name] = float("nan")
                    continue
                col_det = to_detector_space_col(feats[:, idx], name)
                synth_std = float(np.std(col_det))
                ratios[name] = synth_std / max(human_std[name], 1e-12)

            row = {
                "config": cfg_name,
                "steps_remaining": sr,
                "n_paths": n_valid,
                "ratio_curvature_mean": ratios["curvature_mean"],
                "ratio_curvature_std": ratios["curvature_std"],
                "ratio_std_jerk": ratios["std_jerk"],
                "ratio_path_efficiency": ratios["path_efficiency"],
            }
            rows.append(row)
            print(f"[phase0]   sr={sr:>4} n_valid={n_valid}/{n_paths} nan={has_nan} "
                  f"ratio_curv_mean={ratios['curvature_mean']:.4f} "
                  f"ratio_curv_std={ratios['curvature_std']:.4f} "
                  f"ratio_std_jerk={ratios['std_jerk']:.4f} "
                  f"ratio_path_eff={ratios['path_efficiency']:.4f}", flush=True)

    total_elapsed = time.perf_counter() - t_all0
    print(f"[phase0] ALL configs done in {total_elapsed:.1f}s", flush=True)

    import csv
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "config", "steps_remaining", "n_paths",
            "ratio_curvature_mean", "ratio_curvature_std",
            "ratio_std_jerk", "ratio_path_efficiency",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[phase0] wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
