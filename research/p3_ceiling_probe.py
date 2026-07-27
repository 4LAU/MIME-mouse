"""What is the best score P3's output representation can ever reach?

P3 generates paths as coordinates on the 192-slot uniform 125 Hz grid that
prepare_training_data.py builds. Whatever P3 learns, its output lives in that
representation, so the ceiling is measurable without training anything: take
REAL human grid paths, which are what a perfect model would emit, push them
through P3's decode, and score them with the standing contract scorer.

Then repeat with a controlled amount of model error injected, to see how the
score behaves for a model that is very good but not perfect.

The headline finding this script exists to demonstrate: on this representation
the metric is dominated by exact numerical structure rather than by motion
realism. Real human recordings are integer pixels sampled at irregular times.
features.resample_trajectory interpolates them to 125 Hz, which leaves long
runs of exactly collinear points, so consecutive step directions are bitwise
identical and the angle difference between them is exactly zero.
num_direction_changes counts sign changes in that angle difference, and an
exact zero has sign zero, so those runs contribute nothing. A generated path
carries no such exact structure: every angle difference is a small nonzero
number of essentially random sign, and the count roughly doubles. Perturbing
real paths by one billionth of a pixel, far below anything that could change
how the motion looks, moves the AUC from 0.65 to 0.84 on its own.

Model error is injected in increment space (see research/p3_repr_probe.py):
per-step increments normalized to unit magnitude, perturbed, then re-centred so
the path still lands exactly. sigma is therefore a fraction of a typical step.
P3 v1's measured angular velocity corresponds to roughly sigma 0.25.

Usage:
  env PYTHONPATH=. MIME_GRID_DIR=~/mime_data \
    ~/venvs/mime/bin/python research/p3_ceiling_probe.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

import scoring
from features import FEATURE_NAMES
from p3_repr_probe import load_paths, perturb_increment

HZ = 125.0
NDC = FEATURE_NAMES.index("num_direction_changes")
AV = FEATURE_NAMES.index("angular_velocity_mean")


def decode(px, mode, rng):
    """px: float pixel coordinates on the uniform 8 ms grid.

    grid_raw    fractional coordinates, straight off the grid
    grid_round  rounded to whole pixels, what P3 v1 shipped
    grid_snap   whole pixels, lattice-aligned at slow steps
                (the EXPERIMENTS.md EVENT_SNAP=2.5 decode contract)
    event       integer pixels emitted only where the pointer actually moved,
                at their own timestamps, the way a real recording is written.
                The feature extractor then re-interpolates back to 125 Hz.
    """
    t = np.arange(len(px)) / HZ
    if mode == "grid_raw":
        return np.c_[px, t]
    if mode == "grid_round":
        return np.c_[np.round(px), t]
    if mode == "grid_snap":
        d = np.diff(px, axis=0).copy()
        slow = np.hypot(d[:, 0], d[:, 1]) < 2.5
        d[slow] = np.round(d[slow])
        q = np.vstack([np.round(px[0]), np.round(px[0]) + np.cumsum(d, axis=0)])
        return np.c_[q, t]
    if mode == "event":
        q = np.round(px)
        keep = np.ones(len(q), bool)
        keep[1:] = np.any(np.diff(q, axis=0) != 0, axis=1)
        keep[0] = keep[-1] = True
        q, tt = q[keep], t[keep].copy()
        if len(tt) > 1:
            # a real recording's duration is not a multiple of 8 ms
            tt[-1] = tt[-2] + rng.uniform(1e-6, 1.0 / HZ)
        return np.c_[q, tt]
    raise ValueError(mode)


def evaluate(paths, sigma, mode, seed, jitter_px=0.0):
    rng = np.random.default_rng(seed)
    trajs = []
    for p, dist in paths:
        q = p if sigma == 0 else perturb_increment(p, sigma, rng)
        px = np.array([500.0, 500.0]) + q * dist
        if jitter_px:
            px = px + rng.normal(0.0, jitter_px, px.shape)
        trajs.append(decode(px, mode, rng))
    X = scoring.extract_features_from_paths(trajs)
    r = scoring.score_features(X)
    return {"auc_rf_oob": r["auc_rf_oob"],
            "num_direction_changes": float(np.mean(X[:, NDC])),
            "angular_velocity_mean": float(np.mean(X[:, AV])),
            "n_collapsed": len(r["collapse_features"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(R / "research" / "p3_ceiling_results.json"))
    args = ap.parse_args()

    paths = load_paths(args.n, args.seed)
    human = np.load(R / "data" / "human_val_features_grpo.npy")
    print(f"[ceiling] {len(paths)} real grid paths, gate is AUC <= 0.70")
    print(f"[ceiling] reference num_direction_changes "
          f"{np.mean(human[:, NDC]):.2f}, angular_velocity_mean "
          f"{np.mean(human[:, AV]):.2f}\n")

    out = {"n": args.n, "seed": args.seed, "ceiling": {}, "degeneracy": {}}

    print("A perfect model, and models with realistic error, under each decode")
    print(f"{'model error':<22}{'decode':<12}{'AUC':>8}{'n_dir_chg':>11}{'ang_vel':>9}")
    for label, sigma in [("perfect (real paths)", 0.0), ("very good (0.05)", 0.05),
                         ("good (0.10)", 0.10), ("P3 v1 level (0.25)", 0.25)]:
        for mode in ["grid_raw", "grid_round", "grid_snap", "event"]:
            m = evaluate(paths, sigma, mode, args.seed)
            out["ceiling"][f"{sigma}|{mode}"] = m
            print(f"{label:<22}{mode:<12}{m['auc_rf_oob']:>8.4f}"
                  f"{m['num_direction_changes']:>11.2f}"
                  f"{m['angular_velocity_mean']:>9.1f}", flush=True)
        print()

    print("Real paths, perturbed by amounts far too small to change the motion")
    print(f"{'perturbation':<22}{'decode':<12}{'AUC':>8}{'n_dir_chg':>11}")
    for jit in [0.0, 1e-9, 1e-6, 1e-3]:
        m = evaluate(paths, 0.0, "grid_raw", args.seed, jitter_px=jit)
        out["degeneracy"][f"{jit}px"] = m
        lbl = "none (exact)" if jit == 0 else f"{jit:.0e} px"
        print(f"{lbl:<22}{'grid_raw':<12}{m['auc_rf_oob']:>8.4f}"
              f"{m['num_direction_changes']:>11.2f}", flush=True)

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[ceiling] wrote {args.out}")


if __name__ == "__main__":
    main()
