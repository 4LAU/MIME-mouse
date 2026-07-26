"""Why did P3 v1 read 0.98? Attribute it to representation and decode.

P3 v1 generated normalized absolute coordinates on the uniform 125 Hz grid
(start at origin, endpoint at unit distance), rounded the result to whole
pixels, and read angular_velocity_mean 98.7 against a human 22.7 with an AUC
of 0.9815. Two suspects, neither of which is the inpainting design:

  DECODE. The grid IS the resampled representation: prepare_training_data.py
  builds it by calling features.resample_trajectory, so grid coordinates are
  interpolated and fractional, and the feature extractor's own resample is a
  no-op on them. Rounding that smooth off-lattice path alternates lattice
  directions nearly every slow step. EXPERIMENTS.md already recorded this
  exact effect in the event-stream family and fixed it with EVENT_SNAP=2.5,
  worth 0.791 -> 0.755 there. P3 never applied any equivalent.

  REPRESENTATION. A step is 0.035 in endpoint-normalized units while the
  model works at unit scale, so per-slot error has to stay far below 0.035
  before consecutive step directions carry signal. Modelling per-step
  increments instead puts the quantity being predicted at scale 1.

This probe injects white noise into REAL human validation paths under each
representation and each decode contract, then scores them with the standing
contract scorer. No trained model is involved, so whatever it finds belongs to
the pipeline rather than to anything P3 learned. The clean rows (noise 0) are
the floor any perfect model would hit.

Usage:
  env PYTHONPATH=. MIME_GRID_DIR=~/mime_data \
    ~/venvs/mime/bin/python research/p3_repr_probe.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

import scoring
from features import FEATURE_NAMES, extract_feature_matrix

HZ = 125.0
DATA_DIR = Path(os.environ.get("MIME_GRID_DIR", R / "training"))
AV = FEATURE_NAMES.index("angular_velocity_mean")


def load_paths(n_paths, seed):
    """Real val-split paths on the 192-slot grid, with their pixel distance."""
    pos = np.load(DATA_DIR / "val_positions.npy", mmap_mode="r")
    n_real = np.load(DATA_DIR / "val_n_real.npy")
    cond = np.load(DATA_DIR / "val_conditions.npy", mmap_mode="r")
    valid = np.flatnonzero((n_real >= 8) & (n_real < 192))
    rng = np.random.default_rng(seed)
    pick = rng.choice(valid, size=min(n_paths, len(valid)), replace=False)
    out = []
    for i in pick:
        n = int(n_real[i])
        out.append((np.array(pos[i][:n], dtype=np.float64),
                    float(np.exp(cond[i][0]))))
    return out


def perturb_absolute(p, sigma, rng):
    """v1 representation: white noise on absolute normalized coordinates."""
    q = p.copy()
    q[1:-1] += rng.normal(0.0, sigma, size=(len(p) - 2, 2))
    return q


def perturb_increment(p, sigma, rng):
    """v2 proposal: white noise on endpoint-normalized increments.

    d_i are the per-step increments, E their sum (the unit endpoint vector),
    k the step count. e_i = k*d_i - E has zero sum and magnitude near 1, so
    noise at scale sigma is sigma of a typical step rather than sigma of the
    whole path. Re-centring after the noise keeps the sum at E, which is what
    makes exact arrival free.
    """
    d = np.diff(p, axis=0)
    k = len(d)
    E = d.sum(axis=0)
    e = k * d - E + rng.normal(0.0, sigma, size=(k, 2))
    e -= e.mean(axis=0)
    return np.vstack([p[0], p[0] + np.cumsum((e + E) / k, axis=0)])


def decode(p, dist, mode, start=(500.0, 500.0)):
    """mode: raw (fractional, matches the reference's own treatment),
    round (what v1 did), snap (integer but lattice-aligned at slow steps,
    the EXPERIMENTS.md decode contract ported to the grid)."""
    px = np.array(start) + p * dist
    if mode == "round":
        px = np.round(px)
    elif mode == "snap":
        d = np.diff(px, axis=0)
        s = np.hypot(d[:, 0], d[:, 1])
        slow = s < 2.5
        d[slow] = np.round(d[slow])
        px = np.vstack([np.round(px[0]), np.round(px[0]) + np.cumsum(d, axis=0)])
        px[~np.isfinite(px)] = 0.0
    return np.c_[px, np.arange(len(p)) / HZ]


def run(paths, perturb, sigma, mode, seed):
    rng = np.random.default_rng(seed)
    trajs = [decode(p if perturb is None else perturb(p, sigma, rng), dist, mode)
             for p, dist in paths]
    X = extract_feature_matrix(trajs)
    return float(np.mean(X[:, AV])), scoring.score_features(X)["auc_rf_oob"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths = load_paths(args.n, args.seed)
    step = np.mean([np.linalg.norm(np.diff(p, axis=0), axis=1).mean()
                    for p, _ in paths])
    human = np.load(R / "data" / "human_val_features_grpo.npy")
    hm = float(np.mean(human[:, AV]))
    print(f"[probe] {len(paths)} real val paths, mean step {step:.4f} "
          f"in endpoint-normalized units")
    print(f"[probe] human angular_velocity_mean {hm:.1f}; "
          f"P3 v1 read 98.7 at AUC 0.9815\n")

    print(f"{'representation':<14}{'decode':<8}{'noise':>8}{'ang_vel':>10}{'AUC':>9}")

    def row(label, perturb, sigma, mode):
        av, auc = run(paths, perturb, sigma, mode, args.seed)
        print(f"{label:<14}{mode:<8}{sigma if sigma else 0:>8.3f}"
              f"{av:>10.1f}{auc:>9.4f}", flush=True)

    # Floor: a perfect model, i.e. real paths, under each decode contract.
    for mode in ["raw", "round", "snap"]:
        row("perfect", None, 0.0, mode)

    # v1's regime. Its measured angular velocity of 98.7 sits near sigma 0.003.
    for mode in ["round", "raw"]:
        for sigma in [0.002, 0.005]:
            row("absolute", perturb_absolute, sigma, mode)

    # v2's regime, at the same and worse relative accuracy per step.
    for mode in ["raw", "snap", "round"]:
        for sigma in [0.1, 0.2]:
            row("increment", perturb_increment, sigma, mode)


if __name__ == "__main__":
    main()
