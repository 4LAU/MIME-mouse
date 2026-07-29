"""Honest number for the K-filter fallback product: enforce exact arrival.

W0's fallback figure (0.539 at K=32, research/w0_sir_floor_results.json) was
measured on raw candidate paths, which land on the requested pixel only 0.3%
of the time. The product contract is exact arrival, and the landing-price run
(research/w3_landing_price_results.json) showed forcing arrival costs about
+0.078 one-shot. This script measures what the fallback pipeline reads when
every served path actually arrives.

Reuses, verbatim: the July 6 K=32 candidate pool (pool_s42_k32.npz, no fresh
generation), selection_lab.pick_sir (the production per-item judge, fit on ref
half A exactly as research/w0_sir_floor.py does), the magnitude-weighted
additive correction from the landing-price run (integer lattice, endpoints
rounded to whole pixels), and research/autoloop/scoring.score_features as the
only scorer.

Three arms per K in {8, 16, 32}, prefix-of-the-same-draw as in W0:
  raw_raw   select on raw features, score raw winners. Reproduces W0; the
            K=32 number should land near 0.539 or the harness is off.
  raw_corr  select on raw features, then correct the winner. The naive
            pipeline: the judge never sees what correction does to the path.
  corr_corr correct every candidate first, judge the corrected candidates.
            The honest product pipeline; selection gets to avoid paths that
            correction damages most.

Usage:
    env PYTHONPATH=. ~/venvs/mime/bin/python research/w3_fallback_arrival.py
    (optionally --pool other_pool.npz --out other_results.json to score a
    pool built by a different generator with the same machinery)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research" / "autoloop"))

import scoring  # noqa: E402  (metric contract; raises on human_eval paths)
from features import extract_features, resample_trajectory  # noqa: E402
from selection_lab import pick_sir  # noqa: E402

POOL_PATH = REPO_ROOT / "pool_s42_k32.npz"
REF_SIR_PATH = REPO_ROOT / "data" / "human_ref_features_sir.npy"
OUT_JSON = REPO_ROOT / "research" / "w3_fallback_arrival_results.json"
K_VALUES = [8, 16, 32]
SIR_TEMP = 0.7
SIR_SEED = 0


def correct_additive(traj, sx, sy, ex, ey):
    """Magnitude-weighted additive correction, identical to the landing-price
    run: spread the endpoint error along the path in proportion to per-step
    displacement, round to the integer lattice, pin both endpoints."""
    P = np.asarray(traj[:, :2], dtype=np.float64)
    ts = traj[:, 2]
    err = np.array([ex - P[-1, 0], ey - P[-1, 1]])
    step = np.r_[0.0, np.hypot(*np.diff(P, axis=0).T)]
    tot = step.sum()
    w = np.cumsum(step / tot) if tot > 1e-8 else np.linspace(0, 1, len(P))
    Q = P + np.outer(w, err)
    Q = np.round(Q)
    Q[0] = [sx, sy]
    Q[-1] = [ex, ey]
    return np.c_[Q, ts]


class SubPool:
    """The minimal view pick_sir needs (X, spec_rows) over an arbitrary
    feature matrix and per-spec candidate rows, restricted to the first K
    valid candidates per spec in original draw order."""

    def __init__(self, X, spec_rows, k):
        self.X = X
        self.spec_rows = {idx: rows[:k] for idx, rows in spec_rows.items()
                          if len(rows) > 0}


def main():
    global POOL_PATH, OUT_JSON
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(POOL_PATH))
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()
    POOL_PATH = Path(args.pool)
    OUT_JSON = Path(args.out)

    t_all = time.time()
    # allow_pickle: the pool npz is produced by this repo's own poolgen
    # (object-dtype trajs array), never third-party input.
    d = np.load(POOL_PATH, allow_pickle=True)
    specs = d["specs"]
    trajs = d["trajs"]
    X_raw = d["X"]
    owner = d["owner_idx"].astype(int)
    n_specs = len(specs)
    # A real request is for a whole pixel; round each spec's endpoint once and
    # correct every candidate of that spec to the same integer target.
    tgt = np.round(specs).astype(int)
    print(f"[fallback] pool {POOL_PATH.name}: {len(X_raw):,} candidates, "
          f"{n_specs} specs", flush=True)

    print("[fallback] correcting all candidates + re-extracting features...",
          flush=True)
    t0 = time.time()
    X_corr = np.full_like(X_raw, np.nan)
    miss_px = np.full(len(X_raw), np.nan)
    for ci in range(len(X_raw)):
        si = owner[ci]
        sx, sy, ex, ey = tgt[si]
        traj = trajs[ci]
        if traj is None or len(traj) < 3:
            continue
        miss_px[ci] = math.hypot(traj[-1][0] - ex, traj[-1][1] - ey)
        # the 125Hz resample is part of the feature convention: the pool's own
        # X matrix was built as extract_features(resample_trajectory(traj)),
        # and skipping it makes corrected paths detectable for the wrong
        # reason (v1 of this script did exactly that).
        f = extract_features(resample_trajectory(
            correct_additive(np.asarray(traj), sx, sy, ex, ey)))
        if f is not None and np.all(np.isfinite(f)):
            X_corr[ci] = f
    valid = np.all(np.isfinite(X_corr), axis=1)
    print(f"[fallback] corrected features valid for {valid.sum():,}/"
          f"{len(X_raw):,} candidates in {time.time()-t0:.0f}s; raw miss px "
          f"median={np.nanmedian(miss_px):.1f}", flush=True)

    # Compact both feature spaces to the shared valid rows: pick_sir fits its
    # judge on the WHOLE X matrix, so leftover NaN rows would crash it even
    # when spec_rows never references them.
    idx_valid = np.flatnonzero(valid)
    X_raw = X_raw[idx_valid]
    X_corr = X_corr[idx_valid]
    # spec_rows in original draw order over the compacted indices, so all
    # arms select from the same candidate set.
    spec_rows = {}
    for new_ci, ci in enumerate(idx_valid):
        spec_rows.setdefault(int(owner[ci]), []).append(new_ci)
    spec_rows = {idx: np.asarray(rows) for idx, rows in spec_rows.items()}

    ref = np.load(REF_SIR_PATH)
    perm = np.random.default_rng(0).permutation(len(ref))
    ref_a = ref[perm[:len(ref) // 2]]
    print(f"[fallback] judge ref: {REF_SIR_PATH.name} half A, "
          f"{len(ref_a)} rows", flush=True)

    results = {"pool": POOL_PATH.name, "correction": "additive_int",
               "sir_temp": SIR_TEMP, "sir_seed": SIR_SEED, "arms": {}}
    for k in K_VALUES:
        for arm, X_sel_space, X_score_space in (
                ("raw_raw", X_raw, X_raw),
                ("raw_corr", X_raw, X_corr),
                ("corr_corr", X_corr, X_corr)):
            t0 = time.time()
            picks = pick_sir(SubPool(X_sel_space, spec_rows, k), ref_a,
                             temp=SIR_TEMP, seed=SIR_SEED)
            rows = np.asarray(sorted(picks.values()))
            res = scoring.score_features(X_score_space[rows])
            results["arms"][f"K{k}_{arm}"] = {
                "auc": float(res["auc_rf_oob"]),
                "n_per_class": int(res["n_per_class"]),
                "n_specs_selected": len(picks),
                "wall_sec": time.time() - t0,
            }
            print(f"[fallback] K={k:2d} {arm:9s} "
                  f"AUC={res['auc_rf_oob']:.4f} "
                  f"(n={res['n_per_class']}/class, {time.time()-t0:.0f}s)",
                  flush=True)

    results["total_sec"] = time.time() - t_all
    with open(OUT_JSON, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"[fallback] wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
