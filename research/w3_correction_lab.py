"""Correction-scheme lab: can a smarter arrival correction cut the tax?

The product pays about +0.078 AUC for forcing exact arrival with the
magnitude-weighted additive correction. This compares that baseline against
two alternatives on the existing candidate pools, scored exactly like the
fallback product arm (correct every candidate, judge corrected, pick_sir
seeds 0/1/2, K=32):

  additive   spread the endpoint error along the whole path by step size
             (current product scheme, the baseline).
  tail       spread the error over the last 30 percent of the arc length
             only; humans do their homing near the target, so early-path
             dynamics stay untouched.
  similarity rotate and scale the whole path about the start point so the
             realized endpoint maps onto the target; preserves the path's
             shape and relative velocity profile.

All schemes round to the integer lattice and pin both endpoints, matching
the landing-price convention.
"""
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

import scoring
from features import extract_features, resample_trajectory
from selection_lab import pick_sir
from w3_fallback_arrival import correct_additive, SubPool


def correct_tail(traj, sx, sy, ex, ey, frac=0.3):
    P = np.asarray(traj[:, :2], dtype=np.float64)
    ts = traj[:, 2]
    err = np.array([ex - P[-1, 0], ey - P[-1, 1]])
    step = np.r_[0.0, np.hypot(*np.diff(P, axis=0).T)]
    tot = step.sum()
    if tot <= 1e-8:
        return correct_additive(traj, sx, sy, ex, ey)
    arc = np.cumsum(step) / tot
    w = np.clip((arc - (1.0 - frac)) / frac, 0.0, 1.0)
    Q = np.round(P + np.outer(w, err))
    Q[0] = [sx, sy]
    Q[-1] = [ex, ey]
    return np.c_[Q, ts]


def correct_similarity(traj, sx, sy, ex, ey):
    P = np.asarray(traj[:, :2], dtype=np.float64)
    ts = traj[:, 2]
    v = P[-1] - P[0]
    u = np.array([ex - sx, ey - sy], dtype=np.float64)
    nv, nu = np.hypot(*v), np.hypot(*u)
    # degenerate realized or requested displacement: similarity is undefined
    # or wildly amplifying, fall back to the additive scheme
    if nv < 4.0 or nu < 1e-8 or not (0.5 < nu / nv < 2.0):
        return correct_additive(traj, sx, sy, ex, ey)
    s = nu / nv
    ca, sa = (v[0] * u[0] + v[1] * u[1]) / (nv * nu), \
             (v[0] * u[1] - v[1] * u[0]) / (nv * nu)
    M = s * np.array([[ca, -sa], [sa, ca]])
    Q = np.round(np.array([sx, sy]) + (P - P[0]) @ M.T)
    Q[0] = [sx, sy]
    Q[-1] = [ex, ey]
    return np.c_[Q, ts]


SCHEMES = {"additive": correct_additive, "tail": correct_tail,
           "similarity": correct_similarity}


def main():
    ref = np.load(R / "data" / "human_ref_features_sir.npy")
    ref_a = ref[np.random.default_rng(0).permutation(len(ref))[:len(ref) // 2]]
    for pool_name in ("pool_s42_k32.npz", "pool_char_v3_cfg2_s42_k32.npz"):
        # allow_pickle: repo-own poolgen output (object-dtype trajs), not
        # third-party input.
        d = np.load(R / pool_name, allow_pickle=True)
        specs, trajs, owner = d["specs"], d["trajs"], d["owner_idx"].astype(int)
        tgt = np.round(specs).astype(int)
        for scheme, fn in SCHEMES.items():
            X = np.full_like(d["X"], np.nan)
            for ci in range(len(trajs)):
                sx, sy, ex, ey = tgt[owner[ci]]
                t = trajs[ci]
                if t is None or len(t) < 3:
                    continue
                f = extract_features(resample_trajectory(
                    fn(np.asarray(t), sx, sy, ex, ey)))
                if f is not None and np.all(np.isfinite(f)):
                    X[ci] = f
            valid = np.flatnonzero(np.all(np.isfinite(X), axis=1))
            Xv = X[valid]
            spec_rows = {}
            for new_ci, ci in enumerate(valid):
                spec_rows.setdefault(int(owner[ci]), []).append(new_ci)
            spec_rows = {i: np.asarray(r) for i, r in spec_rows.items()}
            aucs = []
            for seed in (0, 1, 2):
                picks = pick_sir(SubPool(Xv, spec_rows, 32), ref_a,
                                 temp=0.7, seed=seed)
                rows = np.asarray(sorted(picks.values()))
                aucs.append(scoring.score_features(Xv[rows])["auc_rf_oob"])
            print(f"[corrlab] {pool_name} {scheme:10s} K=32 corr_corr "
                  f"aucs={['%.4f' % a for a in aucs]} "
                  f"mean={np.mean(aucs):.4f} std={np.std(aucs):.4f} "
                  f"valid={len(valid)}", flush=True)


if __name__ == "__main__":
    main()
