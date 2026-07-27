"""What kinds of human movement does the model never produce?

research/w3_density_ceiling.py found a floor: reweighting how often each
candidate gets emitted, without changing what the model can produce, reaches
0.6018 and stops. About 0.10 of the gap is paths that are simply absent from
tens of thousands of the model's own samples. This asks what they look like.

Method. Fit the contract scorer's RF on real paths against the arm, then read
each real path's own out-of-bag probability. A real path the forest confidently
calls real sits somewhere the model puts nothing; a real path it cannot place
sits inside the model's range. Split the real paths on that, then describe the
two groups in terms anyone can check against a recording: how far the pointer
travelled against how far it needed to, how long it paused, whether it reversed,
how fast it peaked. Cluster the uncovered group so the answer is "these three
kinds of movement" rather than one averaged blur.

The number that matters is at the end: the arm scored against covered real paths
alone, against uncovered real paths alone. If the uncovered group is where the
score lives, the missing paths are the problem and this names them.

Descriptors are computed from the resampled path, never from the 18 features, so
they are an independent read rather than a restatement of what the forest saw.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_missing_paths.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

import scoring  # noqa: E402  (metric contract, imported never edited)
from degeneracy_panel import (_score_against, features_with_jitter,  # noqa: E402
                              real_paths)
from features import FEATURE_NAMES, resample_trajectory  # noqa: E402
from w3_fallback_arrival import correct_additive  # noqa: E402

CACHE = R / "research" / "w3_landing_cache.pkl"
OUT = R / "research" / "w3_missing_paths_results.json"
HZ = 125.0
PAUSE_PX_S = 20.0        # below this the pointer is not meaningfully moving
DESCRIPTORS = ["detour_ratio", "paused_fraction", "n_pauses", "reversals",
               "overshoot", "peak_speed", "duration_s", "straight_dist_px"]


def describe(traj):
    """Interpretable descriptors of one path, from the path itself."""
    p = np.asarray(resample_trajectory(traj, hz=HZ), dtype=np.float64)
    if len(p) < 5:
        return None
    x, y, t = p[:, 0], p[:, 1], p[:, 2]
    dt = np.diff(t)
    dt[dt <= 0] = 1.0 / HZ
    step = np.hypot(np.diff(x), np.diff(y))
    speed = step / dt
    travelled = float(step.sum())
    sx, sy, ex, ey = x[0], y[0], x[-1], y[-1]
    straight = float(np.hypot(ex - sx, ey - sy))
    if straight < 1e-9:
        return None

    slow = speed < PAUSE_PX_S
    # a pause is a run of slow samples, not a single slow sample
    edges = np.diff(np.concatenate([[0], slow.astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    runs = ends - starts
    n_pauses = int(np.sum(runs >= 3))

    ux, uy = (ex - sx) / straight, (ey - sy) / straight
    along = (x - sx) * ux + (y - sy) * uy          # progress toward the target
    prog = np.diff(along)
    moving = np.abs(prog) > 0.05
    reversals = int(np.sum(np.diff(np.sign(prog[moving])) != 0)) if moving.sum() > 1 else 0

    return {"detour_ratio": travelled / straight,
            "paused_fraction": float(slow.mean()),
            "n_pauses": float(n_pauses),
            "reversals": float(reversals),
            "overshoot": float(max(along.max() - straight, 0.0) / straight),
            "peak_speed": float(np.percentile(speed, 99)),
            "duration_s": float(t[-1] - t[0]),
            "straight_dist_px": straight}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", type=float, default=0.25,
                    help="fraction of real paths counted as most/least covered")
    ap.add_argument("--clusters", type=int, default=4)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    # pickle.load: this repo's own artifact from the 2026-07-20 landing-price
    # run on this machine, never third-party input.
    with open(CACHE, "rb") as fh:
        specs, trajs = pickle.load(fh)
    arm = [correct_additive(np.asarray(t), *(int(v) for v in s)) if len(t) >= 3
           else np.asarray(t) for s, t in zip(specs, trajs)]
    real = real_paths(args.n_real, args.seed, "ref")

    Xa = features_with_jitter(arm, 0.0, args.seed)
    Xa = Xa[np.all(np.isfinite(Xa), axis=1)]
    Xr = features_with_jitter(real, 0.0, args.seed)
    keep = np.all(np.isfinite(Xr), axis=1)
    Xr, real = Xr[keep], [p for p, k in zip(real, keep) if k]
    n = min(len(Xa), len(Xr))
    print(f"[missing] {len(Xr)} real paths, {len(Xa)} arm paths, "
          f"balanced to {n} per class")

    clf = RandomForestClassifier(n_estimators=scoring.RF_N_ESTIMATORS,
                                 oob_score=True, n_jobs=-1,
                                 random_state=scoring.RF_SEED)
    clf.fit(np.vstack([Xr[:n], Xa[:n]]),
            np.concatenate([np.zeros(n), np.ones(n)]))
    # p_real for the real rows: high means the forest places it far from
    # anything the model produced
    p_real = 1.0 - clf.oob_decision_function_[:n, 1]

    k = max(int(args.split * n), 10)
    order = np.argsort(-p_real)
    uncovered, covered = order[:k], order[-k:]
    print(f"[missing] {k} least covered (p_real {p_real[uncovered].min():.2f} to "
          f"{p_real[uncovered].max():.2f}), {k} most covered "
          f"({p_real[covered].min():.2f} to {p_real[covered].max():.2f})")

    desc = [describe(real[i]) for i in range(n)]
    ok = np.array([d is not None for d in desc])
    D = np.array([[d[c] for c in DESCRIPTORS] if d is not None
                  else [np.nan] * len(DESCRIPTORS) for d in desc])

    def group(idx):
        m = idx[ok[idx]]
        return {c: float(np.median(D[m, j])) for j, c in enumerate(DESCRIPTORS)}

    g_un, g_cov = group(uncovered), group(covered)
    print(f"\ntypical values, median over each group of {k} real paths")
    print(f"{'descriptor':<20}{'model covers it':>17}{'model misses it':>17}"
          f"{'ratio':>8}")
    for c in DESCRIPTORS:
        a, b = g_cov[c], g_un[c]
        print(f"{c:<20}{a:>17.3f}{b:>17.3f}"
              f"{(b / a if abs(a) > 1e-9 else float('nan')):>8.2f}")

    # what the arm scores against each group on its own
    aucs = {}
    for name, idx in (("covered real paths", covered),
                      ("uncovered real paths", uncovered)):
        aucs[name] = float(_score_against(Xa[:len(idx)], Xr[idx])["auc_rf_oob"])
    aucs["all real paths"] = float(_score_against(Xa[:n], Xr[:n])["auc_rf_oob"])
    print(f"\nthe arm scored against each group on its own")
    for kk in ("all real paths", "covered real paths", "uncovered real paths"):
        print(f"  {kk:<24}{aucs[kk]:.4f}")

    # cluster the uncovered group so the answer is kinds, not an average
    m = uncovered[ok[uncovered]]
    Z = D[m]
    Z = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-12)
    km = KMeans(n_clusters=args.clusters, n_init=10,
                random_state=args.seed).fit(Z)
    clusters = []
    print(f"\nthe missing paths, grouped into {args.clusters} kinds")
    for c in range(args.clusters):
        sel = m[km.labels_ == c]
        if len(sel) == 0:
            continue
        med = {n_: float(np.median(D[sel, j])) for j, n_ in enumerate(DESCRIPTORS)}
        clusters.append({"n": int(len(sel)), "share": len(sel) / len(m),
                         "median": med})
        print(f"  kind {c + 1}: {len(sel):4d} paths ({len(sel)/len(m):5.1%})  "
              f"detour {med['detour_ratio']:.2f}x  "
              f"paused {med['paused_fraction']:.0%}  "
              f"{med['n_pauses']:.0f} pauses  "
              f"{med['reversals']:.0f} reversals  "
              f"overshoot {med['overshoot']:.2f}  "
              f"{med['duration_s']:.2f}s  {med['straight_dist_px']:.0f}px")

    out = {"n_per_class": int(n), "k": int(k), "seed": args.seed,
           "auc": aucs, "median_covered": g_cov, "median_uncovered": g_un,
           "clusters": clusters, "wall_sec": time.time() - t0}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[missing] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
