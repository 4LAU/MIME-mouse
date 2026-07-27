"""What does exact arrival actually cost, and is it the operator or the aim?

w3_lattice_arrival.py measured the arm before and after the endpoint fix:

  raw, no correction    0.6500, arrives on the requested pixel 0.3% of the time
  additive (current)    0.7283, arrives 100%

So enforcing arrival costs 0.078, which is a third of the whole remaining gap to
0.50 and larger than any single lever this program has found. Two attempts to
build a gentler correction both scored worse, and one of them turned out to be
the current correction in disguise: carrying the fractional error forward and
rounding each step is arithmetically the same as rounding the running total.

That leaves two very different explanations, and they imply opposite work:

  operator   the correction damages any path it touches, so the lever is a
             better correction and the model is fine.
  aim        the correction is only as damaging as the error it has to absorb,
             so the lever is generation that lands closer and the correction is
             fine.

Both are measured here, and neither is inferred from the other.

  by error bucket   split the arm by how far its raw endpoint missed, and score
                    raw and corrected within each bucket. The raw column is the
                    control: if paths that miss badly were already worse paths,
                    raw rises across buckets too and nothing is attributable to
                    the correction.

  human injection   take real human paths, invent a target their own endpoint
                    misses by a chosen amount, and run the same correction on
                    them. No model anywhere. Whatever this costs a human path is
                    what the operator costs, and the rest of the arm's 0.078 is
                    aim.

The injected direction is uniform on the circle and the magnitude is fixed per
sweep point, so a bucket is comparable to the arm bucket at the same median
miss. Distances are in pixels, not fractions of path length, because that is the
unit the correction spreads in.

No GPU, no checkpoint touched.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_aiming_price.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

import scoring  # noqa: E402  (metric contract, imported never edited)
from degeneracy_panel import features_with_jitter, real_paths  # noqa: E402
from w3_fallback_arrival import correct_additive  # noqa: E402

CACHE = R / "research" / "w3_landing_cache.pkl"
OUT = R / "research" / "w3_aiming_price_results.json"


def correct_jog(traj, sx, sy, ex, ey):
    """Exact arrival by whole-pixel jogs, leaving every other step untouched.

    correct_additive adds a sub-pixel drift to every position and rounds each
    one, so a miss of any size dithers the whole path and breaks the straight
    lattice runs humans are full of. This instead spends the error as
    |err_x| + |err_y| single-pixel changes, one per chosen step, longest steps
    first, where one pixel bends the least angle. Every step that is not chosen
    comes through byte identical to the model's own.

    At the arm's median 58 px miss over ~57 events this cannot help: nearly
    every step has to move a pixel either way. It is aimed at the regime where
    the miss is small, which is the regime better aiming would put us in.

    Arrival is exact by construction: the steps are integers and they are made
    to sum to the requested displacement.
    """
    P = np.round(np.asarray(traj[:, :2], dtype=np.float64))
    if len(P) < 3:
        return correct_additive(traj, sx, sy, ex, ey)
    d = np.diff(P, axis=0)
    start = np.array([sx, sy], dtype=np.float64)
    err = np.array([ex, ey], dtype=np.float64) - (start + d.sum(0))
    mag = np.hypot(d[:, 0], d[:, 1])
    idx = np.flatnonzero(mag > 0)
    if len(idx) == 0:
        return correct_additive(traj, sx, sy, ex, ey)
    order = idx[np.argsort(-mag[idx])]
    for axis in (0, 1):
        e = int(round(err[axis]))
        if e == 0:
            continue
        s = 1.0 if e > 0 else -1.0
        for k in range(abs(e)):
            d[order[k % len(order)], axis] += s
    Q = np.empty_like(P)
    Q[0] = start
    Q[1:] = start + np.cumsum(d, axis=0)
    return np.c_[Q, traj[:, 2]]


def score(paths, seed):
    X = features_with_jitter(paths, 0.0, seed)
    X = X[np.all(np.isfinite(X), axis=1)]
    if len(X) < 50:
        return float("nan"), int(len(X))
    return float(scoring.score_features(X)["auc_rf_oob"]), int(len(X))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--buckets", type=int, default=6)
    ap.add_argument("--inject", type=float, nargs="+",
                    default=[0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    # pickle.load: this repo's own artifact from the 2026-07-20 landing-price
    # run on this machine, never third-party input.
    with open(CACHE, "rb") as fh:
        specs, trajs = pickle.load(fh)
    keep = [(np.asarray(s), np.asarray(t)) for s, t in zip(specs, trajs)
            if len(t) >= 3]
    specs = [s for s, _ in keep]
    raw = [t for _, t in keep]
    corr = [correct_additive(t, *(int(v) for v in s))
            for s, t in zip(specs, raw)]
    real = [np.asarray(p, dtype=np.float64) for p in
            real_paths(args.n_real, args.seed, "ref")]
    print(f"[aim] {len(raw)} arm paths, {len(real)} real paths", flush=True)

    miss = np.array([float(np.hypot(int(s[2]) - t[-1, 0],
                                    int(s[3]) - t[-1, 1]))
                     for s, t in zip(specs, raw)])
    span = np.array([float(np.hypot(int(s[2]) - int(s[0]),
                                    int(s[3]) - int(s[1]))) for s in specs])
    print(f"[aim] endpoint miss px: p10 {np.percentile(miss, 10):.1f}  "
          f"p50 {np.percentile(miss, 50):.1f}  p90 {np.percentile(miss, 90):.1f}"
          f"  max {miss.max():.0f}")
    print(f"[aim] miss as share of requested distance: p50 "
          f"{np.median(miss / np.maximum(span, 1)):.1%}")

    order = np.argsort(miss)
    chunks = np.array_split(order, args.buckets)
    print(f"\narm split by how far the raw endpoint missed")
    print(f"{'miss px':<18}{'n':>6}{'raw AUC':>10}{'corrected':>12}"
          f"{'correction cost':>18}")
    rows = []
    for ch in chunks:
        m = miss[ch]
        r_auc, _ = score([raw[i] for i in ch], args.seed)
        c_auc, n = score([corr[i] for i in ch], args.seed)
        print(f"{f'{m.min():.0f} to {m.max():.0f}':<18}{n:>6}{r_auc:>10.4f}"
              f"{c_auc:>12.4f}{c_auc - r_auc:>18.4f}")
        rows.append({"miss_lo": float(m.min()), "miss_hi": float(m.max()),
                     "miss_median": float(np.median(m)), "n": n,
                     "raw_auc": r_auc, "corrected_auc": c_auc,
                     "cost": c_auc - r_auc})

    # the operator on its own: real paths, invented targets, same correction
    rng = np.random.default_rng(args.seed)
    ang = rng.uniform(0, 2 * np.pi, len(real))
    print(f"\nthe same correction applied to real human paths, and a jog "
          f"correction beside it")
    print(f"{'injected miss px':<20}{'n':>6}{'additive':>11}{'jog':>10}"
          f"{'jog saves':>12}")
    inj = []
    base = None
    for d in args.inject:
        out_a, out_j = [], []
        for p, a in zip(real, ang):
            sx, sy = int(round(p[0, 0])), int(round(p[0, 1]))
            ex = int(round(p[-1, 0] + d * np.cos(a)))
            ey = int(round(p[-1, 1] + d * np.sin(a)))
            out_a.append(correct_additive(p, sx, sy, ex, ey))
            out_j.append(correct_jog(p, sx, sy, ex, ey))
        auc, n = score(out_a, args.seed)
        aucj, _ = score(out_j, args.seed)
        if base is None:
            base = auc
        print(f"{d:<20.0f}{n:>6}{auc:>11.4f}{aucj:>10.4f}{auc - aucj:>12.4f}")
        inj.append({"miss_px": d, "n": n, "auc_additive": auc,
                    "auc_jog": aucj, "damage": auc - base,
                    "jog_saves": auc - aucj})

    # the jog operator on the real arm, where the miss is not small
    jog = [correct_jog(t, *(int(v) for v in s)) for s, t in zip(specs, raw)]
    ja, jn = score(jog, args.seed)
    ca, _ = score(corr, args.seed)
    print(f"\non the arm itself: additive {ca:.4f}, jog {ja:.4f} "
          f"({ja - ca:+.4f}), n {jn}")

    print(f"\nread: the arm misses by {np.median(miss):.0f} px median over "
          f"~57 events, so about a pixel per event. No operator is gentle at "
          f"that size. The injection curve is what the operator costs once the "
          f"aim is fixed.")

    Path(args.out).write_text(json.dumps(
        {"seed": args.seed, "miss_px": {"p10": float(np.percentile(miss, 10)),
                                        "p50": float(np.median(miss)),
                                        "p90": float(np.percentile(miss, 90))},
         "arm_buckets": rows, "human_injection": inj,
         "wall_sec": time.time() - t0}, indent=2))
    print(f"\n[aim] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
