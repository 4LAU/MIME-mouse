"""The endpoint correction is what makes the model wobble, not the sampler.

w3_turn_floor.py found the model turning 2.5x too much at 100 to 400 px/s and
matching the human exactly above 1000. w3_turn_by_class.py then put the raw
model output next to the corrected one, per speed class, and the story flipped:

  class 1 (1.0 px/event)   mean |turn| human 31.94, raw 26.57, corrected 37.72
  class 12-21 (1.5-2.0)    human 18.85, raw 14.53, corrected 26.14
  class 32 (3.0)           human 10.36, raw 10.47, corrected 18.07
  straight share, class 32 human 66.3%, raw 61.4%, corrected 44.6%

The sampler is at or below the human almost everywhere. correct_additive puts
the excess in. That also explains w3_stall_pattern.py's leftover: excess turn at
a hold is 0.4 degrees raw and 2.8 corrected.

The mechanism is rounding, not the correction's shape. correct_additive adds a
smooth drift w*err to every position and then rounds each position on its own.
Rounding a ramp is a staircase, and the staircase risers land wherever the ramp
happens to cross a half pixel, which is in the middle of straight runs. A human
1 px step run is a repeated identical displacement; one riser in the middle of
it is a 45 or 90 degree turn.

Any correction that must move the endpoint by err whole pixels has to put err
pixel jogs somewhere. The question is where, and the current answer is "spread
thinly over everything", which is the worst one. Two alternatives here:

  diffused   error diffusion along the path. Carry the fractional part forward
             and only spend it when it reaches a whole pixel, so a run of equal
             steps stays equal until one single step absorbs the jog. Ticks are
             skipped so holds are not converted into motion.
  large      the same, but the carry may only discharge on steps at or above
             LARGE px. A 1 px jog on a 40 px step bends it by 1.4 degrees; the
             same jog on a 1 px step bends it by 45 or more. Spend the error
             where it is angularly cheap.

Both keep the contract the correction exists for: the served path starts on the
requested pixel and ends on the requested pixel, exactly. That is asserted, not
assumed, and the assertion is what makes any AUC below comparable.

No GPU, no checkpoint touched. The arm is the landing cache.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_lattice_arrival.py
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
from w3_turn_by_class import gather  # noqa: E402

CACHE = R / "research" / "w3_landing_cache.pkl"
OUT = R / "research" / "w3_lattice_arrival_results.json"


def _diffuse(P, ts, sx, sy, ex, ey, large=0.0):
    """Error-diffused integer endpoint correction.

    Works in the step domain. Each step gets a share of the endpoint error in
    proportion to its own length, the same weighting correct_additive uses, but
    the fractional remainder is carried to the next step instead of being
    rounded away on the spot. A run of identical steps therefore stays identical
    until the carry reaches a whole pixel, and then exactly one step absorbs it.

    large > 0 additionally holds the carry until a step of at least that many
    pixels comes along, so the jog lands where it costs the least angle. The
    carry is always released on the final eligible step regardless, which is
    what keeps arrival exact.
    """
    d = np.diff(P, axis=0)
    n = np.hypot(d[:, 0], d[:, 1])
    tot = float(n.sum())
    start = np.array([sx, sy], dtype=np.float64)
    err = np.array([ex, ey], dtype=np.float64) - (start + d.sum(0))
    if tot < 1e-8 or len(d) == 0:
        return None

    share = err[None, :] * (n / tot)[:, None]
    # a step may absorb the carry if it moves at all, and if it is long enough
    # when a length floor is in force. Ticks never absorb: turning a hold into
    # a 1 px move would trade this defect for the stall defect.
    ok = n > 0 if large <= 0 else n >= large
    if not ok.any():
        ok = n > 0
    if not ok.any():
        return None

    out = np.empty_like(d, dtype=np.float64)
    carry = np.zeros(2)
    for i in range(len(d)):
        want = d[i] + share[i] + carry
        if not ok[i]:
            out[i] = d[i]
        else:
            out[i] = np.round(want)
        carry = want - out[i]

    Q = np.empty((len(P), 2), dtype=np.float64)
    Q[0] = start
    Q[1:] = start + np.cumsum(out, axis=0)
    # the diffusion is exact up to the final rounding; close any 1 px residue
    # on the longest eligible step rather than on the endpoint, so the served
    # path arrives without a visible last-moment jerk
    resid = np.array([ex, ey]) - Q[-1]
    if np.any(resid != 0):
        j = int(np.flatnonzero(ok)[np.argmax(n[ok])])
        Q[j + 1:] += resid
    return np.c_[Q, ts]


def correct_diffused(traj, sx, sy, ex, ey, large=0.0):
    P = np.asarray(traj[:, :2], dtype=np.float64)
    if len(P) < 3:
        return correct_additive(traj, sx, sy, ex, ey)
    out = _diffuse(np.round(P), traj[:, 2], sx, sy, ex, ey, large)
    return correct_additive(traj, sx, sy, ex, ey) if out is None else out


def arrival(paths, specs):
    """Share of served paths that start and end on the requested pixel."""
    hit = 0
    for p, s in zip(paths, specs):
        sx, sy, ex, ey = (int(v) for v in s)
        if (p[0, 0] == sx and p[0, 1] == sy
                and p[-1, 0] == ex and p[-1, 1] == ey):
            hit += 1
    return hit / max(len(paths), 1)


def score(paths, seed):
    X = features_with_jitter(paths, 0.0, seed)
    X = X[np.all(np.isfinite(X), axis=1)]
    return float(scoring.score_features(X)["auc_rf_oob"]), int(len(X))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--large", type=float, nargs="+",
                    default=[0.0, 4.0, 8.0, 16.0])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    # pickle.load: this repo's own artifact from the 2026-07-20 landing-price
    # run on this machine, never third-party input.
    with open(CACHE, "rb") as fh:
        specs, trajs = pickle.load(fh)
    keep = [(s, np.asarray(t)) for s, t in zip(specs, trajs) if len(t) >= 3]
    specs = [s for s, _ in keep]
    raw = [t for _, t in keep]
    real = [np.asarray(p, dtype=np.float64) for p in
            real_paths(args.n_real, args.seed, "ref")]
    print(f"[lattice] {len(raw)} arm paths, {len(real)} real paths", flush=True)

    arms = {"raw, no correction": raw,
            "additive (current)": [correct_additive(t, *(int(v) for v in s))
                                   for s, t in zip(specs, raw)]}
    for L in args.large:
        name = "diffused" if L <= 0 else f"diffused, jog on >= {L:g} px"
        arms[name] = [correct_diffused(t, *(int(v) for v in s), large=L)
                      for s, t in zip(specs, raw)]

    print(f"\n{'':<32}{'arrives':>10}{'n':>7}{'contract AUC':>15}")
    res = {}
    for name, paths in arms.items():
        arr = arrival(paths, specs)
        auc, n = score(paths, args.seed)
        res[name] = {"auc_rf_oob": auc, "n": n, "exact_arrival": arr}
        print(f"{name:<32}{arr:>10.1%}{n:>7}{auc:>15.4f}")

    base = res["additive (current)"]["auc_rf_oob"]
    print(f"\nagainst the current correction, arriving arms only")
    for name, r in res.items():
        if r["exact_arrival"] < 0.999 or name == "additive (current)":
            continue
        print(f"  {name:<30}{r['auc_rf_oob'] - base:+.4f}")

    # the turn statistics that sent us here, so a score move can be traced to
    # the mechanism rather than credited to luck
    print(f"\nmean |turn| deg by speed class, motion events")
    H = gather(real)
    cols = {"human": H}
    for name in ("raw, no correction", "additive (current)", "diffused"):
        cols[name] = gather(arms[name])
    print(f"{'class':<12}" + "".join(f"{k[:14]:>16}" for k in cols))
    bands = [(1, 1), (2, 11), (12, 21), (22, 24), (32, 32), (38, 40),
             (60, 60), (100, 128)]
    turn = {}
    for lo, hi in bands:
        row = {}
        cells = []
        for k, M in cols.items():
            m = (M[:, 0] >= lo) & (M[:, 0] <= hi)
            if m.sum() < 100:
                cells.append(f"{'-':>16}")
                continue
            v = float(M[m, 1].mean())
            z = float((M[m, 1] < 1e-9).mean())
            row[k] = {"mean_turn_deg": v, "straight_share": z,
                      "n": int(m.sum())}
            cells.append(f"{v:>10.2f}{z:>6.0%}")
        label = f"{lo}" if lo == hi else f"{lo} to {hi}"
        print(f"{label:<12}" + "".join(cells))
        turn[label] = row
    print("  (each cell is mean turn, then the share of perfectly straight "
          "continuations)")

    Path(args.out).write_text(json.dumps(
        {"seed": args.seed, "arms": res, "turn_by_class": turn,
         "wall_sec": time.time() - t0}, indent=2))
    print(f"\n[lattice] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
