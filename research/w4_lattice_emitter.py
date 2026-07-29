"""Can a lattice-native emitter rescue a smooth curve?

The live hypothesis after 2026-07-27: the continuous whole-path families here
(DDPM 0.9291, CFM 0.9191, v136 0.9899) may have been convicted on how their
output was put on the pixel grid rather than on their architecture. Every one of
them emits positions between pixels and rounds them, and rounding is expensive:
w4_texture_sensitivity measures real human data plus one pixel of independent
noise at 0.950.

The negative half of that claim is measured. This is the positive half. Take a
smooth curve that is already very close to a real path, a 12 submovement plan at
0.47 px median error reading 0.712, and instead of rounding each sample
independently, CHOOSE the sequence of whole pixel steps.

The channel diagnostic says what to choose for. On the resampled signal the
plan already matches real almost exactly on the things that dominate the
turn-rate features:

  arm                 zero-turn  straight-run  median turn   dwell  step
  real                    0.359          2.42        7.12d   0.102  2.24
  plan N=12, rounded      0.360          2.41        5.37d   0.119  2.24
  real + 0.5 px           0.214          0.85       11.96d   0.049  2.24

So the collinear structure survives rounding intact and is not the defect. The
one visible gap is turn MAGNITUDE: where a human turns 7.12 degrees the rounded
plan turns 5.37. It is too smooth exactly where the forest is looking.

The emitter targets that and only that. At each sample it holds the accumulated
tracking error, forms the candidate integer steps around the error corrected
ideal step, samples a target turn from the human distribution conditioned on
step length, and takes the candidate that best trades turning against tracking.
The error feedback means a step taken to make a turn is repaid on the next
samples, so the path still lands where the plan says.

  rounded (base)       independent rounding, the 0.712 being explained.
  emitted, human turns the arm under test.
  emitted, own turns   the same emitter with target turns drawn from the
                       ROUNDED PLAN's own distribution. Everything that is
                       emitter machinery and not human turn statistics shows up
                       here. If this moves as much as the arm above, the result
                       is the machinery and not the hypothesis.
  emitted, w=0         tracking only, no turn target. Second null.
  real (floor)         the floor.

No generation, no GPU, no checkpoint read or written.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w4_lattice_emitter.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

OUT = R / "research" / "w4_lattice_emitter_results.json"
NSUB = 12
# step length buckets the turn distribution is conditioned on. Turn size and
# step size are strongly related: a one pixel step can only turn in coarse
# increments, a twenty pixel step turns finely.
EDGES = np.array([0.0, 1.5, 3.0, 6.0, 12.0, 25.0, 1e9])


def turn_table(paths, rng_unused=None):
    """Empirical |turn| samples per step-length bucket, from raw whole-pixel
    steps at the recordings' own timestamps, which is where the emitter works."""
    tab = [[] for _ in range(len(EDGES) - 1)]
    for p in paths:
        d = np.diff(np.asarray(p)[:, :2], axis=0)
        m = np.hypot(*d.T)
        ok = m > 0
        if ok.sum() < 3:
            continue
        d, m = d[ok], m[ok]
        a = np.arctan2(d[:, 1], d[:, 0])
        t = np.abs((np.diff(a) + np.pi) % (2 * np.pi) - np.pi)
        b = np.clip(np.digitize(m[:-1], EDGES) - 1, 0, len(tab) - 1)
        for bi, ti in zip(b, t):
            tab[bi].append(ti)
    return [np.array(v) if len(v) else np.array([0.0]) for v in tab]


def emit(C, tab, rng, w_turn=1.0, radius_cap=3):
    """Whole-pixel steps that track C and turn like the table says."""
    out = np.empty_like(C)
    cur = np.rint(C[0])
    out[0] = cur
    e = C[0] - cur
    prev = None
    for k in range(1, len(C)):
        want = C[k] - C[k - 1] + e
        base = np.rint(want)
        mag = float(np.hypot(*want))
        if w_turn <= 0 or prev is None or mag < 1e-9:
            s = base
        else:
            rad = int(min(max(1, round(0.25 * mag)), radius_cap))
            off = np.arange(-rad, rad + 1)
            cand = base + np.stack(np.meshgrid(off, off, indexing="ij"),
                                   -1).reshape(-1, 2)
            cm = np.hypot(*cand.T)
            live = cm > 0
            if not live.any():
                s = base
            else:
                cand, cm = cand[live], cm[live]
                b = int(np.clip(np.digitize([mag], EDGES)[0] - 1, 0, len(tab) - 1))
                pool = tab[b]
                tgt = float(rng.choice(pool)) if len(pool) else 0.0
                ca = np.arctan2(cand[:, 1], cand[:, 0])
                turn = np.abs((ca - prev + np.pi) % (2 * np.pi) - np.pi)
                track = np.hypot(*(want - cand).T)
                cost = w_turn * np.abs(turn - tgt) + track / max(mag, 1.0)
                s = cand[int(np.argmin(cost))]
        cur = cur + s
        out[k] = cur
        e = want - s
        if np.hypot(*s) > 0:
            prev = float(np.arctan2(s[1], s[0]))
    return out


def _fit(p):
    from w4_submovement_ceiling import fit_one, plan_xy
    try:
        params, rms = fit_one(p, NSUB)
        return plan_xy(p, params), float(rms)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-paths", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--w-turn", type=float, nargs="+", default=[0.5, 1.0, 3.0])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t_start = time.time()
    from degeneracy_panel import features_with_jitter, real_paths  # noqa: E402
    import scoring  # noqa: E402
    from w3_raw_column_reread import subset_auc  # noqa: E402

    rng = np.random.default_rng(args.seed)
    ref = [np.asarray(p) for p in real_paths(args.n_paths, args.seed, "ref")
           if len(p) >= 8]
    hold = real_paths(args.n_paths, args.seed, "holdout")
    cols = list(range(18))
    Xh = features_with_jitter(hold, 0.0, args.seed)
    Xh = Xh[np.all(np.isfinite(Xh), axis=1)]

    def score(paths):
        X = features_with_jitter(paths, 0.0, args.seed)
        X = X[np.all(np.isfinite(X), axis=1)]
        return (float(subset_auc(X, Xh, cols)),
                float(scoring.score_features(X)["auc_rf_oob"]))

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        got = list(ex.map(_fit, ref, chunksize=8))
    keep = [(p, g[0]) for p, g in zip(ref, got) if g is not None]
    rms = [g[1] for g in got if g is not None]
    print(f"[emit] {len(keep)} plans at N={NSUB}, fit rms median "
          f"{np.median(rms):.2f} px", flush=True)

    # the human turn table comes from the fit half, and the arms are scored
    # against the disjoint holdout half, so nothing is fitted to its own judge
    tab_h = turn_table([p for p, _ in keep])
    rounded = [np.c_[np.rint(q), p[:, 2]] for p, q in keep]
    tab_own = turn_table(rounded)
    print("[emit] median |turn| per step bucket, degrees")
    print("       human  " + "  ".join(f"{np.degrees(np.median(v)):6.2f}"
                                       for v in tab_h))
    print("       plan   " + "  ".join(f"{np.degrees(np.median(v)):6.2f}"
                                       for v in tab_own), flush=True)

    arms = {"real (floor)": [p for p, _ in keep], "rounded (base)": rounded}
    for w in args.w_turn:
        arms[f"emitted, human turns w={w}"] = [
            np.c_[emit(q, tab_h, rng, w), p[:, 2]] for p, q in keep]
    arms["emitted, own turns w=1.0"] = [
        np.c_[emit(q, tab_own, rng, 1.0), p[:, 2]] for p, q in keep]
    arms["emitted, w=0"] = [np.c_[emit(q, tab_h, rng, 0.0), p[:, 2]]
                            for p, q in keep]

    out = {"n_paths": len(keep), "n_sub": NSUB, "seed": args.seed, "arms": {}}
    print(f"\n{'arm':<30}{'internal':>10}{'contract':>10}")
    for k, v in arms.items():
        a, b = score(v)
        out["arms"][k] = {"auc_internal": a, "auc_contract": b}
        print(f"{k:<30}{a:>10.4f}{b:>10.4f}", flush=True)

    out["wall_sec"] = time.time() - t_start
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[emit] wrote {args.out} ({out['wall_sec']:.0f}s)")


if __name__ == "__main__":
    main()
