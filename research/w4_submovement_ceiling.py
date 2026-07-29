"""Does a compact motor plan preserve what the detector sees?

Every adjacent field that solved "generated sequences look human until a
classifier reads them" stopped generating the signal and started generating a
plan, then rendering it. Handwriting and signature synthesis do it with
Plamondon's Kinematic Theory, where a movement is the vector sum of a few
overlapping ballistic strokes. The motor control literature says the same thing
about aimed pointing specifically: a primary ballistic submovement plus a small
number of corrective ones, each a bell shaped minimum jerk velocity profile,
superposed. The published mouse work puts the count at 2 to 6 for most trials.

Our trunk emits one small increment at a time and never commits to a whole
movement, which is the shape of yesterday's result: error diffuse everywhere,
and repairing any single relationship makes the output MORE detectable. That is
what a model with no plan looks like. Properties that in a person are all
consequences of one commitment are, in the trunk, independent accidents.

Before building any of that, one number decides it.

Fit plans to real human paths. Re-render the plans. Score the re-rendered paths
against a disjoint half of real humans. That is the CEILING of the whole
architecture. Nothing built on this representation can score better than a plan
fitted to the answer. If it lands near the floor at a small stroke count then
the parameterisation keeps whatever the forest is reading and all that remains
is learning the distribution over plans. If it lands high, the representation
throws the signal away and the idea is dead for free.

The model, and why this form.

  Submovement i is a straight minimum jerk displacement a_i (a 2-vector) begun
  at t0_i over duration T_i. Position is

      X(t) = X_0 + sum_i a_i * S((t - t0_i) / T_i),
      S(s) = 10 s^3 - 15 s^4 + 6 s^5 clamped to [0, 1],

  which is the standard minimum jerk profile: zero velocity, acceleration and
  jerk at both ends, one bell shaped speed hump. Four numbers per submovement.

  Curvature is NOT a parameter. It emerges when two submovements pointing in
  different directions overlap in time, which is also where the two thirds
  power law comes from. That is the point of the representation: the coupling
  between speed and bend is a consequence of the plan rather than something a
  model has to learn to correlate.

  X is LINEAR in the a_i, so for any timing the amplitudes solve in closed form
  and only the 2N timing numbers need searching. Variable projection, and it is
  what makes a 2000 path sweep a CPU job instead of a GPU one.

Rendering holds the observation layer fixed on purpose. The plan is evaluated
at the SAME irregular timestamps the human path was sampled at and rounded to
the same integer pixel lattice. Half the contract's features are acceleration
and jerk, which are dominated by sampling and quantisation, so leaving those
identical is what isolates the question actually being asked: does the compact
plan keep the signal, independent of how the signal gets sampled.

Reading the sweep. As N grows the fit approaches the path itself, so AUC has to
fall to the floor eventually. That is the built in control, not a result: if it
did NOT fall, the residual would be the render-and-round pipeline rather than
the parameterisation, and the whole table would be unreadable. The result is
the SHAPE. The number that matters is the smallest N whose AUC reaches the
floor, against a path that carries about 2K numbers at K samples.

  N = 1   the straw man. One smooth ideal stroke, no corrections.
  N = 2   primary plus one correction, the classic Fitts law account.
  N = 3-6 what the mouse submovement literature reports for most trials.
  N = 12  deliberately past the compact regime, to see the floor arrive.

Fitting is done on the "ref" human half and scored against the disjoint
"holdout" half, so no path is ever compared against a plan fitted to itself.

No generation, no GPU, no checkpoint read or written.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w4_submovement_ceiling.py --n-paths 2000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

OUT = R / "research" / "w4_submovement_ceiling_results.json"
NSWEEP = [1, 2, 3, 4, 6, 8, 12]


def minjerk(s):
    """Minimum jerk position profile on [0, 1], clamped outside."""
    s = np.clip(s, 0.0, 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def design(t, t0, T):
    """K by N matrix of each submovement's completed fraction at each sample."""
    return minjerk((t[:, None] - t0[None, :]) / T[None, :])


def solve_amps(M, XY):
    """Closed form amplitudes for a given timing. The linear half of varpro."""
    a, *_ = np.linalg.lstsq(M, XY, rcond=None)
    return a


def _pack(t0, T):
    return np.concatenate([t0, np.log(T)])


def _unpack(p, n):
    return p[:n], np.exp(p[n:])


def fit_one(xyt, n, span_pad=0.35):
    """Fit n minimum jerk submovements to one path. Returns (params, rms)."""
    t = xyt[:, 2] - xyt[0, 2]
    XY = xyt[:, :2] - xyt[0, :2]
    span = max(float(t[-1]), 1e-3)

    # initialise from the speed profile's own humps, which is where the
    # submovement literature says the corrections live. Falls back to an even
    # split when the profile has fewer humps than submovements asked for.
    # the recordings carry repeated timestamps, which make np.gradient divide by
    # zero; nudge to strictly increasing rather than dropping samples, since the
    # duplicates are real dwell and the extractor sees them
    t = np.maximum.accumulate(t + np.arange(len(t)) * 1e-9)
    v = np.hypot(*np.gradient(XY, t, axis=0).T) if len(t) > 2 else np.zeros(len(t))
    k = max(3, len(v) // 8)
    vs = np.convolve(v, np.ones(k) / k, mode="same")
    pk, _ = find_peaks(vs)
    pt = t[pk] if len(pk) else np.array([])
    if len(pt) >= n:
        centres = np.sort(pt[np.argsort(vs[pk])[-n:]])
    else:
        extra = np.linspace(span / (n + 1), span * n / (n + 1), n - len(pt))
        centres = np.sort(np.concatenate([pt, extra])) if len(pt) else \
            np.linspace(span / (n + 1), span * n / (n + 1), n)

    T0 = np.full(n, max(span / max(n, 1) * 1.4, 0.03))
    t00 = np.clip(centres - 0.5 * T0, -span_pad * span, span)

    lo = np.concatenate([np.full(n, -span_pad * span - 1e-6),
                         np.log(np.full(n, 0.015))])
    hi = np.concatenate([np.full(n, span + 1e-6),
                         np.log(np.full(n, max(3.0 * span, 0.06)))])
    p0 = np.clip(_pack(t00, T0), lo + 1e-9, hi - 1e-9)

    def resid(p):
        a, b = _unpack(p, n)
        M = design(t, a, b)
        return (M @ solve_amps(M, XY) - XY).ravel()

    try:
        sol = least_squares(resid, p0, bounds=(lo, hi), method="trf",
                            max_nfev=200 * (2 * n + 1), xtol=1e-8, ftol=1e-8)
        p = sol.x
    except Exception:
        p = p0

    a, b = _unpack(p, n)
    M = design(t, a, b)
    amps = solve_amps(M, XY)
    rms = float(np.sqrt(np.mean(np.sum((M @ amps - XY) ** 2, axis=1))))
    return (a, b, amps), rms


def plan_xy(xyt, params):
    """The plan's continuous position at the path's own timestamps.

    Sampling is held identical to the human path on purpose. Half the contract's
    features are acceleration and jerk, which are dominated by when the signal
    was sampled and how it was quantised, so leaving those fixed is what
    isolates the question being asked.
    """
    t0, T, amps = params
    t = xyt[:, 2] - xyt[0, 2]
    return design(t, t0, T) @ amps + xyt[0, :2]


def _job(args):
    xyt, ns = args
    out = {}
    for n in ns:
        try:
            params, rms = fit_one(xyt, n)
            out[n] = (plan_xy(xyt, params).astype(np.float32), rms)
        except Exception:
            out[n] = None
    return out


def resample_to(r, k):
    """Stretch a residual series onto k samples in normalised time."""
    if len(r) == k:
        return r
    s = np.linspace(0.0, 1.0, k)
    o = np.linspace(0.0, 1.0, len(r))
    return np.c_[np.interp(s, o, r[:, 0]), np.interp(s, o, r[:, 1])]


def ar1(k, sd, rho, rng):
    """One AR(1) residual series per axis, matched variance and lag-1."""
    e = rng.normal(0.0, 1.0, size=(k, 2))
    out = np.empty((k, 2))
    out[0] = e[0]
    for i in range(1, k):
        out[i] = rho * out[i - 1] + np.sqrt(max(1.0 - rho * rho, 1e-6)) * e[i]
    return out * sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-paths", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--ns", type=int, nargs="+", default=NSWEEP)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t_start = time.time()
    from degeneracy_panel import features_with_jitter, real_paths  # noqa: E402
    import scoring  # noqa: E402
    from w3_raw_column_reread import subset_auc  # noqa: E402

    fit_paths = real_paths(args.n_paths, args.seed, "ref")
    hold_paths = real_paths(args.n_paths, args.seed, "holdout")
    # only paths long enough for the extractor and for a timing search to mean
    # anything; the same subset is used for every N so the sweep is like for like
    fit_paths = [np.asarray(p) for p in fit_paths if len(p) >= 8]
    print(f"[w4] {len(fit_paths)} fit paths, {len(hold_paths)} holdout, "
          f"median {int(np.median([len(p) for p in fit_paths]))} samples",
          flush=True)

    Xh = features_with_jitter(hold_paths, 0.0, args.seed)
    Xh = Xh[np.all(np.isfinite(Xh), axis=1)]
    Xf = features_with_jitter(fit_paths, 0.0, args.seed)
    Xf = Xf[np.all(np.isfinite(Xf), axis=1)]
    floor = subset_auc(Xf, Xh, list(range(Xh.shape[1])))
    contract_floor = scoring.score_features(Xf)["auc_rf_oob"]
    print(f"[w4] floor, real against real: internal {floor:.4f}  "
          f"contract {contract_floor:.4f}", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(_job, [(p, args.ns) for p in fit_paths], chunksize=8))
    print(f"[w4] fitted {len(res)} paths x {len(args.ns)} N in "
          f"{time.time()-t_start:.0f}s", flush=True)

    out = {"n_paths": len(fit_paths), "seed": args.seed, "ns": args.ns,
           "floor_internal": float(floor),
           "floor_contract": float(contract_floor), "rows": []}

    cols = list(range(Xh.shape[1]))

    def score(paths):
        X = features_with_jitter(paths, 0.0, args.seed)
        X = X[np.all(np.isfinite(X), axis=1)]
        return (float(subset_auc(X, Xh, cols)),
                float(scoring.score_features(X)["auc_rf_oob"]))

    rng = np.random.default_rng(args.seed)
    print(f"\n{'N':>3}{'par':>5}{'rms':>7}{'p90':>7}  "
          f"{'arm':<12}{'internal':>10}{'contract':>10}{'excess':>9}")
    for n in args.ns:
        keep = [i for i, r in enumerate(res) if r.get(n) is not None]
        rms = np.array([res[i][n][1] for i in keep])
        plans = [res[i][n][0].astype(np.float64) for i in keep]
        obs = [fit_paths[i] for i in keep]
        resid = [o[:, :2] - p for o, p in zip(obs, plans)]

        # the residual's own shape, which decides whether it can be a process
        # rather than 2K free numbers
        sds = np.array([r.std(axis=0).mean() for r in resid])
        # nanmean, not mean: at 2000 paths at least one leftover is constant
        # over a lag, corrcoef returns nan there, and one nan makes the whole
        # AR arm nan and empties the feature matrix
        rho = float(np.nan_to_num(np.nanmean([
            np.corrcoef(r[:-1, 0], r[1:, 0])[0, 1]
            for r in resid if len(r) > 4 and r[:, 0].std() > 1e-9] or [0.0])))

        perm = rng.permutation(len(plans))
        arms = {
            # the plan alone. Smooth by construction, so this is the plan
            # layer's ceiling with no observation layer at all.
            "plan only": [np.c_[np.rint(p), o[:, 2]]
                          for p, o in zip(plans, obs)],
            # plan plus its OWN residual reconstructs the path exactly. Pure
            # pipeline control: anything but the floor here means render and
            # round is leaking and the rest of the column is unreadable.
            "+ own resid": [np.c_[np.rint(p + r), o[:, 2]]
                            for p, r, o in zip(plans, resid, obs)],
            # the decisive arm. Another path's residual, stretched in
            # normalised time. If this lands at the floor the two layers are
            # separable and each can be modelled on its own.
            "+ swap resid": [np.c_[np.rint(p + resample_to(resid[j], len(p))),
                                   o[:, 2]]
                             for p, o, j in zip(plans, obs, perm)],
            # is the leftover just white noise of the right size
            "+ white": [np.c_[np.rint(p + rng.normal(0, s, size=p.shape)),
                              o[:, 2]]
                        for p, o, s in zip(plans, obs, sds)],
            # or does it need memory
            "+ ar1": [np.c_[np.rint(p + ar1(len(p), s, rho, rng)), o[:, 2]]
                      for p, o, s in zip(plans, obs, sds)],
        }
        row = {"n": n, "n_params": 4 * n, "n_fitted": len(plans),
               "rms_median": float(np.median(rms)),
               "rms_p90": float(np.quantile(rms, 0.9)),
               "resid_sd_median": float(np.median(sds)),
               "resid_lag1_rho": rho, "arms": {}}
        for k, paths in arms.items():
            a_int, a_con = score(paths)
            row["arms"][k] = {"auc_internal": a_int, "auc_contract": a_con,
                              "excess_over_floor": a_int - floor}
            head = (f"{n:>3}{4*n:>5}{np.median(rms):>7.2f}"
                    f"{np.quantile(rms,0.9):>7.2f}  ") if k == "plan only" \
                else " " * 24
            print(f"{head}{k:<12}{a_int:>10.4f}{a_con:>10.4f}"
                  f"{a_int-floor:>+9.4f}", flush=True)
        print(f"{'':24}resid sd {np.median(sds):.2f} px, lag1 rho {rho:+.3f}",
              flush=True)
        out["rows"].append(row)

    out["wall_sec"] = time.time() - t_start
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[w4] wrote {args.out} ({out['wall_sec']:.0f}s)")


if __name__ == "__main__":
    main()
