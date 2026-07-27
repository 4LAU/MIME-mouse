"""What is the velocity envelope worth, at its ceiling?

Two axes closed today, style and stalls, both by pricing a perfect repair
before training for it. This prices the third and last of the 2026-05-15
patterns: the shape of the speed profile across the whole movement. It carries
more of the detector's attention than any other family, roughly 0.24 of the RF's
total weight across mean, std, max, skewness and time-to-peak velocity, and it
is the one the repo records as out of reach for per-position heads. The case for
building a different architecture rests on it, so it should be measured on this
arm rather than inherited.

The envelope is separated from the route by retiming. A path's route is the
polyline through its pixels; its envelope is the monotone map from fraction of
elapsed time to fraction of distance covered, which is scale-free. Retiming
keeps one and swaps the other: identical polyline, identical duration, identical
timestamp grid, somebody else's pacing. Human envelopes carry their own pauses,
so this transplants human timing whole rather than smoothing it.

Envelopes are matched by nearest log-distance with a small random window, since
a 40px flick and a 900px sweep do not pace alike even after normalization.

Five arms, because the operation itself moves the score and that has to be
subtracted rather than assumed away:

  arm as scored             the 0.7283 baseline, untouched
  arm on human envelopes    the ceiling this axis can reach
  arm on arm envelopes      the null: retiming with the model's OWN pacing,
                            which isolates what the resample and the rounding
                            do on their own
  human on arm envelopes    real routes paced like the model, scored as if
                            synthetic. This is the cleanest read of how
                            detectable the model's pacing is by itself.
  human on human envelopes  the floor of the same operation on the human side

The pair at the bottom is the one that decides it. If real paths given the
model's pacing become easy to spot, the envelope is the tell and a new
architecture has a target. If they stay near the floor, the envelope is not
where the detection lives and an architecture aimed at it would miss.

Retimed paths are rounded to the integer lattice, because the scored arm is.

No GPU, no checkpoint touched.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_envelope_ceiling.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

import scoring  # noqa: E402  (metric contract, imported never edited)
from degeneracy_panel import (_score_against,  # noqa: E402
                              features_with_jitter, real_paths)
from features import FEATURE_NAMES  # noqa: E402
from w3_fallback_arrival import correct_additive  # noqa: E402

CACHE = R / "research" / "w3_landing_cache.pkl"
OUT = R / "research" / "w3_envelope_ceiling_results.json"
GRID = np.linspace(0.0, 1.0, 101)
WIN = 25          # how far along the distance-sorted order a donor may be drawn


def envelope(p):
    """Fraction of distance covered against fraction of time elapsed.

    Returns the profile on GRID plus the path's log straight-line distance, or
    None if the path never moves or takes no time. Monotone by construction:
    arc length cannot decrease.
    """
    xy, t = p[:, :2], p[:, 2]
    step = np.hypot(*np.diff(xy, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(step)])
    L, T = float(cum[-1]), float(t[-1] - t[0])
    if L < 1e-9 or T < 1e-9 or len(p) < 5:
        return None
    f = np.interp(GRID, (t - t[0]) / T, cum / L)
    f = np.maximum.accumulate(np.clip(f, 0.0, 1.0))
    f[0], f[-1] = 0.0, 1.0
    d = float(np.hypot(*(xy[-1] - xy[0])))
    return f, np.log(max(d, 1.0))


def retime(p, f):
    """Same polyline, same duration, same timestamps, pacing taken from f."""
    xy, t = p[:, :2], p[:, 2]
    step = np.hypot(*np.diff(xy, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(step)])
    L, T = float(cum[-1]), float(t[-1] - t[0])
    if L < 1e-9 or T < 1e-9:
        return p
    target = L * np.interp((t - t[0]) / T, GRID, f)
    x = np.interp(target, cum, xy[:, 0])
    y = np.interp(target, cum, xy[:, 1])
    return np.c_[np.round(x), np.round(y), t]


def _spread(u):
    """Give runs of identical values a share of the gap to the next one.

    Inverting an envelope maps distance to time, and a path that sits still
    has several samples at one distance. Without this they all collapse onto
    the same instant and the path's own holds vanish, which is exactly the
    texture the position-resampling version destroys.
    """
    u = np.asarray(u, dtype=np.float64).copy()
    n, i = len(u), 0
    while i < n:
        j = i
        while j + 1 < n and u[j + 1] == u[i]:
            j += 1
        if j > i:
            nxt = u[j + 1] if j + 1 < n else 1.0
            u[i:j + 1] = np.linspace(u[i], nxt, j - i + 2)[:j - i + 1]
        i = j + 1
    return u


def retime_clock(p, f):
    """Retime by moving the clock, not the pointer.

    Every pixel of the original path is kept exactly, in order. Only the
    timestamps change, so that distance covered against time elapsed follows
    f. Nothing is interpolated and nothing is re-rounded, so the step lattice,
    the sub-pixel structure and the holds all survive untouched. This is the
    honest instrument: retime() above rebuilds positions and the rebuilding
    alone costs more than the effect being measured.
    """
    xy, t = p[:, :2], p[:, 2]
    step = np.hypot(*np.diff(xy, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(step)])
    L, T = float(cum[-1]), float(t[-1] - t[0])
    if L < 1e-9 or T < 1e-9 or len(p) < 5:
        return p
    u = _spread(np.interp(cum / L, f, GRID))
    u = np.maximum.accumulate(np.clip(u, 0.0, 1.0))
    if u[-1] - u[0] < 1e-9:
        return p
    u = (u - u[0]) / (u[-1] - u[0])
    tn = t[0] + T * u
    # the extractor floors dt at 1e-6; keep every step above it honestly
    tn = np.maximum.accumulate(tn + np.arange(len(tn)) * 1e-7)
    return np.c_[xy, tn]


def donors(paths):
    """Envelopes and their log-distances, sorted by distance for matching."""
    out = [(i, envelope(np.asarray(p, dtype=np.float64))) for i, p in enumerate(paths)]
    out = [(i, e) for i, e in out if e is not None]
    F = np.array([e[0] for _, e in out])
    ld = np.array([e[1] for _, e in out])
    order = np.argsort(ld)
    return F[order], ld[order]


def apply_envelopes(paths, F, ld_sorted, rng, how=retime_clock):
    """Retime every path with a donor envelope drawn near its own distance."""
    out = []
    for p in paths:
        q = np.asarray(p, dtype=np.float64)
        e = envelope(q)
        if e is None:
            out.append(q)
            continue
        pos = int(np.searchsorted(ld_sorted, e[1]))
        pos = int(np.clip(pos + rng.integers(-WIN, WIN + 1), 0, len(F) - 1))
        out.append(how(q, F[pos]))
    return out


def score(paths, seed):
    X = features_with_jitter(paths, 0.0, seed)
    X = X[np.all(np.isfinite(X), axis=1)]
    return float(scoring.score_features(X)["auc_rf_oob"]), int(len(X))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    # pickle.load: this repo's own artifact from the 2026-07-20 landing-price
    # run on this machine, never third-party input.
    with open(CACHE, "rb") as fh:
        specs, trajs = pickle.load(fh)
    arm = [correct_additive(np.asarray(t), *(int(v) for v in s)) if len(t) >= 3
           else np.asarray(t) for s, t in zip(specs, trajs)]
    real = [np.asarray(p, dtype=np.float64) for p in
            real_paths(args.n_real, args.seed, "ref")]
    print(f"[envelope] {len(arm)} arm paths, {len(real)} real paths", flush=True)

    Fh, ldh = donors(real)
    Fa, lda = donors(arm)
    print(f"[envelope] {len(Fh)} human envelopes, {len(Fa)} model envelopes")

    def rg():
        return np.random.default_rng(args.seed)

    res, summary = {}, {}
    for tag, how in (("clock only", retime_clock), ("rebuilt positions", retime)):
        cases = [
            ("arm as scored", arm),
            ("arm on human envelopes", apply_envelopes(arm, Fh, ldh, rg(), how)),
            ("arm on arm envelopes", apply_envelopes(arm, Fa, lda, rg(), how)),
            ("human on arm envelopes", apply_envelopes(real, Fa, lda, rg(), how)),
            ("human on human envelopes", apply_envelopes(real, Fh, ldh, rg(), how)),
            ("human untouched", real),
        ]
        print(f"\n{tag:<28}{'n':>7}{'contract AUC':>15}")
        r = {}
        for name, paths in cases:
            auc, n = score(paths, args.seed)
            r[name] = {"auc_rf_oob": auc, "n": n}
            print(f"  {name:<26}{n:>7}{auc:>15.4f}")
        res[tag] = r

        base = r["arm as scored"]["auc_rf_oob"]
        hu = r["human untouched"]["auc_rf_oob"]
        hf = r["human on human envelopes"]["auc_rf_oob"]
        ha = r["human on arm envelopes"]["auc_rf_oob"]
        s = {"ceiling_move": r["arm on human envelopes"]["auc_rf_oob"] - base,
             "operation_null": r["arm on arm envelopes"]["auc_rf_oob"] - base,
             "operation_damage": hf - hu,
             "envelope_alone": ha - hf}
        summary[tag] = s
        print(f"  {'operation damage':<26}{s['operation_damage']:>22.4f}"
              f"   (human on human envelopes vs untouched)")
        print(f"  {'envelope alone costs':<26}{s['envelope_alone']:>22.4f}"
              f"   (real routes, model pacing, same operation both sides)")

    # Both transplants damage the paths more than the effect they were built to
    # measure (0.26 and 0.32 against an effect near 0.01), so neither ceiling is
    # trustworthy. This needs no surgery: run the contract RF on subsets of the
    # feature columns. "without" is what the detector still has if that family
    # were matched perfectly, which is the ceiling the transplant failed to give.
    def subset_auc(Xs, Xh, cols):
        """RF-OOB AUC on a column subset, score_features' own recipe.

        score_features cannot take a subset: its dispersion battery is fixed at
        18 columns. So the AUC half of the recipe is repeated here, balanced the
        same way with the same estimator, seed and OOB decision function. These
        are diagnostics on feature families, never contract numbers. The full-18
        value is printed next to the contract's own so any drift is visible.
        """
        n = min(len(Xs), len(Xh))
        X = np.vstack([Xh[:n][:, cols], Xs[:n][:, cols]])
        y = np.concatenate([np.zeros(n), np.ones(n)])
        clf = RandomForestClassifier(n_estimators=scoring.RF_N_ESTIMATORS,
                                     oob_score=True, n_jobs=-1,
                                     random_state=scoring.RF_SEED)
        clf.fit(X, y)
        return float(roc_auc_score(y, clf.oob_decision_function_[:, 1]))

    Xa = features_with_jitter(arm, 0.0, args.seed)
    Xa = Xa[np.all(np.isfinite(Xa), axis=1)]
    Xr = features_with_jitter(real, 0.0, args.seed)
    Xr = Xr[np.all(np.isfinite(Xr), axis=1)]
    groups = {
        "envelope": ["mean_velocity", "std_velocity", "max_velocity",
                     "velocity_skewness", "time_to_peak_velocity",
                     "movement_duration"],
        "turning": ["curvature_mean", "curvature_std", "angular_velocity_mean",
                    "angular_velocity_std", "num_direction_changes",
                    "path_efficiency", "max_deviation"],
        "derivatives": ["mean_acceleration", "std_acceleration",
                        "max_acceleration", "mean_jerk", "std_jerk"],
        # turning splits by what a per-position head could in principle
        # control. wobble is local texture, decided a few samples at a time.
        # excursion is a whole-path outcome: how far the route strayed and how
        # much longer it was than the straight line. EXPERIMENTS.md:2012 says
        # per-position heads cannot reach the second kind, so which half
        # carries the gap decides whether a different architecture is the
        # answer or a better-trained version of this one is.
        "turning: wobble": ["curvature_mean", "curvature_std",
                            "angular_velocity_mean", "angular_velocity_std",
                            "num_direction_changes"],
        "turning: excursion": ["path_efficiency", "max_deviation"],
    }
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    all_cols = np.arange(len(FEATURE_NAMES))
    contract = float(_score_against(Xa, Xr)["auc_rf_oob"])
    full = subset_auc(Xa, Xr, all_cols)
    print(f"\nfeature families, no surgery         alone      without")
    print(f"{'all 18 features':<30}{full:>11.4f}"
          f"   (contract {contract:.4f})")
    fam = {"_all": full, "_contract": contract}
    for name, cols in groups.items():
        c = np.array([idx[f] for f in cols])
        keep = np.setdiff1d(all_cols, c)
        a = subset_auc(Xa, Xr, c)
        w = subset_auc(Xa, Xr, keep)
        fam[name] = {"alone": a, "without": w, "n_features": len(cols)}
        print(f"{name + f' ({len(cols)})':<30}{a:>11.4f}{w:>13.4f}")

    out = {"seed": args.seed, "cases": res, "summary": summary,
           "families": fam, "wall_sec": time.time() - t0}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[envelope] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
