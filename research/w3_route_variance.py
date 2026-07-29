"""Why is route shape the one thing the model will not vary?

w3_p3_fork established that the model is not globally under-dispersed. On raw
output its spread matches the human's on pausing, pause count, reversals,
duration and peak speed (median std ratio 0.94) and collapses on exactly two
descriptors, both whole-path route shape:

  detour_ratio   0.35 on fc_v2, 0.22 on resid_v2
  overshoot      0.66 on fc_v2, 0.37 on resid_v2

That is the finding a new architecture would be built on, so the mechanism
behind it has to be tested before any design is committed to, and not assumed.
The obvious mechanism is variance concentration in an autoregressive sampler.
detour_ratio is travelled distance over straight-line distance, so it is a sum
over every step the model emits. If the per-step choices are close to
independent, the relative spread of their sum falls like one over the square
root of the number of steps, no matter how loose the per-step sampling
temperature is. Local descriptors do not sum, so they keep their spread. That
would explain the split exactly.

It also makes a prediction that can be wrong, which is the point of running it:
the collapse must DEEPEN as paths get longer, because longer paths sum more
terms. If the ratio is flat across path length, concentration is not the
mechanism and the design reason has to come from somewhere else.

Two instruments.

  length scaling   std(model) / std(human) for both collapsed descriptors,
                   inside bins of path length. Binning on the straight-line
                   distance of the request, so both sides are answering the
                   same requests and a model that simply draws shorter paths
                   cannot fake the trend. Reported alongside the emitted step
                   count per bin, which is the thing actually being summed.

  free or forced   how much of the human's own detour spread is a free choice
                   rather than a consequence of the request. If the request
                   determines the route, a generator has nothing to vary and a
                   route latent is pointless. Measured as within-bin spread
                   against overall spread, plus the R squared of the best
                   linear read of log detour on log distance. High leftover
                   spread means people take visibly different routes between
                   the same two points, and the model has to be able to as
                   well.

Raw model output only. The correction operator moves both descriptors and has
already been shown to invent structure, so it has no business in a design
decision. No generation, no GPU, no checkpoint touched.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_route_variance.py
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

from degeneracy_panel import real_paths  # noqa: E402
from w3_missing_paths import describe  # noqa: E402

OUT = R / "research" / "w3_route_variance_results.json"
CACHES = {
    "fc_v2": R / "research" / "w3_landing_cache.pkl",
    "resid_v2": R / "research" / "w3_jog_cache_event_polar_4m_resid_v2.pkl",
}
COLLAPSED = ["detour_ratio", "overshoot"]
# the control descriptors, the ones w3_p3_fork found at human spread. They must
# NOT show the same length trend, or the trend is an artefact of the binning.
CONTROL = ["paused_fraction", "peak_speed", "duration_s"]
RECORDED = {"fc_v2": {"detour_ratio": 0.35, "overshoot": 0.66},
            "resid_v2": {"detour_ratio": 0.22, "overshoot": 0.37}}


def rows(paths):
    """Descriptors plus emitted step count, and the paths they survived from.

    Returned paired, because the coverage read indexes real paths by their
    descriptor row and a silent length mismatch would misattribute them.
    """
    out, kept = [], []
    for p in paths:
        p = np.asarray(p, dtype=np.float64)
        d = describe(p)
        if d is None or not all(np.isfinite(d[c])
                                for c in COLLAPSED + CONTROL
                                + ["straight_dist_px"]):
            continue
        d = dict(d)
        d["n_steps"] = int(len(p))
        out.append(d)
        kept.append(p)
    return out, kept


def col(rs, c):
    return np.array([r[c] for r in rs], dtype=np.float64)


def load_raw(cache):
    """Raw model paths from a cache this repo wrote on this machine.

    pickle.load: repo-own artifact from the landing-price and jog runs, never
    third-party input.
    """
    with open(cache, "rb") as fh:
        _, trajs = pickle.load(fh)
    return [np.asarray(t, dtype=np.float64) for t in trajs
            if t is not None and len(t) >= 3]


def length_scaling(hr, mr, edges, name, out):
    """std(model) / std(human) per descriptor, inside bins of request length."""
    hd, md = col(hr, "straight_dist_px"), col(mr, "straight_dist_px")
    print(f"\n=== {name}: does the collapse deepen with path length? ===")
    print(f"{'straight px':<16}{'n hum':>7}{'n mod':>7}{'steps h':>9}"
          f"{'steps m':>9}" + "".join(f"{c[:12]:>14}" for c in COLLAPSED)
          + "".join(f"{c[:10]:>12}" for c in CONTROL))
    band = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        h = [r for r, v in zip(hr, hd) if lo <= v < hi]
        m = [r for r, v in zip(mr, md) if lo <= v < hi]
        if len(h) < 50 or len(m) < 50:
            print(f"{f'{lo:.0f} to {hi:.0f}':<16}{len(h):>7}{len(m):>7}"
                  f"   (too few to read)")
            continue
        rec = {"lo": float(lo), "hi": float(hi), "n_human": len(h),
               "n_model": len(m),
               "steps_human": float(np.median(col(h, "n_steps"))),
               "steps_model": float(np.median(col(m, "n_steps"))), "ratio": {}}
        line = (f"{f'{lo:.0f} to {hi:.0f}':<16}{len(h):>7}{len(m):>7}"
                f"{rec['steps_human']:>9.0f}{rec['steps_model']:>9.0f}")
        for c in COLLAPSED + CONTROL:
            sh, sm = float(np.std(col(h, c))), float(np.std(col(m, c)))
            r = sm / sh if sh > 1e-12 else float("nan")
            rec["ratio"][c] = r
            line += (f"{r:>14.3f}" if c in COLLAPSED else f"{r:>12.2f}")
        print(line)
        band.append(rec)
    out[name] = band
    if len(band) >= 2:
        for c in COLLAPSED:
            f, l = band[0]["ratio"][c], band[-1]["ratio"][c]
            verdict = ("deepens" if l < f - 0.05 else
                       "reverses" if l > f + 0.05 else "flat")
            print(f"  {c:<16} shortest {f:.3f} -> longest {l:.3f}  {verdict}")
    return band


def free_or_forced(hr, edges):
    """How much of the human's detour spread survives fixing the request."""
    d = col(hr, "straight_dist_px")
    print(f"\n=== is the human's route a free choice, or set by the request? ===")
    print(f"{'descriptor':<16}{'overall sd':>12}{'within-bin sd':>15}"
          f"{'free share':>12}{'R2 on log dist':>16}")
    res = {}
    for c in COLLAPSED:
        v = col(hr, c)
        overall = float(np.std(v))
        within, tot = [], 0
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (d >= lo) & (d < hi)
            if s.sum() < 50:
                continue
            within.append(s.sum() * np.var(v[s]))
            tot += int(s.sum())
        wsd = float(np.sqrt(sum(within) / tot)) if tot else float("nan")
        lv, ld = np.log(np.maximum(v, 1e-6)), np.log(np.maximum(d, 1e-6))
        b = np.polyfit(ld, lv, 1)
        r2 = float(1.0 - np.var(lv - np.polyval(b, ld)) / np.var(lv))
        res[c] = {"sd_overall": overall, "sd_within_bin": wsd,
                  "free_share": wsd / overall if overall > 1e-12 else float("nan"),
                  "r2_log_dist": r2}
        print(f"{c:<16}{overall:>12.3f}{wsd:>15.3f}"
              f"{res[c]['free_share']:>12.2f}{r2:>16.3f}")
    return res


def who_is_uncovered(hr, real, mr, paths, name, split, seed, out):
    """Profile the quarter of real paths the model places nothing near.

    w3_p3_fork found that split holds on raw output and carries the whole
    score. This asks what those paths actually ARE. If they are the short,
    wandering ones, the coverage finding and the route-shape collapse are one
    fact and not two, and the design target is a single thing.
    """
    from degeneracy_panel import features_with_jitter
    from w3_p3_fork import coverage_split

    Xr = features_with_jitter(real, 0.0, seed)
    kr = np.all(np.isfinite(Xr), axis=1)
    Xr, hk = Xr[kr], [r for r, k in zip(hr, kr) if k]
    Xa = features_with_jitter(paths, 0.0, seed)
    Xa = Xa[np.all(np.isfinite(Xa), axis=1)]
    s, unc, cov = coverage_split(Xa, Xr, split, seed)

    print(f"\n=== {name}: what are the real paths the model does not reach? ===")
    print(f"{'group':<14}{'n':>6}{'straight px':>13}{'detour p50':>12}"
          f"{'detour p90':>12}{'steps p50':>11}")
    res = {"auc": s}
    for lab, idx in (("covered", cov), ("uncovered", unc)):
        g = [hk[i] for i in idx]
        r = {"n": len(g),
             "straight_p50": float(np.median(col(g, "straight_dist_px"))),
             "detour_p50": float(np.median(col(g, "detour_ratio"))),
             "detour_p90": float(np.percentile(col(g, "detour_ratio"), 90)),
             "steps_p50": float(np.median(col(g, "n_steps")))}
        res[lab] = r
        print(f"{lab:<14}{r['n']:>6}{r['straight_p50']:>13.0f}"
              f"{r['detour_p50']:>12.2f}{r['detour_p90']:>12.2f}"
              f"{r['steps_p50']:>11.0f}")
    a = {"n": len(hk),
         "straight_p50": float(np.median(col(hk, "straight_dist_px"))),
         "detour_p50": float(np.median(col(hk, "detour_ratio"))),
         "detour_p90": float(np.percentile(col(hk, "detour_ratio"), 90)),
         "steps_p50": float(np.median(col(hk, "n_steps")))}
    res["all_real"] = a
    print(f"{'all real':<14}{a['n']:>6}{a['straight_p50']:>13.0f}"
          f"{a['detour_p50']:>12.2f}{a['detour_p90']:>12.2f}"
          f"{a['steps_p50']:>11.0f}")
    mo = {"detour_p50": float(np.median(col(mr, "detour_ratio"))),
          "detour_p90": float(np.percentile(col(mr, "detour_ratio"), 90))}
    res["model"] = mo
    print(f"{'model (raw)':<14}{len(mr):>6}{'-':>13}{mo['detour_p50']:>12.2f}"
          f"{mo['detour_p90']:>12.2f}{'-':>11}")
    out[name] = res
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    hr, real = rows(real_paths(args.n_real, args.seed, "ref"))
    print(f"[route] {len(hr)} human paths")
    hd = col(hr, "straight_dist_px")
    edges = np.quantile(hd, np.linspace(0, 1, args.bins + 1))
    edges[-1] = np.inf
    print(f"[route] length bins at {[f'{e:.0f}' for e in edges[:-1]]} px")

    out = {"seed": args.seed, "n_human": len(hr), "recorded": RECORDED,
           "edges": [float(e) for e in edges[:-1]] + [None], "scaling": {},
           "uncovered": {}}
    out["free_or_forced"] = free_or_forced(hr, edges)

    for name, cache in CACHES.items():
        if not cache.exists():
            print(f"[route] MISSING {cache}, skipping {name}")
            continue
        mr, mp = rows(load_raw(cache))
        print(f"\n[route] {name}: {len(mr)} raw model paths")
        for c in COLLAPSED:
            r = float(np.std(col(mr, c))) / float(np.std(col(hr, c)))
            print(f"  whole-arm {c:<14} std ratio {r:.3f}   "
                  f"(w3_p3_fork recorded {RECORDED[name][c]:.2f})")
        length_scaling(hr, mr, edges, name, out["scaling"])
        who_is_uncovered(hr, real, mr, mp, name, 0.25, args.seed,
                         out["uncovered"])

    print("\n=== read ===")
    for name, band in out["scaling"].items():
        if len(band) >= 2:
            print(f"{name}: detour ratio "
                  f"{band[0]['ratio']['detour_ratio']:.3f} at "
                  f"{band[0]['steps_model']:.0f} steps -> "
                  f"{band[-1]['ratio']['detour_ratio']:.3f} at "
                  f"{band[-1]['steps_model']:.0f} steps")

    out["wall_sec"] = time.time() - t0
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[route] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
