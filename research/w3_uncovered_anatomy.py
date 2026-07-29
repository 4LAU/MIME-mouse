"""What actually makes the uncovered quarter detectable?

w3_p3_fork established the one finding that survived the raw-column audit: on
its own raw output the model reads 0.4237 against the three quarters of human
movement the forest places it near and 0.9066 against the quarter it does not,
so the whole score lives in the quarter it misses. The obvious reading of what
that quarter IS was route shape, because its detour p90 is 3.85 against the
model's 1.69. research/w3_oracle_route.py priced that and it is not the gap:
thinning real human paths down to the model's detour distribution moves them
0.0084 once the thinning null control is subtracted, and injecting excursions
into model paths made them worse, 0.6500 to 0.7584.

So detour was a correlate of the uncovered quarter, not its cause, and the
question of what that quarter is is still open. This stops guessing at it. The
forest already knows the answer; it is only a matter of asking which columns it
is using, against that half specifically.

Three instruments, all against the uncovered half and all on raw output.

  family worth    all-18 AUC minus the AUC with one feature family removed.
                  Large means the forest needs that family to tell the
                  uncovered quarter apart. Read next to the same number against
                  the COVERED half, where the arm is already at chance, because
                  a family that matters equally against both is describing the
                  arm in general rather than the deficit.

  family alone    the AUC that family reaches on its own. Worth and alone
                  disagree whenever families are redundant, and the pair is
                  more honest than either: a family can look worthless because
                  another one covers for it.

  per-feature     standardised distance from the model's mean to the uncovered
                  half's mean, in pooled sd, ranked. Names the individual
                  columns rather than the families, and unlike the AUC readings
                  it says which DIRECTION the model is wrong in, which is what
                  an architecture has to act on.

Both checkpoints. A conclusion that only holds on one is a fit to that model.
No generation, no GPU, no checkpoint touched.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_uncovered_anatomy.py
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

from degeneracy_panel import features_with_jitter, real_paths  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402
from w3_p3_fork import coverage_split  # noqa: E402
from w3_raw_column_reread import GROUPS, subset_auc  # noqa: E402

OUT = R / "research" / "w3_uncovered_anatomy_results.json"
CACHES = {
    "fc_v2": R / "research" / "w3_landing_cache.pkl",
    "resid_v2": R / "research" / "w3_jog_cache_event_polar_4m_resid_v2.pkl",
}
IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
RECORDED = {"fc_v2": {"covered": 0.4237, "uncovered": 0.9066},
            "resid_v2": {"covered": 0.4628, "uncovered": 0.9248}}


def load_raw(cache):
    """Raw model paths from a cache this repo wrote on this machine.

    pickle.load: repo-own artifact from the landing-price and jog runs, never
    third-party input.
    """
    with open(cache, "rb") as fh:
        _, trajs = pickle.load(fh)
    return [np.asarray(t, dtype=np.float64) for t in trajs
            if t is not None and len(t) >= 5]


def families(Xa, Xh, label, out):
    """Worth and alone, per family, against one half of the real paths."""
    allc = list(range(len(FEATURE_NAMES)))
    base = subset_auc(Xa, Xh, allc)
    print(f"\n{label}: all 18 features {base:.4f}")
    print(f"{'family':<22}{'without':>10}{'worth':>9}{'alone':>9}")
    rec = {"all18": float(base), "families": {}}
    for g, names in GROUPS.items():
        cols = [IDX[n] for n in names]
        rest = [c for c in allc if c not in cols]
        without = subset_auc(Xa, Xh, rest)
        alone = subset_auc(Xa, Xh, cols)
        rec["families"][g] = {"without": float(without),
                              "worth": float(base - without),
                              "alone": float(alone)}
        print(f"{g:<22}{without:>10.4f}{base-without:>9.4f}{alone:>9.4f}")
    out[label] = rec
    return rec


def per_feature(Xa, Xh, k=8):
    """Standardised distance from the arm's mean to this half's mean."""
    ma, mh = Xa.mean(axis=0), Xh.mean(axis=0)
    sd = np.sqrt(0.5 * (Xa.var(axis=0) + Xh.var(axis=0)))
    z = (ma - mh) / np.maximum(sd, 1e-12)
    order = np.argsort(-np.abs(z))[:k]
    return z, order


def direct_split(Xa, Xr, feats, seed):
    """Split the real paths on a named feature instead of on the forest.

    The covered / uncovered halves are chosen BY the forest, so "the forest
    needs turning to separate them" is partly circular: it selected them for
    separability and turning is its strongest handle overall. This removes the
    circularity. Rank real paths on one raw feature, take the top and bottom
    quartile, and score the arm against each. No model, no forest, nothing
    fitted enters the definition of the halves, so a split that survives here
    is a property of the data.
    """
    n = min(len(Xa), len(Xr))
    k = max(n // 4, 25)
    print(f"\nsplitting the real paths on one feature instead of on the "
          f"forest, {k} per half")
    print(f"{'feature'  :<26}{'arm vs low':>12}{'arm vs high':>13}"
          f"{'spread':>9}")
    rng = np.random.default_rng(seed)
    allc = list(range(len(FEATURE_NAMES)))
    rnd = [subset_auc(Xa[:k], Xr[rng.permutation(n)[:k]], allc)
           for _ in range(3)]
    print(f"{'RANDOM quartile (control)':<26}{np.mean(rnd):>12.4f}"
          f"{'':>13}{'':>9}   sd {np.std(rnd):.4f}")
    res = {"_random_control": {"mean": float(np.mean(rnd)),
                               "sd": float(np.std(rnd)),
                               "runs": [float(v) for v in rnd]}}
    for f in feats:
        v = Xr[:n, IDX[f]]
        o = np.argsort(v)
        lo = subset_auc(Xa[:k], Xr[o[:k]], list(range(len(FEATURE_NAMES))))
        hi = subset_auc(Xa[:k], Xr[o[-k:]], list(range(len(FEATURE_NAMES))))
        res[f] = {"low": float(lo), "high": float(hi),
                  "spread": float(hi - lo)}
        print(f"{f:<26}{lo:>12.4f}{hi:>13.4f}{hi-lo:>9.4f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", type=float, default=0.25)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    real = real_paths(args.n_real, args.seed, "ref")
    Xr = features_with_jitter(real, 0.0, args.seed)
    Xr = Xr[np.all(np.isfinite(Xr), axis=1)]
    print(f"[anat] {len(Xr)} real reference paths")

    out = {"seed": args.seed, "split": args.split, "recorded": RECORDED,
           "arms": {}}
    for name, cache in CACHES.items():
        if not cache.exists():
            print(f"[anat] MISSING {cache}, skipping {name}")
            continue
        Xa = features_with_jitter(load_raw(cache), 0.0, args.seed)
        Xa = Xa[np.all(np.isfinite(Xa), axis=1)]
        s, unc, cov = coverage_split(Xa, Xr, args.split, args.seed)
        print(f"\n{'='*74}\n=== {name}: covered {s['covered']:.4f}, "
              f"uncovered {s['uncovered']:.4f} "
              f"(w3_p3_fork recorded {RECORDED[name]['covered']:.4f} / "
              f"{RECORDED[name]['uncovered']:.4f})\n{'='*74}")

        arm = {"split": s, "halves": {}}
        Xu, Xc = Xr[unc], Xr[cov]
        families(Xa[:len(Xu)], Xu, "uncovered", arm["halves"])
        families(Xa[:len(Xc)], Xc, "covered", arm["halves"])

        print(f"\nwhere the model sits, in pooled sd, against each half. "
              f"Positive means the model reads HIGHER than the real paths.")
        zu, order = per_feature(Xa, Xu)
        zc, _ = per_feature(Xa, Xc)
        print(f"{'feature':<26}{'vs uncovered':>14}{'vs covered':>12}"
              f"{'excess':>9}")
        rows = []
        for i in order:
            rows.append({"feature": FEATURE_NAMES[i], "z_uncovered": float(zu[i]),
                         "z_covered": float(zc[i]),
                         "excess": float(abs(zu[i]) - abs(zc[i]))})
            print(f"{FEATURE_NAMES[i]:<26}{zu[i]:>14.2f}{zc[i]:>12.2f}"
                  f"{abs(zu[i])-abs(zc[i]):>9.2f}")
        arm["per_feature_top"] = rows
        # movement_duration is the control: w3_p3_fork found the model already
        # at human spread on duration, so its split must come out small.
        arm["direct_split"] = direct_split(
            Xa, Xr, ["curvature_std", "curvature_mean", "path_efficiency",
                     "max_deviation", "movement_duration"], args.seed)
        arm["z_uncovered"] = {FEATURE_NAMES[i]: float(zu[i])
                              for i in range(len(FEATURE_NAMES))}
        arm["z_covered"] = {FEATURE_NAMES[i]: float(zc[i])
                            for i in range(len(FEATURE_NAMES))}
        out["arms"][name] = arm

    print(f"\n=== read ===")
    for name, a in out["arms"].items():
        u, c = a["halves"]["uncovered"], a["halves"]["covered"]
        best = max(u["families"], key=lambda g: u["families"][g]["worth"])
        print(f"{name}: the uncovered half needs '{best}' most "
              f"(worth {u['families'][best]['worth']:.4f}, alone "
              f"{u['families'][best]['alone']:.4f}); same family against the "
              f"covered half is worth {c['families'][best]['worth']:.4f}")

    out["wall_sec"] = time.time() - t0
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[anat] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
