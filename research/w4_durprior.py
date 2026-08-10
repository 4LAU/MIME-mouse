"""Which duration prior did the w4 diagnostics actually run under.

research/w4_evprice.py sets EVENT_CHOICE_TEMP and nothing else, so the cache
every w4 diagnostic reads was built with the LIBRARY DEFAULT duration prior:
std_mult 0.7, Gaussian per distance bin. Every serving path in this repo
(generate.py, README.md, w1_oneshot_score.py, w3_*_eval.py) sets
EVENT_DUR_STD=1.0 DUR_EMPIRICAL=1 instead.

That is a difference in the generative model, upstream of the first event. This
script prices it on the duration marginal alone, with no GPU and no contract
read, so the size of the discrepancy is known before any generation is spent
on it.

Feature 14 is movement_duration. Human log duration on the val split reads
mean -0.9882 sd 0.8683 IQR 1.2657; the cache reads sd 0.6675 IQR 0.8976. If the
prior is the cause, the served setting should close most of that gap here.

READ ONLY. No generation, no GPU, no contract scorer. Never touches
data/human_eval_features.npy.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_durprior.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

DUR_COL = 14


def spreads(x):
    q1, q3 = np.percentile(x, [25, 75])
    return {"mean": float(np.mean(x)), "sd": float(np.std(x)),
            "iqr": float(q3 - q1)}


def draw(std_mult, empirical, log_dists, seed):
    """Rebuild the prior at one setting and draw one duration per spec, exactly
    as build_specs does. DUR_EMPIRICAL is read in DurationModel.__init__, so it
    has to be set before the object is constructed, not before sample()."""
    os.environ["DUR_EMPIRICAL"] = "1" if empirical else "0"
    from experiments._common import DurationModel
    dm = DurationModel("./training", std_mult=std_mult)
    dm._rng = np.random.default_rng(seed)
    return np.array([math.log(dm.sample(ld)) for ld in log_dists])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="research/w4_evprice_cache.npz")
    ap.add_argument("--out", default="research/w4_durprior.json")
    args = ap.parse_args()

    import scoring  # noqa: E402
    from phase_a_baseline import make_specs  # noqa: E402

    H = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    hum = spreads(np.log(H[:, DUR_COL]))
    G = np.load(args.cache)["F"]
    cache = spreads(np.log(G[:, DUR_COL]))

    # the same distances build_specs would see, so the between bin component of
    # the spread is held fixed and only the within bin draw changes.
    log_dists = []
    for sx, sy, ex, ey in make_specs(args.n, args.seed):
        d = math.hypot(ex - sx, ey - sy)
        if d >= 1e-6:
            log_dists.append(math.log(d))
    log_dists = np.array(log_dists)

    print(f"\n  {len(log_dists)} specs, one duration draw each, no generation")
    print(f"  human is the realised movement_duration, the rest are the "
          f"COMMANDED duration the trunk is handed\n")
    print(f"  {'setting':<26}{'mean':>9}{'sd':>9}{'iqr':>9}"
          f"{'sd/human':>10}{'iqr/human':>11}")
    print(f"  {'human val (realised)':<26}{hum['mean']:>9.4f}{hum['sd']:>9.4f}"
          f"{hum['iqr']:>9.4f}{1.0:>10.3f}{1.0:>11.3f}")
    print(f"  {'w4 cache (realised)':<26}{cache['mean']:>9.4f}"
          f"{cache['sd']:>9.4f}{cache['iqr']:>9.4f}"
          f"{cache['sd'] / hum['sd']:>10.3f}"
          f"{cache['iqr'] / hum['iqr']:>11.3f}")

    out = {"human": hum, "w4_cache": cache, "settings": {}}
    arms = [(0.7, False), (1.0, False), (1.0, True), (1.25, True), (1.5, True)]
    for sm, emp in arms:
        x = draw(sm, emp, log_dists, args.seed + 1)
        s = spreads(x)
        name = f"std {sm}" + (" empirical" if emp else " gaussian")
        out["settings"][name] = s
        mark = "  <- w4 default" if (sm, emp) == (0.7, False) else (
            "  <- served" if (sm, emp) == (1.0, True) else "")
        print(f"  {name:<26}{s['mean']:>9.4f}{s['sd']:>9.4f}{s['iqr']:>9.4f}"
              f"{s['sd'] / hum['sd']:>10.3f}{s['iqr'] / hum['iqr']:>11.3f}"
              f"{mark}", flush=True)

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n  the commanded spread is an upper bound on the realised spread, "
          f"the model tracks it at corr 0.9990\n  wrote {args.out}\n")


if __name__ == "__main__":
    main()
