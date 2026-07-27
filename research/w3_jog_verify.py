"""Verify the jog endpoint correction against the standing 0.7283 arm.

w3_aiming_price.py measured 0.7144 for correct_jog against 0.7283 for the
correction in service, on the same 6000 cached trajectories. That would be the
best honest single-trajectory number this program has, so it does not get
reported off one print. This checks the four things that could make it fake:

  arrival     the whole point of the correction. Every served path must start
              and end on the requested whole pixel. Anything under 100% and the
              comparison is against a different product.
  collapse    a lower AUC bought by making the paths more alike is not a result.
              The contract's own dispersion battery and collapse flag are read
              for both arms, not just the AUC.
  selection   one path per spec, one correction, no candidates, no picking. The
              spec list is asserted to be the same length as the path list and
              every path is used.
  stability   the move has to survive resampling, and two instruments were
              wrong before this one. Sweeping the seed was useless: the seed
              only feeds the feature jitter, the jitter is set to 0, and the
              contract pins its own RF seed at 42, so five seeds returned five
              identical numbers. Bootstrapping with replacement was worse than
              useless: duplicated paths are easy for the forest, absolute AUC
              climbs to 0.80 on the 6000-path arm and 0.89 on the 2000-path
              one, and near saturation a real gap gets compressed toward zero.
              What is used here is subsampling WITHOUT replacement, which
              leaves the detection problem at its true difficulty.

Nothing here generates. The arm is the landing cache, the model is never loaded
and the checkpoint is never opened.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_jog_verify.py
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
from degeneracy_panel import features_with_jitter  # noqa: E402
from w3_aiming_price import correct_jog  # noqa: E402
from w3_fallback_arrival import correct_additive  # noqa: E402

CACHE = R / "research" / "w3_landing_cache.pkl"
OUT = R / "research" / "w3_jog_verify_results.json"


def full_score(paths, seed):
    X = features_with_jitter(paths, 0.0, seed)
    X = X[np.all(np.isfinite(X), axis=1)]
    r = scoring.score_features(X)
    return r, int(len(X))


def arrival(paths, specs):
    hit = sum(1 for p, s in zip(paths, specs)
              if p[0, 0] == int(s[0]) and p[0, 1] == int(s[1])
              and p[-1, 0] == int(s[2]) and p[-1, 1] == int(s[3]))
    return hit / max(len(paths), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boots", type=int, default=12)
    ap.add_argument("--frac", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", default=str(CACHE),
                    help="path cache to read; defaults to the fc_v2 landing "
                         "cache, or pass a w3_jog_cache_*.pkl from "
                         "w3_jog_on_resid.py")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    # pickle.load: this repo's own artifacts, written by the landing-price run
    # and by research/w3_jog_on_resid.py on this machine, never third-party
    # input.
    with open(args.cache, "rb") as fh:
        specs, trajs = pickle.load(fh)
    keep = [(np.asarray(s), np.asarray(t)) for s, t in zip(specs, trajs)
            if len(t) >= 3]
    specs = [s for s, _ in keep]
    raw = [t for _, t in keep]
    assert len(specs) == len(raw), "spec and path lists desynced"
    print(f"[verify] {len(raw)} specs, {len(raw)} paths, one path per spec, "
          f"no candidate pool", flush=True)

    arms = {"additive (standing)": [correct_additive(t, *(int(v) for v in s))
                                    for s, t in zip(specs, raw)],
            "jog": [correct_jog(t, *(int(v) for v in s))
                    for s, t in zip(specs, raw)]}

    print(f"\n{'':<22}{'arrives':>10}{'n':>7}{'contract AUC':>15}")
    res = {}
    for name, paths in arms.items():
        arr = arrival(paths, specs)
        r, n = full_score(paths, args.seed)
        res[name] = {"exact_arrival": arr, "n": n,
                     "auc_rf_oob": r["auc_rf_oob"],
                     "collapse_flag": bool(r["collapse_flag"]),
                     "collapse_features": list(r["collapse_features"]),
                     "dispersion_ratios": {k: float(v) for k, v in
                                           r["dispersion_ratios"].items()}}
        print(f"{name:<22}{arr:>10.1%}{n:>7}{r['auc_rf_oob']:>15.4f}")
    gap = res["jog"]["auc_rf_oob"] - res["additive (standing)"]["auc_rf_oob"]
    print(f"{'jog minus additive':<22}{'':>10}{'':>7}{gap:>15.4f}")

    rng = np.random.default_rng(args.seed)
    m = int(round(args.frac * len(raw)))
    print(f"\nsubsample without replacement, {args.boots} draws of {m} "
          f"of {len(raw)} paths")
    d = []
    for b in range(args.boots):
        idx = rng.permutation(len(raw))[:m]
        a, _ = full_score([arms["additive (standing)"][i] for i in idx],
                          args.seed)
        j, _ = full_score([arms["jog"][i] for i in idx], args.seed)
        d.append(j["auc_rf_oob"] - a["auc_rf_oob"])
        print(f"  {b + 1:>2}  additive {a['auc_rf_oob']:.4f}  "
              f"jog {j['auc_rf_oob']:.4f}  gap {d[-1]:+.4f}", flush=True)
    d = np.array(d)
    print(f"\n  gap mean {d.mean():+.4f}, sd {d.std():.4f}, "
          f"worst {d.max():+.4f}, wins {int((d < 0).sum())}/{len(d)}")

    print(f"\ncollapse check (a lower AUC from less variety is not a result)")
    for name, r in res.items():
        print(f"  {name:<22}flag {str(r['collapse_flag']):<6}"
              f"features {r['collapse_features'] or 'none'}")
    print(f"\n  dispersion ratio, synth spread over human spread. 1.0 is a "
          f"match, so what matters is distance from 1.0, not direction.")
    print(f"    {'feature':<28}{'additive':>9}{'jog':>8}{'|1-a|':>8}"
          f"{'|1-j|':>8}   verdict")
    better = worse = 0
    for k in res["jog"]["dispersion_ratios"]:
        a = res["additive (standing)"]["dispersion_ratios"][k]
        j = res["jog"]["dispersion_ratios"][k]
        da, dj = abs(1.0 - a), abs(1.0 - j)
        v = ""
        if dj < da - 0.02:
            v, better = "closer to human", better + 1
        elif dj > da + 0.02:
            v, worse = "further from human", worse + 1
        print(f"    {k:<28}{a:>9.3f}{j:>8.3f}{da:>8.3f}{dj:>8.3f}   {v}")
    print(f"\n  {better} features closer to human, {worse} further, "
          f"{18 - better - worse} unchanged")

    Path(args.out).write_text(json.dumps(
        {"seed": args.seed, "arms": res, "gap": gap,
         "subsample_frac": args.frac, "subsample_gaps": d.tolist(),
         "subsample_mean": float(d.mean()), "subsample_sd": float(d.std()),
         "dispersion_closer": better, "dispersion_further": worse,
         "wall_sec": time.time() - t0}, indent=2))
    print(f"\n[verify] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
