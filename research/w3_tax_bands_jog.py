"""Does the arrival tax still scale with how far the model missed, once the
correction operator is the repaired one?

The 2026-07-20 landing-price run split the arm by endpoint miss and priced the
additive correction inside each band (research/w3_landing_price_results.json):

  2 to 5px  +0.019    5 to 15px  +0.020    15 to 40px  +0.034
  40 to 100px  +0.051    100px and up  +0.044

That gradient is the entire case for native aiming. It is what sent six P1
fine-tunes at landing closer, and all six failed. Then 2026-07-26 showed the
operator those bands were priced with was itself injecting the defect, so the
gradient may have been the operator's damage curve rather than the model's.

correct_jog spends the error as whole-pixel changes on the longest steps and
leaves every other step byte identical to the model's own. On the whole arm it
is worth -0.0139 (fc_v2) and -0.0223 (resid_v2). Nobody has priced it per band.
The two possible readings imply opposite work:

  flat      the remaining tax does not care how far the model missed, so better
            aiming buys nothing and P1 stays closed on stronger grounds than
            "six fine-tunes failed".
  scales    aiming still buys something real, and since conditioning tweaks are
            closed, only a different architecture can collect it. That makes it
            a P3 requirement rather than a dead lever.

Two instruments, because they fail differently and neither is inferred from the
other.

  arm bands       the real model's paths, split by their actual miss, scored raw
                  / additive / jog inside each band. The raw column is the
                  control: paths that miss badly may simply be worse paths, and
                  if so raw climbs across bands too and the climb is not the
                  correction's. Only the within-band delta against raw is a fair
                  read. Absolute per-band AUCs are inflated by small n.

  human sweep     real human paths, an invented target a chosen distance from
                  their own endpoint, the same two operators. The path is held
                  fixed and only the miss changes, so this is the operator's
                  damage curve with the model removed entirely. It answers the
                  same question without the arm's confound, and w3_aiming_price
                  already ran a coarse version of it; this one is finer around
                  the 15px region P1 was gated on.

Both caches are read: fc_v2 (6000 paths, the model behind every W3 number) and
resid_v2 (1998 paths, the model behind the standing 0.6986). A conclusion that
only holds on one of them is a fit to that model, not a property of the tax.

No generation, no GPU, no checkpoint touched.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_tax_bands_jog.py
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
from w3_aiming_price import correct_jog  # noqa: E402
from w3_fallback_arrival import correct_additive  # noqa: E402

OUT = R / "research" / "w3_tax_bands_jog_results.json"

# the bands the 2026-07-20 run priced additive in, kept identical so the
# additive column here is directly comparable to the record
BANDS = [(0, 2), (2, 5), (5, 15), (15, 40), (40, 100), (100, 1e9)]
RECORDED_ADDITIVE = {"(2, 5)": 0.0186, "(5, 15)": 0.0204, "(15, 40)": 0.0345,
                     "(40, 100)": 0.0509, "(100, 1000000000.0)": 0.0444}

CACHES = {
    "fc_v2": R / "research" / "w3_landing_cache.pkl",
    "resid_v2": R / "research" / "w3_jog_cache_event_polar_4m_resid_v2.pkl",
}

OPS = {"additive": correct_additive, "jog": correct_jog}


def score(paths, seed):
    X = features_with_jitter(paths, 0.0, seed)
    X = X[np.all(np.isfinite(X), axis=1)]
    if len(X) < 50:
        return float("nan"), int(len(X))
    return float(scoring.score_features(X)["auc_rf_oob"]), int(len(X))


def load_arm(path):
    """(specs, raw paths) from a cache this repo wrote on this machine.

    pickle.load: repo-own artifact from the landing-price and jog runs, never
    third-party input.
    """
    with open(path, "rb") as fh:
        specs, trajs = pickle.load(fh)
    keep = [(np.asarray(s), np.asarray(t, dtype=np.float64))
            for s, t in zip(specs, trajs) if t is not None and len(t) >= 3]
    return [s for s, _ in keep], [t for _, t in keep]


def arm_bands(name, cache, seed, out):
    specs, raw = load_arm(cache)
    ints = [tuple(int(v) for v in s) for s in specs]
    miss = np.array([float(np.hypot(s[2] - t[-1, 0], s[3] - t[-1, 1]))
                     for s, t in zip(ints, raw)])
    corr = {op: [f(t, *s) for s, t in zip(ints, raw)] for op, f in OPS.items()}

    whole = {"raw": score(raw, seed)[0]}
    for op in OPS:
        whole[op] = score(corr[op], seed)[0]
    print(f"\n=== {name}, {len(raw)} paths, miss p50 {np.median(miss):.1f}px ===")
    print(f"whole arm   raw {whole['raw']:.4f}   "
          f"additive {whole['additive']:.4f} (+{whole['additive']-whole['raw']:.4f})"
          f"   jog {whole['jog']:.4f} (+{whole['jog']-whole['raw']:.4f})")

    print(f"\n{'miss px':<14}{'n':>6}{'raw':>9}{'additive':>10}{'jog':>8}"
          f"{'tax add':>10}{'tax jog':>10}{'jog saves':>11}{'recorded':>10}")
    rows = []
    for lo, hi in BANDS:
        sel = np.flatnonzero((miss >= lo) & (miss < hi))
        if len(sel) < 50:
            print(f"{f'{lo} to {hi:g}':<14}{len(sel):>6}   (too few to score)")
            continue
        r, n = score([raw[i] for i in sel], seed)
        a, _ = score([corr["additive"][i] for i in sel], seed)
        j, _ = score([corr["jog"][i] for i in sel], seed)
        rec = RECORDED_ADDITIVE.get(str((lo, hi)))
        print(f"{f'{lo} to {hi:g}':<14}{n:>6}{r:>9.4f}{a:>10.4f}{j:>8.4f}"
              f"{a-r:>10.4f}{j-r:>10.4f}{a-j:>11.4f}"
              + (f"{rec:>10.4f}" if rec else f"{'-':>10}"))
        rows.append({"lo": lo, "hi": hi, "n": n,
                     "miss_median": float(np.median(miss[sel])),
                     "auc_raw": r, "auc_additive": a, "auc_jog": j,
                     "tax_additive": a - r, "tax_jog": j - r,
                     "jog_saves": a - j, "recorded_additive": rec})
    out[name] = {"cache": cache.name, "n_paths": len(raw),
                 "miss_p50": float(np.median(miss)), "whole_arm": whole,
                 "bands": rows}
    return rows


def injection_sweep(paths, label, seed, grid, out, key):
    """Hold the path fixed, invent a target a chosen distance from its own
    endpoint, run both operators. The only thing that varies across a row is
    how far the correction has to move the path, so this is the operator's
    damage curve with path quality held constant.

    Run on human paths it says what the operator costs a perfect generator. Run
    on the arm's own raw paths it says what it costs THIS generator, which is
    the number the aiming question actually turns on: the arm-band split cannot
    answer it, because each band holds different paths at a different baseline
    realism and the correction is being priced against a moving floor.
    """
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, len(paths))
    print(f"\n=== the operators on {label}, path held fixed, "
          f"{len(paths)} paths ===")
    print(f"{'injected miss px':<18}{'n':>6}{'additive':>11}{'jog':>9}"
          f"{'tax add':>10}{'tax jog':>10}{'jog saves':>11}")
    rows, base = [], None
    for dpx in grid:
        arms = {op: [] for op in OPS}
        for p, a in zip(paths, ang):
            sx, sy = int(round(p[0, 0])), int(round(p[0, 1]))
            ex = int(round(p[-1, 0] + dpx * np.cos(a)))
            ey = int(round(p[-1, 1] + dpx * np.sin(a)))
            for op, f in OPS.items():
                arms[op].append(f(p, sx, sy, ex, ey))
        av, n = score(arms["additive"], seed)
        jv, _ = score(arms["jog"], seed)
        if base is None:
            base = av
        print(f"{dpx:<18.0f}{n:>6}{av:>11.4f}{jv:>9.4f}{av-base:>10.4f}"
              f"{jv-base:>10.4f}{av-jv:>11.4f}")
        rows.append({"miss_px": dpx, "n": n, "auc_additive": av, "auc_jog": jv,
                     "tax_additive": av - base, "tax_jog": jv - base,
                     "jog_saves": av - jv})
    out[key] = {"label": label, "baseline_auc": base, "n_paths": len(paths),
                "points": rows}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-real", type=int, default=2000)
    ap.add_argument("--inject", type=float, nargs="+",
                    default=[0, 1, 2, 3, 5, 8, 12, 15, 20, 30, 40, 60, 80])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    out = {"seed": args.seed, "bands": [list(b) for b in BANDS], "arms": {},
           "sweeps": {}}
    raws = {}
    for name, cache in CACHES.items():
        if not cache.exists():
            print(f"[tax] MISSING cache {cache}, skipping {name}")
            continue
        arm_bands(name, cache, args.seed, out["arms"])
        raws[name] = load_arm(cache)[1]

    real = [np.asarray(p, dtype=np.float64)
            for p in real_paths(args.n_real, args.seed, "ref")]
    sweep = injection_sweep(real, "real human paths", args.seed, args.inject,
                            out["sweeps"], "human")
    model_sweeps = {}
    for name, raw in raws.items():
        model_sweeps[name] = injection_sweep(
            raw[:args.n_real], f"{name} raw model paths", args.seed,
            args.inject, out["sweeps"], name)

    print("\n=== read ===")
    for name, a in out["arms"].items():
        b = [r for r in a["bands"] if r["n"] >= 50]
        if len(b) >= 3:
            lo_t, hi_t = b[0]["tax_jog"], b[-1]["tax_jog"]
            lo_a, hi_a = b[0]["tax_additive"], b[-1]["tax_additive"]
            print(f"{name} bands: additive tax {lo_a:+.4f} to {hi_a:+.4f} "
                  f"(span {hi_a-lo_a:+.4f}); jog tax {lo_t:+.4f} to {hi_t:+.4f}"
                  f" (span {hi_t-lo_t:+.4f})")
    print(f"\n{'sweep, path held fixed':<28}" + "".join(
        f"{f'{d:g}px':>9}" for d in args.inject))
    for key, rows in [("human", sweep)] + list(model_sweeps.items()):
        print(f"{key + ' jog tax':<28}" + "".join(
            f"{r['tax_jog']:>9.4f}" for r in rows))
        print(f"{key + ' additive tax':<28}" + "".join(
            f"{r['tax_additive']:>9.4f}" for r in rows))

    out["wall_sec"] = time.time() - t0
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[tax] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
