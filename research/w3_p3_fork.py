"""The P3 architecture fork: is the gap coverage, or is it texture?

The design brief for a new architecture currently says it must be dispersion
calibrated LOCALLY, and that came from w3_envelope_ceiling's split of turning
into a local wobble half (worth 0.102) and a whole-path excursion half (worth
0.019). research/w3_raw_column_reread.py killed that: on raw model output the
two halves are worth 0.0185 and 0.0174, the same, on both checkpoints. The 5x
asymmetry was correct_additive. So the brief has no measured reason to prefer a
per-position architecture, and something has to replace that reason before any
GPU time is spent.

There is a much older finding that would replace it, and it points somewhere
else entirely. w3_missing_paths (row ...27602144) fit the contract forest, read
each real path's own out-of-bag probability, and split the humans into the
quarter the model places nothing near and the three quarters it covers:

  arm against all real paths          0.7353
  arm against covered real paths      0.4821     chance
  arm against uncovered real paths    0.9400

If that holds, the model is not slightly wrong everywhere. It is indistinguishable
from humans on most of human movement and absent from a quarter of it, and the
whole score is coming from the absence. That is a coverage problem, and coverage
and texture want opposite architectures:

  coverage   the model produces one movement style over and over. The fix is
             whatever lets it reach the other styles: mode coverage, a latent
             that actually moves the output, a mixture. Per-position wobble
             calibration would not touch it.
  texture    every path is slightly wrong in the same way. The fix is local
             calibration, which is what the brief currently asks for.

But w3_missing_paths scored a single correct_additive arm, like the seven other
probes audited in w3_raw_column_reread, and its descriptors are all shape and
pacing quantities the operator moves. So its answer cannot be used as it stands.
This re-runs it over raw, additive and jog.

Two instruments.

  coverage split   the covered / uncovered AUC split above, per arm. The
                   forest is refit per arm, because "which real paths this arm
                   fails to cover" is an arm-specific question. A split that
                   only appears on the additive arm was the operator; one that
                   holds on raw is the model's.

  style spread     per-descriptor std(arm) / std(human), per arm. The coverage
                   reading says the model emits one style repeatedly, which is
                   a claim about SPREAD, not about any average. If the model's
                   descriptor spread is far below the human's on raw output,
                   coverage is confirmed from a second direction that does not
                   go through the forest at all.

Descriptors come from w3_missing_paths.describe, computed off the resampled
path rather than the 18 features, so they are an independent read rather than a
restatement of what the forest saw.

No generation, no GPU, no checkpoint touched.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_p3_fork.py
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

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

import scoring  # noqa: E402  (metric contract, imported never edited)
from degeneracy_panel import (_score_against, features_with_jitter,  # noqa: E402
                              real_paths)
from w3_aiming_price import correct_jog  # noqa: E402
from w3_fallback_arrival import correct_additive  # noqa: E402
from w3_missing_paths import DESCRIPTORS, describe  # noqa: E402

OUT = R / "research" / "w3_p3_fork_results.json"
CACHES = {
    "fc_v2": R / "research" / "w3_landing_cache.pkl",
    "resid_v2": R / "research" / "w3_jog_cache_event_polar_4m_resid_v2.pkl",
}
OPS = {"additive": correct_additive, "jog": correct_jog}
RECORDED = {"all": 0.7353, "covered": 0.4821, "uncovered": 0.9400}


def load_arms(cache):
    """raw / additive / jog for one cached model arm.

    pickle.load: repo-own artifact written by the landing-price and jog runs on
    this machine, never third-party input.
    """
    with open(cache, "rb") as fh:
        specs, trajs = pickle.load(fh)
    keep = [(tuple(int(v) for v in np.asarray(s)),
             np.asarray(t, dtype=np.float64))
            for s, t in zip(specs, trajs) if t is not None and len(t) >= 3]
    arms = {"raw": [t for _, t in keep]}
    for op, f in OPS.items():
        arms[op] = [f(t, *s) for s, t in keep]
    return arms


def coverage_split(Xa, Xr, split, seed):
    """Refit the forest for this arm, split the real paths on their own OOB
    probability, and score the arm against each half on its own."""
    n = min(len(Xa), len(Xr))
    clf = RandomForestClassifier(n_estimators=scoring.RF_N_ESTIMATORS,
                                 oob_score=True, n_jobs=-1,
                                 random_state=scoring.RF_SEED)
    clf.fit(np.vstack([Xr[:n], Xa[:n]]),
            np.concatenate([np.zeros(n), np.ones(n)]))
    p_real = 1.0 - clf.oob_decision_function_[:n, 1]
    k = max(int(split * n), 10)
    order = np.argsort(-p_real)
    unc, cov = order[:k], order[-k:]
    return {"n_per_class": n, "k": k,
            "all": float(_score_against(Xa[:n], Xr[:n])["auc_rf_oob"]),
            "covered": float(_score_against(Xa[:k], Xr[cov])["auc_rf_oob"]),
            "uncovered": float(_score_against(Xa[:k], Xr[unc])["auc_rf_oob"]),
            }, unc, cov


def descriptor_matrix(paths):
    d = [describe(p) for p in paths]
    D = np.array([[x[c] for c in DESCRIPTORS] for x in d if x is not None])
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", type=float, default=0.25)
    ap.add_argument("--arms", nargs="+", default=["fc_v2", "resid_v2"])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    real = real_paths(args.n_real, args.seed, "ref")
    Xr = features_with_jitter(real, 0.0, args.seed)
    keep = np.all(np.isfinite(Xr), axis=1)
    Xr, real = Xr[keep], [p for p, k in zip(real, keep) if k]
    Dr = descriptor_matrix(real)
    print(f"[fork] {len(Xr)} real reference paths")

    out = {"seed": args.seed, "split": args.split, "recorded": RECORDED,
           "arms": {}}
    for name in args.arms:
        cache = CACHES[name]
        if not cache.exists():
            print(f"[fork] MISSING {cache}, skipping {name}")
            continue
        arms = load_arms(cache)
        print(f"\n{'='*70}\n=== {name}: {len(arms['raw'])} paths\n{'='*70}")

        print(f"\nthe arm scored against the real paths it covers, and those it "
              f"does not")
        print(f"{'arm':<12}{'all real':>11}{'covered':>10}{'uncovered':>12}"
              f"{'spread':>9}")
        res = {}
        for a, paths in arms.items():
            Xa = features_with_jitter(paths, 0.0, args.seed)
            Xa = Xa[np.all(np.isfinite(Xa), axis=1)]
            s, _, _ = coverage_split(Xa, Xr, args.split, args.seed)
            s["spread"] = s["uncovered"] - s["covered"]
            res[a] = s
            print(f"{a:<12}{s['all']:>11.4f}{s['covered']:>10.4f}"
                  f"{s['uncovered']:>12.4f}{s['spread']:>9.4f}")
        print(f"{'recorded':<12}{RECORDED['all']:>11.4f}"
              f"{RECORDED['covered']:>10.4f}{RECORDED['uncovered']:>12.4f}"
              f"{RECORDED['uncovered']-RECORDED['covered']:>9.4f}")

        print(f"\nstyle spread, std(arm) / std(human) per descriptor. Below 1 "
              f"means the model varies less than people do.")
        print(f"{'descriptor':<20}" + "".join(f"{a:>11}" for a in arms))
        ratios = {}
        for j, c in enumerate(DESCRIPTORS):
            h = float(np.std(Dr[:, j]))
            line = f"{c:<20}"
            ratios[c] = {}
            for a in arms:
                Da = descriptor_matrix(arms[a])
                r = float(np.std(Da[:, j])) / h if h > 1e-12 else float("nan")
                ratios[c][a] = r
                line += f"{r:>11.2f}"
            print(line)
        med = {a: float(np.median([ratios[c][a] for c in DESCRIPTORS]))
               for a in arms}
        print(f"{'median ratio':<20}" + "".join(f"{med[a]:>11.2f}" for a in arms))

        out["arms"][name] = {"coverage": res, "spread_ratios": ratios,
                             "median_spread_ratio": med}

    print(f"\n=== read ===")
    for name, o in out["arms"].items():
        r = o["coverage"]["raw"]
        print(f"{name} raw: covered {r['covered']:.4f}, uncovered "
              f"{r['uncovered']:.4f}, spread {r['spread']:.4f}; "
              f"median style spread {o['median_spread_ratio']['raw']:.2f}")

    out["wall_sec"] = time.time() - t0
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[fork] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
