"""How much apparent separability does narrowing buy, and which recorded
numbers were bought that way?

research/w3_uncovered_anatomy.py found that scoring the arm against a QUARTER
of the real paths chosen on any single feature reads 0.89 to 0.93, while a
random quarter of the same size reads 0.6286 and the whole arm reads 0.6451.
The inflation appeared even on movement_duration, a feature the model already
matches. So it is not a deficit being revealed, it is the geometry of comparing
a broad distribution against a narrow slice of one.

That confound has now bitten three times in one day: the Task 2 arm-band
instrument, w3_p3_fork, and w3_uncovered_anatomy. Rather than re-run every
affected probe, this measures the artefact once, as a curve, so any recorded
number can be checked against it. The readable quantity is never the raw AUC
against a subset. It is the EXCESS over a random subset of identical size.

Four instruments.

  calibration    narrow the real side to a fraction f, three ways: at random,
                 on a single feature's extreme, and by the forest's own opinion
                 of which real paths look synthetic. Sweep f. Random is the
                 floor the other two have to beat to mean anything.

  which side     the same sweep narrowing the ARM instead. The Task 2 band
                 instrument narrowed that side, not the real one, so it needs
                 its own curve rather than an assumption that the two behave
                 alike.

  feature blind  narrow the real side on each of the 18 features in turn. If
                 the resulting AUCs cluster regardless of which feature was
                 used, the inflation is generic and no feature-defined subset
                 says anything about a deficit.

  corrected      the covered / uncovered numbers this session has been quoting,
                 restated as excess over a random subset at the same k. This is
                 what those rows should have reported.

Both checkpoints. Raw output only, since the correction operator has its own
audit. No generation, no GPU, no checkpoint touched.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_narrowing_audit.py
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
from degeneracy_panel import features_with_jitter, real_paths  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402
from w3_raw_column_reread import subset_auc  # noqa: E402

OUT = R / "research" / "w3_narrowing_audit_results.json"
CACHES = {
    "fc_v2": R / "research" / "w3_landing_cache.pkl",
    "resid_v2": R / "research" / "w3_jog_cache_event_polar_4m_resid_v2.pkl",
}
ALLC = list(range(len(FEATURE_NAMES)))
FRACS = [1.0, 0.5, 0.25, 0.125, 0.0625]

# Every recorded probe that scores an arm against a NARROWED subset of one
# side, and what the narrowing was. Recorded here so the audit is part of the
# artifact rather than a claim in a summary. "exposed" means the conclusion
# rests on an absolute AUC against a subset, with no same-size random control.
AUDIT = {
    "w3_missing_paths": "EXPOSED. Forest-split real paths, quotes 0.4821 "
                        "covered vs 0.9400 uncovered as a deficit. No random "
                        "control. Conclusion does not stand.",
    "w3_p3_fork": "EXPOSED. Same split re-run across raw/additive/jog this "
                  "session. The raw 0.4237 / 0.9066 is the same artefact on "
                  "the uncovered side. Retracted the day it was written.",
    "w3_uncovered_anatomy": "EXPOSED but SELF-CAUGHT. Family decomposition "
                            "against the uncovered half; the random control "
                            "added in the same run is what found this.",
    "w3_critic_coverage": "EXPOSED. Reuses the w3_missing_paths split verbatim "
                          "and reads critic performance per half. The halves "
                          "do not mean what the row says they mean.",
    "w3_corpus_coverage": "INTERPRETATION ONLY. Uses the same split but reads "
                          "corpus survival rates per group, not an AUC, so the "
                          "inflation does not enter its instrument. The group "
                          "LABEL 'model cannot produce' is still unsupported.",
    "w3_tax_bands_jog": "PARTLY EXPOSED, already flagged in-file. The arm-band "
                        "column narrows the ARM by endpoint miss and was "
                        "called confounded on 2026-07-27; the fixed-path "
                        "injection sweep that replaced it narrows nothing and "
                        "stands.",
    "w3_landing_price": "EXPOSED, same arm-band design as above, and it is the "
                        "source of the 2 to 100px additive gradient that sent "
                        "six P1 fine-tunes. Superseded by the injection sweep.",
    "w3_envelope_ceiling": "NOT EXPOSED. Narrows FEATURES, not rows; both "
                           "sides keep every path.",
    "w3_raw_column_reread": "NOT EXPOSED. Same family decomposition over whole "
                            "arms, no row subsetting.",
    "w3_oracle_route": "NOT EXPOSED. Thins the real side, but ships its own "
                       "same-size null control (duration thinning) and the "
                       "conclusion is stated net of it.",
}


def load_raw(cache):
    """Raw model paths from a cache this repo wrote on this machine.

    pickle.load: repo-own artifact from the landing-price and jog runs, never
    third-party input.
    """
    with open(cache, "rb") as fh:
        _, trajs = pickle.load(fh)
    return [np.asarray(t, dtype=np.float64) for t in trajs
            if t is not None and len(t) >= 5]


def forest_order(Xa, Xr, seed):
    """Real paths ranked by how synthetic the forest thinks they look."""
    n = min(len(Xa), len(Xr))
    clf = RandomForestClassifier(n_estimators=scoring.RF_N_ESTIMATORS,
                                 oob_score=True, n_jobs=-1,
                                 random_state=scoring.RF_SEED)
    clf.fit(np.vstack([Xr[:n], Xa[:n]]),
            np.concatenate([np.zeros(n), np.ones(n)]))
    p_real = 1.0 - clf.oob_decision_function_[:n, 1]
    return np.argsort(-p_real), n


def sweep(Xbroad, Xnarrow, order, label, fracs, rng, reps, out):
    """AUC as one side is narrowed three ways, at matched sample size.

    subset_auc balances to the smaller class, so narrowing one side cuts both
    to the same k and sample size is held constant across the three ways. Only
    the SELECTION differs, which is the whole point.
    """
    n = min(len(Xbroad), len(Xnarrow))
    print(f"\n=== {label} ===")
    print(f"{'kept':<10}{'k':>7}{'random':>10}{'sd':>8}{'by feature':>13}"
          f"{'min':>8}{'max':>8}{'by forest':>12}{'excess':>9}")
    rows = []
    for f in fracs:
        k = max(int(f * n), 30)
        rnd = [subset_auc(Xbroad, Xnarrow[rng.permutation(n)[:k]], ALLC)
               for _ in range(reps)]
        feats = [subset_auc(Xbroad, Xnarrow[np.argsort(-Xnarrow[:n, j])[:k]],
                            ALLC) for j in range(len(FEATURE_NAMES))]
        forest = subset_auc(Xbroad, Xnarrow[order[:k]], ALLC)
        rows.append({"frac": f, "k": k, "random": float(np.mean(rnd)),
                     "random_sd": float(np.std(rnd)),
                     "by_feature_mean": float(np.mean(feats)),
                     "by_feature_min": float(np.min(feats)),
                     "by_feature_max": float(np.max(feats)),
                     "by_forest": float(forest),
                     "excess_forest": float(forest - np.mean(rnd)),
                     "excess_feature": float(np.mean(feats) - np.mean(rnd))})
        r = rows[-1]
        print(f"{f:<10.4g}{k:>7}{r['random']:>10.4f}{r['random_sd']:>8.4f}"
              f"{r['by_feature_mean']:>13.4f}{r['by_feature_min']:>8.4f}"
              f"{r['by_feature_max']:>8.4f}{r['by_forest']:>12.4f}"
              f"{r['excess_forest']:>9.4f}")
    out[label] = rows
    return rows


def corrected(Xa, Xr, order, n, seed, reps, rng):
    """The covered / uncovered pair restated as excess over a random subset."""
    k = max(n // 4, 10)
    unc, cov = order[:k], order[-k:]
    rnd = [subset_auc(Xa, Xr[rng.permutation(n)[:k]], ALLC) for _ in range(reps)]
    u = subset_auc(Xa[:k], Xr[unc], ALLC)
    c = subset_auc(Xa[:k], Xr[cov], ALLC)
    base = float(np.mean(rnd))
    # the sweep only ever took each feature's TOP quartile, which controls the
    # uncovered side. The covered side is the other tail and needs the other
    # one, or "the forest found real paths inside the arm" is still an
    # uncontrolled claim.
    lo = [subset_auc(Xa[:k], Xr[np.argsort(Xr[:n, j])[:k]], ALLC)
          for j in range(len(FEATURE_NAMES))]
    print(f"\nthe covered / uncovered pair, restated against a random subset "
          f"of the same {k}")
    print(f"{'half':<14}{'as quoted':>12}{'random':>10}{'excess':>10}"
          f"{'best single feature':>22}")
    print(f"{'uncovered':<14}{u:>12.4f}{base:>10.4f}{u-base:>10.4f}")
    print(f"{'covered':<14}{c:>12.4f}{base:>10.4f}{c-base:>10.4f}"
          f"{np.min(lo):>22.4f}")
    print(f"  the lowest AUC any single feature's bottom quartile reaches is "
          f"{np.min(lo):.4f} ({FEATURE_NAMES[int(np.argmin(lo))]}), "
          f"mean {np.mean(lo):.4f}. The forest reaches {c:.4f}.")
    return {"k": k, "uncovered": float(u), "covered": float(c),
            "random": base, "random_sd": float(np.std(rnd)),
            "excess_uncovered": float(u - base), "excess_covered": float(c - base),
            "feature_bottom_min": float(np.min(lo)),
            "feature_bottom_mean": float(np.mean(lo)),
            "feature_bottom_argmin": FEATURE_NAMES[int(np.argmin(lo))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    Xr = features_with_jitter(real_paths(args.n_real, args.seed, "ref"), 0.0,
                              args.seed)
    Xr = Xr[np.all(np.isfinite(Xr), axis=1)]
    print(f"[audit] {len(Xr)} real reference paths")

    out = {"seed": args.seed, "reps": args.reps, "fracs": FRACS,
           "audit": AUDIT, "arms": {}}
    for name, cache in CACHES.items():
        if not cache.exists():
            print(f"[audit] MISSING {cache}, skipping {name}")
            continue
        Xa = features_with_jitter(load_raw(cache), 0.0, args.seed)
        Xa = Xa[np.all(np.isfinite(Xa), axis=1)]
        order, n = forest_order(Xa, Xr, args.seed)
        whole = subset_auc(Xa, Xr, ALLC)
        print(f"\n{'='*84}\n=== {name}: {len(Xa)} arm paths, whole-arm "
              f"{whole:.4f}\n{'='*84}")

        arm = {"whole_arm": float(whole), "sweeps": {}}
        sweep(Xa, Xr, order, "narrowing the REAL side", FRACS, rng, args.reps,
              arm["sweeps"])
        # narrowing the arm needs the forest's opinion of ARM paths, which is
        # the other tail of the same fit, so the order is rebuilt on that side
        aorder, _ = forest_order(Xr, Xa, args.seed)
        sweep(Xr, Xa, aorder, "narrowing the ARM side", FRACS, rng, args.reps,
              arm["sweeps"])
        arm["corrected_split"] = corrected(Xa, Xr, order, n, args.seed,
                                           args.reps, rng)
        out["arms"][name] = arm

    print(f"\n=== read ===")
    for name, a in out["arms"].items():
        r = a["sweeps"]["narrowing the REAL side"][2]      # the quarter row
        c = a["corrected_split"]
        print(f"{name}: at a quarter, random {r['random']:.4f} against a "
              f"whole arm of {a['whole_arm']:.4f}; any single feature buys "
              f"{r['excess_feature']:+.4f} and the forest buys "
              f"{r['excess_forest']:+.4f}")
        print(f"{'':<{len(name)+2}}the quoted uncovered figure is "
              f"{c['uncovered']:.4f}, excess over random "
              f"{c['excess_uncovered']:+.4f}; covered {c['covered']:.4f}, "
              f"excess {c['excess_covered']:+.4f}")

    print(f"\n=== static audit ===")
    for k, v in AUDIT.items():
        print(f"{k:<24}{v}")

    out["wall_sec"] = time.time() - t0
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[audit] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
