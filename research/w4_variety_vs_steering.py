"""Is the model's shape variety ABSENT, or present but UNSTEERABLE?

cond_realization_probe measured how well the feature-conditioned trunk obeys
its commanded 18-dim character vector. Mean correlation is 0.41, and the shape
family is near zero: mean_jerk -0.005, mean_acceleration 0.167,
time_to_peak_velocity 0.168, movement_duration 0.172, curvature_std 0.228,
curvature_mean 0.306. w3_guidance_capacity then found that turning the
conditioning volume up cannot reach curvature at all, and concluded "curvature
variety is absent from the model, not merely unexpressed."

Those two readings have very different consequences and the record does not
separate them:

  ABSENT      the model can only draw one kind of bend. The marginal spread of
              its curvature is a fraction of human. No amount of conditioning
              work reaches it, because there is nothing to reach. The fix is a
              representation that states curvature directly, which is a rewrite.

  UNSTEERABLE the model draws the full human range of bends, but at random,
              disconnected from what it was told. Marginal spread is human, the
              correlation with the command is what is broken. The fix is how
              the instruction is wired into the network, which is a retrain of
              the same architecture.

This measures the marginals directly against real humans and puts them beside
the adherence numbers, so the two cannot be confused again.

Three quantities per feature, all in units of the human sd so they are
comparable across features:

  spread ratio   model sd / human sd. ABSENT predicts much less than 1 on the
                 shape family. UNSTEERABLE predicts about 1.
  mean shift     (model mean - human mean) / human sd. Location error, which is
                 a third thing again and is what a bias correction would fix.
  1-feature AUC  the contract estimator on that column alone. Says which
                 columns actually carry the detector, so the diagnosis is
                 weighted by what matters rather than by feature count.

The human ref/holdout split gives every column its own floor, so a ratio near 1
and an AUC near the floor mean the same thing: that column is not the problem.

Reads cached generations only. No GPU, no checkpoint written, no generation.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w4_variety_vs_steering.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

OUT = R / "research" / "w4_variety_vs_steering_results.json"
PROBE = R / "research" / "cond_probe_artifacts" / "measure_summary.json"

# the family the two readings disagree about, by the names in features.py
SHAPE = ["curvature_mean", "curvature_std", "angular_velocity_mean",
         "angular_velocity_std", "num_direction_changes", "path_efficiency",
         "max_deviation"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-paths", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    from degeneracy_panel import features_with_jitter, real_paths  # noqa: E402
    from features import FEATURE_NAMES  # noqa: E402
    from w3_coupling_gate import CACHES  # noqa: E402
    from w3_joint_structure import load_raw  # noqa: E402
    from w3_raw_column_reread import subset_auc  # noqa: E402

    def feats(paths):
        X = features_with_jitter(paths, 0.0, args.seed)
        return X[np.all(np.isfinite(X), axis=1)]

    Xh = feats(real_paths(args.n_paths, args.seed, "ref"))
    Xf = feats(real_paths(args.n_paths, args.seed, "holdout"))
    print(f"[vs] human ref {len(Xh)}, human holdout {len(Xf)}", flush=True)

    arms = {"human holdout (floor)": Xf}
    for name, cache in CACHES.items():
        if cache.exists():
            arms[name] = feats([p for p in load_raw(cache) if len(p) >= 8])
            print(f"[vs] {name}: {len(arms[name])} cached paths", flush=True)

    # adherence, so spread and steering sit in one table
    adh = {}
    if PROBE.exists():
        d = json.load(open(PROBE))
        adh = dict(zip(d["feature_names"], d["r"]))

    hmu, hsd = Xh.mean(0), Xh.std(0)
    hsd = np.where(hsd > 1e-12, hsd, 1.0)

    out = {"n_human": len(Xh), "seed": args.seed, "features": {}, "arms": {}}
    for name, X in arms.items():
        rows = {}
        for i, f in enumerate(FEATURE_NAMES):
            rows[f] = {
                "spread_ratio": float(X[:, i].std() / hsd[i]),
                "mean_shift": float((X[:, i].mean() - hmu[i]) / hsd[i]),
                "auc_1feat": float(subset_auc(X, Xh, [i])),
            }
        out["arms"][name] = rows

    for name in arms:
        rows = out["arms"][name]
        print(f"\n=== {name} ===")
        print(f"{'feature':<24}{'spread':>8}{'shift':>8}{'1f AUC':>8}{'obeys':>8}")
        order = sorted(FEATURE_NAMES, key=lambda f: -rows[f]["auc_1feat"])
        for f in order:
            r = rows[f]
            a = adh.get(f)
            mark = "  <- shape" if f in SHAPE else ""
            print(f"{f:<24}{r['spread_ratio']:>8.2f}{r['mean_shift']:>8.2f}"
                  f"{r['auc_1feat']:>8.3f}"
                  f"{'   n/a' if a is None else f'{a:>8.2f}'}{mark}")

    # the one line the whole probe exists to produce
    print("\n=== verdict ===")
    fl = out["arms"]["human holdout (floor)"]
    fsp = np.mean([abs(fl[f]["spread_ratio"] - 1.0) for f in SHAPE])
    print(f"human-vs-human shape spread error   {fsp:.3f}  (measurement noise)")
    for name in arms:
        if name.startswith("human"):
            continue
        rows = out["arms"][name]
        sp = np.mean([rows[f]["spread_ratio"] for f in SHAPE])
        al = np.mean([rows[f]["spread_ratio"] for f in FEATURE_NAMES])
        ad = np.mean([adh[f] for f in SHAPE if f in adh]) if adh else float("nan")
        print(f"{name:<12} shape spread {sp:.2f} of human, all-feature "
              f"{al:.2f}, shape adherence {ad:.2f}")
        out["arms"][name]["_summary"] = {"shape_spread": float(sp),
                                         "all_spread": float(al),
                                         "shape_adherence": float(ad)}
    print("\nABSENT predicts shape spread well under 1. UNSTEERABLE predicts "
          "about 1 with low adherence.")

    # ---- second stage: is any single column worth fixing? --------------
    # Human timestamps sit on a 1 ms grid, so human durations do too: 903
    # distinct values in 2000 samples. The model's are continuous, 2000
    # distinct in 2000. A tree splits between grid points and everything in
    # the gaps is synthetic. Snapping the model's timestamps to the same grid
    # is therefore a clean, targeted repair of the largest single carrier, and
    # the question is what repairing it buys on the joint score.
    import scoring  # noqa: E402

    def snap(paths, q=0.001):
        o = []
        for p in paths:
            c = np.asarray(p, float).copy()
            c[:, 2] = np.maximum.accumulate(
                np.rint((c[:, 2] - c[0, 2]) / q) * q) + c[0, 2]
            o.append(c)
        return o

    raw = [np.asarray(p, float) for p in load_raw(CACHES["fc_v2"])
           if len(p) >= 8]
    rng = np.random.default_rng(args.seed)
    A, Bs = feats(raw), feats(snap(raw))
    n = min(len(Xh), len(Xf), len(A), len(Bs))
    sel = rng.choice(len(A), n, replace=False)
    A, Bs, Hn, Fn = A[sel], Bs[sel], Xh[:n], Xf[:n]
    allc = list(range(len(FEATURE_NAMES)))

    print(f"\n=== redundancy test, balanced n={n} ===")
    red = {}
    for nm, X in [("floor", Fn), ("fc_v2 as-is", A), ("fc_v2 t snapped", Bs)]:
        red[nm] = {"auc_all": float(subset_auc(X, Hn, allc)),
                   "auc_contract": float(scoring.score_features(X)["auc_rf_oob"])}
        print(f"  {nm:<18}all-18 {red[nm]['auc_all']:.4f}   "
              f"contract {red[nm]['auc_contract']:.4f}")
    print(f"\n  {'feature':<24}{'floor':>8}{'as-is':>8}{'snapped':>9}{'delta':>8}")
    for i, f in enumerate(FEATURE_NAMES):
        a, b, c = (subset_auc(Fn, Hn, [i]), subset_auc(A, Hn, [i]),
                   subset_auc(Bs, Hn, [i]))
        red.setdefault("per_feature", {})[f] = {
            "floor": float(a), "as_is": float(b), "snapped": float(c)}
        print(f"  {f:<24}{a:>8.3f}{b:>8.3f}{c:>9.3f}{c - b:>+8.3f}")
    out["redundancy"] = red
    print("\nIf the two largest single carriers can be removed without moving "
          "the joint score, no per-feature repair can ever work here.")

    out["wall_sec"] = time.time() - t0
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[vs] wrote {args.out} ({out['wall_sec']:.0f}s)")


if __name__ == "__main__":
    main()
