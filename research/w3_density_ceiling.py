"""Is the model emitting the wrong paths, or the right paths at wrong rates?

D3 in research/w3_gap_anatomy.py picked, per request, whichever of 32 candidates
a detector called least synthetic. That set scored 0.7494, WORSE than picking
one at random (0.7204), while the recorded SIR product on the same pool reads
0.5833. Greedy picking chases the picker's own errors and flattens the spread of
what gets served, and the scorer reads that flattening immediately. So D3's
instrument answered a narrower question than intended.

The question it exposed is better. The contract scorer compares a SET of
generated paths against a SET of human ones, so "one trajectory at 0.50" means
the distribution a model produces, sampled one at a time, is indistinguishable
from the human distribution. That splits the failure in two:

  wrong support   the model cannot produce human-like paths at all, and no
                  change to how often it emits each one would help
  wrong density   it can, but emits some kinds too often and others too rarely

This separates them. Estimate how much more or less often humans produce each
candidate than the model does, then resample one path per request in proportion,
sharpening by a temperature. Uniform is what the model does today. If sharpening
walks the score down toward chance, the support is fine and the whole remaining
problem is which paths get emitted, which is a sampling fix rather than a new
architecture. If it stalls, the support is wrong and the architecture must change.

This is a ceiling, not a method. The reweighter sees the candidates it reweights,
so the number is optimistic; it bounds what fixing the density could buy. Two
things keep it from being pure circularity: the reweighter is fitted against the
rebuilt real-path reference from research/degeneracy_panel.py, while the reported
score is the contract scorer against its own separate reference, so the estimator
and the judge never share a sample. Every row also carries score_features' own
dispersion battery, because a reweighting that wins by collapsing the spread of
what is served has not found anything.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_density_ceiling.py
"""
from __future__ import annotations

import argparse
import json
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
from degeneracy_panel import build_reference  # noqa: E402
from w3_gap_anatomy import POOL, pool_features  # noqa: E402

OUT = R / "research" / "w3_density_ceiling_results.json"
# research/w3_gap_anatomy_results.json and the SIR product row, for orientation.
REFERENCE_POINTS = {"typical, one at random": 0.7204,
                    "greedy best of 32": 0.7494,
                    "recorded SIR product": 0.5833}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(POOL))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--temps", type=float, nargs="+",
                    default=[1.0, 0.7, 0.5, 0.3, 0.15])
    ap.add_argument("--n-ref", type=int, default=2000)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    X, owner = pool_features(args.pool)
    ok = np.all(np.isfinite(X), axis=1)
    groups = {}
    for ci in np.flatnonzero(ok):
        groups.setdefault(int(owner[ci]), []).append(ci)
    groups = [np.asarray(v) for v in groups.values() if len(v) >= 2]
    print(f"[density] {ok.sum():,} valid candidates over {len(groups)} requests")

    # the estimator's human sample: rebuilt real paths, never the contract
    # reference the reported score is measured against
    ref = build_reference(0.0, args.n_ref, 42)["X"]
    est = RandomForestClassifier(n_estimators=scoring.RF_N_ESTIMATORS,
                                 n_jobs=-1, random_state=0)
    est.fit(np.vstack([ref, X[ok]]),
            np.concatenate([np.zeros(len(ref)), np.ones(int(ok.sum()))]))
    p = np.clip(est.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    logw = np.log(1.0 - p) - np.log(p)      # log how much humans favour it
    print(f"[density] log weight spread {logw[ok].std():.3f}, "
          f"range {logw[ok].min():.2f} to {logw[ok].max():.2f}")

    rows = []
    for temp in [None] + list(args.temps):
        aucs, collapsed = [], set()
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            pick = []
            for g in groups:
                if temp is None:
                    pick.append(rng.choice(g))
                    continue
                z = logw[g] / temp
                w = np.exp(z - z.max())
                pick.append(rng.choice(g, p=w / w.sum()))
            res = scoring.score_features(X[np.asarray(pick)])
            aucs.append(res["auc_rf_oob"])
            collapsed.update(res["collapse_features"])
        rows.append({"temp": temp, "auc_mean": float(np.mean(aucs)),
                     "auc_sd": float(np.std(aucs)), "aucs": aucs,
                     "collapse_features": sorted(collapsed)})

    print(f"\n{'sampling':<28}{'AUC':>8}{'sd':>8}   collapsed features")
    for r in rows:
        name = ("uniform (what the model does)" if r["temp"] is None
                else f"reweighted, temperature {r['temp']}")
        flag = ", ".join(r["collapse_features"]) or "none"
        print(f"{name:<28}{r['auc_mean']:>8.4f}{r['auc_sd']:>8.4f}   {flag}")
    print("\nfor orientation, same pool, contract scorer")
    for k, v in REFERENCE_POINTS.items():
        print(f"  {k:<26}{v:.4f}")

    best = min(rows, key=lambda r: r["auc_mean"])
    print(f"\nlowest reachable by reweighting alone: {best['auc_mean']:.4f} "
          f"at temperature {best['temp']}")

    out = {"n_requests": len(groups), "seeds": args.seeds, "rows": rows,
           "reference_points": REFERENCE_POINTS, "best": best,
           "wall_sec": time.time() - t0}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[density] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
