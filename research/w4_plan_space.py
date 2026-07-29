"""Is the trunk's deficit in the plan, or in the emitter?

Today's chain, in order. A plan of 6 minimum jerk submovements tracks a real
human path to about 1 px. Rendered on the lattice it still reads 0.83, and the
control that explains why is not about plans at all: a REAL human path with half
a pixel of noise added reads 0.867, and with one pixel 0.950. The contract is
overwhelmingly sensitive to sample-scale texture. Then the transfer arms: plan
plus its own leftover is exactly the floor, but plan plus that same leftover
REVERSED, identical magnitude and spectrum and lag structure, is 0.830. The
texture is not a process that can be added to a smooth curve. It is the exact
integer sequence.

Which inverts the picture of the trunk. The trunk emits integer steps natively
and never holds a fractional position, so it never pays the sub-pixel penalty at
all, and that is very likely why the discrete event stream is the only survivor
here while every continuous whole-path model landed at 0.92 to 0.99: those
render off-lattice positions and round them, which is the 0.950 arm above. The
trunk is therefore GOOD at texture and, at 0.65, bad at something else.

The two layer claim is that the something else is the plan: the trunk is handed
four numbers and never commits to a whole movement, so its whole-path properties
emerge as an average, which is the shape of yesterday's diffuse deficit.

That claim is now decidable without training anything. Fit plans to the trunk's
own cached output and to real paths, throw the paths away, and score the PLANS
against each other. Plan space is where the commitment lives.

  If the trunk's plans separate from human plans, the deficit is in the plan
  layer, the architecture argument holds, and the build is plan model plus the
  existing emitter.

  If the trunk's plans are already human, the plan layer adds nothing, the
  deficit is in the emitter, and this whole direction dies here for the price of
  a CPU hour.

Floor is human ref plans against human holdout plans, measured the same way, so
the fitter's own noise is charged to the floor and not to the arms.

No generation, no GPU, no checkpoint read or written.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w4_plan_space.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

OUT = R / "research" / "w4_plan_space_results.json"
NSUB = 6

PLAN_NAMES = [
    "n_effective",        # submovements carrying more than 5 pct of the total
    "primary_share",      # largest submovement's share of total displacement
    "amp_gini",           # how unequal the submovement sizes are
    "log_total_amp",      # summed submovement length, before cancellation
    "amp_over_net",       # that total against the straight line distance
    "mean_overlap",       # mean pairwise temporal overlap, 0 is strictly serial
    "max_overlap",
    "mean_dur",           # submovement durations
    "std_dur",
    "log_span",           # first onset to last offset
    "mean_turn",          # angle between consecutive submovement directions
    "max_turn",
    "reversal_share",     # share of displacement pointing backwards along net
    "onset_gap_mean",     # rhythm of the corrections
    "onset_gap_std",
    "tail_share",         # displacement in the last third of the plan
]


def gini(v):
    v = np.sort(np.abs(v))
    n = len(v)
    if n == 0 or v.sum() <= 0:
        return 0.0
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def plan_features(t0, T, amps):
    """Describe a plan without reference to the path it came from."""
    mag = np.hypot(*amps.T)
    tot = float(mag.sum())
    if tot <= 0 or not np.all(np.isfinite(mag)):
        return None
    net = amps.sum(axis=0)
    netmag = float(np.hypot(*net))
    o = np.argsort(t0)
    t0s, Ts, A = t0[o], T[o], amps[o]
    ms = mag[o]

    lo, hi = t0s, t0s + Ts
    ov = []
    for i in range(len(t0s)):
        for j in range(i + 1, len(t0s)):
            inter = max(0.0, min(hi[i], hi[j]) - max(lo[i], lo[j]))
            ov.append(inter / max(min(Ts[i], Ts[j]), 1e-6))
    ov = np.array(ov) if ov else np.array([0.0])

    keep = ms > 0.05 * tot
    ang = np.arctan2(A[:, 1], A[:, 0])
    dang = np.abs(np.diff(np.unwrap(ang[keep]))) if keep.sum() > 1 \
        else np.array([0.0])

    ndir = net / max(netmag, 1e-6)
    proj = A @ ndir
    back = float(np.sum(np.abs(proj[proj < 0])))

    span = float(hi.max() - lo.min())
    gaps = np.diff(t0s) if len(t0s) > 1 else np.array([0.0])
    third = lo.min() + 2.0 * span / 3.0
    tail = float(ms[t0s >= third].sum())

    return [
        float(keep.sum()), float(ms.max() / tot), gini(ms),
        float(np.log1p(tot)), float(tot / max(netmag, 1e-6)),
        float(ov.mean()), float(ov.max()),
        float(Ts.mean()), float(Ts.std()), float(np.log1p(max(span, 0.0))),
        float(dang.mean()), float(dang.max()),
        back / tot, float(gaps.mean()), float(gaps.std()), tail / tot,
    ]


def _job(xyt):
    from w4_submovement_ceiling import fit_one
    try:
        (t0, T, amps), rms = fit_one(np.asarray(xyt, dtype=np.float64), NSUB)
        f = plan_features(t0, T, amps)
        if f is None or not np.all(np.isfinite(f)):
            return None
        return f, float(rms)
    except Exception:
        return None


def fit_many(paths, ex):
    got = [g for g in ex.map(_job, paths, chunksize=8) if g is not None]
    return np.array([g[0] for g in got]), np.array([g[1] for g in got])


def auc(Xa, Xb, cols=None):
    """The contract's estimator and seed, on plan columns instead of motion
    ones. A diagnostic, never a contract number: the contract's 18 features are
    fixed and this deliberately is not them."""
    import scoring
    c = cols if cols is not None else list(range(Xa.shape[1]))
    n = min(len(Xa), len(Xb))
    X = np.vstack([Xb[:n][:, c], Xa[:n][:, c]])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    clf = RandomForestClassifier(n_estimators=scoring.RF_N_ESTIMATORS,
                                 oob_score=True, n_jobs=-1,
                                 random_state=scoring.RF_SEED)
    clf.fit(X, y)
    return (float(roc_auc_score(y, clf.oob_decision_function_[:, 1])),
            clf.feature_importances_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-paths", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t_start = time.time()
    from degeneracy_panel import real_paths  # noqa: E402
    from w3_coupling_gate import CACHES  # noqa: E402
    from w3_joint_structure import load_raw  # noqa: E402

    src = {"human ref": [np.asarray(p) for p in
                         real_paths(args.n_paths, args.seed, "ref")
                         if len(p) >= 8],
           "human holdout": [np.asarray(p) for p in
                             real_paths(args.n_paths, args.seed, "holdout")
                             if len(p) >= 8]}
    for name, cache in CACHES.items():
        if cache.exists():
            src[name] = [p for p in load_raw(cache) if len(p) >= 8][:args.n_paths]
        else:
            print(f"[plan] MISSING {cache}, skipping {name}")

    P, RMS = {}, {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, v in src.items():
            P[k], RMS[k] = fit_many(v, ex)
            print(f"[plan] {k:<14} {len(P[k]):>5} plans, fit rms median "
                  f"{np.median(RMS[k]):.2f} px", flush=True)

    ref = P["human ref"]
    out = {"n_sub": NSUB, "seed": args.seed, "features": PLAN_NAMES, "arms": {}}

    floor, _ = auc(P["human holdout"], ref)
    print(f"\nfloor, human plans against human plans: {floor:.4f}")
    print(f"\n{'arm':<16}{'plan AUC':>10}{'excess':>9}   top plan features")
    for k in P:
        if k == "human ref":
            continue
        a, imp = auc(P[k], ref)
        top = [PLAN_NAMES[i] for i in np.argsort(imp)[::-1][:4]]
        out["arms"][k] = {"plan_auc": a, "excess_over_floor": a - floor,
                          "n": int(len(P[k])),
                          "fit_rms_median": float(np.median(RMS[k])),
                          "top_features": top,
                          "importances": dict(zip(PLAN_NAMES, imp.tolist()))}
        print(f"{k:<16}{a:>10.4f}{a-floor:>+9.4f}   {', '.join(top)}")

    print(f"\nper-feature, standardised mean gap against human ref "
          f"(model minus human, in human SDs)")
    mu, sd = ref.mean(0), ref.std(0) + 1e-9
    print(f"{'feature':<16}" + "".join(f"{k:>16}" for k in P if k != "human ref"))
    gaps = {}
    for i, nme in enumerate(PLAN_NAMES):
        row = {k: float((P[k][:, i].mean() - mu[i]) / sd[i])
               for k in P if k != "human ref"}
        gaps[nme] = row
        print(f"{nme:<16}" + "".join(f"{row[k]:>16.2f}" for k in row))
    out["standardised_gaps"] = gaps
    out["floor"] = float(floor)
    out["wall_sec"] = time.time() - t_start
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[plan] wrote {args.out} ({out['wall_sec']:.0f}s)")


if __name__ == "__main__":
    main()
