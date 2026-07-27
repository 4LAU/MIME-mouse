"""Did the model ever see the movements it cannot produce?

research/w3_guidance_capacity.py established that the missing smooth movement is
absent from the model rather than suppressed in it, which leaves two very
different explanations. Either the training objective failed to learn something
that was in front of it, which argues for a new model, or the training corpus
never contained it, which is a data pipeline bug and far cheaper to fix.

The event corpus is not the segmented recordings. training/prepare_events.py
re-encodes each movement as a stream of (dt, dx, dy) events and then applies two
hard filters: fewer than 5 events is dropped, more than 256 events is dropped.
Neither is neutral with respect to the kinds of movement that turned out to be
missing. event_codec.encode_events splits any step longer than 63 px into
several, so a fast movement arrives at the length filter already inflated. A
movement that hesitates spends its pause emitting events too. Both of the kinds
research/w3_missing_paths.py named are pushed toward the 256 cap by the encoder
itself, and over it they are discarded rather than truncated.

So this runs the real pipeline over real recordings and asks which ones survive.
Two populations, taken from the same coverage split w3_missing_paths defined:
the real paths the model reproduces at chance, and the quarter it cannot. If the
uncovered quarter is dropped by the corpus builder at a much higher rate, the
model never saw them and the fix is upstream of any architecture decision.

encode_events and the thresholds are imported, never reimplemented, so what runs
here is what built the corpus.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_corpus_coverage.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

import scoring  # noqa: E402  (metric contract, imported never edited)
from event_codec import encode_events  # noqa: E402
from degeneracy_panel import features_with_jitter, real_paths  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from training.prepare_events import MAX_EVENTS, MIN_EVENTS  # noqa: E402
from w3_missing_paths import DESCRIPTORS, describe  # noqa: E402

OUT = R / "research" / "w3_corpus_coverage_results.json"


def corpus_fate(traj):
    """What training/prepare_events.py would do with this recording."""
    p = np.asarray(traj, dtype=np.float64)
    xy = np.round(p[:, :2]).astype(np.int64)
    t = p[:, 2]
    if len(p) < MIN_EVENTS:
        return "dropped_short", 0
    enc = encode_events(xy, t)
    if enc is None:
        return "dropped_bad", 0
    dt_s, _, _ = enc
    m = len(dt_s)
    if m < MIN_EVENTS:
        return "dropped_short", m
    if m > MAX_EVENTS:
        return "dropped_long", m
    if float(dt_s.sum()) < 1e-4:
        return "dropped_bad", m
    return "kept", m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    # the same coverage split w3_missing_paths defines: fit the scorer on real
    # against the arm, read each real path's own out-of-bag probability
    import pickle
    from w3_fallback_arrival import correct_additive
    # pickle.load: this repo's own artifact from the 2026-07-20 landing-price
    # run on this machine, never third-party input.
    with open(R / "research" / "w3_landing_cache.pkl", "rb") as fh:
        specs, trajs = pickle.load(fh)
    arm = [correct_additive(np.asarray(t), *(int(v) for v in s)) if len(t) >= 3
           else np.asarray(t) for s, t in zip(specs, trajs)]
    Xa = features_with_jitter(arm, 0.0, args.seed)
    Xa = Xa[np.all(np.isfinite(Xa), axis=1)]
    real = real_paths(args.n, args.seed, "ref")
    Xr = features_with_jitter(real, 0.0, args.seed)
    keep = np.all(np.isfinite(Xr), axis=1)
    Xr, real = Xr[keep], [p for p, k in zip(real, keep) if k]
    n = min(len(Xa), len(Xr))

    clf = RandomForestClassifier(n_estimators=scoring.RF_N_ESTIMATORS,
                                 oob_score=True, n_jobs=-1,
                                 random_state=scoring.RF_SEED)
    clf.fit(np.vstack([Xr[:n], Xa[:n]]),
            np.concatenate([np.zeros(n), np.ones(n)]))
    p_real = 1.0 - clf.oob_decision_function_[:n, 1]
    order = np.argsort(-p_real)
    k = max(n // 4, 10)
    groups = {"model cannot produce": order[:k], "model reproduces": order[-k:],
              "all real paths": np.arange(n)}

    print(f"[corpus] MIN_EVENTS={MIN_EVENTS} MAX_EVENTS={MAX_EVENTS}, "
          f"{n} real recordings, {k} per group")
    out = {"n": int(n), "k": int(k), "min_events": MIN_EVENTS,
           "max_events": MAX_EVENTS, "groups": {}}

    fates = [corpus_fate(real[i]) for i in range(n)]
    kinds = np.array([f[0] for f in fates])
    lens = np.array([f[1] for f in fates])

    print(f"\n{'group':<24}{'kept':>8}{'too long':>10}{'too short':>11}"
          f"{'bad':>7}{'median events':>15}")
    for name, idx in groups.items():
        kk = kinds[idx]
        row = {"kept": float(np.mean(kk == "kept")),
               "dropped_long": float(np.mean(kk == "dropped_long")),
               "dropped_short": float(np.mean(kk == "dropped_short")),
               "dropped_bad": float(np.mean(kk == "dropped_bad")),
               "median_events": float(np.median(lens[idx]))}
        out["groups"][name] = row
        print(f"{name:<24}{row['kept']:>7.1%}{row['dropped_long']:>10.1%}"
              f"{row['dropped_short']:>11.1%}{row['dropped_bad']:>7.1%}"
              f"{row['median_events']:>15.0f}")

    # what the surviving corpus looks like against what was fed in
    desc = [describe(real[i]) for i in range(n)]
    ok = np.array([d is not None for d in desc])
    D = np.array([[d[c] for c in DESCRIPTORS] if d is not None
                  else [np.nan] * len(DESCRIPTORS) for d in desc])
    kept = ok & (kinds == "kept")
    lost = ok & (kinds != "kept")
    print(f"\nwhat the length filter removes, median over "
          f"{int(kept.sum())} kept and {int(lost.sum())} dropped recordings")
    print(f"{'descriptor':<20}{'kept':>12}{'dropped':>12}{'ratio':>8}")
    shape = {}
    for j, c in enumerate(DESCRIPTORS):
        a, b = float(np.median(D[kept, j])), float(np.median(D[lost, j]))
        shape[c] = {"kept": a, "dropped": b}
        print(f"{c:<20}{a:>12.3f}{b:>12.3f}"
              f"{(b / a if abs(a) > 1e-9 else float('nan')):>8.2f}")
    out["descriptors"] = shape
    out["overall_drop_rate"] = float(np.mean(kinds != "kept"))
    print(f"\noverall, the corpus builder discards "
          f"{out['overall_drop_rate']:.1%} of real recordings")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[corpus] wrote {args.out}")


if __name__ == "__main__":
    main()
