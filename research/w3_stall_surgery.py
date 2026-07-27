"""What is the stall axis worth, at its ceiling?

research/w3_stall_pattern.py measured the four-part stop pattern and found the
opposite of what was proposed. The model does not under-produce stops. It
over-produces them: 12.2 holds per path against the human 8.6, 15.8 percent of
elapsed time held against 9.7, and 6.8 percent exact-zero steps after resample
against the human 5.3. The 2.5 percent deficit at EXPERIMENTS.md:1539 belongs to
the WS4-era flow model, not to this arm, and carrying it across was an error.

Length and duration of a hold already match (2.34 samples against 2.33, 7.8ms
against 7.0, identical p50 and p90). So the only part of the pattern that is
wrong by more than a little is how many there are, and how fast the pointer is
moving when it stops.

This bounds what fixing that is worth before anyone trains for it. Hold removal
is done by hand on the arm's own paths: delete the repeated samples of a hold
and leave every surviving timestamp untouched, so total duration is unchanged
and the resample interpolates through the gap instead of sitting still. Sweep
the removal rate from none to all, score each with the contract scorer, and read
the best the axis can do.

The surgery is a measurement, not a deliverable. It is a hand edit of finished
paths and no model produces it. If the curve is flat the axis is closed; if it
dips hard at the human-matched rate then a fine-tune has something to aim at,
and the decoder already carries EVENT_TICKMERGE as the in-model version of the
same edit.

No GPU, no checkpoint touched.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_stall_surgery.py
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
from w3_fallback_arrival import correct_additive  # noqa: E402
from w3_stall_pattern import holds, measure  # noqa: E402

CACHE = R / "research" / "w3_landing_cache.pkl"
OUT = R / "research" / "w3_stall_surgery_results.json"


def drop_holds(traj, frac, rng):
    """Delete a random `frac` of this path's holds.

    A hold spanning samples s..e-1 is one pixel repeated. Removing rows
    s+1..e-1 leaves the pointer at that pixel for a single sample instead of
    several. Every surviving row keeps its own timestamp, so the path still
    starts and ends when it did and only the stationary stretch is gone.
    """
    p = np.asarray(traj, dtype=np.float64)
    st, en = holds(p)
    if len(st) == 0 or frac <= 0:
        return p
    pick = rng.random(len(st)) < frac
    if not pick.any():
        return p
    kill = np.zeros(len(p), dtype=bool)
    for s, e in zip(st[pick], en[pick]):
        kill[s + 1:e] = True
    out = p[~kill]
    return out if len(out) >= 5 else p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fracs", type=float, nargs="+",
                    default=[0.0, 0.15, 0.30, 0.50, 0.75, 1.0])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    # pickle.load: this repo's own artifact from the 2026-07-20 landing-price
    # run on this machine, never third-party input.
    with open(CACHE, "rb") as fh:
        specs, trajs = pickle.load(fh)
    arm = [correct_additive(np.asarray(t), *(int(v) for v in s)) if len(t) >= 3
           else np.asarray(t) for s, t in zip(specs, trajs)]
    print(f"[surgery] {len(arm)} arm paths", flush=True)

    # 0.30 is the rate that takes 12.2 holds per path down to the human 8.6
    print(f"\n{'removed':>9}{'holds/path':>12}{'time held':>11}"
          f"{'zero steps':>12}{'contract AUC':>14}")
    rows = []
    for f in args.fracs:
        rng = np.random.default_rng(args.seed)
        cut = [drop_holds(t, f, rng) for t in arm]
        st = measure(cut, np.random.default_rng(args.seed))
        X = features_with_jitter(cut, 0.0, args.seed)
        X = X[np.all(np.isfinite(X), axis=1)]
        auc = float(scoring.score_features(X)["auc_rf_oob"])
        print(f"{f:>9.2f}{st['holds_per_path']:>12.2f}"
              f"{st['hold_time_frac']:>11.1%}"
              f"{st['resampled_zero_step_frac']:>12.3%}{auc:>14.4f}")
        rows.append({"frac_removed": f, "holds_per_path": st["holds_per_path"],
                     "hold_time_frac": st["hold_time_frac"],
                     "zero_step_frac": st["resampled_zero_step_frac"],
                     "auc_rf_oob": auc, "n": int(len(X))})

    best = min(rows, key=lambda r: r["auc_rf_oob"])
    base = rows[0]["auc_rf_oob"]
    print(f"\nbaseline {base:.4f}, best {best['auc_rf_oob']:.4f} at "
          f"{best['frac_removed']:.0%} removed, move {best['auc_rf_oob']-base:+.4f}")
    print("human targets: 8.57 holds/path, 9.7% time held, 5.286% zero steps")

    Path(args.out).write_text(json.dumps(
        {"seed": args.seed, "sweep": rows, "baseline_auc": base,
         "best": best, "wall_sec": time.time() - t0}, indent=2))
    print(f"\n[surgery] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
