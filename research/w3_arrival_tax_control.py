"""Re-measure the arrival tax with the degeneracy control.

The finding W3 was launched on (HANDOFF_W3.md, row W3_groundwork_...0e65cfd1,
research/w3_landing_price_results.json): forcing a generated path to land on the
requested pixel costs about +0.078 AUC.

  uncorrected (arrives exactly 0.3% of the time)           0.6500
  magnitude-weighted additive correction, 100% arrival     0.7283
  rotate + scale correction, 100% arrival                  0.7342

Every correction shifts each point and re-rounds it, and that is exactly the
operation research/p3_ceiling_probe.py showed erases the exact-collinearity
structure the contract scorer reads (a 1e-9 px nudge alone is worth +0.19 on
real paths). So the +0.078 could be the cost of moving the path, or the cost of
erasing that structure, or both, and the original measurement cannot tell them
apart. research/degeneracy_panel.py can: it re-reads the same three arms with
both sides nudged, which removes the structure from all of them symmetrically.

  tax survives the control    correction really does damage the motion, W3's
                              premise stands
  tax collapses               the +0.078 was mostly arithmetic, and P1's six
                              cycles were chasing a measurement artifact

No generation. research/w3_landing_cache.pkl holds the 6000 one-shot
event_polar_4m_fc_v2 paths and their integer targets from the original run, so
this is the same paths through a second reading. The correction functions are
imported from research/w3_fallback_arrival.py and research/w3_correction_lab.py,
not reimplemented.

Usage:
  env PYTHONPATH=. \
    ~/venvs/mime/bin/python research/w3_arrival_tax_control.py
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

from degeneracy_panel import panel, print_panel  # noqa: E402
from w3_correction_lab import correct_similarity  # noqa: E402
from w3_fallback_arrival import correct_additive  # noqa: E402

CACHE = R / "research" / "w3_landing_cache.pkl"
OUT = R / "research" / "w3_arrival_tax_control_results.json"

# The numbers this is re-reading, from w3_landing_price_results.json.
ORIGINAL = {"none": 0.6500, "additive": 0.7283, "rotate": 0.7342}


def build_arms(specs, trajs):
    """Three arms over the same paths in the same order: uncorrected, and the
    two corrections that force exact arrival on the integer target."""
    arms = {"none": [], "additive": [], "rotate": []}
    for spec, tr in zip(specs, trajs):
        sx, sy, ex, ey = (int(v) for v in spec)
        a = np.asarray(tr, dtype=np.float64)
        arms["none"].append(a)
        if len(a) < 3:
            # too short to correct; keep the row so all arms stay aligned, the
            # panel's shared-validity mask will drop it from every arm at once
            arms["additive"].append(a)
            arms["rotate"].append(a)
            continue
        arms["additive"].append(correct_additive(a, sx, sy, ex, ey))
        arms["rotate"].append(correct_similarity(a, sx, sy, ex, ey))
    return arms


def arrival_rate(arm, specs):
    hit = 0
    for spec, tr in zip(specs, arm):
        ex, ey = int(spec[2]), int(spec[3])
        if abs(tr[-1][0] - ex) < 1e-9 and abs(tr[-1][1] - ey) < 1e-9:
            hit += 1
    return hit / len(arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000,
                    help="reference and holdout size, each")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    # pickle.load: w3_landing_cache.pkl is this repo's own artifact, written by
    # the 2026-07-20 landing-price run on this machine, never third-party input.
    with open(CACHE, "rb") as fh:
        specs, trajs = pickle.load(fh)
    print(f"[tax] {len(trajs)} cached one-shot paths from the original "
          f"landing-price run, no regeneration", flush=True)

    arms = build_arms(specs, trajs)
    for k in arms:
        print(f"[tax] arm {k:9s} exact arrival "
              f"{arrival_rate(arms[k], specs):.1%}", flush=True)

    res = panel(arms, n_paths=args.n, seed=args.seed)
    print_panel(res, "Arrival tax, original reading and controlled reading")

    print("\nTax = corrected minus uncorrected, per column")
    print(f"{'scheme':<14}{'original':>10}{'contract':>10}{'rebuilt':>10}"
          f"{'control':>10}")
    taxes = {}
    for k in ("additive", "rotate"):
        row = {c: res[k][c] - res["none"][c]
               for c in ("contract", "rebuilt", "control")}
        row["original"] = ORIGINAL[k] - ORIGINAL["none"]
        taxes[k] = row
        print(f"{k:<14}{row['original']:>+10.4f}{row['contract']:>+10.4f}"
              f"{row['rebuilt']:>+10.4f}{row['control']:>+10.4f}")

    out = {"n_paths": len(trajs), "n_ref": args.n, "seed": args.seed,
           "original_results": ORIGINAL, "panel": res, "tax": taxes,
           "wall_sec": time.time() - t0}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[tax] wrote {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
