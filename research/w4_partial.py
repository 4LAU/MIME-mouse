"""Is the coupling error just the command showing through?

`w4_couplemap` found the model's speed features are far more negatively coupled
to its turning features than a person's, ten out of ten largest errors, same
sign. `w4_speedturn` then refuted the obvious explanation. Event by event the
model turns at speed exactly as often as a person does: sharp turn share in the
three fastest bands 0.032, 0.035, 0.039 against a human 0.032, 0.027, 0.037,
and within path speed against turn correlation -0.225 against a human -0.203.
What it emits is right. What its whole paths add up to is not.

This is the second time today that pattern has appeared. `w4_stillcal` found
the still conditional correct to three decimals while the sampled rate came out
five times too low. Local statistics right, trajectory aggregate wrong, twice.

One mechanism explains both. The model is handed a distance and a duration and
generates against them. Given a fixed distance, a longer commanded duration
means a slower path, and a slower path turns more per event, so speed and
curvature get locked into a trade off that the command itself creates. A person
handed the same distance and the same duration produces paths that still differ
a lot in how curved they are, for reasons no command captures. If that is the
story, then controlling for duration should collapse the model's coupling error
and leave the human's roughly alone.

The test is a partial Spearman correlation. Rank every column, regress the two
features of interest on the control columns, and correlate what is left. Three
control sets:

  none            the raw rank correlation, reproducing `w4_couplemap`
  duration        movement_duration held fixed
  dur + speed     movement_duration and mean_velocity held fixed, which pins
                  the commanded distance too since distance is close to their
                  product

If the model's error survives all three, the command is not the cause and the
extra variation a person has is somewhere else entirely.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_partial.py
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from scipy.stats import rankdata

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import scoring  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402

PAIRS = [
    ("max_velocity", "angular_velocity_std"),
    ("max_velocity", "angular_velocity_mean"),
    ("std_velocity", "angular_velocity_mean"),
    ("std_velocity", "angular_velocity_std"),
    ("max_acceleration", "angular_velocity_std"),
    ("max_deviation", "angular_velocity_std"),
    ("max_acceleration", "angular_velocity_mean"),
    ("max_velocity", "curvature_mean"),
    ("max_acceleration", "num_direction_changes"),
    ("std_acceleration", "num_direction_changes"),
]
CONTROLS = [("none", []),
            ("duration", ["movement_duration"]),
            ("dur + speed", ["movement_duration", "mean_velocity"])]


def partial_rho(X, a, b, ctrl):
    """Spearman of columns a and b with the ctrl columns regressed out."""
    R = np.column_stack([rankdata(X[:, c]) for c in (a, b, *ctrl)])
    R = (R - R.mean(0)) / R.std(0)
    u, v = R[:, 0], R[:, 1]
    if ctrl:
        C = np.column_stack([np.ones(len(R)), R[:, 2:]])
        u = u - C @ np.linalg.lstsq(C, u, rcond=None)[0]
        v = v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    return float(np.corrcoef(u, v)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="research/w4_ar_features.npy")
    ap.add_argument("--out", default="research/w4_partial.json")
    args = ap.parse_args()

    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    F = np.load(args.gen)
    F = F[np.all(np.isfinite(F), 1)]
    H = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    H = H[np.all(np.isfinite(H), 1)]
    print(f"  {len(F):,} generated, {len(H):,} human\n")

    out = {}
    hdr = "".join(f"{lbl + ' H':>14}{lbl + ' G':>10}{'err':>8}"
                  for lbl, _ in CONTROLS)
    print(f"  {'pair':<44}{hdr}")
    for a, b in PAIRS:
        cells, rec = [], {}
        for lbl, cn in CONTROLS:
            ctrl = [idx[c] for c in cn if c not in (a, b)]
            rh = partial_rho(H, idx[a], idx[b], ctrl)
            rg = partial_rho(F, idx[a], idx[b], ctrl)
            rec[lbl] = dict(human=rh, gen=rg, err=rg - rh)
            cells.append(f"{rh:>14.3f}{rg:>10.3f}{rg - rh:>8.3f}")
        out[f"{a} + {b}"] = rec
        print(f"  {a + ' + ' + b:<44}" + "".join(cells), flush=True)

    for lbl, _ in CONTROLS:
        e = [abs(out[k][lbl]["err"]) for k in out]
        print(f"\n  mean absolute error with control '{lbl}': "
              f"{np.mean(e):.4f}", flush=True)

    json.dump(out, open(args.out, "w"), indent=2)
    print("\n  if the error column shrinks toward zero as controls are added,")
    print("  the coupling defect is the command showing through and the fix is")
    print("  variation the command does not carry; if it does not shrink, the")
    print("  command is not the cause")


if __name__ == "__main__":
    main()
