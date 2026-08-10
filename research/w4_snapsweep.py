"""The lattice snap threshold, swept, with a mechanism check attached.

Turning `EVENT_SNAP` on at 2.5 was worth 0.0425 of contract AUC this morning,
the largest single improvement in W4's record. The value 2.5 was inherited from
the flow model's serving recipe and has never been moved for this checkpoint.

`research/w4_disp.py` and the percentile table in HANDOFF.md say the model's
paths are systematically less straight than human paths: curvature larger at
every percentile, maximum deviation larger at every percentile, path efficiency
lower at every percentile. `_decode` rounds `dx` and `dy` to whole lattice units
for steps slower than `_SNAP`, which removes exactly the sub pixel jitter that
inflates those three. The knob acts on the defect.

Arms 2.5, 3.5, 5.0, 8.0. Specs, duration draws and the torch stream are paired
inside a seed, so an arm to arm difference at a fixed seed is attributable to the
threshold. One trajectory per spec, nothing generated twice, nothing selected.

The mechanism check. An AUC improvement counts as a straightness correction only
if the median ratios of curvature_mean, max_deviation and path_efficiency all
move toward one. An improvement without that is an unexplained win.

Safety. Scores through research/autoloop/scoring.py only. Never modifies scoring
code, never training/candi_polar_flow_best.pt. Paces itself on GPU temperature:
this machine crashed on this workload on 2026-08-06, kill line tightened to 79C.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_snapsweep.py --seeds 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

os.environ.setdefault("EVENT_CHOICE_TEMP", "10")
os.environ.setdefault("EVENT_DUR_STD", "1.0")
os.environ.setdefault("DUR_EMPIRICAL", "1")
os.environ.setdefault("EVENT_SNAP", "2.5")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
import scoring  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402
from models.event_ar import EventARModel  # noqa: E402
from w4_latent import cooldown, gpu_c  # noqa: E402
from w4_paired import gen, specs_for  # noqa: E402

SNAPS = (2.5, 3.5, 5.0, 8.0)
MECH = ("curvature_mean", "max_deviation", "path_efficiency")


def mech_ratios(F, H):
    """Median of the generated over median of the human, per mechanism feature.
    One is correct; the served configuration reads about 1.58, 1.39 and 0.96."""
    return {n: float(np.median(F[:, FEATURE_NAMES.index(n)])
                     / np.median(H[:, FEATURE_NAMES.index(n)]))
            for n in MECH}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=0,
                    help="first rng seed; a confirmation run uses fresh ones")
    ap.add_argument("--snaps", default=None)
    ap.add_argument("--out", default="research/w4_snapsweep.json")
    args = ap.parse_args()

    snaps = ([float(x) for x in args.snaps.split(",")] if args.snaps
             else list(SNAPS))
    dev = esp._DEVICE
    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    H = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)

    print(f"\n  EVENT_DUR_STD={os.environ['EVENT_DUR_STD']} "
          f"DUR_EMPIRICAL={os.environ['DUR_EMPIRICAL']}, snap swept")
    print(f"  {args.n} specs, {args.seeds} seeds, {len(snaps)} arms, "
          f"one trajectory each, no selection\n")
    print(f"  {'seed':>5}{'snap':>6}{'contract':>10}{'collapse':>10}{'n':>7}"
          + "".join(f"{n.split('_')[0][:8]:>10}" for n in MECH) + f"{'gpu':>6}")

    res = {s: [] for s in snaps}
    col = {s: [] for s in snaps}
    mech = {s: [] for s in snaps}
    with torch.no_grad():
        for sd in range(args.seed0, args.seed0 + args.seeds):
            for s in snaps:
                cooldown()
                esp._SNAP = s
                rows, meta = specs_for(args.n, args.seed, sd)
                F = gen(model, rows, meta, args.batch, 1.0, dev, sd)
                r = scoring.score_features(F)
                m = mech_ratios(F, H)
                res[s].append(float(r["auc_rf_oob"]))
                col[s].append(len(r["collapse_features"]))
                mech[s].append(m)
                print(f"  {sd:>5}{s:>6.1f}{r['auc_rf_oob']:>10.4f}"
                      f"{len(r['collapse_features']):>10}{len(F):>7}"
                      + "".join(f"{m[n]:>10.3f}" for n in MECH)
                      + f"{gpu_c():>6}", flush=True)

    base = np.array(res[snaps[0]])
    print(f"\n  {'snap':>6}{'mean AUC':>11}{'sd':>9}{'collapse':>10}"
          f"{'paired vs ' + str(snaps[0]):>18}")
    out = {"auc": {str(s): res[s] for s in snaps},
           "collapse": {str(s): col[s] for s in snaps},
           "mech": {str(s): mech[s] for s in snaps},
           "n": args.n, "seeds": args.seeds}
    for s in snaps:
        v = np.array(res[s])
        d = v - base
        txt = (f"{d.mean():+.4f} sd {d.std(ddof=1):.4f}" if len(d) > 1
               else f"{d.mean():+.4f}")
        out[f"paired_{s}"] = {"mean": float(d.mean()),
                              "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                              "wins": int((d < 0).sum())}
        print(f"  {s:>6.1f}{v.mean():>11.4f}{v.std(ddof=1):>9.4f}"
              f"{np.mean(col[s]):>10.2f}{txt:>18}")

    best = min(snaps[1:], key=lambda s: out[f"paired_{s}"]["mean"])
    d = out[f"paired_{best}"]
    b0 = {n: float(np.mean([m[n] for m in mech[snaps[0]]])) for n in MECH}
    b1 = {n: float(np.mean([m[n] for m in mech[best]])) for n in MECH}
    toward = all(abs(b1[n] - 1.0) < abs(b0[n] - 1.0) for n in MECH)
    print(f"\n  best arm {best}, paired {d['mean']:+.4f}, "
          f"{d['wins']}/{args.seeds} seeds improved")
    print(f"  mechanism ratios to human medians, {snaps[0]} then {best}")
    for n in MECH:
        print(f"    {n:<22}{b0[n]:>9.3f}{b1[n]:>9.3f}"
              f"{'  toward 1' if abs(b1[n] - 1) < abs(b0[n] - 1) else '  away'}")

    rose = np.mean(col[best]) > np.mean(col[snaps[0]])
    if rose:
        verdict = (f"LOSS. The collapse count rises from "
                   f"{np.mean(col[snaps[0]]):.1f} to {np.mean(col[best]):.1f} "
                   "at the best AUC arm, and the registration reports that as a "
                   "loss regardless of AUC.")
    elif d["mean"] <= -0.015 and d["wins"] == args.seeds and toward:
        verdict = (f"REAL. Snap {best} improves the contract by "
                   f"{-d['mean']:.4f} against 2.5 on every seed, and all three "
                   "mechanism ratios move toward one, so the win is the "
                   "straightness correction it was predicted to be.")
    elif d["mean"] <= -0.015 and d["wins"] == args.seeds:
        verdict = (f"UNEXPLAINED WIN. Snap {best} improves the contract by "
                   f"{-d['mean']:.4f} on every seed but the mechanism ratios do "
                   "not all move toward one, so the registered mechanism is not "
                   "what produced it.")
    else:
        verdict = (f"NOISE. Best arm {best} moves the contract {d['mean']:+.4f} "
                   f"on {d['wins']}/{args.seeds} seeds, inside the 0.015 "
                   "threshold or not consistent in sign. The threshold "
                   "inherited from the flow recipe is not leaving anything on "
                   "the table.")
    out["verdict"] = verdict
    print(f"\n  VERDICT  {verdict}\n")

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}\n")


if __name__ == "__main__":
    main()
