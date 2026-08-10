"""What the lattice snap threshold actually moves.

Eight seeds say raising `EVENT_SNAP` from the served 2.5 to 5.0 is worth 0.0095
of contract AUC with a standard error of 0.0016. The same eight seeds say the
curvature, maximum deviation and path efficiency ratios do not move, so the
straightness account the sweep was built on is refuted and the route is unknown.

Two readouts on one paired pair of samples.

  1. In the contract's own coordinates. Per feature, the change in mean from 2.5
     to 5.0 divided by the HUMAN standard deviation of that feature, so the
     eighteen numbers are comparable, plus the distance from the human mean in
     the same units before and after, which says whether the move is toward the
     human or away. Branched.

  2. In path space. Per step displacement magnitude at the low end, the share of
     steps under one pixel, and the share of steps whose x and y displacements
     both sit within 0.05 of a whole pixel. Reported on the raw decoded grid,
     where snap acts, and on the 125 Hz resampled grid, which is what the
     feature extractor actually sees. Reported, not branched.

Diagnostic. One trajectory per spec, nothing selected, nothing regenerated.
Never touches data/human_eval_features.npy, never modifies scoring code, never
training/candi_polar_flow_best.pt. Paces itself on GPU temperature: this machine
crashed on this workload on 2026-08-06, kill line tightened to 79C.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python -u research/w4_snapwhy.py
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
from features import (FEATURE_NAMES, extract_features,  # noqa: E402
                      resample_trajectory)
from models.event_ar import EventARModel  # noqa: E402
from phase0_critic import (VAL_N, VAL_SEED,  # noqa: E402
                           reconstruct_human_val_paths)
from w4_latent import cooldown, gpu_c  # noqa: E402
from w4_paired import gen_paths, specs_for  # noqa: E402

HZ = 125.0
LAT_TOL = 0.05
PCT = (10, 25, 50, 75, 90)


def featurise(paths):
    """Feature matrix and the surviving paths, filtered together. The contract's
    own extractor drops rows silently, so doing this by hand is the only way the
    path list and the feature matrix cannot fall out of step."""
    F, keep = [], []
    for p in paths:
        f = extract_features(resample_trajectory(p, hz=HZ))
        if f is not None and np.all(np.isfinite(f)):
            F.append(f)
            keep.append(p)
    return np.asarray(F, dtype=np.float64), keep


def steps(paths):
    """Per step displacements pooled over paths, on whatever grid is passed in."""
    return np.vstack([np.diff(np.asarray(p, dtype=np.float64)[:, :2], axis=0)
                      for p in paths if len(p) > 1])


def lattice_row(name, paths):
    raw = steps(paths)
    res = steps([resample_trajectory(p, hz=HZ) for p in paths])
    out = {}
    for grid, d in (("raw", raw), ("resampled", res)):
        mag = np.hypot(d[:, 0], d[:, 1])
        frac = np.abs(d - np.round(d))
        exact = float(np.mean((frac[:, 0] < LAT_TOL) & (frac[:, 1] < LAT_TOL)))
        q = np.percentile(mag, PCT)
        print(f"  {name:<12}{grid:<11}{np.mean(mag < 1.0):>9.3f}{exact:>9.3f}"
              + "".join(f"{v:>9.3f}" for v in q) + f"{len(mag):>10}")
        out[grid] = {"under_one_px": float(np.mean(mag < 1.0)),
                     "lattice_exact": exact,
                     "mag_percentiles": [float(v) for v in q],
                     "n_steps": int(len(mag))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rngseed", type=int, default=8)
    ap.add_argument("--lo", type=float, default=2.5)
    ap.add_argument("--hi", type=float, default=5.0)
    ap.add_argument("--out", default="research/w4_snapwhy.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    H = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)

    print(f"\n  snap {args.lo} against {args.hi}, {args.n} specs, rng seed "
          f"{args.rngseed}, one trajectory each, no selection\n")
    P, F = {}, {}
    with torch.no_grad():
        for s in (args.lo, args.hi):
            cooldown()
            esp._SNAP = s
            rows, meta = specs_for(args.n, args.seed, args.rngseed)
            F[s], P[s] = featurise(gen_paths(model, rows, meta, args.batch, 1.0,
                                             dev, args.rngseed))
            r = scoring.score_features(F[s])
            print(f"  snap {s:.1f}  contract {r['auc_rf_oob']:.4f}  "
                  f"collapse {len(r['collapse_features'])}  n {len(F[s])}  "
                  f"gpu {gpu_c()}", flush=True)

    hm, hs = H.mean(0), H.std(0)
    shift = (F[args.hi].mean(0) - F[args.lo].mean(0)) / hs
    dlo = np.abs(F[args.lo].mean(0) - hm) / hs
    dhi = np.abs(F[args.hi].mean(0) - hm) / hs

    order = np.argsort(-np.abs(shift))
    tot = float(np.abs(shift).sum())
    top3 = float(np.abs(shift)[order[:3]].sum())
    frac = top3 / tot if tot > 0 else 0.0

    print(f"\n  per feature mean shift from snap {args.lo} to {args.hi}, in "
          f"human standard deviations")
    print(f"  {'feature':<26}{'shift':>10}{'gap lo':>10}{'gap hi':>10}{'':>9}")
    for j in order:
        print(f"  {FEATURE_NAMES[j]:<26}{shift[j]:>+10.4f}{dlo[j]:>10.4f}"
              f"{dhi[j]:>10.4f}"
              f"{'  toward' if dhi[j] < dlo[j] else '  away':>9}")
    n_toward = int((dhi < dlo).sum())
    print(f"\n  total absolute shift {tot:.4f}, top three {top3:.4f}, "
          f"{frac:.1%}, {n_toward}/18 features moved toward the human mean")

    print(f"\n  path space")
    print(f"  {'sample':<12}{'grid':<11}{'<1 px':>9}{'lattice':>9}"
          + "".join(f"{'p' + str(q):>9}" for q in PCT) + f"{'n':>10}")
    hp, hf = reconstruct_human_val_paths(VAL_N, VAL_SEED, verbose=False)
    assert np.allclose(np.asarray(hf), H, rtol=1e-6, atol=1e-6), (
        "reconstructed human paths do not reproduce the scorer's human features")
    lat = {"human": lattice_row("human", hp)}
    for s in (args.lo, args.hi):
        lat[f"snap_{s}"] = lattice_row(f"snap {s:.1f}", P[s])

    verdict = ("NAMED ROUTE. The three largest standardised shifts carry "
               f"{frac:.1%} of the total absolute movement, so the knob acts on "
               "a nameable corner of the feature space and that corner can be "
               "attacked directly."
               if frac > 0.60 else
               f"DIFFUSE. The three largest standardised shifts carry {frac:.1%} "
               "of the total absolute movement. The knob nudges the whole "
               "feature vector, like every other real effect in W4, and there is "
               "no single corner to attack.")
    print(f"\n  VERDICT  {verdict}\n")

    json.dump({"lo": args.lo, "hi": args.hi, "n": args.n,
               "rngseed": args.rngseed,
               "shift": {FEATURE_NAMES[j]: float(shift[j])
                         for j in range(len(FEATURE_NAMES))},
               "gap_lo": {FEATURE_NAMES[j]: float(dlo[j])
                          for j in range(len(FEATURE_NAMES))},
               "gap_hi": {FEATURE_NAMES[j]: float(dhi[j])
                          for j in range(len(FEATURE_NAMES))},
               "top3_share": frac, "n_toward": n_toward,
               "lattice": lat, "verdict": verdict},
              open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}\n")


if __name__ == "__main__":
    main()
