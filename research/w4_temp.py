"""Sampling temperature, on the corrected configuration, paired across seeds.

Everything W4 has measured says the same shape of thing. `w4_featmap` said the
detector's power is DIFFUSE, no feature or group carrying it. `w4_block` says
the gap is not concentrated in any correlation block. `w4_modes` says it is not
a missing mixture component. `w4_tail` says pairwise higher order dependence
differs on 3 of 153 pairs, with mean absolute excess 0.064 human against 0.058
generated. There is no concentrated defect at any order; the model is slightly
wrong nearly everywhere and a forest with eighteen inputs aggregates that.

The one generation time control that acts uniformly on a uniformly slight error
is the sampling temperature, and it has never been swept on this path with the
duration prior and the lattice snap set correctly. The contract's own collapse
flag has been firing on the acceleration and jerk features, which is
UNDER dispersion, so the direction to test is mainly above 1.0.

One scalar, applied to every trajectory, one trajectory per request. Nothing
generated twice, nothing selected, nothing read that serving would not have.

Paired: within a seed the specs, the duration draws and the torch stream are
identical across temperatures, so a temperature to temperature difference is
attributable to the temperature. The seed to seed spread is the noise floor, and
on this configuration replicates read 0.6519, 0.6443 and 0.6563, so that floor
is around 0.006 and no single arm difference below about 0.015 means anything.

Safety. Scores through research/autoloop/scoring.py only. Never modifies scoring
code, never training/candi_polar_flow_best.pt. Paces itself on GPU temperature:
this machine crashed on this workload on 2026-08-06, kill line tightened to 79C.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        EVENT_SNAP=2.5 EVENT_DUR_STD=1.0 DUR_EMPIRICAL=1 \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_temp.py --seeds 3
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
from models.event_ar import EventARModel  # noqa: E402
from w4_latent import cooldown, gpu_c  # noqa: E402
from w4_paired import gen, specs_for  # noqa: E402

TEMPS = (0.9, 1.0, 1.05, 1.1, 1.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--temps", default=None)
    ap.add_argument("--out", default="research/w4_temp.json")
    args = ap.parse_args()

    temps = ([float(x) for x in args.temps.split(",")] if args.temps
             else list(TEMPS))

    dev = esp._DEVICE
    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])

    print(f"\n  EVENT_DUR_STD={os.environ['EVENT_DUR_STD']} "
          f"DUR_EMPIRICAL={os.environ['DUR_EMPIRICAL']} "
          f"EVENT_SNAP={esp._SNAP}")
    print(f"  {args.n} specs, {args.seeds} seeds, {len(temps)} temperatures, "
          f"one trajectory each, no selection\n")
    print(f"  {'seed':>6}{'temp':>7}{'contract':>10}{'collapse':>10}"
          f"{'n':>7}{'gpu':>6}")

    res = {t: [] for t in temps}
    col = {t: [] for t in temps}
    with torch.no_grad():
        for sd in range(args.seeds):
            rows, meta = specs_for(args.n, args.seed, sd)
            for t in temps:
                cooldown()
                F = gen(model, rows, meta, args.batch, t, dev, sd)
                r = scoring.score_features(F)
                a = float(r["auc_rf_oob"])
                res[t].append(a)
                col[t].append(len(r["collapse_features"]))
                print(f"  {sd:>6}{t:>7.2f}{a:>10.4f}"
                      f"{len(r['collapse_features']):>10}{len(F):>7}"
                      f"{gpu_c():>6}", flush=True)

    print(f"\n  {'temp':>7}{'mean AUC':>11}{'sd':>9}{'collapse':>11}"
          f"{'vs 1.0 paired':>15}")
    base = np.array(res[1.0]) if 1.0 in res else None
    out = {"auc": {str(t): res[t] for t in temps},
           "collapse": {str(t): col[t] for t in temps},
           "n": args.n, "seeds": args.seeds}
    for t in temps:
        v = np.array(res[t])
        if base is not None:
            d = v - base
            pd = f"{d.mean():+.4f} sd {d.std(ddof=1):.4f}" if len(d) > 1 else ""
            out[f"paired_{t}"] = {"mean": float(d.mean()),
                                  "sd": float(d.std(ddof=1)) if len(d) > 1
                                  else 0.0}
        else:
            pd = ""
        print(f"  {t:>7.2f}{v.mean():>11.4f}{v.std(ddof=1):>9.4f}"
              f"{np.mean(col[t]):>11.2f}{pd:>15}")

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n  a paired difference under about 0.015 is inside the replicate "
          f"noise measured on this configuration\n  wrote {args.out}\n")


if __name__ == "__main__":
    main()
