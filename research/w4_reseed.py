"""Served configuration feature caches at further seeds.

Every structural conclusion in the 2026-08-07 entries of HANDOFF.md rests on one
generated sample, `research/w4_snap_cache.npz`, drawn at a single seed. The
replicate spread on this configuration is about 0.006 of contract AUC, which is
small, but `w4_modes` reports a component populated at 0.65 of the human rate
and `w4_tail` reports three pairs of a hundred and fifty three beyond three
sigma, and neither of those is an AUC. Nothing has ever checked whether they
reproduce.

This writes the same cache at further seeds, in the same format and under the
same key, so `w4_modes`, `w4_block`, `w4_tail` and `w4_copula` can be pointed at
them with `--cache` and no other change.

One trajectory per spec, nothing generated twice, nothing selected. The specs
themselves are held fixed across seeds; only the duration draws and the sampling
stream move, which is the thing being resampled.

Safety. Scores through research/autoloop/scoring.py only. Never modifies scoring
code, never training/candi_polar_flow_best.pt. Paces itself on GPU temperature:
this machine crashed on this workload on 2026-08-06, kill line tightened to 79C.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_reseed.py --rngseeds 1,2
"""
from __future__ import annotations

import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rngseeds", default="1,2")
    args = ap.parse_args()

    dev = esp._DEVICE
    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])

    print(f"\n  EVENT_SNAP={esp._SNAP} "
          f"DUR_EMPIRICAL={os.environ['DUR_EMPIRICAL']} "
          f"EVENT_DUR_STD={os.environ['EVENT_DUR_STD']}")
    print(f"  {'rngseed':>8}{'contract':>10}{'collapse':>10}{'n':>7}{'gpu':>6}"
          f"   cache")
    with torch.no_grad():
        for s in [int(x) for x in args.rngseeds.split(",")]:
            cooldown()
            rows, meta = specs_for(args.n, args.seed, s)
            F = gen(model, rows, meta, args.batch, 1.0, dev, s)
            r = scoring.score_features(F)
            path = f"research/w4_snap_cache_s{s}.npz"
            np.savez_compressed(path, F=F)
            print(f"  {s:>8}{r['auc_rf_oob']:>10.4f}"
                  f"{len(r['collapse_features']):>10}{len(F):>7}{gpu_c():>6}"
                  f"   {path}", flush=True)
    print()


if __name__ == "__main__":
    main()
