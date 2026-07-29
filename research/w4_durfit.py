"""Duration is implicated twice. Where exactly is it wrong?

`w4_whatsees` found `movement_duration` is the single most detectable feature
the detector has, 0.5699 alone against 0.5310 for the next one and chance for
eight of the eighteen. `w4_partial` then found that holding duration and mean
velocity fixed collapses the model's cross feature coupling error from a mean
absolute 0.169 to 0.025, an eighty five percent reduction, so conditional on
those two the model's behaviour is human.

Duration is also the one quantity the model does not produce. It is drawn from
`esp._duration`, a one dimensional empirical fit of p(duration | distance), and
handed in as part of the conditioning vector. So a wrong duration is not a
training failure, it is a failure of that sampler or of the model honouring it.

Three things can be wrong and this separates them, all on the corpus, no GPU:

  convention    does a human token stream's own intervals sum to the duration
                its conditioning vector claims? If not, every comparison below
                is measuring the wrong thing and nothing else in this file is
                meaningful.
  sampler       for the SAME distances a person actually moved, draw from
                `esp._duration` and compare the conditional mean and the
                conditional spread against what those people did. A sampler
                that is right on average and too narrow or too wide in spread
                produces exactly the kind of marginal error the detector sees.
  spec drift    do the evaluation specs from `make_specs` cover the same
                distances as the human recordings? If they do not, the sampler
                is being asked for durations at distances it was never fitted
                on, and the defect is in the harness rather than the model.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_durfit.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

os.environ.setdefault("DUR_EMPIRICAL", "1")
os.environ.setdefault("EVENT_DUR_STD", "1.0")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
from phase_a_baseline import make_specs  # noqa: E402

EDGES = [0, 100, 200, 350, 500, 750, 1100, 1e9]


def _bucket(d):
    return int(np.searchsorted(EDGES, d, side="right") - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="research/w4_durfit.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cond = np.load("training/events_cond.npy")
    dt = np.load("training/events_dt.npy", mmap_mode="r")
    lengths = np.load("training/events_len.npy")
    out = {}

    pick = np.sort(rng.choice(len(cond), min(args.n, len(cond)),
                              replace=False))
    ld, lu = cond[pick, 0].astype(np.float64), cond[pick, 1].astype(np.float64)

    chk = pick[:400]
    claimed = np.exp(cond[chk, 1].astype(np.float64))
    actual = np.array([float(np.asarray(dt[i, :int(min(lengths[i], 256))],
                                        dtype=np.float64).sum())
                       for i in chk])
    ratio = actual / np.maximum(claimed, 1e-9)
    out["convention"] = dict(ratio_p10=float(np.percentile(ratio, 10)),
                             ratio_p50=float(np.median(ratio)),
                             ratio_p90=float(np.percentile(ratio, 90)))
    print(f"  interval sum over claimed duration: p10 "
          f"{np.percentile(ratio, 10):.3f}  p50 {np.median(ratio):.3f}  "
          f"p90 {np.percentile(ratio, 90):.3f}\n", flush=True)

    sampled = np.array([math.log(esp._duration.sample(float(x))) for x in ld])
    print(f"  {'distance px':<14}{'n':>7}{'humanMu':>10}{'sampMu':>10}"
          f"{'dMu':>8}{'humanSd':>10}{'sampSd':>10}{'sdRatio':>9}")
    rows = []
    d = np.exp(ld)
    for b in range(len(EDGES) - 1):
        m = (d >= EDGES[b]) & (d < EDGES[b + 1])
        if m.sum() < 40:
            continue
        hm, sm = float(lu[m].mean()), float(sampled[m].mean())
        hs, ss = float(lu[m].std()), float(sampled[m].std())
        lbl = f"{EDGES[b]:.0f}-{EDGES[b+1]:.0f}" if EDGES[b + 1] < 1e8 \
            else f"{EDGES[b]:.0f}+"
        rows.append(dict(bucket=lbl, n=int(m.sum()), human_mu=hm, samp_mu=sm,
                         human_sd=hs, samp_sd=ss, sd_ratio=ss / hs))
        print(f"  {lbl:<14}{m.sum():>7}{hm:>10.3f}{sm:>10.3f}{sm - hm:>8.3f}"
              f"{hs:>10.3f}{ss:>10.3f}{ss / hs:>9.3f}", flush=True)
    out["by_distance"] = rows
    out["overall"] = dict(human_mu=float(lu.mean()), samp_mu=float(sampled.mean()),
                          human_sd=float(lu.std()), samp_sd=float(sampled.std()),
                          sd_ratio=float(sampled.std() / lu.std()))
    print(f"\n  overall log duration: human mu {lu.mean():.3f} sd "
          f"{lu.std():.3f}, sampled mu {sampled.mean():.3f} sd "
          f"{sampled.std():.3f}, sd ratio "
          f"{sampled.std() / lu.std():.3f}", flush=True)

    specs = make_specs(args.n, args.seed)
    sd = np.array([math.hypot(ex - sx, ey - sy) for sx, sy, ex, ey in specs])
    sd = sd[sd > 1e-6]
    qs = [5, 25, 50, 75, 95]
    hq = [float(np.percentile(d, q)) for q in qs]
    sq = [float(np.percentile(sd, q)) for q in qs]
    out["spec_drift"] = dict(quantiles=qs, human_px=hq, spec_px=sq)
    print(f"\n  distance percentiles  " + "".join(f"{q:>10}" for q in qs))
    print(f"  human recordings      " + "".join(f"{v:>10.1f}" for v in hq))
    print(f"  evaluation specs      " + "".join(f"{v:>10.1f}" for v in sq))

    json.dump(out, open(args.out, "w"), indent=2)
    print("\n  a sd ratio far from 1 means the duration sampler has the right")
    print("  average and the wrong spread, which is a marginal error the")
    print("  detector reads directly and no amount of training can fix")


if __name__ == "__main__":
    main()
