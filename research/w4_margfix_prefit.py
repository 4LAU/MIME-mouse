"""w4_margfix, the CPU half. Fit the first bias vector without touching the GPU.

Measures the corpus token marginal against the model's ALREADY SAVED free
running marginal, forms the first round bias, and reports where the correction
actually lands. Costs nothing and validates the fitting code before the GPU
half spends four hours of sampling on it.

Nothing here decides anything. The primary is the contract, and the contract
needs new samples.
"""
from __future__ import annotations

import os
import sys

import numpy as np

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.event_ar import DT_MAX_MS                              # noqa: E402
from models.event_stream_polar import (S_PAD_CLASS, TH_BINS,       # noqa: E402
                                       TICK_CLASS)
from w4_detcap import corpus_ids, corpus_tokens                    # noqa: E402
from w4_margfix import marginals, tv, update                       # noqa: E402

STREAMS = [f"research/w4_texcover_streams_s{i}.npz" for i in (0, 1)]
HUMAN_ROWS = 200000
NAMES = ("s", "th", "dt")


def main():
    d0 = np.load(STREAMS[0])
    print(f"  model streams {STREAMS}  ckpt {d0['ckpt']}  temps "
          f"s {float(d0['s_temp']):.2f} th {float(d0['th_temp']):.2f} "
          f"dt {float(d0['dt_temp']):.2f}\n", flush=True)

    acc = []
    for f in STREAMS:
        d = np.load(f)
        acc.append(marginals(d["s"].astype(np.int64), d["th"].astype(np.int64),
                             d["dt"].astype(np.int64)))
    PM = tuple(np.mean([p[i] for p in acc], axis=0) for i in range(3))

    hs, hth, hdt, _, hL = corpus_tokens(
        corpus_ids(np.random.default_rng(3), HUMAN_ROWS))
    PH = marginals(hs, hth, hdt, hL)
    print(f"  human target from {HUMAN_ROWS} corpus rows\n")

    print(f"  {'ch':>4}{'classes':>9}{'tv':>9}{'kl_h||m':>10}"
          f"{'m support gaps':>16}")
    for i, nm in enumerate(NAMES):
        h, m = PH[i], PM[i]
        kl = float((h[h > 0] * np.log(h[h > 0] / (m[h > 0] + 1e-12))).sum())
        gaps = int(((h > 0) & (m == 0)).sum())
        print(f"  {nm:>4}{len(h):>9}{tv(h, m):>9.4f}{kl:>10.4f}{gaps:>16}")

    print("\n  first round bias, lambda 1.0, centred on the model rates")
    bias = [update(np.zeros(len(PH[i])), PH[i], PM[i], 1.0) for i in range(3)]
    for i, nm in enumerate(NAMES):
        b, m = bias[i], PM[i]
        top = np.argsort(-np.abs(b) * (m + PH[i]))[:5]
        mass = float((m * np.abs(b)).sum())
        print(f"  {nm:>4}  |b| max {np.abs(b).max():>7.3f}   rate weighted "
              f"mean |b| {mass:>6.4f}   clipped {int((np.abs(b) >= 2.0).sum()):>4}"
              f"   biggest movers {list(zip(top.tolist(), np.round(b[top], 2).tolist()))}")

    # the classes the bias must not touch, checked rather than assumed
    print(f"\n  s bias covers classes 0..{S_PAD_CLASS - 1}, PAD {S_PAD_CLASS} "
          f"is left at zero")
    print(f"  th bias covers classes 0..{TH_BINS - 1}, NULL is left at zero")
    print(f"  dt bias covers classes 0..{DT_MAX_MS}")
    print(f"  TICK class {TICK_CLASS} model rate {PM[0][TICK_CLASS]:.4f} "
          f"human {PH[0][TICK_CLASS]:.4f}  bias {bias[0][TICK_CLASS]:+.3f}")

    np.savez("research/w4_margfix_prefit.npz",
             ph_s=PH[0], ph_th=PH[1], ph_dt=PH[2],
             pm_s=PM[0], pm_th=PM[1], pm_dt=PM[2],
             b_s=bias[0], b_th=bias[1], b_dt=bias[2])
    print("\n  wrote research/w4_margfix_prefit.npz")


if __name__ == "__main__":
    main()
