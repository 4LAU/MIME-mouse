"""Aggregate the four arm by five seed contract AUC grid from the coupled head
fine tune, and answer the one question the seed 0 table could not.

WHY THIS EXISTS. At seed 0 the four arms scored base 0.6412, mlp 0.6150,
mlp_nos 0.6365, add 0.6130. The coupled arm, which had bought 0.2848 held out
nats, scored WORSE than the architecturally unchanged `add` control, which had
bought 0.0029. Read literally that says the 0.28 nats were worth nothing. But
the four numbers span 0.028 and `research/w4_arcurve.py` records contract AUC
moving plus or minus 0.03 run to run, so at one seed the table cannot separate
any arm from any other and a literal reading would be a reading of noise.

WHAT THIS SETTLES. Five seeds per arm gives a within arm standard error, which
turns three loose questions into arithmetic:

  1. Is the seed to seed noise actually 0.03, or is that band, quoted from a
     different experiment, too wide for this one? Everything else depends on it.
  2. Does the coupled arm differ from `add`? That is the contrast the whole
     pre registration was about, mlp minus add, isolating the coupling from the
     fine tuning that both arms received.
  3. Across all twenty points, does held out likelihood still predict contract
     AUC at all? Regressing AUC on nats over this family gives an in family
     exchange rate with an error bar, to be read against the 0.1904 per nat that
     `w4_arcurve` fitted along a single training trajectory. If that slope is
     indistinguishable from zero the screening instrument this programme has
     steered by is dead on this model, which is exactly what P3 of
     /home/aaronadmin/w4_arms/headmlp_prereg.md registered in advance as the
     outcome that must not be buried.

Paired reading. Every arm is scored at the same five seeds, and the seed sets
the specification sample the arms are asked to satisfy, so seed is a genuine
blocking factor and the paired contrast has less variance than the unpaired
one. Both are printed. If they disagree the paired one is the one that answers
question 2, and the gap between them is how much of the spread was the
specifications rather than the models.

This reads existing JSON only. It runs no model and scores nothing, so it can
be re run freely.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# Held out th nats gained over the shared 4.3834 start, from the three matched
# 6000 step fine tunes. The base is the checkpoint they all started from, so by
# construction it gained nothing.
NATS = {
    "event_ar_v2_s40000": 0.0,
    "event_ar_hm_add": 0.0029,
    "event_ar_hm_mlp_nos": 0.0062,
    "event_ar_hm_mlp": 0.2848,
}
LABEL = {
    "event_ar_v2_s40000": "base",
    "event_ar_hm_add": "add     (unchanged)",
    "event_ar_hm_mlp_nos": "mlp_nos (depth ctl)",
    "event_ar_hm_mlp": "mlp     (COUPLED)",
}
ARCURVE_SLOPE = 0.1904  # AUC per nat, fitted across 8 v2 snapshots


def load(ck: str, seed: int) -> float | None:
    """Contract AUC for one arm at one seed, or None if that run is missing.

    Seed 0 was written by run_headmlp_score.sh under an unsuffixed name; every
    later seed carries an _s<seed> suffix.
    """
    stem = f"w4_hmscore_{ck}" + ("" if seed == 0 else f"_s{seed}")
    p = REPO / "research" / f"{stem}.json"
    if not p.exists():
        return None
    return float(json.loads(p.read_text())["t1.0"]["contract"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()

    grid, missing = {}, []
    for ck in NATS:
        vals = []
        for s in a.seeds:
            v = load(ck, s)
            if v is None:
                missing.append(f"{ck} s{s}")
            else:
                vals.append((s, v))
        grid[ck] = vals
    if missing:
        print(f"  MISSING {len(missing)}: {', '.join(missing)}")

    print("\n=== contract AUC, four arms by seed ===")
    print(f"  {'arm':<21} {'nats':>7} " +
          " ".join(f"{'s'+str(s):>7}" for s in a.seeds) +
          f" {'mean':>8} {'se':>7}")
    stats = {}
    for ck, vals in grid.items():
        d = dict(vals)
        arr = np.array([v for _, v in vals], dtype=float)
        m = arr.mean()
        # se of the mean; ddof=1 because these seeds are a sample of the
        # sampler's randomness, not the whole population of it.
        se = arr.std(ddof=1) / np.sqrt(arr.size) if arr.size > 1 else float("nan")
        stats[ck] = (m, se, arr, d)
        cells = " ".join(f"{d[s]:7.4f}" if s in d else f"{'.':>7}" for s in a.seeds)
        print(f"  {LABEL[ck]:<21} {NATS[ck]:7.4f} {cells} {m:8.4f} {se:7.4f}")

    # Question 1. How big is the seed to seed noise really? Pool the within arm
    # sd across all four arms; each arm is a fixed model so all of its spread is
    # sampler and specification noise.
    pooled = np.sqrt(np.mean([stats[c][2].var(ddof=1) for c in NATS]))
    print(f"\n  pooled within arm sd across seeds  {pooled:.4f}")
    print(f"  w4_arcurve quoted run to run band  0.0300")
    print("  -> the quoted band is "
          f"{0.03 / pooled:.1f}x the sd actually measured here")

    # Question 2. mlp minus add, the contrast the pre registration was about.
    # Paired first: same seed means the same specification sample, so seed is a
    # blocking factor and differencing inside it removes that variance.
    print("\n=== the contrast that decides it: COUPLED minus unchanged ===")
    dm, da = stats["event_ar_hm_mlp"], stats["event_ar_hm_add"]
    shared = sorted(set(dm[3]) & set(da[3]))
    diffs = np.array([dm[3][s] - da[3][s] for s in shared])
    pd_m = diffs.mean()
    pd_se = diffs.std(ddof=1) / np.sqrt(diffs.size) if diffs.size > 1 else float("nan")
    print(f"  paired over {diffs.size} seeds   {pd_m:+.4f}  se {pd_se:.4f}"
          f"   t {pd_m / pd_se:+.2f}" if diffs.size > 1 else "")
    un_m = dm[0] - da[0]
    un_se = np.hypot(dm[1], da[1])
    print(f"  unpaired              {un_m:+.4f}  se {un_se:.4f}"
          f"   t {un_m / un_se:+.2f}")
    print("  NOTE sign convention: contract AUC is a detector score, so LOWER "
          "is better.")
    print(f"  exchange rate predicted {(NATS['event_ar_hm_mlp'] - NATS['event_ar_hm_add']) * ARCURVE_SLOPE:.4f} "
          "of AUC improvement, i.e. a difference of "
          f"{-(NATS['event_ar_hm_mlp'] - NATS['event_ar_hm_add']) * ARCURVE_SLOPE:+.4f}")

    # Question 3. Does likelihood predict the contract at all across this
    # family? One point per arm, weighted equally, regressed on nats.
    print("\n=== in family exchange rate: AUC regressed on nats ===")
    x = np.array([NATS[c] for c in NATS])
    y = np.array([stats[c][0] for c in NATS])
    ns = np.array([stats[c][2].size for c in NATS])
    slope, icept = np.polyfit(x, y, 1)
    resid = y - (slope * x + icept)
    # se of the slope from the residuals; only 4 points, so 2 dof.
    ss_x = ((x - x.mean()) ** 2).sum()
    s_err = np.sqrt((resid ** 2).sum() / (len(x) - 2)) / np.sqrt(ss_x)
    print(f"  slope {slope:+.4f} AUC per nat, se {s_err:.4f}  "
          f"(negative = nats help)")
    print(f"  w4_arcurve slope on the v2 trajectory  {-ARCURVE_SLOPE:+.4f}")
    print(f"  n per point {ns.tolist()}, 4 points so 2 dof, treat as indicative")
    lo, hi = slope - 2 * s_err, slope + 2 * s_err
    print(f"  rough 2 se interval [{lo:+.4f}, {hi:+.4f}]  "
          f"{'INCLUDES ZERO' if lo < 0 < hi else 'excludes zero'}")
    print(f"  arcurve rate is {'INSIDE' if lo <= -ARCURVE_SLOPE <= hi else 'OUTSIDE'} that interval")


if __name__ == "__main__":
    main()
