"""w4_mserve aggregator. Reads registered in AMENDMENT 19, step0_prereg.md.

READ 1 (PRIMARY, serve decision): paired mq1 minus k0, ten seeds.
mean <= -0.008 at 2 paired se -> SERVE (headline = mq1 mean); |mean| <=
0.004 -> NO SERVE; else BETWEEN, no rerun without a design change.
READ 2 (support only): mq1 mean vs the fifteen seed baseline 0.5881
(se 0.0022), cross run t. Cannot overturn READ 1.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import ledger  # noqa: E402

BASELINE_MEAN, BASELINE_SE = 0.5881, 0.0022


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="research")
    ap.add_argument("--no-ledger", action="store_true")
    a = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(a.dir, "w4_mserve_s*.json"))
                   if re.search(r"_s\d+\.json$", f))
    runs = [json.load(open(f)) for f in files]
    if not runs:
        print("no runs found"); sys.exit(1)
    runs.sort(key=lambda r: r["seed"])
    seeds = [r["seed"] for r in runs]

    k0 = np.array([r["arms"]["k0"]["contract"] for r in runs])
    mq1 = np.array([r["arms"]["mq1"]["contract"] for r in runs])

    print(f"  seeds {seeds}")
    print(f"  {'seed':>6}{'k0':>9}{'mq1':>9}{'diff':>9}")
    for i, r in enumerate(runs):
        print(f"  {r['seed']:>6}{k0[i]:>9.4f}{mq1[i]:>9.4f}{mq1[i]-k0[i]:>+9.4f}")

    d = mq1 - k0
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    t = m / se if se > 0 else float("inf")
    print(f"\n  READ 1 (PRIMARY), mq1 minus k0 paired: mean {m:+.4f}  se {se:.4f}"
          f"  t {t:+.2f}")
    if m <= -0.008 and abs(m) >= 2 * se:
        verdict = "SERVE"
    elif abs(m) <= 0.004:
        verdict = "NO SERVE"
    else:
        verdict = "BETWEEN"
    print(f"  VERDICT: {verdict}")

    mm, mse = mq1.mean(), mq1.std(ddof=1) / np.sqrt(len(mq1))
    ct = (mm - BASELINE_MEAN) / np.hypot(mse, BASELINE_SE)
    print(f"\n  READ 2 (support), mq1 mean {mm:.4f} se {mse:.4f} vs baseline "
          f"{BASELINE_MEAN} se {BASELINE_SE}: cross run t {ct:+.2f}")

    if not a.no_ledger:
        rid = ledger.append_row(
            "w4_mserve",
            {"seeds": seeds, "n": 2000, "arms": ["k0", "mq1"],
             "k0": "protocol durations, no q",
             "mq1": "matched k=64 heldout durations plus full q temps 1,1,1"},
            "ok",
            metrics={"k0_mean": float(k0.mean()), "mq1_mean": float(mm),
                     "mq1_se": float(mse),
                     "mq1_minus_k0": float(m), "mq1_minus_k0_se": float(se),
                     "vs_baseline_t": float(ct)},
            artifacts=files,
            notes=f"AMENDMENT 19 serve candidate test, ten seeds, paired. "
                  f"{verdict}. Registered in advance.",
            tier=1)
        ledger.regenerate_leaderboard()
        print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
