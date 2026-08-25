"""w4_kfill aggregator. Reads registered in AMENDMENT 25,
step0_prereg.md.

In run ladder on seeds 40 to 45: k0 and h04 from the AMENDMENT 22
table (pairing validated twice today, delta 0.0), h02 and h03 from the
kf runs. READ 1 (PRIMARY): rung increments inc23 = h03 minus h02 and
inc34 = h04 minus h03; both |inc| <= 0.005 COMPOUNDING; either
inc <= -0.010 at 2 se CONDITIONALS DEGRADE IN SAMPLE SPACE; else
BETWEEN. READ 2: h02 minus k0 against OUTCOME 11's k=2 (-0.0220).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="research")
    ap.add_argument("--no-ledger", action="store_true")
    a = ap.parse_args()

    kf_files = sorted(f for f in glob.glob(os.path.join(a.dir, "w4_qladder_kf_s*.json"))
                      if re.search(r"_s\d+\.json$", f))
    kf_runs = sorted((json.load(open(f)) for f in kf_files), key=lambda r: r["seed"])
    if not kf_runs:
        print("no kf runs found"); sys.exit(1)
    seeds = [r["seed"] for r in kf_runs]
    a22 = {s: json.load(open(os.path.join(a.dir, f"w4_qladder_s{s}.json")))
           for s in seeds}

    k0 = np.array([a22[s]["arms"]["k0"]["contract"] for s in seeds])
    h04 = np.array([a22[s]["arms"]["h04"]["contract"] for s in seeds])
    h02 = np.array([r["arms"]["h02"]["contract"] for r in kf_runs])
    h03 = np.array([r["arms"]["h03"]["contract"] for r in kf_runs])

    print(f"  seeds {seeds}")
    print("    seed       k0      h02      h03      h04")
    for i, s in enumerate(seeds):
        print(f"  {s:>6}{k0[i]:>9.4f}{h02[i]:>9.4f}{h03[i]:>9.4f}{h04[i]:>9.4f}")

    def stats(d, name):
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        t = m / se if se > 0 else float("inf")
        print(f"  {name}: mean {m:+.4f}  se {se:.4f}  t {t:+.2f}  per seed "
              + " ".join(f"{v:+.4f}" for v in d))
        return m, se, t

    print("\n  READ 1 (PRIMARY), rung increments, each replacing a one step"
          " draw with the human event:")
    m23, se23, _ = stats(h03 - h02, "inc23 (h03 - h02)")
    m34, se34, _ = stats(h04 - h03, "inc34 (h04 - h03)")
    deg23 = m23 <= -0.010 and abs(m23) >= 2 * se23
    deg34 = m34 <= -0.010 and abs(m34) >= 2 * se34
    if abs(m23) <= 0.005 and abs(m34) <= 0.005:
        verdict = "COMPOUNDING"
    elif deg23 or deg34:
        verdict = "CONDITIONALS DEGRADE IN SAMPLE SPACE"
    else:
        verdict = "BETWEEN"
    print(f"  VERDICT: {verdict}")

    print("\n  READ 2, h02 minus k0, against OUTCOME 11's k=2 -0.0220:")
    m2, se2, _ = stats(h02 - k0, "READ 2")

    print("\n  READ 3 (informational), full ladder means:")
    print(f"    k0 {k0.mean():.4f}  h02 {h02.mean():.4f}  h03 {h03.mean():.4f}"
          f"  h04 {h04.mean():.4f}")

    print("\n  no headline, no serve decision from this arm (registered)")

    if not a.no_ledger:
        rid = ledger.append_row(
            "w4_kfill",
            {"seeds": seeds, "n": 2000, "arms": ["h02", "h03"],
             "paired_against": "w4_qladder_2026-08-22T211310+0000_90347fa1"},
            "ok",
            metrics={"h02_mean": float(h02.mean()), "h03_mean": float(h03.mean()),
                     "inc23": float(m23), "inc23_se": float(se23),
                     "inc34": float(m34), "inc34_se": float(se34),
                     "h02_minus_k0": float(m2), "h02_minus_k0_se": float(se2)},
            artifacts=kf_files,
            notes=f"AMENDMENT 25 ladder rungs 2 and 3, compounding depth"
                  f" read. {verdict}. No headline, registered in advance.",
            tier=1)
        ledger.regenerate_leaderboard()
        print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
