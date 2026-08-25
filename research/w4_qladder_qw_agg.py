"""w4_qwarm aggregator. Reads registered in AMENDMENT 26,
step0_prereg.md.

Cross run pairing against the AMENDMENT 22 per seed table (q0) and the
AMENDMENT 25 table (h02). No gate rerun: pairing was validated at delta
0.0 in AMENDMENTS 23 and 24 and no sampler code changed here. READ 1
(PRIMARY): q0w1 minus q0; <= -0.010 at 2 se SHARPENING HURTS THE GOOD
PREFIX; |mean| <= 0.004 TEMPERATURE FULLY CLEARED; else BETWEEN.
READ 2 (informational): q0w1 minus h02.
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

    qw_files = sorted(f for f in glob.glob(os.path.join(a.dir, "w4_qladder_qw_s*.json"))
                      if re.search(r"_s\d+\.json$", f))
    qw_runs = sorted((json.load(open(f)) for f in qw_files), key=lambda r: r["seed"])
    if not qw_runs:
        print("no qw runs found"); sys.exit(1)
    seeds = [r["seed"] for r in qw_runs]
    a22 = {s: json.load(open(os.path.join(a.dir, f"w4_qladder_s{s}.json")))
           for s in seeds}
    kf = {s: json.load(open(os.path.join(a.dir, f"w4_qladder_kf_s{s}.json")))
          for s in seeds}

    qw = np.array([r["arms"]["q0w1"]["contract"] for r in qw_runs])
    q0 = np.array([a22[s]["arms"]["q0"]["contract"] for s in seeds])
    h02 = np.array([kf[s]["arms"]["h02"]["contract"] for s in seeds])

    print(f"  seeds {seeds}")
    print("    seed     q0w1       q0      h02")
    for i, s in enumerate(seeds):
        print(f"  {s:>6}{qw[i]:>9.4f}{q0[i]:>9.4f}{h02[i]:>9.4f}")

    def stats(d, name):
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        t = m / se if se > 0 else float("inf")
        print(f"  {name}: mean {m:+.4f}  se {se:.4f}  t {t:+.2f}  per seed "
              + " ".join(f"{v:+.4f}" for v in d))
        return m, se, t

    print("\n  READ 1 (PRIMARY), q0w1 minus q0, temperature 1 at position 1"
          " behind the served q event 0:")
    m1, se1, _ = stats(qw - q0, "READ 1")
    if m1 <= -0.010 and abs(m1) >= 2 * se1:
        verdict = "SHARPENING HURTS THE GOOD PREFIX"
    elif abs(m1) <= 0.004:
        verdict = "TEMPERATURE FULLY CLEARED"
    else:
        verdict = "BETWEEN"
    print(f"  VERDICT: {verdict}")

    print("\n  READ 2 (informational), q0w1 minus h02:")
    m2, se2, _ = stats(qw - h02, "READ 2")

    print("\n  no headline, no serve decision from this arm (registered)")

    if not a.no_ledger:
        rid = ledger.append_row(
            "w4_qwarm",
            {"seeds": seeds, "n": 2000, "arms": ["q0w1"],
             "schedule": "q event 0 forced (draw +7), temps 1,1,1 at"
                         " position 1, served s 0.95 th 0.90 dt 1.00"
                         " from position 2",
             "paired_against": ["w4_qladder_2026-08-22T211310+0000_90347fa1",
                                "w4_kfill_2026-08-23T013629+0000_dc718032"]},
            "ok",
            metrics={"q0w1_mean": float(qw.mean()),
                     "read1_q0w1_minus_q0": float(m1), "read1_se": float(se1),
                     "read2_q0w1_minus_h02": float(m2), "read2_se": float(se2)},
            artifacts=qw_files,
            notes=f"AMENDMENT 26 qwarm, does serve sharpening at position 1"
                  f" explain the gap between the q serve and the human pair."
                  f" {verdict}. No gate rerun, pairing validated twice."
                  f" No headline, registered in advance.",
            tier=1)
        ledger.regenerate_leaderboard()
        print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
