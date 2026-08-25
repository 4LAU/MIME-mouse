"""w4_pairq contract aggregator. Reads registered in AMENDMENT 27,
step0_prereg.md.

Cross run pairing against the AMENDMENT 22 table (q0, k0) and the
AMENDMENT 25 table (h02). READ 1 (PRIMARY): p01 minus q0;
<= -0.015 at 2 se PAIR MODEL CLOSES THE GAP; <= -0.008 at 2 se PARTIAL
CLOSE; |mean| <= 0.005 NO GAIN; >= +0.010 at 2 se PAIR MODEL HURTS;
else BETWEEN. READ 2: p01 minus h02. READ 3 (informational): p01
minus k0.
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

    p_files = sorted(f for f in glob.glob(os.path.join(a.dir, "w4_qladder_p_s*.json"))
                     if re.search(r"_s\d+\.json$", f))
    p_runs = sorted((json.load(open(f)) for f in p_files), key=lambda r: r["seed"])
    if not p_runs:
        print("no p runs found"); sys.exit(1)
    seeds = [r["seed"] for r in p_runs]
    a22 = {s: json.load(open(os.path.join(a.dir, f"w4_qladder_s{s}.json")))
           for s in seeds}
    kf = {s: json.load(open(os.path.join(a.dir, f"w4_qladder_kf_s{s}.json")))
          for s in seeds}

    p01 = np.array([r["arms"]["p01"]["contract"] for r in p_runs])
    q0 = np.array([a22[s]["arms"]["q0"]["contract"] for s in seeds])
    k0 = np.array([a22[s]["arms"]["k0"]["contract"] for s in seeds])
    h02 = np.array([kf[s]["arms"]["h02"]["contract"] for s in seeds])

    print(f"  seeds {seeds}")
    print("    seed      p01       q0       k0      h02")
    for i, s in enumerate(seeds):
        print(f"  {s:>6}{p01[i]:>9.4f}{q0[i]:>9.4f}{k0[i]:>9.4f}{h02[i]:>9.4f}")

    def stats(d, name):
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        t = m / se if se > 0 else float("inf")
        print(f"  {name}: mean {m:+.4f}  se {se:.4f}  t {t:+.2f}  per seed "
              + " ".join(f"{v:+.4f}" for v in d))
        return m, se, t

    print("\n  READ 1 (PRIMARY), p01 minus q0, the pair model against the"
          " current serve:")
    m1, se1, _ = stats(p01 - q0, "READ 1")
    if m1 <= -0.015 and abs(m1) >= 2 * se1:
        verdict = "PAIR MODEL CLOSES THE GAP"
    elif m1 <= -0.008 and abs(m1) >= 2 * se1:
        verdict = "PARTIAL CLOSE"
    elif abs(m1) <= 0.005:
        verdict = "NO GAIN"
    elif m1 >= 0.010 and abs(m1) >= 2 * se1:
        verdict = "PAIR MODEL HURTS"
    else:
        verdict = "BETWEEN"
    print(f"  VERDICT: {verdict}")

    print("\n  READ 2, p01 minus h02, remaining distance to the human pair:")
    m2, se2, _ = stats(p01 - h02, "READ 2")

    print("\n  READ 3 (informational), p01 minus k0:")
    m3, se3, _ = stats(p01 - k0, "READ 3")

    print("\n  no headline, no serve decision from this arm (registered)")

    if not a.no_ledger:
        rid = ledger.append_row(
            "w4_pairq",
            {"seeds": seeds, "n": 2000, "arms": ["p01"],
             "serve": "e0 and e1 forced from PairHead.sample temps 1,1,1"
                      " draw +13, served s 0.95 th 0.90 dt 1.00 from"
                      " position 2",
             "paired_against": ["w4_qladder_2026-08-22T211310+0000_90347fa1",
                                "w4_kfill_2026-08-23T013629+0000_dc718032"]},
            "ok",
            metrics={"p01_mean": float(p01.mean()),
                     "read1_p01_minus_q0": float(m1), "read1_se": float(se1),
                     "read2_p01_minus_h02": float(m2), "read2_se": float(se2),
                     "read3_p01_minus_k0": float(m3), "read3_se": float(se3)},
            artifacts=p_files,
            notes=f"AMENDMENT 27 pair model contract run, does q2(e0, e1 |"
                  f" cond) close the gap between the q serve and the human"
                  f" pair. {verdict}. No headline, registered in advance.",
            tier=1)
        ledger.regenerate_leaderboard()
        print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
