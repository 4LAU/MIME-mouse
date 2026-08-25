"""w4_ceplus contract aggregator. Reads registered in AMENDMENT 33,
step0_prereg.md.

Cross run pairing against the AMENDMENT 27 p01 files (PRIMARY), the
AMENDMENT 22 q0, and the AMENDMENT 25 h02. READ 1: r01 minus p01;
IMPROVES <= -0.008 at 2 se; NO GAIN |mean| <= 0.004; HURTS >= +0.008
at 2 se; else BETWEEN. READ 2: r01 minus h02. READ 3: r01 minus q0.
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

    r_files = sorted(f for f in glob.glob(os.path.join(a.dir, "w4_qladder_r_s*.json"))
                     if re.search(r"_s\d+\.json$", f))
    r_runs = sorted((json.load(open(f)) for f in r_files), key=lambda r: r["seed"])
    if not r_runs:
        print("no r runs found"); sys.exit(1)
    seeds = [r["seed"] for r in r_runs]
    pf = {s: json.load(open(os.path.join(a.dir, f"w4_qladder_p_s{s}.json")))
          for s in seeds}
    a22 = {s: json.load(open(os.path.join(a.dir, f"w4_qladder_s{s}.json")))
           for s in seeds}
    kf = {s: json.load(open(os.path.join(a.dir, f"w4_qladder_kf_s{s}.json")))
          for s in seeds}

    r01 = np.array([r["arms"]["r01"]["contract"] for r in r_runs])
    p01 = np.array([pf[s]["arms"]["p01"]["contract"] for s in seeds])
    q0 = np.array([a22[s]["arms"]["q0"]["contract"] for s in seeds])
    h02 = np.array([kf[s]["arms"]["h02"]["contract"] for s in seeds])

    print(f"  seeds {seeds}")
    print("    seed      r01      p01       q0      h02")
    for i, s in enumerate(seeds):
        print(f"  {s:>6}{r01[i]:>9.4f}{p01[i]:>9.4f}{q0[i]:>9.4f}{h02[i]:>9.4f}")

    def stats(d, name):
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        t = m / se if se > 0 else float("inf")
        print(f"  {name}: mean {m:+.4f}  se {se:.4f}  t {t:+.2f}  per seed "
              + " ".join(f"{v:+.4f}" for v in d))
        return m, se, t

    print("\n  READ 1 (PRIMARY), r01 minus p01, does the CE continuation's"
          " pair level gain reach the contract:")
    m1, se1, _ = stats(r01 - p01, "READ 1")
    if m1 <= -0.008 and abs(m1) >= 2 * se1:
        verdict = "IMPROVES"
    elif abs(m1) <= 0.004:
        verdict = "NO GAIN"
    elif m1 >= 0.008 and abs(m1) >= 2 * se1:
        verdict = "HURTS"
    else:
        verdict = "BETWEEN"
    print(f"  VERDICT: {verdict}")

    print("\n  READ 2, r01 minus h02, distance to the human pair:")
    m2, se2, _ = stats(r01 - h02, "READ 2")

    print("\n  READ 3, r01 minus q0:")
    m3, se3, _ = stats(r01 - q0, "READ 3")

    print("\n  no headline, no serve decision from this arm (registered)")

    if not a.no_ledger:
        rid = ledger.append_row(
            "w4_ceplus",
            {"phase": "contract", "seeds": seeds, "n": 2000, "arms": ["r01"],
             "serve": "e0 from q seed +7 (identical to p01), e1 from the"
                      " pure CE continuation checkpoint (w4_pairadv.pt lam 0"
                      " best epoch 1), draw +13, served temps from position 2",
             "paired_against": ["w4_pairq_2026-08-24T233541+0000_59f551ca",
                                "w4_qladder_2026-08-22T211310+0000_90347fa1",
                                "w4_kfill_2026-08-23T013629+0000_dc718032"]},
            "ok",
            metrics={"r01_mean": float(r01.mean()),
                     "read1_r01_minus_p01": float(m1), "read1_se": float(se1),
                     "read2_r01_minus_h02": float(m2), "read2_se": float(se2),
                     "read3_r01_minus_q0": float(m3), "read3_se": float(se3)},
            artifacts=r_files,
            notes=f"AMENDMENT 33 calibration contract run, does the pair"
                  f" instrument's 40 percent visibility reduction translate."
                  f" {verdict}. No headline, registered in advance.",
            tier=1)
        ledger.regenerate_leaderboard()
        print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
