"""w4_pairsplit aggregator. Reads registered in AMENDMENT 28,
step0_prereg.md.

Cross run pairing against the AMENDMENT 22 table (q0, k0) and the
AMENDMENT 25 table (h02). READ 1 (PRIMARY): h0p1 minus h02;
|mean| <= 0.005 CONDITIONAL ADEQUATE GIVEN A REAL e0; >= +0.010 at
2 se SAMPLED e1 IS THE DEFECT; else BETWEEN. READ 2: h01 minus q0.
READ 3 (informational): h01 minus h02, h0p1 minus q0. Consistency
check per the registration.
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

    ps_files = sorted(f for f in glob.glob(os.path.join(a.dir, "w4_qladder_ps_s*.json"))
                      if re.search(r"_s\d+\.json$", f))
    ps_runs = sorted((json.load(open(f)) for f in ps_files), key=lambda r: r["seed"])
    if not ps_runs:
        print("no ps runs found"); sys.exit(1)
    seeds = [r["seed"] for r in ps_runs]
    a22 = {s: json.load(open(os.path.join(a.dir, f"w4_qladder_s{s}.json")))
           for s in seeds}
    kf = {s: json.load(open(os.path.join(a.dir, f"w4_qladder_kf_s{s}.json")))
          for s in seeds}

    h01 = np.array([r["arms"]["h01"]["contract"] for r in ps_runs])
    h0p1 = np.array([r["arms"]["h0p1"]["contract"] for r in ps_runs])
    q0 = np.array([a22[s]["arms"]["q0"]["contract"] for s in seeds])
    k0 = np.array([a22[s]["arms"]["k0"]["contract"] for s in seeds])
    h02 = np.array([kf[s]["arms"]["h02"]["contract"] for s in seeds])

    print(f"  seeds {seeds}")
    print("    seed      h01     h0p1       q0       k0      h02")
    for i, s in enumerate(seeds):
        print(f"  {s:>6}{h01[i]:>9.4f}{h0p1[i]:>9.4f}{q0[i]:>9.4f}"
              f"{k0[i]:>9.4f}{h02[i]:>9.4f}")

    def stats(d, name):
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        t = m / se if se > 0 else float("inf")
        print(f"  {name}: mean {m:+.4f}  se {se:.4f}  t {t:+.2f}  per seed "
              + " ".join(f"{v:+.4f}" for v in d))
        return m, se, t

    print("\n  READ 1 (PRIMARY), h0p1 minus h02, the conditional's e1 vs the"
          " human e1 behind the same human e0:")
    m1, se1, _ = stats(h0p1 - h02, "READ 1")
    if abs(m1) <= 0.005:
        verdict = "CONDITIONAL ADEQUATE GIVEN A REAL e0"
    elif m1 >= 0.010 and abs(m1) >= 2 * se1:
        verdict = "SAMPLED e1 IS THE DEFECT"
    else:
        verdict = "BETWEEN"
    print(f"  VERDICT: {verdict}")

    print("\n  READ 2, h01 minus q0, the human e0 realization vs q's draw:")
    m2, se2, _ = stats(h01 - q0, "READ 2")

    print("\n  READ 3 (informational):")
    m3a, _, _ = stats(h01 - h02, "h01 minus h02")
    m3b, _, _ = stats(h0p1 - q0, "h0p1 minus q0")

    if abs(m1) <= 0.005 and m2 >= -0.005:
        print("\n  CONSISTENCY CHECK TRIGGERED: READ 1 adequate and READ 2"
              " shows no human e0 value, which contradicts AMENDMENT 27."
              " Report as interaction, do not force a verdict.")

    print("\n  no headline, no serve decision from this arm (registered)")

    if not a.no_ledger:
        rid = ledger.append_row(
            "w4_pairsplit",
            {"seeds": seeds, "n": 2000, "arms": ["h01", "h0p1"],
             "serve": "h01 human e0 forced alone; h0p1 human e0 forced plus"
                      " e1 from q1g0 given that e0, draw +13, served temps"
                      " from position 2",
             "paired_against": ["w4_qladder_2026-08-22T211310+0000_90347fa1",
                                "w4_kfill_2026-08-23T013629+0000_dc718032"]},
            "ok",
            metrics={"h01_mean": float(h01.mean()),
                     "h0p1_mean": float(h0p1.mean()),
                     "read1_h0p1_minus_h02": float(m1), "read1_se": float(se1),
                     "read2_h01_minus_q0": float(m2), "read2_se": float(se2),
                     "read3_h01_minus_h02": float(m3a),
                     "read3_h0p1_minus_q0": float(m3b)},
            artifacts=ps_files,
            notes=f"AMENDMENT 28 pair split, is the pair defect in the drawn"
                  f" e0 or the sampled e1. {verdict}. No headline, registered"
                  f" in advance.",
            tier=1)
        ledger.regenerate_leaderboard()
        print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
