"""w4_qladder wr0h13 aggregator. Reads registered in AMENDMENT 23,
step0_prereg.md.

Cross run pairing against the AMENDMENT 22 per seed table, valid only if
the seed 40 k0 rerun matches the AMENDMENT 22 value to the fourth
decimal. READ 1 (PRIMARY): wr0h13 minus q0h13; |mean| <= 0.015
INCOHERENCE PER SE; <= -0.10 Q SPECIFIC; else PARTIAL. READ 2
(informational): wr0h13 minus h04.
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

    wr_files = sorted(f for f in glob.glob(os.path.join(a.dir, "w4_qladder_wr_s*.json"))
                      if re.search(r"_s\d+\.json$", f))
    wr_runs = sorted((json.load(open(f)) for f in wr_files), key=lambda r: r["seed"])
    if not wr_runs:
        print("no wr runs found"); sys.exit(1)
    seeds = [r["seed"] for r in wr_runs]
    a22 = {s: json.load(open(os.path.join(a.dir, f"w4_qladder_s{s}.json")))
           for s in seeds}

    k0_rerun = wr_runs[0]["arms"].get("k0")
    k0_orig = a22[seeds[0]]["arms"]["k0"]["contract"]
    if k0_rerun is None:
        print("!! seed 40 k0 rerun missing, pairing unverified"); sys.exit(1)
    ok = abs(k0_rerun["contract"] - k0_orig) < 5e-5
    print(f"  reproduction check seed {seeds[0]} k0: rerun "
          f"{k0_rerun['contract']:.4f} vs A22 {k0_orig:.4f} -> "
          f"{'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        print("!! pairing invalid, rerun all arms in run per registration")
        sys.exit(1)

    wr = np.array([r["arms"]["wr0h13"]["contract"] for r in wr_runs])
    q0h13 = np.array([a22[s]["arms"]["q0h13"]["contract"] for s in seeds])
    h04 = np.array([a22[s]["arms"]["h04"]["contract"] for s in seeds])

    print(f"  seeds {seeds}")
    print("    seed   wr0h13    q0h13      h04")
    for i, s in enumerate(seeds):
        print(f"  {s:>6}{wr[i]:>9.4f}{q0h13[i]:>9.4f}{h04[i]:>9.4f}")

    def stats(d, name):
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        t = m / se if se > 0 else float("inf")
        print(f"  {name}: mean {m:+.4f}  se {se:.4f}  t {t:+.2f}  per seed "
              + " ".join(f"{v:+.4f}" for v in d))
        return m, se, t

    print("\n  READ 1 (PRIMARY), wr0h13 minus q0h13, is the splice artifact"
          " q specific:")
    m1, se1, _ = stats(wr - q0h13, "READ 1")
    if abs(m1) <= 0.015:
        verdict = "INCOHERENCE PER SE"
    elif m1 <= -0.10:
        verdict = "Q SPECIFIC"
    else:
        verdict = "PARTIAL"
    print(f"  VERDICT: {verdict}")

    print("\n  READ 2 (informational), wr0h13 minus h04:")
    m2, se2, _ = stats(wr - h04, "READ 2")

    print("\n  no headline, no serve decision from this arm (registered)")

    if not a.no_ledger:
        rid = ledger.append_row(
            "w4_qladder",
            {"seeds": seeds, "n": 2000, "arms": ["wr0h13"],
             "donor": "pick index (i+1) mod n human event 0",
             "paired_against": "w4_qladder_2026-08-22T211310+0000_90347fa1"},
            "ok",
            metrics={"wr0h13_mean": float(wr.mean()),
                     "read1_wr_minus_q0h13": float(m1), "read1_se": float(se1),
                     "read2_wr_minus_h04": float(m2), "read2_se": float(se2),
                     "k0_reproduction_delta": float(k0_rerun["contract"] - k0_orig)},
            artifacts=wr_files,
            notes=f"AMENDMENT 23 splice control, wrong row human event 0 plus"
                  f" own events 1 to 3. {verdict}. Cross run pairing validated"
                  f" by seed {seeds[0]} k0 reproduction. No headline,"
                  f" registered in advance.",
            tier=1)
        ledger.regenerate_leaderboard()
        print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
