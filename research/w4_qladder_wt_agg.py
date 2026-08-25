"""w4_warmtemp aggregator. Reads registered in AMENDMENT 24,
step0_prereg.md.

Cross run pairing against the AMENDMENT 22 per seed table, valid only if
the seed 40 k0 rerun on the MODIFIED sampler matches 0.5827 to the
fourth decimal (that gate also proves the scalar temperature path is
untouched on the real seam). READ 1 (PRIMARY): warm4 minus k0;
<= -0.010 at 2 se SHARPENING COSTS THE OPENING; >= +0.010 at 2 se
SHARPENING HELPS THE OPENING; else NULL RANGE. READ 2 (informational):
warm4 minus q0.
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

    wt_files = sorted(f for f in glob.glob(os.path.join(a.dir, "w4_qladder_wt_s*.json"))
                      if re.search(r"_s\d+\.json$", f))
    wt_runs = sorted((json.load(open(f)) for f in wt_files), key=lambda r: r["seed"])
    if not wt_runs:
        print("no wt runs found"); sys.exit(1)
    seeds = [r["seed"] for r in wt_runs]
    a22 = {s: json.load(open(os.path.join(a.dir, f"w4_qladder_s{s}.json")))
           for s in seeds}

    k0_rerun = wt_runs[0]["arms"].get("k0")
    k0_orig = a22[seeds[0]]["arms"]["k0"]["contract"]
    if k0_rerun is None:
        print("!! seed 40 k0 rerun missing, gate unverified"); sys.exit(1)
    ok = abs(k0_rerun["contract"] - k0_orig) < 5e-5
    print(f"  gate seed {seeds[0]} k0 on modified sampler: rerun "
          f"{k0_rerun['contract']:.4f} vs A22 {k0_orig:.4f} -> "
          f"{'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        print("!! scalar path or pairing broken, do not read further")
        sys.exit(1)

    warm = np.array([r["arms"]["warm4"]["contract"] for r in wt_runs])
    k0 = np.array([a22[s]["arms"]["k0"]["contract"] for s in seeds])
    q0 = np.array([a22[s]["arms"]["q0"]["contract"] for s in seeds])

    print(f"  seeds {seeds}")
    print("    seed    warm4       k0       q0")
    for i, s in enumerate(seeds):
        print(f"  {s:>6}{warm[i]:>9.4f}{k0[i]:>9.4f}{q0[i]:>9.4f}")

    def stats(d, name):
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        t = m / se if se > 0 else float("inf")
        print(f"  {name}: mean {m:+.4f}  se {se:.4f}  t {t:+.2f}  per seed "
              + " ".join(f"{v:+.4f}" for v in d))
        return m, se, t

    print("\n  READ 1 (PRIMARY), warm4 minus k0, temperature 1 opening on"
          " the AR's own draws:")
    m1, se1, _ = stats(warm - k0, "READ 1")
    if m1 <= -0.010 and abs(m1) >= 2 * se1:
        verdict = "SHARPENING COSTS THE OPENING"
    elif m1 >= 0.010 and abs(m1) >= 2 * se1:
        verdict = "SHARPENING HELPS THE OPENING"
    else:
        verdict = "NULL RANGE"
    print(f"  VERDICT: {verdict}")

    print("\n  READ 2 (informational), warm4 minus q0:")
    m2, se2, _ = stats(warm - q0, "READ 2")

    print("\n  no headline, no serve decision from this arm (registered)")

    if not a.no_ledger:
        rid = ledger.append_row(
            "w4_warmtemp",
            {"seeds": seeds, "n": 2000, "arms": ["warm4"],
             "schedule": "temps 1,1,1 at positions 0 to 3, served"
                         " s 0.95 th 0.90 dt 1.00 from position 4",
             "paired_against": "w4_qladder_2026-08-22T211310+0000_90347fa1"},
            "ok",
            metrics={"warm4_mean": float(warm.mean()),
                     "read1_warm4_minus_k0": float(m1), "read1_se": float(se1),
                     "read2_warm4_minus_q0": float(m2), "read2_se": float(se2),
                     "k0_gate_delta": float(k0_rerun["contract"] - k0_orig)},
            artifacts=wt_files,
            notes=f"AMENDMENT 24 warm opening, does serve sharpening at"
                  f" positions 0 to 3 explain the ladder increment."
                  f" {verdict}. Gate passed on the modified sampler."
                  f" No headline, registered in advance.",
            tier=1)
        ledger.regenerate_leaderboard()
        print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
