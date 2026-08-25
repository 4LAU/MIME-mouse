"""w4_qladder aggregator. Reads registered in AMENDMENT 22, step0_prereg.md.

READ 1 (PRIMARY): paired q0h13 minus q0. Mean <= -0.010 at 2 se HUMAN
OPENING STILL CARRIES; |mean| <= 0.004 LADDER CASHED; else BETWEEN.
READ 2: h04 minus q0h13, does replacing q event 0 with the human event 0
matter once human events 1 to 3 are present; expect about 0.
READ 3: q0 minus k0, the served q effect on held out rows; expect -0.015
to -0.025.
READ 4 (informational): h04 minus k0 vs OUTCOME 11's k=4 read -0.0366.
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

LABELS = ("k0", "q0", "q0h13", "h04")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="research")
    ap.add_argument("--no-ledger", action="store_true")
    a = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(a.dir, "w4_qladder_s*.json"))
                   if re.search(r"_s\d+\.json$", f))
    runs = [json.load(open(f)) for f in files]
    if not runs:
        print("no runs found"); sys.exit(1)
    runs.sort(key=lambda r: r["seed"])
    seeds = [r["seed"] for r in runs]
    arm = {lab: np.array([r["arms"][lab]["contract"] for r in runs])
           for lab in LABELS}

    print(f"  seeds {seeds}")
    print("  " + "seed".rjust(6) + "".join(l.rjust(9) for l in LABELS))
    for i, r in enumerate(runs):
        print(f"  {r['seed']:>6}" + "".join(f"{arm[l][i]:>9.4f}" for l in LABELS))

    def stats(d, name):
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        t = m / se if se > 0 else float("inf")
        print(f"  {name}: mean {m:+.4f}  se {se:.4f}  t {t:+.2f}  per seed "
              + " ".join(f"{v:+.4f}" for v in d))
        return m, se, t

    print("\n  READ 1 (PRIMARY), q0h13 minus q0, do human events 1 to 3 still"
          " carry given the served q event 0:")
    m1, se1, t1 = stats(arm["q0h13"] - arm["q0"], "READ 1")
    if m1 <= -0.010 and abs(m1) >= 2 * se1:
        verdict = "HUMAN OPENING STILL CARRIES"
    elif abs(m1) <= 0.004:
        verdict = "LADDER CASHED"
    else:
        verdict = "BETWEEN"
    print(f"  VERDICT: {verdict}")

    print("\n  READ 2, h04 minus q0h13, human vs q event 0 with human 1 to 3"
          " present (expect about 0):")
    m2, se2, _ = stats(arm["h04"] - arm["q0h13"], "READ 2")

    print("\n  READ 3, q0 minus k0, served q effect on held out rows"
          " (expect -0.015 to -0.025):")
    m3, se3, _ = stats(arm["q0"] - arm["k0"], "READ 3")

    print("\n  READ 4 (informational), h04 minus k0 vs OUTCOME 11 k=4 -0.0366:")
    m4, se4, _ = stats(arm["h04"] - arm["k0"], "READ 4")

    print("\n  no headline, no serve decision from this arm (registered)")

    if not a.no_ledger:
        rid = ledger.append_row(
            "w4_qladder",
            {"seeds": seeds, "n": 2000, "arms": list(LABELS),
             "rows": "held out, rng 1000+seed, length > 4",
             "q": "w4_firsthead_q.pt temps 1,1,1 torch offset +7, shared"
                  " draw across q0 and q0h13"},
            "ok",
            metrics={lab + "_mean": float(arm[lab].mean()) for lab in LABELS} |
                    {"read1_q0h13_minus_q0": float(m1), "read1_se": float(se1),
                     "read2_h04_minus_q0h13": float(m2), "read2_se": float(se2),
                     "read3_q0_minus_k0": float(m3), "read3_se": float(se3),
                     "read4_h04_minus_k0": float(m4), "read4_se": float(se4)},
            artifacts=files,
            notes=f"AMENDMENT 22 q ladder, is the k=4 increment already cashed"
                  f" by the served q event 0. {verdict}. No headline,"
                  f" registered in advance.",
            tier=1)
        ledger.regenerate_leaderboard()
        print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
