"""w4_durmech aggregator. Reads registered in AMENDMENT 20, step0_prereg.md.

E(src) = src-q1 minus src-k0 per seed, paired. READ 1 (PRIMARY): J =
E(MJ) minus E(M); J >= +0.010 at 2 se JITTER IS THE MECHANISM; |J| <=
0.005 JITTER INNOCENT; else BETWEEN. READ 2: E(M), E(P) vs AMENDMENT 18
(-0.019, 0.000), descriptive. READ 3: MJ-k0 minus M-k0.
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

LABELS = ("M-k0", "M-q1", "MJ-k0", "MJ-q1", "P-k0", "P-q1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="research")
    ap.add_argument("--no-ledger", action="store_true")
    a = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(a.dir, "w4_durmech_s*.json"))
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

    E = {s: arm[f"{s}-q1"] - arm[f"{s}-k0"] for s in ("M", "MJ", "P")}
    J = E["MJ"] - E["M"]

    print("\n  READ 1 (PRIMARY), J = E(MJ) minus E(M), jitter cost of the q effect:")
    mJ, seJ, tJ = stats(J, "J")
    if mJ >= 0.010 and abs(mJ) >= 2 * seJ:
        verdict = "JITTER IS THE MECHANISM"
    elif abs(mJ) <= 0.005:
        verdict = "JITTER INNOCENT"
    else:
        verdict = "BETWEEN"
    print(f"  VERDICT: {verdict}")

    print("\n  READ 2 (descriptive), do E(M), E(P) reproduce AMENDMENT 18:")
    mM, seM, _ = stats(E["M"], "E(M) (A18 read -0.0192)")
    mP, seP, _ = stats(E["P"], "E(P) (A18 read -0.0006)")
    mMJ, seMJ, _ = stats(E["MJ"], "E(MJ) (new)")

    print("\n  READ 3, MJ-k0 minus M-k0, jitter alone on the AR baseline:")
    m3, se3, _ = stats(arm["MJ-k0"] - arm["M-k0"], "MJ-k0 - M-k0")

    print("\n  no headline, no serve decision from this arm (registered)")

    if not a.no_ledger:
        rid = ledger.append_row(
            "w4_durmech",
            {"seeds": seeds, "n": 2000, "arms": list(LABELS),
             "MJ": "M plus normal(0, 0.02) log jitter rng 17000+seed, clipped"},
            "ok",
            metrics={lab.replace("-", "_") + "_mean": float(arm[lab].mean())
                     for lab in LABELS} |
                    {"J": float(mJ), "J_se": float(seJ),
                     "E_M": float(mM), "E_M_se": float(seM),
                     "E_MJ": float(mMJ), "E_MJ_se": float(seMJ),
                     "E_P": float(mP), "E_P_se": float(seP),
                     "MJk0_minus_Mk0": float(m3), "MJk0_minus_Mk0_se": float(se3)},
            artifacts=files,
            notes=f"AMENDMENT 20 mechanism split, jitter factor isolated. "
                  f"{verdict}. No headline, registered in advance.",
            tier=1)
        ledger.regenerate_leaderboard()
        print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
