"""w4_dursrc aggregator. Reads registered in AMENDMENT 18, step0_prereg.md.

READ 1 (PRIMARY, the interaction): D = (M-q1 minus M-k0) minus (P-q1 minus
P-k0) per seed, paired; five seed mean and se. D <= -0.010 at 2 se ->
MODULATION REAL. |D| <= 0.005 -> A DRAW (durmatch's H_dur verdict retracted
in substance; population closes at neutral, bounded by AMENDMENT 15).
Else BETWEEN.
READ 2 (descriptive): does M-q1 minus M-k0 reproduce durmatch's -0.0215,
and P-q1 minus P-k0 the confirm's +0.0012?
READ 3: P-k0 minus M-k0, the duration source's own effect on the AR
baseline, paired on identical specs.
No headline, no serve decision from this arm.
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

    files = sorted(f for f in glob.glob(os.path.join(a.dir, "w4_dursrc_s*.json"))
                   if re.search(r"_s\d+\.json$", f))
    runs = [json.load(open(f)) for f in files]
    if not runs:
        print("no runs found"); sys.exit(1)
    runs.sort(key=lambda r: r["seed"])
    seeds = [r["seed"] for r in runs]

    arm = {lab: np.array([r["arms"][lab]["contract"] for r in runs])
           for lab in ("P-k0", "P-q1", "M-k0", "M-q1")}

    print(f"  seeds {seeds}")
    print(f"  {'seed':>6}{'P-k0':>9}{'P-q1':>9}{'M-k0':>9}{'M-q1':>9}")
    for i, r in enumerate(runs):
        print(f"  {r['seed']:>6}" + "".join(f"{arm[lab][i]:>9.4f}"
              for lab in ("P-k0", "P-q1", "M-k0", "M-q1")))

    def stats(d, name):
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        t = m / se if se > 0 else float("inf")
        print(f"  {name}: mean {m:+.4f}  se {se:.4f}  t {t:+.2f}  per seed "
              + " ".join(f"{v:+.4f}" for v in d))
        return m, se, t

    qP = arm["P-q1"] - arm["P-k0"]
    qM = arm["M-q1"] - arm["M-k0"]
    D = qM - qP

    print("\n  READ 1 (PRIMARY), interaction D = (M-q1 - M-k0) - (P-q1 - P-k0):")
    mD, seD, tD = stats(D, "D")
    if mD <= -0.010 and abs(mD) >= 2 * seD:
        verdict = "MODULATION REAL"
    elif abs(mD) <= 0.005:
        verdict = "A DRAW"
    else:
        verdict = "BETWEEN"
    print(f"  VERDICT: {verdict}")

    print("\n  READ 2 (descriptive), per source q effects vs prior runs:")
    mM, seM, tM = stats(qM, "M q effect (durmatch read -0.0215)")
    mP, seP, tP = stats(qP, "P q effect (confirm read +0.0012)")

    print("\n  READ 3, P-k0 minus M-k0, duration source effect on the AR baseline:")
    m3, se3, t3 = stats(arm["P-k0"] - arm["M-k0"], "P-k0 - M-k0")

    print("\n  no headline, no serve decision from this arm (registered)")

    if not a.no_ledger:
        rid = ledger.append_row(
            "w4_dursrc",
            {"seeds": seeds, "n": 2000,
             "arms": ["P-k0", "P-q1", "M-k0", "M-q1"],
             "P": "protocol esp._duration.sample",
             "M": "matched k=64 heldout, rng 9000+seed"},
            "ok",
            metrics={f"{lab}_mean".replace("-", "_"): float(arm[lab].mean())
                     for lab in arm} |
                    {"D_interaction": float(mD), "D_se": float(seD),
                     "qM": float(mM), "qM_se": float(seM),
                     "qP": float(mP), "qP_se": float(seP),
                     "Pk0_minus_Mk0": float(m3), "Pk0_minus_Mk0_se": float(se3)},
            artifacts=files,
            notes=f"AMENDMENT 18 interaction test, both duration sources inside "
                  f"one run on identical specs. {verdict}. No headline, no "
                  f"serve decision, registered in advance.",
            tier=1)
        ledger.regenerate_leaderboard()
        print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
