"""w4_nodur stage B aggregator. Reads registered in AMENDMENT 17.

CROSS-RUN NOTE: k0 is NOT bit-reproducible against the confirm's k0;
DurationModel._rng is unseeded, so each process draws different spec
durations (AMENDMENT 17 DESIGN NOTE). The comparison is informational.
Within-run pairing (durations drawn once, shared across arms) is intact.
PRIMARY: paired n1 minus k0; <= -0.008 at 2 paired se -> TRANSFERS (no
headline claim, fresh-seed confirm to be registered); |mean| <= 0.004 ->
DOES NOT TRANSFER; else BETWEEN. READ 2 informational: nT minus n1, and
n1 effect vs durmatch's q1 effect.
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

    files = sorted(f for f in glob.glob(os.path.join(a.dir, "w4_nodur_s*.json"))
                   if re.search(r"_s\d+\.json$", f))
    runs = [json.load(open(f)) for f in files]
    if not runs:
        print("no runs found"); sys.exit(1)
    runs.sort(key=lambda r: r["seed"])
    seeds = [r["seed"] for r in runs]

    k0 = np.array([r["arms"]["k0"]["contract"] for r in runs])
    n1 = np.array([r["arms"]["n1"]["contract"] for r in runs])
    nT = np.array([r["arms"]["nT"]["contract"] for r in runs])

    print(f"  seeds {seeds}")
    print(f"  {'seed':>6}{'k0':>9}{'n1':>9}{'nT':>9}")
    for r in runs:
        print(f"  {r['seed']:>6}{r['arms']['k0']['contract']:>9.4f}"
              f"{r['arms']['n1']['contract']:>9.4f}{r['arms']['nT']['contract']:>9.4f}")

    print("\n  CROSS-RUN (informational), k0 vs the confirm's k0. Durations are")
    print("  process-random (unseeded DurationModel rng): bit-identity NOT expected.")
    for s, v in zip(seeds, k0):
        f = os.path.join(a.dir, f"w4_fhconfirm_s{s}.json")
        if os.path.exists(f):
            c = json.load(open(f))["arms"]["k0"]["contract"]
            print(f"    seed {s}: {v:.4f} vs confirm {c:.4f}  diff {v - c:+.4f}")

    def paired(x, y, name):
        d = x - y
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        t = m / se if se > 0 else float("inf")
        print(f"  {name}: mean {m:+.4f}  se {se:.4f}  t {t:+.2f}  per seed "
              + " ".join(f"{v:+.4f}" for v in d))
        return m, se, t

    print("\n  PRIMARY, n1 minus k0 paired:")
    m1, se1, t1 = paired(n1, k0, "n1 - k0")
    if m1 <= -0.008 and abs(m1) >= 2 * se1:
        verdict = "TRANSFERS"
    elif abs(m1) <= 0.004:
        verdict = "DOES NOT TRANSFER"
    else:
        verdict = "BETWEEN"
    print(f"  VERDICT: {verdict}  (headline untouched; fresh-seed confirm required first)")

    print("\n  READ 2 (informational):")
    m2, se2, t2 = paired(nT, k0, "nT - k0")
    paired(nT, n1, "nT - n1")

    if not a.no_ledger:
        rid = ledger.append_row(
            "w4_nodur",
            {"seeds": seeds, "n": 2000, "q": "nodur cond=[logdist,cos,sin]",
             "arms": ["k0", "n1", "nT"]},
            "ok",
            metrics={"k0_mean": float(k0.mean()), "n1_mean": float(n1.mean()),
                     "nT_mean": float(nT.mean()),
                     "n1_minus_k0": float(m1), "n1_minus_k0_se": float(se1),
                     "nT_minus_k0": float(m2), "nT_minus_k0_se": float(se2),
                     },
            artifacts=files,
            notes=f"AMENDMENT 17 stage B. {verdict}. q_nodur (duration input "
                  f"dropped, stage A retained 0.776 of 0.88 nats) serving e0 "
                  f"on the headline population. Cross-run k0 informational only "
                  f"(unseeded DurationModel rng). No headline claim from "
                  f"this arm, registered.",
            tier=1)
        ledger.regenerate_leaderboard()
        print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
