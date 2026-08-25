"""w4_durmatch aggregator. Reads registered in AMENDMENT 16, step0_prereg.md.

READ 1 (descriptive): k0(durmatch) minus k0(confirm) per seed, same seeds so
same specs, durations differ. READ 2 (PRIMARY): paired q1 minus k0 inside
durmatch; <= -0.008 at 2 paired se -> H_dur SUPPORTED; |mean| <= 0.004 ->
H_small SUPPORTED; else BETWEEN. READ 3: qT minus k0, informational.
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

    files = sorted(f for f in glob.glob(os.path.join(a.dir, "w4_durmatch_s*.json"))
                   if re.search(r"_s\d+\.json$", f))
    runs = [json.load(open(f)) for f in files]
    if not runs:
        print("no runs found"); sys.exit(1)
    runs.sort(key=lambda r: r["seed"])
    seeds = [r["seed"] for r in runs]

    k0 = np.array([r["arms"]["k0"]["contract"] for r in runs])
    q1 = np.array([r["arms"]["q1"]["contract"] for r in runs])
    qT = np.array([r["arms"]["qT"]["contract"] for r in runs])

    print(f"  seeds {seeds}")
    print(f"  {'seed':>6}{'k0':>9}{'q1':>9}{'qT':>9}")
    for r in runs:
        print(f"  {r['seed']:>6}{r['arms']['k0']['contract']:>9.4f}"
              f"{r['arms']['q1']['contract']:>9.4f}{r['arms']['qT']['contract']:>9.4f}")

    def paired(x, y, name):
        d = x - y
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        t = m / se if se > 0 else float("inf")
        print(f"  {name}: mean {m:+.4f}  se {se:.4f}  t {t:+.2f}  per seed "
              + " ".join(f"{v:+.4f}" for v in d))
        return m, se, t

    # READ 1, descriptive: k0 shift vs the confirm's k0 on the same seeds.
    ck0 = {}
    for s in seeds:
        f = os.path.join(a.dir, f"w4_fhconfirm_s{s}.json")
        if os.path.exists(f):
            ck0[s] = json.load(open(f))["arms"]["k0"]["contract"]
    print("\n  READ 1 (descriptive), k0 durmatch minus k0 confirm, same seeds:")
    if len(ck0) == len(seeds):
        m1, se1, t1 = paired(k0, np.array([ck0[s] for s in seeds]), "k0 shift")
    else:
        print(f"    confirm jsons missing for seeds "
              f"{[s for s in seeds if s not in ck0]}, read skipped")
        m1 = se1 = None

    print("\n  READ 2 (PRIMARY), q1 minus k0 paired inside durmatch:")
    m2, se2, t2 = paired(q1, k0, "q1 - k0")
    if m2 <= -0.008 and abs(m2) >= 2 * se2:
        verdict = "H_dur SUPPORTED"
    elif abs(m2) <= 0.004:
        verdict = "H_small SUPPORTED"
    else:
        verdict = "BETWEEN"
    print(f"  VERDICT: {verdict}")

    print("\n  READ 3 (informational), qT minus k0 paired inside durmatch:")
    m3, se3, t3 = paired(qT, k0, "qT - k0")

    print("\n  no headline, no serve decision from this arm (registered)")

    if not a.no_ledger:
        rid = ledger.append_row(
            "w4_durmatch",
            {"seeds": seeds, "n": 2000, "durations": "matched k=64 heldout",
             "arms": ["k0", "q1", "qT"]},
            "ok",
            metrics={"k0_mean": float(k0.mean()), "q1_mean": float(q1.mean()),
                     "qT_mean": float(qT.mean()),
                     "q1_minus_k0": float(m2), "q1_minus_k0_se": float(se2),
                     "qT_minus_k0": float(m3), "qT_minus_k0_se": float(se3),
                     "k0_shift_vs_confirm": None if m1 is None else float(m1),
                     "k0_shift_se": None if se1 is None else float(se1)},
            artifacts=files,
            notes=f"AMENDMENT 16 discriminator. {verdict}. Matched durations "
                  f"(empirical p(log dur | log dist), 64-NN heldout rows) in "
                  f"place of esp._duration.sample. No headline, no serve "
                  f"decision, registered in advance.",
            tier=1)
        ledger.regenerate_leaderboard()
        print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
