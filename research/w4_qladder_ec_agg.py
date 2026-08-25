"""w4_e1chan aggregator. Reads registered in AMENDMENT 29,
step0_prereg.md.

Nested channel decomposition of the human e1's value behind the human
e0, within the q1g0 completion family. Pairs h0p1 from the AMENDMENT
28 files and h02 from the AMENDMENT 25 files. inc_s = h0s1p minus
h0p1; inc_th = h0st1p minus h0s1p; inc_dt = h02 minus h0st1p. Per
channel: CARRIES if <= -0.008 at 2 se; NULL RANGE if |mean| <= 0.004;
else BETWEEN.
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

    ec_files = sorted(f for f in glob.glob(os.path.join(a.dir, "w4_qladder_ec_s*.json"))
                      if re.search(r"_s\d+\.json$", f))
    ec_runs = sorted((json.load(open(f)) for f in ec_files), key=lambda r: r["seed"])
    if not ec_runs:
        print("no ec runs found"); sys.exit(1)
    seeds = [r["seed"] for r in ec_runs]
    ps = {s: json.load(open(os.path.join(a.dir, f"w4_qladder_ps_s{s}.json")))
          for s in seeds}
    kf = {s: json.load(open(os.path.join(a.dir, f"w4_qladder_kf_s{s}.json")))
          for s in seeds}

    hs = np.array([r["arms"]["h0s1p"]["contract"] for r in ec_runs])
    hst = np.array([r["arms"]["h0st1p"]["contract"] for r in ec_runs])
    hp = np.array([ps[s]["arms"]["h0p1"]["contract"] for s in seeds])
    h02 = np.array([kf[s]["arms"]["h02"]["contract"] for s in seeds])

    print(f"  seeds {seeds}")
    print("    seed     h0p1    h0s1p   h0st1p      h02")
    for i, s in enumerate(seeds):
        print(f"  {s:>6}{hp[i]:>9.4f}{hs[i]:>9.4f}{hst[i]:>9.4f}{h02[i]:>9.4f}")

    def stats(d, name):
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        t = m / se if se > 0 else float("inf")
        print(f"  {name}: mean {m:+.4f}  se {se:.4f}  t {t:+.2f}  per seed "
              + " ".join(f"{v:+.4f}" for v in d))
        return m, se, t

    def verdict(m, se):
        if m <= -0.008 and abs(m) >= 2 * se:
            return "CARRIES"
        if abs(m) <= 0.004:
            return "NULL RANGE"
        return "BETWEEN"

    print("\n  Nested increments, each the value of the human realization of"
          " one channel:")
    m_s, se_s, _ = stats(hs - hp, "inc_s  (h0s1p - h0p1) ")
    v_s = verdict(m_s, se_s)
    print(f"    s verdict: {v_s}")
    m_t, se_t, _ = stats(hst - hs, "inc_th (h0st1p - h0s1p)")
    v_t = verdict(m_t, se_t)
    print(f"    th verdict: {v_t}")
    m_d, se_d, _ = stats(h02 - hst, "inc_dt (h02 - h0st1p)  ")
    v_d = verdict(m_d, se_d)
    print(f"    dt verdict: {v_d}")
    tot = (h02 - hp).mean()
    print(f"\n  sum of increments {m_s + m_t + m_d:+.4f} = h02 minus h0p1"
          f" {tot:+.4f} (construction check)")

    print("\n  no headline, no serve decision from this arm (registered)")

    if not a.no_ledger:
        rid = ledger.append_row(
            "w4_e1chan",
            {"seeds": seeds, "n": 2000, "arms": ["h0s1p", "h0st1p"],
             "serve": "human e0 forced; q1g0 completes position 1 behind"
                      " forced human s1 (and th1), draw +13, served temps"
                      " from position 2",
             "paired_against": ["w4_pairsplit_2026-08-25T012510+0000_489c9097",
                                "w4_kfill_2026-08-23T013629+0000_dc718032"]},
            "ok",
            metrics={"h0s1p_mean": float(hs.mean()),
                     "h0st1p_mean": float(hst.mean()),
                     "inc_s": float(m_s), "inc_s_se": float(se_s),
                     "inc_th": float(m_t), "inc_th_se": float(se_t),
                     "inc_dt": float(m_d), "inc_dt_se": float(se_d)},
            artifacts=ec_files,
            notes=f"AMENDMENT 29 e1 channel decomposition within the q1g0"
                  f" completion family. s {v_s}, th {v_t}, dt {v_d}."
                  f" No headline, registered in advance.",
            tier=1)
        ledger.regenerate_leaderboard()
        print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
