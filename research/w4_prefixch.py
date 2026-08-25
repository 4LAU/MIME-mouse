"""w4_prefixch. Which channel carries the flat lift, and does its own
redundancy explain the flatness?

Registered in /home/aaronadmin/w4_arms/step0_prereg.md, AMENDMENT 8. Identical
to `w4_prefix` in rows, condition matching, draws, seeds and detector. The ONE
change is that the detector reads one channel's columns at a time, because
`w4_ess` measured wildly different within row redundancy per channel (s 6.81,
th 9.51, dt 2.05 independent events out of forty) and the 49 column detector
cannot say which of those the flat curve is riding on.

CPU only. Nothing generated.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

os.environ.setdefault("EVENT_CHOICE_TEMP", "10")
for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

from w4_detcap import CORPUS, corpus_tokens, model_tokens          # noqa: E402
from w4_prefix import auc, uni_m                                   # noqa: E402
from w4_step0 import NCELL, cell_cuts, cells, draw_matched, prep    # noqa: E402

MMAX = 40
MS = (1, 2, 3, 5, 10, 20, 40)
DRAWS = 8
NULL_SEEDS = (401, 402, 403, 404, 405, 406, 407, 408)

# uni_m concatenates 16 step size columns, 17 direction columns and 16 timing
# columns, in that order. Asserted at run time against its own output width.
COLS = {"s": np.arange(0, 16), "th": np.arange(16, 33), "dt": np.arange(33, 49)}
# Constants from w4_ess OUTCOME 5, not recomputed here.
SQRT_ESS = {"s": 2.61, "th": 3.08, "dt": 1.43}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="+",
                    default=["research/w4_texcover_streams_s0.npz",
                             "research/w4_texcover_streams_s1.npz"])
    ap.add_argument("--hpool", type=int, default=400000)
    ap.add_argument("--out", default="research/w4_prefixch_results.json")
    args = ap.parse_args()
    print("w4_prefixch. which channel carries the flat lift. CPU, nothing "
          "generated.", flush=True)
    rng = np.random.default_rng(8000 + MMAX)

    S, TH, DT, CD, LL = [], [], [], [], []
    for f in args.streams:
        a, b, c, d, L = model_tokens(f)
        S.append(a); TH.append(b); DT.append(c); CD.append(d); LL.append(L)
    S, TH, DT = np.concatenate(S), np.concatenate(TH), np.concatenate(DT)
    CD, LL = np.concatenate(CD), np.concatenate(LL)
    S, TH, DT, keep = prep(S, TH, DT, LL, MMAX)
    CD = CD[keep]

    Lall = np.load(CORPUS / "events_len.npy")
    Call = np.load(CORPUS / "events_cond.npy")
    elig = np.flatnonzero(Lall >= MMAX)
    elig = rng.choice(elig, min(args.hpool, len(elig)), replace=False)
    hc = np.asarray(Call[elig], dtype=np.float32)
    ccuts = cell_cuts(hc)
    hcell, mcell = cells(hc, ccuts), cells(CD, ccuts)
    want = np.bincount(mcell, minlength=NCELL)
    pool = [elig[hcell == c] for c in range(NCELL)]
    print(f"  MMAX {MMAX}   model rows {len(S)}   human pool {len(elig)}",
          flush=True)

    def human(ids):
        a, b, c, _, L = corpus_tokens(ids)
        a, b, c, _ = prep(a, b, c, L, MMAX)
        return a, b, c

    MU = {m: uni_m(S, TH, DT, m) for m in MS}
    assert MU[1].shape[1] == 49, MU[1].shape

    # Human draws are decoded ONCE and shared by all three channels, so the
    # three readings differ only in which columns the forest is given.
    arm_h = [human(draw_matched(pool, want, np.random.default_rng(8200 + d))[0])
             for d in range(DRAWS)]
    HU = [{m: uni_m(*h, m) for m in MS} for h in arm_h]

    null_h = []
    for sd in NULL_SEEDS:
        ids, _ = draw_matched(pool, want, np.random.default_rng(sd), mult=2)
        o = np.random.default_rng(sd + 50).permutation(len(ids))
        ids = ids[o]
        h = len(ids) // 2
        A, B = human(ids[:h]), human(ids[h:])
        null_h.append(({m: uni_m(*A, m) for m in MS},
                       {m: uni_m(*B, m) for m in MS}))

    res, ntable = {}, {}
    print(f"\n  {'ch':>4}{'m':>5}{'arm':>9}{'null':>9}{'lift':>9}{'null se':>10}")
    for ch, cc in COLS.items():
        arm = np.array([[auc(HU[d][m][:, cc], MU[m][:, cc], 8300 + d)
                         for m in MS] for d in range(DRAWS)]).mean(0)
        nac = np.array([[auc(A[m][:, cc], B[m][:, cc], NULL_SEEDS[i] + 7)
                         for m in MS] for i, (A, B) in enumerate(null_h)])
        null = nac.mean(0)
        nsd = nac.std(0, ddof=1) / np.sqrt(len(NULL_SEEDS))
        lift = arm - null
        res[ch] = {"arm": arm.tolist(), "null": null.tolist(),
                   "lift": lift.tolist(), "nsd": nsd.tolist()}
        ntable[ch] = (lift, nsd)
        for i, m in enumerate(MS):
            print(f"  {ch:>4}{m:>5}{arm[i]:>9.4f}{null[i]:>9.4f}"
                  f"{lift[i]:>+9.4f}{nsd[i]:>10.4f}", flush=True)

    # ---- S1. which channel carries it -------------------------------------
    print(f"\n  S1 WHICH CHANNEL. lift(40) against the 49 column detector's "
          f"+0.0174")
    seen = {}
    for ch in COLS:
        lift, nsd = ntable[ch]
        z = lift[-1] / max(nsd[-1], 1e-9)
        seen[ch] = bool(lift[-1] > 3 * nsd[-1])
        print(f"  {ch:>4}  lift(40) {lift[-1]:>+8.4f}  {z:>6.2f} sd  "
              f"{'READ' if seen[ch] else 'not resolved'}")
    if not any(seen.values()):
        print("  S1 FAIL, no channel resolves. OUTCOME 4 downgraded to PARTIAL "
              "on row count alone.")

    # ---- S2. shortfall on the leading channel -----------------------------
    print(f"\n  S2 GROWTH AND SHORTFALL")
    print(f"  {'ch':>4}{'lift(1)':>10}{'lift(40)':>10}{'growth':>9}"
          f"{'sqrt(ESS)':>11}{'shortfall':>11}")
    short = {}
    for ch in COLS:
        lift, _ = ntable[ch]
        g = float(lift[-1] / lift[0]) if abs(lift[0]) > 1e-9 else float("nan")
        sf = SQRT_ESS[ch] / g if g == g and abs(g) > 1e-9 else float("nan")
        short[ch] = sf
        print(f"  {ch:>4}{lift[0]:>+10.4f}{lift[-1]:>+10.4f}{g:>9.2f}"
              f"{SQRT_ESS[ch]:>11.2f}{sf:>11.2f}")

    lead = max(COLS, key=lambda c: ntable[c][0][-1])
    sfl = short[lead]
    if not any(seen.values()):
        verdict = "PARTIAL, no channel resolves"
    elif not seen[lead]:
        verdict = "PARTIAL, the leading channel does not resolve"
    elif sfl != sfl:
        verdict = "PARTIAL, growth undefined on the leading channel"
    elif sfl > 2:
        verdict = (f"PER TRAJECTORY READING STANDS on {lead}, shortfall "
                   f"{sfl:.2f}")
    elif sfl < 1.5:
        verdict = (f"DOWNGRADED TO PARTIAL, {lead} shortfall only {sfl:.2f}")
    else:
        verdict = f"PARTIAL, {lead} shortfall {sfl:.2f} between 1.5 and 2"
    print(f"\n  leading channel {lead}")
    # S3 is a contradiction check, not a gate.
    if lead == "th":
        print("  S3 CONTRADICTION. th leads here, yet w4_k0power could not "
              "plant on th at ANY strength and OUTCOME 2 measured why. Two of "
              "my own measurements disagree and neither is resolved in favour "
              "of the other.")
    print(f"\n  VERDICT  {verdict}", flush=True)

    out = {"ms": list(MS), "channels": res, "seen": seen,
           "shortfall": short, "lead": lead, "sqrt_ess": SQRT_ESS,
           "verdict": verdict}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
