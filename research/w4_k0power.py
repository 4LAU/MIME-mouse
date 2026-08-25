"""w4_k0power. How strong must a FIRST EVENT ONLY defect be before the prefix
detector can see it, and does its curve decay?

Registered in /home/aaronadmin/w4_arms/step0_prereg.md, AMENDMENT 6. Read it
first. The short version: w4_prefix returned UNREADABLE because its k0 power
control was invisible, and the reason is arithmetic, not a property of the
model. A plant at rate p on events 5..39 perturbs every row's prefix average a
little. The same p at event 0 perturbs p of ROWS completely and leaves the rest
identical. Those are not the same size of signal, and the registration treated
them as if they were.

So the plant rate is swept instead of asserted, and the read is the SHAPE of the
resulting curve, not its height. If a k0 confined effect decays with m and the
model's curve is flat, the model's defect is not confined to the first event. If
a k0 confined effect is ALSO flat, this statistic cannot tell the two apart and
the whole construction returns NOT SEPARABLE.

No model token is read. Model condition vectors are, because the matched draw
needs the arm's own per cell counts.

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
from w4_poskl import channels, make_cuts                           # noqa: E402
from w4_prefix import MOD, auc, uni_m                              # noqa: E402
from w4_step0 import (NCELL, bin_shift, cell_cuts, cells,          # noqa: E402
                      draw_matched, plant_at, prep)

MMAX = 40
MS = (1, 2, 3, 5, 10, 20, 40)
NULL_SEEDS = (401, 402, 403, 404, 405, 406, 407, 408)
PSEEDS = NULL_SEEDS[:4]

# AMENDMENT 6, fixed before the run. p = 1.00 is the ceiling of what a k0
# confined perturbation can be at this shift size, and it is in the ladder so
# that R1 can fail.
LADDER = {"s":  (0.05, 0.15, 0.35, 0.70, 1.00),
          "th": (0.02, 0.10, 0.30, 0.80, 1.00),
          "dt": (0.05, 0.15, 0.35, 0.70, 1.00)}

# The model's own measured curve at MMAX 40, from w4_prefix's paired rerun.
# Constants, not recomputed here. Nothing in this file reads a model token.
MODEL_LIFT1 = 0.0176
MODEL_RATIO = 0.0174 / 0.0176


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="+", required=True)
    ap.add_argument("--hpool", type=int, default=400000)
    ap.add_argument("--out", default="research/w4_k0power_results.json")
    args = ap.parse_args()

    print("w4_k0power. how strong must a first event defect be to be seen. "
          "CPU, nothing generated.", flush=True)
    rng = np.random.default_rng(9000 + MMAX)

    # ---- rows, matched exactly as w4_prefix matched them -------------------
    S, TH, DT, CD, LL = [], [], [], [], []
    for f in args.streams:
        a, b, c, d, L = model_tokens(f)
        S.append(a); TH.append(b); DT.append(c); CD.append(d); LL.append(L)
    S, TH, DT = np.concatenate(S), np.concatenate(TH), np.concatenate(DT)
    CD, LL = np.concatenate(CD), np.concatenate(LL)
    S, TH, DT, keep = prep(S, TH, DT, LL, MMAX)
    CD = CD[keep]
    nmodel = len(S)
    del S, TH, DT           # tokens are not used past this point, by design

    Lall = np.load(CORPUS / "events_len.npy")
    Call = np.load(CORPUS / "events_cond.npy")
    elig = np.flatnonzero(Lall >= MMAX)
    elig = rng.choice(elig, min(args.hpool, len(elig)), replace=False)
    hc = np.asarray(Call[elig], dtype=np.float32)

    ccuts = cell_cuts(hc)
    hcell, mcell = cells(hc, ccuts), cells(CD, ccuts)
    want = np.bincount(mcell, minlength=NCELL)
    pool = [elig[hcell == c] for c in range(NCELL)]
    print(f"  MMAX {MMAX}   cells from {nmodel} model rows   "
          f"human pool {len(elig)}", flush=True)

    ref_ids = rng.choice(elig, min(20000, len(elig)), replace=False)
    rs, rth, rdt, _, rL = corpus_tokens(ref_ids)
    rs, rth, rdt, _ = prep(rs, rth, rdt, rL, MMAX)
    rcuts = make_cuts(channels(rs, rth, rdt))
    shifts = {c: bin_shift(rcuts[c]) for c in ("s", "th", "dt")}
    del rs, rth, rdt
    print(f"  plant shifts from a plain human reference {shifts}", flush=True)

    def human(ids):
        a, b, c, _, L = corpus_tokens(ids)
        a, b, c, _ = prep(a, b, c, L, MMAX)
        return a, b, c

    ks0 = np.array([0])

    def null_pair(seed, plant=None):
        """Identical to w4_prefix's. Same seed, same ids, same split."""
        ids, _ = draw_matched(pool, want, np.random.default_rng(seed), mult=2)
        o = np.random.default_rng(seed + 50).permutation(len(ids))
        ids = ids[o]
        h = len(ids) // 2
        A, B = human(ids[:h]), human(ids[h:])
        if plant is not None:
            pr, ch = plant
            B = plant_at(*B, np.random.default_rng(seed + 90), pr, ks0, ch,
                         shifts[ch], MOD[ch])
        return [auc(uni_m(*A, m), uni_m(*B, m), seed + 7) for m in MS]

    # ---- the unplanted baseline. eight seeds for the se, four for pairing --
    nacc = np.array([null_pair(sd) for sd in NULL_SEEDS])
    nsd = nacc.std(0, ddof=1) / np.sqrt(len(NULL_SEEDS))
    base = np.array([nacc[NULL_SEEDS.index(sd)] for sd in PSEEDS])
    se1 = float(nsd[0])
    gate1 = 3.0 * se1
    print(f"\n  unplanted null se by m  " +
          "".join(f"{v:>8.4f}" for v in nsd), flush=True)
    print(f"  R1/R2 bar at m = 1 is 3 x {se1:.4f} = {gate1:.4f}. the model's "
          f"own lift(1) is {MODEL_LIFT1:+.4f}", flush=True)

    # ---- the ladder --------------------------------------------------------
    res = {}
    print(f"\n  k0 ONLY PLANT LADDER, paired, {len(PSEEDS)} seeds")
    print(f"  {'ch':>4}{'p':>7}" + "".join(f"{m:>8}" for m in MS) +
          f"{'psd(1)':>9}{'r40/1':>8}")
    for ch, rates in LADDER.items():
        res[ch] = []
        for pr in rates:
            pa = np.array([null_pair(sd, plant=(pr, ch)) for sd in PSEEDS])
            d = pa - base
            lift, psd = d.mean(0), d.std(0, ddof=1) / np.sqrt(len(PSEEDS))
            ratio = (float(lift[-1] / lift[0])
                     if abs(lift[0]) > 1e-9 else float("nan"))
            res[ch].append({"p": pr, "lift": lift.tolist(),
                            "psd": psd.tolist(), "ratio": ratio,
                            "seen1": bool(lift[0] > gate1)})
            print(f"  {ch:>4}{pr:>7.2f}" +
                  "".join(f"{v:>+8.4f}" for v in lift) +
                  f"{psd[0]:>9.4f}{ratio:>8.2f}", flush=True)

    # ---- R1, R2, R3 --------------------------------------------------------
    ceil_ok = {ch: res[ch][-1]["seen1"] for ch in LADDER}
    r1 = any(ceil_ok.values())
    print(f"\n  R1 CEILING. at p = 1.00 lift(1) clears {gate1:.4f} on "
          f"{[c for c in LADDER if ceil_ok[c]]}  -> "
          f"{'PASS' if r1 else 'FAIL, the statistic is structurally blind'}")

    thr, shape = {}, {}
    for ch in LADDER:
        hit = next((r for r in res[ch] if r["seen1"]), None)
        thr[ch] = hit["p"] if hit else None
        shape[ch] = hit["ratio"] if hit else None
    print(f"\n  R2 THRESHOLD. smallest p whose lift(1) clears the bar")
    for ch in LADDER:
        t = thr[ch]
        print(f"  {ch:>4}  p* = {('%.2f' % t) if t is not None else 'none'}"
              f"   lift(1) there "
              f"{(('%+.4f' % next(r['lift'][0] for r in res[ch] if r['seen1'])) if t is not None else '   n/a')}"
              f"   against the model's {MODEL_LIFT1:+.4f}")

    print(f"\n  R3 SHAPE. lift(40)/lift(1) for the k0 plant at p*, predicted "
          f"near 1/sqrt(40) = 0.158")
    seps = []
    for ch in LADDER:
        if shape[ch] is None:
            print(f"  {ch:>4}  no p* reached, no shape")
            continue
        sep = shape[ch] < 0.5
        seps.append(sep)
        print(f"  {ch:>4}  ratio {shape[ch]:>6.2f}  -> "
              f"{'SEPARATES' if sep else 'does NOT separate'}")
    if not r1:
        verdict = "STRUCTURALLY BLIND"
    elif not seps:
        verdict = "NO THRESHOLD REACHED"
    elif any(seps):
        verdict = ("SEPARATES. the model's own ratio is "
                   f"{MODEL_RATIO:.2f}, so the model's defect is NOT confined "
                   "to the first event")
    else:
        verdict = ("NOT SEPARABLE. a k0 confined effect is flat in m too, so "
                   "this statistic cannot answer the question at any strength")
    print(f"\n  VERDICT  {verdict}", flush=True)

    out = {"mmax": MMAX, "ms": list(MS), "nmodel": nmodel, "shifts": shifts,
           "null_se": nsd.tolist(), "gate1": gate1, "ladder": res,
           "r1": r1, "threshold": thr, "shape": shape,
           "model_lift1": MODEL_LIFT1, "model_ratio": MODEL_RATIO,
           "verdict": verdict}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
