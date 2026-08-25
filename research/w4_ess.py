"""w4_ess. How many INDEPENDENT events does a forty event trajectory carry?

Registered in /home/aaronadmin/w4_arms/step0_prereg.md, AMENDMENT 7, which
exists to attack w4_k0power's conclusion rather than to support it. Read it
first.

The short version. w4_prefix measures a detector lift that is FLAT in the number
of events read. w4_k0power excludes two explanations with a plant that is drawn
INDEPENDENTLY at every position. The model's signal rides on a mouse
trajectory, where consecutive events are smooth. If forty consecutive events
carry the information of two, a flat curve is what ANY signal would produce and
the flatness says nothing about where the defect lives.

The statistic, for a coarse bin b with indicator x and rowmean_m the average of
x over a row's first m events:

    ESS(m) = Var_pooled(x) / Var_rows(rowmean_m(x))

which is m under within row independence and 1 under perfect redundancy. Bins
are aggregated by summing both variances, which weights by variance and stops
rare bins dominating.

Note on interpretation, fixed before the run. Var_rows(rowmean_m) contains a
between row component that does NOT shrink with m, so ESS saturates. That
cannot fake a small reading: for a Bernoulli indicator Var_within(1) = p(1-p)
is the largest the within component can be and the between component can never
exceed it, so ESS(1) = 1 always and any saturation below 40 is redundancy, not
heterogeneity.

CPU only. Nothing generated, no checkpoint touched.
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

from models.event_ar import N_DT_CLASSES                          # noqa: E402
from models.event_stream_polar import S_PAD_CLASS, TH_NULL_CLASS  # noqa: E402
from w4_detcap import (CORPUS, NDT_B, NS_B, NTH_B, corpus_tokens,  # noqa: E402
                       model_tokens)

MMAX = 40
MS = (1, 2, 3, 5, 10, 20, 40)
NBINS = {"s": NS_B, "th": NTH_B + 1, "dt": NDT_B}


def coarse(s, th, dt):
    """The detector's own alphabet, copied from `bin_streams`."""
    sb = np.minimum(s, S_PAD_CLASS - 1) * NS_B // S_PAD_CLASS
    tb = np.where(th >= TH_NULL_CLASS, NTH_B,
                  np.minimum(th, TH_NULL_CLASS - 1) * NTH_B // TH_NULL_CLASS)
    db = np.minimum(dt, N_DT_CLASSES - 1) * NDT_B // N_DT_CLASSES
    return {"s": sb, "th": tb, "dt": db}


def ess_curve(codes, nb):
    """codes is (n rows, MMAX). Returns ESS at every m in MS, and the bin
    marginals so the caller can rebuild an independent surrogate."""
    n = len(codes)
    p = np.bincount(codes.ravel(), minlength=nb) / codes.size
    num = float((p * (1 - p)).sum())
    out = []
    for m in MS:
        vr = 0.0
        for b in range(nb):
            if p[b] <= 0:
                continue
            rm = (codes[:, :m] == b).mean(1)
            vr += float(rm.var(ddof=1))
        out.append(num / vr if vr > 0 else float("nan"))
    return out, p, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="+",
                    default=["research/w4_texcover_streams_s0.npz",
                             "research/w4_texcover_streams_s1.npz"])
    ap.add_argument("--nhuman", type=int, default=20000)
    ap.add_argument("--out", default="research/w4_ess_results.json")
    args = ap.parse_args()
    print("w4_ess. how many independent events does a 40 event row carry. "
          "CPU, nothing generated.", flush=True)
    rng = np.random.default_rng(9800)

    Lall = np.load(CORPUS / "events_len.npy")
    elig = np.flatnonzero(Lall >= MMAX)
    ids = rng.choice(elig, min(args.nhuman, len(elig)), replace=False)
    hs, hth, hdt, _, _ = corpus_tokens(ids)
    H = coarse(hs[:, :MMAX], hth[:, :MMAX], hdt[:, :MMAX])
    del hs, hth, hdt

    S, TH, DT, LL = [], [], [], []
    for f in args.streams:
        a, b, c, _, L = model_tokens(f)
        S.append(a); TH.append(b); DT.append(c); LL.append(L)
    S, TH, DT = np.concatenate(S), np.concatenate(TH), np.concatenate(DT)
    LL = np.concatenate(LL)
    lng = np.asarray(LL) >= MMAX
    M = coarse(S[lng][:, :MMAX], TH[lng][:, :MMAX], DT[lng][:, :MMAX])
    print(f"  human rows {len(ids)}   model rows {int(lng.sum())}   "
          f"first {MMAX} events", flush=True)

    res = {"ms": list(MS), "human": {}, "model": {}, "iid": {}}
    print(f"\n  ESS(m). {MMAX} means the events are independent, 1 means they "
          f"repeat")
    print(f"  {'arm':>7}{'ch':>5}" + "".join(f"{m:>8}" for m in MS))
    for tag, D in (("human", H), ("model", M)):
        for ch, nb in NBINS.items():
            e, p, n = ess_curve(D[ch], nb)
            res[tag][ch] = e
            print(f"  {tag:>7}{ch:>5}" + "".join(f"{v:>8.2f}" for v in e),
                  flush=True)
            if tag == "human":
                # E0 SANITY. Same n, same marginal, drawn independently.
                sur = rng.choice(nb, size=(n, MMAX), p=p)
                es, _, _ = ess_curve(sur, nb)
                res["iid"][ch] = es
    print(f"\n  E0 SANITY. independent surrogate at the human marginal, must "
          f"read near {MMAX} at m = {MMAX}")
    print(f"  {'arm':>7}{'ch':>5}" + "".join(f"{m:>8}" for m in MS))
    e0 = True
    for ch in NBINS:
        v = res["iid"][ch]
        print(f"  {'iid':>7}{ch:>5}" + "".join(f"{x:>8.2f}" for x in v))
        e0 = e0 and abs(v[-1] - MMAX) <= 0.15 * MMAX
    print(f"  E0  -> {'PASS' if e0 else 'FAIL, the estimator is broken and '
                                        'nothing below is read'}")

    # ---- E1 / E2 / E3 ------------------------------------------------------
    hs_, hd_ = res["human"]["s"][-1], res["human"]["dt"][-1]
    key = max(hs_, hd_)
    if not e0:
        verdict = "VOID, E0 failed"
    elif hs_ < 2 and hd_ < 2:
        verdict = ("REDUNDANCY EXPLAINS IT. OUTCOME 4's PER TRAJECTORY reading "
                   "is WITHDRAWN")
    elif key > 6:
        verdict = ("REDUNDANCY DOES NOT EXPLAIN IT. the PER TRAJECTORY reading "
                   "stands")
    else:
        verdict = "PARTIAL"
    print(f"\n  E1/E2/E3. human ESS(40) is {hs_:.2f} on s and {hd_:.2f} on dt")
    print(f"    the lift curve SHOULD have grown by sqrt(ESS) = "
          f"{np.sqrt(key):.2f}x from m = 1 to m = 40")
    print(f"    it grew by 0.0174 / 0.0176 = 0.99x")
    ratios = {ch: res["model"][ch][-1] / res["human"][ch][-1]
              for ch in NBINS if res["human"][ch][-1] > 0}
    print(f"\n  E4 MODEL vs HUMAN ESS(40) ratio  " +
          "  ".join(f"{c} {v:.2f}" for c, v in ratios.items()))
    big = {c: v for c, v in ratios.items() if v > 2 or v < 0.5}
    print(f"    channels differing by more than 2x  {list(big) or 'none'}")
    print(f"\n  VERDICT  {verdict}", flush=True)

    res.update({"e0": e0, "human_ess40_s": hs_, "human_ess40_dt": hd_,
                "sqrt_expected_growth": float(np.sqrt(key)),
                "measured_growth": 0.0174 / 0.0176,
                "model_over_human": ratios, "verdict": verdict})
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
