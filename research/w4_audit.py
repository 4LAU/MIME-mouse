"""An auditor that does not use the eighteen features, and that we never train
against.

L asked the right question: how do we know we are not fitting the detector
rather than building something human like? Three answers apply. Nothing here
has ever been trained against the detector, all training is likelihood on real
recordings. Every property we act on is measured on human data, not chosen to
please the scorer. And the eighteen summary features are a narrow lens, so a
model could score 0.50 on them and still be obviously wrong in some way nobody
measures.

This is the control for the third one. It is a second, independent read of the
same question that shares nothing with `scoring.py`: no resampling to 125Hz, no
summary statistics, no random forest. It works on the raw event token stream,
which is what a real defender would see.

The instrument is a likelihood ratio. Fit two order k Markov models over the
joint token alphabet, one on human streams and one on generated streams, both
on a training split. For a held out sequence read the average log likelihood
under each and take the difference. If the generator matched the human process
the two models would be interchangeable and the difference would carry no
information, so the AUC would sit at chance. Anything above chance is sequence
structure the generator gets wrong, whether or not the eighteen features can
see it.

Three alphabets, because which one separates says WHERE the structure is wrong:

  s        displacement class only, coarsened. Speed sequence structure.
  th       heading change class only, coarsened and signed. Turn sequence
           structure, which is where every measurement today has pointed.
  s,th     the joint, which catches structure in neither alone.

Two guards against fooling ourselves:

  length   sequence length alone is scored as its own arm. If length separates
           on its own then any likelihood result may just be reading length,
           and the honest report is the excess over that arm.
  shuffled the same sequences with their event order permuted within each
           trajectory. Order is the only thing a Markov model can see beyond
           the marginal, so this arm must fall to chance. If it does not, the
           instrument is reading marginals and not sequence structure.

This auditor must never be used as a training signal. Its only job is to be
the thing we did not aim at.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_audit.py --ar event_ar_v1.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

os.environ.setdefault("EVENT_CHOICE_TEMP", "10")
os.environ.setdefault("EVENT_SNAP", "2.5")
os.environ.setdefault("EVENT_DUR_STD", "1.0")
os.environ.setdefault("DUR_EMPIRICAL", "1")
os.environ.setdefault("EVENT_BESTOF", "1")
os.environ.setdefault("EVENT_SIR", "1")
os.environ.setdefault("EVENT_ORDER", "gumbel")
os.environ.setdefault("EVENT_STEPS", "100")
os.environ.setdefault("EVENT_CFG_W", "0")
os.environ["EVENT_TICKMERGE"] = "0"

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
from models.event_ar import EventARModel  # noqa: E402
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TH_NULL_CLASS, TICK_CLASS, class_to_dtheta, class_to_speed,
    dth_lattice_to_class, s2_to_class,
)
from phase_a_baseline import make_specs  # noqa: E402

S_BINS = 8
TH_BINS = 9
TH_EDGES = np.array([-2.5, -1.2, -0.5, -0.18, -0.02, 0.02, 0.18, 0.5, 1.2, 2.5])


def _symbols(s_cls, th_cls, n, kind):
    """Coarsen one trajectory into a small symbol alphabet."""
    s_cls, th_cls = s_cls[:n], th_cls[:n]
    mo = s_cls > TICK_CLASS
    sp = class_to_speed(torch.from_numpy(s_cls.astype(np.int64))).numpy()
    sp = np.where(mo, sp, 0.0)
    sb = np.clip(np.digitize(sp, [0.01, 1.0, 2.0, 3.5, 5.5, 8.5, 14.0]), 0,
                 S_BINS - 1)
    th = class_to_dtheta(torch.from_numpy(th_cls.astype(np.int64))).numpy()
    th = np.where(mo, th, 0.0)
    tb = np.clip(np.digitize(th, TH_EDGES), 0, TH_BINS - 1)
    if kind == "s":
        return sb, S_BINS
    if kind == "th":
        return tb, TH_BINS
    return sb * TH_BINS + tb, S_BINS * TH_BINS


def _fit(seqs, V, k, alpha=0.5):
    C = np.full((V ** k, V), alpha, dtype=np.float64)
    for q in seqs:
        if len(q) <= k:
            continue
        ctx = np.zeros(len(q) - k, dtype=np.int64)
        for j in range(k):
            ctx = ctx * V + q[j:len(q) - k + j]
        np.add.at(C, (ctx, q[k:]), 1.0)
    return np.log(C / C.sum(1, keepdims=True))


def _score(q, LP, V, k):
    if len(q) <= k:
        return 0.0
    ctx = np.zeros(len(q) - k, dtype=np.int64)
    for j in range(k):
        ctx = ctx * V + q[j:len(q) - k + j]
    return float(LP[ctx, q[k:]].mean())


def _auc(hs, gs):
    y = np.concatenate([np.zeros(len(hs)), np.ones(len(gs))])
    v = np.concatenate([hs, gs])
    a = roc_auc_score(y, v)
    return float(max(a, 1 - a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1500)
    ap.add_argument("--ar", default="event_ar_v1.pt")
    ap.add_argument("--masked", action="store_true")
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--out", default="research/w4_audit.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    rng = np.random.default_rng(args.seed)
    lengths = np.load("training/events_len.npy")
    pick = np.sort(rng.choice(len(lengths), args.n * 2, replace=False))
    s2 = np.load("training/events_s2.npy", mmap_mode="r")[pick]
    dth = np.load("training/events_dth.npy", mmap_mode="r")[pick]
    L = np.minimum(lengths[pick], 256)
    hs = s2_to_class(torch.from_numpy(s2.astype(np.int64))).numpy()
    hth = np.where(s2 > 0,
                   dth_lattice_to_class(torch.from_numpy(dth.astype(np.int64))).numpy(),
                   TH_NULL_CLASS)
    human = [(hs[i], hth[i], int(L[i])) for i in range(len(L))
             if int(L[i]) >= 12]

    specs = make_specs(args.n, args.seed)
    rows = []
    for sx, sy, ex, ey in specs:
        dd = math.hypot(ex - sx, ey - sy)
        if dd < 1e-6:
            continue
        ang = math.atan2(ey - sy, ex - sx)
        rows.append([math.log(dd), math.log(esp._duration.sample(math.log(dd))),
                     math.cos(ang), math.sin(ang)])

    def collect(fn):
        S, T, N = [], [], []
        for c0 in range(0, len(rows), args.batch):
            cond = torch.tensor(rows[c0:c0 + args.batch], dtype=torch.float32,
                                device=dev)
            a, b = fn(cond)
            a, b = a.cpu().numpy(), b.cpu().numpy()
            pad = a >= S_PAD_CLASS
            S.append(a)
            T.append(b)
            N.append(np.where(pad.any(1), pad.argmax(1), a.shape[1]))
            print(f"    sampled {sum(len(x) for x in S):,}", flush=True)
        a, b, n = np.concatenate(S), np.concatenate(T), np.concatenate(N)
        return [(a[i], b[i], int(n[i])) for i in range(len(n))
                if int(n[i]) >= 12]

    arms = {}
    if args.masked:
        m, seq_len = esp._model, esp._cfg["max_seq_len"]

        def masked_fn(cond):
            with torch.no_grad():
                _, a, b = m.sample(
                    cond, seq_len, n_steps=100, temperature=args.temp,
                    order="gumbel", choice_temp=10.0,
                    feat=torch.zeros(cond.shape[0], esp._FEAT_BANK.shape[1],
                                     device=dev) if esp._FEAT_BANK is not None else None)
            return a, b
        arms["masked served"] = collect(masked_fn)

    for ck_name in [c for c in args.ar.split(",") if c]:
        ck = torch.load(f"training/{ck_name}", map_location=dev,
                        weights_only=False)
        model = EventARModel(**ck["config"]).to(dev).eval()
        model.load_state_dict(ck["model_state_dict"])

        def ar_fn(cond, _m=model):
            a, b, _c = _m.sample(cond, temperature=args.temp)
            return a, b
        arms[f"ar {ck_name.replace('.pt', '')}"] = collect(ar_fn)

    out = {}
    print(f"\n  order {args.order} markov likelihood ratio, held out halves")
    print(f"  {'arm':<20}{'alphabet':>10}{'auc':>9}{'lenAuc':>9}"
          f"{'shufAuc':>9}{'nH':>7}{'nG':>7}")
    for label, gen in arms.items():
        for kind in ("s", "th", "s,th"):
            res = {}
            for tag, shuf in (("real", False), ("shuf", True)):
                H, G = [], []
                for src, dst in ((human, H), (gen, G)):
                    for s_cls, th_cls, n in src:
                        q, V = _symbols(s_cls, th_cls, n, kind)
                        if shuf:
                            q = q[rng.permutation(len(q))]
                        dst.append(q)
                hi = rng.permutation(len(H))
                gi = rng.permutation(len(G))
                ht, he = hi[:len(hi) // 2], hi[len(hi) // 2:]
                gt, ge = gi[:len(gi) // 2], gi[len(gi) // 2:]
                LPh = _fit([H[i] for i in ht], V, args.order)
                LPg = _fit([G[i] for i in gt], V, args.order)
                sh = [_score(H[i], LPh, V, args.order)
                      - _score(H[i], LPg, V, args.order) for i in he]
                sg = [_score(G[i], LPh, V, args.order)
                      - _score(G[i], LPg, V, args.order) for i in ge]
                res[tag] = _auc(np.asarray(sh), np.asarray(sg))
                if tag == "real":
                    res["nH"], res["nG"] = len(he), len(ge)
            lens_h = np.array([len(q) for q in H])
            lens_g = np.array([len(q) for q in G])
            res["len"] = _auc(lens_h, lens_g)
            out[f"{label} | {kind}"] = res
            print(f"  {label:<20}{kind:>10}{res['real']:>9.4f}"
                  f"{res['len']:>9.4f}{res['shuf']:>9.4f}"
                  f"{res['nH']:>7}{res['nG']:>7}", flush=True)

    json.dump(out, open(args.out, "w"), indent=2)
    print("\n  this auditor shares nothing with scoring.py and must never be")
    print("  used as a training signal. shufAuc must sit near chance or the")
    print("  instrument is reading marginals rather than sequence order, and")
    print("  any result below lenAuc is really a statement about length.")


if __name__ == "__main__":
    main()
