"""How much of the turning is organised, and how much just cancels?

Three hypotheses died getting here and the negative results are the useful part.

`w4_statevisit` killed the idea that the AR model never enters the state where
people correct. It enters it MORE than they do, 0.1398 against 0.1282.

`w4_speedturn` killed the idea that the model cannot turn while moving fast.
Its sharp turn share in the three fastest bands is 0.032, 0.035, 0.039 against
a human 0.032, 0.027, 0.037. Event by event the speed and turn joint is right.

A direct look at the corpus then killed the idea that human corrections are
sustained arcs. Human signed heading change has lag 1 autocorrelation -0.337,
strongly ALTERNATING, with a median same sign run of one event. Most human
turning is cancelling micro jitter, exactly like a model's.

What survives is the tail. 9.4 percent of human same sign runs are longer than
two events and the longest reach 49. So human turning is alternating noise plus
a rare heavy tail of organised runs, and it is the tail that produces real
displacement from the straight line. That is consistent with everything else
measured: at matched distance AND matched duration the model turns MORE than a
person (angular spread 44.9 against 34.4, curvature spread 3.36 against 1.04 in
the slowest short band), yet its turning does not accumulate into deviation
(rank correlation with max_deviation 0.004 against a human 0.172) and does not
arrive with acceleration bursts (0.093 against 0.248). More turning, less
achieved by it.

This measures the tail on all three arms:

  acf             signed heading change autocorrelation, lags 1 to 6. Sets the
                  alternating baseline each arm sits on.
  run lengths     the same sign run distribution out to p99 and the max, not
                  just the mean, because the mean is dominated by the jitter
                  and the mean is the part that already matches.
  organised share the fraction of TOTAL absolute heading change delivered by
                  runs longer than two events. This is the money number. Two
                  arms can have identical total turning and differ completely
                  here, and that is precisely the shape of the defect.
  net per run     the heading actually achieved by a run, p90 and max. A run of
                  eight tiny alternating steps achieves nothing; a real
                  correction achieves most of a reversal.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_turnruns.py --masked \
        --ar event_ar_v1.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

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
    S_PAD_CLASS, TH_NULL_CLASS, TICK_CLASS, class_to_dtheta,
    dth_lattice_to_class, s2_to_class,
)
from phase_a_baseline import make_specs  # noqa: E402

LONG = 2


def _measure(label, streams, out):
    acc = [[] for _ in range(6)]
    runs, nets, organised, total = [], [], 0.0, 0.0
    for s_cls, th_cls, n in streams:
        if n < 12:
            continue
        mo = np.flatnonzero(s_cls[:n] > TICK_CLASS)
        if len(mo) < 20:
            continue
        t = class_to_dtheta(torch.from_numpy(th_cls[mo].astype(np.int64))).numpy()
        if t.std() < 1e-9:
            continue
        z = (t - t.mean()) / t.std()
        for k in range(1, 7):
            acc[k - 1].append(z[:-k] * z[k:])
        nz = np.flatnonzero(t != 0.0)
        if len(nz) < 2:
            continue
        tv, sg = t[nz], np.sign(t[nz])
        b = np.flatnonzero(np.diff(sg) != 0)
        starts = np.concatenate([[0], b + 1])
        ends = np.concatenate([b + 1, [len(sg)]])
        for a, e in zip(starts, ends):
            L = e - a
            s = float(np.abs(tv[a:e].sum()))
            runs.append(L)
            total += s
            if L > LONG:
                organised += s
                nets.append(s)
    R = np.asarray(runs, dtype=np.float64)
    N = np.asarray(nets, dtype=np.float64) if nets else np.zeros(1)
    rec = dict(n_runs=int(len(R)),
               acf=[float(np.concatenate(a).mean()) for a in acc],
               run_mean=float(R.mean()), run_p50=float(np.median(R)),
               run_p90=float(np.percentile(R, 90)),
               run_p99=float(np.percentile(R, 99)), run_max=float(R.max()),
               share_long=float((R > LONG).mean()),
               organised_share=float(organised / max(total, 1e-9)),
               net_p90=float(np.percentile(N, 90)), net_max=float(N.max()))
    out[label] = rec
    print(f"  {label:<20}{rec['acf'][0]:>8.3f}{rec['run_mean']:>8.3f}"
          f"{rec['run_p90']:>7.0f}{rec['run_p99']:>7.0f}{rec['run_max']:>7.0f}"
          f"{rec['share_long']:>9.4f}{rec['organised_share']:>10.4f}"
          f"{rec['net_p90']:>9.3f}{rec['net_max']:>9.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--ar", default="")
    ap.add_argument("--masked", action="store_true")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--out", default="research/w4_turnruns.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    out = {}
    print(f"  {'arm':<20}{'acf1':>8}{'runMu':>8}{'p90':>7}{'p99':>7}"
          f"{'max':>7}{'shrLong':>9}{'organis':>10}{'netP90':>9}{'netMax':>9}")

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
    _measure("human", ((hs[i], hth[i], int(L[i]))
                       for i in range(min(args.n, len(L)))), out)

    specs = make_specs(args.n, args.seed)
    rows = []
    for sx, sy, ex, ey in specs:
        dd = math.hypot(ex - sx, ey - sy)
        if dd < 1e-6:
            continue
        ang = math.atan2(ey - sy, ex - sx)
        rows.append([math.log(dd), math.log(esp._duration.sample(math.log(dd))),
                     math.cos(ang), math.sin(ang)])

    def collect(fn, label):
        S, T, N = [], [], []
        for c0 in range(0, len(rows), args.batch):
            cond = torch.tensor(rows[c0:c0 + args.batch], dtype=torch.float32,
                                device=dev)
            s_cls, th_cls = fn(cond)
            s_np, th_np = s_cls.cpu().numpy(), th_cls.cpu().numpy()
            pad = s_np >= S_PAD_CLASS
            S.append(s_np)
            T.append(th_np)
            N.append(np.where(pad.any(1), pad.argmax(1), s_np.shape[1]))
        s_np, th_np, n = np.concatenate(S), np.concatenate(T), np.concatenate(N)
        _measure(label, ((s_np[i], th_np[i], int(n[i])) for i in range(len(n))),
                 out)

    if args.masked:
        m, seq_len = esp._model, esp._cfg["max_seq_len"]

        def masked_fn(cond):
            with torch.no_grad():
                _, s_cls, th_cls = m.sample(
                    cond, seq_len, n_steps=100, temperature=args.temp,
                    order="gumbel", choice_temp=10.0,
                    feat=torch.zeros(cond.shape[0], esp._FEAT_BANK.shape[1],
                                     device=dev) if esp._FEAT_BANK is not None else None)
            return s_cls, th_cls
        collect(masked_fn, "masked served")

    for ck_name in [c for c in args.ar.split(",") if c]:
        ck = torch.load(f"training/{ck_name}", map_location=dev,
                        weights_only=False)
        model = EventARModel(**ck["config"]).to(dev).eval()
        model.load_state_dict(ck["model_state_dict"])

        def ar_fn(cond, _m=model):
            a, b, _c = _m.sample(cond, temperature=args.temp)
            return a, b
        collect(ar_fn, f"ar {ck_name.replace('.pt', '')}")

    json.dump(out, open(args.out, "w"), indent=2)
    print("\n  organis is the fraction of all absolute heading change carried")
    print("  by same sign runs longer than two events. Two arms can turn the")
    print("  same total amount and differ entirely here, and that is the")
    print("  shape every other measurement today has pointed at.")


if __name__ == "__main__":
    main()
