"""Which states does each model actually visit?

`w4_stillcal` teacher forced `event_ar_v1` on real human prefixes and found its
conditional is right: it predicts a still token at 0.0968 against an empirical
0.0979, tracks the human rate to three decimals in every remaining distance
bucket, and puts 6.32 times more probability on a still at a real pause onset
than at an ordinary moving event. The model knows when a person pauses.

`w4_submove` found it produces 0.05 real pauses per trajectory against a human
0.23. A right conditional and a wrong sample rate can only be reconciled one
way: the model visits a different distribution of states than a person does, so
the conditional it gets right is being asked the wrong questions.

`w4_stillcal` also says which states matter. The human still rate is U shaped in
how far the pointer still is from where it ends up, 0.1402 within 10 percent,
falling to 0.0729 in the middle, and rising again to 0.1082 once the pointer is
FARTHER from the target than the whole movement was long. Real pause onsets
concentrate at exactly those two extremes, 0.00907 and 0.00913 against 0.00203
in the middle. The second one is a person who has gone badly wrong.

This measures how much time each arm spends in each of those states. Everything
is relative to each trajectory's OWN endpoint, which is what the human number
is relative to, so a model that misses its commanded target is not penalised
here for that; only the shape of its approach is measured.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_statevisit.py --masked \
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
    S_PAD_CLASS, TH_NULL_CLASS, TICK_CLASS, class_to_dtheta, class_to_speed,
    dth_lattice_to_class, s2_to_class,
)
from phase_a_baseline import make_specs  # noqa: E402

EDGES = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70, 1.00, 1e9]


def _rem(s_cls, th_cls, n):
    """Distance from each event to the trajectory's own endpoint, over the
    straight line length of the whole movement."""
    s = class_to_speed(torch.from_numpy(s_cls[:n].astype(np.int64))).numpy()
    d = class_to_dtheta(torch.from_numpy(th_cls[:n].astype(np.int64))).numpy()
    mo = s_cls[:n] > TICK_CLASS
    hd = np.cumsum(np.where(mo, d, 0.0))
    x = np.concatenate([[0.0], np.cumsum(np.where(mo, s * np.cos(hd), 0.0))])
    y = np.concatenate([[0.0], np.cumsum(np.where(mo, s * np.sin(hd), 0.0))])
    D = float(np.hypot(x[-1], y[-1]))
    if D < 20:
        return None
    return np.hypot(x[:n] - x[-1], y[:n] - y[-1]) / D


def _measure(label, streams, out):
    R, mx = [], []
    for s_cls, th_cls, n in streams:
        if n < 12:
            continue
        r = _rem(s_cls, th_cls, n)
        if r is None:
            continue
        R.append(r)
        mx.append(float(r.max()))
    R = np.concatenate(R)
    shares = []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        shares.append(float(((R >= a) & (R < b)).mean()))
    rec = dict(n=len(mx), n_events=int(len(R)), shares=shares,
               share_over_1=shares[-1], max_rem_p50=float(np.median(mx)),
               max_rem_p90=float(np.percentile(mx, 90)))
    out[label] = rec
    print(f"  {label:<20}" + "".join(f"{v:>9.4f}" for v in shares)
          + f"{rec['max_rem_p50']:>9.3f}{rec['max_rem_p90']:>9.3f}"
          + f"{len(mx):>7}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--ar", default="")
    ap.add_argument("--masked", action="store_true")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--out", default="research/w4_statevisit.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    out = {}
    hdr = "".join(f"{f'{a:.2f}-{b:.2f}' if b < 1e8 else '>1.00':>9}"
                  for a, b in zip(EDGES[:-1], EDGES[1:]))
    print(f"  {'arm':<20}{hdr}{'maxP50':>9}{'maxP90':>9}{'n':>7}")

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
    print("\n  columns are the share of events at that distance from the")
    print("  trajectory's own endpoint, over the movement's straight line")
    print("  length. The >1.00 column is the badly wrong state where human")
    print("  pauses concentrate. maxP50 is the median worst excursion.")


if __name__ == "__main__":
    main()
