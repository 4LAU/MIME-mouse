"""Does the AR model KNOW about pauses, or has it merely stopped emitting them?

`w4_tickstruct` found `event_ar_v1` emits 6.21 percent still events against a
human 9.48 percent, with moving runs of p90 37 against a human 21. `w4_submove`
found the human pauses that matter, the ones lasting at least 40ms, are re-aims:
the heading changes 7.5 to 21 times more across one than across speed matched
uninterrupted motion, and 94 percent of paths overshoot and come back.

A shortfall in the SAMPLED share has two very different causes and they need
different fixes:

  learning     the model never learned when a person pauses, so its predicted
               probability of a still token is low even standing on a real human
               prefix at the exact moment that person paused
  drift        the model learned it, but sampling walks the state channel away
               from anything it saw in training, so the sampled share falls even
               though the conditional is right

Teacher forcing separates them. Run real human prefixes through the model and
read off the probability it assigns to a still token. That is the conditional
with no sampling in the loop, so drift cannot touch it.

Three readings:

  overall      mean predicted P(still) against the empirical still rate. Equal
               means calibrated in the marginal.
  at onset     mean predicted P(still) at the first event of a real pause,
               against the same at ordinary moving positions. A model that
               learned the trigger separates these; one that did not will not.
  by remaining the same conditional bucketed by how much of the straight line
               distance is still to go, which is the channel `prefix_state`
               feeds the model exactly. If the human rate rises somewhere and
               the model's prediction stays flat, that is the missed structure.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_stillcal.py --ckpt event_ar_v1.pt
"""
from __future__ import annotations

import argparse
import json
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

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
from models.event_ar import (  # noqa: E402
    DT_MAX_MS, EventARModel, prefix_state,
)
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TH_NULL_CLASS, TICK_CLASS, class_to_dtheta, class_to_speed,
    dth_lattice_to_class, s2_to_class,
)

PAUSE_MS = 40.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v1.pt")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="research/w4_stillcal.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    T = ck["config"]["max_seq_len"]
    print(f"  {args.ckpt} step {ck.get('step')}, teacher forced on human "
          f"prefixes, no sampling\n", flush=True)

    rng = np.random.default_rng(args.seed)
    lengths = np.load("training/events_len.npy")
    pick = np.sort(rng.choice(len(lengths), args.n, replace=False))
    s2 = np.load("training/events_s2.npy", mmap_mode="r")[pick]
    dth = np.load("training/events_dth.npy", mmap_mode="r")[pick]
    dtm = np.load("training/events_dt.npy", mmap_mode="r")[pick].astype(np.float64)
    conds = np.load("training/events_cond.npy")[pick]
    L = np.minimum(lengths[pick], T)

    p_all, y_all, onset_all, rem_all, valid = [], [], [], [], []

    for c0 in range(0, len(L), args.batch):
        sl = slice(c0, min(c0 + args.batch, len(L)))
        b = sl.stop - sl.start
        s_np = s2[sl].astype(np.int64)
        d_np = dth[sl].astype(np.int64)
        t_np = dtm[sl]

        s_cls = np.full((b, T), S_PAD_CLASS, dtype=np.int64)
        th_cls = np.full((b, T), TH_NULL_CLASS, dtype=np.int64)
        dt_cls = np.zeros((b, T), dtype=np.int64)
        for j in range(b):
            n = int(L[sl][j])
            s_cls[j, :n] = s2_to_class(torch.from_numpy(s_np[j, :n])).numpy()
            th_cls[j, :n] = np.where(
                s_np[j, :n] > 0,
                dth_lattice_to_class(torch.from_numpy(d_np[j, :n])).numpy(),
                TH_NULL_CLASS)
            dt_cls[j, :n] = np.clip(np.round(t_np[j, :n]), 0, DT_MAX_MS)

        sc = torch.from_numpy(s_cls).to(dev)
        tc = torch.from_numpy(th_cls).to(dev)
        dc = torch.from_numpy(dt_cls).to(dev)
        cond = torch.from_numpy(conds[sl].astype(np.float32)).to(dev)

        with torch.no_grad():
            sp, tp, dp = model.shift_inputs(sc, tc, dc)
            state = prefix_state(sc, tc, dc, cond)
            s_logits, _, _ = model(sp, tp, dp, state, cond, sc, tc, dc)
            p_still = torch.softmax(s_logits.float(), -1)[..., TICK_CLASS]
        p = p_still.cpu().numpy()

        for j in range(b):
            n = int(L[sl][j])
            if n < 12:
                continue
            cls = s_cls[j, :n]
            mo = cls > TICK_CLASS
            # geometry for remaining distance, same construction as _decode
            sspd = class_to_speed(torch.from_numpy(cls)).numpy()
            ddth = class_to_dtheta(torch.from_numpy(th_cls[j, :n])).numpy()
            hd = np.cumsum(np.where(mo, ddth, 0.0))
            dx = np.where(mo, sspd * np.cos(hd), 0.0)
            dy = np.where(mo, sspd * np.sin(hd), 0.0)
            x = np.concatenate([[0.0], np.cumsum(dx)])
            y = np.concatenate([[0.0], np.cumsum(dy)])
            D = float(np.hypot(x[-1], y[-1]))
            if D < 20:
                continue
            rem = np.hypot(x[:n] - x[-1], y[:n] - y[-1]) / D

            onset = np.zeros(n, dtype=bool)
            k = 0
            while k < n:
                if not mo[k]:
                    m = k
                    while m < n and not mo[m]:
                        m += 1
                    if t_np[j, k:m].sum() >= PAUSE_MS:
                        onset[k] = True
                    k = m
                else:
                    k += 1

            p_all.append(p[j, :n])
            y_all.append(~mo)
            onset_all.append(onset)
            rem_all.append(rem)
            valid.append(1)

    P = np.concatenate(p_all)
    Y = np.concatenate(y_all)
    O = np.concatenate(onset_all)
    R = np.concatenate(rem_all)
    moving = ~Y

    out = dict(
        n_traj=int(len(valid)), n_events=int(len(P)),
        empirical_still_rate=float(Y.mean()),
        predicted_still_rate=float(P.mean()),
        pred_at_pause_onset=float(P[O].mean()),
        pred_at_moving=float(P[moving].mean()),
        pred_at_any_still=float(P[Y].mean()),
        n_onsets=int(O.sum()),
    )
    print(f"  events {len(P):,} over {len(valid):,} trajectories, "
          f"{int(O.sum()):,} real pause onsets\n")
    print(f"  empirical still rate           {out['empirical_still_rate']:.4f}")
    print(f"  mean predicted P(still)        {out['predicted_still_rate']:.4f}")
    print(f"  predicted at a moving event    {out['pred_at_moving']:.4f}")
    print(f"  predicted at any still event   {out['pred_at_any_still']:.4f}")
    print(f"  predicted at a REAL pause onset{out['pred_at_pause_onset']:>10.4f}")
    lift = out['pred_at_pause_onset'] / max(out['pred_at_moving'], 1e-9)
    out["onset_lift"] = float(lift)
    print(f"  lift at onset over moving      {lift:>10.2f}x\n")

    edges = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1e9]
    print(f"  {'remaining':>12}{'n':>10}{'empirical':>11}{'predicted':>11}"
          f"{'onsetRate':>11}")
    buckets = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (R >= a) & (R < b)
        if m.sum() < 100:
            continue
        row = dict(lo=a, hi=b, n=int(m.sum()), emp=float(Y[m].mean()),
                   pred=float(P[m].mean()), onset=float(O[m].mean()))
        buckets.append(row)
        print(f"  {f'{a:.2f}-{b:.2f}':>12}{row['n']:>10}{row['emp']:>11.4f}"
              f"{row['pred']:>11.4f}{row['onset']:>11.5f}")
    out["by_remaining"] = buckets

    json.dump(out, open(args.out, "w"), indent=2)
    print("\n  teacher forced, so sampling drift cannot affect any number here.")
    print("  if predicted tracks empirical the model learned it and the")
    print("  shortfall is drift; if it is flat the model never learned it.")


if __name__ == "__main__":
    main()
