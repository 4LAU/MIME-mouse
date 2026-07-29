"""Is the AR model starving or saturated? Held out likelihood says which.

Every measurement in this programme now points the same way. `w4_dvjoint` was
the fourth hypothesis to die and the ninth channel to price small: the defect is
distributed, not located, and no low dimensional summary holds it. What moves a
distributed error is capacity and data on the plain objective. Before spending
hours on that, this establishes WHICH of the two, because they need opposite
responses and the answer was never recorded.

`train_event_ar.py` keeps no validation split and saves only a training loss
EMA, so the checkpoint cannot answer this. But its training subset is drawn with
a hardcoded `default_rng(123)` over the full corpus, so the split is exactly
reproducible after the fact. 1,500,000 of 4,028,855 trajectories were trained
on for 3.41 epochs and 2,528,855 were never seen once.

Both sides are scored with the trainer's own loss, in eval mode so dropout is
off, on identically sized samples.

  train close to held out   the model has not memorised anything and is limited
                            by capacity or optimisation. More parameters and
                            more steps are the lever, and the 2.5M unused
                            trajectories are not the constraint.
  train well below held out the model is fitting its subset. More data is the
                            lever and more capacity would make it worse.

The three streams are reported separately, because they can disagree and the
one that is saturated is not necessarily the one that matters. `th` is the
heading stream, which is where every structural measurement today has pointed.

This is a measurement, not a training signal, and it touches no checkpoint.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_arfit.py --ckpt event_ar_v1.pt
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

for p in (".", "research", "research/autoloop", "training"):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.event_ar import EventARModel, N_DT_CLASSES, prefix_state  # noqa: E402
from models.event_stream_polar import (  # noqa: E402
    N_S_CLASSES, N_TH_CLASSES, S_PAD_CLASS,
)
from train_event_ar import ARDataset  # noqa: E402

TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000


@torch.no_grad()
def evaluate(model, ds, device, batch, amp):
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4,
                    pin_memory=True, drop_last=False)
    tot = {k: 0.0 for k in ("s", "th", "dt")}
    cnt = {k: 0.0 for k in ("s", "th", "dt")}
    for bi, b in enumerate(dl):
        s_cls, th_cls, dt_cls, n_sup, cond = (x.to(device, non_blocking=True)
                                              for x in b)
        B, T = s_cls.shape
        s_prev, th_prev, dt_prev = model.shift_inputs(s_cls, th_cls, dt_cls)
        state = prefix_state(s_cls, th_cls, dt_cls, cond)
        with torch.amp.autocast("cuda", enabled=amp):
            sl, tl, dl_ = model(s_prev, th_prev, dt_prev, state, cond,
                                s_cls, th_cls, dt_cls)
        ar = torch.arange(T, device=device).unsqueeze(0)
        sup = (ar < n_sup.unsqueeze(1)).float()
        motion = (s_cls > 0) & (s_cls < S_PAD_CLASS)
        sup_th = sup * motion.float()
        for key, lg, tg, m, V in (
                ("s", sl, s_cls, sup, N_S_CLASSES),
                ("th", tl, th_cls, sup_th, N_TH_CLASSES),
                ("dt", dl_, dt_cls, sup, N_DT_CLASSES)):
            ce = F.cross_entropy(lg.reshape(-1, V).float(), tg.reshape(-1),
                                 reduction="none").view(B, T)
            tot[key] += float((ce * m).sum())
            cnt[key] += float(m.sum())
        if bi % 20 == 0:
            print(f"    batch {bi}", flush=True)
    return {k: tot[k] / max(cnt[k], 1.0) for k in tot}, \
           {k: int(cnt[k]) for k in cnt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v1.pt")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--n-train", type=int, default=N_TRAIN_DEFAULT)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="research/w4_arfit.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s2 = np.load("training/events_s2.npy", mmap_mode="r")
    dth = np.load("training/events_dth.npy", mmap_mode="r")
    dt = np.load("training/events_dt.npy", mmap_mode="r")
    lengths = np.load("training/events_len.npy")
    cond = np.load("training/events_cond.npy")
    N = len(lengths)

    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(N, min(N, args.n_train), replace=False))
    held = np.setdiff1d(np.arange(N), trained)
    rng = np.random.default_rng(args.seed)
    pick = {
        "trained on": np.sort(rng.choice(trained, min(args.n, len(trained)),
                                         replace=False)),
        "held out": np.sort(rng.choice(held, min(args.n, len(held)),
                                       replace=False)),
    }
    print(f"  corpus {N:,}, trained on {len(trained):,}, never seen "
          f"{len(held):,}, scoring {args.n:,} from each\n", flush=True)

    ck = torch.load(f"training/{args.ckpt}", map_location=device,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(device).eval()
    model.load_state_dict(ck["model_state_dict"])
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  {args.ckpt}: {n_par/1e6:.2f}M params, step {ck.get('step')}, "
          f"train loss ema {ck.get('loss_ema'):.4f}\n", flush=True)

    out = {"ckpt": args.ckpt, "params": int(n_par), "step": ck.get("step"),
           "train_loss_ema": ck.get("loss_ema"), "n_scored": args.n,
           "n_trained_on": int(len(trained)), "n_never_seen": int(len(held))}
    res = {}
    for label, idx in pick.items():
        ds = ARDataset(s2[idx], dth[idx], dt[idx], lengths[idx], cond[idx],
                       ck["config"]["max_seq_len"])
        m, c = evaluate(model, ds, device, args.batch, device.type == "cuda")
        m["total"] = m["s"] + m["th"] + m["dt"]
        res[label] = dict(loss=m, tokens=c)
        print(f"  {label:<12} s {m['s']:.4f}  th {m['th']:.4f}  "
              f"dt {m['dt']:.4f}  total {m['total']:.4f}", flush=True)

    a, b = res["trained on"]["loss"], res["held out"]["loss"]
    gap = {k: b[k] - a[k] for k in a}
    out["arms"] = res
    out["generalisation_gap"] = gap
    print(f"\n  {'gap held out minus trained on':<32}"
          f"s {gap['s']:+.4f}  th {gap['th']:+.4f}  dt {gap['dt']:+.4f}  "
          f"total {gap['total']:+.4f}")

    json.dump(out, open(args.out, "w"), indent=2)
    print("\n  a gap near zero means nothing was memorised and the lever is")
    print("  capacity and steps, not the 2.5M unused trajectories. a clearly")
    print("  positive gap means the opposite and more parameters would hurt.")


if __name__ == "__main__":
    main()
