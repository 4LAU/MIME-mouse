"""Ask the critic what it is looking at, by taking things away from it.

PRE REGISTERED in HANDOFF.md 2026-08-10, "## What the critic is reading".
Variants, controls, thresholds and the prediction were fixed before the run.

`w4_critic` put a learned critic at 0.7509 against a 0.5235 floor, where every
hand written statistic in `w4_views` together reached 0.6201 against 0.5095. The
critic sees roughly twice the excess. It therefore knows something this file
does not, and the cheapest way to find out what is to blind it and retrain.

This is the question `w4_channels` was built for and could not answer. That arm
spliced human and generated streams together and every hybrid scored worse than
both parents, because the splice made objects that could not exist. Nothing is
spliced here. Each critic sees real sequences from both sides and is simply
denied a stream, which is a restriction on the observer rather than a change to
the data.

  only_x      the critic sees stream x and nothing else. How much that stream
              carries by itself.
  blind_x     the critic sees everything except stream x. What is lost when it
              is removed, which is what x carries UNIQUELY.
  first32     every stream, but only the first thirty two events of each
              trajectory, and every row padded to exactly that. Length is a cue
              and this removes it, while asking whether the tell is in the way a
              movement starts.
  floor       the A half of the human pool against the B half, full streams.

A blinded stream is replaced by a constant, not deleted, so every variant sees
the same sequence length and the same padding. Everything else, the
architecture, the optimiser, the epochs, the rows and the held out split, is
identical to `w4_critic` and to every other variant.

Safety. Reads the training corpus, one checkpoint and the cached generation
written by `w4_critic`. Writes no model. Touches no evaluation data and no
scoring code, never `training/candi_polar_flow_best.pt`. Paces on GPU
temperature through `w4_latent`.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python -u research/w4_critic_ablate.py --cache PATH
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("EVENT_SNAP", "2.5")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
from models.event_stream_polar import S_PAD_CLASS, TH_NULL_CLASS  # noqa: E402
from w4_critic import human_rows, pack, train_critic  # noqa: E402
from w4_latent import gpu_c  # noqa: E402

DATA = Path("training")
FULL_REF = 0.7509
FLOOR_REF = 0.5235

# A stream is SUFFICIENT if a critic given only that stream comes within this of
# the full critic, and NECESSARY if blinding it costs the full critic more than
# this. Both are set at roughly four times the spread between the two
# independent runs of w4_critic, which read 0.7518 and 0.7509.
MARGIN = 0.030
NEED = 0.080


def blind(X, which):
    """Replace a stream with a constant. The mask is untouched, so every variant
    sees the same lengths and the same padding, and only the content goes."""
    s, th, dt, m = (x.copy() for x in X)
    if "speed" in which:
        s[:] = 0
    if "turn" in which:
        th[:] = 0
    if "time" in which:
        dt[:] = 0
    return s, th, dt, m


def head(X, T):
    """The first T events of every row, all rows exactly that long."""
    s, th, dt, m = X
    keep = m[:, T - 1]
    return (s[keep, :T], th[keep, :T], dt[keep, :T],
            np.ones((int(keep.sum()), T), dtype=bool)), keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--min-len", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--head-len", type=int, default=32)
    ap.add_argument("--out", default="research/w4_critic_ablate.json")
    args = ap.parse_args()

    arr = {k: np.load(DATA / f"events_{k}.npy", mmap_mode="r")
           for k in ("s2", "dth", "dt", "len", "cond")}
    ok = np.load(DATA / "events_feat18_ok.npy")

    # Reconstructed exactly as w4_critic built it, same seed, same pool, same
    # A and B assignment, so the cached generation lines up row for row.
    rng = np.random.default_rng(args.seed)
    pool = np.sort(rng.choice(np.flatnonzero(ok), 2 * args.n, replace=False))
    assign = rng.permutation(2 * args.n)
    ai, bi = np.sort(assign[:args.n]), np.sort(assign[args.n:])
    cond = np.asarray(arr["cond"][pool], dtype=np.float32)

    z = np.load(args.cache)
    gl, gs, gt, gd = z["len"], z["s"], z["th"], z["dt"]
    gen = [None if gl[j] < args.min_len
           else (gs[j, :gl[j]].astype(np.int64), gt[j, :gl[j]].astype(np.int64),
                 gd[j, :gl[j]].astype(np.int64)) for j in range(len(gl))]

    humA = human_rows(arr, pool[ai], args.min_len)
    humB = human_rows(arr, pool[bi], args.min_len)
    ka = [j for j in range(args.n) if humA[j] is not None
          and gen[j] is not None]
    kb = [j for j in range(args.n) if humB[j] is not None]
    m = min(len(ka), len(kb))
    ka, kb = ka[:m], kb[:m]

    T = 256
    HA = pack([humA[j] for j in ka], T)
    G = pack([gen[j] for j in ka], T)
    HB = pack([humB[j] for j in kb], T)
    CA, CB = cond[ai][ka], cond[bi][kb]
    dev = esp._DEVICE
    print(f"\n  {m} rows per class, reference full critic {FULL_REF}\n")

    runs = [("full", ()), ("only_speed", ("turn", "time")),
            ("only_turn", ("speed", "time")), ("only_time", ("speed", "turn")),
            ("blind_speed", ("speed",)), ("blind_turn", ("turn",)),
            ("blind_time", ("time",))]
    res = {}
    for name, which in runs:
        print(f"  {name}")
        res[name] = train_critic(blind(HA, which), blind(G, which), CA, CA,
                                 dev, args.epochs, args.bs, args.lr,
                                 args.seed + 2, name)

    # A row survives only if BOTH sides reach the head length, so the two
    # classes keep the same rows and the pairing is not quietly broken.
    (HAh, kh), (Gh, kg) = head(HA, args.head_len), head(G, args.head_len)
    both = kh & kg
    HAh = tuple(x[both[kh]] for x in HAh)
    Gh = tuple(x[both[kg]] for x in Gh)
    print(f"  first{args.head_len}, {int(both.sum())} rows per class")
    res[f"first{args.head_len}"] = train_critic(
        HAh, Gh, CA[both], CA[both], dev, args.epochs, args.bs, args.lr,
        args.seed + 2, f"first{args.head_len}")

    print("  floor")
    res["floor"] = train_critic(HA, HB, CA, CB, dev, args.epochs, args.bs,
                                args.lr, args.seed + 3, "floor")

    full, floor = res["full"], res["floor"]
    print(f"\n  {'variant':>14}{'AUC':>9}{'vs full':>10}")
    for k in res:
        print(f"  {k:>14}{res[k]:>9.4f}{res[k] - full:>10.4f}")

    sufficient = [c for c in ("speed", "turn", "time")
                  if full - res[f"only_{c}"] <= MARGIN]
    necessary = [c for c in ("speed", "turn", "time")
                 if full - res[f"blind_{c}"] >= NEED]
    alone = sum(res[f"only_{c}"] - floor for c in ("speed", "turn", "time"))

    if res["floor"] > 0.60:
        verdict = (f"VOID. The floor reads {floor:.4f} and nothing may be "
                   f"attributed to the model.")
    elif sufficient:
        verdict = (f"LOCALISED in {', '.join(sufficient)}. A critic given that "
                   f"stream alone comes within {MARGIN} of the full one, so "
                   f"the tell is carried there and an arm may target it.")
    elif not necessary:
        verdict = (f"JOINT. No stream alone comes within {MARGIN} of the full "
                   f"critic and blinding no stream costs more than {NEED}, so "
                   f"no stream is sufficient and none is necessary. The single "
                   f"stream excesses sum to {alone:.4f} against a full excess "
                   f"of {full - floor:.4f}. The information is redundant across "
                   f"streams and lives in their joint structure, which is the "
                   f"same answer w4_views gave for statistics and w4_channels "
                   f"gave for free when every hybrid lost to both parents.")
    else:
        verdict = (f"NECESSARY, {', '.join(necessary)}. No stream is "
                   f"sufficient but blinding those costs more than {NEED}, so "
                   f"they carry something no other stream duplicates.")
    print(f"\n  VERDICT  {verdict}\n")

    json.dump({"n_per_class": m, "seed": args.seed, "runs": res,
               "full_ref": FULL_REF, "floor_ref": FLOOR_REF,
               "alone_sum": alone, "verdict": verdict, "gpu_c": gpu_c()},
              open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}\n")


if __name__ == "__main__":
    main()
