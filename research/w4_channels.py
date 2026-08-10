"""Which of the three token channels carries the 0.129 gap.

PRE REGISTERED in HANDOFF.md 2026-08-09, "## Attributing the gap to a token
channel". Arms, controls, branch thresholds and the prediction were all fixed
before this file was run.

The setting. `w4_token_ceiling` and the 2026-08-09 ceiling both put real human
tokens, decoded through the serving decoder, at about 0.512 to 0.516. The model
as served reads about 0.644. The representation is therefore not the limit and
the whole of the 0.129 belongs to the sequences the model emits. Feature
conditioning has just failed to close it, and its obedience table says the
failure lives in fine motion texture rather than in the shape of the path.

This asks a narrower question than any arm so far. The decoder consumes three
channels, a speed class, a turn class and a millisecond timing class. For a
matched pair of sequences, one human and one generated on that human's own
command, every one of the eight ways of choosing each channel from either
source can be decoded and scored. Two of the eight are the ceiling and the
served number. The other six partition the gap between the channels.

Nothing is trained. Nothing is selected. One trajectory per command.

The consistency repair. A speed class and a turn class are not independent in
this vocabulary: a step of zero length carries `TH_NULL_CLASS` and a moving step
carries a lattice class. A mixed arm can therefore pair a moving speed with a
null turn, which is not a sequence either source could produce. Every arm,
including the two pure ones, is passed through the same repair: a moving step
whose turn is null takes the zero turn class, and a still step takes null. On
the pure arms the repair is the identity and that is asserted, not assumed.

Safety. Scores through `research/autoloop/scoring.py` only. Reads the training
corpus and one checkpoint, writes neither. Never touches evaluation data or
scoring code, never `training/candi_polar_flow_best.pt`. Paces on GPU
temperature through `w4_latent`.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python -u research/w4_channels.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("EVENT_SNAP", "2.5")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
import scoring  # noqa: E402
from features import extract_features, resample_trajectory  # noqa: E402
from models.event_ar import (DT_MAX_MS, EventARModel, class_to_dt_ms)  # noqa: E402
from models.event_stream_polar import (S_PAD_CLASS, TH_NULL_CLASS,  # noqa: E402
                                       dth_lattice_to_class, s2_to_class)
from w4_latent import cooldown, gpu_c, throttle  # noqa: E402

DATA = Path("training")
CHANNELS = ("speed", "turn", "time")

# The ceiling this file's all human arm has to reproduce, from w4_token_ceiling
# on 2026-07-28 and again from an independent path on 2026-08-09. If truncation
# to the common length contaminates the comparison it will show up here first.
CEILING_REF = 0.5163
CEILING_TOL = 0.045

ZERO_MOVE = int(s2_to_class(torch.tensor([0], dtype=torch.long))[0])
ZERO_TURN = int(dth_lattice_to_class(torch.tensor([0], dtype=torch.long))[0])


def repair(s_cls, th_cls):
    """Make a mixed pair of channels a sequence the vocabulary allows.

    A moving step must carry a lattice turn and a still step must carry the
    null. Mixing channels breaks that pairing; this restores it without
    touching either channel's own content anywhere the pairing already holds.
    """
    moving = s_cls != ZERO_MOVE
    out = np.where(moving,
                   np.where(th_cls == TH_NULL_CLASS, ZERO_TURN, th_cls),
                   TH_NULL_CLASS)
    return out.astype(np.int64)


def human_tokens(arr, i):
    """One corpus row in the decoder's own class space. Mirrors
    `training/prepare_event_features.decode_row` exactly."""
    L = int(arr["len"][i])
    if L < 2:
        return None
    s2 = torch.from_numpy(arr["s2"][i, :L].astype(np.int64))
    dth = torch.from_numpy(arr["dth"][i, :L].astype(np.int64))
    s_cls = s2_to_class(s2).numpy()
    th_cls = torch.where(s2 > 0, dth_lattice_to_class(dth),
                         torch.full_like(dth, TH_NULL_CLASS)).numpy()
    dt_cls = torch.round(torch.from_numpy(arr["dt"][i, :L].astype(np.float32))
                         ).long().clamp(0, DT_MAX_MS).numpy()
    return s_cls, th_cls, dt_cls


def generate(model, cond, batch, temp, dev, seed):
    """One trajectory per command, in corpus order. No selection."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    out = []
    with torch.no_grad():
        for c0 in range(0, len(cond), batch):
            throttle()
            blk = torch.tensor(cond[c0:c0 + batch], dtype=torch.float32,
                               device=dev)
            s_cls, th_cls, dt_cls = model.sample(blk, temperature=temp)
            s_np = s_cls.cpu().numpy()
            th_np = th_cls.cpu().numpy()
            dt_np = dt_cls.cpu().numpy()
            pad = s_np >= S_PAD_CLASS
            for j in range(s_np.shape[0]):
                L = int(pad[j].argmax()) if pad[j].any() else s_np.shape[1]
                out.append((s_np[j, :L], th_np[j, :L], dt_np[j, :L]))
            del g
            g = torch.Generator(device="cpu").manual_seed(seed + c0)
    return out


def decode(s_cls, th_cls, dt_cls, angle):
    dt_ms = class_to_dt_ms(torch.from_numpy(dt_cls.astype(np.int64)))
    dt_z = ((torch.log(dt_ms.clamp(min=0.05)) - esp._DT_MEAN)
            / esp._DT_STD).numpy()
    return esp._decode(dt_z, s_cls, th_cls, 0.0, 0.0, angle)


def featurise(s_cls, th_cls, dt_cls, angle):
    p = decode(s_cls, th_cls, dt_cls, angle)
    if p is None or len(p) < 2:
        return None
    f = extract_features(resample_trajectory(np.asarray(p, dtype=np.float64),
                                             hz=125.0))
    if f is None or not np.all(np.isfinite(f)):
        return None
    return np.asarray(f, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="training/event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--min-len", type=int, default=8)
    ap.add_argument("--out", default="research/w4_channels.json")
    args = ap.parse_args()

    arr = {k: np.load(DATA / f"events_{k}.npy", mmap_mode="r")
           for k in ("s2", "dth", "dt", "len", "cond")}
    ok = np.load(DATA / "events_feat18_ok.npy")

    # Random rows, never a prefix. The corpus is ordered by session and any
    # prefix of it is a measurement of one person's mouse, which HANDOFF
    # records as having curvature medians four orders of magnitude off.
    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(np.flatnonzero(ok), args.n, replace=False))
    cond = np.asarray(arr["cond"][rows], dtype=np.float32)
    ang = np.arctan2(cond[:, 3], cond[:, 2])

    dev = esp._DEVICE
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])

    print(f"\n  {len(rows)} random corpus rows, one generated trajectory each")
    print(f"  ckpt {args.ckpt} at step {ck.get('step', '?')}")
    print(f"  all human arm must land within {CEILING_TOL} of {CEILING_REF}\n")

    cooldown()
    gen = generate(model, cond, args.batch, args.temp, dev, args.seed + 1)

    arms = list(product((0, 1), repeat=3))          # 0 human, 1 generated
    feats = {a: [] for a in arms}
    kept = 0
    for j, i in enumerate(rows):
        h = human_tokens(arr, int(i))
        if h is None:
            continue
        g = gen[j]
        L = min(len(h[0]), len(g[0]))
        if L < args.min_len:
            continue
        src = (h, g)
        row = {}
        for a in arms:
            s = src[a[0]][0][:L]
            th = src[a[1]][1][:L]
            dt = src[a[2]][2][:L]
            f = featurise(s.copy(), repair(s, th), dt.copy(), float(ang[j]))
            if f is None:
                row = None
                break
            row[a] = f
        if row is None:
            continue
        for a in arms:
            feats[a].append(row[a])
        kept += 1

    print(f"  {kept} rows survived every arm, paired across all eight\n")
    if kept < 500:
        print("  ABORT, too few paired rows to read anything")
        return

    # The pure arms must be untouched by the repair. Assert rather than trust.
    for a in ((0, 0, 0), (1, 1, 1)):
        s, th, _ = (human_tokens(arr, int(rows[0])) if a[0] == 0 else gen[0])
        assert np.array_equal(repair(s, th), th.astype(np.int64)), \
            "the repair is not the identity on a pure arm"

    res = {}
    print(f"  {'speed':>7}{'turn':>7}{'time':>7}{'contract':>11}")
    for a in arms:
        F = np.asarray(feats[a])
        auc = float(scoring.score_features(F)["auc_rf_oob"])
        res["".join("g" if x else "h" for x in a)] = auc
        lbl = ["human", "gen"]
        print(f"  {lbl[a[0]]:>7}{lbl[a[1]]:>7}{lbl[a[2]]:>7}{auc:>11.4f}")

    hhh, ggg = res["hhh"], res["ggg"]
    gap = ggg - hhh
    print(f"\n  ceiling {hhh:.4f}   served {ggg:.4f}   gap {gap:.4f}")

    # Main effect of a channel: the mean change from switching it from human to
    # generated, averaged over the four settings of the other two.
    main = {}
    for c, ci in zip(CHANNELS, range(3)):
        d = []
        for a in arms:
            if a[ci]:
                continue
            b = list(a)
            b[ci] = 1
            d.append(res["".join("g" if x else "h" for x in b)]
                     - res["".join("g" if x else "h" for x in a)])
        main[c] = float(np.mean(d))
    share = {c: main[c] / gap for c in CHANNELS}
    inter = gap - sum(main.values())

    print("\n  main effect, share of the gap")
    for c in CHANNELS:
        print(f"    {c:>7}  {main[c]:+.4f}   {100 * share[c]:5.1f} percent")
    print(f"    {'joint':>7}  {inter:+.4f}   {100 * inter / gap:5.1f} percent")

    top = max(CHANNELS, key=lambda c: share[c])
    pair = sorted(CHANNELS, key=lambda c: -share[c])[:2]
    if abs(hhh - CEILING_REF) > CEILING_TOL:
        verdict = (f"VOID. The all human arm reads {hhh:.4f} against a "
                   f"reference of {CEILING_REF}, so truncation to the common "
                   f"length has contaminated the comparison and no attribution "
                   f"may be read.")
    elif share[top] > 0.60 and all(share[c] < 0.20 for c in CHANNELS
                                   if c != top):
        verdict = (f"SINGLE CHANNEL, {top}. It carries "
                   f"{100 * share[top]:.0f} percent of the gap and neither "
                   f"other channel reaches 20. The next arm targets {top} "
                   f"alone.")
    elif sum(share[c] for c in pair) > 0.75 and all(share[c] > 0.25
                                                    for c in pair):
        verdict = (f"TWO CHANNEL, {pair[0]} and {pair[1]}, carrying "
                   f"{100 * sum(share[c] for c in pair):.0f} percent between "
                   f"them.")
    else:
        verdict = (f"DISTRIBUTED. No channel reaches 60 percent and no pair "
                   f"reaches 75, joint term {100 * inter / gap:.0f} percent. "
                   f"The defect is in the structure across channels rather "
                   f"than inside any one of them, and an arm that fixes a "
                   f"single channel cannot close it.")
    print(f"\n  VERDICT  {verdict}\n")

    json.dump({"n_kept": kept, "seed": args.seed, "temp": args.temp,
               "ckpt": args.ckpt, "arms": res, "ceiling": hhh, "served": ggg,
               "gap": gap, "main": main, "share": share, "joint": inter,
               "verdict": verdict, "gpu_c": gpu_c()},
              open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}\n")


if __name__ == "__main__":
    main()
