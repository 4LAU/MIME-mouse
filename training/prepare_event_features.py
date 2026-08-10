"""The eighteen contract features of every human training trajectory, as the
SERVING DECODER would render them.

Why this exists. The event stream AR trunk is conditioned on four numbers: log
distance, log duration in seconds, and the cosine and sine of the straight line
angle. Nothing tells it what KIND of movement to make. The older flow family WAS
conditioned on the full eighteen dimensional character vector, and HANDOFF's
2026-07-27 section prices its obedience at a mean commanded to realized
correlation of 0.41, with curvature carrying essentially no variety at all. That
same section closes by saying the only emitter left which could pass is a
learned autoregressive model over integer steps, which is exactly what this
trunk is. The two halves have never been put together, because the AR trunk has
no feature conditioning to put them together with.

This script builds the missing label.

The label is deliberately NOT the feature vector of the original polled human
path. It is the feature vector of the path the SERVING DECODER produces from
that human's own tokens, quantised speed classes, quantised turn classes,
millisecond timing classes, lattice snap and integer rounding included. Three
reasons.

  1. That is the only thing the model can ever be asked to hit. Commanding a
     feature vector the decoder cannot render is commanding disobedience.
  2. It makes the ceiling of the whole idea explicit and already measured. If
     obedience were perfect and the commands were drawn from this distribution,
     the generated sample would BE the decoded human token sample, which HANDOFF
     prices at 0.5118 against the 0.6449 the model serves today.
  3. It goes through the same `_decode` the contract is scored on, so a label
     and a realized value are the same kind of object.

Reconstruction note. Every feature is a function of the increments, so the start
point does not matter and every path is decoded from the origin. The angle comes
from the stored condition vector, because `_decode` integrates heading from it.

Output:
    training/events_feat18.npy      float32 (N, 18), NaN where the extractor or
                                    the decoder rejects the row
    training/events_feat18_ok.npy   bool (N,)

Safety. Reads only training corpus arrays. Touches no evaluation data, no
scoring code, no checkpoint. CPU only. Honours EVENT_SNAP, which must match
whatever the serving recipe is when the labels are used; the default is the
served 2.5.

Run:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" EVENT_SNAP=2.5 PYTHONPATH=. \
        ~/venvs/mime/bin/python -u training/prepare_event_features.py \
        --workers 12
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("EVENT_SNAP", "2.5")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import FEATURE_NAMES, extract_features, resample_trajectory  # noqa: E402

DATA = Path(__file__).resolve().parent
_ARR = {}
_MOD = {}


def _init():
    """Each worker opens its own memory maps and its own copy of the decoder.
    Passing slices through the pool would copy gigabytes per chunk."""
    import experiments.event_stream_polar as esp
    from models.event_ar import DT_MAX_MS, class_to_dt_ms
    from models.event_stream_polar import (S_PAD_CLASS, TH_NULL_CLASS,
                                           dth_lattice_to_class, s2_to_class)
    _MOD.update(esp=esp, class_to_dt_ms=class_to_dt_ms, DT_MAX_MS=DT_MAX_MS,
                s2_to_class=s2_to_class, dth_lattice_to_class=dth_lattice_to_class,
                S_PAD_CLASS=S_PAD_CLASS, TH_NULL_CLASS=TH_NULL_CLASS)
    for k in ("s2", "dth", "dt"):
        _ARR[k] = np.load(DATA / f"events_{k}.npy", mmap_mode="r")
    _ARR["len"] = np.load(DATA / "events_len.npy", mmap_mode="r")
    _ARR["cond"] = np.load(DATA / "events_cond.npy", mmap_mode="r")


def decode_row(i):
    """One training row through the serving decoder. Mirrors the token to path
    conversion in `w4_paired.gen_paths` exactly, so a label and a generated
    value are produced by the same code path."""
    m = _MOD
    L = int(_ARR["len"][i])
    if L < 2:
        return None
    s2 = torch.from_numpy(_ARR["s2"][i, :L].astype(np.int64))
    dth = torch.from_numpy(_ARR["dth"][i, :L].astype(np.int64))
    s_cls = m["s2_to_class"](s2)
    th_cls = torch.where(s2 > 0, m["dth_lattice_to_class"](dth),
                         torch.full_like(dth, m["TH_NULL_CLASS"]))
    dt_cls = torch.round(torch.from_numpy(_ARR["dt"][i, :L].astype(np.float32))
                         ).long().clamp(0, m["DT_MAX_MS"])
    dt_ms = m["class_to_dt_ms"](dt_cls)
    esp = m["esp"]
    dt_z = ((torch.log(dt_ms.clamp(min=0.05)) - esp._DT_MEAN) / esp._DT_STD
            ).numpy()
    ang = float(np.arctan2(_ARR["cond"][i, 3], _ARR["cond"][i, 2]))
    return esp._decode(dt_z, s_cls.numpy(), th_cls.numpy(), 0.0, 0.0, ang)


def chunk(bounds):
    lo, hi = bounds
    out = np.full((hi - lo, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    for j, i in enumerate(range(lo, hi)):
        p = decode_row(i)
        if p is None or len(p) < 2:
            continue
        f = extract_features(resample_trajectory(p, hz=125.0))
        if f is not None and np.all(np.isfinite(f)):
            out[j] = f
    return lo, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--n", type=int, default=0, help="0 means the whole corpus")
    ap.add_argument("--out", default="training/events_feat18.npy")
    args = ap.parse_args()

    N = len(np.load(DATA / "events_len.npy", mmap_mode="r"))
    if args.n:
        N = min(N, args.n)
    bounds = [(lo, min(lo + args.chunk, N)) for lo in range(0, N, args.chunk)]
    print(f"  EVENT_SNAP={os.environ['EVENT_SNAP']}")
    print(f"  {N:,} rows, {len(bounds)} chunks, {args.workers} workers",
          flush=True)

    F = np.full((N, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    done = 0
    with Pool(args.workers, initializer=_init) as pool:
        for lo, block in pool.imap_unordered(chunk, bounds):
            F[lo:lo + len(block)] = block
            done += len(block)
            if done % (args.chunk * 50) < args.chunk:
                print(f"  {done:,}/{N:,}", flush=True)

    ok = np.all(np.isfinite(F), axis=1)
    np.save(args.out, F)
    np.save(args.out.replace(".npy", "_ok.npy"), ok)
    print(f"  {ok.sum():,}/{N:,} extracted, {N - ok.sum():,} rejected")
    print(f"  wrote {args.out}")
    for j, n in enumerate(FEATURE_NAMES):
        c = F[ok, j]
        print(f"    {n:<26}{c.mean():>16.4f}{c.std():>16.4f}")


if __name__ == "__main__":
    main()
