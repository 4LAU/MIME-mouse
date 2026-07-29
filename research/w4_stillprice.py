"""What is still placement worth to the detector?

`w4_dtstruct` showed the speed alternation that `w4_seqstats` flagged is really
the interleaving of zero displacement still events among moving ones, and
`w4_tickstruct` showed both models get that interleaving wrong: human stills are
9.48 percent of events in clumps of gap dispersion 21.6, the served masked model
emits 15.67 percent spread more evenly, and moving runs are cut from a p90 of 21
to 12. Neither model's speed is wrong once the stills are removed, 0.7406 and
0.6894 at lag 1 with matching lag2 over lag1 ratios.

Knowing a property is wrong does not say it is worth fixing. This prices it the
way `w4_arrangement` priced speed order: take real human data, corrupt ONLY that
property, and see how much the detector notices. Any lift is attributable,
because the corruption is a permutation.

The corruption is a riffle. Each trajectory splits into its still events and its
moving events. Both subsequences keep their internal order exactly; only the
interleaving between them is redrawn. So the multiset of speeds, turns and
intervals is untouched, every marginal is preserved to the last bit, and the run
length and gap dispersion statistics are the only things that move.

Arms:
  passthrough    corpus tokens through the serving decoder, no change. This is
                 the floor for this instrument and must be read first: if it is
                 already high, the decode path is lossy and nothing below it
                 means anything.
  riffle W       stills reinterleaved uniformly inside non overlapping windows
                 of W events. W=0 is the whole sequence. Window 16 is the size
                 `w4_arrangement` used, where permuting speeds took human data
                 from 0.5576 to 0.8595, so that arm is directly comparable.
  share P        the still SHARE moved to P, which the riffle cannot test
                 because it holds the count fixed. Below the human share stills
                 are dropped at random and their interval is added to the next
                 event; above it, stills are duplicated and their interval is
                 split. Total elapsed time is preserved either way and the
                 moving subsequence is untouched, so this isolates how many
                 stills there are from where they sit. The arms that matter are
                 0.0621, what `event_ar_v1` emits, and 0.1567, what the served
                 masked model emits, against a human 0.0948.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_stillprice.py --n 1500
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

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
import scoring  # noqa: E402
from features import extract_feature_matrix  # noqa: E402
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TH_NULL_CLASS, TICK_CLASS, dth_lattice_to_class, s2_to_class,
)
from w4_seqstats import _acf, _gap_stats  # noqa: E402


def _riffle(order_still, n, window, rng):
    """Return the new index order for one trajectory. `order_still` is the
    boolean still mask in original order. Both subsequences keep their internal
    order; only which slots the stills occupy is redrawn."""
    out = np.empty(n, dtype=np.int64)
    w = n if window <= 0 else window
    for a in range(0, n, w):
        b = min(a + w, n)
        seg = np.arange(a, b)
        st = seg[order_still[a:b]]
        mo = seg[~order_still[a:b]]
        if len(st) == 0 or len(mo) == 0:
            out[a:b] = seg
            continue
        slots = rng.permutation(b - a)[:len(st)]
        slots.sort()
        pos = np.empty(b - a, dtype=np.int64)
        mask = np.zeros(b - a, dtype=bool)
        mask[slots] = True
        pos[mask] = st
        pos[~mask] = mo
        out[a:b] = pos
    return out


def _decode_all(s_cls, th_cls, dt_ms, lens, conds):
    paths, stills = [], []
    for i in range(len(lens)):
        n = int(lens[i])
        if n < 12:
            continue
        s = np.full(256, S_PAD_CLASS, dtype=np.int64)
        t = np.full(256, TH_NULL_CLASS, dtype=np.int64)
        d = np.zeros(256, dtype=np.float64)
        s[:n] = s_cls[i, :n]
        t[:n] = th_cls[i, :n]
        d[:n] = dt_ms[i, :n]
        dz = (np.log(np.maximum(d, 0.05)) - esp._DT_MEAN) / esp._DT_STD
        ang = math.atan2(float(conds[i, 3]), float(conds[i, 2]))
        p = esp._decode(dz, s, t, 0.0, 0.0, ang)
        if p is not None and len(p) >= 4:
            paths.append(np.asarray(p, dtype=np.float64))
            stills.append(s[:n] <= TICK_CLASS)
    return paths, stills


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--windows", default="0,16",
                    help="riffle window sizes; 0 is the whole sequence")
    ap.add_argument("--shares", default="",
                    help="target still shares, e.g. 0.0621,0.1567")
    ap.add_argument("--out", default="research/w4_stillprice.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    lengths = np.load("training/events_len.npy")
    pick = np.sort(rng.choice(len(lengths), args.n, replace=False))
    s2 = np.load("training/events_s2.npy", mmap_mode="r")[pick]
    dth = np.load("training/events_dth.npy", mmap_mode="r")[pick]
    dt = np.load("training/events_dt.npy", mmap_mode="r")[pick].astype(np.float64)
    conds = np.load("training/events_cond.npy")[pick]
    L = np.minimum(lengths[pick], 256)

    s_cls = s2_to_class(torch.from_numpy(s2.astype(np.int64))).numpy()
    th_cls = np.where(s2 > 0,
                      dth_lattice_to_class(torch.from_numpy(dth.astype(np.int64))).numpy(),
                      TH_NULL_CLASS)

    out = {}
    print(f"  {'arm':<18}{'auc':>9}{'gapVMR':>9}{'run50':>8}{'run90':>8}"
          f"{'allAc1':>9}{'ratio':>8}{'n':>7}")

    def run(label, sc, tc, dm):
        paths, stills = _decode_all(sc, tc, dm, L, conds)
        F = extract_feature_matrix(paths)
        F = F[np.all(np.isfinite(F), 1)]
        auc = float(scoring.score_features(F)["auc_rf_oob"])
        gs = _gap_stats(stills)
        sp = [np.asarray(p, dtype=np.float64) for p in
              [np.hypot(np.diff(a[:, 0]), np.diff(a[:, 1])) for a in paths]]
        ac = _acf(sp, maxlag=2)
        out[label] = dict(auc=auc, gap_vmr=gs["gap_vmr"], run_p50=gs["run_p50"],
                          run_p90=gs["run_p90"], ac1=ac[0], ac2=ac[1],
                          n=len(F))
        print(f"  {label:<18}{auc:>9.4f}{gs['gap_vmr']:>9.4f}"
              f"{gs['run_p50']:>8.1f}{gs['run_p90']:>8.1f}{ac[0]:>9.4f}"
              f"{ac[1] / ac[0]:>8.3f}{len(F):>7}", flush=True)

    run("passthrough", s_cls, th_cls, dt)

    for wtxt in [w for w in args.windows.split(",") if w != ""]:
        w = int(wtxt)
        r = np.random.default_rng(args.seed + 1000 + w)
        sc = s_cls.copy()
        tc = th_cls.copy()
        dm = dt.copy()
        for i in range(len(L)):
            n = int(L[i])
            if n < 12:
                continue
            still = s_cls[i, :n] <= TICK_CLASS
            if still.sum() == 0 or still.sum() == n:
                continue
            o = _riffle(still, n, w, r)
            sc[i, :n] = s_cls[i, o]
            tc[i, :n] = th_cls[i, o]
            dm[i, :n] = dt[i, o]
        run(f"riffle w={w}" if w else "riffle whole", sc, tc, dm)

    for stxt in [s for s in args.shares.split(",") if s != ""]:
        target = float(stxt)
        r = np.random.default_rng(args.seed + 2000 + int(target * 1e4))
        sc = np.full_like(s_cls, S_PAD_CLASS)
        tc = np.full_like(th_cls, TH_NULL_CLASS)
        dm = np.zeros_like(dt)
        newL = L.copy()
        for i in range(len(L)):
            n = int(L[i])
            if n < 12:
                continue
            still = np.flatnonzero(s_cls[i, :n] <= TICK_CLASS)
            cur = len(still) / n
            keep_s, keep_t, keep_d = [], [], []
            if target < cur and len(still):
                # drop stills at random; their interval moves to the next event
                ndrop = int(round((cur - target) * n))
                drop = set(r.choice(still, min(ndrop, len(still)),
                                    replace=False).tolist())
                carry = 0.0
                for k in range(n):
                    if k in drop:
                        carry += dt[i, k]
                        continue
                    keep_s.append(s_cls[i, k])
                    keep_t.append(th_cls[i, k])
                    keep_d.append(dt[i, k] + carry)
                    carry = 0.0
                if carry > 0 and keep_d:
                    keep_d[-1] += carry
            elif target > cur and len(still):
                # duplicate stills; their interval is split across the copies
                nadd = int(round((target - cur) * n))
                add = r.choice(still, nadd, replace=True)
                extra = np.bincount(add, minlength=n)
                for k in range(n):
                    reps = 1 + int(extra[k])
                    for _ in range(reps):
                        keep_s.append(s_cls[i, k])
                        keep_t.append(th_cls[i, k])
                        keep_d.append(dt[i, k] / reps)
            else:
                keep_s = list(s_cls[i, :n])
                keep_t = list(th_cls[i, :n])
                keep_d = list(dt[i, :n])
            m = min(len(keep_s), s_cls.shape[1])
            sc[i, :m] = keep_s[:m]
            tc[i, :m] = keep_t[:m]
            dm[i, :m] = keep_d[:m]
            newL[i] = m
        saveL = L.copy()
        L[:] = newL
        run(f"share {target:.4f}", sc, tc, dm)
        L[:] = saveL

    json.dump(out, open(args.out, "w"), indent=2)
    print("\n  passthrough is the floor for this instrument, read it first")
    print("  every riffle arm is a permutation, so all marginals are exact")
    print("  share arms change the count, not the placement, and preserve the")
    print("  total elapsed time and the moving subsequence exactly")


if __name__ == "__main__":
    main()
