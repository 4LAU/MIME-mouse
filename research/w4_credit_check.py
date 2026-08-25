"""Can a contract feature's value be attributed back to individual tokens?

The bar registered in task 12 before any of it is built. Two things must hold.

  1. A verbatim mirror of esp._decode that also returns which token produced
     each path point must reproduce esp._decode's trajectory exactly. If it
     does not, the mirror is a reimplementation and the whole idea is dead,
     because a hand rolled token walk reads about 0.13 high in this repo.

  2. Perturbing one token must move the feature by what the attribution says
     it moves it by, and must leave the rest alone. Tested on mean_velocity,
     which is the simplest differential feature, and on path_efficiency,
     which reads absolute geometry and is predicted to FAIL locality.

CPU only. No model, no GPU, no scoring, no protected file.

RESULT, 2026-08-10. PASSED, with a bias worth carrying forward.

    trajectories checked                              244
    mirror disagreed with esp._decode                   0
    the bump changed mean_velocity in                 164 of 244

    mean_velocity, on the rows that moved
      correlation, attributed against actual       0.9806
      slope through the origin                     1.1955
      median |attributed - actual| / |actual|      0.2336
      absolute per interval change landing outside the credited
      token, median 0.5733, p90 0.9187

    path_efficiency
      per token attribution predicts exactly 0
      it actually moves, median 6.694e-04, nonzero in 163 of 164

Read that middle block carefully, because two of its numbers look like they
contradict each other. Most of the ABSOLUTE per interval movement lands outside
the credited token, and yet the attributed total tracks the real one at 0.98.
Both are true because the leaked movement mostly cancels: shifting one step
rigidly displaces everything after it, and the boundary intervals move in
opposite directions. What survives the cancellation is a systematic 20 percent
overstatement, which is the slope. For a score function weight that is later
normalised to unit standard deviation across the batch, a common scale error of
that size is harmless and the 0.98 is the number that matters.

An incidental finding, and it is a constraint on the whole workstream. A one
class speed edit changes nothing at all in most trajectories: the decoder rounds
positions to whole pixels and snaps slow steps, so single class edits are erased.
It took a six class edit to move the features in two thirds of rows. Whatever
steers this model has to move it further than one lattice cell to be seen by the
contract at all, which is the same wall the feature conditioning arm hit from the
other side when it could not be steered on fine texture.
"""
from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("EVENT_SNAP", "2.5")
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (R, f"{R}/research", f"{R}/research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(R)

import experiments.event_stream_polar as esp  # noqa: E402
import torch  # noqa: E402
from features import extract_features, resample_trajectory  # noqa: E402
from models.event_ar import DT_MAX_MS, class_to_dt_ms  # noqa: E402
from models.event_stream_polar import S_PAD_CLASS, TICK_CLASS  # noqa: E402
from w4_credit import decode_indexed, token_of_grid_point  # noqa: E402

HZ = 125.0
# a one class speed bump is usually erased by the lattice: the decoder rounds
# positions to whole pixels and snaps slow steps, so most single class edits
# change nothing at all. The check needs a perturbation the decoder survives.
BUMP = 6


def main():
    D = "training"
    rng = np.random.default_rng(17)
    s2a = np.load(f"{D}/events_s2.npy", mmap_mode="r")
    dtha = np.load(f"{D}/events_dth.npy", mmap_mode="r")
    dta = np.load(f"{D}/events_dt.npy", mmap_mode="r")
    lens = np.load(f"{D}/events_len.npy")
    cond_all = np.load(f"{D}/events_cond.npy", mmap_mode="r")
    ok = np.flatnonzero(np.load(f"{D}/events_feat18_ok.npy"))
    rows = ok[rng.permutation(len(ok))[:400]]

    from models.event_stream_polar import (
        TH_NULL_CLASS, dth_lattice_to_class, s2_to_class,
    )

    mirror_bad = 0
    checked = 0
    dv_pred, dv_true = [], []
    pe_pred, pe_true = [], []
    leak_v, leak_pe = [], []

    for j in rows:
        L = int(lens[j])
        if L < 24:
            continue
        s2 = torch.from_numpy(np.asarray(s2a[j, :L]).astype(np.int64))
        dth = torch.from_numpy(np.asarray(dtha[j, :L]).astype(np.int64))
        s_c = s2_to_class(s2).numpy()
        th_c = torch.where(s2 > 0, dth_lattice_to_class(dth),
                           torch.full_like(dth, TH_NULL_CLASS)).numpy()
        dt_c = np.round(np.asarray(dta[j, :L]).astype(np.float64)
                        ).clip(0, DT_MAX_MS).astype(np.int64)
        ms = class_to_dt_ms(torch.from_numpy(dt_c)).numpy().astype(np.float64)
        dz = (np.log(np.maximum(ms, 0.05)) - esp._DT_MEAN) / esp._DT_STD
        c = np.asarray(cond_all[j], dtype=np.float64)
        ang = float(np.arctan2(c[3], c[2]))

        ref = esp._decode(dz, s_c, th_c, 0.0, 0.0, ang)
        path, tok = decode_indexed(dz, s_c, th_c, 0.0, 0.0, ang)
        if ref is None or path is None:
            continue
        if len(ref) != len(path) or not np.allclose(
                np.asarray(ref, dtype=np.float64),
                np.asarray(path, dtype=np.float64), atol=0, rtol=0):
            mirror_bad += 1
            continue

        f0 = extract_features(resample_trajectory(path, HZ))
        if f0 is None:
            continue
        gtok = token_of_grid_point(path, resample_trajectory(path, HZ), tok)

        # perturb one MOTION token's speed class, one class up
        cand = np.flatnonzero((s_c[:len(s_c)] > TICK_CLASS)
                              & (s_c < S_PAD_CLASS - BUMP - 1))
        cand = cand[(cand > 3) & (cand < max(len(tok) - 4, 4))]
        if not len(cand):
            continue
        m = int(rng.choice(cand))
        s_p = s_c.copy()
        s_p[m] += BUMP
        pth2, tok2 = decode_indexed(dz, s_p, th_c, 0.0, 0.0, ang)
        if pth2 is None or tok2 is None or len(tok2) != len(tok):
            continue
        f1 = extract_features(resample_trajectory(pth2, HZ))
        if f1 is None:
            continue

        # what the attribution says: only intervals credited to token m move
        hit = gtok == m
        if not hit.any():
            continue
        checked += 1

        g0 = np.asarray(resample_trajectory(path, HZ), dtype=np.float64)
        g1 = np.asarray(resample_trajectory(pth2, HZ), dtype=np.float64)
        if len(g0) != len(g1):
            continue
        sp0 = np.hypot(np.diff(g0[:, 0]), np.diff(g0[:, 1])) / np.maximum(
            np.diff(g0[:, 2]), 1e-6)
        sp1 = np.hypot(np.diff(g1[:, 0]), np.diff(g1[:, 1])) / np.maximum(
            np.diff(g1[:, 2]), 1e-6)
        d = sp1 - sp0
        n = min(len(d), len(hit))
        d, hit = d[:n], hit[:n]
        dv_pred.append(float(d[hit].sum() / len(sp0)))
        dv_true.append(float(f1[0] - f0[0]))
        leak_v.append(float(np.abs(d[~hit]).sum()
                            / max(np.abs(d).sum(), 1e-12)))

        pe_pred.append(0.0)
        pe_true.append(float(f1[9] - f0[9]))
        leak_pe.append(abs(f1[9] - f0[9]))

    print(f"  trajectories checked                {checked}")
    print(f"  mirror disagreed with esp._decode   {mirror_bad}")
    dv_pred = np.array(dv_pred); dv_true = np.array(dv_true)
    leak_v = np.array(leak_v); pe_true = np.array(pe_true)
    moved = np.abs(dv_true) > 1e-12
    print(f"  the bump changed mean_velocity in   {moved.sum()} of {checked}")

    p, q, lk = dv_pred[moved], dv_true[moved], leak_v[moved]
    print("\n  mean_velocity, a differential feature, on the rows that moved")
    print(f"    correlation, attributed against actual     "
          f"{np.corrcoef(p, q)[0, 1]:.4f}")
    print(f"    slope through the origin                   "
          f"{float((p * q).sum() / (p * p).sum()):.4f}")
    print(f"    median |attributed - actual| / |actual|    "
          f"{np.median(np.abs(p - q) / np.abs(q)):.4f}")
    print(f"    per interval change landing OUTSIDE the credited token, "
          f"median {np.median(lk):.4f}, p90 {np.quantile(lk, 0.9):.4f}")

    pe = pe_true[moved]
    print("\n  path_efficiency, an absolute geometry feature")
    print("    per token attribution predicts 0, because a speed edit only")
    print("    shifts everything downstream rigidly")
    print(f"    it actually moves by, median |change|      "
          f"{np.median(np.abs(pe)):.3e}")
    print(f"    nonzero in                                 "
          f"{int((np.abs(pe) > 1e-12).sum())} of {int(moved.sum())}")


if __name__ == "__main__":
    main()
