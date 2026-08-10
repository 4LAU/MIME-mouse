"""Does a real history make the model's own continuation better?

`w4_prefix` swept a teacher forced opening and the score fell steadily with how
much real opening the model got, from 0.6337 at k 0 to 0.5445 at k 128. Two
things move together in that sweep and it cannot separate them.

  a. The model is on firmer ground because its history is real.
  b. More of the scored output IS real human data. At k 64 against a median
     length of 44, three quarters of what gets scored was never generated.

The scattered placement arm was meant to break the tie and could not: splicing
real tokens into mid flight positions is destructive on its own, so it reads
0.8120 against the prefix arm's 0.6028 at the same forced fraction. Position
clearly matters. The direction is not readable off that arm.

This script removes b entirely. SCORE ONLY WHAT THE MODEL WROTE. For each k,
the scored object is tokens k onward, and the only difference between the two
arms is what sat in positions 0 to k while those tokens were being produced.

  self    the model's own first k tokens, free running exactly as served
  forced  a real held out human's first k tokens
  human   the real continuation, model never consulted, as a floor

Same rows, same window, same decode, same reference class. `forced` below
`self` is compounding error with nothing else in the picture: identical
generation, identical scored segment, and the only thing that changed is
whether the history the model conditioned on was its own. `forced` level with
`self` is a local defect, and the per step conditional is what needs work.

The contract scorer cannot be the instrument here and a pilot proved it. A
suffix cut out of the middle of a movement is not a point A to point B
trajectory, and the contract scorer's human class still is, so scoring a suffix
against it read 0.83 on the whole suffix and 0.92 on a truncated one FOR REAL
HUMAN DATA. At a floor that high the classifier has already separated the two
classes and any further difference between arms is compressed into what is
left, so a null there would mean nothing.

So the readout is a two sample test between suffixes and suffixes. Real human
suffixes are split in half. One half is the reference class for all three
comparisons, the other half is the probe. Every comparison then has the same
class sizes, the same reference, and the same kind of object on both sides, and
the human probe against the human reference is a genuine floor at 0.5 rather
than an assumed one.

This is a DIAGNOSTIC and never a contract number. `scoring.py` is untouched and
is not in this path. The recipe below is copied from `score_features` so the
diagnostic behaves like the thing it is standing in for, but a number out of
this script must never be written to a ledger `score` field or compared to the
served 0.65. It answers one yes or no question about where the defect lives.

Length is the one thing that can still leak. If a real history makes the model
run longer, and human continuations are longer, the arm improves for a reason
that is not shape. So every arm is scored twice, once over the whole suffix and
once truncated to the first `--fix` events, where length cannot vary at all.
The truncated column is the strict one. If the two disagree, believe it.

Rows come from the 2,528,855 trajectories the `default_rng(123)` training
subset never selected. Rows are dropped unless all three arms yield at least
`--minsuf` events after the cut, so the three arms at one k are the same rows.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_suffix.py \
        --ckpt event_ar_v2_s40000.pt --n 5000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
from features import extract_feature_matrix  # noqa: E402
from models.event_ar import (  # noqa: E402
    DT_MAX_MS, EventARModel, class_to_dt_ms, dt_ms_to_class,
)
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TH_NULL_CLASS, dth_lattice_to_class, s2_to_class,
)

TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
MAX_T = 256
RF_SEED = 42
RF_N_ESTIMATORS = 100


def two_sample_auc(ref: np.ndarray, probe: np.ndarray) -> tuple:
    """RF out of bag AUC separating two feature matrices, same recipe as
    `scoring.score_features`: balance to the shorter, reference is class 0,
    probe is class 1, RandomForest(100, oob_score=True, random_state=42), AUC
    of the out of bag decision function.

    Not the contract. The contract's class 0 is a fixed protected file of whole
    human movements, which is the right reference for a whole synthetic
    movement and the wrong one for a fragment. Here both classes are fragments
    cut at the same position, so 0.5 means indistinguishable and the number is
    only ever compared to the other arms in the same block.
    """
    n = min(len(ref), len(probe))
    if n < 40:
        return float("nan"), n
    X = np.vstack([ref[:n], probe[:n]])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    clf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, oob_score=True,
                                 n_jobs=-1, random_state=RF_SEED)
    clf.fit(X, y)
    return float(roc_auc_score(y, clf.oob_decision_function_[:, 1])), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--ks", default="8,16,32",
                    help="how many leading tokens differ between the arms. The "
                         "scored window is everything after this")
    ap.add_argument("--minsuf", type=int, default=16,
                    help="events required after the cut, in every arm, for a "
                         "row to count at this k")
    ap.add_argument("--fix", type=int, default=16,
                    help="the strict column truncates every suffix to this "
                         "many events so length cannot differ between arms")
    ap.add_argument("--out", default="research/w4_suffix.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    lengths = np.load("training/events_len.npy")
    N = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(N, min(N, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(N), trained)
    pick = np.sort(np.random.default_rng(args.seed)
                   .choice(held, args.n, replace=False))
    print(f"  corpus {N:,}, never seen {len(held):,}, drew {args.n:,}",
          flush=True)

    s2 = np.load("training/events_s2.npy", mmap_mode="r")[pick]
    dth = np.load("training/events_dth.npy", mmap_mode="r")[pick]
    dt_ms = np.load("training/events_dt.npy", mmap_mode="r")[pick].astype(np.float64)
    conds = np.load("training/events_cond.npy")[pick]
    L = np.minimum(lengths[pick], MAX_T).astype(np.int64)

    ks = [int(t) for t in args.ks.split(",") if t != ""]
    keep = L >= max(ks) + args.minsuf
    # one row set for the whole run, so the k columns are on the same rows and
    # the trend across k is not a trend in which trajectories got included
    s2, dth, dt_ms, conds, L = (s2[keep], dth[keep], dt_ms[keep],
                                conds[keep], L[keep])
    B = len(L)
    print(f"  {B:,} rows at least {max(ks) + args.minsuf} events, "
          f"median length {int(np.median(L))}\n", flush=True)

    real_s = np.full((B, MAX_T), S_PAD_CLASS, dtype=np.int64)
    real_th = np.full((B, MAX_T), TH_NULL_CLASS, dtype=np.int64)
    real_dt = np.zeros((B, MAX_T), dtype=np.int64)
    sc = s2_to_class(torch.from_numpy(np.asarray(s2, dtype=np.int64))).numpy()
    tc = np.where(np.asarray(s2) > 0,
                  dth_lattice_to_class(
                      torch.from_numpy(np.asarray(dth, dtype=np.int64))).numpy(),
                  TH_NULL_CLASS)
    dc = dt_ms_to_class(torch.from_numpy(dt_ms)).numpy()
    for i in range(B):
        n = int(L[i])
        real_s[i, :n] = sc[i, :n]
        real_th[i, :n] = tc[i, :n]
        real_dt[i, :n] = dc[i, :n].clip(0, DT_MAX_MS)

    cond_t = torch.from_numpy(conds[:, :4].astype(np.float32))
    angs = np.arctan2(conds[:, 3].astype(np.float64),
                      conds[:, 2].astype(np.float64))

    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    print(f"  {args.ckpt} step {ck.get('step')} "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params\n",
          flush=True)

    pos = np.arange(MAX_T)[None, :]
    valid = pos < L[:, None]

    def generate(mask):
        g_s = np.full((B, MAX_T), S_PAD_CLASS, dtype=np.int64)
        g_th = np.full((B, MAX_T), TH_NULL_CLASS, dtype=np.int64)
        g_dt = np.zeros((B, MAX_T), dtype=np.int64)
        for c0 in range(0, B, args.batch):
            sl = slice(c0, min(c0 + args.batch, B))
            f = None if mask is None else (
                torch.from_numpy(real_s[sl]).to(dev),
                torch.from_numpy(real_th[sl]).to(dev),
                torch.from_numpy(real_dt[sl]).to(dev),
                torch.from_numpy(mask[sl]).to(dev),
            )
            s_o, th_o, dt_o = model.sample(
                cond_t[sl].to(dev), temperature=args.temp, force=f)
            m = s_o.shape[1]
            g_s[sl, :m] = s_o.cpu().numpy()
            g_th[sl, :m] = th_o.cpu().numpy()
            g_dt[sl, :m] = dt_o.cpu().numpy()
        return g_s, g_th, g_dt

    def suffix_len(s_arr, k):
        """Events before the first PAD, counting from the cut."""
        tail = s_arr[:, k:] >= S_PAD_CLASS
        return np.where(tail.any(1), tail.argmax(1), tail.shape[1])

    def decode_window(streams, k, rows, cap):
        s_arr, th_arr, dt_arr = streams
        paths = []
        for i in rows:
            s_i = s_arr[i, k:].copy()
            if cap is not None:
                s_i[cap:] = S_PAD_CLASS
            d = class_to_dt_ms(torch.from_numpy(dt_arr[i, k:])).numpy()
            dz = (np.log(np.maximum(d, 0.05)) - esp._DT_MEAN) / esp._DT_STD
            p = esp._decode(dz, s_i, th_arr[i, k:], 0.0, 0.0, float(angs[i]))
            if p is not None and len(p) >= 4:
                paths.append(np.asarray(p, dtype=np.float64))
        return paths

    def feats(streams, k, rows, cap):
        F = extract_feature_matrix(decode_window(streams, k, rows, cap))
        return F[np.all(np.isfinite(F), 1)]

    print("  generating free running arm once, it does not depend on k",
          flush=True)
    free = generate(None)
    human = (real_s, real_th, real_dt)

    out = {"ckpt": args.ckpt, "fix": args.fix, "rows_drawn": int(B),
           "diagnostic_only": True, "k": {}}
    print(f"\n  {'k':>5}{'probe':>9}{'whole suffix':>14}"
          f"{f'first {args.fix}':>14}{'n/class':>9}")

    for k in ks:
        forced = generate(valid & (pos < k))
        arms = {"self": free, "forced": forced, "human": human}
        ok = np.ones(B, dtype=bool)
        for a in arms.values():
            ok &= suffix_len(a[0], k) >= args.minsuf
        rows = np.flatnonzero(ok)
        # half the real suffixes are the reference class for all three
        # comparisons, the other half is the human probe, so the floor arm has
        # exactly the class sizes and the class 0 set the model arms do
        ref_rows, probe_rows = rows[0::2], rows[1::2]

        rec = {"n_rows": int(len(rows)), "dropped": int(B - len(rows)),
               "arms": {}}
        out["k"][str(k)] = rec
        for cap, col in ((None, "whole"), (args.fix, "fixed")):
            ref = feats(human, k, ref_rows, cap)
            for name, st in arms.items():
                auc, n = two_sample_auc(ref, feats(st, k, probe_rows, cap))
                rec["arms"].setdefault(name, {})[col] = auc
                rec["arms"][name][f"n_{col}"] = n
        for name in ("human", "self", "forced"):
            d = rec["arms"][name]
            tag = name + (" (floor)" if name == "human" else "")
            print(f"  {k:>5}{tag:>9}{d['whole']:>14.4f}{d['fixed']:>14.4f}"
                  f"{d['n_whole']:>9}", flush=True)
        s, f = rec["arms"]["self"], rec["arms"]["forced"]
        print(f"  {'':>5}{'gap':>9}{f['whole'] - s['whole']:>14.4f}"
              f"{f['fixed'] - s['fixed']:>14.4f}{'':>9}"
              f"  dropped {rec['dropped']}\n", flush=True)

    json.dump(out, open(args.out, "w"), indent=2)
    print("  one trajectory per row, no selection, no reranking")
    print("  DIAGNOSTIC ONLY. these are suffix against suffix two sample")
    print("  numbers and are not contract scores. never ledger them as one.")
    print("  the human row is the floor. read it first: if it is not near 0.5")
    print("  the reference and probe halves differ for some reason of their")
    print("  own and the model rows below it are unreadable.")
    print("  gap clearly negative is compounding error, the model conditions")
    print("  on its own past and drifts. gap at zero is a local defect, the")
    print("  conditional is wrong everywhere and history does not matter.")
    print("  the gap must clear the floor's own distance from 0.5 to count")


if __name__ == "__main__":
    main()
