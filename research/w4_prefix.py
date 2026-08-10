"""Is the defect in the model's per-step conditional, or in what its own
history does to that conditional over 256 steps?

These are the two remaining shapes the gap can have and they need opposite
responses, so the fork is worth one measurement before another training run.

  LOCAL. The conditional p(next token | history) is wrong everywhere, by a
  little, and handing the model a real human opening changes nothing. Then the
  lever is the objective or the factorization, and exposure to its own output
  is irrelevant.

  COMPOUNDING. The conditional is close to right, but the model conditions on
  its own past, and small errors accumulate over a 256 step sequence into a
  trajectory no human would produce. Then the lever is a training time
  exposure fix, and no amount of objective work touches it.

The instrument is a teacher forced opening. Row b takes its first k tokens from
a real held out human sequence and generates the rest itself. k sweeps from 0
to the full length.

Two built in controls, and the run is worthless without both.

  k = 0 must reproduce the ordinary served score for this checkpoint, near
  0.65. If it does not, the conditioning or the decode differs from the
  standard path and nothing else in the table can be read.

  k = full is pure passthrough with the model never consulted. `w4_token_
  ceiling` measured that arm at 0.5118 against a split half floor of 0.467 to
  0.497, so this column must land near 0.51. A number above that means the
  forcing path itself is lossy and the middle of the sweep is measuring the
  instrument rather than the model.

Rows come from the 2,528,855 trajectories the `default_rng(123)` training
subset never selected, so a forced opening is not something the model has
memorised.

READ THE SHAPE, NOT ANY SINGLE CELL. Run to run contract noise is plus or minus
0.03, so only a monotone trend across the sweep means anything. A curve that
falls steadily with k is compounding error. A curve that sits flat at 0.65
until it collapses at k = full is a local defect, because in that case the real
opening buys nothing and only removing the model entirely helps.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_prefix.py \
        --ckpt event_ar_v2_s40000.pt --n 2000
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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
import scoring  # noqa: E402
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--ks", default="0,2,4,8,16,32,64,128,999",
                    help="forced prefix lengths; 999 means the whole sequence")
    ap.add_argument("--scatter-ks", default="16,64",
                    help="which k also get the scattered placement control. "
                         "A pilot put scattered k 16 at 0.7551 against a free "
                         "running 0.6082, so splicing real tokens into "
                         "mid-flight positions is itself damaging and this arm "
                         "is documentation of that, not a baseline to read the "
                         "prefix column against")
    ap.add_argument("--out", default="research/w4_prefix.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    lengths = np.load("training/events_len.npy")
    N = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(N, min(N, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(N), trained)
    rng = np.random.default_rng(args.seed)
    pick = np.sort(rng.choice(held, args.n, replace=False))
    print(f"  corpus {N:,}, never seen {len(held):,}, using {args.n:,}",
          flush=True)

    s2 = np.load("training/events_s2.npy", mmap_mode="r")[pick]
    dth = np.load("training/events_dth.npy", mmap_mode="r")[pick]
    dt_ms = np.load("training/events_dt.npy", mmap_mode="r")[pick].astype(np.float64)
    conds = np.load("training/events_cond.npy")[pick]
    L = np.minimum(lengths[pick], MAX_T).astype(np.int64)

    keep = L >= 12
    s2, dth, dt_ms, conds, L = s2[keep], dth[keep], dt_ms[keep], conds[keep], L[keep]
    print(f"  {len(L):,} rows at least 12 events, median length "
          f"{int(np.median(L))}\n", flush=True)

    # Real token streams, padded exactly as the sampler emits them so a forced
    # row and a generated row are the same object to the decoder.
    B = len(L)
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

    def decode_all(s_np, th_np, dtc_np):
        paths = []
        for i in range(len(s_np)):
            d = class_to_dt_ms(torch.from_numpy(dtc_np[i])).numpy()
            dz = (np.log(np.maximum(d, 0.05)) - esp._DT_MEAN) / esp._DT_STD
            p = esp._decode(dz, s_np[i], th_np[i], 0.0, 0.0, float(angs[i]))
            if p is not None and len(p) >= 4:
                paths.append(np.asarray(p, dtype=np.float64))
        return paths

    pos = np.arange(MAX_T)[None, :]
    valid = pos < L[:, None]

    def prefix_mask(k):
        return valid & (pos < k)

    def scatter_mask(k, salt):
        """Same count of forced tokens per row as the prefix arm, placed at
        random positions instead of at the front. This is the arm that makes
        the sweep readable: forcing real tokens lowers the score all by itself,
        simply by putting human material in the output, so the prefix column
        alone cannot distinguish a real opening from a real anything."""
        m = np.zeros((B, MAX_T), dtype=bool)
        r = np.random.default_rng(args.seed + salt)
        for i in range(B):
            n = int(L[i])
            c = min(k, n)
            if c > 0:
                m[i, r.choice(n, c, replace=False)] = True
        return m

    def generate(mask):
        g_s = np.empty((B, MAX_T), dtype=np.int64)
        g_th = np.empty((B, MAX_T), dtype=np.int64)
        g_dt = np.empty((B, MAX_T), dtype=np.int64)
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
            n_got = s_o.shape[1]
            g_s[sl] = S_PAD_CLASS
            g_th[sl] = TH_NULL_CLASS
            g_dt[sl] = 0
            g_s[sl, :n_got] = s_o.cpu().numpy()
            g_th[sl, :n_got] = th_o.cpu().numpy()
            g_dt[sl, :n_got] = dt_o.cpu().numpy()
        return decode_all(g_s, g_th, g_dt)

    out = {"ckpt": args.ckpt, "n_rows": int(B), "arms": {}}
    print(f"  {'k':>6}{'placement':>12}{'contract':>10}{'n':>7}"
          f"{'frac forced':>13}")

    def run(label, placement, k, paths, frac):
        F = extract_feature_matrix(paths)
        F = F[np.all(np.isfinite(F), 1)]
        auc = float(scoring.score_features(F)["auc_rf_oob"])
        out["arms"][label] = dict(contract=auc, n=int(len(F)), k=k,
                                  placement=placement, frac_forced=frac)
        print(f"  {k if k < MAX_T else 'full':>6}{placement:>12}{auc:>10.4f}"
              f"{len(F):>7}{frac:>13.4f}", flush=True)

    total = float(L.sum())
    scatter_at = {int(t) for t in args.scatter_ks.split(",") if t != ""}
    for ktxt in args.ks.split(","):
        k = int(ktxt)
        if k >= MAX_T:
            run("full", "all", MAX_T, decode_all(real_s, real_th, real_dt), 1.0)
            continue
        pm = prefix_mask(k)
        run(f"prefix{k}", "prefix", k,
            generate(None if k == 0 else pm), float(pm.sum()) / total)
        if k in scatter_at and k > 0:
            sm = scatter_mask(k, salt=k)
            run(f"scatter{k}", "scattered", k, generate(sm),
                float(sm.sum()) / total)

    json.dump(out, open(args.out, "w"), indent=2)
    print("\n  one trajectory per row, no selection, no reranking")
    print("  controls: k 0 must land near the served 0.65, k full near the")
    print("  0.5118 token round trip. If either misses, read nothing else.")
    print("  prefix BELOW scattered at the same k is compounding error: the")
    print("  opening is worth more than the same tokens spread out. prefix")
    print("  equal to scattered means position does not matter and the")
    print("  conditional is wrong everywhere.")
    print("  run to run contract noise is plus or minus 0.03")
    print("  reference split-half floor 0.467 to 0.512")


if __name__ == "__main__":
    main()
