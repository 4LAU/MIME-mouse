"""Are the model's within event and step to step dependences too weak?

`w4_drift` closed the exposure fix: a genuinely human history removes only about
15 to 45 percent of the defect, so compounding error is a minority contributor
and the damage is in what one step of the model does. `w4_copula` and
`w4_couplemap` said the same thing from the feature side, roughly 56 percent of
the gap is coupling rather than marginals. This run looks for that coupling at
the only place it can be caused, the raw token stream.

Every event carries three numbers: how far the pointer moved, how much the
heading turned, and how long the step took. `models/event_ar.py` already
factorises them correctly, s then theta given s then dt given s and theta, so
the chain rule is intact and this is NOT the composition failure the AR model
was built to fix. But the mechanism carrying the within event conditioning is
the weakest one in the network,

    th_head(th_norm(x + s_ctx_embed(s_cur)))

an additive context embedding into a LayerNorm before a linear head, which is
close to a fixed logit offset per conditioning class. The cond vector by
contrast gets FiLM, a learned scale and shift. So the dependence CAN be
represented but only faintly, and faint but correctly shaped is exactly the
attenuation signature `w4_attenuation` found at the feature level on the older
masked model.

The readout is Spearman rank correlation, chosen because it is invariant to each
variable's own marginal. `w4_redundancy` established the marginals are already
right, so a measure that cannot see marginals isolates what is left.

WITHIN TRAJECTORY, not pooled. Pooling every event from every row would be
dominated by between trajectory variation: a long fast movement has both larger
steps and longer total time because of the cond vector, which is conditioning
the model already gets and is not what is under test. So the correlation is
computed inside each trajectory and then averaged across trajectories through a
Fisher z transform.

Both arms go through the IDENTICAL class to physical maps, so quantisation is
the same on both sides and cannot be the difference.

  ratio near 1 across the board   the couplings are the right strength and this
                                  suspect is closed. The remaining gap is not
                                  reachable by rewiring the within event path.
  ratio well below 1, same signs  attenuation. The model has the shape and not
                                  the strength, which is what a fixed logit
                                  offset would produce, and the conditioning
                                  path is worth changing.
  signs disagree                  something other than attenuation, and the
                                  reading above does not apply.

DIAGNOSTIC ONLY. `scoring.py` is untouched and is not in this path. Nothing here
is a contract score. Rows come from the 2,528,855 trajectories the
`default_rng(123)` training subset never selected.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_coupletok.py \
        --ckpt event_ar_v2_s40000.pt --n 8000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import rankdata

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
from models.event_ar import (  # noqa: E402
    DT_MAX_MS, EventARModel, class_to_dt_ms, dt_ms_to_class,
)
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TH_NULL_CLASS, TICK_CLASS, class_to_dtheta, class_to_speed,
    dth_lattice_to_class, s2_to_class,
)

TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
MAX_T = 256

# (label, variable a, lag on a, variable b, lag on b)
PAIRS = [
    ("speed x |turn|      ", "sp", 0, "at", 0),
    ("speed x dt          ", "sp", 0, "dt", 0),
    ("|turn| x dt         ", "at", 0, "dt", 0),
    ("speed t x speed t+1 ", "sp", 0, "sp", 1),
    ("|turn| t x |turn|t+1", "at", 0, "at", 1),
    ("turn t x turn t+1   ", "st", 0, "st", 1),
    ("dt t x dt t+1       ", "dt", 0, "dt", 1),
    ("speed t x |turn|t+1 ", "sp", 0, "at", 1),
    ("|turn| t x speed t+1", "at", 0, "sp", 1),
    ("speed t x speed t+2 ", "sp", 0, "sp", 2),
    ("dt t x dt t+2       ", "dt", 0, "dt", 2),
]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of the ranks. Returns nan when either side is
    constant, which happens on short rows where every step lands in one bin."""
    if len(a) < 8:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    da, db = np.sqrt((ra * ra).sum()), np.sqrt((rb * rb).sum())
    if da == 0 or db == 0:
        return float("nan")
    return float((ra * rb).sum() / (da * db))


def fisher_mean(rhos: np.ndarray) -> tuple:
    """Mean correlation through the Fisher z transform, and the standard error
    of that mean, both back in correlation units."""
    r = rhos[np.isfinite(rhos)]
    if len(r) < 10:
        return float("nan"), float("nan"), 0
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    m, se = z.mean(), z.std(ddof=1) / np.sqrt(len(z))
    return float(np.tanh(m)), float(np.tanh(m + se) - np.tanh(m)), len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--minev", type=int, default=24,
                    help="motion events a row needs before its own rank "
                         "correlation is stable enough to average in")
    ap.add_argument("--out", default="research/w4_coupletok.json")
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

    keep = L >= args.minev + 4
    s2, dth, dt_ms, conds, L = (s2[keep], dth[keep], dt_ms[keep],
                                conds[keep], L[keep])
    B = len(L)
    print(f"  {B:,} rows at least {args.minev + 4} events, median length "
          f"{int(np.median(L))}\n", flush=True)

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

    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    print(f"  {args.ckpt} step {ck.get('step')} "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params\n",
          flush=True)

    print("  generating, free running exactly as served", flush=True)
    g_s = np.full((B, MAX_T), S_PAD_CLASS, dtype=np.int64)
    g_th = np.full((B, MAX_T), TH_NULL_CLASS, dtype=np.int64)
    g_dt = np.zeros((B, MAX_T), dtype=np.int64)
    for c0 in range(0, B, args.batch):
        sl = slice(c0, min(c0 + args.batch, B))
        s_o, th_o, dt_o = model.sample(cond_t[sl].to(dev),
                                       temperature=args.temp)
        w = s_o.shape[1]
        g_s[sl, :w] = s_o.cpu().numpy()
        g_th[sl, :w] = th_o.cpu().numpy()
        g_dt[sl, :w] = dt_o.cpu().numpy()

    def channels(s_arr, th_arr, dt_arr, i):
        """One row's motion events as physical quantities. Identical maps on
        both arms, so quantisation is not a difference between them."""
        s_i = s_arr[i]
        end = np.argmax(s_i >= S_PAD_CLASS) if (s_i >= S_PAD_CLASS).any() \
            else len(s_i)
        s_i = s_i[:end]
        m = (s_i > TICK_CLASS) & (th_arr[i, :end] < TH_NULL_CLASS)
        if m.sum() < args.minev:
            return None
        sp = class_to_speed(torch.from_numpy(s_i[m])).numpy()
        st = class_to_dtheta(torch.from_numpy(th_arr[i, :end][m])).numpy()
        dt = class_to_dt_ms(torch.from_numpy(dt_arr[i, :end][m])).numpy()
        return {"sp": sp, "st": st, "at": np.abs(st), "dt": dt}

    def measure(streams):
        s_arr, th_arr, dt_arr = streams
        acc = {lab: [] for lab, *_ in PAIRS}
        rows = 0
        for i in range(B):
            ch = channels(s_arr, th_arr, dt_arr, i)
            if ch is None:
                continue
            rows += 1
            n = len(ch["sp"])
            for lab, ka, la, kb, lb in PAIRS:
                w = n - max(la, lb)
                if w < args.minev:
                    acc[lab].append(float("nan"))
                    continue
                acc[lab].append(spearman(ch[ka][la:la + w], ch[kb][lb:lb + w]))
        return {lab: np.asarray(v) for lab, v in acc.items()}, rows

    hum, n_h = measure((real_s, real_th, real_dt))
    gen, n_g = measure((g_s, g_th, g_dt))
    print(f"  {n_h:,} human rows and {n_g:,} generated rows carry at least "
          f"{args.minev} motion events\n", flush=True)

    out = {"ckpt": args.ckpt, "rows_human": n_h, "rows_gen": n_g,
           "minev": args.minev, "diagnostic_only": True, "pairs": {}}
    print(f"  {'pair':<22}{'human':>9}{'se':>8}{'model':>9}{'se':>8}"
          f"{'ratio':>8}{'diff':>9}")
    for lab, *_ in PAIRS:
        rh, sh, ch_ = fisher_mean(hum[lab])
        rg, sg, cg = fisher_mean(gen[lab])
        ratio = rg / rh if abs(rh) > 0.02 else float("nan")
        out["pairs"][lab.strip()] = {"human": rh, "human_se": sh,
                                     "model": rg, "model_se": sg,
                                     "ratio": ratio, "n_human": ch_,
                                     "n_model": cg}
        print(f"  {lab:<22}{rh:>9.4f}{sh:>8.4f}{rg:>9.4f}{sg:>8.4f}"
              f"{ratio:>8.2f}{rg - rh:>9.4f}", flush=True)

    json.dump(out, open(args.out, "w"), indent=2)
    print("\n  DIAGNOSTIC ONLY, never a contract score. one trajectory per row,")
    print("  no selection, no reranking.")
    print("  Spearman inside each trajectory, Fisher z averaged across rows.")
    print("  rank correlation ignores each variable's own marginal on purpose,")
    print("  because w4_redundancy already established the marginals are right.")
    print("  ratio near 1 everywhere closes this suspect. ratio well below 1")
    print("  with matching signs is attenuation and points at the additive")
    print("  within event conditioning. disagreeing signs are neither, and the")
    print("  attenuation reading does not apply.")
    print("  a ratio is only meaningful where the human correlation is large")
    print("  compared to its own se. check the se columns before reading one.")


if __name__ == "__main__":
    main()
