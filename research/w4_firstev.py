"""w4_firstev. How much of the CONTRACT residue does the first event carry.

w4_prefix on ten streams (step0_prereg.md OUTCOME 9 and 10) found the model's
first event, emitted with no history, is the single most detectable thing it
does: one event read alone separates from human at 0.539 and forty events read
together at 0.535, on the coarse token detector. That detector is not the
contract. The contract reads eighteen whole path features, and on 2026-07-29 an
older checkpoint forced with two human events read 0.6405 against 0.6337 free
running, one seed, inside noise. So the token level finding may not be load
bearing on the goal metric, and this arm measures whether it is.

Design. Held out human rows (never in the default_rng(123) training pick). The
model is conditioned on each row's own condition, so the forced first event is
a draw from the EXACT conditional p(e0 | cond), not a cell match. Arms by k,
the number of leading events taken from the human row, model generating the
rest through the served sampler at the closed optimum:

    k 0   free running, the in arm baseline on THIS condition population
    k 1   first event human
    k 2   first two
    k 4   first four

Paired on seed: same rows, same conditions, same sampling seed, only the mask
differs. The read is contract(k) minus contract(0) averaged over seeds, with its
paired se. Human material in the scored output at k=1 is about one event in
thirty nine, so the passthrough inflation bounded by w4_token_ceiling is under
0.003 and is stated, not corrected.

Reads, registered in step0_prereg.md AMENDMENT 13 before this ran:
    k=1 drop of 0.015 or more at 3 paired se   the first event conditional is
                                               load bearing on the contract and
                                               is the next build target
    k=1 within 0.005 of zero                   real but not load bearing, the
                                               first event line closes there
    between                                    reported with the number

One trajectory per row, no selection. Diagnostic, never a serving change. Reads
training/events_*.npy and one checkpoint, never the protected eval file, never
candi_polar_flow_best.pt.
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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp                      # noqa: E402
import scoring                                                    # noqa: E402
from features import extract_feature_matrix                       # noqa: E402
from models.event_ar import (DT_MAX_MS, EventARModel,             # noqa: E402
                             class_to_dt_ms, dt_ms_to_class)
from models.event_stream_polar import (S_PAD_CLASS, TH_NULL_CLASS,  # noqa: E402
                                       dth_lattice_to_class, s2_to_class)

TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
MAX_T = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_hm_mlp.pt")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ks", default="0,1,2,4")
    ap.add_argument("--s-temp", type=float, default=0.95)
    ap.add_argument("--th-temp", type=float, default=0.90)
    ap.add_argument("--dt-temp", type=float, default=1.00)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",")]

    dev = esp._DEVICE
    lengths = np.load("training/events_len.npy")
    N = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(N, min(N, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(N), trained)
    rng = np.random.default_rng(1000 + a.seed)
    # Rows of at least max(ks)+1 events, so every forced event exists and the
    # model always has at least one event of its own to write. No other length
    # filter: the old w4_prefix's length 12 filter handed the detector a
    # population tell.
    kmax = max(ks)
    elig = held[lengths[held] > kmax]
    pick = np.sort(rng.choice(elig, a.n, replace=False))
    print(f"  corpus {N:,}, held out {len(held):,}, eligible {len(elig):,}, "
          f"using {a.n:,}, seed {a.seed}", flush=True)

    s2 = np.load("training/events_s2.npy", mmap_mode="r")[pick]
    dth = np.load("training/events_dth.npy", mmap_mode="r")[pick]
    dt_ms = np.load("training/events_dt.npy", mmap_mode="r")[pick].astype(np.float64)
    conds = np.load("training/events_cond.npy")[pick]
    L = np.minimum(lengths[pick], MAX_T).astype(np.int64)
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

    ck = torch.load(f"training/{a.ckpt}", map_location=dev, weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    print(f"  {a.ckpt} step {ck.get('step')}  temps s {a.s_temp} th {a.th_temp} "
          f"dt {a.dt_temp}", flush=True)

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

    def generate(k):
        mask = None if k == 0 else (valid & (pos < k))
        g_s = np.full((B, MAX_T), S_PAD_CLASS, dtype=np.int64)
        g_th = np.full((B, MAX_T), TH_NULL_CLASS, dtype=np.int64)
        g_dt = np.zeros((B, MAX_T), dtype=np.int64)
        for c0 in range(0, B, a.batch):
            sl = slice(c0, min(c0 + a.batch, B))
            # Same sampling seed per batch across k, so the only difference
            # between arms is the mask. Forced positions consume no draws in
            # the sampler but later positions see a different prefix, so
            # this pairs the row and the opening randomness, not every token.
            torch.manual_seed(a.seed * 100003 + c0)
            f = None if mask is None else (
                torch.from_numpy(real_s[sl]).to(dev),
                torch.from_numpy(real_th[sl]).to(dev),
                torch.from_numpy(real_dt[sl]).to(dev),
                torch.from_numpy(mask[sl]).to(dev))
            with torch.no_grad():
                s_o, th_o, dt_o = model.sample(
                    cond_t[sl].to(dev), temperature=a.s_temp,
                    th_temperature=a.th_temp, dt_temperature=a.dt_temp,
                    force=f)
            n_got = s_o.shape[1]
            g_s[sl, :n_got] = s_o.cpu().numpy()
            g_th[sl, :n_got] = th_o.cpu().numpy()
            g_dt[sl, :n_got] = dt_o.cpu().numpy()
            if c0 % (a.batch * 20) == 0:
                print(f"    k {k}  {c0}/{B}", flush=True)
        return decode_all(g_s, g_th, g_dt)

    out = {"ckpt": a.ckpt, "seed": a.seed, "n_rows": int(B),
           "temps": [a.s_temp, a.th_temp, a.dt_temp], "arms": {}}
    total = float(L.sum())
    print(f"\n  {'k':>4}{'contract':>10}{'n':>7}{'frac forced':>13}"
          f"{'collapse':>10}", flush=True)
    for k in ks:
        paths = generate(k)
        F = extract_feature_matrix(paths)
        F = F[np.all(np.isfinite(F), 1)]
        F = F[np.random.default_rng(a.seed).permutation(len(F))]
        r = scoring.score_features(F)
        frac = float((valid & (pos < k)).sum()) / total
        out["arms"][str(k)] = dict(contract=float(r["auc_rf_oob"]),
                                   n=int(len(F)), frac_forced=frac,
                                   collapse=bool(r["collapse_flag"]),
                                   collapse_features=list(r["collapse_features"]),
                                   len_p50=float(np.median([len(p) for p in paths])))
        print(f"  {k:>4}{out['arms'][str(k)]['contract']:>10.4f}{len(F):>7}"
              f"{frac:>13.4f}{str(bool(r['collapse_flag'])):>10}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"  wrote {a.out}\n  one trajectory per row, no selection", flush=True)


if __name__ == "__main__":
    main()
