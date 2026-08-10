"""Where along a movement the over dispersion lives.

PRE REGISTERED in HANDOFF.md 2026-08-06, "## Where the over dispersion lives
along a movement". Four branches and the bootstrap convention fixed before this
file existed.

Same corpus, checkpoint, split seeds, pairing and generation seed as
`w4_typicality`, so the whole sequence numbers reproduce as a check. The
generated streams are saved this time so later questions cost no GPU.

Safety. Reads training/events_*.npy and one checkpoint. Touches no evaluation
data, never scoring.py, never training/candi_polar_flow_best.pt.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_typpos.py --n 1500
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

os.environ.setdefault("EVENT_CHOICE_TEMP", "10")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
from models.event_ar import (  # noqa: E402
    EventARModel, dt_ms_to_class, prefix_state,
)
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, dth_lattice_to_class, s2_to_class,
)
from w4_beta_curve import MAX_T, N_TRAIN_DEFAULT, TRAIN_PICK_SEED  # noqa: E402
from w4_launch import N_REAL  # noqa: E402

TH_NULL_CLASS = N_REAL["th"]
NBIN = 8
BOOT = 400
NABS = 6


def per_token_nll(model, s_cls, th_cls, dt_cls, cond, lens, batch, dev):
    """Per POSITION nll and live mask, pooled over the three heads.

    Returns (nll, cnt), both (B, MAX_T). nll is the summed negative log
    likelihood of every head that is live at that position and cnt is how many
    heads that was, so nll/cnt is a per token average and the two can be pooled
    across sequences without weighting long ones more.
    """
    B = len(lens)
    NLL = np.zeros((B, MAX_T))
    CNT = np.zeros((B, MAX_T))
    with torch.no_grad():
        for c0 in range(0, B, batch):
            sl = slice(c0, min(c0 + batch, B))
            s_b, th_b = s_cls[sl].to(dev), th_cls[sl].to(dev)
            dt_b, cnd = dt_cls[sl].to(dev), cond[sl].to(dev)
            n = s_b.shape[0]
            s_p, th_p, dt_p = model.shift_inputs(s_b, th_b, dt_b)
            st = prefix_state(s_b, th_b, dt_b, cnd)
            lg_s, lg_th, lg_dt = model.forward(s_p, th_p, dt_p, st, cnd,
                                               s_b, th_b, dt_b)
            pos = torch.arange(MAX_T, device=dev).unsqueeze(0)
            live = pos < torch.from_numpy(lens[sl]).to(dev).unsqueeze(1)
            live_th = live & (th_b < TH_NULL_CLASS)
            tot = torch.zeros_like(live, dtype=torch.float32)
            cnt = torch.zeros_like(live, dtype=torch.float32)
            for lg, tgt, msk in ((lg_s, s_b, live),
                                 (lg_th, th_b.clamp(max=TH_NULL_CLASS - 1),
                                  live_th),
                                 (lg_dt, dt_b, live)):
                ll = torch.log_softmax(lg.float(), dim=-1)
                v = -ll.gather(-1, tgt.clamp(max=ll.shape[-1] - 1)
                               .unsqueeze(-1)).squeeze(-1)
                tot += torch.where(msk, v, torch.zeros_like(v))
                cnt += msk.float()
            NLL[c0:c0 + n] = tot.cpu().numpy()
            CNT[c0:c0 + n] = cnt.cpu().numpy()
    return NLL, CNT


def frac_profile(NLL, CNT, lens, nbin=NBIN):
    """Per sequence numerator and denominator in each fractional position bin.

    Fractional binning is what removes survival conditioning: every sequence
    contributes to every bin regardless of its length.
    """
    B = len(lens)
    num = np.zeros((B, nbin))
    den = np.zeros((B, nbin))
    for i in range(B):
        n = int(lens[i])
        if n <= 0:
            continue
        b = np.minimum(nbin - 1, (nbin * np.arange(n)) // n)
        np.add.at(num[i], b, NLL[i, :n])
        np.add.at(den[i], b, CNT[i, :n])
    return num, den


def boot_gap(rn, rd, gn, gd, seed, nboot=BOOT):
    """Bootstrap the generated minus real gap profile over SEQUENCES."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(nboot):
        i = rng.integers(0, len(rn), len(rn))
        j = rng.integers(0, len(gn), len(gn))
        r = rn[i].sum(0) / np.maximum(rd[i].sum(0), 1)
        g = gn[j].sum(0) / np.maximum(gd[j].sum(0), 1)
        vals.append(g - r)
    v = np.array(vals)
    return v.std(0, ddof=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--gen-batch", type=int, default=100)
    ap.add_argument("--streams", default="research/w4_typpos_streams.npz")
    ap.add_argument("--out", default="research/w4_typpos.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    lengths = np.load("training/events_len.npy")
    N = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(N, min(N, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(N), trained)
    pick = np.sort(np.random.default_rng(args.seed + 991)
                   .choice(held, min(args.n * 4, len(held)), replace=False))

    s2 = np.asarray(np.load("training/events_s2.npy", mmap_mode="r")[pick])
    dth = np.asarray(np.load("training/events_dth.npy", mmap_mode="r")[pick])
    dt_raw = np.load("training/events_dt.npy", mmap_mode="r")[pick].astype(np.float64)
    conds = np.load("training/events_cond.npy")[pick]
    L = np.minimum(lengths[pick], MAX_T).astype(np.int64)
    keep = np.flatnonzero(L >= 12)[:args.n]
    s2, dth, dt_raw, conds, L = (s2[keep], dth[keep], dt_raw[keep],
                                 conds[keep], L[keep])
    B = len(L)
    print(f"\n  {B:,} paired sequences from the never seen half\n", flush=True)

    real_s = np.full((B, MAX_T), S_PAD_CLASS, dtype=np.int64)
    real_th = np.full((B, MAX_T), TH_NULL_CLASS, dtype=np.int64)
    real_dt = np.zeros((B, MAX_T), dtype=np.float64)
    sc = s2_to_class(torch.from_numpy(s2.astype(np.int64))).numpy()
    tc = np.where(s2 > 0,
                  dth_lattice_to_class(
                      torch.from_numpy(dth.astype(np.int64))).numpy(),
                  TH_NULL_CLASS)
    for i in range(B):
        n = int(L[i])
        real_s[i, :n] = sc[i, :n]
        real_th[i, :n] = tc[i, :n]
        real_dt[i, :n] = dt_raw[i, :n]
    real_dt_cls = dt_ms_to_class(torch.from_numpy(real_dt)).numpy()

    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    cond_t = torch.from_numpy(conds[:, :4].astype(np.float32))

    print(f"  generating {B:,}, same seed and batching as w4_typicality",
          flush=True)
    torch.manual_seed(args.seed)
    gs, gt, gd = [], [], []
    with torch.no_grad():
        for c0 in range(0, B, args.gen_batch):
            sl = slice(c0, min(c0 + args.gen_batch, B))
            a, b, c = model.sample(cond_t[sl].to(dev), temperature=1.0)
            gs.append(a.cpu())
            gt.append(b.cpu())
            gd.append(c.cpu())
            if (c0 // args.gen_batch) % 3 == 0:
                print(f"    {min(c0 + args.gen_batch, B):,} / {B:,}", flush=True)
    gen_s = torch.cat(gs).numpy().astype(np.int64)
    gen_th = torch.cat(gt).numpy().astype(np.int64)
    gen_dt = torch.cat(gd).numpy().astype(np.int64)
    gen_L = np.array([int(np.argmax(gen_s[i] >= S_PAD_CLASS))
                      if (gen_s[i] >= S_PAD_CLASS).any() else MAX_T
                      for i in range(B)], dtype=np.int64)
    ok = gen_L >= 12
    np.savez_compressed(args.streams, gen_s=gen_s, gen_th=gen_th,
                        gen_dt=gen_dt, gen_L=gen_L, real_s=real_s,
                        real_th=real_th, real_dt=real_dt_cls, real_L=L,
                        cond=conds[:, :4].astype(np.float32))
    print(f"\n  saved streams to {args.streams}")
    print(f"  generated at least 12 events: {int(ok.sum()):,} / {B:,}\n",
          flush=True)

    rN, rC = per_token_nll(model, torch.from_numpy(real_s),
                           torch.from_numpy(real_th),
                           torch.from_numpy(real_dt_cls), cond_t, L,
                           args.batch, dev)
    gN, gC = per_token_nll(model, torch.from_numpy(gen_s),
                           torch.from_numpy(gen_th),
                           torch.from_numpy(gen_dt), cond_t, gen_L,
                           args.batch, dev)
    gN, gC, gL = gN[ok], gC[ok], gen_L[ok]

    whole_r = rN.sum() / rC.sum()
    whole_g = gN.sum() / gC.sum()
    print(f"  whole sequence check   real {whole_r:.4f}  gen {whole_g:.4f}"
          f"  gap {whole_g - whole_r:+.4f}   (w4_typicality had +0.2659)\n")

    rn, rd = frac_profile(rN, rC, L)
    gn, gd = frac_profile(gN, gC, gL)
    r = rn.sum(0) / np.maximum(rd.sum(0), 1)
    g = gn.sum(0) / np.maximum(gd.sum(0), 1)
    gap = g - r
    se = boot_gap(rn, rd, gn, gd, args.seed + 31)

    print("  FRACTIONAL POSITION, eight bins, no survival conditioning")
    print(f"    {'bin':<6}{'real':>9}{'gen':>9}{'gap':>10}{'se':>8}")
    for k in range(NBIN):
        print(f"    {(k + 0.5) / NBIN:<6.2f}{r[k]:>9.4f}{g[k]:>9.4f}"
              f"{gap[k]:>+10.4f}{se[k]:>8.4f}")

    dse = float(np.hypot(se[-1], se[0]))
    contrast = float(gap[-1] - gap[0])
    print(f"\n    last minus first gap  {contrast:+.4f}  se {dse:.4f}")

    print("\n  ABSOLUTE INDEX, secondary read")
    print(f"    {'idx':<6}{'real':>9}{'gen':>9}{'gap':>10}")
    abs_rows = []
    for t in range(NABS):
        rr = rN[:, t].sum() / max(rC[:, t].sum(), 1)
        gg = gN[:, t].sum() / max(gC[:, t].sum(), 1)
        print(f"    {t:<6}{rr:>9.4f}{gg:>9.4f}{gg - rr:>+10.4f}")
        abs_rows.append({"idx": t, "real": float(rr), "gen": float(gg),
                         "gap": float(gg - rr)})

    first_pos = gap[0] > 2 * se[0]
    flat = all(abs(gap[k] - gap[j]) <= 2 * float(np.hypot(se[k], se[j]))
               for k in range(NBIN) for j in range(NBIN))
    if contrast > 2 * dse:
        verdict = ("GROWS. Accumulation. CONTRADICTS w4_position and w4_drift, "
                   "and the disagreement must be resolved before any build.")
    elif contrast < -2 * dse:
        verdict = ("SHRINKS. Concentrated at launch, the same shape and "
                   "direction w4_position found in the band power.")
    elif first_pos and flat:
        verdict = ("IMMEDIATE. Not accumulation. A broader regime from the "
                   "first event, which points at the objective.")
    else:
        verdict = "MIXED. Report the curve, no verdict."
    print(f"\n  VERDICT  {verdict}\n")

    json.dump({"n": int(B), "n_gen_ok": int(ok.sum()),
               "whole": {"real": float(whole_r), "gen": float(whole_g),
                         "gap": float(whole_g - whole_r)},
               "frac": [{"bin": k, "real": float(r[k]), "gen": float(g[k]),
                         "gap": float(gap[k]), "se": float(se[k])}
                        for k in range(NBIN)],
               "last_minus_first": {"diff": contrast, "se": dse},
               "absolute": abs_rows, "verdict": verdict},
              open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}\n")


if __name__ == "__main__":
    main()
