"""Can the model's own likelihood see what the contract detector sees.

PRE REGISTERED in HANDOFF.md 2026-08-06, "## The typicality test. Can the
model's own likelihood see what the detector sees". Branches and both guards
were fixed before this file existed.

Paired by construction: every generated sequence uses the conditioning vector of
the held out human sequence it is compared against, so the conditioning
distribution is identical on both sides and cannot carry the result.

The difference of mean NLLs is H(q,p) - H(p) = KL(q||p) + H(q) - H(p). That is
NOT KL and is not reported as KL anywhere in this file.

Safety. Reads training/events_*.npy and one checkpoint. Touches no evaluation
data, never scoring.py, never training/candi_polar_flow_best.pt.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_typicality.py --n 1500
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
from sklearn.metrics import roc_auc_score  # noqa: E402
from w4_beta_curve import MAX_T, N_TRAIN_DEFAULT, TRAIN_PICK_SEED  # noqa: E402
from w4_launch import N_REAL  # noqa: E402

TH_NULL_CLASS = N_REAL["th"]
OBSERVED_EXCESS = 0.1512          # 0.6612 base contract AUC against a 0.51 floor
TARGETED_RATE = 1.4231            # AUC per trajectory nat, w4_beta_curve
TARGETED_RATE_FLOOR = 0.0927      # its conservative 2 sigma floor


def auc_and_se(neg, pos, nboot=400, seed=0):
    """ROC AUC of a single scalar, with a bootstrap se over SEQUENCES."""
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    x = np.concatenate([neg, pos])
    a = float(roc_auc_score(y, x))
    a = max(a, 1.0 - a)           # separability, direction reported separately
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(nboot):
        i = rng.integers(0, len(neg), len(neg))
        j = rng.integers(0, len(pos), len(pos))
        yy = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
        xx = np.concatenate([neg[i], pos[j]])
        v = roc_auc_score(yy, xx)
        vals.append(max(v, 1.0 - v))
    return a, float(np.std(vals, ddof=1))


def nll_streams(model, s_cls, th_cls, dt_cls, cond, lens, batch, dev):
    """Teacher forced per sequence mean NLL, per head, over live positions.

    Returns four arrays over sequences: total mean nll per token, and the three
    per head means. Positions at or beyond the sequence length are excluded, so
    the terminator is not scored, matching w4_beta_curve.
    """
    B = len(lens)
    out = {k: np.zeros(B) for k in ("all", "s", "th", "dt")}
    cnt = {k: np.zeros(B) for k in ("all", "s", "th", "dt")}
    with torch.no_grad():
        for c0 in range(0, B, batch):
            sl = slice(c0, min(c0 + batch, B))
            s_b = s_cls[sl].to(dev)
            th_b = th_cls[sl].to(dev)
            dt_b = dt_cls[sl].to(dev)
            cnd = cond[sl].to(dev)
            n = s_b.shape[0]
            s_p, th_p, dt_p = model.shift_inputs(s_b, th_b, dt_b)
            st = prefix_state(s_b, th_b, dt_b, cnd)
            lg_s, lg_th, lg_dt = model.forward(s_p, th_p, dt_p, st, cnd,
                                               s_b, th_b, dt_b)
            pos = torch.arange(MAX_T, device=dev).unsqueeze(0)
            live = pos < torch.from_numpy(lens[sl]).to(dev).unsqueeze(1)
            live_th = live & (th_b < TH_NULL_CLASS)
            for name, lg, tgt, msk in (
                ("s", lg_s, s_b, live),
                ("th", lg_th, th_b.clamp(max=TH_NULL_CLASS - 1), live_th),
                ("dt", lg_dt, dt_b, live),
            ):
                ll = torch.log_softmax(lg.float(), dim=-1)
                v = -ll.gather(-1, tgt.clamp(max=ll.shape[-1] - 1)
                               .unsqueeze(-1)).squeeze(-1)
                v = torch.where(msk, v, torch.zeros_like(v))
                out[name][c0:c0 + n] = v.sum(1).cpu().numpy()
                cnt[name][c0:c0 + n] = msk.sum(1).cpu().numpy()
            out["all"][c0:c0 + n] = (out["s"][c0:c0 + n] + out["th"][c0:c0 + n]
                                     + out["dt"][c0:c0 + n])
            cnt["all"][c0:c0 + n] = (cnt["s"][c0:c0 + n] + cnt["th"][c0:c0 + n]
                                     + cnt["dt"][c0:c0 + n])
    mean = {k: out[k] / np.maximum(cnt[k], 1) for k in out}
    return mean, out, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--gen-batch", type=int, default=100)
    ap.add_argument("--out", default="research/w4_typicality.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    lengths = np.load("training/events_len.npy")
    N = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(N, min(N, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(N), trained)
    # drawn from the never seen half exactly as w4_beta_curve does, but with a
    # different draw seed so this is not the same 100k rows read twice
    pick = np.sort(np.random.default_rng(args.seed + 991)
                   .choice(held, min(args.n * 4, len(held)), replace=False))

    s2 = np.load("training/events_s2.npy", mmap_mode="r")[pick]
    dth = np.load("training/events_dth.npy", mmap_mode="r")[pick]
    dt_raw = np.load("training/events_dt.npy", mmap_mode="r")[pick].astype(np.float64)
    conds = np.load("training/events_cond.npy")[pick]
    L = np.minimum(lengths[pick], MAX_T).astype(np.int64)
    keep = np.flatnonzero(L >= 12)[:args.n]
    s2, dth, dt_raw, conds, L = (np.asarray(s2)[keep], np.asarray(dth)[keep],
                                 dt_raw[keep], conds[keep], L[keep])
    B = len(L)
    print(f"\n  corpus {N:,}, never seen {len(held):,}, using {B:,} paired\n",
          flush=True)

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
    print(f"  {args.ckpt} step {ck.get('step')} "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params\n",
          flush=True)

    cond_t = torch.from_numpy(conds[:, :4].astype(np.float32))

    # generation, one sequence per held out human conditioning vector
    print(f"  generating {B:,} sequences, one shot, paired conditioning",
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
    gen_s = torch.cat(gs).numpy()
    gen_th = torch.cat(gt).numpy()
    gen_dt = torch.cat(gd).numpy()

    gen_L = np.array([int(np.argmax(gen_s[i] >= S_PAD_CLASS))
                      if (gen_s[i] >= S_PAD_CLASS).any() else MAX_T
                      for i in range(B)], dtype=np.int64)
    ok = gen_L >= 12
    print(f"\n  generated at least 12 events: {int(ok.sum()):,} / {B:,}\n",
          flush=True)

    print("  two teacher forced passes\n", flush=True)
    real_mean, _, _ = nll_streams(
        model, torch.from_numpy(real_s), torch.from_numpy(real_th),
        torch.from_numpy(real_dt_cls), cond_t, L, args.batch, dev)
    gen_mean, _, _ = nll_streams(
        model, torch.from_numpy(gen_s.astype(np.int64)),
        torch.from_numpy(gen_th.astype(np.int64)),
        torch.from_numpy(gen_dt.astype(np.int64)),
        cond_t, gen_L, args.batch, dev)

    out = {"n_pairs": int(B), "n_gen_ok": int(ok.sum()), "ckpt": args.ckpt}

    print(f"  {'head':<8}{'real nll':>11}{'gen nll':>11}{'gen - real':>12}"
          f"{'auc':>9}{'se':>8}")
    for head in ("all", "s", "th", "dt"):
        r = real_mean[head]
        g = gen_mean[head][ok]
        a, se = auc_and_se(r, g, seed=args.seed + 5)
        d = float(g.mean() - r.mean())
        print(f"  {head:<8}{r.mean():>11.4f}{g.mean():>11.4f}{d:>+12.4f}"
              f"{a:>9.4f}{se:>8.4f}")
        out[f"nll_{head}"] = {"real": float(r.mean()), "gen": float(g.mean()),
                              "diff": d, "auc": a, "se": se}

    # GUARDS
    rng = np.random.default_rng(args.seed + 17)
    perm = rng.permutation(B)
    h1, h2 = perm[:B // 2], perm[B // 2:]
    fa, fse = auc_and_se(real_mean["all"][h1], real_mean["all"][h2],
                         seed=args.seed + 6)
    la, lse = auc_and_se(L.astype(float), gen_L[ok].astype(float),
                         seed=args.seed + 7)
    print(f"\n  FLOOR GUARD   real vs real, same instrument   {fa:.4f} se {fse:.4f}"
          f"   {'PASS' if abs(fa - 0.5) <= 2 * fse else 'FAIL'}")
    print(f"  LENGTH GUARD  sequence length alone           {la:.4f} se {lse:.4f}")
    out["floor_guard"] = {"auc": fa, "se": fse,
                          "pass": bool(abs(fa - 0.5) <= 2 * fse)}
    out["length_guard"] = {"auc": la, "se": lse}

    # the primary branch
    a = out["nll_all"]["auc"]
    se = out["nll_all"]["se"]
    if not out["floor_guard"]["pass"]:
        verdict = "VOID, the floor guard failed"
    elif a <= 0.53 and abs(a - fa) <= 2 * float(np.hypot(se, fse)):
        verdict = "BRANCH A, BLIND. The objective cannot see the defect."
    elif a >= 0.58:
        verdict = "BRANCH B, VISIBLE. The model knows; the defect is in sampling."
    else:
        verdict = "BRANCH C, in between. Report only."
    print(f"\n  VERDICT  {verdict}")

    # the registered conversion, reported whichever way the branch fell
    ev_real = float(np.mean(L))
    gap_per_traj = out["nll_all"]["diff"] * ev_real * 3.0
    pred = abs(gap_per_traj) * TARGETED_RATE
    pred_lo = abs(gap_per_traj) * TARGETED_RATE_FLOOR
    print(f"\n  typicality gap        {gap_per_traj:+.4f} nats per trajectory")
    print(f"  predicted AUC at the targeted rate   {pred:.4f}"
          f"   floor {pred_lo:.4f}")
    print(f"  observed excess to explain           {OBSERVED_EXCESS:.4f}")
    if pred > OBSERVED_EXCESS:
        print("  the gap OVER explains the excess, linearity refuted here")
    out["conversion"] = {"gap_per_traj": gap_per_traj, "pred_auc": pred,
                         "pred_auc_floor": pred_lo,
                         "observed_excess": OBSERVED_EXCESS}
    out["verdict"] = verdict

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n  wrote {args.out}\n")


if __name__ == "__main__":
    main()
