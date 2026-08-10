"""Split the over dispersion into an objective part and a state part.

PRE REGISTERED in HANDOFF.md 2026-08-06, "### The decomposition that separates
the two remaining explanations". Thresholds fixed before this file existed.

    A = H(q,p | real states)   cross entropy on real data
    B = H(p   | real states)   the model's own entropy at real states
    C = H(p   | gen  states)   the model's own entropy at its own states

    C - A = (B - A) + (C - B)   exactly

B - A is intrinsic over dispersion where the model is known to do well. C - B is
the model's own history taking it into hotter states. No generation: the streams
saved by `w4_typpos` are reused, so this is one forward pass over each set.

Safety. Reads the saved streams and one checkpoint. Touches no evaluation data,
never scoring.py, never training/candi_polar_flow_best.pt.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_entdecomp.py
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
from models.event_ar import EventARModel, prefix_state  # noqa: E402
from w4_beta_curve import MAX_T  # noqa: E402
from w4_launch import N_REAL  # noqa: E402

TH_NULL_CLASS = N_REAL["th"]
BOOT = 400


def pass_over(model, s, th, dt, cond, lens, batch, dev):
    """Per sequence summed cross entropy and summed entropy, and the count.

    Cross entropy is the NLL of the token actually present. Entropy is the full
    -sum p log p of the same conditional, which is what the model would pay on
    average for a token it drew itself at that state. Both are accumulated over
    the same live positions and the same three heads, so they are directly
    comparable and their difference is not a mixture artefact.
    """
    B = len(lens)
    CE = np.zeros(B)
    EN = np.zeros(B)
    CT = np.zeros(B)
    with torch.no_grad():
        for c0 in range(0, B, batch):
            sl = slice(c0, min(c0 + batch, B))
            s_b, th_b = s[sl].to(dev), th[sl].to(dev)
            dt_b, cnd = dt[sl].to(dev), cond[sl].to(dev)
            n = s_b.shape[0]
            s_p, th_p, dt_p = model.shift_inputs(s_b, th_b, dt_b)
            st = prefix_state(s_b, th_b, dt_b, cnd)
            lg_s, lg_th, lg_dt = model.forward(s_p, th_p, dt_p, st, cnd,
                                               s_b, th_b, dt_b)
            pos = torch.arange(MAX_T, device=dev).unsqueeze(0)
            live = pos < torch.from_numpy(lens[sl]).to(dev).unsqueeze(1)
            live_th = live & (th_b < TH_NULL_CLASS)
            ce = torch.zeros_like(live, dtype=torch.float64)
            en = torch.zeros_like(live, dtype=torch.float64)
            ct = torch.zeros_like(live, dtype=torch.float64)
            for lg, tgt, msk in ((lg_s, s_b, live),
                                 (lg_th, th_b.clamp(max=TH_NULL_CLASS - 1),
                                  live_th),
                                 (lg_dt, dt_b, live)):
                ll = torch.log_softmax(lg.float(), dim=-1).double()
                v = -ll.gather(-1, tgt.clamp(max=ll.shape[-1] - 1)
                               .unsqueeze(-1)).squeeze(-1)
                h = -(ll.exp() * ll).sum(-1)
                ce += torch.where(msk, v, torch.zeros_like(v))
                en += torch.where(msk, h, torch.zeros_like(h))
                ct += msk.double()
            CE[c0:c0 + n] = ce.sum(1).cpu().numpy()
            EN[c0:c0 + n] = en.sum(1).cpu().numpy()
            CT[c0:c0 + n] = ct.sum(1).cpu().numpy()
    return CE, EN, CT


def boot_se(num, den, seed, nboot=BOOT):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(nboot):
        i = rng.integers(0, len(num), len(num))
        v.append(num[i].sum() / max(den[i].sum(), 1))
    return float(np.std(v, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--streams", default="research/w4_typpos_streams.npz")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="research/w4_entdecomp.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    z = np.load(args.streams)
    cond = torch.from_numpy(z["cond"])
    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])

    rL = z["real_L"].astype(np.int64)
    gL = z["gen_L"].astype(np.int64)
    ok = gL >= 12
    print(f"\n  {len(rL):,} real, {int(ok.sum()):,} generated, no generation "
          f"in this run\n", flush=True)

    rCE, rEN, rCT = pass_over(model, torch.from_numpy(z["real_s"].astype(np.int64)),
                              torch.from_numpy(z["real_th"].astype(np.int64)),
                              torch.from_numpy(z["real_dt"].astype(np.int64)),
                              cond, rL, args.batch, dev)
    gCE, gEN, gCT = pass_over(model, torch.from_numpy(z["gen_s"].astype(np.int64)),
                              torch.from_numpy(z["gen_th"].astype(np.int64)),
                              torch.from_numpy(z["gen_dt"].astype(np.int64)),
                              cond, gL, args.batch, dev)
    gCE, gEN, gCT = gCE[ok], gEN[ok], gCT[ok]

    A = rCE.sum() / rCT.sum()
    Bq = rEN.sum() / rCT.sum()
    C = gEN.sum() / gCT.sum()
    Cce = gCE.sum() / gCT.sum()

    seA = boot_se(rCE, rCT, args.seed + 1)
    seB = boot_se(rEN, rCT, args.seed + 2)
    seC = boot_se(gEN, gCT, args.seed + 3)

    print(f"  A  H(q,p | real states)   cross entropy on real   {A:.4f}"
          f"  se {seA:.4f}")
    print(f"  B  H(p   | real states)   model entropy, real     {Bq:.4f}"
          f"  se {seB:.4f}")
    print(f"  C  H(p   | gen  states)   model entropy, own      {C:.4f}"
          f"  se {seC:.4f}")
    print(f"     cross check, gen NLL at gen states             {Cce:.4f}"
          f"   (w4_typpos had 1.5914)\n")

    obj = Bq - A
    sta = C - Bq
    tot = C - A
    print(f"  B - A  intrinsic over dispersion at real states   {obj:+.4f}")
    print(f"  C - B  the state distribution effect              {sta:+.4f}")
    print(f"  C - A  total, must equal the sum                  {tot:+.4f}"
          f"   sum {obj + sta:+.4f}")
    share = obj / tot if abs(tot) > 1e-12 else float("nan")
    print(f"\n  objective share of the total   {share:.1%}")

    if abs(sta) > 1e-12 and obj >= 2 * sta:
        verdict = ("OBJECTIVE DOMINANT. Maximum likelihood left breadth "
                   "unpenalised and the model is broad wherever it has "
                   "history. Sharpening at serving is a real correction.")
    elif abs(obj) > 1e-12 and sta >= 2 * obj:
        verdict = ("STATES DOMINANT. The conditional is right and the model's "
                   "own history is taking it somewhere hotter.")
    else:
        verdict = "NEITHER DOMINANT, within a factor of two. Report both."
    print(f"\n  VERDICT  {verdict}\n")

    json.dump({"A_cross_entropy_real": float(A), "B_entropy_real": float(Bq),
               "C_entropy_gen": float(C), "gen_ce_check": float(Cce),
               "se": {"A": seA, "B": seB, "C": seC},
               "objective_part": float(obj), "state_part": float(sta),
               "total": float(tot), "objective_share": float(share),
               "verdict": verdict}, open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}\n")


if __name__ == "__main__":
    main()
