"""Which coordinate of prefix_state carries the 0.3185 nat state effect.

PRE REGISTERED in HANDOFF.md 2026-08-06, "## Which coordinate of the state
carries the excess". Branches, the length confound prediction and the secondary
nearest neighbour read were all fixed before this file existed.

For one state coordinate, binned at the REAL distribution's quantiles,

    C - B = sum_b (wgen[b] - wreal[b]) * Hbar_real[b]      BETWEEN
          + sum_b  wgen[b] * (Hbar_gen[b] - Hbar_real[b])  WITHIN

exactly. BETWEEN is the part explained by visiting that coordinate's values in
the wrong proportions. WITHIN is what survives holding it fixed.

No generation. Reuses the streams saved by `w4_typpos`.

Safety. Reads the saved streams and one checkpoint. Touches no evaluation data,
never scoring.py, never training/candi_polar_flow_best.pt.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_statecoord.py
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
NBIN = 20
BOOT = 300
NN_SAMPLE = 20000
COORDS = ("log1p remaining distance", "unit x remaining", "unit y remaining",
          "elapsed / commanded", "index / buffer", "log1p distance travelled")
CONFOUNDED = (3, 4)          # registered in advance as length confounded


def entropy_and_states(model, s, th, dt, cond, lens, batch, dev):
    """Per token entropy pooled over the three heads, with its state vector.

    Entropy is summed over the same heads and the same live positions as
    `w4_entdecomp`, so the token means reproduce that run's B and C exactly.
    """
    B = len(lens)
    H, S, R, F = [], [], [], []
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
            Lb = torch.from_numpy(lens[sl]).to(dev).unsqueeze(1)
            live = pos < Lb
            live_th = live & (th_b < TH_NULL_CLASS)
            h = torch.zeros_like(live, dtype=torch.float64)
            for lg, msk in ((lg_s, live), (lg_th, live_th), (lg_dt, live)):
                ll = torch.log_softmax(lg.float(), dim=-1).double()
                e = -(ll.exp() * ll).sum(-1)
                h += torch.where(msk, e, torch.zeros_like(e))
            idx = torch.arange(c0, c0 + n, device=dev).unsqueeze(1).expand(-1, MAX_T)
            frac = pos.expand(n, -1).double() / Lb.double().clamp(min=1)
            H.append(h[live].cpu().numpy())
            S.append(st[live].float().cpu().numpy())
            R.append(idx[live].cpu().numpy())
            F.append(frac[live].cpu().numpy())
    return (np.concatenate(H), np.concatenate(S, axis=0),
            np.concatenate(R), np.concatenate(F))


def decompose(hr, sr, hg, sg, nbin=NBIN):
    """BETWEEN and WITHIN for one coordinate, using real quantile bins."""
    edges = np.quantile(sr, np.linspace(0, 1, nbin + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    br = np.clip(np.digitize(sr, edges[1:-1]), 0, len(edges) - 2)
    bg = np.clip(np.digitize(sg, edges[1:-1]), 0, len(edges) - 2)
    nb = len(edges) - 1
    cr = np.bincount(br, minlength=nb).astype(float)
    cg = np.bincount(bg, minlength=nb).astype(float)
    sr_h = np.bincount(br, weights=hr, minlength=nb)
    sg_h = np.bincount(bg, weights=hg, minlength=nb)
    wr, wg = cr / cr.sum(), cg / cg.sum()
    # empty real bins carry no real mean; give them the generated mean so the
    # identity still closes and the mass shows up in BETWEEN as it should
    Hr = np.where(cr > 0, sr_h / np.maximum(cr, 1), 0.0)
    Hg = np.where(cg > 0, sg_h / np.maximum(cg, 1), 0.0)
    Hr = np.where(cr > 0, Hr, Hg)
    between = float(((wg - wr) * Hr).sum())
    within = float((wg * np.where(cg > 0, Hg - Hr, 0.0)).sum())
    return between, within, wr, wg, Hr, Hg


def boot_between(hr, sr, rr, hg, sg, rg, seed, nboot=BOOT):
    """Sequence clustered bootstrap of BETWEEN for one coordinate."""
    rng = np.random.default_rng(seed)
    ur, ug = np.unique(rr), np.unique(rg)
    pr = {v: np.flatnonzero(rr == v) for v in ur}
    pg = {v: np.flatnonzero(rg == v) for v in ug}
    out = []
    for _ in range(nboot):
        ir = np.concatenate([pr[v] for v in rng.choice(ur, len(ur))])
        ig = np.concatenate([pg[v] for v in rng.choice(ug, len(ug))])
        out.append(decompose(hr[ir], sr[ir], hg[ig], sg[ig])[0])
    return float(np.std(out, ddof=1))


def nn_read(sr, fr, rr, sg, fg, rg, seed, lo=0.2, hi=0.8, n=NN_SAMPLE):
    """Off manifold check in the whitened joint state, matched on position.

    Same sequence reference points are excluded for the real query. Without
    that the real query's nearest neighbour is almost always the adjacent
    position of its own movement, which is trivially close, while a generated
    query has no such partner. That asymmetry inflates the ratio and is not a
    finding. With it excluded both sides answer the same question, how far is
    this state from the nearest state of a DIFFERENT real movement.
    """
    rng = np.random.default_rng(seed)
    mr = (fr >= lo) & (fr < hi)
    mg = (fg >= lo) & (fg < hi)
    A, ia = sr[mr], rr[mr]
    Bm = sg[mg]
    mu, C = A.mean(0), np.cov(A.T)
    W = np.linalg.inv(np.linalg.cholesky(C + 1e-6 * np.eye(C.shape[0])))
    Aw = (A - mu) @ W.T
    Bw = (Bm - mu) @ W.T
    pick = rng.choice(len(Aw), min(n, len(Aw)), replace=False)
    ref, ref_seq = Aw[pick], ia[pick]

    def nearest(Q, qseq):
        d = []
        for c0 in range(0, len(Q), 1024):
            q = Q[c0:c0 + 1024]
            dd = ((q[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
            if qseq is not None:
                dd[qseq[c0:c0 + 1024][:, None] == ref_seq[None, :]] = np.inf
            d.append(np.sqrt(dd.min(1)))
        return np.concatenate(d)

    pa = rng.choice(len(Aw), min(4096, len(Aw)), replace=False)
    pb = rng.choice(len(Bw), min(4096, len(Bw)), replace=False)
    return (float(np.median(nearest(Aw[pa], ia[pa]))),
            float(np.median(nearest(Bw[pb], None))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--streams", default="research/w4_typpos_streams.npz")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="research/w4_statecoord.json")
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
    ok = np.flatnonzero(gL >= 12)

    hr, sr, rr, fr = entropy_and_states(
        model, torch.from_numpy(z["real_s"].astype(np.int64)),
        torch.from_numpy(z["real_th"].astype(np.int64)),
        torch.from_numpy(z["real_dt"].astype(np.int64)),
        cond, rL, args.batch, dev)
    hg, sg, rg, fg = entropy_and_states(
        model, torch.from_numpy(z["gen_s"][ok].astype(np.int64)),
        torch.from_numpy(z["gen_th"][ok].astype(np.int64)),
        torch.from_numpy(z["gen_dt"][ok].astype(np.int64)),
        cond[ok], gL[ok], args.batch, dev)

    Bq, C = float(hr.mean()), float(hg.mean())
    gap = C - Bq
    print(f"\n  {len(hr):,} real positions, {len(hg):,} generated")
    print(f"  B {Bq:.4f}   C {C:.4f}   C - B {gap:+.4f}"
          f"   (w4_entdecomp had 1.2834, 1.6020, +0.3185)\n")

    print(f"  {'coordinate':<28}{'BETWEEN':>10}{'se':>8}{'share':>9}"
          f"{'WITHIN':>10}{'gen mean':>11}{'real mean':>11}")
    rows = []
    for k, name in enumerate(COORDS):
        b, w, _, _, _, _ = decompose(hr, sr[:, k], hg, sg[:, k])
        se = boot_between(hr, sr[:, k], rr, hg, sg[:, k], rg, args.seed + k)
        share = b / gap
        flag = "  (confounded)" if k in CONFOUNDED else ""
        print(f"  {k} {name:<26}{b:>+10.4f}{se:>8.4f}{share:>9.1%}"
              f"{w:>+10.4f}{sg[:, k].mean():>11.4f}{sr[:, k].mean():>11.4f}"
              f"{flag}")
        rows.append({"k": k, "name": name, "between": b, "se": se,
                     "share": share, "within": w,
                     "gen_mean": float(sg[:, k].mean()),
                     "real_mean": float(sr[:, k].mean()),
                     "confounded": k in CONFOUNDED})

    clean = [r for r in rows if not r["confounded"]]
    top = max(rows, key=lambda r: abs(r["share"]))
    topc = max(clean, key=lambda r: abs(r["share"]))
    if abs(top["share"]) >= 0.50:
        verdict = f"NAMED. coordinate {top['k']}, {top['name']}, carries " \
                  f"{top['share']:.1%} of the state effect."
        if top["confounded"]:
            verdict += (" REGISTERED IN ADVANCE AS LENGTH CONFOUNDED, so this "
                        "is not a mechanism. Best clean coordinate is "
                        f"{topc['k']} at {topc['share']:.1%}.")
    elif abs(top["share"]) < 0.25:
        verdict = ("DIFFUSE. No single coordinate reaches 25 percent. The "
                   "marginals are close and the difference is in the joint. "
                   "The nearest neighbour read is primary.")
    else:
        verdict = (f"PARTIAL. Largest is coordinate {top['k']}, "
                   f"{top['name']}, at {top['share']:.1%}. Claim only the table.")
    print(f"\n  VERDICT  {verdict}\n")

    dself, dgen = nn_read(sr, fr, rr, sg, fg, rg, args.seed + 100)
    ratio = dgen / dself if dself > 0 else float("nan")
    print("  JOINT READ, whitened six dimensional state, fractional position "
          "0.2 to 0.8")
    print(f"    median nearest real distance, real query        {dself:.4f}")
    print(f"    median nearest real distance, generated query   {dgen:.4f}")
    print(f"    ratio                                           {ratio:.3f}")
    print("    ratio near 1 means generated states sit on the real manifold, "
          "\n    well above 1 means they sit off it\n")

    json.dump({"B": Bq, "C": C, "gap": gap, "coords": rows,
               "verdict": verdict,
               "nn": {"real_self": dself, "gen": dgen, "ratio": ratio},
               "n_real_pos": int(len(hr)), "n_gen_pos": int(len(hg))},
              open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}\n")


if __name__ == "__main__":
    main()
