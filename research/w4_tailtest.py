"""A few ruined moments, or a uniform shift.

PRE REGISTERED in HANDOFF.md 2026-08-06, "## Is it a few ruined moments or a
uniform shift". Branches fixed before this file existed, including the
correction to the sentence that closed the preceding section.

A mean is the integral of its quantile function, so

    mean_gen - mean_real = integral of [Qgen(u) - Qreal(u)] du

splits the measured 0.3342 by quantile band exactly, with no binning choice.

No generation. Reuses the streams saved by `w4_typpos`.

Safety. Reads the saved streams and one checkpoint. Touches no evaluation data,
never scoring.py, never training/candi_polar_flow_best.pt.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_tailtest.py
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
HEADS = ("s", "th", "dt")
BANDS = ((0.00, 0.50), (0.50, 0.75), (0.75, 0.90), (0.90, 0.95), (0.95, 1.00))
EXTREME = 0.001


def collect(model, s, th, dt, cond, lens, batch, dev):
    """Per token surprise, kept flat with the sequence index alongside it."""
    B = len(lens)
    vals = {h: [] for h in HEADS}
    rows = {h: [] for h in HEADS}
    with torch.no_grad():
        for c0 in range(0, B, batch):
            sl = slice(c0, min(c0 + batch, B))
            s_b, th_b = s[sl].to(dev), th[sl].to(dev)
            dt_b, cnd = dt[sl].to(dev), cond[sl].to(dev)
            n = s_b.shape[0]
            s_p, th_p, dt_p = model.shift_inputs(s_b, th_b, dt_b)
            st = prefix_state(s_b, th_b, dt_b, cnd)
            lg = dict(zip(HEADS, model.forward(s_p, th_p, dt_p, st, cnd,
                                               s_b, th_b, dt_b)))
            pos = torch.arange(MAX_T, device=dev).unsqueeze(0)
            live = pos < torch.from_numpy(lens[sl]).to(dev).unsqueeze(1)
            tgts = {"s": s_b, "th": th_b.clamp(max=TH_NULL_CLASS - 1),
                    "dt": dt_b}
            msks = {"s": live, "th": live & (th_b < TH_NULL_CLASS), "dt": live}
            idx = torch.arange(c0, c0 + n, device=dev).unsqueeze(1)
            idx = idx.expand(-1, MAX_T)
            for h in HEADS:
                ll = torch.log_softmax(lg[h].float(), dim=-1)
                v = -ll.gather(-1, tgts[h].clamp(max=ll.shape[-1] - 1)
                               .unsqueeze(-1)).squeeze(-1)
                m = msks[h]
                vals[h].append(v[m].cpu().numpy())
                rows[h].append(idx[m].cpu().numpy())
    return ({h: np.concatenate(vals[h]) for h in HEADS},
            {h: np.concatenate(rows[h]) for h in HEADS})


def band_split(a, b, bands=BANDS, ngrid=200000):
    """Contribution of each quantile band to mean(b) - mean(a), exactly."""
    u = (np.arange(ngrid) + 0.5) / ngrid
    Qa = np.quantile(a, u)
    Qb = np.quantile(b, u)
    d = Qb - Qa
    total = float(d.mean())
    out = []
    for lo, hi in bands:
        i0, i1 = int(lo * ngrid), int(hi * ngrid)
        contrib = float(d[i0:i1].sum() / ngrid)
        out.append({"lo": lo, "hi": hi, "contrib": contrib,
                    "share": contrib / total if abs(total) > 1e-12 else float("nan")})
    return total, out, float(np.quantile(b, 0.5) - np.quantile(a, 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--streams", default="research/w4_typpos_streams.npz")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="research/w4_tailtest.json")
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

    rv, rr = collect(model, torch.from_numpy(z["real_s"].astype(np.int64)),
                     torch.from_numpy(z["real_th"].astype(np.int64)),
                     torch.from_numpy(z["real_dt"].astype(np.int64)),
                     cond, rL, args.batch, dev)
    gv, gr = collect(model, torch.from_numpy(z["gen_s"][ok].astype(np.int64)),
                     torch.from_numpy(z["gen_th"][ok].astype(np.int64)),
                     torch.from_numpy(z["gen_dt"][ok].astype(np.int64)),
                     cond[ok], gL[ok], args.batch, dev)

    allr = np.concatenate([rv[h] for h in HEADS])
    allg = np.concatenate([gv[h] for h in HEADS])
    out = {"n_real_tok": int(len(allr)), "n_gen_tok": int(len(allg))}

    print(f"\n  {len(allr):,} real tokens, {len(allg):,} generated\n")
    for name, a, b in [("POOLED", allr, allg)] + \
                      [(h.upper(), rv[h], gv[h]) for h in HEADS]:
        total, bands, medshift = band_split(a, b)
        print(f"  {name}   mean gap {total:+.4f}   median shift {medshift:+.4f}"
              f"   ({medshift / total:.1%} of the gap)" if abs(total) > 1e-12
              else f"  {name}")
        print(f"    {'quantile band':<18}{'contribution':>14}{'share':>9}")
        for bd in bands:
            print(f"    {bd['lo']:.2f} to {bd['hi']:.2f}      "
                  f"{bd['contrib']:>+12.4f}{bd['share']:>9.1%}")
        print()
        out[name.lower()] = {"mean_gap": total, "median_shift": medshift,
                             "bands": bands}

    # the top decile, the registered quantity
    pooled = out["pooled"]
    top10 = sum(b["contrib"] for b in pooled["bands"] if b["lo"] >= 0.90)
    top10_share = top10 / pooled["mean_gap"]
    med_share = pooled["median_shift"] / pooled["mean_gap"]
    print(f"  top 10 percent of positions carry {top10_share:.1%} of the gap")
    print(f"  median shift is {med_share:.1%} of the gap")

    if top10_share >= 0.50 and med_share < 0.25:
        verdict = ("TAIL. A small number of very bad moments. Serving time "
                   "truncation removes exactly those draws.")
    elif med_share >= 0.75:
        verdict = ("UNIFORM. Every position a little worse. The tail account "
                   "is wrong and truncation cannot help.")
    else:
        verdict = "MIXED. Report the curve, claim neither."
    print(f"\n  VERDICT  {verdict}\n")
    out["top10_share"] = top10_share
    out["median_share"] = med_share
    out["verdict"] = verdict

    # the second axis, reported and not used to decide
    thr = -np.log(EXTREME)
    def seq_stats(vals, rows, n):
        bad = {}
        for h in HEADS:
            m = vals[h] > thr
            for r in np.unique(rows[h][m]):
                bad[r] = bad.get(r, 0) + int((rows[h][m] == r).sum())
        return bad
    rbad = seq_stats(rv, rr, len(rL))
    gbad = seq_stats(gv, gr, len(ok))
    print(f"  share of sequences with at least one token below p={EXTREME}")
    print(f"    real       {len(rbad) / len(rL):.1%}")
    print(f"    generated  {len(gbad) / len(ok):.1%}")
    out["extreme"] = {"p": EXTREME, "real_share": len(rbad) / len(rL),
                      "gen_share": len(gbad) / len(ok)}

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n  wrote {args.out}\n")


if __name__ == "__main__":
    main()
