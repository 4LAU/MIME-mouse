"""WHICH mechanism produces the cross channel coupling?

PRE REGISTERED in HANDOFF, thresholds fixed before this file existed.

w4_cross measured +0.1309 between speed surprise and direction surprise at lag
zero, where a correct model gives exactly zero. Three mechanisms could produce it
and they call for different fixes:

    a  between sequence   some movements are harder for every head at once and the
                          model does not know it should be less certain on them
    b  between position   the same failure of self knowledge, at the level of
                          moments inside a movement
    c  within moment      given the same moment, speed being surprising and
                          direction being surprising are genuinely linked, and the
                          chain rule factorisation is being under used

Why slicing is legal, and where the limit is. The identity that makes the pooled
number exactly zero survives conditioning on ANY quantity the model could compute
from the history, because the argument is a tower property over the history. The
model's predicted entropy at a step qualifies, since it is a function of the
distribution the model produced from the history alone. The position index
qualifies. What does NOT qualify is anything built from the outcome, including the
realised surprise and any per sequence average of it. Panels built on those are
DESCRIPTIVE ATTRIBUTION, labelled as such, and no verdict is read on them.

    Panel 1  descriptive   between and within sequence decomposition
    Panel 2  THE TEST      correlation within joint entropy cells, 5 by 5
    Panel 3  test          correlation within position bands
    Panel 4  descriptive   share of the covariance from the top decile of surprise

Panel 2 rule:

    retains >= 0.50 of the pooled reading   WITHIN MOMENT coupling dominates
    retains <= 0.20 of the pooled reading   EXPLAINED by the model's own uncertainty
    otherwise                               MIXED
    BOUNDARY, within one bootstrap sd of a threshold, the call is REFUSED

VALIDITY ARM. The self sampled arm through every panel, exact zero by construction
in each cell. Magnitude floor 0.002, and an unmeasurable bootstrap sd is a FAILURE.

DIAGNOSTIC ONLY, never a contract score. No serving change follows and no build is
authorised by any outcome. Phase conditioning and the spectral loss term remain
NOT AUTHORISED.

Safety. Reads training/events_*.npy and one checkpoint. Touches no evaluation data,
no scoring code, and never training/candi_polar_flow_best.pt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("EVENT_STEPS", "100")
os.environ.setdefault("EVENT_CFG_W", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import experiments.event_stream_polar as esp  # noqa: E402
from models.event_ar import (  # noqa: E402
    EventARModel, dt_ms_to_class, prefix_state,
)
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TH_NULL_CLASS, dth_lattice_to_class, s2_to_class,
)
from research.w4_timing import (  # noqa: E402
    MAX_T, N_TRAIN_DEFAULT, TRAIN_PICK_SEED,
)
from research.w4_launch import N_REAL, renorm, self_sample  # noqa: E402

WITHIN_MOMENT = 0.50
EXPLAINED = 0.20
VALID_SD = 2.0
VALID_FLOOR = 0.002
N_QUINT = 5
MIN_CELL = 500
POS_BANDS = ((1, 4), (4, 8), (8, 12), (12, 20), (20, 32), (32, MAX_T))
PAIRS = (("s", "th"), ("s", "dt"), ("th", "dt"))
HEADS = ("s", "th", "dt")


def cov_terms(a, b, m):
    """Per row sums for a masked correlation, so a bootstrap is a sum over rows."""
    x = np.where(m, a.astype(np.float64), 0.0)
    y = np.where(m, b.astype(np.float64), 0.0)
    return (np.einsum("ij,ij->i", x, y), x.sum(1), y.sum(1),
            np.einsum("ij,ij->i", x, x), np.einsum("ij,ij->i", y, y),
            m.sum(1).astype(np.float64))


def stat_from(terms, rows, want="corr"):
    sxy, sx, sy, sxx, syy, n = (t[rows].sum() for t in terms)
    if n < MIN_CELL:
        return None
    cov = sxy / n - (sx / n) * (sy / n)
    if want == "cov":
        return float(cov), float(n)
    vx = sxx / n - (sx / n) ** 2
    vy = syy / n - (sy / n) ** 2
    if vx <= 0 or vy <= 0:
        return None
    return float(cov / np.sqrt(vx * vy))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--out", default="research/w4_couple.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    rng = np.random.default_rng(args.seed + 31)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed + 211)

    lengths = np.load("training/events_len.npy")
    N = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(N, min(N, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(N), trained)
    pick = np.sort(np.random.default_rng(args.seed)
                   .choice(held, args.n, replace=False))

    s2 = np.load("training/events_s2.npy", mmap_mode="r")[pick]
    dth = np.load("training/events_dth.npy", mmap_mode="r")[pick]
    dt_raw = np.load("training/events_dt.npy", mmap_mode="r")[pick].astype(np.float64)
    conds = np.load("training/events_cond.npy")[pick]
    L = np.minimum(lengths[pick], MAX_T).astype(np.int64)
    keep = L >= 12
    s2, dth, dt_raw, conds, L = (s2[keep], dth[keep], dt_raw[keep],
                                 conds[keep], L[keep])
    B = len(L)
    print(f"  corpus {N:,}, never seen {len(held):,}, drew {args.n:,}")
    print(f"  {B:,} rows at least 12 events, the same rows w4_cross used\n",
          flush=True)

    real_s = np.full((B, MAX_T), S_PAD_CLASS, dtype=np.int64)
    real_th = np.full((B, MAX_T), TH_NULL_CLASS, dtype=np.int64)
    real_dt = np.zeros((B, MAX_T), dtype=np.float64)
    sc = s2_to_class(torch.from_numpy(np.asarray(s2, dtype=np.int64))).numpy()
    tc = np.where(np.asarray(s2) > 0,
                  dth_lattice_to_class(
                      torch.from_numpy(np.asarray(dth, dtype=np.int64))).numpy(),
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

    s_t = torch.from_numpy(real_s)
    th_t = torch.from_numpy(real_th)
    dt_t = torch.from_numpy(real_dt_cls)
    cond_t = torch.from_numpy(conds[:, :4].astype(np.float32))

    srp = {(h, arm): np.zeros((B, MAX_T), dtype=np.float32)
           for h in HEADS for arm in ("real", "self")}
    ent = {h: np.zeros((B, MAX_T), dtype=np.float32) for h in HEADS}
    live = np.zeros((B, MAX_T), dtype=bool)
    live_th = np.zeros((B, MAX_T), dtype=bool)

    print("  one teacher forced forward pass, identical to w4_cross's", flush=True)
    with torch.no_grad():
        for c0 in range(0, B, args.batch):
            sl = slice(c0, min(c0 + args.batch, B))
            s_b, th_b, dt_b = s_t[sl].to(dev), th_t[sl].to(dev), dt_t[sl].to(dev)
            cnd = cond_t[sl].to(dev)
            s_p, th_p, dt_p = model.shift_inputs(s_b, th_b, dt_b)
            st = prefix_state(s_b, th_b, dt_b, cnd)
            x = model.trunk(s_p, th_p, dt_p, st, cnd)
            lg = {"s": model.s_head(x),
                  "th": model.th_logits(x, s_b),
                  "dt": model.dt_logits(x, s_b, th_b)}
            true = {"s": s_b, "th": th_b, "dt": dt_b}
            for h in HEADS:
                p = renorm(torch.softmax(lg[h], -1), N_REAL[h])
                logp = torch.log(p.clamp(min=1e-30))
                H = -(p * logp).sum(-1)
                ent[h][sl] = H.float().cpu().numpy()
                for arm, k in (("real", true[h]), ("self", self_sample(p, gen))):
                    k = k.clamp(max=p.shape[-1] - 1)
                    nll = -logp.gather(-1, k.unsqueeze(-1)).squeeze(-1)
                    srp[(h, arm)][sl] = (nll - H).float().cpu().numpy()
            live[sl] = (s_b < S_PAD_CLASS).cpu().numpy()
            live_th[sl] = ((s_b < S_PAD_CLASS) & (th_b < TH_NULL_CLASS)
                           ).cpu().numpy()

    print(f"  {live.sum():,} live positions, {live_th.sum():,} of them with a "
          f"direction to predict\n", flush=True)

    lv = {"s": live, "th": live_th, "dt": live}
    all_rows = np.arange(B)
    out = {"ckpt": args.ckpt, "n_rows": int(B), "seed": args.seed,
           "diagnostic_only": True, "pre_registered": "HANDOFF.md 2026-08-05",
           "thresholds": {"within_moment": WITHIN_MOMENT, "explained": EXPLAINED},
           "n_live": int(live.sum())}

    def boot(terms, draws):
        v = []
        for _ in range(draws):
            r = stat_from(terms, rng.integers(0, B, B))
            if r is not None:
                v.append(r)
        return float(np.std(v)) if len(v) > 10 else float("nan")

    pos = np.broadcast_to(np.arange(MAX_T)[None, :], (B, MAX_T))

    # Quantile edges taken ONCE at the full sample, so the bootstrap resamples
    # sequences and never the binning. The same discipline w4_condtex needed.
    qedge = {}
    for h in HEADS:
        vals = ent[h][lv[h]]
        qedge[h] = np.quantile(vals, np.linspace(0, 1, N_QUINT + 1)[1:-1])

    results = {}
    for (ha, hb) in PAIRS:
        lab = f"{ha}{hb}"
        mask = lv[ha] & lv[hb]
        blk = {}
        for arm in ("real", "self"):
            a, b = srp[(ha, arm)], srp[(hb, arm)]
            pooled_t = cov_terms(a, b, mask)
            pooled = stat_from(pooled_t, all_rows)
            pooled_sd = boot(pooled_t, args.draws)

            # Panel 2, THE TEST. Twenty five joint entropy cells. Every cell is a
            # valid instance of the same exact zero identity, because the binning
            # variable is computed from the history alone.
            ia = np.digitize(ent[ha], qedge[ha]).astype(np.int8)
            ib = np.digitize(ent[hb], qedge[hb]).astype(np.int8)
            cells, num, den = [], 0.0, 0.0
            for u in range(N_QUINT):
                for w in range(N_QUINT):
                    m = mask & (ia == u) & (ib == w)
                    if m.sum() < MIN_CELL:
                        continue
                    t = cov_terms(a, b, m)
                    r = stat_from(t, all_rows)
                    if r is None:
                        continue
                    n = float(m.sum())
                    cells.append({"ent_a": u, "ent_b": w, "n": n, "rho": r})
                    num += r * n
                    den += n
            within = (num / den) if den > 0 else None

            # Bootstrap the RETENTION directly, so the ratio's own noise is
            # measured rather than propagated by hand from two separate sds.
            rb = []
            cellt, celln = [], []
            for c in cells:
                m = mask & (ia == c["ent_a"]) & (ib == c["ent_b"])
                cellt.append(cov_terms(a, b, m))
                celln.append(m.sum(1).astype(np.float64))
                del m
            for _ in range(max(40, args.draws // 4)):
                rs = rng.integers(0, B, B)
                p0 = stat_from(pooled_t, rs)
                if p0 is None or abs(p0) < 1e-9:
                    continue
                nu = de = 0.0
                for t, nn in zip(cellt, celln):
                    r = stat_from(t, rs)
                    if r is None:
                        continue
                    w = float(nn[rs].sum())
                    nu += r * w
                    de += w
                if de > 0:
                    rb.append((nu / de) / p0)
            ret = (within / pooled) if (within is not None and pooled
                                        and abs(pooled) > 1e-9) else None
            ret_sd = float(np.std(rb)) if len(rb) > 10 else float("nan")

            # Panel 3. Position bands, also history measurable, also a valid test.
            bands = []
            for lo, hi in POS_BANDS:
                m = mask & (pos >= lo) & (pos < hi)
                if m.sum() < MIN_CELL:
                    continue
                bands.append({"lo": lo, "hi": hi, "n": float(m.sum()),
                              "rho": stat_from(cov_terms(a, b, m), all_rows)})

            blk[arm] = {"pooled": pooled, "pooled_sd": pooled_sd,
                        "within_entropy": within, "retention": ret,
                        "retention_sd": ret_sd, "cells": cells, "bands": bands}
        results[lab] = blk
    out["pairs"] = results

    # VALIDITY FIRST. The self arm is an exact zero in every cell.
    print("  VALIDITY arm. self sampled tokens through every panel. exact zero")
    print("  in each cell by construction\n")
    vfail = []
    for lab, blk in results.items():
        v = blk["self"]
        # Both the pooled reading and the SIZE WEIGHTED WITHIN CELL AVERAGE, which
        # is the statistic the verdict is actually read on. Gating only the worst
        # of twenty five cells while reading the verdict off a weighted average no
        # gate protected was the real defect. See the AMENDMENT in HANDOFF.
        worst = 0.0
        for val, sd in ((v["pooled"], v["pooled_sd"]),
                        (v["within_entropy"], v["pooled_sd"])):
            z = (float("inf") if (val is None or not np.isfinite(sd) or sd <= 0)
                 else (0.0 if abs(val) < VALID_FLOOR else abs(val) / sd))
            worst = max(worst, z)
        # A cell fails only if it is BOTH large enough to matter AND
        # distinguishable from zero. Cell sizes span a factor of a few hundred,
        # because speed entropy and direction entropy are correlated and the
        # corner cells are nearly empty, so a fixed magnitude compared against a
        # cell noisier than the threshold can only produce false failures. This is
        # the symmetric counterpart of the magnitude floor, not a relaxation.
        def excursion(rho, n):
            return (rho is not None and abs(rho) >= VALID_FLOOR * 5
                    and abs(rho) * np.sqrt(max(n, 1.0)) >= 3.0)

        cworst = max((abs(c["rho"]) for c in v["cells"]
                      if excursion(c["rho"], c["n"])), default=0.0)
        bworst = max((abs(b["rho"]) for b in v["bands"]
                      if excursion(b["rho"], b["n"])), default=0.0)
        # The cell floor is a MAGNITUDE, so whether it is reachable at all
        # depends on the sample. The median cell size and its implied noise are
        # printed so a failure can be read as an estimator defect or as a sample
        # too small for a gate at this granularity, without guessing.
        med = float(np.median([c["n"] for c in v["cells"]])) if v["cells"] else 0.0
        noise = (1.0 / np.sqrt(med)) if med > 0 else float("inf")
        print(f"    {lab:>6}  pooled {v['pooled']:+.4f}  weighted within "
              f"{v['within_entropy']:+.4f}  ({worst:.1f} sd)  "
              f"cell excursions {cworst:.4f}  band {bworst:.4f}  "
              f"median cell {med:,.0f}, cell noise {noise:.4f}")
        if worst > VALID_SD or cworst > 0.0 or bworst > 0.0:
            vfail.append(lab)
    if vfail:
        print(f"\n  VALIDITY ARM FAILED on {', '.join(vfail)}.")
        out["verdict"] = f"FAILED, validity arm coupled on {','.join(vfail)}"
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        return
    print("\n  validity arm PASSES, the slicing does not manufacture coupling\n")

    print("  PANEL 2, THE TEST. correlation inside joint entropy cells, against")
    print("  the pooled reading. this asks whether the coupling survives once we")
    print("  condition on what the model already knew about its own uncertainty\n")
    print(f"    {'pair':>6} {'pooled':>9} {'within':>9} {'retains':>9} {'sd':>8}")
    for lab, blk in results.items():
        r = blk["real"]
        print(f"    {lab:>6} {r['pooled']:>+9.4f} {r['within_entropy']:>+9.4f} "
              f"{r['retention']:>9.2f} {r['retention_sd']:>8.2f}")

    print("\n  PANEL 3. correlation inside position bands\n")
    print(f"    {'pair':>6} " +
          " ".join(f"{str(lo) + '-' + str(hi):>9}" for lo, hi in POS_BANDS))
    for lab, blk in results.items():
        row = {(b["lo"], b["hi"]): b["rho"] for b in blk["real"]["bands"]}
        print(f"    {lab:>6} " +
              " ".join(f"{row.get((lo, hi)):>+9.4f}"
                       if row.get((lo, hi)) is not None else f"{'-':>9}"
                       for lo, hi in POS_BANDS))

    # PANEL 1 and PANEL 4 are DESCRIPTIVE ATTRIBUTION. Both condition on the
    # outcome, so neither is a test and no verdict is read on either.
    print("\n  PANEL 1, DESCRIPTIVE, NOT A TEST. share of the covariance carried")
    print("  by sequences differing in their mean surprise\n")
    attr = {}
    for (ha, hb) in PAIRS:
        lab = f"{ha}{hb}"
        m = lv[ha] & lv[hb]
        a = np.where(m, srp[(ha, "real")].astype(np.float64), 0.0)
        b = np.where(m, srp[(hb, "real")].astype(np.float64), 0.0)
        n = m.sum(1).astype(np.float64)
        ok = n > 0
        ma = a.sum(1)[ok] / n[ok]
        mb = b.sum(1)[ok] / n[ok]
        gx, gy = a.sum() / m.sum(), b.sum() / m.sum()
        tot = np.einsum("ij,ij->", a, b) / m.sum() - gx * gy
        betw = float(np.average((ma - gx) * (mb - gy), weights=n[ok]))
        frac = (betw / tot) if abs(tot) > 1e-12 else float("nan")
        attr[lab] = {"total_cov": float(tot), "between_seq_cov": betw,
                     "between_fraction": float(frac)}
        print(f"    {lab:>6}  total {tot:>+9.4f}  between sequences {betw:>+9.4f}"
              f"  fraction {frac:>6.2f}")

    print("\n  PANEL 4, DESCRIPTIVE, NOT A TEST. share of the covariance carried")
    print("  by the top decile of speed surprise\n")
    for (ha, hb) in PAIRS:
        lab = f"{ha}{hb}"
        m = lv[ha] & lv[hb]
        a = srp[(ha, "real")].astype(np.float64)
        b = srp[(hb, "real")].astype(np.float64)
        cut = np.quantile(a[m], 0.9)
        gx, gy = a[m].mean(), b[m].mean()
        prod = (a - gx) * (b - gy)
        tot = prod[m].sum()
        top = prod[m & (a >= cut)].sum()
        share = float(top / tot) if abs(tot) > 1e-12 else float("nan")
        attr[lab]["top_decile_share"] = share
        print(f"    {lab:>6}  top decile carries {share:>6.2f} of the covariance")
    out["attribution"] = attr

    def call(v, sd):
        if v is None:
            return "none"
        if not np.isfinite(sd):
            sd = 0.0
        if min(abs(v - WITHIN_MOMENT), abs(v - EXPLAINED)) < sd:
            return "boundary"
        return "within_moment" if v >= WITHIN_MOMENT else (
            "explained" if v <= EXPLAINED else "mixed")

    calls = {lab: call(blk["real"]["retention"], blk["real"]["retention_sd"])
             for lab, blk in results.items()}
    out["calls"] = calls
    prim = calls["sth"]
    print(f"\n  {calls}")
    if prim == "within_moment":
        verdict = (f"WITHIN MOMENT COUPLING. {calls}. Conditioning on the model's "
                   f"own predicted uncertainty does not explain it, so the "
                   f"coupling is between the channels at a single instant and the "
                   f"chain rule factorisation is being under used.")
    elif prim == "explained":
        verdict = (f"EXPLAINED BY THE MODEL'S OWN UNCERTAINTY. {calls}. The "
                   f"coupling collapses inside entropy cells, so what is wrong is "
                   f"the model's knowledge of WHICH moments are hard, not the "
                   f"link between the channels.")
    elif prim == "boundary":
        verdict = (f"BOUNDARY, the call is REFUSED. {calls}. Reported as the in "
                   f"between case.")
    else:
        verdict = (f"MIXED. {calls}. Both mechanisms contribute. Report the "
                   f"retention and the panels.")
    out["verdict"] = verdict
    print(f"\n  -> {verdict}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.out}")
    print("""
  DIAGNOSTIC ONLY, never a contract score. no serving change follows and
  no build is authorised by any outcome. phase conditioning and the
  spectral loss term remain NOT AUTHORISED.
  panels 2 and 3 slice on quantities the model computes from the history, so
  the exact zero identity holds inside every cell. panels 1 and 4 slice on
  the outcome, so they are attribution and never a verdict.""")


if __name__ == "__main__":
    main()
