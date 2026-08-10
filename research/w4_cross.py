"""Are the three channels wrong TOGETHER?

PRE REGISTERED in HANDOFF, thresholds fixed before this file existed.

Every instrument so far is per head. w4_launch, w4_condtex, w4_progress and
w4_indep all look at speed, direction and timing one at a time, and all four come
back clean. None of them can see whether the three channels are wrong TOGETHER,
and the contract scorer's features are mostly about exactly that: curvature is
direction against speed, the velocity profile is speed against time. A model can be
calibrated on each channel separately and still couple them wrongly.

The identity is the one that made w4_indep exact. The model emits s, then th given
s, then dt given s and th. So given the history at step t:

    r_s(t)   is a function of the history and s(t)
    r_th(t)  has conditional mean zero given the history and s(t)
    r_dt(t)  has conditional mean zero given the history, s(t) and th(t)

Each residual is centred on a conditional the earlier ones are already measurable
with respect to, so every cross channel correlation is EXACTLY zero under a correct
model, at lag zero and at every lag in both directions.

Three residual kinds per head:

    lvl(t) = m(t) - 1/2                   signed surprise, m is the mid PIT
    vol(t) = |m(t) - 1/2| - E|m - 1/2|    surprise magnitude
    srp(t) = -log p(k(t)) - H(p(t))       surprise, ORDERING FREE

srp is the primary. lvl and vol depend on where the alphabet is cut, which is
arbitrary for the circular direction head, so they are profile only.

A distinction that has to be stated, given the history in this file. srp is built
from the same two quantities as the WITHDRAWN KLhat estimator, negative log
probability and entropy. KLhat was withdrawn because their MEANS do not form a
divergence: the difference is a KL plus an entropy gap, so it is not non negative
and goes negative exactly when the model is too wide. None of that is used here.
srp is used only as a conditionally mean zero residual inside a CORRELATION, and
the property relied on is that E[srp | history] is exactly zero, which is true and
was never the disputed part. Its mean is not interpreted as a distance.

    any |rho| >= 0.05    MATERIAL cross channel dependence
    all |rho| <= 0.01    the channels are not coupled wrongly at lag zero
    otherwise            MIXED
    BOUNDARY             within one bootstrap sd of a threshold the call is REFUSED

Same numbers w4_indep used, on the same correlation scale, so the two readings are
comparable and neither is fitted to what was observed. Bootstrap over whole
SEQUENCES.

VALIDITY ARM. Tokens drawn from the model at each step, each head from its own
conditional given the REAL preceding tokens. Given the history and the real earlier
tokens the drawn heads are independent, so every cross correlation here is an exact
zero and any deviation is estimator error. It is a weaker null than the real arm in
one respect worth stating: it does not reproduce the real arm's chain, because the
drawn token is not what the next head conditions on. It is an exact null for the
ESTIMATOR, which is what a validity arm is for.

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
from itertools import combinations

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

MATERIAL = 0.05
NEGLIGIBLE = 0.01
VALID_SD = 2.0          # single test, at lag 0, the lag the verdict uses
VALID_FAMILY_SD = 3.0   # the family
VALID_FLOOR = NEGLIGIBLE / 5    # 0.002, the magnitude floor, see HANDOFF
LAGS = (1, 2, 3, 4)
KINDS = ("srp", "vol", "lvl")
HEADS = ("s", "th", "dt")
PAIRS = tuple(combinations(HEADS, 2))


def x_terms(a, b, live_a, live_b, lag):
    """Sums for a lag k CROSS correlation between two series, WITHIN sequences.

    lag 0 pairs a position with itself. Positive lag pairs a(t) with b(t+lag).
    Returns per row sums so a bootstrap over sequences never re walks positions.
    """
    if lag == 0:
        x, y = a, b
        m = live_a & live_b
    else:
        x, y = a[:, :-lag], b[:, lag:]
        m = live_a[:, :-lag] & live_b[:, lag:]
    x = np.where(m, x.astype(np.float64), 0.0)
    y = np.where(m, y.astype(np.float64), 0.0)
    return (np.einsum("ij,ij->i", x, y), x.sum(1), y.sum(1),
            np.einsum("ij,ij->i", x, x), np.einsum("ij,ij->i", y, y),
            m.sum(1).astype(np.float64))


def corr_from(terms, rows):
    """Pearson correlation of the paired positions on a set of rows."""
    sxy, sx, sy, sxx, syy, n = (t[rows].sum() for t in terms)
    if n < 100:
        return None
    cov = sxy / n - (sx / n) * (sy / n)
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
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--out", default="research/w4_cross.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    rng = np.random.default_rng(args.seed + 23)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed + 149)

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
    print(f"  {B:,} rows at least 12 events, the same rows w4_indep used\n",
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

    # Residuals are stored float32 and cast to float64 inside the term sums. Eighteen
    # full arrays at float64 would be over three gigabytes, and the accumulation is
    # what needs the precision, not the storage.
    res = {(h, k, arm): np.zeros((B, MAX_T), dtype=np.float32)
           for h in HEADS for k in KINDS for arm in ("real", "self")}
    live = np.zeros((B, MAX_T), dtype=bool)
    live_th = np.zeros((B, MAX_T), dtype=bool)

    print("  one teacher forced forward pass, identical to w4_indep's",
          flush=True)
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
                cdf = torch.cumsum(p, dim=-1)
                m_all = cdf - 0.5 * p
                # Both centrings in closed form from p at this step. This is what
                # makes each residual exactly conditionally mean zero.
                mu = (p * (m_all - 0.5).abs()).sum(-1)
                logp = torch.log(p.clamp(min=1e-30))
                ent = -(p * logp).sum(-1)
                k_self = self_sample(p, gen)

                def resid(k):
                    k = k.clamp(max=p.shape[-1] - 1)
                    kk = k.unsqueeze(-1)
                    m = m_all.gather(-1, kk).squeeze(-1) - 0.5
                    srp = -logp.gather(-1, kk).squeeze(-1) - ent
                    return (m.float().cpu().numpy(),
                            (m.abs() - mu).float().cpu().numpy(),
                            srp.float().cpu().numpy())

                for arm, k in (("real", true[h]), ("self", k_self)):
                    a, b, c = resid(k)
                    res[(h, "lvl", arm)][sl] = a
                    res[(h, "vol", arm)][sl] = b
                    res[(h, "srp", arm)][sl] = c
            live[sl] = (s_b < S_PAD_CLASS).cpu().numpy()
            live_th[sl] = ((s_b < S_PAD_CLASS) & (th_b < TH_NULL_CLASS)
                           ).cpu().numpy()

    print(f"  {live.sum():,} live positions, {live_th.sum():,} of them with a "
          f"direction to predict\n", flush=True)

    lv = {"s": live, "th": live_th, "dt": live}
    all_rows = np.arange(B)
    ALL_LAGS = (0,) + LAGS + tuple(-k for k in LAGS)

    # Negative lag means the SECOND head leads, which is a different question from
    # the first head leading and is not the same number, so both directions are
    # measured rather than assumed symmetric.
    terms = {}
    for (ha, hb) in PAIRS:
        for kind in KINDS:
            for lag in ALL_LAGS:
                a, b = (ha, hb) if lag >= 0 else (hb, ha)
                for arm in ("real", "self"):
                    terms[((ha, hb), kind, arm, lag)] = x_terms(
                        res[(a, kind, arm)], res[(b, kind, arm)],
                        lv[a], lv[b], abs(lag))

    def boot_sd(key, draws):
        v = []
        for _ in range(draws):
            rs = rng.integers(0, B, B)
            r = corr_from(terms[key], rs)
            if r is not None:
                v.append(r)
        # An unmeasurable sd must FAIL the gate rather than disable it, the defect
        # w4_indep's first smoke test found.
        return float(np.std(v)) if len(v) > 10 else float("nan")

    out = {"ckpt": args.ckpt, "n_rows": int(B), "seed": args.seed,
           "diagnostic_only": True, "pre_registered": "HANDOFF.md 2026-08-05",
           "thresholds": {"material": MATERIAL, "negligible": NEGLIGIBLE},
           "lags": list(ALL_LAGS), "n_live": int(live.sum()),
           "validity": {}, "real": {}}

    def lagcols():
        return " ".join(f"{('lag' + str(k)):>8}" for k in ALL_LAGS)

    print("  VALIDITY arm. tokens drawn from the model, each head from its own")
    print("  conditional given the REAL earlier tokens, so the drawn heads are")
    print("  independent and every cross correlation here is an exact zero\n")
    print(f"    {'pair kind':>14} " + lagcols())
    vfail = []
    for (ha, hb) in PAIRS:
        for kind in KINDS:
            row, worst, z0 = [], 0.0, float("inf")
            for lag in ALL_LAGS:
                key = ((ha, hb), kind, "self", lag)
                r = corr_from(terms[key], all_rows)
                sd = boot_sd(key, max(40, args.draws // 8))
                row.append({"lag": lag, "rho": r, "sd": sd})
                z = (float("inf") if (r is None or not np.isfinite(sd) or sd <= 0)
                     else (0.0 if abs(r) < VALID_FLOOR else abs(r) / sd))
                worst = max(worst, z)
                if lag == 0:
                    z0 = z
            lab = f"{ha}{hb} {kind}"
            out["validity"][lab] = row
            print(f"    {lab:>14} " +
                  " ".join(f"{d['rho']:>+8.4f}" if d["rho"] is not None
                           else f"{'nan':>8}" for d in row) +
                  f"   lag0 {z0:.1f} sd, worst {worst:.1f} sd")
            if not (z0 <= VALID_SD and worst <= VALID_FAMILY_SD):
                vfail.append(lab)
    if vfail:
        print(f"\n  VALIDITY ARM FAILED on {', '.join(vfail)}. the estimator "
              f"manufactures coupling, so nothing below would mean anything.")
        out["verdict"] = f"FAILED, validity arm coupled on {','.join(vfail)}"
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        return
    print("\n  validity arm PASSES at every lag, the statistic is sound\n")

    print("  REAL tokens. cross channel correlation of the residuals. under a")
    print("  correct model every entry is EXACTLY zero, at lag zero and at every")
    print("  lag in both directions. negative lag means the second head leads\n")
    print(f"    {'pair kind':>14} " + lagcols())
    for (ha, hb) in PAIRS:
        for kind in KINDS:
            row = [{"lag": lag,
                    "rho": corr_from(terms[((ha, hb), kind, "real", lag)],
                                     all_rows)}
                   for lag in ALL_LAGS]
            lab = f"{ha}{hb} {kind}"
            out["real"][lab] = row
            print(f"    {lab:>14} " +
                  " ".join(f"{d['rho']:>+8.4f}" if d["rho"] is not None
                           else f"{'nan':>8}" for d in row))

    prim = {}
    for (ha, hb) in PAIRS:
        key = ((ha, hb), "srp", "real", 0)
        r = corr_from(terms[key], all_rows)
        sd = boot_sd(key, args.draws)
        prim[f"{ha}{hb}"] = {"rho": r, "bootstrap_sd": sd}
    out["primary_srp_lag0"] = prim
    print("\n  PRIMARY. surprise, lag zero, the three channel pairs")
    for k, d in prim.items():
        print(f"    {k:>6}  {d['rho']:+.4f}  bootstrap sd {d['bootstrap_sd']:.4f}")

    def call(v, sd):
        if v is None:
            return "none"
        a = abs(v)
        if min(abs(a - MATERIAL), abs(a - NEGLIGIBLE)) < sd:
            return "boundary"
        return "material" if a >= MATERIAL else (
            "negligible" if a <= NEGLIGIBLE else "mixed")

    calls = {k: call(d["rho"], d["bootstrap_sd"]) for k, d in prim.items()}
    out["calls"] = calls
    print(f"\n  {calls}")
    if any(v == "material" for v in calls.values()):
        verdict = (f"MATERIAL CROSS CHANNEL DEPENDENCE. {calls}. Under a correct "
                   f"model these are exactly zero, so the model gets the three "
                   f"channels wrong together in a way no per head instrument "
                   f"could see. This is the coupling the contract features are "
                   f"built out of.")
    elif all(v == "negligible" for v in calls.values()):
        verdict = (f"NO CROSS CHANNEL DEPENDENCE AT LAG ZERO. {calls}. The three "
                   f"channels are not coupled wrongly, so the defect is not in "
                   f"how speed, direction and timing move together either.")
    elif any(v == "boundary" for v in calls.values()):
        verdict = (f"BOUNDARY, the call is REFUSED. {calls}. Reported as the in "
                   f"between case.")
    else:
        verdict = f"MIXED. {calls}. Report the lag profile and the numbers."
    out["verdict"] = verdict
    print(f"\n  -> {verdict}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.out}")
    print("""
  DIAGNOSTIC ONLY, never a contract score. no serving change follows and
  no build is authorised by any outcome. phase conditioning and the
  spectral loss term remain NOT AUTHORISED.
  srp is used only as a conditionally mean zero residual inside a
  correlation. its MEAN is not a divergence and is not interpreted as one,
  which is the error that withdrew KLhat.""")


if __name__ == "__main__":
    main()
