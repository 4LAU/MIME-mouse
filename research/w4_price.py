"""What is the w4_selfsurprise defect actually WORTH, in nats and then in AUC?

PRE REGISTERED in HANDOFF, thresholds fixed before this file existed.

THE IDENTITY THAT MAKES THIS FREE. With logits z and inverse temperature b,

    NLL(b) = -b z_k + logsumexp(b z)
    d NLL / d b at b = 1 = E_p[z] - z_k = srp

The surprise residual IS the gradient of the head's loss with respect to a
confidence correction. So the curve w4_selfsurprise measured is a gradient, and
the loss recoverable by correcting it is a one dimensional convex minimisation on
logits that are already computed. No training, no generation, one forward pass.

THE EXCHANGE RATE. `w4_arcurve` measured 0.1904 AUC per nat across eight snapshots
of this model on one ruler, r 0.953, residual sd 0.0131. `event_ar_v2_s40000` sits
at 0.6526 contract AUC and 4.4024 nats held out.

    >= 0.05 nats   MATERIAL, the fine tune is authorised
    <= 0.01 nats   NEGLIGIBLE, this line closes for AUC purposes
    otherwise      MIXED, the number goes to L

THREE FAMILIES, cheapest first, each a superset of the last.

    1  one global inverse temperature per head
    2  per decile inverse temperature indexed by the DRIVER's surprise, which
       is exactly the w4_selfsurprise curve
    3  per decile temperature AND per decile mix toward the head's own marginal

Family 3 exists because a temperature can only rescale. If the defect is partly a
SHAPE error and not a WIDTH error, family 2 underprices it, and if family 3 beats
family 2 by a wide margin then the pricing is wrong and must be redone richer.

Every parameter is fitted on one half of the SEQUENCES and scored on the other
half, so every gain quoted is a held out gain.

DIAGNOSTIC ONLY, never a contract score. No serving change follows.

Safety. Reads training/events_*.npy and one checkpoint. Touches no evaluation
data, no scoring code, and never training/candi_polar_flow_best.pt.
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
from research.w4_launch import N_REAL, renorm  # noqa: E402

MATERIAL_NATS = 0.05
NEGLIGIBLE_NATS = 0.01
AUC_PER_NAT = 0.1904          # w4_arcurve, 8 v2 snapshots, one ruler
AUC_PER_NAT_RESID_SD = 0.0131
BASE_AUC = 0.6526             # event_ar_v2_s40000, n 2000, seed 0, temp 1.0
N_BINS = 10
HEADS = ("s", "th", "dt")
# (driver, responder). The driver's surprise indexes the correction applied to
# the responder. Legal only where the responder's head already conditions on the
# driver's token, which is what makes E[srp_resp | srp_driver] = 0 exact.
SLICES = (("s", "th"), ("s", "dt"), ("th", "dt"))


def nll_of(z, k, beta, mix, marg):
    """Mean NLL of the true class under softmax(beta*z) mixed toward `marg`.

    z is (M, K) float64 logits, k is (M,) int, beta and mix are (M,) or scalar,
    marg is (K,) the head's own marginal. mix = 0 recovers pure temperature.
    """
    zb = z * np.asarray(beta).reshape(-1, 1)
    zb -= zb.max(axis=1, keepdims=True)
    p = np.exp(zb)
    p /= p.sum(axis=1, keepdims=True)
    if np.any(mix):
        p = (1.0 - np.asarray(mix).reshape(-1, 1)) * p + \
            np.asarray(mix).reshape(-1, 1) * marg.reshape(1, -1)
    return -np.log(np.clip(p[np.arange(len(k)), k], 1e-30, None))


def fit_1d(f, lo, hi, iters=60):
    """Golden section on a unimodal 1-D objective. NLL is convex in beta."""
    g = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--cap", type=int, default=250000,
                    help="positions kept per head for the logit side. the fit "
                         "has 20 parameters, so this is far past sufficient and "
                         "keeps 257 wide float64 logits inside memory.")
    ap.add_argument("--out", default="research/w4_price.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    rng = np.random.default_rng(args.seed + 77)

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
    print(f"  {B:,} rows at least 12 events\n", flush=True)

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

    # Which positions get their logits kept. Fixed up front so the pass writes
    # into a preallocated buffer instead of growing a list.
    live_full = np.zeros((B, MAX_T), dtype=bool)
    for i in range(B):
        live_full[i, :int(L[i])] = True
    live_th_full = live_full & (real_th < TH_NULL_CLASS)
    n_live = int(live_full.sum())
    take = min(args.cap, n_live)
    flat = np.flatnonzero(live_full.reshape(-1))
    sel_flat = np.sort(rng.choice(flat, take, replace=False))
    selmask = np.zeros(B * MAX_T, dtype=bool)
    selmask[sel_flat] = True
    selmask = selmask.reshape(B, MAX_T)
    print(f"  {n_live:,} live positions, keeping logits at {take:,} of them\n",
          flush=True)

    Z = {h: np.zeros((take, N_REAL[h]), dtype=np.float32) for h in HEADS}
    K = {h: np.zeros(take, dtype=np.int64) for h in HEADS}
    SRP = {h: np.zeros(take, dtype=np.float64) for h in HEADS}
    ROW = np.zeros(take, dtype=np.int64)
    OKTH = np.zeros(take, dtype=bool)
    w = 0

    print("  one teacher forced forward pass", flush=True)
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
            m = torch.from_numpy(selmask[sl]).to(dev)
            cnt = int(m.sum())
            if cnt == 0:
                continue
            for h in HEADS:
                zz = lg[h][..., :N_REAL[h]]
                p = renorm(torch.softmax(lg[h], -1), N_REAL[h])
                logp = torch.log(p.clamp(min=1e-30))
                H = -(p * logp).sum(-1)
                kk = true[h].clamp(max=N_REAL[h] - 1)
                srp = -logp.gather(-1, kk.unsqueeze(-1)).squeeze(-1) - H
                Z[h][w:w + cnt] = zz[m].float().cpu().numpy()
                K[h][w:w + cnt] = kk[m].cpu().numpy()
                SRP[h][w:w + cnt] = srp[m].double().cpu().numpy()
            rows = np.broadcast_to(np.arange(c0, c0 + s_b.shape[0])[:, None],
                                   selmask[sl].shape)[selmask[sl]]
            ROW[w:w + cnt] = rows
            OKTH[w:w + cnt] = live_th_full[sl][selmask[sl]]
            w += cnt
    assert w == take, (w, take)

    # Held out by SEQUENCE. Parameters are fitted on the fit half and every
    # number quoted is scored on the score half, so a gain is a real gain.
    fit_row = rng.random(B) < 0.5
    isfit = fit_row[ROW]
    print(f"  fit half {int(isfit.sum()):,} positions, "
          f"score half {int((~isfit).sum()):,}\n", flush=True)

    out = {"ckpt": args.ckpt, "n_rows": int(B), "n_kept": int(take),
           "seed": args.seed, "diagnostic_only": True,
           "pre_registered": "HANDOFF.md 2026-08-06",
           "exchange_rate_auc_per_nat": AUC_PER_NAT,
           "base_auc": BASE_AUC, "families": {}}

    print("  BASELINE. mean NLL of each head at b = 1, on the score half.")
    print("  this is what there is to win from, and every gain below is a")
    print("  fraction of it\n")
    base = {}
    for h in HEADS:
        ok = ~isfit if h != "th" else (~isfit & OKTH)
        z = Z[h][ok].astype(np.float64)
        base[h] = float(nll_of(z, K[h][ok], 1.0, 0.0, None).mean())
        print(f"    {h:>3}  {base[h]:8.4f} nats over {int(ok.sum()):,} positions")
        del z
    out["baseline_nll"] = base

    def gain_for(h, idx_fit, idx_sc, nb, fam):
        """Fit per bin parameters on the fit half, score on the score half.

        fam 2 is temperature only. fam 3 adds a mix toward the head's own GLOBAL
        marginal. fam 4 mixes toward the marginal OF THAT BIN, which is the
        richest of the three: it can move probability toward whatever directions
        actually occur at surprising moments, not merely widen. A temperature
        rescales, fam 4 can also re rank.
        """
        zf, kf = Z[h][idx_fit].astype(np.float64), K[h][idx_fit]
        zs, ks = Z[h][idx_sc].astype(np.float64), K[h][idx_sc]
        bf = BIN[idx_fit]
        bs = BIN[idx_sc]
        gmarg = np.bincount(kf, minlength=N_REAL[h]).astype(np.float64)
        gmarg = np.clip(gmarg / gmarg.sum(), 1e-12, None)
        beta = np.ones(nb)
        mix = np.zeros(nb)
        # Per bin target the mixture leans toward. Fitted on the fit half only,
        # so leaning on it is scored honestly on the other half.
        targ = np.repeat(gmarg.reshape(1, -1), nb, axis=0)
        if fam == 4:
            for b in range(nb):
                mf = bf == b
                if mf.sum() >= 500:
                    t = np.bincount(kf[mf], minlength=N_REAL[h]).astype(np.float64)
                    targ[b] = np.clip(t / t.sum(), 1e-12, None)
        for b in range(nb):
            mf = bf == b
            if mf.sum() < 500:
                continue
            zz, kk, tg = zf[mf], kf[mf], targ[b]
            beta[b] = fit_1d(
                lambda t: nll_of(zz, kk, t, 0.0, tg).mean(), 0.3, 3.0)
            if fam >= 3:
                mix[b] = fit_1d(
                    lambda u: nll_of(zz, kk, beta[b], np.full(len(kk), u),
                                     tg).mean(), 0.0, 0.9)
        nll = 0.0
        for b in range(nb):
            ms = bs == b
            if not ms.any():
                continue
            nll += nll_of(zs[ms], ks[ms], beta[b],
                          np.full(int(ms.sum()), mix[b]), targ[b]).sum()
        return float(nll / len(ks)), beta, mix

    results = {}
    for (dr, resp) in SLICES:
        lab = f"{dr}->{resp}"
        ok_all = OKTH if resp == "th" or dr == "th" else np.ones(take, bool)
        idx_fit = np.flatnonzero(isfit & ok_all)
        idx_sc = np.flatnonzero(~isfit & ok_all)
        d = SRP[dr]
        edges = np.quantile(d[idx_fit], np.linspace(0, 1, N_BINS + 1)[1:-1])
        BIN = np.digitize(d, edges).astype(np.int64)
        # Bin CENTRES in nats, recorded so the fitted table can be applied by
        # srp VALUE rather than by rank. At generation the model samples its own
        # speeds and its own surprise distribution is narrower than the real
        # one, so a rank indexed table would apply the wrong correction.
        centres = [float(np.median(d[idx_fit][BIN[idx_fit] == b]))
                   if (BIN[idx_fit] == b).sum() >= 500 else None
                   for b in range(N_BINS)]
        b0 = float(nll_of(Z[resp][idx_sc].astype(np.float64), K[resp][idx_sc],
                          1.0, 0.0, None).mean())
        # Family 1 is a single global temperature, so it is the one bin case.
        BIN_SAVE = BIN
        BIN = np.zeros(take, dtype=np.int64)
        n1, be1, _ = gain_for(resp, idx_fit, idx_sc, 1, 2)
        BIN = BIN_SAVE
        n2, be2, _ = gain_for(resp, idx_fit, idx_sc, N_BINS, 2)
        n3, be3, mi3 = gain_for(resp, idx_fit, idx_sc, N_BINS, 3)
        n4, be4, mi4 = gain_for(resp, idx_fit, idx_sc, N_BINS, 4)
        # The ceiling uses the RICHEST family, so the verdict is never made on
        # an underpriced correction.
        best = min(n2, n3, n4)
        results[lab] = {
            "base_nll": b0,
            "srp_bin_edges": edges.tolist(), "srp_bin_centres": centres,
            "fam1_nll": n1, "fam1_gain": b0 - n1, "fam1_beta": float(be1[0]),
            "fam2_nll": n2, "fam2_gain": b0 - n2, "fam2_beta": be2.tolist(),
            "fam3_nll": n3, "fam3_gain": b0 - n3, "fam3_beta": be3.tolist(),
            "fam3_mix": mi3.tolist(),
            "fam4_nll": n4, "fam4_gain": b0 - n4, "fam4_beta": be4.tolist(),
            "fam4_mix": mi4.tolist(),
            "gain_over_global": (b0 - best) - (b0 - n1),
        }
    out["families"] = results

    print("\n  WHAT THE CORRECTION IS WORTH, in nats, held out\n")
    print(f"    {'slice':>8} {'base':>8} {'fam1':>8} {'fam2':>8} {'fam3':>8} "
          f"{'fam4':>8} {'best-f1':>8}")
    for lab, r in results.items():
        print(f"    {lab:>8} {r['base_nll']:>8.4f} {r['fam1_gain']:>+8.4f} "
              f"{r['fam2_gain']:>+8.4f} {r['fam3_gain']:>+8.4f} "
              f"{r['fam4_gain']:>+8.4f} {r['gain_over_global']:>+8.4f}")
    print("\n  fam1 is one global temperature and is NOT the defect, it is the")
    print("  uniform over confidence w4_launch and w4_condtex already found.")
    print("  the defect w4_selfsurprise found is the last column, the part a")
    print("  single global temperature cannot reach.")
    print("  fam4 leans toward the class distribution OF THAT BIN, so it can")
    print("  re rank and not merely widen. if fam4 barely beats fam2 then the")
    print("  defect really is a width error and the pricing is not too tight\n")

    print("  the fitted per decile inverse temperature. above 1 means the head")
    print("  should be SHARPER there, below 1 means it should be WIDER\n")
    print(f"    {'slice':>8} " + " ".join(f"{'d'+str(i):>6}" for i in range(N_BINS)))
    for lab, r in results.items():
        print(f"    {lab:>8} " + " ".join(f"{b:>6.3f}" for b in r["fam2_beta"]))

    # The primary is the speed to direction slice, the one w4_selfsurprise read
    # its verdict from, and the quantity is the part a global temperature cannot
    # already reach.
    prim = results["s->th"]["gain_over_global"]
    tot = sum(r["gain_over_global"] for r in results.values())
    pa_prim = AUC_PER_NAT * prim
    pa_tot = AUC_PER_NAT * tot
    print(f"\n  PRIMARY, s->th beyond a global temperature: {prim:+.4f} nats")
    print(f"  all three slices summed, an optimistic ceiling:  {tot:+.4f} nats")
    print(f"\n  at the measured {AUC_PER_NAT} AUC per nat, resid sd "
          f"{AUC_PER_NAT_RESID_SD}")
    print(f"    predicted AUC gain, primary  {pa_prim:.4f}   "
          f"{BASE_AUC:.4f} -> {BASE_AUC - pa_prim:.4f}")
    print(f"    predicted AUC gain, ceiling  {pa_tot:.4f}   "
          f"{BASE_AUC:.4f} -> {BASE_AUC - pa_tot:.4f}")

    if tot >= MATERIAL_NATS:
        verdict = (f"MATERIAL. the correction is worth {tot:.4f} nats beyond a "
                   f"global temperature, {AUC_PER_NAT * tot:.4f} AUC at the "
                   f"measured exchange rate. the fine tune is authorised.")
    elif tot <= NEGLIGIBLE_NATS:
        verdict = (f"NEGLIGIBLE. the correction is worth {tot:.4f} nats beyond a "
                   f"global temperature, {AUC_PER_NAT * tot:.4f} AUC at the "
                   f"measured exchange rate, against a 0.15 gap to close. the "
                   f"defect is real, correctly measured and worth nothing at the "
                   f"scorer. this line CLOSES for AUC purposes with no training "
                   f"step spent.")
    else:
        verdict = (f"MIXED. {tot:.4f} nats beyond a global temperature, "
                   f"{AUC_PER_NAT * tot:.4f} AUC. the decision goes to L with "
                   f"the number attached.")
    out["verdict"] = verdict
    out["primary_nats"] = prim
    out["ceiling_nats"] = tot
    print(f"\n  -> {verdict}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.out}")
    print("""
  DIAGNOSTIC ONLY, never a contract score. no serving change follows and the
  locked serving recipe does not move. this prices a CORRECTION and cannot
  authorise a MECHANISM, so phase conditioning, the spectral loss term and the
  FiLM rewrite all remain NOT AUTHORISED whatever the number says.""")


if __name__ == "__main__":
    main()
