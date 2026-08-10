"""The exchange rate for a TARGETED correction, measured instead of assumed.

PRE REGISTERED in HANDOFF.md 2026-08-06, thresholds fixed before this file
existed. See "## The exchange rate for a TARGETED correction".

WHY. `w4_arcurve`'s 0.1904 AUC per nat was withdrawn: it priced GENERIC
likelihood improvement along one training trajectory, where the loss falls
everywhere at once, and it was then used to price a TARGETED correction. Two
conclusions rested on it. One has since been settled by measurement. The other,
that closing the gap costs 0.65 to 0.80 nats and therefore three to four orders
of magnitude more capacity, is UNPRICED.

WHAT IS FREE AND HAS NEVER BEEN READ. The constant temperature arm is a targeted
intervention with a measured AUC effect of +0.0192 and a likelihood cost that is
a closed form function of logits already computed. That is one point on the
targeted exchange rate curve. This file reads it, and reads the whole curve
around it so the sweep has something to hang its five AUC numbers on.

THE CURVE. Direction head mean NLL at FIXED inverse temperature beta, held out by
SEQUENCE, over a dense grid. No fitting is needed for the primary quantity. The
grid contains the five settings the sweep would use, so the sweep becomes a set
of paired (nats, AUC) points rather than five AUC numbers separated by less than
their own noise.

THE CONFOUND, named in the registration. Every NLL here is TEACHER FORCED and
every AUC is FREE RUNNING. The two need not share an optimum and the gap between
them is exposure bias. This instrument bounds the disagreement, it does not
attribute it.

Same corpus, same checkpoint, same split seeds as `w4_price.py`, so the numbers
compose with what is already recorded.

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

# The AUC winning constant, from the registered control that closed the table.
# Direction head temperature 0.928049, which is this inverse temperature.
BETA_AUC_ARM = 1.077529
# The five settings the sweep would use, as inverse temperatures. Named here so
# the grid cannot be chosen after the fact.
SWEEP_TH_TEMPS = (1.00, 0.96, 0.928049, 0.90, 0.85)
# What the const arm bought on the contract scorer, mean of three paired seeds.
ARM_DELTA_AUC = 0.0192
ARM_DELTA_AUC_SEM = 0.0081
# Median events per generated trajectory, from the arms themselves.
EVENTS_PER_TRAJ = 39.0
# The withdrawn generic rate, quoted only for the comparison it loses.
GENERIC_AUC_PER_NAT_TOKEN = 0.1904
TOKENS_PER_TRAJ = 113.0
# Observed contract AUC of the unmodified baseline, mean of three seeds.
BASE_AUC = 0.6612


def nll_at(z, k, beta):
    """Per position NLL of the true class under softmax(beta * z).

    z is (M, K) float64 logits already truncated to the real classes, so the
    softmax here is the renormalised one `w4_price` uses. k is (M,) int.
    """
    zb = z * beta
    zb -= zb.max(axis=1, keepdims=True)
    p = np.exp(zb)
    p /= p.sum(axis=1, keepdims=True)
    return -np.log(np.clip(p[np.arange(len(k)), k], 1e-30, None))


def cluster_sem(v, row):
    """Standard error clustered by SEQUENCE.

    Positions inside one trajectory are not independent, so the naive per
    position sem understates by a large factor. Average within sequence, then
    take the sem over sequences.
    """
    order = np.argsort(row, kind="stable")
    r = row[order]
    x = v[order]
    bnd = np.flatnonzero(np.diff(r)) + 1
    means = np.array([g.mean() for g in np.split(x, bnd)])
    return float(means.std(ddof=1) / np.sqrt(len(means))), len(means)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--cap", type=int, default=250000)
    ap.add_argument("--out", default="research/w4_beta_curve.json")
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

    live_full = np.zeros((B, MAX_T), dtype=bool)
    for i in range(B):
        live_full[i, :int(L[i])] = True
    live_th_full = live_full & (real_th < TH_NULL_CLASS)
    n_live = int(live_full.sum())
    n_dir = int(live_th_full.sum())
    frac_dir = n_dir / n_live
    take = min(args.cap, n_live)
    flat = np.flatnonzero(live_full.reshape(-1))
    sel_flat = np.sort(rng.choice(flat, take, replace=False))
    selmask = np.zeros(B * MAX_T, dtype=bool)
    selmask[sel_flat] = True
    selmask = selmask.reshape(B, MAX_T)
    print(f"  {n_live:,} live positions, {n_dir:,} of them carry a direction, "
          f"fraction {frac_dir:.4f}")
    print(f"  keeping logits at {take:,} positions\n", flush=True)

    W = N_REAL["th"]
    Z = np.zeros((take, W), dtype=np.float32)
    K = np.zeros(take, dtype=np.int64)
    ROW = np.zeros(take, dtype=np.int64)
    OKTH = np.zeros(take, dtype=bool)
    w = 0

    print("  one teacher forced forward pass, direction head only", flush=True)
    with torch.no_grad():
        for c0 in range(0, B, args.batch):
            sl = slice(c0, min(c0 + args.batch, B))
            s_b, th_b, dt_b = s_t[sl].to(dev), th_t[sl].to(dev), dt_t[sl].to(dev)
            cnd = cond_t[sl].to(dev)
            s_p, th_p, dt_p = model.shift_inputs(s_b, th_b, dt_b)
            st = prefix_state(s_b, th_b, dt_b, cnd)
            x = model.trunk(s_p, th_p, dt_p, st, cnd)
            lg = model.th_logits(x, s_b)
            m = torch.from_numpy(selmask[sl]).to(dev)
            cnt = int(m.sum())
            if cnt == 0:
                continue
            zz = lg[..., :W]
            kk = th_b.clamp(max=W - 1)
            Z[w:w + cnt] = zz[m].float().cpu().numpy()
            K[w:w + cnt] = kk[m].cpu().numpy()
            rows = np.broadcast_to(np.arange(c0, c0 + s_b.shape[0])[:, None],
                                   selmask[sl].shape)[selmask[sl]]
            ROW[w:w + cnt] = rows
            OKTH[w:w + cnt] = live_th_full[sl][selmask[sl]]
            w += cnt
    assert w == take, (w, take)

    # Same split rule and same rng state consumption order as w4_price, so the
    # fit and score halves are the identical partition and every number here
    # composes with the ones already recorded.
    fit_row = rng.random(B) < 0.5
    isfit = fit_row[ROW]
    fit_idx = np.flatnonzero(isfit & OKTH)
    sc_idx = np.flatnonzero(~isfit & OKTH)
    print(f"  direction positions: fit half {len(fit_idx):,}, "
          f"score half {len(sc_idx):,}\n", flush=True)

    zf = Z[fit_idx].astype(np.float64)
    kf = K[fit_idx]
    zs = Z[sc_idx].astype(np.float64)
    ks = K[sc_idx]
    rs = ROW[sc_idx]

    # Dense grid, plus the five sweep settings exactly. Sorted and deduplicated.
    grid = sorted(set([round(b, 6) for b in np.arange(0.90, 1.4001, 0.01)]
                      + [round(1.0 / t, 6) for t in SWEEP_TH_TEMPS]))

    base_pos = nll_at(zs, ks, 1.0)
    base = float(base_pos.mean())
    print("  BASELINE. direction head mean NLL at beta = 1, score half")
    print(f"    {base:.4f} nats over {len(ks):,} positions\n", flush=True)

    print("  the curve. gain is base minus NLL(beta), positive means the")
    print("  sharpening makes the model BETTER by its own held out likelihood\n")
    print(f"    {'beta':>8}{'th_temp':>9}{'nll':>10}{'gain':>10}{'sem':>9}")
    curve = []
    for b in grid:
        v = nll_at(zs, ks, b)
        d = base_pos - v
        sem, nclust = cluster_sem(d, rs)
        rec = {"beta": b, "th_temp": round(1.0 / b, 6),
               "nll": float(v.mean()), "gain": float(d.mean()), "sem": sem}
        curve.append(rec)
        mark = ""
        if abs(b - BETA_AUC_ARM) < 1e-4:
            mark = "  <- the AUC winning arm"
        elif any(abs(b - 1.0 / t) < 1e-4 for t in SWEEP_TH_TEMPS):
            mark = "  <- sweep grid"
        print(f"    {b:>8.4f}{rec['th_temp']:>9.4f}{rec['nll']:>10.4f}"
              f"{rec['gain']:>+10.4f}{sem:>9.4f}{mark}")

    # beta*, fitted on the fit half only, then reported with its score half gain.
    fine = np.arange(0.90, 1.4001, 0.001)
    fit_nll = np.array([nll_at(zf, kf, b).mean() for b in fine])
    beta_star = float(fine[int(np.argmin(fit_nll))])
    v_star = nll_at(zs, ks, beta_star)
    d_star = base_pos - v_star
    sem_star, _ = cluster_sem(d_star, rs)
    gain_star = float(d_star.mean())
    # The score half's own argmin, reported as a consistency check only. It is
    # not the held out number and is not used for any verdict.
    sc_nll = np.array([nll_at(zs, ks, b).mean() for b in fine])
    beta_star_sc = float(fine[int(np.argmin(sc_nll))])

    # The crossover: the beta above 1 at which the gain returns to zero, so the
    # model is no better than leaving it alone. Linear interpolation on the fine
    # grid, evaluated on the score half.
    g = base - sc_nll
    cross = None
    for i in range(1, len(fine)):
        if fine[i] > beta_star and g[i] <= 0.0 < g[i - 1]:
            t = g[i - 1] / (g[i - 1] - g[i])
            cross = float(fine[i - 1] + t * (fine[i] - fine[i - 1]))
            break

    print(f"\n  beta*        {beta_star:.4f}  fitted on the fit half"
          f"   th_temp {1.0/beta_star:.4f}")
    print(f"    held out gain at beta*  {gain_star:+.6f} nats  sem {sem_star:.6f}")
    print(f"  beta* on the score half itself {beta_star_sc:.4f}, "
          f"consistency check only")
    print(f"  crossover    {cross if cross is None else round(cross, 4)}"
          "   above this the model is WORSE by its own held out likelihood")

    # The AUC winning arm, converted to trajectory level nats.
    v_arm = nll_at(zs, ks, BETA_AUC_ARM)
    d_arm = base_pos - v_arm
    sem_arm, _ = cluster_sem(d_arm, rs)
    gain_arm = float(d_arm.mean())
    per_traj = gain_arm * EVENTS_PER_TRAJ * frac_dir
    per_traj_sem = sem_arm * EVENTS_PER_TRAJ * frac_dir

    print(f"\n  THE AUC WINNING ARM, beta {BETA_AUC_ARM}, th_temp 0.928049")
    print(f"    held out gain          {gain_arm:+.6f} nats per direction "
          f"prediction  sem {sem_arm:.6f}")
    print(f"    per trajectory         {per_traj:+.6f} nats "
          f"({EVENTS_PER_TRAJ:.0f} events x {frac_dir:.4f} carrying a direction)")
    print(f"    it bought              {ARM_DELTA_AUC:+.4f} AUC "
          f"sem {ARM_DELTA_AUC_SEM:.4f}, three paired seeds")

    rate = ARM_DELTA_AUC / per_traj if per_traj > 0 else float("nan")
    generic_traj = GENERIC_AUC_PER_NAT_TOKEN / TOKENS_PER_TRAJ
    print(f"\n  TARGETED exchange rate   {rate:>10.4f} AUC per trajectory nat")
    print(f"  GENERIC, withdrawn       {generic_traj:>10.6f} AUC per trajectory "
          f"nat  ({GENERIC_AUC_PER_NAT_TOKEN} per token / {TOKENS_PER_TRAJ:.0f})")
    if per_traj > 0:
        print(f"  ratio                    {rate / generic_traj:>10.1f}x")

    # Pinsker floor, recomputed here so the comparison is in one place. TV is
    # approximately 2(AUC - 0.5) for a balanced optimal test and the random
    # forest is suboptimal, so this is a conservative LOWER bound on the
    # divergence the detector's own score requires.
    tv = 2.0 * (BASE_AUC - 0.5)
    kl_floor = 2.0 * tv * tv
    need_auc = BASE_AUC - 0.50
    implied = need_auc / rate if per_traj > 0 else float("nan")

    print(f"\n  THE HYPOTHESIS THE SWEEP EXISTS TO TEST. linear, one point,")
    print(f"  and the linearity is the assumption, not a result.")
    print(f"    observed baseline contract AUC        {BASE_AUC:.4f}")
    print(f"    implied total variation               {tv:.4f}")
    print(f"    Pinsker floor on trajectory KL        {kl_floor:.4f} nats")
    print(f"    nats implied by the targeted rate     {implied:.4f} nats")
    print(f"    ratio of the two                      "
          f"{implied / kl_floor:.2f}x")

    out = {
        "ckpt": args.ckpt, "n_rows": int(B), "n_kept": int(take),
        "seed": args.seed, "diagnostic_only": True,
        "pre_registered": "HANDOFF.md 2026-08-06",
        "n_live": n_live, "n_dir": n_dir, "frac_dir": frac_dir,
        "score_half_positions": int(len(ks)),
        "base_nll": base, "curve": curve,
        "beta_star": beta_star, "beta_star_score_half": beta_star_sc,
        "gain_at_beta_star": gain_star, "sem_at_beta_star": sem_star,
        "crossover": cross,
        "beta_auc_arm": BETA_AUC_ARM,
        "gain_at_auc_arm": gain_arm, "sem_at_auc_arm": sem_arm,
        "per_trajectory_nats_at_auc_arm": per_traj,
        "per_trajectory_nats_sem": per_traj_sem,
        "arm_delta_auc": ARM_DELTA_AUC, "arm_delta_auc_sem": ARM_DELTA_AUC_SEM,
        "targeted_auc_per_trajectory_nat": rate,
        "generic_auc_per_trajectory_nat": generic_traj,
        "pinsker_floor_nats": kl_floor,
        "implied_gap_nats_linear": implied,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.out}")
    print("""
  DIAGNOSTIC ONLY. every NLL here is TEACHER FORCED and every AUC it is compared
  against is FREE RUNNING, so an overshoot of beta* by the AUC winning arm is
  ambiguous between detector fitting and exposure bias and this file does not
  attribute it. no serving change follows. phase conditioning, the spectral loss
  term and the FiLM rewrite all remain NOT AUTHORISED.""")


if __name__ == "__main__":
    main()
