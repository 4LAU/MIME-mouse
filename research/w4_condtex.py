"""Does the model CONDITION its texture on the history, or regress to the mean?

PRE REGISTERED in HANDOFF, thresholds fixed before this file existed.

w4_launch found every one step conditional calibrated. By the chain rule, if the
conditionals were exactly right the joint would be right too, so the conditionals
must be calibrated MARGINALLY and wrong CONDITIONALLY. A randomised PIT pools over
every history at a position, and a model can be uniform there while being wrong
after every individual history, as long as the errors cancel in the pool. Too
narrow after volatile histories and too wide after calm ones averages to uniform.
That failure is regression to the mean, and it is what under modulation IS.

So the PIT is re sliced, not by position but by a property of the REAL HISTORY:
the local texture of the previous LOOKBACK events, measured as the mean absolute
successive difference of the speed class. A correct conditional is uniform inside
EVERY slice of ANY function of the history.

The statistic is SHAPE, not tilt. Regression to the mean is a width error whose
direction depends on which way the truth happened to move, so it cancels in the
PIT mean and does not cancel in the width. After a volatile history a mean
regressing model is too NARROW and its PIT is a U with shape above 1. After a calm
history it is too WIDE and its PIT is a hump with shape below 1.

    D = shape(top quintile) - shape(bottom quintile), speed head

    D >= 0.06    REGRESSION TO THE MEAN, addressable in training
    D <= 0.02    the model DOES condition its texture properly
    in between   MIXED
    BOUNDARY     within one bootstrap sd of a threshold the call is REFUSED

Position is confounded, since volatile histories happen later in a movement and
shape drifts mildly with position. D is therefore computed WITHIN each position
band and pooled, and the quintile edges are taken WITHIN each band, so a slope
that is really position in disguise cannot survive.

VALIDITY ARM, and here it is a real one. The same quintile slicing applied to
tokens drawn from the model's own predictive law. The covariate comes from the
real history and is IDENTICAL between the two arms, so an estimator that
manufactures a slope manufactures it there too. If the validity arm is not flat
the run is reported as failed rather than interpreted.

Caveat, stated rather than buried. Uniformity inside every slice is a NECESSARY
condition and not a sufficient one. A flat D proves the model conditions on THIS
function of the history, not on every function of it.

DIAGNOSTIC ONLY, never a contract score. No serving change follows and no build is
authorised by either outcome. Phase conditioning and the spectral loss term remain
NOT AUTHORISED.

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
from research.w4_launch import (  # noqa: E402
    N_REAL, pit_of, pit_shape, renorm, self_sample,
)

REGRESSES = 0.06
CONDITIONS = 0.02
# The validity arm is flat by construction, so the criterion is "flat within
# noise" and the noise is measured rather than assumed. A fixed constant was tried
# first and is wrong: the shape statistic reads four histogram bins at each end
# against four in the middle, so its own sd is near 0.03 at a few thousand samples
# per cell, and any fixed bound sits below the noise floor at small n. See the
# AMENDMENT in HANDOFF. Chosen from the validity arm alone, with the real token
# side never printed, so it cannot leak the answer.
VALID_SD = 2.0
LOOKBACK = 8
N_QUINT = 5
BANDS = [(8, 12), (12, 20), (20, 32), (32, MAX_T)]
MIN_CELL = 200          # pit_shape refuses below this anyway


def texture(s_cls, live):
    """Mean absolute successive difference of the speed class over the previous
    LOOKBACK events. Defined only where the whole window is live.

    The speed alphabet is ordinal in magnitude, so a successive difference is a
    texture and not an arbitrary label distance.
    """
    d = np.abs(np.diff(s_cls.astype(np.float64), axis=1))         # (B, T-1)
    ok = live[:, 1:] & live[:, :-1]
    d = np.where(ok, d, 0.0)
    cs = np.concatenate([np.zeros((len(d), 1)), np.cumsum(d, 1)], 1)
    ck = np.concatenate([np.zeros((len(d), 1)), np.cumsum(ok, 1)], 1)

    cov = np.full(s_cls.shape, np.nan)
    for t in range(LOOKBACK, s_cls.shape[1]):
        # differences at indices t-LOOKBACK .. t-2, which is LOOKBACK-1 of them
        lo, hi = t - LOOKBACK, t - 1
        n = ck[:, hi] - ck[:, lo]
        full = n == (LOOKBACK - 1)
        cov[full, t] = (cs[full, hi] - cs[full, lo]) / (LOOKBACK - 1)
    return cov


def band_D(u, cov, edges, rows_mask):
    """shape(top quintile) minus shape(bottom quintile) inside one position band.

    Returns (D, per quintile shapes, per quintile counts).
    """
    q = np.digitize(cov[rows_mask], edges)
    uu = u[rows_mask]
    sh, cnt = [], []
    for k in range(N_QUINT):
        m = q == k
        cnt.append(int(m.sum()))
        sh.append(pit_shape(uu[m]) if m.sum() >= MIN_CELL else None)
    D = (sh[-1] - sh[0]) if (sh[-1] is not None and sh[0] is not None) else None
    return D, sh, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--out", default="research/w4_condtex.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    rng = np.random.default_rng(args.seed + 5)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed + 91)

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
    print(f"  {B:,} rows at least 12 events, the same rows w4_launch used\n",
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

    heads = ("s", "th", "dt")
    pit = {h: np.zeros((B, MAX_T), dtype=np.float64) for h in heads}
    vpit = {h: np.zeros((B, MAX_T), dtype=np.float64) for h in heads}
    live = np.zeros((B, MAX_T), dtype=bool)
    live_th = np.zeros((B, MAX_T), dtype=bool)

    print("  one teacher forced forward pass, identical to w4_launch's",
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
            for h in heads:
                p = renorm(torch.softmax(lg[h], -1), N_REAL[h])
                pit[h][sl] = pit_of(p, true[h], rng, dev).double().cpu().numpy()
                vpit[h][sl] = pit_of(p, self_sample(p, gen), rng,
                                     dev).double().cpu().numpy()
            live[sl] = (s_b < S_PAD_CLASS).cpu().numpy()
            live_th[sl] = ((s_b < S_PAD_CLASS) & (th_b < TH_NULL_CLASS)
                           ).cpu().numpy()

    cov = texture(real_s, live)
    have = live & np.isfinite(cov)
    print(f"  {live.sum():,} live positions, {have.sum():,} of them with a "
          f"complete {LOOKBACK} event history behind them\n", flush=True)

    # Quintile edges are taken WITHIN each band and FIXED at the full sample, so
    # a bootstrap draw does not also resample the binning.
    band_mask, band_edges = [], []
    for lo, hi in BANDS:
        m = np.zeros_like(have)
        m[:, lo:hi] = have[:, lo:hi]
        band_mask.append(m)
        band_edges.append(np.quantile(cov[m],
                                      [k / N_QUINT for k in range(1, N_QUINT)]))

    def pooled_D(store, head, rows):
        """D per band on a row set, and the count weighted pool across bands."""
        lv = live_th if head == "th" else live
        U, CV, LV = store[head][rows], cov[rows], lv[rows]
        per, num, den = [], 0.0, 0.0
        for bi, m in enumerate(band_mask):
            mm = m[rows] & LV
            d, sh, cnt = band_D(U, CV, band_edges[bi], mm)
            per.append({"D": d, "shapes": sh, "n": cnt})
            if d is not None:
                w = cnt[0] + cnt[-1]
                num += d * w
                den += w
        return (num / den if den > 0 else None), per

    all_rows = np.arange(B)
    out = {"ckpt": args.ckpt, "n_rows": int(B), "seed": args.seed,
           "diagnostic_only": True, "pre_registered": "HANDOFF.md 2026-08-05",
           "lookback": LOOKBACK, "n_quintiles": N_QUINT,
           "thresholds": {"regresses": REGRESSES, "conditions": CONDITIONS},
           "bands": [list(b) for b in BANDS],
           "band_edges": [e.tolist() for e in band_edges],
           "n_live": int(live.sum()), "n_with_history": int(have.sum())}

    # VALIDITY ARM FIRST, before anything touching a real token is printed.
    print("  VALIDITY arm. tokens drawn from the model's OWN law, sliced by the")
    print("  SAME covariate computed from the SAME real history. flat by")
    print("  construction, so any slope here is estimator error\n")
    print(f"    {'head':>6} {'D pooled':>10} {'boot sd':>9} {'in sd':>7}")
    vfail = []
    for h in heads:
        d, per = pooled_D(vpit, h, all_rows)
        vb = []
        for _ in range(max(20, args.draws // 2)):
            rs = rng.integers(0, B, B)
            x, _ = pooled_D(vpit, h, rs)
            if x is not None:
                vb.append(x)
        vsd = float(np.std(vb)) if len(vb) > 10 else float("nan")
        z = abs(d) / vsd if (d is not None and vsd > 0) else float("nan")
        out.setdefault("validity", {})[h] = {"D": d, "bootstrap_sd": vsd,
                                             "z": z, "per_band": per}
        print(f"    {h:>6} " + (f"{d:>+10.4f}" if d is not None else f"{'nan':>10}")
              + f" {vsd:>9.4f} {z:>7.1f}")
        if d is None or not (z <= VALID_SD):
            vfail.append(h)
    if vfail:
        print(f"\n  VALIDITY ARM FAILED on {', '.join(vfail)}. the estimator "
              f"manufactures a slope, so nothing below would mean anything.")
        out["verdict"] = f"FAILED, validity arm slopes on {','.join(vfail)}"
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        return
    print("\n  validity arm PASSES on every head, the slicing is sound\n")

    print("  PIT shape by history texture quintile, against the REAL token.")
    print("  above 1 is a U and too NARROW, below 1 is a hump and too WIDE.")
    print("  a model that conditions properly is flat across the row\n")
    for h in ("s", "dt"):
        d, per = pooled_D(pit, h, all_rows)
        out.setdefault("real", {})[h] = {"D": d, "per_band": per}
        print(f"    {h} head, pooled D {d:+.4f}")
        print(f"      {'band':>10} " +
              " ".join(f"{'Q' + str(k + 1):>8}" for k in range(N_QUINT)) +
              f" {'D':>9}")
        for bi, (lo, hi) in enumerate(BANDS):
            sh = per[bi]["shapes"]
            lab = f"{lo} to {hi - 1}"
            print(f"      {lab:>10} " +
                  " ".join(f"{v:>8.3f}" if v is not None else f"{'nan':>8}"
                           for v in sh) +
                  (f" {per[bi]['D']:>+9.4f}" if per[bi]["D"] is not None
                   else f" {'nan':>9}"))
        print()

    # Tilt on the direction head, reported and not what the thresholds are for.
    tth = []
    for bi, m in enumerate(band_mask):
        mm = m & live_th
        q = np.digitize(cov[mm], band_edges[bi])
        u = pit["th"][mm]
        tth.append([float(u[q == k].mean() - 0.5) if (q == k).sum() >= MIN_CELL
                    else None for k in range(N_QUINT)])
    out["th_tilt_by_quintile"] = tth
    print("  direction head, signed PIT tilt by quintile, reported only\n")
    for bi, (lo, hi) in enumerate(BANDS):
        print(f"      {f'{lo} to {hi - 1}':>10} " +
              " ".join(f"{v:>+8.4f}" if v is not None else f"{'nan':>8}"
                       for v in tth[bi]))

    D = out["real"]["s"]["D"]
    boot = []
    for _ in range(args.draws):
        rs = rng.integers(0, B, B)
        d, _ = pooled_D(pit, "s", rs)
        if d is not None:
            boot.append(d)
    sd = float(np.std(boot)) if len(boot) > 20 else float("nan")
    out["D_speed"] = {"D": D, "bootstrap_sd": sd, "draws": len(boot)}
    print(f"\n  SPEED HEAD  D {D:+.4f}  bootstrap sd {sd:.4f}\n")

    margin = min(abs(D - REGRESSES), abs(D - CONDITIONS))
    if margin < sd:
        verdict = (f"BOUNDARY. D {D:+.4f}, nearest threshold {margin:.4f} away "
                   f"against a bootstrap sd of {sd:.4f}, so the threshold call is "
                   f"REFUSED and this is reported as the in between case.")
    elif D >= REGRESSES:
        verdict = (f"REGRESSION TO THE MEAN. D {D:+.4f} >= {REGRESSES}. The model "
                   f"is too NARROW after volatile histories and too WIDE after "
                   f"calm ones, so it does not condition its texture on the "
                   f"history's texture. That is w4_position's under modulation "
                   f"seen one step at a time, and a conditional that ignores an "
                   f"available covariate is addressable in training.")
    elif D <= CONDITIONS:
        verdict = (f"THE MODEL CONDITIONS PROPERLY. D {D:+.4f} <= {CONDITIONS}. "
                   f"Its width tracks the history's texture, so branch (a) is "
                   f"dead as well as branch (b), and the defect is somewhere none "
                   f"of these instruments has yet reached.")
    else:
        verdict = (f"MIXED. D {D:+.4f} sits between {CONDITIONS} and {REGRESSES}. "
                   f"Report the table and the number, neither alone.")
    out["verdict"] = verdict
    print(f"  -> {verdict}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.out}")
    print("""
  DIAGNOSTIC ONLY, never a contract score. no serving change follows and
  no build is authorised by either outcome. phase conditioning and the
  spectral loss term remain NOT AUTHORISED.
  uniformity inside every slice is a NECESSARY condition and not a
  sufficient one. a flat D proves the model conditions on THIS function of
  the history, not on every function of it.""")


if __name__ == "__main__":
    main()
