"""How is the dt conditional wrong? The calibration characterisation.

CHARACTERISATION, NOT A HYPOTHESIS TEST. There is no single registered statistic
here and no pass or fail. The readings for every shape the primary plot can take
were written into HANDOFF before this file existed, so that a calibration curve,
which admits a story for every shape, cannot be narrated after the fact.

Background. `w4_timing` arm G established that this checkpoint's one step clock
puts roughly 25 percent too much speed power into 12 to 22 Hz with every
contamination removed: real history, real speed, real direction, no
autoregression. That says the conditional is tilted. It does not say in what way.

The instrument is arm G's forward pass reused, and it costs one pass. Teacher
force a held out human row completely, and at every position the model hands back
its whole distribution over the next interval given a fully real history and that
position's own real speed and direction. The interval the human actually produced
sits next to it. That is a predicted distribution against a realised value at a
million positions, which is a calibration question.

    PRIMARY   randomised PIT histogram
              U shaped    -> OVER confident, intervals too narrow
              hump shaped -> UNDER confident, intervals too wide
              tilted      -> BIASED clock, systematically fast or slow
              flat        -> conditional is calibrated, the arm G tilt is NOT in
                             any single interval and has to live in the
                             dependence between them, which would make the
                             current HANDOFF description WRONG rather than
                             merely incomplete

    DECISIVE  PIT uniformity inside slices of what the model ALREADY KNEW,
              the previous interval and the one before it. A calibrated
              conditional is uniform in EVERY such slice, not merely on
              average, so this is what separates a head that carries the
              dependence from one that does not. It is the unconfounded test
              for the flat branch above.

    SECONDARY reliability by the model's OWN predicted mean, and by the real
              speed at the position, so a bias and a dispersion error are told
              apart rather than blurred into one PIT shape

    Two diagnostics a draft of this file carried were REMOVED before the
    deciding run rather than reported with a caveat, and the amendment is in
    HANDOFF. Slicing by the REALISED interval conditions on the outcome, so
    regression to the mean manufactures a bias pattern in any model. Comparing
    the interval autocorrelation of a one step draw against the human is biased
    by construction, because the draws are conditionally independent given a
    real history while a human interval feeds forward into its successor, so the
    draw is attenuated even if the conditional is perfect. A number that is
    wrong by construction is worse than no number.

Nothing about serving follows from this file. No build is authorised by it. Phase
conditioning and the spectral loss term stay NOT AUTHORISED; neither was gated on
calibration and a calibration result cannot revive them.

Safety. Reads `training/events_*.npy` and one checkpoint. Touches no evaluation
data, no scoring code, and never `training/candi_polar_flow_best.pt`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments.event_stream_polar as esp  # noqa: E402
from models.event_ar import (  # noqa: E402
    DT_PAD_CLASS, EventARModel, N_DT_VALS, class_to_dt_ms, dt_ms_to_class,
    prefix_state,
)
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TH_NULL_CLASS, class_to_speed, dth_lattice_to_class,
    s2_to_class,
)

TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
MAX_T = 256


def pit_uniformity(u: np.ndarray, nbins: int = 20):
    """Randomised PIT summary. A calibrated conditional gives a flat histogram.

    Reported as the histogram itself plus three scalars that name the departure
    without needing the plot: the end mass against the middle mass separates over
    from under confidence, and the mean against 0.5 catches a one sided bias.
    """
    h, edges = np.histogram(u, bins=nbins, range=(0.0, 1.0))
    dens = h / h.sum() * nbins          # 1.0 everywhere if uniform
    k = max(1, nbins // 10)             # outer decile each side
    ends = float(dens[:k].mean() + dens[-k:].mean()) / 2.0
    mid = float(dens[nbins // 2 - k: nbins // 2 + k].mean())
    return {
        "density": [round(float(x), 4) for x in dens],
        "edges": [round(float(x), 3) for x in edges],
        "end_density": round(ends, 4),
        "mid_density": round(mid, 4),
        "end_over_mid": round(ends / mid, 4) if mid > 0 else float("nan"),
        "mean": round(float(u.mean()), 5),
        "mean_minus_half": round(float(u.mean()) - 0.5, 5),
        "ks_vs_uniform": round(float(np.abs(
            np.sort(u) - (np.arange(len(u)) + 0.5) / len(u)).max()), 5),
        "n": int(len(u)),
    }


def pit_by(u, by, edges, label, nbins=20):
    """PIT uniformity inside bins of a CONDITIONING variable.

    This is the decisive instrument for the dependence question and it is the
    one that is not confounded. A conditional that carries the dependence
    correctly is uniform inside EVERY slice of anything already known at the
    time of prediction, not merely uniform on average. Slicing by the previous
    interval therefore asks directly whether the head uses it.

    Never slice by the REALISED interval. That conditions on the outcome, and
    regression to the mean then manufactures a bias pattern in any model,
    correct or not.
    """
    rows = []
    idx = np.digitize(by, edges)
    for b in range(len(edges) + 1):
        m = idx == b
        if m.sum() < 500:
            continue
        lo = "-inf" if b == 0 else f"{edges[b - 1]:g}"
        hi = "inf" if b == len(edges) else f"{edges[b]:g}"
        s = pit_uniformity(u[m], nbins)
        rows.append({"bin": f"{lo}..{hi}", "n": int(m.sum()),
                     "pit_mean": s["mean"],
                     "mean_minus_half": s["mean_minus_half"],
                     "end_over_mid": s["end_over_mid"],
                     "ks": s["ks_vs_uniform"]})
    return {"by": label, "rows": rows}


def pit_acf(u_rows, lag):
    """Autocorrelation of the PIT values themselves.

    THIS is the test the raw interval autocorrelation could not be. Under a
    conditional that is exactly right the PIT values are i.i.d. uniform, so
    their autocorrelation is ZERO at every lag with no attenuation and no
    reference model needed. Any departure is a genuine defect: the model is
    leaving dependence on the table that a correct conditional would have
    absorbed.

    What it does NOT do is clear `w4_timing` arm G, and an earlier draft of this
    docstring claimed it did. That claim was wrong and is corrected here rather
    than softened. Arm G is under correlated relative to the human EVEN WHEN THE
    CONDITIONAL IS EXACTLY RIGHT, because a human interval feeds forward into
    the conditional of its own successor and a resampled one does not. Writing
    the lag one covariance both ways makes the gap explicit:

        human   Cov(dt_i, dt_i+1) = E[Cov(dt_i, mu_i+1(dt_i) | H_i)]
                                    + Cov(mu_i, mu_i+1)
        arm G   Cov(dt_i, dt_i+1) =   Cov(mu_i, mu_i+1)

    The first term survives in the human and vanishes in arm G, and it vanishes
    for a perfect model exactly as readily as for a bad one. So a zero here means
    the conditional is right; it says nothing about whether arm G's spectral
    excess is real. If anything a zero makes arm G look WORSE, because it removes
    the only explanation under which arm G's excess could have been a model
    defect. The control that actually settles it has to generate data whose true
    conditional is known, which is a different run.
    """
    num = den_a = den_b = 0.0
    n_eff = 0
    for u in u_rows:
        if len(u) < lag + 8:
            continue
        a, b = u[:-lag], u[lag:]
        a = a - 0.5
        b = b - 0.5
        num += float((a * b).sum())
        den_a += float((a * a).sum())
        den_b += float((b * b).sum())
        n_eff += len(a)
    if n_eff == 0 or den_a <= 0 or den_b <= 0:
        return {"lag": lag, "acf": float("nan"), "n": 0}
    r = num / np.sqrt(den_a * den_b)
    se = 1.0 / np.sqrt(n_eff)          # null sd of a correlation under i.i.d.
    return {"lag": lag, "acf": round(float(r), 6), "n": int(n_eff),
            "null_se": round(float(se), 6),
            "sigma": round(float(r / se), 2)}


def slice_table(real, pred_mean, pred_sd, by, edges, label):
    """Bias and dispersion inside bins of `by`, so an error confined to one
    regime is not averaged against the regimes where the model is fine."""
    rows = []
    idx = np.digitize(by, edges)
    for b in range(len(edges) + 1):
        m = idx == b
        if m.sum() < 200:
            continue
        lo = "-inf" if b == 0 else f"{edges[b - 1]:g}"
        hi = "inf" if b == len(edges) else f"{edges[b]:g}"
        rows.append({
            "bin": f"{lo}..{hi}", "n": int(m.sum()),
            "realised_mean": round(float(real[m].mean()), 4),
            "predicted_mean": round(float(pred_mean[m].mean()), 4),
            "bias": round(float((pred_mean[m] - real[m]).mean()), 4),
            "realised_sd": round(float(real[m].std()), 4),
            "predicted_sd": round(float(pred_sd[m].mean()), 4),
            "sd_ratio": round(float(pred_sd[m].mean() /
                                    max(real[m].std(), 1e-9)), 4),
        })
    return {"by": label, "rows": rows}



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--pit-bins", type=int, default=20)
    ap.add_argument("--out", default="research/w4_dtcal.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    lengths = np.load("training/events_len.npy")
    N = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(N, min(N, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(N), trained)
    pick = np.sort(np.random.default_rng(args.seed)
                   .choice(held, args.n, replace=False))
    print(f"  corpus {N:,}, never seen {len(held):,}, drew {args.n:,}", flush=True)

    s2 = np.load("training/events_s2.npy", mmap_mode="r")[pick]
    dth = np.load("training/events_dth.npy", mmap_mode="r")[pick]
    dt_ms = np.load("training/events_dt.npy", mmap_mode="r")[pick].astype(np.float64)
    conds = np.load("training/events_cond.npy")[pick]
    L = np.minimum(lengths[pick], MAX_T).astype(np.int64)

    keep = L >= 12
    s2, dth, dt_ms, conds, L = (s2[keep], dth[keep], dt_ms[keep],
                                conds[keep], L[keep])
    B = len(L)
    print(f"  {B:,} rows at least 12 events, median length "
          f"{int(np.median(L))}\n", flush=True)

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
        real_dt[i, :n] = dt_ms[i, :n]

    real_dt_cls = dt_ms_to_class(torch.from_numpy(real_dt)).numpy()
    cond_t = torch.from_numpy(conds[:, :4].astype(np.float32))
    s_t = torch.from_numpy(real_s)
    th_t = torch.from_numpy(real_th)
    dt_t = torch.from_numpy(real_dt_cls)

    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    print(f"  {args.ckpt} step {ck.get('step')} "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params\n",
          flush=True)

    # Support of the dt alphabet in milliseconds. Class k is k ms, which is why
    # this is simply arange.
    #
    # Class 151 is DT_PAD_CLASS and is input context only, never a real
    # interval, but class_to_dt_ms CLAMPS rather than rejecting, so it decodes
    # to 150 ms. Any probability the head leaves on it would be counted as a
    # 150 ms interval and would inflate every conditional mean. The conditional
    # is therefore taken over real intervals only, classes 0 to 150, and
    # renormalised. The mass that had to be removed is measured and reported,
    # because `w4_timing` arm G sampled without doing this and the size of that
    # mass is exactly how much arm G was affected.
    support = class_to_dt_ms(torch.arange(N_DT_VALS)).to(dev).double()
    pad_mass_sum, pad_mass_max, pad_n = 0.0, 0.0, 0

    pit, nll = [], []
    pmean, psd, realised, real_spd = [], [], [], []
    prev1, prev2 = [], []
    u_rows = []
    rng = np.random.default_rng(args.seed + 5)

    print("  one teacher forced forward pass, the model's whole dt "
          "distribution at every position", flush=True)
    with torch.no_grad():
        for c0 in range(0, B, args.batch):
            sl = slice(c0, min(c0 + args.batch, B))
            s_b, th_b = s_t[sl].to(dev), th_t[sl].to(dev)
            dt_b, cnd = dt_t[sl].to(dev), cond_t[sl].to(dev)
            s_p, th_p, dt_p = model.shift_inputs(s_b, th_b, dt_b)
            st = prefix_state(s_b, th_b, dt_b, cnd)
            x = model.trunk(s_p, th_p, dt_p, st, cnd)
            logits = model.dt_logits(x, s_b, th_b)
            p_full = torch.softmax(logits.double() / args.temp, dim=-1)

            # live positions: every real event including a TICK, which carries a
            # real interval. Only the PAD tail is excluded.
            live = s_b < S_PAD_CLASS
            nb = s_b.shape[0]
            lens = live.sum(1)

            pm = p_full[..., DT_PAD_CLASS]
            lvm = live
            pad_mass_sum += float(pm[lvm].sum())
            pad_mass_max = max(pad_mass_max, float(pm[lvm].max()) if lvm.any() else 0.0)
            pad_n += int(lvm.sum())
            p = p_full[..., :N_DT_VALS]
            p = p / p.sum(-1, keepdim=True).clamp(min=1e-12)

            cdf = torch.cumsum(p, dim=-1)
            k = dt_b.clamp(max=N_DT_VALS - 1)
            p_k = p.gather(-1, k.unsqueeze(-1)).squeeze(-1)
            cdf_lo = (cdf.gather(-1, k.unsqueeze(-1)).squeeze(-1) - p_k)
            v = torch.from_numpy(rng.random(tuple(k.shape))).to(dev)
            u = cdf_lo + v * p_k                      # randomised PIT

            mu = (p * support).sum(-1)
            var = (p * support * support).sum(-1) - mu * mu
            sd = torch.sqrt(var.clamp(min=0))


            # The two intervals immediately before this position, both already
            # known to the model when it predicts. Position 0 and 1 have no such
            # history and are marked -1 so the slices drop them.
            rd = class_to_dt_ms(dt_b.cpu()).numpy()
            p1 = np.full_like(rd, -1.0)
            p2 = np.full_like(rd, -1.0)
            p1[:, 1:] = rd[:, :-1]
            p2[:, 2:] = rd[:, :-2]

            lv = live.cpu().numpy()
            # PIT kept with its row structure intact, for the autocorrelation.
            un = u.cpu().numpy()
            for i in range(nb):
                n_i = int(lens[i].item())
                if n_i >= 12:
                    u_rows.append(un[i, :n_i].copy())
            prev1.append(p1[lv])
            prev2.append(p2[lv])
            pit.append(u.cpu().numpy()[lv])
            nll.append(-torch.log(p_k.clamp(min=1e-12)).cpu().numpy()[lv])
            pmean.append(mu.cpu().numpy()[lv])
            psd.append(sd.cpu().numpy()[lv])
            realised.append(class_to_dt_ms(dt_b.cpu()).numpy()[lv])
            real_spd.append(class_to_speed(s_b.cpu()).numpy()[lv])

    pit = np.concatenate(pit)
    nll = np.concatenate(nll)
    pmean = np.concatenate(pmean)
    psd = np.concatenate(psd)
    realised = np.concatenate(realised)
    real_spd = np.concatenate(real_spd)
    prev1 = np.concatenate(prev1)
    prev2 = np.concatenate(prev2)
    print(f"  {len(pit):,} live positions scored\n", flush=True)

    out = {"n_rows": B, "n_positions": int(len(pit)), "ckpt": args.ckpt,
           "temp": args.temp, "seed": args.seed}

    # How much probability the head leaves on the PAD class at live positions.
    # This is also the impurity in `w4_timing` arm G, which sampled the full
    # alphabet, and it is reported whether it is large or small.
    out["dt_pad_mass"] = {"mean": round(pad_mass_sum / max(pad_n, 1), 8),
                          "max": round(pad_mass_max, 8), "n": pad_n}
    pmm = out["dt_pad_mass"]["mean"]
    print(f"  dt PAD class mass at live positions, mean {pmm:.2e}, max "
          f"{out['dt_pad_mass']['max']:.2e}")
    print("    this is the impurity in w4_timing arm G, which sampled the full "
          "alphabet without renormalising")

    # ---- PRIMARY -------------------------------------------------------
    P = pit_uniformity(pit, args.pit_bins)
    out["pit"] = P
    print("  PRIMARY, randomised PIT histogram, flat means calibrated")
    for i, d in enumerate(P["density"]):
        lo = i / args.pit_bins
        bar = "#" * int(round(d * 30))
        print(f"    {lo:4.2f}-{lo + 1 / args.pit_bins:4.2f}  {d:6.3f}  {bar}")
    print(f"\n    end density {P['end_density']:.3f}, mid density "
          f"{P['mid_density']:.3f}, ratio {P['end_over_mid']:.3f}")
    print(f"    mean {P['mean']:.4f}, departure from 0.5 "
          f"{P['mean_minus_half']:+.4f}, KS {P['ks_vs_uniform']:.4f}")

    r = P["end_over_mid"]
    if r > 1.15:
        shape = ("U SHAPED -> the model is OVER CONFIDENT, its intervals are "
                 "too narrow and reality lands in its tails too often")
    elif r < 0.87:
        shape = ("HUMP SHAPED -> the model is UNDER CONFIDENT, its intervals "
                 "are too wide and it hedges")
    else:
        shape = ("FLAT on the ends versus middle axis, read the mean and the "
                 "slices before concluding calibrated")
    if abs(P["mean_minus_half"]) > 0.02:
        shape += (f"\n    AND TILTED, mean is {P['mean_minus_half']:+.4f} off "
                  f"centre, so the clock is also BIASED "
                  f"{'SLOW' if P['mean_minus_half'] > 0 else 'FAST'}")
    print(f"    -> {shape}")
    out["pit_shape"] = shape

    # ---- SECONDARY -----------------------------------------------------
    out["overall"] = {
        "realised_mean_ms": round(float(realised.mean()), 4),
        "predicted_mean_ms": round(float(pmean.mean()), 4),
        "bias_ms": round(float((pmean - realised).mean()), 4),
        "realised_sd_ms": round(float(realised.std()), 4),
        "mean_predicted_sd_ms": round(float(psd.mean()), 4),
        "sd_ratio": round(float(psd.mean() / max(realised.std(), 1e-9)), 4),
        "mean_nll_nats": round(float(nll.mean()), 4),
    }
    o = out["overall"]
    print(f"\n  OVERALL  realised mean {o['realised_mean_ms']:.3f} ms, "
          f"predicted {o['predicted_mean_ms']:.3f} ms, bias {o['bias_ms']:+.3f}")
    print(f"           realised sd {o['realised_sd_ms']:.3f} ms, mean "
          f"conditional sd {o['mean_predicted_sd_ms']:.3f} ms, ratio "
          f"{o['sd_ratio']:.3f}")
    print(f"           mean NLL {o['mean_nll_nats']:.4f} nats")
    print("           the sd ratio compares a WITHIN position spread against a "
          "POOLED spread and is not a calibration test on its own, it is here "
          "to tell a bias apart from a dispersion error")

    # THE DECISIVE TEST for the flat branch. A conditional that carries the
    # dependence is uniform inside every slice of anything already known when it
    # predicts, not merely uniform on average. Slicing the PIT by the previous
    # interval and by the one before that asks exactly whether the head uses
    # them. A tilt inside a lag 2 slice with none inside a lag 1 slice would say
    # the head tracks the immediately preceding interval and misses the
    # alternation one step further back.
    #
    # The autocorrelation comparison that a draft of this file carried has been
    # REMOVED rather than reported with a caveat, because it cannot answer this.
    # One step draws are conditionally independent given a real history, while a
    # human interval feeds forward into its own successor, so the draws are
    # attenuated relative to the human EVEN IF THE CONDITIONAL IS PERFECT. That
    # is a property of the estimator, not of the model, and a number that is
    # biased by construction is worse than no number.
    dt_edges = np.array([2, 4, 6, 8, 10, 12, 16, 20, 30, 50, 90])
    print("\n  DECISIVE, PIT uniformity inside slices of what the model "
          "ALREADY KNEW. uniform in every slice means the dependence is "
          "carried, a tilt means it is not")
    for tbl, arr, lab in (("pit_by_prev1", prev1, "previous interval, ms"),
                          ("pit_by_prev2", prev2,
                           "interval two steps back, ms")):
        ok = arr >= 0
        t = pit_by(pit[ok], arr[ok], dt_edges, lab)
        out[tbl] = t
        print(f"\n    sliced by {lab}")
        print(f"      {'bin':>12}  {'n':>9}  {'PIT mean':>9}  "
              f"{'off 0.5':>8}  {'end/mid':>8}  {'KS':>7}")
        for rw in t["rows"]:
            flag = "  <-- TILTED" if abs(rw["mean_minus_half"]) > 0.02 else ""
            print(f"      {rw['bin']:>12}  {rw['n']:>9,}  "
                  f"{rw['pit_mean']:>9.4f}  {rw['mean_minus_half']:>+8.4f}  "
                  f"{rw['end_over_mid']:>8.3f}  {rw['ks']:>7.4f}{flag}")
        worst = max((abs(r["mean_minus_half"]) for r in t["rows"]), default=0.0)
        out[tbl]["worst_abs_tilt"] = round(float(worst), 5)
        print(f"      worst absolute tilt {worst:+.4f}")

    sp_edges = np.array([0.5, 1, 2, 4, 8, 16, 32])
    t = pit_by(pit, real_spd, sp_edges, "real speed at the position")
    out["pit_by_real_speed"] = t
    print("\n    sliced by real speed at the position")
    print(f"      {'bin':>12}  {'n':>9}  {'PIT mean':>9}  "
          f"{'off 0.5':>8}  {'end/mid':>8}  {'KS':>7}")
    for rw in t["rows"]:
        flag = "  <-- TILTED" if abs(rw["mean_minus_half"]) > 0.02 else ""
        print(f"      {rw['bin']:>12}  {rw['n']:>9,}  "
              f"{rw['pit_mean']:>9.4f}  {rw['mean_minus_half']:>+8.4f}  "
              f"{rw['end_over_mid']:>8.3f}  {rw['ks']:>7.4f}{flag}")

    # Reliability, sliced by the model's OWN predicted mean. This is the
    # legitimate version of a bias table. Slicing by the realised interval, as a
    # draft of this file did, conditions on the outcome and manufactures a bias
    # pattern in any model through regression to the mean.
    pm_edges = np.array([4, 6, 8, 9, 10, 11, 12, 14, 18, 25])
    for tbl, arr, edges, lab in (
            ("by_predicted_mean", pmean, pm_edges,
             "the model's own predicted mean, ms"),
            ("by_real_speed", real_spd, sp_edges, "real speed at the position")):
        t = slice_table(realised, pmean, psd, arr, edges, lab)
        out[tbl] = t
        print(f"\n  sliced by {lab}")
        print(f"    {'bin':>12}  {'n':>9}  {'real mu':>8}  {'pred mu':>8}  "
              f"{'bias':>7}  {'real sd':>8}  {'pred sd':>8}  {'ratio':>6}")
        for rw in t["rows"]:
            print(f"    {rw['bin']:>12}  {rw['n']:>9,}  "
                  f"{rw['realised_mean']:>8.3f}  {rw['predicted_mean']:>8.3f}  "
                  f"{rw['bias']:>+7.3f}  {rw['realised_sd']:>8.3f}  "
                  f"{rw['predicted_sd']:>8.3f}  {rw['sd_ratio']:>6.3f}")

    # THE SECOND DECISIVE READ, and the one that prices w4_timing arm G's own
    # construction. Under a correct conditional the PIT values are i.i.d., so
    # this is exactly zero at every lag with no attenuation and no reference
    # model. It is the test the raw interval autocorrelation could never be.
    print("\n  DECISIVE, autocorrelation of the PIT values themselves. exactly "
          "zero under a correct conditional, no attenuation, no reference "
          "model needed")
    out["pit_acf"] = {}
    for lag in (1, 2, 3, 4, 5, 8):
        a = pit_acf(u_rows, lag)
        out["pit_acf"][f"lag{lag}"] = a
        flag = "  <-- DEPENDENCE LEFT ON THE TABLE" if abs(a["sigma"]) > 4 else ""
        print(f"    lag {lag}   acf {a['acf']:+.5f}   null se {a['null_se']:.5f}"
              f"   {a['sigma']:+6.1f} sd   n {a['n']:,}{flag}")
    worst = max(abs(v["sigma"]) for v in out["pit_acf"].values())
    out["pit_acf_worst_sigma"] = worst
    print(f"    worst {worst:.1f} sd")
    print("    a nonzero value here is a genuine defect. a ZERO does NOT clear "
          "w4_timing arm G: arm G is under correlated against the human even "
          "when the conditional is exactly right, because a human interval "
          "feeds forward into its successor's conditional and a resampled one "
          "does not. see the pit_acf docstring for the covariance both ways")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.out}")
    print("""
  CHARACTERISATION ONLY, never a contract score. no serving change follows.
  no build is authorised by this run. phase conditioning and the spectral
  loss term remain NOT AUTHORISED, neither was gated on calibration.
  the PIT is the primary read and the slices exist so a bias and a
  dispersion error are told apart rather than blurred into one shape.
  a FLAT PIT would make the current HANDOFF description of arm G WRONG
  rather than incomplete, and it gets corrected, not qualified.""")


if __name__ == "__main__":
    main()
