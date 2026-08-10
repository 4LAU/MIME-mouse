"""Is the model's high frequency speed excess there from the first moment of a
movement, or does it build up along it?

PRE REGISTERED in HANDOFF, thresholds fixed before this file existed:

    SLOPE_DIFF >= 0.17   ACCUMULATION, at least half the excess is growth along
                         the movement, the work is long horizon consistency
    SLOPE_DIFF <= 0.05   PRESENT FROM THE START, the work is the texture itself
    in between           MIXED, report the number and both shares
    BOUNDARY             within one null sd of a threshold the call is REFUSED
                         and the in between case is reported instead

A NEGATIVE SLOPE_DIFF is a registered third outcome. It would mean the excess is
concentrated at the START of movements and fades, which points at how a movement
is launched rather than at its texture or its drift.

Why this design cannot carry the fault that withdrew arm G and arm E. Both of
those forced part of a human sequence into a model generation, which creates a
context the joint never produces, and that context alone reproduced their entire
signature on data with no defect in it. Here nothing is forced and nothing is
mixed. Free running model on one side, real human recordings on the other.

The statistic. Resample to the contract's 125 Hz grid, take speed, slide a 64
sample window with hop 8. Per window: remove the mean, divide by that window's
own sd, Hann, periodogram, average the 11 to 41.5 Hz bins. Standardising per
window removes amplitude and leaves the SHAPE, so a person being faster mid
movement than at its ends does not enter. Regress log band power on the window's
normalised centre position within the trace. The per trace slope is the unit.

The duration confound. w4_timing's windows() takes ONE centred window per trace
precisely so a duration difference cannot become a spectral one. This takes
several, so that protection is gone and is replaced by length matching: bin by
resampled length into deciles of the HUMAN distribution, compute inside each bin,
pool weighted by human count, drop and report any bin under 100 traces a side.

DIAGNOSTIC ONLY, never a contract score. One trajectory per row, no selection, no
reranking. No serving change follows and no build is authorised by either outcome.
Phase conditioning and the spectral loss term remain NOT AUTHORISED.

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

# Set BEFORE experiments.event_stream_polar is imported, because it reads these
# at import time.
os.environ.setdefault("EVENT_STEPS", "100")
os.environ.setdefault("EVENT_CFG_W", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import experiments.event_stream_polar as esp  # noqa: E402
from models.event_ar import (  # noqa: E402
    EventARModel, class_to_dt_ms, dt_ms_to_class,
)
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TH_NULL_CLASS, dth_lattice_to_class, s2_to_class,
)
from research.w4_timing import (  # noqa: E402
    BAND_HI_HZ, BAND_LO_HZ, HZ, MAX_T, N_TRAIN_DEFAULT, TRAIN_PICK_SEED,
    signals,
)

ACCUM_DOMINATES = 0.17
FLAT = 0.05
ARM_C_EXCESS_LOG = 0.1677      # log(1 + 0.1825), the excess these anchor to
MIN_BIN = 100                  # traces per side, below this a length bin is dropped


def trace_windows(sp: np.ndarray, w: int, hop: int,
                  sel: np.ndarray) -> tuple | None:
    """Every window of one trace as (normalised centre position, log band power).

    AMENDED estimator, see the HANDOFF amendment. The registered design fitted a
    line inside each trace and averaged those slopes. The smoke test on human
    data alone put the null sd of that estimator at 0.58 against thresholds of
    0.17 and 0.05, so every possible outcome was a boundary refusal. The cause is
    structural rather than a sample size problem: the median trace covers only
    about a third of the 0 to 1 position range, and a within trace regression
    divides by the variance of that narrow span, which multiplies the noise. The
    estimand is unchanged, log band power per unit of normalised position. Only
    the estimator changes, from fit inside each trace then average, to pool then
    fit. Returns None when the trace yields no usable window.
    """
    n = len(sp)
    starts = list(range(0, n - w + 1, hop))
    if not starts:
        return None
    win = np.hanning(w)
    norm = (win * win).sum()
    pos, lp = [], []
    for s0 in starts:
        seg = sp[s0:s0 + w]
        seg = seg - seg.mean()
        sd = seg.std()
        if sd < 1e-9:
            continue
        seg = seg / sd
        p = np.abs(np.fft.rfft(seg * win)) ** 2 / norm
        b = float(p[sel].mean())
        if b <= 0:
            continue
        pos.append((s0 + w / 2.0) / (n - 1.0))
        lp.append(np.log(b))
    if not pos:
        return None
    return np.asarray(pos), np.asarray(lp)


N_POS_BINS = 5
POS_EDGES = np.linspace(0.0, 1.0, N_POS_BINS + 1)
POS_CENTRES = (POS_EDGES[:-1] + POS_EDGES[1:]) / 2.0


def set_pos_bins(n: int) -> None:
    """Position binning is a POWER knob, calibrated on human data alone before
    the model side is generated. Tuning it against a human against human null
    cannot leak the answer, because no model output exists when it is chosen."""
    global N_POS_BINS, POS_EDGES, POS_CENTRES
    N_POS_BINS = n
    POS_EDGES = np.linspace(0.0, 1.0, n + 1)
    POS_CENTRES = (POS_EDGES[:-1] + POS_EDGES[1:]) / 2.0


def summarise(traces):
    """Reduce each trace to a fixed size summary: per position bin, the sum of
    log band power and the count of windows.

    Everything downstream is then matrix indexing, which is what makes a 400 draw
    bootstrap over whole traces affordable. Windows inside one trace are
    correlated, so the resampling unit is the trace and never the window.
    """
    S = np.zeros((len(traces), N_POS_BINS))
    C = np.zeros((len(traces), N_POS_BINS))
    for i, (pos, lp) in enumerate(traces):
        j = np.clip(np.digitize(pos, POS_EDGES) - 1, 0, N_POS_BINS - 1)
        np.add.at(S[i], j, lp)
        np.add.at(C[i], j, 1.0)
    return S, C


def excess_slope(aS, aC, a_len, bS, bC, b_len, edges, want_rows=False):
    """Slope of the (a minus b) excess against normalised path position.

    Within each length bin the mean log band power profile of each population is
    taken across position bins and differenced; the length bins are pooled
    weighted by b's trace count; a weighted line is fitted to the pooled excess
    profile. AMENDED estimator: pool then fit, rather than fit inside each trace
    then average.
    """
    ai = np.digitize(a_len, edges)
    bi = np.digitize(b_len, edges)
    num = np.zeros(N_POS_BINS)
    den = np.zeros(N_POS_BINS)
    rows, dropped = [], []
    for k in range(0, len(edges) + 1):
        ka = ai == k
        kb = bi == k
        na, nb = int(ka.sum()), int(kb.sum())
        if na < MIN_BIN or nb < MIN_BIN:
            if want_rows and (na or nb):
                dropped.append({"bin": int(k), "n_a": na, "n_b": nb})
            continue
        ca, cb = aC[ka].sum(0), bC[kb].sum(0)
        ok = (ca >= MIN_BIN) & (cb >= MIN_BIN)
        d = np.where(ok, aS[ka].sum(0) / np.maximum(ca, 1)
                     - bS[kb].sum(0) / np.maximum(cb, 1), 0.0)
        wgt = np.where(ok, float(nb), 0.0)
        num += d * wgt
        den += wgt
        if want_rows:
            rows.append({"bin": int(k), "n_a": na, "n_b": nb,
                         "excess_by_position": [
                             (float(x) if o else None) for x, o in zip(d, ok)]})
    ok = den > 0
    if ok.sum() < 2:
        return None, None, rows, dropped
    prof = np.where(ok, num / np.maximum(den, 1), np.nan)
    slope = float(np.polyfit(POS_CENTRES[ok], prof[ok], 1,
                             w=np.sqrt(den[ok]))[0])
    return slope, prof, rows, dropped


def boot_sd(aS, aC, a_len, bS, bC, b_len, edges, draws, seed):
    """Bootstrap whole TRACES and refit, so the error bar carries the pooling,
    the per trace variance and the correlation between windows inside a trace
    together."""
    rng = np.random.default_rng(seed)
    out = []
    na, nb = len(a_len), len(b_len)
    for _ in range(draws):
        ia = rng.integers(0, na, na)
        ib = rng.integers(0, nb, nb)
        s, _, _, _ = excess_slope(aS[ia], aC[ia], a_len[ia],
                                  bS[ib], bC[ib], b_len[ib], edges)
        if s is not None:
            out.append(s)
    if len(out) < 20:
        return float("nan")
    return float(np.std(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--temp", type=float, default=1.0)
    # w=32 hop=4 pos=8 len=5 is the FROZEN design, chosen by a power calibration
    # run on human data alone before any model output existed. w=64 gave a null
    # sd of 0.089 against thresholds of 0.05 and 0.17; w=32 gives 0.047 and also
    # admits shorter traces, so 73% of rows contribute rather than 43%. Halving
    # the hop to 2 doubled the windows and moved the null sd by 0.0006, which is
    # what says the variance is limited by traces and not by windows.
    ap.add_argument("--w", type=int, default=32)
    ap.add_argument("--hop", type=int, default=4)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--pos-bins", type=int, default=8)
    ap.add_argument("--len-bins", type=int, default=5,
                    help="quantiles of the HUMAN resampled length used to match "
                         "the two populations. More bins is a stronger control "
                         "on the duration confound and a weaker error bar.")
    ap.add_argument("--no-gpu", action="store_true",
                    help="human arms only, for a pipeline smoke test. The "
                         "verdict cannot be produced without the model side and "
                         "the script says so rather than printing one.")
    ap.add_argument("--out", default="research/w4_position.json")
    args = ap.parse_args()
    set_pos_bins(args.pos_bins)

    dev = esp._DEVICE
    lengths = np.load("training/events_len.npy")
    N = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(N, min(N, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(N), trained)
    pick = np.sort(np.random.default_rng(args.seed)
                   .choice(held, args.n, replace=False))

    s2 = np.load("training/events_s2.npy", mmap_mode="r")[pick]
    dth = np.load("training/events_dth.npy", mmap_mode="r")[pick]
    dt_ms = np.load("training/events_dt.npy", mmap_mode="r")[pick].astype(np.float64)
    conds = np.load("training/events_cond.npy")[pick]
    L = np.minimum(lengths[pick], MAX_T).astype(np.int64)
    keep = L >= 12
    s2, dth, dt_ms, conds, L = s2[keep], dth[keep], dt_ms[keep], conds[keep], L[keep]
    B = len(L)
    print(f"  corpus {N:,}, never seen {len(held):,}, drew {args.n:,}")
    print(f"  {B:,} rows at least 12 events, the same rows w4_timing used\n",
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
        real_dt[i, :n] = dt_ms[i, :n]

    # The human clock through the model's own whole ms alphabet, so the two
    # sides differ by the model and not by a quantisation the model cannot
    # express. This is arm Aq, the reference w4_timing settled on.
    real_dt_cls = dt_ms_to_class(torch.from_numpy(real_dt)).numpy()
    quant_dt = class_to_dt_ms(torch.from_numpy(real_dt_cls)).numpy().astype(np.float64)
    quant_dt[real_dt == 0.0] = 0.0

    angs = np.arctan2(conds[:, 3].astype(np.float64),
                      conds[:, 2].astype(np.float64))
    cond_t = torch.from_numpy(conds[:, :4].astype(np.float32))

    arms = {"H_human": (real_s, real_th, quant_dt)}

    if not args.no_gpu:
        ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                        weights_only=False)
        model = EventARModel(**ck["config"]).to(dev).eval()
        model.load_state_dict(ck["model_state_dict"])
        print(f"  {args.ckpt} step {ck.get('step')} "
              f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params\n",
              flush=True)
        print("  arm M, model free running, nothing forced", flush=True)
        o_s = np.full((B, MAX_T), S_PAD_CLASS, dtype=np.int64)
        o_th = np.full((B, MAX_T), TH_NULL_CLASS, dtype=np.int64)
        o_dt = np.zeros((B, MAX_T), dtype=np.float64)
        torch.manual_seed(args.seed + 17)
        with torch.no_grad():
            for c0 in range(0, B, args.batch):
                sl = slice(c0, min(c0 + args.batch, B))
                s_o, th_o, dt_o = model.sample(cond_t[sl].to(dev),
                                               temperature=args.temp)
                wd = s_o.shape[1]
                o_s[sl, :wd] = s_o.cpu().numpy()
                o_th[sl, :wd] = th_o.cpu().numpy()
                o_dt[sl, :wd] = class_to_dt_ms(dt_o.cpu()).numpy()
        arms["M_model_free"] = (o_s, o_th, o_dt)

    freqs = np.fft.rfftfreq(args.w, d=1.0 / HZ)
    sel = (freqs >= BAND_LO_HZ) & (freqs <= BAND_HI_HZ)

    got = {}
    for name, (sa, ta, da) in arms.items():
        tr, ln_ = [], []
        for i in range(B):
            dz = ((np.log(np.maximum(da[i], 0.05)) - esp._DT_MEAN) / esp._DT_STD)
            p = esp._decode(dz, sa[i], ta[i], 0.0, 0.0, float(angs[i]))
            if p is None or len(p) < 8:
                continue
            sg = signals(np.asarray(p, dtype=np.float64))
            if sg is None:
                continue
            tw = trace_windows(sg["speed"], args.w, args.hop, sel)
            if tw is None:
                continue
            tr.append(tw)
            ln_.append(len(sg["speed"]))
        S, C = summarise(tr)
        got[name] = (S, C, np.asarray(ln_, dtype=np.float64))
        nw = int(C.sum())
        print(f"  {name:<14} {len(tr):>6,} traces with at least one window, "
              f"{len(tr) / B * 100:5.1f}% of rows, {nw:>7,} windows, median "
              f"resampled length {int(np.median(ln_))} samples", flush=True)

    hS, hC, h_ln = got["H_human"]
    q = np.arange(1, args.len_bins) / args.len_bins
    edges = np.quantile(h_ln, q)
    out = {"ckpt": args.ckpt, "w": args.w, "hop": args.hop, "hz": HZ,
           "n_rows": int(B), "temp": args.temp, "diagnostic_only": True,
           "pre_registered": "HANDOFF.md 2026-08-05, amended before the model "
                             "side was generated, see the HANDOFF amendment",
           "band_hz": [BAND_LO_HZ, BAND_HI_HZ],
           "n_position_bins": N_POS_BINS,
           "thresholds": {"accum_dominates": ACCUM_DOMINATES, "flat": FLAT},
           "arm_C_excess_log": ARM_C_EXCESS_LOG,
           "length_bin_edges": edges.tolist(),
           "n_traces": {k: int(len(v[2])) for k, v in got.items()},
           "n_windows": {k: int(v[1].sum()) for k, v in got.items()}}

    print("\n  mean log band power by position along the path, per arm")
    print(f"    {'arm':<14} " + " ".join(f"{c:>8.2f}" for c in POS_CENTRES))
    for k, (S, C, _) in got.items():
        cs, ss = C.sum(0), S.sum(0)
        prof = np.where(cs > 0, ss / np.maximum(cs, 1), np.nan)
        print(f"    {k:<14} " + " ".join(f"{x:>+8.4f}" for x in prof))

    # VALIDITY GATE. Two halves of the SAME human population, identical pipeline.
    rng = np.random.default_rng(args.seed + 5)
    perm = rng.permutation(len(h_ln))
    h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
    g, _, _, _ = excess_slope(hS[h1], hC[h1], h_ln[h1],
                              hS[h2], hC[h2], h_ln[h2], edges)
    gsd = boot_sd(hS[h1], hC[h1], h_ln[h1], hS[h2], hC[h2], h_ln[h2],
                  edges, args.draws, args.seed + 7)
    gate_ok = g is not None and abs(g) <= 2.0 * gsd
    out["validity_gate"] = {"diff": g, "null_sd": gsd,
                            "sigma": (g / gsd) if g is not None else None,
                            "pass": bool(gate_ok)}
    print(f"\n  VALIDITY   human half against human half: "
          f"{g:+.4f} against a null sd of {gsd:.4f} ({g / gsd:+.1f} sd)")
    print(f"             must be within 2 sd of zero, "
          f"{'PASS' if gate_ok else 'FAIL, no verdict is reported'}")
    print(f"  POWER      the null sd of the model against human comparison is "
          f"what decides whether the registered")
    print(f"             thresholds {FLAT:.2f} and {ACCUM_DOMINATES:.2f} are "
          f"separable at all. Half against half is a proxy for it.")

    if "M_model_free" not in got:
        out["verdict"] = ("model side not run, --no-gpu. No verdict exists and "
                          "none is printed.")
        print("\n  -> " + out["verdict"])
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  wrote {args.out}")
        return

    mS, mC, m_ln = got["M_model_free"]
    d, prof, rows, dropped = excess_slope(mS, mC, m_ln, hS, hC, h_ln, edges,
                                          want_rows=True)
    dsd = boot_sd(mS, mC, m_ln, hS, hC, h_ln, edges, args.draws, args.seed + 9)
    out["slope_diff"] = {"diff": d, "null_sd": dsd, "sigma": d / dsd,
                         "pooled_excess_profile": [
                             (None if np.isnan(x) else float(x)) for x in prof],
                         "position_centres": POS_CENTRES.tolist(),
                         "bins": rows, "dropped_bins": dropped}

    print("\n  pooled model minus human excess in log band power, by position")
    print(f"    {'position':<14} " + " ".join(f"{c:>8.2f}" for c in POS_CENTRES))
    print(f"    {'excess':<14} " + " ".join(
        ("     nan" if np.isnan(x) else f"{x:>+8.4f}") for x in prof))
    if dropped:
        print(f"    length bins dropped for fewer than {MIN_BIN} traces a side: "
              + ", ".join(f"bin {x['bin']} (M {x['n_a']}, H {x['n_b']})"
                          for x in dropped))

    print(f"\n  SLOPE_DIFF {d:+.4f} against a null sd of {dsd:.4f} "
          f"({d / dsd:+.1f} sd)")

    # The threshold anchors were derived from a w=64 measurement of the total
    # excess. This pipeline runs at w=32 on a longer-reaching population, so its
    # own total excess is measured here and the share is quoted against BOTH,
    # with which is which stated. An interval is always reported, whatever branch
    # fires, because a threshold branch alone hides how much the number could
    # move.
    okp = ~np.isnan(prof)
    mean_excess = float(np.average(prof[okp]))
    lo, hi = d - 2.0 * dsd, d + 2.0 * dsd
    out["mean_excess_log_in_pipeline"] = mean_excess
    out["growth_share"] = {
        "vs_registered_anchor": d / (2.0 * ARM_C_EXCESS_LOG),
        "vs_in_pipeline_total": (d / (2.0 * mean_excess)
                                 if abs(mean_excess) > 1e-9 else None),
        "interval_2sd": [lo, hi],
    }
    print(f"\n  mean excess across position in THIS pipeline {mean_excess:+.4f} "
          f"in logs, against the w=64 anchor of {ARM_C_EXCESS_LOG:+.4f}")
    print(f"  SLOPE_DIFF 2 sd interval [{lo:+.4f}, {hi:+.4f}], reported whatever "
          f"branch fires below")

    if not gate_ok:
        verdict = ("VALIDITY GATE FAILED, the pipeline manufactures a slope "
                   "difference between two samples of the same population, so "
                   "no verdict is reported.")
    else:
        margin = min(abs(d - ACCUM_DOMINATES), abs(d - FLAT))
        share = d / (2.0 * ARM_C_EXCESS_LOG)
        if d < 0 and abs(d) > dsd:
            verdict = (f"NEGATIVE, {d:+.4f} at {d / dsd:+.1f} sd. This is the "
                       f"registered third outcome. The excess is concentrated "
                       f"at the START of movements and fades, which points at "
                       f"how a movement is launched rather than at its texture "
                       f"or at drift.")
        elif margin < dsd:
            verdict = (f"BOUNDARY, the nearest threshold is {margin:.4f} away "
                       f"against a null sd of {dsd:.4f}, so the threshold call "
                       f"is REFUSED. Reported as the in between case: "
                       f"SLOPE_DIFF {d:+.4f}, roughly {share * 100:.0f} percent "
                       f"of the excess attributable to growth, the rest present "
                       f"from the start.")
        elif d >= ACCUM_DOMINATES:
            verdict = (f"ACCUMULATION. SLOPE_DIFF {d:+.4f} >= "
                       f"{ACCUM_DOMINATES:.2f}, roughly {share * 100:.0f} "
                       f"percent of the excess is growth along the movement. "
                       f"The work is long horizon consistency.")
        elif d <= FLAT:
            verdict = (f"PRESENT FROM THE START. SLOPE_DIFF {d:+.4f} <= "
                       f"{FLAT:.2f}, roughly {share * 100:.0f} percent is "
                       f"growth. The work is the texture itself.")
        else:
            verdict = (f"MIXED. SLOPE_DIFF {d:+.4f} sits between {FLAT:.2f} and "
                       f"{ACCUM_DOMINATES:.2f}, roughly {share * 100:.0f} "
                       f"percent growth and the rest present from the start. "
                       f"Both reported, neither alone.")
    out["verdict"] = verdict
    print(f"\n  -> {verdict}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.out}")
    print("""
  DIAGNOSTIC ONLY, never a contract score. no serving change follows and
  no build is authorised by either outcome. phase conditioning and the
  spectral loss term remain NOT AUTHORISED.
  read the VALIDITY gate first. a call landing within one null sd of a
  threshold is REFUSED, not rounded.
  the share of the excess is an APPROXIMATION. it converts a many window
  slope into a one centred window statistic and is an anchor for the
  thresholds, not an accounting.
  only traces long enough for one 32 sample window appear here, which is
  73 percent of rows, so this population is close to the one arm C
  measured rather than a long tail of it.""")


if __name__ == "__main__":
    main()
