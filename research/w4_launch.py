"""Is the LAUNCH conditional itself wrong, or only what the model does with it?

PRE REGISTERED in HANDOFF, and CORRECTED there on the smoke test before any
deciding run. The original primary statistic, mean NLL minus mean predicted
entropy, was claimed to estimate a KL and to be non negative. It does neither.
Writing p for the truth and q for the model it equals KL(p||q) + H(p) - H(q), so
it goes negative whenever the model is wider than the truth, which is exactly the
thing this run exists to detect. The registered sanity floor caught it on 1,359
rows. Both are kept below as DESCRIPTIVE quantities and neither decides anything.

The primary instrument is the randomised PIT, which is entropy free and is what
w4_dtcal used correctly. Two scale free readings per head per position slice, both
comparable across slices of different size:

    TILT   |PIT mean - 0.5|, a bias in the conditional
    SHAPE  end density over middle density. above 1 is a U, the model is UNDER
           dispersed and too confident. below 1 is a hump, the model is OVER
           dispersed and hedges wider than the truth.

Registered thresholds on the SPEED head pooled over positions 0 to 3, which is the
launch. w4_dtcal's worst tilt over every slice it tested was 0.0062:

    TILT >= 0.02    a real bias at launch, three times anything previously seen
    TILT <= 0.008   no detectable bias at launch
    in between      MIXED
    BOUNDARY        within one bootstrap sd of a threshold the call is REFUSED

    SHAPE <= 0.95   OVER dispersed at launch, consistent with w4_position. the
                    model hedges wider than the truth at movement onset, which is
                    what too much launch texture looks like one step at a time.
    SHAPE >= 1.05   UNDER dispersed, which CONTRADICTS w4_position and means one
                    of the two runs is wrong.

SHAPE is read twice, against 1.0 and against the model's own mid sequence value,
so a global bias in the statistic cannot be mistaken for a launch specific one.

Standard caveat, stated rather than buried. PIT uniformity is a NECESSARY
condition for a correct conditional and not a sufficient one. Non uniformity
proves miscalibration; uniformity does not prove correctness.

The direction head is CIRCULAR. Uniformity of its PIT under a fixed ordering of
the alphabet is still a valid test of miscalibration, so TILT is computed for it.
The U versus hump reading requires an ordered alphabet, so SHAPE is not.

VALIDITY ARM, and the replacement for the broken floor. At every position a token
is also drawn from the model's OWN renormalised distribution and put through the
identical PIT path. That PIT is uniform BY CONSTRUCTION, so any tilt or shape it
shows is estimator error: bad CDF arithmetic, a renormalisation mistake, a masking
mistake. It cannot leak the answer because it never touches the real token. If the
validity arm fails, the run is reported as failed rather than interpreted.

Why this instrument carries no construction artefact. One teacher forced forward
pass over real held out sequences. At every position the model's own predicted
distribution is read against the REAL next token given the REAL history. Nothing
is generated and nothing hybrid exists, so the failure that withdrew arm G and
arm E cannot occur here. This is the w4_dtcal setup, the one measurement in this
workstream that has never had to be withdrawn.

DIAGNOSTIC ONLY, never a contract score. No serving change follows and no build is
authorised by any outcome. Phase conditioning and the spectral loss term remain
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
    EventARModel, N_DT_VALS, dt_ms_to_class, prefix_state,
)
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TH_NULL_CLASS, dth_lattice_to_class, s2_to_class,
)
from research.w4_timing import (  # noqa: E402
    MAX_T, N_TRAIN_DEFAULT, TRAIN_PICK_SEED,
)

TILT_REAL = 0.02
TILT_CLEAN = 0.008
OVER_DISPERSED = 0.95
UNDER_DISPERSED = 1.05
LAUNCH = (0, 4)         # the launch, positions 0 to 3
LATE_FROM = 12          # "mid sequence", for the relative shape read

# The validity arm is uniform by construction, so these are tight. They are what
# estimator error looks like, not what a wrong model looks like.
VALID_TILT = 0.005
VALID_SHAPE = 0.04

SLICES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 8), (8, 12),
          (12, 20), (20, 32), (32, MAX_T)]

N_REAL = {"s": S_PAD_CLASS, "th": TH_NULL_CLASS, "dt": N_DT_VALS}


def renorm(p_full, n_vals):
    """Drop the terminator class and renormalise the rest.

    A PIT over an alphabet that includes a terminator is not a PIT over the
    quantity of interest. w4_dtcal established that this mass is nil for dt; it is
    dropped here for every head regardless rather than assumed nil for a head that
    was never checked.
    """
    p = p_full[..., :n_vals]
    return p / p.sum(-1, keepdim=True).clamp(min=1e-12)


def pit_of(p, cls, rng, dev):
    """u = F(k-1) + v p(k), uniform iff the conditional is calibrated."""
    cdf = torch.cumsum(p, dim=-1)
    k = cls.clamp(max=p.shape[-1] - 1)
    p_k = p.gather(-1, k.unsqueeze(-1)).squeeze(-1)
    cdf_lo = cdf.gather(-1, k.unsqueeze(-1)).squeeze(-1) - p_k
    v = torch.from_numpy(rng.random(tuple(k.shape))).to(dev).float()
    return cdf_lo + v * p_k


def self_sample(p, gen):
    """Draw one token per position from the model's own renormalised law."""
    sh = p.shape
    flat = p.reshape(-1, sh[-1])
    k = torch.multinomial(flat, 1, generator=gen).squeeze(-1)
    return k.reshape(sh[:-1])


def pit_shape(u, nbins=20):
    """End density over middle density of a randomised PIT histogram.

    Above 1 is a U shape, the model is UNDER dispersed and too confident. Below 1
    is a hump, the model is OVER dispersed and hedging wider than the truth.
    """
    if len(u) < 200:
        return None
    h, _ = np.histogram(u, bins=nbins, range=(0.0, 1.0))
    d = h / h.sum() * nbins
    k = max(1, nbins // 10)
    ends = (d[:k].mean() + d[-k:].mean()) / 2.0
    mid = d[nbins // 2 - k: nbins // 2 + k].mean()
    if mid <= 0:
        return None
    return float(ends / mid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--out", default="research/w4_launch.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    rng = np.random.default_rng(args.seed + 3)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed + 77)

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
    nll = {h: np.zeros((B, MAX_T), dtype=np.float64) for h in heads}
    ent = {h: np.zeros((B, MAX_T), dtype=np.float64) for h in heads}
    pit = {h: np.zeros((B, MAX_T), dtype=np.float64) for h in heads}
    vpit = {h: np.zeros((B, MAX_T), dtype=np.float64) for h in heads}
    live = np.zeros((B, MAX_T), dtype=bool)
    # th is only defined where the event actually moved. A TH_NULL position
    # carries no direction to predict and is excluded from the th head alone.
    live_th = np.zeros((B, MAX_T), dtype=bool)

    print("  one teacher forced forward pass, every head, every position",
          flush=True)
    with torch.no_grad():
        for c0 in range(0, B, args.batch):
            sl = slice(c0, min(c0 + args.batch, B))
            s_b, th_b, dt_b = s_t[sl].to(dev), th_t[sl].to(dev), dt_t[sl].to(dev)
            cnd = cond_t[sl].to(dev)
            s_p, th_p, dt_p = model.shift_inputs(s_b, th_b, dt_b)
            # dt CLASSES, not float ms. The alphabet is whole milliseconds so the
            # two agree numerically, and classes are what model.sample and
            # w4_dtcal both pass. Consistency with the established path matters
            # more here than the parameter's name.
            st = prefix_state(s_b, th_b, dt_b, cnd)
            x = model.trunk(s_p, th_p, dt_p, st, cnd)

            lg = {"s": model.s_head(x),
                  "th": model.th_logits(x, s_b),
                  "dt": model.dt_logits(x, s_b, th_b)}
            true = {"s": s_b, "th": th_b, "dt": dt_b}
            for h in heads:
                lp = torch.log_softmax(lg[h], dim=-1)
                # Descriptive only. Their difference is NOT a divergence, see the
                # correction in HANDOFF, and nothing below reads it as one.
                nll[h][sl] = (-lp.gather(-1, true[h].unsqueeze(-1)).squeeze(-1)
                              .double().cpu().numpy())
                ent[h][sl] = (-(lp.exp() * lp).sum(-1).double().cpu().numpy())
                p = renorm(torch.softmax(lg[h], -1), N_REAL[h])
                pit[h][sl] = pit_of(p, true[h], rng, dev).double().cpu().numpy()
                vpit[h][sl] = pit_of(p, self_sample(p, gen), rng,
                                     dev).double().cpu().numpy()
            live[sl] = (s_b < S_PAD_CLASS).cpu().numpy()
            live_th[sl] = ((s_b < S_PAD_CLASS) & (th_b < TH_NULL_CLASS)
                           ).cpu().numpy()

    print(f"  {live.sum():,} live positions, {live_th.sum():,} of them with a "
          f"direction to predict\n", flush=True)

    def pool(store, h, rows, lo, hi):
        lv = live_th if h == "th" else live
        m = lv[rows, lo:hi]
        return store[h][rows, lo:hi][m]

    all_rows = np.arange(B)
    out = {"ckpt": args.ckpt, "n_rows": int(B), "seed": args.seed,
           "diagnostic_only": True,
           "pre_registered": "HANDOFF.md 2026-08-05, corrected same day",
           "thresholds": {"tilt_real": TILT_REAL, "tilt_clean": TILT_CLEAN,
                          "over_dispersed": OVER_DISPERSED,
                          "under_dispersed": UNDER_DISPERSED},
           "n_live": int(live.sum()), "n_live_th": int(live_th.sum()),
           "validity": {}, "pit_by_position": {}, "descriptive": {}}

    # VALIDITY ARM FIRST, before anything that touches the real token is printed.
    print("  VALIDITY arm. a token drawn from the model's OWN law, through the")
    print("  identical PIT path. uniform by construction, so any deviation here")
    print("  is estimator error and not a property of the model\n")
    print(f"    {'head':>6} {'n':>12} {'tilt':>9} {'shape':>9}")
    vfail = []
    for h in heads:
        u = pool(vpit, h, all_rows, 0, MAX_T)
        t = abs(float(u.mean()) - 0.5)
        sh = pit_shape(u)
        out["validity"][h] = {"tilt": t, "shape": sh, "n": int(len(u))}
        print(f"    {h:>6} {len(u):>12,} {t:>9.4f} "
              + (f"{sh:>9.3f}" if sh is not None else f"{'nan':>9}"))
        if t > VALID_TILT or (sh is not None and abs(sh - 1.0) > VALID_SHAPE):
            vfail.append(h)
    if vfail:
        print(f"\n  VALIDITY ARM FAILED on {', '.join(vfail)}. the PIT path is "
              f"broken, so nothing below would mean anything and it is not read.")
        out["verdict"] = f"FAILED, validity arm broken on {','.join(vfail)}"
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        return
    print("\n  validity arm PASSES on every head, the PIT path is sound\n")

    print("  randomised PIT against the REAL token, by position. tilt is a bias,")
    print("  shape above 1 is a U and UNDER dispersed, below 1 is a hump and OVER")
    print("  dispersed. the direction head is circular so it gets no shape\n")
    print(f"    {'positions':>12} {'n':>10} {'tilt s':>8} {'shp s':>8} "
          f"{'tilt th':>8} {'tilt dt':>8} {'shp dt':>8}")
    for lo, hi in SLICES:
        if not live[:, lo:hi].any():
            continue
        label = f"{lo}" if hi == lo + 1 else f"{lo} to {hi - 1}"
        r = {}
        for h in heads:
            u = pool(pit, h, all_rows, lo, hi)
            r[h] = {"tilt": abs(float(u.mean()) - 0.5) if len(u) else None,
                    "shape": pit_shape(u) if h != "th" else None,
                    "n": int(len(u))}
        out["pit_by_position"][label] = r

        def f4(v, w=8, p=4):
            return f"{v:>{w}.{p}f}" if v is not None else f"{'nan':>{w}}"
        print(f"    {label:>12} {r['s']['n']:>10,} "
              f"{f4(r['s']['tilt'])} {f4(r['s']['shape'], 8, 3)} "
              f"{f4(r['th']['tilt'])} "
              f"{f4(r['dt']['tilt'])} {f4(r['dt']['shape'], 8, 3)}")

    # Bootstrap over whole SEQUENCES, because positions within a sequence are not
    # independent. The signed mean is resampled and the tilt taken after, so the
    # sd is not the sd of a folded quantity.
    def stats(rows, lo, hi):
        u = pool(pit, "s", rows, lo, hi)
        if len(u) < 200:
            return None, None
        return float(u.mean()), pit_shape(u)

    m0, sh0 = stats(all_rows, *LAUNCH)
    m1, sh1 = stats(all_rows, LATE_FROM, MAX_T)
    bm, bs, brel = [], [], []
    for _ in range(args.draws):
        rs = rng.integers(0, B, B)
        a, b = stats(rs, *LAUNCH)
        c, d = stats(rs, LATE_FROM, MAX_T)
        if a is not None:
            bm.append(a)
            if b is not None:
                bs.append(b)
                if d is not None and d > 1e-6:
                    brel.append(b / d)
    sd_m = float(np.std(bm)) if len(bm) > 20 else float("nan")
    sd_s = float(np.std(bs)) if len(bs) > 20 else float("nan")
    tilt = abs(m0 - 0.5)
    rel = sh0 / sh1 if (sh0 is not None and sh1 and sh1 > 1e-6) else None
    sd_rel = float(np.std(brel)) if len(brel) > 20 else float("nan")

    out["launch"] = {"positions": list(LAUNCH), "pit_mean": m0, "tilt": tilt,
                     "tilt_sd": sd_m, "shape": sh0, "shape_sd": sd_s,
                     "mid_shape": sh1, "shape_relative": rel,
                     "shape_relative_sd": sd_rel}

    print(f"\n  LAUNCH, positions {LAUNCH[0]} to {LAUNCH[1] - 1}, speed head")
    print(f"    PIT mean       {m0:.4f}   tilt {tilt:.4f}  bootstrap sd {sd_m:.4f}")
    print(f"    shape          {sh0:.3f}    bootstrap sd {sd_s:.3f}")
    print(f"    mid sequence   {sh1:.3f}    relative {rel:.3f}  sd {sd_rel:.3f}")

    margin = min(abs(tilt - TILT_REAL), abs(tilt - TILT_CLEAN))
    if margin < sd_m:
        verdict = (f"BOUNDARY on the speed head tilt. {tilt:.4f}, nearest "
                   f"threshold {margin:.4f} away against a bootstrap sd of "
                   f"{sd_m:.4f}, so the threshold call is REFUSED and this is "
                   f"reported as the in between case.")
    elif tilt >= TILT_REAL:
        verdict = (f"THE LAUNCH CONDITIONAL IS BIASED. Speed head tilt "
                   f"{tilt:.4f} >= {TILT_REAL}, against a worst previously "
                   f"observed tilt of 0.0062. The model's one step speed "
                   f"prediction at movement onset is itself wrong, and a wrong "
                   f"conditional is addressable in training.")
    elif tilt <= TILT_CLEAN:
        verdict = (f"NO BIAS AT LAUNCH. Speed head tilt {tilt:.4f} <= "
                   f"{TILT_CLEAN}. The one step conditional at onset is not "
                   f"detectably biased, so w4_position's launch excess is not a "
                   f"shifted conditional. Either it is a dispersion fault, which "
                   f"the shape read below decides, or it lives in the joint.")
    else:
        verdict = (f"MIXED on the speed head tilt. {tilt:.4f} sits between "
                   f"{TILT_CLEAN} and {TILT_REAL}. Report the curve and the "
                   f"number, neither alone.")
    out["verdict"] = verdict
    print(f"\n  -> {verdict}")

    def read_shape(v, sd, what):
        if v is None:
            return f"{what} not computable"
        margin = min(abs(v - OVER_DISPERSED), abs(v - UNDER_DISPERSED))
        if margin < sd:
            return (f"{what} {v:.3f} is BOUNDARY, {margin:.3f} from the nearest "
                    f"threshold against a bootstrap sd of {sd:.3f}, call REFUSED")
        if v <= OVER_DISPERSED:
            return (f"{what} {v:.3f} <= {OVER_DISPERSED}, OVER dispersed. "
                    f"Consistent with w4_position: the model hedges wider than "
                    f"the truth at onset, which is what too much launch texture "
                    f"looks like one step at a time.")
        if v >= UNDER_DISPERSED:
            return (f"{what} {v:.3f} >= {UNDER_DISPERSED}, UNDER dispersed. This "
                    f"CONTRADICTS w4_position and one of the two runs is wrong. "
                    f"Registered in advance so it cannot be reinterpreted.")
        return (f"{what} {v:.3f} sits between {OVER_DISPERSED} and "
                f"{UNDER_DISPERSED}, not detectably mis dispersed either way.")

    d_abs = read_shape(sh0, sd_s, "absolute shape at launch")
    d_rel = read_shape(rel, sd_rel, "shape at launch relative to mid sequence")
    out["dispersion_absolute"] = d_abs
    out["dispersion_relative"] = d_rel
    print(f"  -> {d_abs}")
    print(f"  -> {d_rel}")

    print("\n  DESCRIPTIVE ONLY, mean NLL and mean predicted entropy by position.")
    print("  their DIFFERENCE IS NOT A DIVERGENCE, it is KL plus an entropy gap,")
    print("  and nothing above is decided on it. see the correction in HANDOFF\n")
    print(f"    {'positions':>12} {'nll s':>8} {'ent s':>8} {'nll th':>8} "
          f"{'ent th':>8} {'nll dt':>8} {'ent dt':>8}")
    for lo, hi in SLICES:
        if not live[:, lo:hi].any():
            continue
        label = f"{lo}" if hi == lo + 1 else f"{lo} to {hi - 1}"
        row = {}
        for h in heads:
            a = pool(nll, h, all_rows, lo, hi)
            b = pool(ent, h, all_rows, lo, hi)
            row[h] = {"nll": float(a.mean()) if len(a) else None,
                      "entropy": float(b.mean()) if len(b) else None}
        out["descriptive"][label] = row
        print(f"    {label:>12} " + " ".join(
            f"{row[h][k]:>8.3f}" if row[h][k] is not None else f"{'nan':>8}"
            for h in heads for k in ("nll", "entropy")))

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.out}")
    print("""
  DIAGNOSTIC ONLY, never a contract score. no serving change follows and
  no build is authorised by any outcome. phase conditioning and the
  spectral loss term remain NOT AUTHORISED.
  PIT uniformity is a NECESSARY condition for a correct conditional and not
  a sufficient one. non uniformity proves miscalibration, uniformity does
  not prove correctness.
  the direction head is circular, so it gets a tilt and no shape rather than
  a shape being computed and read as if it meant something.""")


if __name__ == "__main__":
    main()
