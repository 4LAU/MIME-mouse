"""w4_poskl. Does the AR model's error actually grow with position?

Registered in /home/aaronadmin/w4_arms/poskl_prereg.md.

The case for the diffusion arm rests on COMPOUNDING: a small per step bias
accumulating over about 39 events. Every piece of evidence for that in the
record is indirect. This measures the thing the word claims, which is that the
discrepancy is small early in a trajectory and larger late in it.

CPU only. Reads streams that already exist. Generates nothing, trains nothing.
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

from models.event_ar import class_to_dt_ms                      # noqa: E402
from models.event_stream_polar import (S_PAD_CLASS, TH_BINS,  # noqa: E402
                                       TH_NULL_CLASS, TICK_CLASS,
                                       class_to_dtheta, class_to_speed)
from w4_detcap import corpus_ids, corpus_tokens, model_tokens   # noqa: E402

KMAX = 40          # positions 0..39, and every row must be at least this long
NBIN = 16          # quantile bins per channel, cut points from HUMAN only
NULL_SEEDS = (101, 102, 103, 104, 105, 106, 107, 108)


def cuts_from(x, nbin):
    """Quantile cut points defined on the human reference and nowhere else."""
    q = np.linspace(0, 100, nbin + 1)[1:-1]
    return np.unique(np.percentile(x, q))


def tv(a, b, nbin):
    """Total variation between two binned columns. Bounded, and its finite n
    bias is exactly what the null arm measures, so it never needs correcting
    by hand."""
    pa = np.bincount(a, minlength=nbin).astype(np.float64)
    pb = np.bincount(b, minlength=nbin).astype(np.float64)
    pa /= pa.sum()
    pb /= pb.sum()
    return 0.5 * float(np.abs(pa - pb).sum())


def slope(y):
    """OLS slope of y on its own index. Reported per position, so a value of
    1e-4 means the divergence grows by 0.004 over the 40 positions read."""
    k = np.arange(len(y), dtype=np.float64)
    return float(np.polyfit(k, np.asarray(y, dtype=np.float64), 1)[0])


def cumulative(s, th, dt):
    """The cumulative counterpart of each token channel, reconstructed from the
    token classes alone with no decoder involved.

    AMENDMENT 3. A per step bias compounds in the RUNNING SUM of the channel it
    biases, so every token channel gets its running sum. `disp` alone was close
    to blind to a heading drift, because a drift rotates a trajectory far more
    than it lengthens it, and the power control caught that.

    The start angle is set to zero for both populations. That is exact rather
    than an approximation for `disp`: rotating every step by a common angle
    leaves the magnitude of the sum unchanged. `cumhead` is likewise read as a
    displacement from the start heading, so the common angle drops out.
    """
    sp = class_to_speed(torch.from_numpy(s)).numpy()
    dth = class_to_dtheta(torch.from_numpy(th)).numpy()
    ms = class_to_dt_ms(torch.from_numpy(dt)).numpy()
    head = np.cumsum(dth, axis=1)
    step = sp * ms
    x = np.cumsum(step * np.cos(head), axis=1)
    y = np.cumsum(step * np.sin(head), axis=1)
    return {"cumhead": head, "pathlen": np.cumsum(step, axis=1),
            "disp": np.hypot(x, y)}


def prep(s, th, dt, L, rng=None):
    """Rows of length >= KMAX, trimmed to the first KMAX positions. Every
    position is then computed on exactly the same rows, so survivorship cannot
    manufacture a trend.

    SHUFFLE IS NOT OPTIONAL. corpus_tokens SORTS the ids it is handed, so
    slicing the rows it returns into halves compares low indexed corpus rows
    against high indexed ones, which is a systematic subpopulation split rather
    than a random one. Caught by the power control, which read the same
    divergence at every planted size because the split, not the plant, was
    producing it. Same failure as the shuffle before score_features rule.
    """
    keep = np.asarray(L) >= KMAX
    s, th, dt = s[keep, :KMAX], th[keep, :KMAX], dt[keep, :KMAX]
    if rng is not None:
        o = rng.permutation(len(s))
        s, th, dt = s[o], th[o], dt[o]
    return s, th, dt, len(s)


CHANNELS = ("s", "th", "dt", "cumhead", "pathlen", "disp")
TOKEN_CH = ("s", "th", "dt")
CUM_CH = ("cumhead", "pathlen", "disp")


def channels(s, th, dt):
    """The six position resolved quantities the arm reads: three token
    channels and their three cumulative counterparts."""
    return {"s": s, "th": th, "dt": dt, **cumulative(s, th, dt)}


def make_cuts(ch):
    """Cut points pooled over every position, so the alphabet is the same at
    k = 0 and k = 39 and only the DISTRIBUTION over it is allowed to move."""
    cuts = {}
    for name, v in ch.items():
        if name == "th":
            live = v[v < TH_NULL_CLASS]
            cuts[name] = cuts_from(live, NBIN)
        else:
            cuts[name] = cuts_from(v.reshape(-1), NBIN)
    return cuts


def digitize(ch, cuts):
    out = {}
    for name, v in ch.items():
        if name == "th":
            # NULL is a real event type, not a missing value, so it gets its
            # own bin rather than being folded into the top one.
            b = np.digitize(v, cuts[name])
            out[name] = np.where(v >= TH_NULL_CLASS, len(cuts[name]) + 1, b)
        else:
            out[name] = np.digitize(v, cuts[name])
    return out


def curve(A, B, nbin):
    return [tv(A[:, k], B[:, k], nbin) for k in range(KMAX)]


def arm(hs, ms, cuts, nbin):
    """One TV curve and one slope per channel."""
    hb, mb = digitize(hs, cuts), digitize(ms, cuts)
    out = {}
    for name in hs:
        c = curve(hb[name], mb[name], nbin)
        out[name] = {"tv": c, "slope": slope(c), "tv_mean": float(np.mean(c)),
                     "tv_first5": float(np.mean(c[:5])),
                     "tv_last5": float(np.mean(c[-5:]))}
    return out



# ------------------------------------------------------------ power control --
# A flat reading is worthless until the instrument is shown to see a planted
# effect of comparable size. Registered as AMENDMENT 2 in poskl_prereg.md.
#
# Both arms are built from HUMAN streams, so the ONLY difference between the two
# populations being compared is the perturbation itself.


def plant_compound(s, th, dt, rng, p):
    """Per step heading bias. Heading is a cumulative sum, so this makes the
    heading error random walk WITH DRIFT and the displacement error grow faster
    than linearly. Compounding in its textbook form."""
    live = th < TH_NULL_CLASS
    bump = (rng.random(th.shape) < p) & live
    return s, np.where(bump, (th + 1) % TH_BINS, th), dt


def plant_stationary(s, th, dt, rng, q):
    """Per position speed bias applied independently at every position, with
    no cumulation in token space. Displacement is a cumulative sum so this
    still moves disp, but it moves it MULTIPLICATIVELY: both distributions
    scale with k together, so the divergence should sit flat rather than grow.
    That contrast is the whole point of having two planted arms."""
    live = (s > TICK_CLASS) & (s < S_PAD_CLASS)
    bump = (rng.random(s.shape) < q) & live
    return np.where(bump, np.minimum(s + 1, S_PAD_CLASS - 1), s), th, dt


def size_to(fn, A, Braw, cuts, nbin, target, grid, rng):
    """Pick the perturbation size whose MEAN disp divergence is closest to what
    the model arm actually shows. Sizing on the mean and reading the SLOPE
    keeps the two independent."""
    best = None
    for g in grid:
        B = channels(*fn(*Braw, rng, g))
        m = float(np.mean(curve(digitize(A, cuts)["disp"],
                                digitize(B, cuts)["disp"], nbin)))
        print(f"      size {g:<8.4f} disp mean tv {m:.4f}"
              + ("   (unperturbed baseline)" if g == 0 else ""), flush=True)
        # size 0 is printed as the control's own control: it must land near the
        # null level, and it is never eligible to be CHOSEN as the plant.
        if g == 0:
            continue
        if best is None or abs(m - target) < abs(best[1] - target):
            best = (g, m, B)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="+",
                    default=[f"research/w4_texcover_streams_s{d}.npz"
                             for d in (0, 1)])
    ap.add_argument("--label", default="event_ar rollout at the optimum")
    ap.add_argument("--out", default="research/w4_poskl_results.json")
    a = ap.parse_args()

    print("w4_poskl. position resolved divergence, CPU only, nothing "
          "generated.")
    print(f"  arm: {a.label}\n")

    ms, mth, mdt = [], [], []
    for sp in a.streams:
        s, th, dt, _, L = model_tokens(sp)
        s, th, dt, n = prep(s, th, dt, L)
        print(f"  model {sp.split('/')[-1]}  {n} rows of length >= {KMAX}")
        ms.append(s); mth.append(th); mdt.append(dt)
    MS, MTH, MDT = np.vstack(ms), np.vstack(mth), np.vstack(mdt)
    n_model = len(MS)

    # human reference, oversampled because the length >= KMAX cut discards rows
    rng = np.random.default_rng(7)
    HS = HTH = HDT = None
    want = n_model
    ids = corpus_ids(rng, int(want * 4.0))
    hs, hth, hdt, _, hL = corpus_tokens(ids)
    hs, hth, hdt, nh = prep(hs, hth, hdt, hL, rng)
    assert nh >= want, f"only {nh} human rows of length >= {KMAX}, need {want}"
    HS, HTH, HDT = hs[:want], hth[:want], hdt[:want]
    print(f"  human reference {len(HS)} rows, drawn from {nh} eligible\n")

    hch = channels(HS, HTH, HDT)
    mch = channels(MS, MTH, MDT)
    cuts = make_cuts(hch)                   # HUMAN only, never the model
    nbin = NBIN + 2                         # digitize range plus the th NULL bin

    res = arm(hch, mch, cuts, nbin)
    print("  MODEL arm, total variation against the human reference")
    print(f"    {'channel':<8}{'first5':>9}{'last5':>9}{'mean':>9}"
          f"{'slope/pos':>12}")
    for name in CHANNELS:
        r = res[name]
        print(f"    {name:<8}{r['tv_first5']:>9.4f}{r['tv_last5']:>9.4f}"
              f"{r['tv_mean']:>9.4f}{r['slope']:>12.6f}")

    # ---- NULL arm, human split against human, same n per side --------------
    print(f"\n  NULL, human against human, {len(NULL_SEEDS)} seeds, "
          f"{want} rows per side")
    nulls = {name: [] for name in res}
    for sd in NULL_SEEDS:
        r2 = np.random.default_rng(sd)
        i2 = corpus_ids(r2, int(want * 8.5))
        s2, t2, d2, _, L2 = corpus_tokens(i2)
        s2, t2, d2, n2 = prep(s2, t2, d2, L2, r2)
        assert n2 >= 2 * want, f"null seed {sd}: {n2} rows, need {2 * want}"
        A = channels(s2[:want], t2[:want], d2[:want])
        B = channels(s2[want:2 * want], t2[want:2 * want], d2[want:2 * want])
        rr = arm(A, B, cuts, nbin)
        for name in nulls:
            nulls[name].append(rr[name]["slope"])
        print(f"    seed {sd}  " + "  ".join(
            f"{name} {rr[name]['slope']:+.6f}" for name in
            CHANNELS), flush=True)

    print(f"\n  PRIMARY. slope of TV on position, model against the null band")
    print(f"    {'channel':<8}{'model':>11}{'null mean':>11}{'null sd':>10}"
          f"{'corrected':>11}{'z':>8}   reading")
    out = {"config": dict(kmax=KMAX, nbin=NBIN, n_model=n_model,
                          n_human=int(want), streams=a.streams,
                          label=a.label, null_seeds=list(NULL_SEEDS)),
           "model": res, "null": {}, "z": {}}
    readings = {}
    for name in CHANNELS:
        v = np.array(nulls[name])
        mu, sd = float(v.mean()), float(v.std(ddof=1))
        # AMENDMENT 1: the statistic is NULL CORRECTED. the null does not sit
        # at zero, and requiring the raw slope to be positive would fold that
        # artefact back into the reading.
        corr = res[name]["slope"] - mu
        z = corr / sd if sd > 0 else float("nan")
        rd = ("GROWING" if z > 3 and corr > 0
              else "flat" if abs(z) <= 2 else "ambiguous")
        readings[name] = rd
        out["null"][name] = {"slopes": nulls[name], "mean": mu, "sd": sd}
        out["z"][name] = z
        out["model"][name]["corrected_slope"] = corr
        print(f"    {name:<8}{res[name]['slope']:>+11.6f}{mu:>+11.6f}"
              f"{sd:>10.6f}{corr:>+11.6f}{z:>8.2f}   {rd}")

    tok = [readings[n] for n in TOKEN_CH]
    verdict = ("GROWING" if "GROWING" in tok
               else "FLAT" if all(t == "flat" for t in tok) else "AMBIGUOUS")
    print(f"\n  TOKEN VERDICT   {verdict}")
    print("  PATH READOUT    " + "   ".join(f"{n} {readings[n]}"
                                             for n in CUM_CH))
    print("  (the path readout is only readable if the power control below "
          "passes on that channel)")
    out["verdict"] = verdict
    out["readings"] = readings

    # ---- POWER CONTROL, AMENDMENT 2 -------------------------------------
    # The disp channel is the one whose FLAT reading would lower the prior on
    # the arm now training, so it is the one that most needs this.
    print("\n  POWER CONTROL. two planted effects, both from human streams,")
    print("  each sized so its mean divergence matches the model arm's own on")
    print("  the channel being read.")
    pr = np.random.default_rng(909)
    pid = corpus_ids(pr, int(want * 8.5))
    ps, pt, pd, _, pL = corpus_tokens(pid)
    ps, pt, pd, pn = prep(ps, pt, pd, pL, pr)
    assert pn >= 2 * want, f"power arm: {pn} rows, need {2 * want}"
    PA = channels(ps[:want], pt[:want], pd[:want])
    Braw = (ps[want:2 * want], pt[want:2 * want], pd[want:2 * want])
    PAd = digitize(PA, cuts)

    out["power"] = {}
    for tag, fn, grid in (
            ("COMPOUNDING", plant_compound,
             (0.0, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30, 0.60)),
            ("STATIONARY", plant_stationary,
             (0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0))):
        print(f"\n    {tag}")
        prng = np.random.default_rng(4242)
        sized = {}
        for g in grid:
            B = digitize(channels(*fn(*Braw, prng, g)), cuts)
            row = {}
            for name in CUM_CH:
                c = curve(PAd[name], B[name], nbin)
                row[name] = (float(np.mean(c)), c)
            sized[g] = row
            print("      size {:<6.3f}".format(g)
                  + "  ".join(f"{n} {row[n][0]:.4f}" for n in CUM_CH)
                  + ("   (unperturbed baseline)" if g == 0 else ""), flush=True)
        out["power"][tag] = {}
        for name in CUM_CH:
            target = res[name]["tv_mean"]
            mu_c = out["null"][name]["mean"]
            sd_c = out["null"][name]["sd"]
            # size 0 is the control's own control. it is printed but is never
            # eligible to be chosen as the plant.
            g = min((k for k in grid if k > 0),
                    key=lambda k: abs(sized[k][name][0] - target))
            c = sized[g][name][1]
            sl = slope(c)
            zc = (sl - mu_c) / sd_c
            rd = ("GROWING" if zc > 3 and (sl - mu_c) > 0
                  else "flat" if abs(zc) <= 2 else "ambiguous")
            print(f"      {name:<8} size {g:<6.3f} mean tv {np.mean(c):.4f} "
                  f"vs target {target:.4f}   first5 {np.mean(c[:5]):.4f} "
                  f"last5 {np.mean(c[-5:]):.4f}   corrected {sl - mu_c:+.6f} "
                  f"z {zc:+.2f}  -> {rd}")
            out["power"][tag][name] = dict(size=g, mean_tv=float(np.mean(c)),
                                           slope=sl, corrected=sl - mu_c,
                                           z=zc, reading=rd, tv=c)

    # POWERED means the instrument separates the two mechanisms: the plant that
    # compounds registers, and the plant that does not compound does not.
    powered_ch = [n for n in CUM_CH
                  if out["power"]["COMPOUNDING"][n]["reading"] == "GROWING"
                  and out["power"]["STATIONARY"][n]["reading"] != "GROWING"]
    out["powered_channels"] = powered_ch
    out["powered"] = bool(powered_ch)
    print(f"\n  POWER  {'POWERED' if powered_ch else 'UNPOWERED'}"
          + (f"  on {powered_ch}" if powered_ch else ""))
    if powered_ch:
        print("  the cumulative readings on those channels are informative:")
        for n in powered_ch:
            print(f"    {n:<8} model {readings[n]}")
    else:
        print("  the cumulative readings carry NO information. this arm "
              "failed to measure what it set out to measure.")

    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
