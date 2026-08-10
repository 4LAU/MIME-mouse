"""Are the model's errors INDEPENDENT across steps, as a correct model requires?

PRE REGISTERED in HANDOFF, thresholds fixed before this file existed.

Everything measured so far is a MARGINAL property: the conditional is uniform
within this slice, within that slice. Marginal calibration at every step does not
imply a correct joint, and what it fails to constrain is exactly the DEPENDENCE
between steps. That is where a three percent per step error becomes a movement
level defect.

Under a correct model the randomised PIT values u(t) are not merely uniform, they
are INDEPENDENT across t. That is the Rosenblatt transform and it is an identity.
So any autocorrelation in u proves the joint is wrong, without knowing p, without
generating anything, and without the construction artefact that withdrew arms G
and E.

That randomisation is what makes u exactly uniform on a discrete alphabet, and it
is also what dilutes the very dependence the test is looking for, because it adds
noise that is independent by construction. Measured on the smoke test, the
randomised series retains fifteen percent of the speed signal and one percent of
the timing signal. A null on an instrument that blind is close to worthless, so
the primary is the CENTERED MID PIT instead, which carries the same signal with
none of the added noise. See the THIRD AMENDMENT in HANDOFF.

    m(t) = F(k(t) - 1) + p(k(t))/2          the mid PIT, no added noise
    e(t) = m(t) - 1/2                       the level residual
    r(t) = |m(t) - 1/2| - E[|m - 1/2| ; p]  the volatility residual

Both are centred on their own conditional mean, which is available in closed form
from p at that step. That centring is what makes them exact. Under a correct model
E[e(t+1) | history] = 0 and E[r(t+1) | history] = 0, and both e(t) and r(t) are
functions of the history at t+1, so every autocorrelation at every lag is EXACTLY
zero. Same identity as the Rosenblatt one, undiluted.

    rho_level = autocorrelation of e(t)
    rho_vol   = autocorrelation of r(t)

rho_vol is the primary. It measures whether being surprised at step t predicts
being surprised at step t+1 BEYOND what the model already expects, which is
volatility clustering, which is texture. A person's jitter comes in bursts. A
model that reproduces the marginal amount of jitter but not its clustering passes
every marginal instrument and assembles a movement that is smooth in the wrong
way.

    either rho >= 0.05   MATERIAL residual dependence, addressable in training
    both rho <= 0.01     no residual dependence at lag 1
    otherwise            MIXED
    BOUNDARY             within one bootstrap sd of a threshold the call is REFUSED

These are MATERIALITY thresholds, not significance thresholds. At five million
positions the analytic null sd is about 0.0005 and almost anything is significant,
so the question is whether it is big enough to matter. Bootstrap is over whole
SEQUENCES, since positions inside one are the very thing being measured.

A REJECTED alternative, recorded because it was built and run before it was seen
through. The obvious null for the mid PIT is the same statistic on a token drawn
from the model rather than from the truth, and the difference between them. That
difference is BIASED. In the self sampled arm the drawn token is discarded, so it
cannot influence the next step's predictive distribution. In the real arm the
token IS the history, so a surprising token at t moves p at t+1, and the resulting
correlation is present even under a perfect model. The bias falls entirely on the
volatility series, which is the primary, and the smoke test showed exactly that
signature: all three level series flat, both volatility series positive. Centring
on the conditional mean removes it, which is why the statistic above is a residual
rather than a difference.

VALIDITY ARM, exact here. The self sampled tokens' residuals have conditional mean
zero BY CONSTRUCTION and their draws are independent across steps, so their
autocorrelation is the exact null for the estimator. Two gates, both must hold, and
an unmeasurable sd is a FAILURE rather than a pass:

    lag 1      |z| <= 2.0   a single test, at the lag the verdict is read on
    every lag  |z| <= 3.0   a family of eight, about two percent family wise

The registration asked for 2 sd at every lag. That is a multiplicity error: a
maximum over eight tests against a per test threshold fails about a third of the
time on a PERFECT estimator. See the AMENDMENT in HANDOFF. Both defects were found
on the smoke test, from the null arm's own arithmetic, before any real token
statistic was printed.

The randomised series is still computed and reported, since it is what the original
registration named, and the ratio between the two is how the dilution is measured
rather than assumed.

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
    EventARModel, dt_ms_to_class, prefix_state,
)
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TH_NULL_CLASS, dth_lattice_to_class, s2_to_class,
)
from research.w4_timing import (  # noqa: E402
    MAX_T, N_TRAIN_DEFAULT, TRAIN_PICK_SEED,
)
from research.w4_launch import (  # noqa: E402
    N_REAL, pit_of, renorm, self_sample,
)

MATERIAL = 0.05
NEGLIGIBLE = 0.01
VALID_SD = 2.0          # single test, at lag 1, the lag the verdict uses
VALID_FAMILY_SD = 3.0   # family of eight lags, about two percent family wise
VALID_FLOOR = NEGLIGIBLE / 5    # a fifth of the smallest meaningful level, 0.002
LAGS = (1, 2, 3, 4, 5, 6, 7, 8)


def acf_terms(x, live, lag):
    """Sums needed for a lag k autocorrelation, WITHIN sequences only.

    Returns (sum_xy, sum_x, sum_y, sum_xx, sum_yy, n) so a bootstrap over
    sequences is a sum over rows and never re walks the positions.
    """
    a, b = x[:, :-lag], x[:, lag:]
    m = live[:, :-lag] & live[:, lag:]
    a = np.where(m, a, 0.0)
    b = np.where(m, b, 0.0)
    return (np.einsum("ij,ij->i", a, b), a.sum(1), b.sum(1),
            np.einsum("ij,ij->i", a, a), np.einsum("ij,ij->i", b, b),
            m.sum(1).astype(np.float64))


def acf_from(terms, rows):
    """Pearson correlation of the lagged pairs on a set of rows."""
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
    ap.add_argument("--out", default="research/w4_indep.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    rng = np.random.default_rng(args.seed + 11)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed + 137)

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
    print(f"  {B:,} rows at least 12 events, the same rows w4_progress used\n",
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
    # The MID PIT, u = F(k-1) + p(k)/2, carries no added randomisation noise. It
    # is not exactly uniform on discrete data so it is not used for the verdict,
    # but its autocorrelation is UNDILUTED, and the ratio against the randomised
    # series measures empirically how conservative this test is. Without that
    # number "no dependence" is a claim about an instrument of unknown power.
    mpit = {h: np.zeros((B, MAX_T), dtype=np.float64) for h in heads}
    # The same statistic on a token drawn from the model. This is the VALIDITY
    # arm, not the null for a difference: see the rejected alternative in the
    # module docstring for why the difference is biased and the residual is not.
    mspit = {h: np.zeros((B, MAX_T), dtype=np.float64) for h in heads}
    # E[|mid - 0.5|] under p, in closed form at every position. Subtracting it is
    # what makes the volatility series conditionally mean zero, and therefore
    # exactly uncorrelated under a correct model at every lag. E[mid] is 1/2
    # identically, so the level series needs no stored counterpart.
    mu = {h: np.zeros((B, MAX_T), dtype=np.float64) for h in heads}
    live = np.zeros((B, MAX_T), dtype=bool)
    live_th = np.zeros((B, MAX_T), dtype=bool)

    print("  one teacher forced forward pass, identical to w4_progress's",
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
                k_self = self_sample(p, gen)
                pit[h][sl] = pit_of(p, true[h], rng, dev).double().cpu().numpy()
                vpit[h][sl] = pit_of(p, k_self, rng, dev).double().cpu().numpy()
                cdf = torch.cumsum(p, dim=-1)

                def mid(k):
                    k = k.clamp(max=p.shape[-1] - 1)
                    p_k = p.gather(-1, k.unsqueeze(-1)).squeeze(-1)
                    cdf_k = cdf.gather(-1, k.unsqueeze(-1)).squeeze(-1)
                    return (cdf_k - 0.5 * p_k).double().cpu().numpy()

                mpit[h][sl] = mid(true[h])
                # The SAME drawn token, so the two mid arms differ only in
                # whether the token came from the truth or from the model.
                mspit[h][sl] = mid(k_self)
                m_all = cdf - 0.5 * p
                mu[h][sl] = ((p * (m_all - 0.5).abs()).sum(-1)
                             .double().cpu().numpy())
            live[sl] = (s_b < S_PAD_CLASS).cpu().numpy()
            live_th[sl] = ((s_b < S_PAD_CLASS) & (th_b < TH_NULL_CLASS)
                           ).cpu().numpy()

    print(f"  {live.sum():,} live positions, {live_th.sum():,} of them with a "
          f"direction to predict\n", flush=True)

    # Every statistic is precomputed as per row sums, so a bootstrap draw is a
    # sum over rows and never re walks five million positions.
    all_rows = np.arange(B)

    # Four kinds. level and vol are the CENTERED MID residuals, the primary,
    # undiluted and exactly mean zero given the history. rlevel and rvol are the
    # randomised series the original registration named, kept so the dilution is
    # a measured number rather than an assumption.
    KINDS = ("level", "vol", "rlevel", "rvol")

    def build(h, kind, arm):
        u = {"real": (pit[h], mpit[h]), "self": (vpit[h], mspit[h])}[arm]
        if kind == "level":
            return u[1] - 0.5
        if kind == "vol":
            return np.abs(u[1] - 0.5) - mu[h]
        if kind == "rlevel":
            return u[0] - 0.5
        return np.abs(u[0] - 0.5) - 0.25   # E|u - 1/2| = 1/4 for a uniform u

    # Built one at a time and freed, so peak memory holds one derived array
    # rather than twenty four of them.
    terms, series = {}, set()
    for h in heads:
        lv = live_th if h == "th" else live
        for kind in KINDS:
            if h == "th" and kind in ("vol", "rvol"):
                continue
            for arm in ("real", "self"):
                x = build(h, kind, arm)
                series.add((h, kind, arm))
                for lag in LAGS:
                    terms[((h, kind, arm), lag)] = acf_terms(x, lv, lag)
                del x

    def boot_sd(key, lag, draws):
        v = []
        for _ in range(draws):
            rs = rng.integers(0, B, B)
            r = acf_from(terms[(key, lag)], rs)
            if r is not None:
                v.append(r)
        # len(v) > 10, not > draws. An earlier version asked for more draws than
        # it requested, returned nan, and the nan then failed every "sd > 0" test
        # downstream, which silently disabled the validity gate rather than
        # tripping it. A gate that can no op is worse than no gate.
        return float(np.std(v)) if len(v) > 10 else float("nan")

    out = {"ckpt": args.ckpt, "n_rows": int(B), "seed": args.seed,
           "diagnostic_only": True, "pre_registered": "HANDOFF.md 2026-08-05",
           "thresholds": {"material": MATERIAL, "negligible": NEGLIGIBLE},
           "lags": list(LAGS), "n_live": int(live.sum()),
           "validity": {}, "real": {}}

    # VALIDITY ARM FIRST. These PIT values are i.i.d. by construction, so this is
    # the exact null for the statistic, dilution included.
    print("  VALIDITY arm. residuals of a token drawn from the model, mean zero")
    print("  BY CONSTRUCTION and drawn independently at each step, so this is the")
    print("  exact null for the estimator on all four series\n")
    print(f"    {'series':>12} " + " ".join(f"{'lag' + str(k):>8}" for k in LAGS))
    vfail = []
    for h in heads:
        for kind in KINDS:
            key = (h, kind, "self")
            if key not in series:
                continue
            row, worst, z1 = [], 0.0, float("inf")
            for lag in LAGS:
                r = acf_from(terms[(key, lag)], all_rows)
                sd = boot_sd(key, lag, max(40, args.draws // 8))
                row.append({"lag": lag, "rho": r, "sd": sd})
                # An unmeasurable sd is a FAILURE, not a pass. This is the arm
                # that decides whether anything below gets read.
                # MAGNITUDE floor, not a third significance threshold. An
                # estimator bias bounded below a fifth of the NEGLIGIBLE level
                # cannot create, mask or move a reading at 0.01 or at 0.05. Ten
                # series times eight lags is eighty tests, and chasing the right
                # sigma for eighty would rationalise the gate out of existence.
                # See the SECOND AMENDMENT in HANDOFF.
                z = (float("inf") if (r is None or not np.isfinite(sd) or sd <= 0)
                     else (0.0 if abs(r) < VALID_FLOOR else abs(r) / sd))
                worst = max(worst, z)
                if lag == 1:
                    z1 = z
            out["validity"][f"{h}:{kind}"] = row
            print(f"    {h + ' ' + kind:>12} " +
                  " ".join(f"{d['rho']:>+8.4f}" if d["rho"] is not None
                           else f"{'nan':>8}" for d in row) +
                  f"   lag1 {z1:.1f} sd, worst {worst:.1f} sd")
            # Two gates. lag 1 is a single test at the lag the verdict is read
            # on, so 2 sd. The family of eight gets 3 sd, because a max over
            # eight against a per test threshold fails about a third of the time
            # on a PERFECT estimator. See the AMENDMENT in HANDOFF.
            if not (z1 <= VALID_SD and worst <= VALID_FAMILY_SD):
                vfail.append(f"{h}:{kind}")
    if vfail:
        print(f"\n  VALIDITY ARM FAILED on {', '.join(vfail)}. the estimator "
              f"manufactures dependence, so nothing below would mean anything.")
        out["verdict"] = f"FAILED, validity arm autocorrelated on {','.join(vfail)}"
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        return
    print("\n  validity arm PASSES at every lag, the statistic is sound\n")

    print("  REAL tokens. autocorrelation of the residual within sequences. under")
    print("  a correct model these are EXACTLY zero, so anything here proves the")
    print("  joint is wrong. vol is surprise magnitude beyond what the model")
    print("  already expects, which is volatility clustering. r prefixed rows are")
    print("  the randomised series, same quantity through a blunter instrument\n")
    print(f"    {'series':>12} " + " ".join(f"{'lag' + str(k):>8}" for k in LAGS))
    for h in heads:
        for kind in KINDS:
            key = (h, kind, "real")
            if key not in series:
                continue
            row = [{"lag": lag, "rho": acf_from(terms[(key, lag)], all_rows)}
                   for lag in LAGS]
            out["real"][f"{h}:{kind}"] = row
            print(f"    {h + ' ' + kind:>12} " +
                  " ".join(f"{d['rho']:>+8.4f}" if d["rho"] is not None
                           else f"{'nan':>8}" for d in row))

    prim = {}
    for kind in KINDS:
        key = ("s", kind, "real")
        r = acf_from(terms[(key, 1)], all_rows)
        sd = boot_sd(key, 1, args.draws)
        prim[kind] = {"rho": r, "bootstrap_sd": sd}
        out[f"rho_{kind}_lag1_speed"] = prim[kind]
    print(f"\n  SPEED HEAD at lag 1")
    for kind in KINDS:
        print(f"    rho_{kind:<7} {prim[kind]['rho']:+.4f}  "
              f"bootstrap sd {prim[kind]['bootstrap_sd']:.4f}")

    def call(v, sd):
        if v is None:
            return "none"
        a = abs(v)
        if min(abs(a - MATERIAL), abs(a - NEGLIGIBLE)) < sd:
            return "boundary"
        return "material" if a >= MATERIAL else (
            "negligible" if a <= NEGLIGIBLE else "mixed")

    # How blunt was the registered instrument? The randomised series dilutes any
    # real autocorrelation toward zero, so its null is only as strong as the
    # dilution is mild. Both series estimate the same zero, so the ratio measures
    # the dilution empirically rather than assuming it.
    print("\n  DILUTION. the randomised series destroys dependence along with")
    print("  noise. both rows below estimate the same quantity, so the ratio is")
    print("  how much of the signal the registered instrument was keeping\n")
    print(f"    {'series':>12} {'centered':>10} {'randomised':>12} {'kept':>8}")
    dil = {}
    for h in heads:
        for kind in ("level", "vol"):
            if (h, kind, "real") not in series:
                continue
            a = acf_from(terms[((h, kind, "real"), 1)], all_rows)
            b = acf_from(terms[((h, "r" + kind, "real"), 1)], all_rows)
            r = (b / a) if (a is not None and b is not None
                            and abs(a) > 1e-9) else float("nan")
            dil[f"{h}:{kind}"] = {"centered": a, "randomised": b, "kept": r}
            print(f"    {h + ' ' + kind:>12} {a:>+10.4f} {b:>+12.4f} {r:>8.2f}")
    out["dilution_lag1"] = dil

    calls = {k: call(prim[k]["rho"], prim[k]["bootstrap_sd"])
             for k in ("vol", "level")}
    rcalls = {k: call(prim["r" + k]["rho"], prim["r" + k]["bootstrap_sd"])
              for k in ("vol", "level")}
    out["calls"] = calls
    out["calls_randomised"] = rcalls
    ratio = dil.get("s:vol", {}).get("kept", float("nan"))
    out["dilution_kept_s_vol"] = ratio
    print(f"\n  centered residual, the primary   {calls}")
    print(f"  randomised, as registered        {rcalls}")
    if np.isfinite(ratio) and abs(ratio) < 0.5:
        print(f"  the randomised instrument kept {abs(ratio):.0%} of the speed "
              f"volatility signal, which is why it is no longer the primary")

    src, lbl = calls, "centered residual"
    if any(v == "material" for v in src.values()):
        verdict = (f"MATERIAL RESIDUAL DEPENDENCE on the {lbl}. {src}. Under a "
                   f"correct model these autocorrelations are exactly zero, so "
                   f"the joint is wrong in its step to step structure. Every "
                   f"marginal instrument passed because marginal calibration "
                   f"does not constrain dependence, and this is the structure "
                   f"they could not see.")
    elif all(v == "negligible" for v in src.values()):
        verdict = (f"NO RESIDUAL DEPENDENCE AT LAG 1 on the {lbl}. {src}. The "
                   f"series is i.i.d. to within a hundredth, so the step to step "
                   f"structure is not where the defect lives either.")
    elif any(v == "boundary" for v in src.values()):
        verdict = (f"BOUNDARY on the {lbl}, the call is REFUSED. {src}. Reported "
                   f"as the in between case.")
    else:
        verdict = (f"MIXED on the {lbl}. {src}. Report the lag profile and the "
                   f"numbers.")
    out["verdict"] = verdict
    out["verdict_basis"] = lbl
    print(f"\n  -> {verdict}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.out}")
    print("""
  DIAGNOSTIC ONLY, never a contract score. no serving change follows and
  no build is authorised by any outcome. phase conditioning and the
  spectral loss term remain NOT AUTHORISED.
  the primary is the CENTERED MID residual, which is exactly mean zero given
  the history and therefore exactly uncorrelated at every lag under a correct
  model, with no added noise. the randomised series the registration named is
  reported beside it, and the ratio is how much of the signal it was keeping.
  the self sampled arm is the estimator's null, not a null for a difference.""")


if __name__ == "__main__":
    main()
