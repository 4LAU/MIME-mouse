"""WHERE in the input does the conditioning fail?

PRE REGISTERED in HANDOFF, thresholds fixed before this file existed.

w4_cross found the three channels are wrong together at a single instant, and
w4_couple localised it to the top decile of surprise and ruled out the model's own
uncertainty as the explanation. This asks where in the INPUT the failure sits.

Relationship to the CLOSED additive conditioning suspect, stated so this is not a
quiet reopening. `w4_coupletok` measured Spearman rank correlations among speed,
turn magnitude and dt inside trajectories and found no attenuation. That closure
stands. Spearman is a monotone rank statistic over all events in a trajectory, so
it is dominated by the typical ninety percent and deliberately compresses extremes,
while the defect found here lives in the top decile. Bulk coupling being right and
tail coupling being wrong are not in conflict. The FiLM rewrite remains NOT
AUTHORISED and this test does not bear on it.

The identity again, sliced somewhere new. The direction head's residual satisfies

    E[ srp_th | history, s(t) ] = 0     exactly, under a correct model

and therefore E[srp_th | s(t)] = 0. The emitted speed is part of what the direction
head conditions on, so slicing by it is legal, and the curve

    g(b) = mean srp_th over positions whose emitted speed falls in bin b

is zero at every b under a correct model. Its shape says where in the input space
the conditioning fails, which a single correlation cannot say.

    explained = Cov( srp_s , g(bin of s) ) / Cov( srp_s , srp_th )

    >= 0.50   the defect IS a function of the emitted value and the curve
              characterises it completely
    <= 0.20   the coupling is finer grained than any function of the emitted value
    otherwise MIXED
    BOUNDARY, within one bootstrap sd of a threshold, the call is REFUSED

Same numbers as w4_couple on the same explained fraction scale, deliberately.

`--driver srp` runs the SECOND registered question with the identical estimator
path: slice by the model's OWN SURPRISE at the driver token rather than by the
token's value. `srp_s(t)` is a function of (history, s(t)) and nothing else, so
E[srp_th | srp_s] = 0 holds exactly and the slice is equally legal. It is legal
only for the RESPONDER's residual. Conditioning a head's own residual on a
function of itself is conditioning on the outcome and manufactures structure under
a perfect model, so `s->s` is never computed.

VALIDITY ARM. Self sampled tokens, curve zero at every bin by construction. A bin
fails only if it is both above the magnitude floor and beyond three sd of its own
size implied noise, the rule as amended after w4_couple's false failure.

DIAGNOSTIC ONLY, never a contract score. No serving change follows and no build is
authorised by any outcome. Phase conditioning, the spectral loss term and the FiLM
rewrite all remain NOT AUTHORISED.

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

EXPLAINS = 0.50
FINER = 0.20
VALID_FLOOR = 0.002
N_BINS = 10
MIN_BIN = 500
HEADS = ("s", "th", "dt")

# (driver, responder). The driver must be something the responder's head is
# ALREADY conditioned on, otherwise the identity does not hold and the slice is
# not legal. th is conditioned on s, dt is conditioned on s and th.
SLICES = (("s", "th"), ("s", "dt"), ("th", "dt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--driver", choices=("value", "srp"), default="value",
                    help="slice by the emitted VALUE, or by the model's own "
                         "SURPRISE at that value. both are legal slices.")
    ap.add_argument("--out", default="research/w4_condshape.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    rng = np.random.default_rng(args.seed + 41)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed + 307)

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
    print(f"  {B:,} rows at least 12 events, the same rows w4_couple used\n",
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

    print("  one teacher forced forward pass, identical to w4_couple's", flush=True)
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
    # The DRIVER is the emitted value the responder's head already conditions on.
    # Turn magnitude rather than signed turn, because the direction alphabet is
    # circular and a signed ordering of it is arbitrary while its distance from
    # straight ahead is not.
    # THE DRIVER. Either the emitted VALUE, which is in the responder head's
    # input by construction, or the model's own SURPRISE at that value, which is
    # a function of the same information but is nowhere computed in the forward
    # pass. Both are (history, driver token) measurable, so the identity
    # E[srp_resp | driver] = 0 holds exactly for either, and the slice is legal.
    half = (N_REAL["th"] - 1) / 2.0
    value = {"s": real_s.astype(np.float64),
             "th": np.abs(real_th.astype(np.float64) - half)}

    def driver(h, arm):
        # In the self arm the driver must be the SELF arm's own quantity, so the
        # null is internally consistent. The two heads are sampled independently
        # there, so E[srp_resp_self | srp_driver_self] = 0 by independence rather
        # than by the tower property. Still exact, but by a different argument.
        return value[h] if args.driver == "value" else srp[(h, arm)].astype(
            np.float64)

    out = {"ckpt": args.ckpt, "n_rows": int(B), "seed": args.seed,
           "driver": args.driver,
           "diagnostic_only": True, "pre_registered": "HANDOFF.md 2026-08-06",
           "thresholds": {"explains": EXPLAINS, "finer": FINER},
           "n_live": int(live.sum()), "slices": {}}

    results = {}
    for (dr, resp) in SLICES:
        lab = f"{dr}->{resp}"
        mask = lv[dr] & lv[resp]
        blk = {}
        for arm in ("real", "self"):
            a = srp[(dr, arm)].astype(np.float64)
            b = srp[(resp, arm)].astype(np.float64)
            # Edges fixed once at the full sample of THIS arm, so the bootstrap
            # never resamples the binning.
            d = driver(dr, arm)
            edges = np.quantile(d[mask], np.linspace(0, 1, N_BINS + 1)[1:-1])
            idx = np.digitize(d, edges).astype(np.int8)
            curve, gmap = [], np.zeros(N_BINS)
            for bi in range(N_BINS):
                m = mask & (idx == bi)
                nb = int(m.sum())
                if nb < MIN_BIN:
                    curve.append({"bin": bi, "n": nb, "g": None, "rel": None,
                                  "lo": None, "hi": None})
                    continue
                g = float(b[m].mean())
                gmap[bi] = g
                # The bin's own noise, from its own spread. Positions inside a
                # bin are very nearly independent, w4_indep put the residual
                # step to step dependence at 0.0033, so this understates the
                # noise only slightly and understating it makes the validity
                # gate STRICTER, which is the safe direction for a null arm.
                se = float(b[m].std()) / np.sqrt(nb)
                curve.append({"bin": bi, "n": nb, "g": g, "se": se,
                              "rel": g / float(ent[resp][m].mean()),
                              "lo": float(d[m].min()), "hi": float(d[m].max())})
            # The part of the coupling reproduced by the curve alone. Both
            # covariances on the identical mask, so the ratio is a share.
            av, bv = a[mask], b[mask]
            gv = gmap[idx[mask]]
            cov_full = float(np.cov(av, bv, bias=True)[0, 1])
            cov_curve = float(np.cov(av, gv, bias=True)[0, 1])
            share = (cov_curve / cov_full) if abs(cov_full) > 1e-12 else None
            # Bootstrap over SEQUENCES. The curve is refitted inside each draw,
            # so the ratio carries the curve's own estimation noise rather than
            # treating the fitted curve as if it were known.
            rowid = np.broadcast_to(np.arange(B)[:, None], (B, MAX_T))[mask]
            order = np.argsort(rowid, kind="stable")
            rs_row, rs_start = np.unique(rowid[order], return_index=True)
            seg = np.split(order, rs_start[1:])
            boot = []
            for _ in range(max(40, args.draws // 3)):
                pick_rows = rng.integers(0, len(rs_row), len(rs_row))
                sel = np.concatenate([seg[p] for p in pick_rows])
                aa, bb, ii = av[sel], bv[sel], idx[mask][sel]
                gm = np.zeros(N_BINS)
                for bi in range(N_BINS):
                    mm = ii == bi
                    if mm.sum() >= MIN_BIN:
                        gm[bi] = bb[mm].mean()
                cf = np.cov(aa, bb, bias=True)[0, 1]
                if abs(cf) < 1e-12:
                    continue
                boot.append(np.cov(aa, gm[ii], bias=True)[0, 1] / cf)
            sd = float(np.std(boot)) if len(boot) > 10 else float("nan")
            blk[arm] = {"curve": curve, "cov_full": cov_full,
                        "cov_curve": cov_curve, "share": share, "share_sd": sd}
        results[lab] = blk
    out["slices"] = results

    print("  VALIDITY arm. self sampled tokens. the curve is zero at every bin")
    print("  by construction, so any bin away from zero is estimator error\n")
    vfail = []
    for lab, blk in results.items():
        v = blk["self"]
        bad = [c for c in v["curve"] if c["g"] is not None
               and abs(c["g"]) >= VALID_FLOOR
               and c["se"] > 0 and abs(c["g"]) / c["se"] >= 3.0]
        worst = max((abs(c["g"]) for c in v["curve"]
                     if c["g"] is not None), default=0.0)
        wz = max((abs(c["g"]) / c["se"] for c in v["curve"]
                  if c["g"] is not None and c["se"] > 0), default=0.0)
        smallest = min((c["n"] for c in v["curve"] if c["g"] is not None),
                       default=0)
        print(f"    {lab:>8}  worst bin {worst:+.4f}  worst {wz:.1f} sd  "
              f"excursions {len(bad)}  smallest bin {smallest:,}")
        if bad:
            vfail.append(lab)
    if vfail:
        print(f"\n  VALIDITY ARM FAILED on {', '.join(vfail)}.")
        out["verdict"] = f"FAILED, validity curve away from zero on {','.join(vfail)}"
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        return
    print("\n  validity arm PASSES, the slicing does not manufacture a curve\n")

    what = ("the DRIVER value it is already conditioned on" if args.driver ==
            "value" else "the model's OWN SURPRISE at the driver token")
    print(f"  THE CURVE. mean surprise of the RESPONDER head, in nats, by decile")
    print(f"  of {what}. zero at every bin")
    print("  under a correct model. positive means the head is worse there\n")
    print(f"    {'slice':>8} " + " ".join(f"{'d' + str(i):>8}" for i in range(N_BINS)))
    for lab, blk in results.items():
        c = blk["real"]["curve"]
        print(f"    {lab:>8} " +
              " ".join(f"{x['g']:>+8.4f}" if x["g"] is not None else f"{'-':>8}"
                       for x in c))
    print("\n  the same curve as a FRACTION of what that head knows, its entropy\n")
    print(f"    {'slice':>8} " + " ".join(f"{'d' + str(i):>8}" for i in range(N_BINS)))
    for lab, blk in results.items():
        c = blk["real"]["curve"]
        print(f"    {lab:>8} " +
              " ".join(f"{x['rel']:>+8.3f}" if x["rel"] is not None else f"{'-':>8}"
                       for x in c))

    print("\n  HOW MUCH OF THE COUPLING THE CURVE ALONE REPRODUCES\n")
    print(f"    {'slice':>8} {'cov full':>10} {'cov curve':>10} {'share':>8} {'sd':>7}")
    for lab, blk in results.items():
        r = blk["real"]
        print(f"    {lab:>8} {r['cov_full']:>+10.4f} {r['cov_curve']:>+10.4f} "
              f"{r['share']:>8.2f} {r['share_sd']:>7.2f}")

    def call(v, sd):
        if v is None:
            return "none"
        if not np.isfinite(sd):
            sd = 0.0
        if min(abs(v - EXPLAINS), abs(v - FINER)) < sd:
            return "boundary"
        return "explains" if v >= EXPLAINS else (
            "finer" if v <= FINER else "mixed")

    calls = {lab: call(blk["real"]["share"], blk["real"]["share_sd"])
             for lab, blk in results.items()}
    out["calls"] = calls
    prim = calls["s->th"]
    print(f"\n  {calls}")
    obj = "EMITTED VALUE" if args.driver == "value" else "MODEL'S OWN SURPRISE"
    if prim == "explains":
        verdict = (f"THE DEFECT IS A FUNCTION OF THE {obj}. {calls}. The "
                   f"curve characterises it completely, so the failure is that "
                   f"the responder head does not adapt its distribution to that "
                   f"quantity, and the curve says at which values.")
    elif prim == "finer":
        verdict = (f"FINER GRAINED THAN THE {obj}. {calls}. No function "
                   f"of it reproduces the coupling, so the failure "
                   f"depends on more than that one quantity and the curve is not "
                   f"the description.")
    elif prim == "boundary":
        verdict = (f"BOUNDARY, the call is REFUSED. {calls}.")
    else:
        verdict = (f"MIXED. {calls}. Part of the coupling is a function of the "
                   f"emitted value and part is not. Report both.")
    out["verdict"] = verdict
    print(f"\n  -> {verdict}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.out}")
    print("""
  DIAGNOSTIC ONLY, never a contract score. no serving change follows and
  no build is authorised by any outcome. phase conditioning, the spectral
  loss term and the FiLM rewrite all remain NOT AUTHORISED.
  w4_coupletok's closure of the additive conditioning suspect STANDS. it
  measured bulk rank correlation, which compresses exactly the tail this
  finding lives in, so the two do not conflict and neither overturns the
  other.""")


if __name__ == "__main__":
    main()
