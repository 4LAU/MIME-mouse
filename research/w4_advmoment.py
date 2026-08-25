"""Match the DISTRIBUTION of a learned critic's output, rather than maximise it.

WHY THIS ARM EXISTS. `w4_advpath` closed the per row critic family by training on
it: base 0.6338, step 50 0.7131, step 100 0.7794, about fifteen standard errors
the wrong way, with the location error flat and the spread error climbing and the
collapse flag on from step 50. That is the objective's minimum and not a failure
to reach it. A sum over rows of `softplus(-critic(row))` is minimised by emitting
the single most human looking trajectory every time.

That result plus the seven moment arms before it gives a dichotomy, and this file
is aimed at the one cell the dichotomy leaves open:

  A batch statistic is distribution aware, so it punishes collapse, but moments
  are all it reaches and `w4_copula` closed those off. Marginals are worth
  nothing, rank correlation is worth 0.0062, a full copula transport still reads
  about 0.595 against a floor near 0.517.

  A per row critic reaches the higher order structure that actually separates,
  but it grades each trajectory alone, so it is not a comparison of
  distributions, and its optimum is a point mass.

The objective here is both at once. The critic is a learned nonlinear map onto
directions chosen to separate maximally, so it is higher order. What the
generator minimises is a two sample distance between the generated batch's
distribution and a human batch's distribution ALONG those directions, so it is
distribution aware and a point mass is maximally penalised rather than optimal.

WHAT THIS CHANGES RELATIVE TO THE ARM REGISTERED IN RESUME JOB 3e, AND WHY. Job
3e registered matching the mean and standard deviation of ONE critic scalar, and
registered its own objection: a collapsed generator could match one scalar's two
moments while being wrong in every other direction. This file generalises along
exactly the axis that objection names, at no extra machinery and no extra cost.
The critic emits K outputs instead of one, each an independently trained
discriminator sharing a trunk, and the generator matches the FULL one dimensional
distribution of each rather than its first two moments. The one scalar two moment
version is the K equals 1 corner of this, so a failure here is a failure of the
whole cell rather than of the weakest member of it. The deviation is deliberate
and is registered here rather than glossed.

THE TWO SAMPLE DISTANCE is the one dimensional Wasserstein 1 distance, which for
two equal sized samples is the mean absolute difference of the sorted values. It
is exact rather than estimated, it has no bandwidth or kernel to choose, and its
gradient is a per row sign, which is what makes the chunking below exact.

THE ONE PIECE OF NEW MATHEMATICS, and it is the part that had to be checked
rather than asserted. `w4_advpath` could chunk its relaxed decode because a sum
over rows has an exactly additive gradient. A batch statistic does not decompose
that way, and RESUME job 3b says so explicitly. It can still be chunked exactly,
by a route the per row arms never needed:

  Let l be the critic outputs and F(l) the batch statistic. The chain rule gives
  dF/dtheta as a SUM OVER ROWS of (dF/dl_i) times (dl_i/dtheta). The first factor
  depends only on the forward values. The straight through join means the forward
  values are the served decoder's, which are already computed before any chunk
  runs. So compute every dF/dl_i first, hold them CONSTANT, and backward the
  surrogate sum of (dF/dl_i) times l_i chunk by chunk. That surrogate's gradient
  is the true gradient, not an approximation of it, and it is additive over rows
  by construction.

For the sorted matching distance, dF/dl_i,k is sign(l_i,k minus the human value
it sorts against) divided by n times K. `research/w4_advmoment_check.py` verifies
the chunked surrogate against autograd through the whole undivided statistic and
must be run before this file is; it is cheap and needs no GPU model.

A DEFECT THIS ARM DOES NOT INHERIT, found while it was first running and fixed
before the run that counts. Every arm in this family standardised the critic's
input by the human reference's MEAN AND STANDARD DEVIATION. On eight of the
twelve trained features that standard deviation is worthless: the reference's
median and ninetieth percentile agree with the scorer's reference to within five
percent, but its standard deviation is seven to eight times larger, because a
handful of extreme rows out of four thousand control it. The reference is not the
wrong distribution, which was the first thing checked; it is the same
distribution with a longer sampled tail, and the standard deviation is simply the
wrong summary of it. The consequence is that the body of every speed family
feature standardises into a band of about a hundredth, and those are the features
the record says the remaining gap lives in.

The cost was measured rather than assumed, on a saved generated sample against
the same reference, same architecture, same budget:

    forest on the raw columns                 0.6515
    critic on squashed mean/sd z, as shipped  0.6315
    critic on squashed median/IQR z           0.6563
    critic on rank transformed columns        0.6838

So the shipped standardisation cost the critic about 0.025 against an affine fix
and 0.052 against the best transform. The rank transform is the strongest and is
unusable here, because its gradient is zero almost everywhere and the generator
has to differentiate through this map. The median and interquartile range is
affine, so it keeps the pathwise gradient, and it is what this file uses. The
squash then does what it was always meant to do, resolve the body and clip the
tail, instead of never engaging on the body at all.

This does NOT overturn `w4_advpath`. That arm failed by +0.1456 with the collapse
flag on, which a 0.025 handicap on the critic does not produce. It is a defect in
its own right and it is fixed here. The diagnostics `summarise` reports still use
the mean and standard deviation, unchanged, so every spread and location number
stays comparable to every earlier run.

THE BALANCE RULE, carried from `w4_advpath`'s failure. A constant lam cannot hold
an anchor against a term whose size is set by an adversary that is still
learning. Measured there: anchor 20.9 against total 23.7 at step 10, anchor 17.5
against total 384.6 at step 100, so a lam set from an early measurement was a
tenth of the critic forty steps later. Here the generator and the anchor are
backwarded separately, both gradient norms are read, and the anchor is rescaled
every step so its norm is a fixed multiple beta of the generator's. The anchor's
share is then a property of the run rather than of the step it was tuned at.

SCORING. Evaluation is at the contract's own n of 2000. The same checkpoint reads
0.6037 at n 800 and 0.6338 at n 2000, so a small eval sample does not merely add
variance, it reads about 0.03 low.

PREDICTION: the endpoint AUC minus the base AUC is below -0.02 and its two
standard error band excludes zero.

FALSIFIER: that difference is above +0.02, meaning training made it worse, OR it
is within two standard errors of zero, meaning this arm lands in the same plateau
band as the arms before it. Both bounds are stated. `w4_advpath` triggered the
upper one and would have passed a one sided gate.

WHAT THIS CANNOT SETTLE. One seed. The two draw endpoint noise here is about
0.0043, estimated from the three n 2000 evaluations `w4_advpath` wrote, whose
draw ranges were 0.0013, 0.0004 and 0.0048. The plateau band across arms is
0.0097, so an effect below about 0.02 cannot be read from a single run, which is
where the gate is set. A confirmed result needs a fresh seed and the tier 2 panel.

A RISK TO REGISTER RATHER THAN ASSUME AWAY. The 1536 row fact says energy and
kernel two sample statistics read near null on this gap where a forest reads it
at 3.47 null standard deviations, because only a classifier concentrates power
where the difference is. The distance here is computed on 96 rows, far fewer.
What is supposed to rescue it is that it is computed in the critic's learned
space, which is where the difference is concentrated, so it is the classifier's
own power being reused rather than an omnibus statistic being asked for it. That
is a claim about the learned representation and it is not established.

The protected checkpoint is never written. The eval sample is never read.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("EVENT_SNAP", "2.5")
for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import scoring  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402
from models.event_ar import EventARModel  # noqa: E402
from models.event_stream_polar import S_PAD_CLASS, TH_NULL_CLASS  # noqa: E402
from w4_rollout import (  # noqa: E402
    CKPT, COOL_C, COOL_MAX_S, D, GATE_C, HELD_OUT, KILL_C, RESUME_C, TRAINED,
    anchor_nll, decode_batch, gpu_temp, load_human, summarise,
)
from w4_softdec import soft_forward, straight_through  # noqa: E402

OUT_JSON = "research/w4_advmoment.json"
OUT_CKPT = "research/w4_advmoment.pt"
# excluded from the generator's gradient, kept in the critic's view. Their
# relaxed Jacobian is not faithful; see w4_softdec.
DROP = ("mean_acceleration", "mean_jerk", "curvature_mean")
SQUASH = 6.0


def gradnorm(model):
    """Total gradient norm of whatever is currently accumulated."""
    t = 0.0
    for p in model.parameters():
        if p.grad is not None:
            t += float(p.grad.detach().double().pow(2).sum())
    return t ** 0.5


def squash(z):
    return SQUASH * torch.tanh(z / SQUASH)


def w1_terms(lg, lh):
    """Sorted matching distance between two equal sized samples, and its
    derivative with respect to every generated value.

    lg and lh are (n, K) critic outputs for the generated and the human batch.
    Neither needs a graph: this reads forward values only, which is the whole
    point of the construction. Returns the per column distance (K,) and the
    coefficient matrix (n, K) holding d(mean over K of W1)/d lg.

    The objective is the mean over columns, so the coefficients carry the 1/K.
    Verified against autograd in research/w4_advmoment_check.py.
    """
    n = lg.shape[0]
    k = lg.shape[1]
    order = torch.argsort(lg, dim=0)
    hs, _ = torch.sort(lh, dim=0)
    diff = torch.gather(lg, 0, order) - hs
    w1 = diff.abs().mean(0)
    coeff = torch.zeros_like(lg)
    coeff.scatter_(0, order, torch.sign(diff) / (n * k))
    return w1, coeff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--tau", type=float, default=4.0)
    ap.add_argument("--chunk", type=int, default=24,
                    help="rows per relaxed decode pass. Exact here because the "
                         "surrogate is a sum over rows even though the "
                         "objective it stands for is not")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="anchor gradient norm as a multiple of the "
                         "generator's, rescaled every step. Replaces the "
                         "constant lam that failed in w4_advpath")
    ap.add_argument("--clip-g", type=float, default=1.0)
    ap.add_argument("--critic-hid", type=int, default=128)
    ap.add_argument("--critic-out", type=int, default=8,
                    help="independently trained discriminator heads on a "
                         "shared trunk. Each is a separating direction whose "
                         "output distribution the generator has to match")
    ap.add_argument("--critic-lr", type=float, default=1e-3)
    ap.add_argument("--critic-steps", type=int, default=40)
    ap.add_argument("--critic-buf", type=int, default=16,
                    help="steps of generated rows the critic is fitted on. 16 "
                         "at batch 96 is the 1536 rows at which a forest reads "
                         "the remaining gap at 3.47 null sd")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-n", type=int, default=2000,
                    help="the contract's own n. 800 reads about 0.03 low")
    ap.add_argument("--eval-draws", type=int, default=2)
    ap.add_argument("--draw-sd", type=float, default=0.0043,
                    help="sd of a single n 2000 draw, from the three "
                         "evaluations w4_advpath wrote at that n")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--init", type=str, default=None)
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()

    t = gpu_temp()
    if t > GATE_C:
        print(f"  GPU at {t}C, above the {GATE_C}C launch gate. Not starting.")
        return
    out_json = OUT_JSON.replace(".json", f"_{a.tag}.json") if a.tag else OUT_JSON
    out_ckpt = OUT_CKPT.replace(".pt", f"_{a.tag}.pt") if a.tag else OUT_CKPT

    dev = torch.device("cuda")
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    ck = torch.load(a.init or CKPT, map_location=dev, weights_only=False)
    cap = int(ck["config"]["max_seq_len"])
    a.cap = cap

    s2a = np.load(f"{D}/events_s2.npy", mmap_mode="r")
    dtha = np.load(f"{D}/events_dth.npy", mmap_mode="r")
    dta = np.load(f"{D}/events_dt.npy", mmap_mode="r")
    lens = np.load(f"{D}/events_len.npy")
    cond_all = np.load(f"{D}/events_cond.npy", mmap_mode="r")
    ok = np.flatnonzero(np.load(f"{D}/events_feat18_ok.npy"))
    perm = ok[rng.permutation(len(ok))]

    HX = np.load("data/human_ref_features_sir.npy").astype(np.float64)
    HX = HX[np.isfinite(HX).all(1)]
    rng.shuffle(HX)
    # mu and sd are for the reported diagnostics only, so spread and location
    # stay comparable to every earlier run in the record.
    mu, sd = HX.mean(0), HX.std(0)
    sd[sd == 0] = 1.0
    # ctr and scl are what the critic and the generator actually see. On the
    # speed family features this reference's sd is set by about eight rows out
    # of four thousand; the interquartile range is not. See the docstring for
    # the measured cost of getting this wrong.
    ctr = np.median(HX, 0)
    scl = np.percentile(HX, 75, 0) - np.percentile(HX, 25, 0)
    ctr_t = torch.tensor(ctr, dtype=torch.float32, device=dev)
    scl_t = torch.tensor(scl, dtype=torch.float32, device=dev)
    tk = [FEATURE_NAMES.index(f) for f in TRAINED]
    live = torch.tensor([i for i, f in enumerate(TRAINED) if f not in DROP],
                        device=dev)
    ZH = torch.tensor(((HX - ctr) / scl)[:, tk], dtype=torch.float32, device=dev)
    print(f"\n  human reference sir, {len(HX)} rows, buffer width {cap}")
    print(f"  critic scaled by median and IQR, smallest IQR {scl.min():.4g}")
    print(f"  critic sees {len(tk)} trained features on {a.critic_out} heads, "
          f"generator differentiates through {len(live)}", flush=True)

    model = EventARModel(**ck["config"]).to(dev)
    model.load_state_dict(ck["model_state_dict"])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)

    # K independent discriminators on a shared trunk. Each column of the output
    # is trained by its own binary cross entropy, so each is a separating
    # direction in its own right rather than a coordinate of some embedding
    # that only means something after a further readout.
    critic = nn.Sequential(nn.Linear(len(tk), a.critic_hid), nn.ReLU(),
                           nn.Linear(a.critic_hid, a.critic_hid), nn.ReLU(),
                           nn.Linear(a.critic_hid, a.critic_out)).to(dev)
    copt = torch.optim.Adam(critic.parameters(), lr=a.critic_lr,
                            weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    eval_rows = perm[:a.eval_n]
    eval_cond = torch.tensor(np.asarray(cond_all[np.sort(eval_rows)],
                                        dtype=np.float32))
    eval_ang = np.arctan2(eval_cond[:, 3].numpy().astype(np.float64),
                          eval_cond[:, 2].numpy().astype(np.float64))
    train_rows = perm[a.eval_n:a.eval_n + 400000]

    therm = {"peak": 0, "cooled_s": 0.0}

    def thermal():
        t = gpu_temp()
        therm["peak"] = max(therm["peak"], t)
        if t >= COOL_C:
            c0 = time.time()
            while gpu_temp() > RESUME_C and time.time() - c0 < COOL_MAX_S:
                time.sleep(10)
            therm["cooled_s"] += time.time() - c0
            t = gpu_temp()
            therm["peak"] = max(therm["peak"], t)
        return t

    def evaluate(tag):
        model.eval()
        draws = []
        for _ in range(a.eval_draws):
            S, TH, DT = [], [], []
            for c0 in range(0, len(eval_cond), a.batch):
                if thermal() >= KILL_C:
                    raise SystemExit(
                        f"  GPU at or above the {KILL_C}C kill during the "
                        f"{tag} evaluation. Stopping.")
                c = eval_cond[c0:c0 + a.batch].to(dev)
                with torch.no_grad():
                    s, th, dt = model.sample(c, seq_len=cap)
                S.append(s.cpu().numpy()); TH.append(th.cpu().numpy())
                DT.append(dt.cpu().numpy())
            S = np.concatenate(S); TH = np.concatenate(TH)
            DT = np.concatenate(DT)
            X, _, _ = decode_batch(list(S), list(TH), list(DT), eval_ang)
            np.random.default_rng(a.seed).shuffle(X)
            r = scoring.score_features(X)
            draws.append({"auc_rf": float(r["auc_rf_oob"]),
                          "collapse": bool(r["collapse_flag"]),
                          "n": float(len(X)), **summarise(X, mu, sd)})
        aucs = [d["auc_rf"] for d in draws]
        row = {"tag": tag, "auc_rf": float(np.mean(aucs)), "auc_rf_draws": aucs,
               "auc_rf_range": float(max(aucs) - min(aucs)),
               "collapse": any(d["collapse"] for d in draws),
               "n": int(np.mean([d["n"] for d in draws])),
               **{k: float(np.mean([d[k] for d in draws]))
                  for k in ("spread_err_trained", "spread_err_held",
                            "loc_err_trained", "loc_err_held")}}
        print(f"  EVAL {tag:<10} rf {row['auc_rf']:.4f}"
              f" (range {row['auc_rf_range']:.4f} over {len(aucs)})"
              f"  spread err trained {row['spread_err_trained']:.4f}"
              f"  held {row['spread_err_held']:.4f}"
              f"  collapse {row['collapse']}", flush=True)
        model.train()
        return row

    hist = {"config": vars(a), "held_out": HELD_OUT, "evals": [], "steps": []}
    hist["evals"].append(evaluate("base"))
    with open(out_json, "w") as f:
        json.dump(hist, f, indent=2)

    cbuf = collections.deque(maxlen=max(2, a.critic_buf))
    t0 = time.time()
    model.train()

    for step in range(1, a.steps + 1):
        if thermal() >= KILL_C:
            print(f"  GPU at or above the {KILL_C}C kill after "
                  f"{COOL_MAX_S}s of cooling. Stopping.", flush=True)
            break

        rows = np.sort(train_rows[rng.choice(len(train_rows), a.batch,
                                             replace=False)])
        cond = torch.tensor(np.asarray(cond_all[rows],
                                       dtype=np.float32)).to(dev)
        ang = np.arctan2(cond[:, 3].cpu().numpy().astype(np.float64),
                         cond[:, 2].cpu().numpy().astype(np.float64))

        model.eval()
        with torch.no_grad():
            s, th, dt = model.sample(cond, seq_len=cap)
        model.train()
        torch.cuda.empty_cache()
        X, keep, _ = decode_batch(list(s.cpu().numpy()), list(th.cpu().numpy()),
                                  list(dt.cpu().numpy()), ang)
        if len(keep) < 16:
            continue
        Xht = torch.tensor(X, dtype=torch.float32, device=dev)
        zt = squash((Xht - ctr_t) / scl_t)[:, tk]

        nbuf = sum(len(b) for b in cbuf)
        auc_c = float("nan")
        w1_v = float("nan")
        if nbuf >= 2 * len(zt):
            # fitted on rows from earlier steps only, so every value the
            # generator is graded on is out of sample. Each output column gets
            # its own loss against the same labels.
            XG = torch.cat(list(cbuf), 0)
            lab = torch.cat([torch.zeros(256, device=dev),
                             torch.ones(256, device=dev)])
            for _ in range(a.critic_steps):
                gi = torch.randint(0, len(XG), (256,), device=dev)
                hi = torch.randint(0, len(ZH), (256,), device=dev)
                inp = torch.cat([XG[gi], squash(ZH[hi])], 0)
                out = critic(inp)
                copt.zero_grad(set_to_none=True)
                sum(bce(out[:, j], lab)
                    for j in range(a.critic_out)).backward()
                copt.step()

            # The coefficients of the surrogate. Read off the HARD forward
            # values, which the straight through join guarantees are exactly
            # the values the relaxed pass will produce, so holding them fixed
            # while the chunks run loses nothing.
            with torch.no_grad():
                lg = critic(zt)
                lh = critic(squash(ZH[torch.randint(0, len(ZH),
                                                    (len(zt),), device=dev)]))
                w1, coeff = w1_terms(lg, lh)
                w1_v = float(w1.mean())
                auc_c = float(torch.stack(
                    [(lh[:, j].unsqueeze(1) > lg[:, j].unsqueeze(0))
                     .float().mean() for j in range(a.critic_out)]).mean())

            angt = torch.tensor(ang, dtype=torch.float32, device=dev)

            # The generator, first and alone, so its norm is its own.
            opt.zero_grad(set_to_none=True)
            for i0 in range(0, len(ang), a.chunk):
                i1 = min(i0 + a.chunk, len(ang))
                sel = [j for j, k in enumerate(keep) if i0 <= k < i1]
                if not sel:
                    continue
                Xs, _ = soft_forward(model, s[i0:i1], th[i0:i1], dt[i0:i1],
                                     cond[i0:i1], angt[i0:i1], tau=a.tau)
                loc = torch.tensor([keep[j] - i0 for j in sel], device=dev)
                row = torch.tensor(sel, device=dev)
                zs = squash((straight_through(Xht[row], Xs[loc])
                             - ctr_t) / scl_t)[:, tk]
                zmix = zt[row].clone()
                zmix[:, live] = zs[:, live]
                (coeff[row] * critic(zmix)).sum().backward()
                del Xs, zs, zmix
                torch.cuda.empty_cache()
            gn_g = gradnorm(model)
            gsnap = [None if p.grad is None else p.grad.detach().clone()
                     for p in model.parameters()]

            # The anchor, second and alone, at unit weight, then rescaled to
            # beta times the generator's norm. w4_advpath used a constant lam
            # tuned at step 10 and it was a tenth of the critic by step 40,
            # because the generator's gradient grows with how confidently the
            # critic separates and the critic keeps sharpening.
            opt.zero_grad(set_to_none=True)
            nll_v = 0.0
            arows = np.sort(train_rows[rng.choice(len(train_rows), a.batch,
                                                  replace=False)])
            ah, akept = load_human(arows, cap, s2a, dtha, dta, lens, cond_all)
            M = len(ah)
            for j0 in range(0, M, a.chunk):
                j1 = min(j0 + a.chunk, M)
                grp = ah[j0:j1]
                w = len(grp) / M
                L = max(len(r[0]) for r in grp)
                AS = torch.full((len(grp), L), S_PAD_CLASS, dtype=torch.long)
                ATH = torch.full((len(grp), L), TH_NULL_CLASS, dtype=torch.long)
                ADT = torch.zeros((len(grp), L), dtype=torch.long)
                for i, r in enumerate(grp):
                    AS[i, :len(r[0])] = torch.from_numpy(r[0])
                    ATH[i, :len(r[1])] = torch.from_numpy(r[1])
                    ADT[i, :len(r[2])] = torch.from_numpy(r[2])
                acond = torch.tensor(np.asarray(cond_all[akept[j0:j1]],
                                                dtype=np.float32)).to(dev)
                nll = anchor_nll(model, (AS.to(dev), ATH.to(dev), ADT.to(dev)),
                                 acond, False)
                (w * nll).backward()
                nll_v += float(nll.detach()) * w
                del nll, AS, ATH, ADT, acond
                torch.cuda.empty_cache()
            gn_a1 = gradnorm(model)
            lam = a.beta * gn_g / gn_a1
            for p, g in zip(model.parameters(), gsnap):
                if p.grad is None:
                    p.grad = g
                else:
                    p.grad.mul_(lam)
                    if g is not None:
                        p.grad.add_(g)
            del gsnap
            gn_t = gradnorm(model)
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                      a.clip_g))
            opt.step()
        else:
            gn = lam = float("nan")
            gn_g = gn_a1 = gn_t = float("nan")
            nll_v = 0.0

        cbuf.append(zt.detach())
        if step % 10 == 0 or step == 1:
            print(f"  step {step:>4}  rows {len(keep):>3}  "
                  f"critic auc {auc_c:.3f}  w1 {w1_v:.4f}  "
                  f"nll {nll_v:6.3f}  lam {lam:.3e}  gen {gn_g:.3e}  "
                  f"tot {gn_t:.3e}  {gpu_temp()}C  "
                  f"{time.time() - t0:.0f}s", flush=True)
        hist["steps"].append({"step": step, "n": len(keep),
                              "critic_auc": auc_c, "w1": w1_v, "nll": nll_v,
                              "lam": lam, "grad_norm": gn, "gn_gen": gn_g,
                              "gn_anchor_unit": gn_a1, "gn_total": gn_t})
        if step % a.eval_every == 0:
            hist["evals"].append(evaluate(f"step{step}"))
            torch.save({"model_state_dict": model.state_dict(),
                        "config": ck["config"], "step": step}, out_ckpt)
            with open(out_json, "w") as f:
                json.dump(hist, f, indent=2)

    if not hist["evals"] or hist["evals"][-1]["tag"] == "base":
        hist["evals"].append(evaluate("end"))
    base = hist["evals"][0]["auc_rf"]
    endp = hist["evals"][-1]["auc_rf"]
    se = float(np.sqrt(2) * a.draw_sd / np.sqrt(a.eval_draws))
    d = endp - base
    met = d < -0.02 and d + 2 * se < 0
    fals = d > 0.02 or abs(d) < 2 * se
    print(f"\n  base {base:.4f}  endpoint {endp:.4f}  "
          f"difference {d:+.4f}  se {se:.4f}  {d / se:.1f} se")
    print(f"  2 se band [{d - 2 * se:+.4f}, {d + 2 * se:+.4f}]")
    print(f"  PREDICTION {'MET' if met else 'NOT MET'}    "
          f"FALSIFIER {'TRIGGERED' if fals else 'not met'}")
    if fals and d > 0.02:
        print("  training on this objective made the model worse. Both cells "
              "of the dichotomy are now closed by measurement.")
    elif fals:
        print("  this arm lands in the same plateau band as the arms before "
              "it. A distribution match on a learned direction is not the "
              "missing piece either.")
    hist["verdict"] = {"base": base, "end": endp, "diff": d, "se": se,
                       "met": bool(met), "falsified": bool(fals)}
    hist["thermal"] = therm
    with open(out_json, "w") as f:
        json.dump(hist, f, indent=2)
    print(f"\n  peak {therm['peak']}C, cooled {therm['cooled_s']:.0f}s, "
          f"wrote {out_json}")


if __name__ == "__main__":
    main()
