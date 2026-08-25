"""Do the pathwise and score function gradients of one surrogate agree? One flag.

REGISTRATION REWRITTEN 2026-08-12, BEFORE RUNNING. The original version of this
docstring argued that `w4_advmoment` walked toward the worst value of its own
objective, that an optimiser handed an unbiased gradient cannot do that, and that
the pathwise straight through estimator's bias was therefore the explanation.

`w4_bulktail` withdrew the premise. The arm did NOT mostly walk toward a point
mass. Through the 1st to 99th percentile range it moved the whole acceleration
family TOWARD human, from 30 to 60 percent too wide onto target:

    feature              base    step 100
    max_velocity        1.343       1.018
    max_acceleration    1.579       1.038
    mean_jerk           1.457       1.071
    std_acceleration    1.388       0.919

and the contract AUC rose 0.0522 anyway. Mean change over the trained twelve was
p1_99 -0.152 against iqr -0.038, so the movement was four to one tails over bulk,
and what bulk narrowing there is sits mostly on two held out features. There is no
contradiction left for estimator bias to explain, and the collapse story must not
be carried into how this arm is read.

WHAT STILL JUSTIFIES RUNNING IT. Two things, and they are the whole registration.

ONE, the cosine between the two gradients is diagnostic on its own. If RELAX or
REBAR is ever built here it uses the relaxed decode as a learned control variate
for an unbiased score function gradient, and a control variate reduces variance
only in so far as it CORRELATES with the thing it corrects. That correlation is
exactly this cosine. Near zero and RELAX is not worth building whatever else is
true. Near one and the two estimators are the same direction at different noise
levels, which is also decisive and in the other direction.

TWO, an independent worry already in RESUME, untouched by any of the above: SIX
score function objectives landed within 0.0048 of each other. That is more easily
explained by one shared noisy estimator than by six independently good ideas
agreeing. Measuring what the score function gradient on this surrogate actually
looks like, beside a low variance alternative on the SAME surrogate, speaks to
that directly.

WHY THE SURROGATE IS AFFORDABLE FOR BOTH. The sorted matching objective looks like
a batch statistic, and a score function gradient of a batch statistic is one
scalar per batch and hopelessly noisy. But the arm holds the matching coefficients
fixed, read off the hard forward values, and with them fixed the surrogate is a
plain sum over rows, r_i = sum_j coeff[i,j] * critic(z_i)[j]. So the score
function form is per row, sum_i (r_i - b) * grad log p(x_i), and gets the whole
batch as samples rather than one. Both estimators target the SAME surrogate, which
is what makes the cosine meaningful.

Caveat to state rather than bury: the coefficients depend on the sample through
its ranks, so holding them fixed means both estimators target the same SURROGATE
and not exactly the same true gradient. That is the right comparison anyway,
because the surrogate is what the arm actually descends.

MUST NOT BE RUN ON `w4_advpath`'s PER ROW OBJECTIVE, even though its score
function variance would be lower. For that objective collapse genuinely IS the
optimum, so both estimators would reduce spread, both would be right to, and
nothing would be discriminated.

UNANCHORED BY DEFAULT, because an anchor is exactly the term that would hide each
estimator's own direction. The update is rescaled to EXACTLY unit norm rather than
clipped to at most unit norm, and that is load bearing: clipping only ever scales
down, so a small score function gradient would take shorter steps and "sf did not
collapse" and "sf barely moved" would be the same picture. Fixing the norm makes
step SIZE identical by construction so that DIRECTION is the only thing left.

EVAL IS n 800 ON 1 DRAW DELIBERATELY. Those AUCs are comparable to EACH OTHER and
to nothing else in the record, which stands at n 2000, where n 800 reads about
0.03 low. Do not quote them against any other arm.

READING IT. The paired dispersion change is the headline and the cosine is the
supplement, never the substitute, because this file already contains a case where
a cosine near one hid a two order of magnitude scale error.
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
    token_logprob_pos,
)
from w4_softdec import soft_forward, straight_through  # noqa: E402

OUT_JSON = "research/w4_estimator.json"
OUT_CKPT = "research/w4_estimator.pt"
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
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--sample-chunk", type=int, default=96,
                    help="rows per sampler forward pass. Decoupled from "
                         "--batch so the batch can be raised past what the "
                         "card fits in one pass, which is what the reliability "
                         "measurement says is needed")
    ap.add_argument("--tau", type=float, default=4.0)
    ap.add_argument("--chunk", type=int, default=24,
                    help="rows per relaxed decode pass. Exact here because the "
                         "surrogate is a sum over rows even though the "
                         "objective it stands for is not")
    ap.add_argument("--estimator", choices=("st", "sf"), default="st",
                    help="st is the pathwise straight through gradient the "
                         "collapsing arms used. sf is a score function estimate "
                         "of the SAME surrogate with no relaxation anywhere. "
                         "This is the only difference between the two runs")
    ap.add_argument("--beta", type=float, default=0.0,
                    help="anchor pull as a multiple of the generator's gradient "
                         "norm. ZERO here on purpose: the question is what each "
                         "estimator's own direction does, and an anchor is "
                         "exactly the term that would hide it")
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
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-n", type=int, default=800,
                    help="800 and not the contract's 2000, deliberately. This "
                         "file compares two arms to each other on SPREAD; it "
                         "does not produce a number for the record. 800 reads "
                         "about 0.03 low and its AUCs must never be quoted "
                         "against an n 2000 verdict")
    ap.add_argument("--eval-draws", type=int, default=1)
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
            # --sample-chunk, not --batch. The evaluation used to borrow the
            # training batch as its memory chunk, which was harmless while the
            # two were the same number and is not once --batch is raised past
            # what one sampler pass fits.
            for c0 in range(0, len(eval_cond), a.sample_chunk):
                if thermal() >= KILL_C:
                    raise SystemExit(
                        f"  GPU at or above the {KILL_C}C kill during the "
                        f"{tag} evaluation. Stopping.")
                c = eval_cond[c0:c0 + a.sample_chunk].to(dev)
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
        # Chunked so --batch is limited by wall clock rather than by what one
        # sampler forward pass fits. The gradient side below already chunks at
        # --chunk and accumulates, so this was the only ceiling on the batch,
        # and w4_estimator_rel measured the batch to be the binding constraint
        # on whether either estimator carries signal at all.
        with torch.no_grad():
            parts = [model.sample(cond[c0:c0 + a.sample_chunk], seq_len=cap)
                     for c0 in range(0, len(cond), a.sample_chunk)]
        s = torch.cat([p[0] for p in parts])
        th = torch.cat([p[1] for p in parts])
        dt = torch.cat([p[2] for p in parts])
        del parts
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

            # The generator, first and alone, so its norm is its own. The two
            # branches descend the SAME first order surrogate, sum over rows
            # and columns of coeff times the critic output, and differ only in
            # how that surrogate's derivative with respect to the parameters is
            # estimated. Everything above this point is shared.
            def gen_grad(kind):
                """Accumulate one estimator's gradient into p.grad and return
                its norm. Caller is responsible for zeroing first."""
                if kind == "sf":
                    # Score function. With the coefficients fixed the surrogate
                    # is a plain sum over rows, so row i carries its own reward
                    # and the estimator gets the whole batch as samples instead
                    # of one. The relaxed decode is not called on this path.
                    adv = (coeff * lg).sum(1)
                    adv = adv - adv.mean()
                for i0 in range(0, len(ang), a.chunk):
                    i1 = min(i0 + a.chunk, len(ang))
                    sel = [j for j, k in enumerate(keep) if i0 <= k < i1]
                    if not sel:
                        continue
                    loc = torch.tensor([keep[j] - i0 for j in sel], device=dev)
                    row = torch.tensor(sel, device=dev)
                    if kind == "sf":
                        lp, _ = token_logprob_pos(model, s[i0:i1], th[i0:i1],
                                                  dt[i0:i1], cond[i0:i1], False)
                        (adv[row] * lp.sum(1)[loc]).sum().backward()
                        del lp
                    else:
                        Xs, _ = soft_forward(model, s[i0:i1], th[i0:i1],
                                             dt[i0:i1], cond[i0:i1],
                                             angt[i0:i1], tau=a.tau)
                        zs = squash((straight_through(Xht[row], Xs[loc])
                                     - ctr_t) / scl_t)[:, tk]
                        zmix = zt[row].clone()
                        zmix[:, live] = zs[:, live]
                        (coeff[row] * critic(zmix)).sum().backward()
                        del Xs, zs, zmix
                    torch.cuda.empty_cache()
                return gradnorm(model)

            opt.zero_grad(set_to_none=True)
            gn_g = gen_grad(a.estimator)

            # The cosine between the two estimators, on print steps only so the
            # cheaper one's cost lands about a tenth of the time. It is NOT the
            # verdict, and the record contains a case where a cosine near one
            # sat beside a two order of magnitude scale error, so the dispersion
            # measurement is what decides. It is here because it is the number
            # that predicts whether RELAX is worth building: a control variate
            # only reduces variance if it correlates with what it corrects, so a
            # cosine near zero would say the relaxed decode is useless as one
            # even if the rest of this file blames it for the collapse.
            cos_est = float("nan")
            if step % 10 == 0 or step == 1:
                snap = [None if p.grad is None else p.grad.detach().clone()
                        for p in model.parameters()]
                opt.zero_grad(set_to_none=True)
                gn_o = gen_grad("sf" if a.estimator == "st" else "st")
                dot = 0.0
                for p, g in zip(model.parameters(), snap):
                    if p.grad is not None and g is not None:
                        dot += float((p.grad.detach().double()
                                      * g.double()).sum())
                if gn_g > 0 and gn_o > 0:
                    cos_est = dot / (gn_g * gn_o)
                opt.zero_grad(set_to_none=True)
                for p, g in zip(model.parameters(), snap):
                    p.grad = g
                del snap
            gsnap = [None if p.grad is None else p.grad.detach().clone()
                     for p in model.parameters()]

            # The anchor, second and alone, at unit weight, then rescaled to
            # beta times the generator's norm. w4_advpath used a constant lam
            # tuned at step 10 and it was a tenth of the critic by step 40,
            # because the generator's gradient grows with how confidently the
            # critic separates and the critic keeps sharpening.
            if a.beta == 0.0:
                # Unanchored, which is the default here. The generator gradient
                # IS the update, so what the step does is exactly what the
                # estimator asked for.
                #
                # Set to EXACTLY clip_g rather than clipped to at most clip_g,
                # and that difference decides whether this file can answer its
                # question at all. Clipping only ever scales down. The two
                # estimators have no reason to produce gradients of comparable
                # magnitude, and if the score function arm's norm came out below
                # the threshold it would take shorter steps, move less, and
                # report a smaller spread change for a reason that has nothing
                # to do with its direction. "sf did not collapse" and "sf barely
                # moved" would then be the same picture. Fixing the norm makes
                # step SIZE identical by construction so that DIRECTION is the
                # only thing left that differs.
                nll_v = 0.0
                lam = 0.0
                gn_a1 = float("nan")
                del gsnap
                gn_t = gn_g
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(a.clip_g / gn_g)
                gn = a.clip_g
                opt.step()
            else:
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
            gn = lam = cos_est = float("nan")
            gn_g = gn_a1 = gn_t = float("nan")
            nll_v = 0.0

        cbuf.append(zt.detach())
        if step % 10 == 0 or step == 1:
            print(f"  step {step:>4}  rows {len(keep):>3}  "
                  f"critic auc {auc_c:.3f}  w1 {w1_v:.4f}  "
                  f"nll {nll_v:6.3f}  lam {lam:.3e}  gen {gn_g:.3e}  "
                  f"cos {cos_est:+.3f}  "
                  f"tot {gn_t:.3e}  {gpu_temp()}C  "
                  f"{time.time() - t0:.0f}s", flush=True)
        hist["steps"].append({"step": step, "n": len(keep),
                              "critic_auc": auc_c, "w1": w1_v, "nll": nll_v,
                              "lam": lam, "grad_norm": gn, "gn_gen": gn_g,
                              "gn_anchor_unit": gn_a1, "gn_total": gn_t,
                              "cos_estimators": cos_est})
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
        # One estimator, one cell. The sentence this replaced claimed both
        # cells of the dichotomy were closed and printed it unconditionally
        # from a single arm, which is inherited boilerplate from the parent
        # file and was false the first time it ran here.
        print(f"  training on this objective through the {a.estimator} "
              f"estimator made the model worse. That closes ONE cell of the "
              f"dichotomy. The other estimator has to be run before anything "
              f"is concluded about the objective itself.")
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
