"""Does the DIRECTION of the KL anchor matter more than the states it is
evaluated at? `w4_klanchor` says it does, and this arm is the test.

WHY THIS EXISTS. `w4_klanchor` replaced the teacher forced anchor with an exact
per position KL at the states the live model visits when it samples. It scored
WORSE than the anchor it replaced, on the same objective, from a base matching to
four decimals:

    no anchor at all, advpath run 1                     +0.0806
    constant lam decaying to about a tenth, advpath 3   +0.0793
    sampled state REVERSE KL held at target, klanchor   +0.0669
    teacher forced likelihood at full equal pull        +0.0436

Two properties had been conflated. The teacher forced anchor is maximum
likelihood, and maximum likelihood is a FORWARD KL: mass covering, and it charges
the model for abandoning mass the parent had. `w4_klanchor`'s divergence is the
expectation under the LIVE model, which is REVERSE KL: mode seeking and zero
forcing, minimised by concentrating onto a subset of the parent's support, and it
charges nothing at all for abandoned mass because those terms are weighted by a
live probability going to zero. An anchor in that direction is cheapest in
exactly the configuration it exists to prevent.

    right states, wrong direction   sampled, reverse    +0.0669
    wrong states, right direction   teacher forced      +0.0436

So direction is worth more than states, which is the opposite of `w4_klanchor`'s
premise. This arm takes the right value on both axes: FORWARD KL, expectation
under the frozen base, at the states the live model visits.

    sum_v base(v) * log( base(v) / live(v) )

It is unbounded as the live model puts zero mass where the base puts mass, which
is precisely the charge that collapse should incur and that the reverse form
waives. Cost is identical: both full logit tensors were already being computed,
so the change is which one carries the expectation, not an extra forward pass.

`--kl-direction reverse` reproduces `w4_klanchor` exactly, so the control is the
same script and not a different one.

ON THE TARGET. The adaptive controller holds the divergence at 0.05 nats per
decided token whichever direction is chosen, so the two runs budget the same
divergence and the comparison is fair in that sense. Forward KL will sit higher
at equal drift, so expect the controller to settle on a LARGER coefficient than
`w4_klanchor`'s 1.156e-02. That is the anchor doing more work, not a
misconfiguration.

A CAVEAT TO STATE RATHER THAN BURY. Forward KL evaluated at states the LIVE model
visits is not the textbook object, which would take the expectation over the
base's own states as well. Live states are what is available without a second
sampling pass, and they are the states where the live model's behaviour is what
the contract scores. It remains a valid divergence at each fixed state, and the
prefixes are held fixed, so it still carries no score function term.

PREDICTION, registered before running. Endpoint damage below +0.0436, beating the
teacher forced anchor, with the 2se band excluding that value.

FALSIFIER, two sided. Above +0.0436, in which case the direction argument is
wrong and BOTH anchor axes are closed together, which is the more valuable
outcome because it closes the whole anchor family rather than one member. Or
within 2se of `w4_klanchor`'s +0.0669, in which case direction does nothing and
the ordering that motivated this arm was noise.

READING THE KL TRACE. Registered because `w4_klanchor` taught the lesson that a
three point trace invites over reading:

    coef settles well above 1.156e-02, damage falls   direction is the axis
    coef settles near klanchor's, damage unchanged    the two directions are not
                                                      distinguishable at this
                                                      budget, target too loose
    coef climbs to the 1e6 clamp                      forward KL cannot be held
                                                      at 0.05 without freezing
                                                      the model, report the clamp
                                                      fraction and do not read AUC

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
from models.event_ar import prefix_state  # noqa: E402
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TICK_CLASS,
)
from w4_rollout import (  # noqa: E402
    CKPT, COOL_C, COOL_MAX_S, D, GATE_C, HELD_OUT, KILL_C, RESUME_C, TRAINED,
    decode_batch, gpu_temp, summarise,
)
from w4_softdec import soft_forward, straight_through  # noqa: E402

OUT_JSON = "research/w4_kldir.json"
OUT_CKPT = "research/w4_kldir.pt"
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


def token_kl(model, base, s, th, dt, cond, forward):
    """Exact per position KL from the live model to the frozen base, at the
    states the live model actually visits when it samples.

    Masking follows `token_logprob_pos` exactly, because it has to be the same
    set of decisions: the s and dt heads are live up to and including the first
    PAD decision, the th head only at motion events. Positions that were not
    decisions contribute zero.

    Both models are evaluated on the SAME prefixes, which are the ones the live
    model produced. The prefixes are held fixed, so this is a divergence between
    two conditional distributions at a given state and carries no score function
    term. Returned per decided token so the number means the same thing at every
    sequence length and the adaptive controller has a stable target.
    """
    s_lg, th_lg, dt_lg = model(*model.shift_inputs(s, th, dt),
                               prefix_state(s, th, dt, cond), cond, s, th, dt)
    with torch.no_grad():
        b_s, b_th, b_dt = base(*base.shift_inputs(s, th, dt),
                               prefix_state(s, th, dt, cond), cond, s, th, dt)
    pad = s >= S_PAD_CLASS
    first_pad = torch.where(pad.any(1), pad.float().argmax(1),
                            torch.full_like(s[:, 0], s.shape[1] - 1))
    pos = torch.arange(s.shape[1], device=s.device).unsqueeze(0)
    live = pos <= first_pad.unsqueeze(1)
    motion = (s > TICK_CLASS) & (s < S_PAD_CLASS) & live

    def kl(lg, bg, mask):
        # Only the expectation changes between the two directions. Under
        # forward the weights come from the frozen base and the gradient
        # reaches the live model through -bp.exp() * lp, which is the cross
        # entropy of live under base, so the term is unbounded exactly where
        # the live model has abandoned mass the base holds.
        lp = torch.log_softmax(lg.float(), -1)
        bp = torch.log_softmax(bg.float(), -1)
        if forward:
            return ((bp.exp() * (bp - lp)).sum(-1) * mask)
        return ((lp.exp() * (lp - bp)).sum(-1) * mask)

    k = kl(s_lg, b_s, live) + kl(dt_lg, b_dt, live) + kl(th_lg, b_th, motion)
    n = (live.sum(1) * 2 + motion.sum(1)).clamp(min=1)
    return (k.sum(1) / n).mean()


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
    ap.add_argument("--kl-direction", type=str, default="forward",
                    choices=("forward", "reverse"),
                    help="forward is mass covering and charges for abandoned "
                         "mass; reverse reproduces w4_klanchor exactly")
    ap.add_argument("--kl-target", type=float, default=0.05,
                    help="nats per decided token the sampled distribution is "
                         "allowed to move from the base. The one knob, picked "
                         "and not tuned; read the KL trace before the verdict")
    ap.add_argument("--kl-coef", type=float, default=1.0,
                    help="starting KL coefficient. The controller moves it, so "
                         "this only sets where the search begins")
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
    # The frozen reference. Never stepped, never in an optimiser, eval mode for
    # the whole run. This is what "the base" means in the KL.
    base = EventARModel(**ck["config"]).to(dev)
    base.load_state_dict(ck["model_state_dict"])
    base.eval()
    for q in base.parameters():
        q.requires_grad_(False)

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
    kl_coef = a.kl_coef
    fwd = a.kl_direction == "forward"
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
            # Normalised to unit norm and held aside. THIS is the inversion of
            # w4_advmoment's rule. There the anchor was rescaled to a fixed
            # share of the critic; here the critic is put on a fixed footing so
            # that the anchor's own magnitude, which is what measures drift, is
            # left free to grow. A KL term that has been renormalised every step
            # cannot pull harder when the model wanders further, which is the
            # only thing it is for.
            gsnap = [None if p.grad is None else p.grad.detach().div(gn_g)
                     for p in model.parameters()]

            # The anchor, second and alone, on the tokens the model just
            # produced itself. No human batch is loaded and none is wanted:
            # w4_advmoment held a teacher forced loss flat at 1.505 to 1.576
            # for fifty steps while the sampled distribution collapsed.
            opt.zero_grad(set_to_none=True)
            kl_v = 0.0
            M = len(s)
            for j0 in range(0, M, a.chunk):
                j1 = min(j0 + a.chunk, M)
                w = (j1 - j0) / M
                kl = token_kl(model, base, s[j0:j1], th[j0:j1], dt[j0:j1],
                              cond[j0:j1], fwd)
                (w * kl).backward()
                kl_v += float(kl.detach()) * w
                del kl
                torch.cuda.empty_cache()
            gn_a1 = gradnorm(model)

            # The multiplicative controller. The coefficient is what gets
            # searched over; the target is the interpretable quantity and the
            # only knob a reader has to hold in mind.
            if kl_v > 1.5 * a.kl_target:
                kl_coef *= 1.5
            elif kl_v < a.kl_target / 1.5:
                kl_coef /= 1.5
            kl_coef = min(max(kl_coef, 1e-3), 1e6)

            for p, g in zip(model.parameters(), gsnap):
                if p.grad is None:
                    p.grad = g
                else:
                    p.grad.mul_(kl_coef)
                    if g is not None:
                        p.grad.add_(g)
            del gsnap
            gn_t = gradnorm(model)
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                      a.clip_g))
            opt.step()
        else:
            gn = float("nan")
            gn_g = gn_a1 = gn_t = float("nan")
            kl_v = 0.0

        cbuf.append(zt.detach())
        if step % 10 == 0 or step == 1:
            print(f"  step {step:>4}  rows {len(keep):>3}  "
                  f"critic auc {auc_c:.3f}  w1 {w1_v:.4f}  "
                  f"kl {kl_v:.4f}  coef {kl_coef:.3e}  gen {gn_g:.3e}  "
                  f"kln {gn_a1:.3e}  tot {gn_t:.3e}  {gpu_temp()}C  "
                  f"{time.time() - t0:.0f}s", flush=True)
        hist["steps"].append({"step": step, "n": len(keep),
                              "critic_auc": auc_c, "w1": w1_v, "kl": kl_v,
                              "kl_coef": kl_coef, "grad_norm": gn,
                              "gn_gen": gn_g, "gn_kl": gn_a1,
                              "gn_total": gn_t})
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
