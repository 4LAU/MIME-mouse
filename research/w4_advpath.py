"""Adversarial training in feature space with a pathwise generator gradient.
Registered before it ran.

WHY THIS CELL AND NOT ANOTHER

Two facts, from opposite directions, point at the same missing arm.

From the scoring side, recorded in `w4_rollout.py`'s own critic comment on
2026-08-11: on a generated sample already transported to corpus marginals and
corpus rank correlations, no distance statistic can see the remaining gap at a
1536 row batch. Energy reads -0.46 null standard deviations, a gaussian MMD
between -0.52 and +0.45 across five bandwidths, energy on the quadratic lift
-0.48, the largest of 256 random projections 0.15. The contract's forest reads
the same sample at 3.47 and a logistic regression on the lift at 1.42. A
distance test spreads its power over every direction at once; a classifier
concentrates it where the difference is. Only a classifier can see what is left.

From the gradient side, established today by `w4_residual`, the pairwise
gradient cosine measures how far the model is from its target and not how the
objective is shaped. That retracts the ranking of objectives this workstream has
been doing, including the verdict `w4_softdec_critic` printed against the
critic. It leaves exactly one way to rank an objective: train on it and score.

Crossing the two gives a cell nobody has run. Every critic arm so far used
REINFORCE, whose per row credit is a scalar reward times a log probability.
Every pathwise arm so far used a moment objective, which is the thing with no
detection power. Pathwise gradient into a classifier objective is untried.

WHAT IS NEW HERE AND WHAT IS BORROWED

Borrowed unchanged: the sampler, `decode_batch` and its served decoder mirror,
the thermal gates, the human reference, the contract scorer. Nothing about the
forward path is new and nothing about it is hand rolled.

New: the generator's gradient comes through `soft_forward` and the straight
through join at the eighteen features, so forward values are exactly what the
served decoder produced and the Jacobian is the relaxed decode's. That
construction was built and validated over the last two days and it is the one
result from that line that survived scrutiny.

THE FIRST RUN OF THIS FILE FAILED, AND WHY IT IS BEING RUN AGAIN

Run one moved the contract AUC from 0.6037 to 0.6843 in fifty steps, which is
+0.0806 in the wrong direction at about six standard errors. It was stopped
there rather than carried to its endpoint: the falsifier was already triggered
by a margin no further step could undo, and the cause was identifiable.

The cause is that run one carried neither of the two things every other arm in
this family carries. It had no teacher forced anchor, the negative log
likelihood on real corpus batches that holds the model near its supervised
solution while the objective pulls on it, which HANDOFF records as what the GRPO
pilot lacked when it collapsed the model's variety. And it did not clip the
generator gradient to norm 1.0, which `w4_rollout` does on every step. Both were
omissions in construction and neither is a finding about the objective.

So run one says nothing about whether pathwise gradient into a classifier works.
It says an unanchored adversarial objective in feature space destroys a model in
fifty steps, which was already known. The gate below is unchanged and this run
carries `--lam 1.0` and a clip at 1.0.

THREE CHOICES THAT NEED STATING

The critic sees the twelve trained features. The generator differentiates
through nine of them. `mean_acceleration`, `mean_jerk` and `curvature_mean` are
detached, because every arm in the relaxed decode family has excluded them:
their relaxed Jacobian is not faithful enough to steer on. The critic keeps them
so its detection power is not cut, and the generator moves what it can honestly
move.

Features reaching the critic pass through `6 tanh(z/6)`. This is not a guard
against something that cannot happen: `mean_acceleration` reaches 158 standard
deviations on corpus rows and the existing critic code clips for the same
measured reason. A hard clip would give those rows no gradient at all, which is
wrong when outliers are part of what the critic detects, so the squash is smooth
and saturating rather than flat.

The human reference is `data/human_ref_features_sir.npy` and not the training
corpus. RESUME records the two as different populations separated by 0.0149 of
AUC at 3.5 standard errors, with `sir` drawn from the same pool as the contract's
validation humans. Aiming the objective at the corpus was a known defect in
every earlier arm.

RE REGISTERED BEFORE RUN THREE, 2026-08-11, three conditions changed and all
three are corrections rather than tuning:

  Scored at n 2000, the contract's own n, not the 800 runs one and two used.
  800 rows balance to about 760, and today's 0.6407 human floor scare was an
  artifact of scoring 1160 row folds. A bias at small n largely cancels in a
  paired base against endpoint difference, but the variance does not.

  lam 10 rather than 1. Measured on a clean GPU at lam 1: the generator's own
  gradient norm is 77.7 and the total after the anchor is 78.1, so the anchor
  was carrying roughly a tenth of the critic's pull and was not holding
  anything. lam 10 puts the two at about equal norm. The anchor's own norm is
  now printed every step, so this is auditable rather than assumed.

  The anchor is chunked. Run two died at step 40 in the anchor's backward with
  "CUDA driver error: device not ready", the WSL paravirt layer's way of
  reporting out of memory. Run two was also sharing the 8 GB card with run
  one's python, which survived a kill of its parent shell and ran orphaned for
  an hour; that is fixed too, and it is why run two was half the speed of run
  one and why it ran out of memory at all.

PREDICTION: the endpoint AUC minus the base AUC is below -0.02 and its two
standard error band excludes zero.

FALSIFIER: that difference is above +0.02, meaning training made it worse, OR it
is within two standard errors of zero, meaning this arm lands in the same
plateau band as the seven before it. Both bounds are stated because the last
four gates in this workstream were one sided and one of them passed an arm that
had gone catastrophically the wrong way.

WHAT THIS CANNOT SETTLE. A single seed. The draw noise on a two draw endpoint is
about 0.0094 and the plateau band is 0.0097, so an effect smaller than about
0.02 cannot be read from this run at all, which is why the gate is set where it
is. A confirmed result needs a fresh seed and the tier 2 panel before it is
anything more than a lead.

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

OUT_JSON = "research/w4_advpath.json"
OUT_CKPT = "research/w4_advpath.pt"
# excluded from the generator's gradient, kept in the critic's view. Their
# relaxed Jacobian is not faithful; see the module docstring.
DROP = ("mean_acceleration", "mean_jerk", "curvature_mean")
SQUASH = 6.0


def gradnorm(model):
    """Total gradient norm of whatever is currently accumulated.

    Read once after the generator backward and once after the anchor, so the
    balance between the two is measured rather than assumed. The second run of
    this file assumed it and was wrong by about two orders of magnitude.
    """
    t = 0.0
    for p in model.parameters():
        if p.grad is not None:
            t += float(p.grad.detach().double().pow(2).sum())
    return t ** 0.5


def squash(z):
    return SQUASH * torch.tanh(z / SQUASH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--tau", type=float, default=4.0)
    ap.add_argument("--chunk", type=int, default=24,
                    help="rows per relaxed decode pass. The loss is a sum over\n                         rows so chunking is exact")
    ap.add_argument("--lam", type=float, default=1.0,
                    help="weight on the teacher forced anchor NLL. Every arm "
                         "in this family carries one; the first run of this "
                         "file did not and diverged by +0.08 in 50 steps")
    ap.add_argument("--clip-g", type=float, default=1.0,
                    help="generator gradient norm clip, as in w4_rollout")
    ap.add_argument("--critic-hid", type=int, default=128)
    ap.add_argument("--critic-lr", type=float, default=1e-3)
    ap.add_argument("--critic-steps", type=int, default=40,
                    help="critic updates per generator step")
    ap.add_argument("--critic-buf", type=int, default=16,
                    help="steps of generated rows the critic is fitted on. 16 "
                         "at batch 96 is the 1536 rows at which a forest reads "
                         "the remaining gap at 3.47 null sd")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-n", type=int, default=800)
    ap.add_argument("--eval-draws", type=int, default=2)
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
    mu, sd = HX.mean(0), HX.std(0)
    sd[sd == 0] = 1.0
    mu_t = torch.tensor(mu, dtype=torch.float32, device=dev)
    sd_t = torch.tensor(sd, dtype=torch.float32, device=dev)
    tk = [FEATURE_NAMES.index(f) for f in TRAINED]
    # positions within tk whose gradient is trusted
    live = torch.tensor([i for i, f in enumerate(TRAINED) if f not in DROP],
                        device=dev)
    ZH = torch.tensor(((HX - mu) / sd)[:, tk], dtype=torch.float32, device=dev)
    print(f"\n  human reference sir, {len(HX)} rows, buffer width {cap}")
    print(f"  critic sees {len(tk)} trained features, generator "
          f"differentiates through {len(live)}", flush=True)

    model = EventARModel(**ck["config"]).to(dev)
    model.load_state_dict(ck["model_state_dict"])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)

    critic = nn.Sequential(nn.Linear(len(tk), a.critic_hid), nn.ReLU(),
                           nn.Linear(a.critic_hid, a.critic_hid), nn.ReLU(),
                           nn.Linear(a.critic_hid, 1)).to(dev)
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
        torch.cuda.empty_cache()   # the sampler's cache, before the backward
        X, keep, _ = decode_batch(list(s.cpu().numpy()), list(th.cpu().numpy()),
                                  list(dt.cpu().numpy()), ang)
        if len(keep) < 16:
            continue
        Xht = torch.tensor(X, dtype=torch.float32, device=dev)
        zt = squash((Xht - mu_t) / sd_t)[:, tk]

        # the critic is fitted on rows from earlier steps only, so every logit
        # the generator is graded on is out of sample and none of it is the
        # critic recognising the batch it just saw
        nbuf = sum(len(b) for b in cbuf)
        auc_c = float("nan")
        if nbuf >= 2 * len(zt):
            XG = torch.cat(list(cbuf), 0)
            for _ in range(a.critic_steps):
                gi = torch.randint(0, len(XG), (256,), device=dev)
                hi = torch.randint(0, len(ZH), (256,), device=dev)
                inp = torch.cat([XG[gi], squash(ZH[hi])], 0)
                lab = torch.cat([torch.zeros(256, device=dev),
                                 torch.ones(256, device=dev)])
                copt.zero_grad(set_to_none=True)
                bce(critic(inp).squeeze(1), lab).backward()
                copt.step()
            with torch.no_grad():
                lg = critic(zt).squeeze(1)
                lh = critic(squash(ZH[torch.randint(0, len(ZH),
                                                    (len(zt),), device=dev)])
                            ).squeeze(1)
                auc_c = float((lh.unsqueeze(1) > lg.unsqueeze(0))
                              .float().mean())

            # The generator step. Forward values are the served decoder's, the
            # Jacobian is the relaxed decode's, and only the trusted columns
            # carry gradient.
            #
            # Rows go through the relaxed decode in chunks. The loss is a sum
            # over rows, so summing the chunks' gradients is exactly the whole
            # batch's gradient and not an approximation of it. This is not a
            # style choice: one teacher forced pass over 96 rows at buffer width
            # 256 holds three class probability tensors at once and the WSL GPU
            # paravirt layer returns ENOMEM as a driver error rather than a
            # clean torch OOM, which is what killed the first run of this file.
            # The generator loss is a MEAN over kept rows, not a sum, because
            # the anchor is a mean over rows too and that is what puts the two
            # on the same footing. The second run of this file used a sum over
            # roughly ninety five rows and its teacher forced loss drifted
            # 1.460 to 1.564 over thirty steps while nominally anchored.
            # Dividing by a count fixed before the loop keeps the chunked
            # accumulation exact.
            #
            # The anchor runs FIRST so gn_a is its own gradient norm exactly,
            # rather than something inferred from the change in the total under
            # an orthogonality assumption. Measured on a clean GPU at lam 1.0:
            # generator alone 77.7, total 78.1, so the anchor was carrying
            # about a tenth of the critic's pull. lam is set from that.
            angt = torch.tensor(ang, dtype=torch.float32, device=dev)
            opt.zero_grad(set_to_none=True)
            # The anchor. A teacher forced negative log likelihood on real
            # corpus batches, which is what holds the model near its supervised
            # solution while the critic pulls on it. HANDOFF records that
            # dropping it is how the GRPO pilot collapsed the model's variety,
            # and the first run of this file dropped it and moved +0.0806 the
            # wrong way in fifty steps. Gradients accumulate, so running it
            # here is the same update a single combined backward would give.
            nll_v = 0.0
            if a.lam > 0:
                arows = np.sort(train_rows[rng.choice(len(train_rows), a.batch,
                                                      replace=False)])
                ah, akept = load_human(arows, cap, s2a, dtha, dta, lens,
                                       cond_all)
                if len(ah) >= 8:
                    M = len(ah)
                    # Chunked for the same reason the generator loop is, and
                    # exact for the same reason: anchor_nll is a mean over
                    # ROWS, so a chunk's mean scaled by its share of the rows
                    # sums to the whole batch's mean. One 96 row teacher forced
                    # backward at buffer width 256 is what the WSL paravirt
                    # layer reports as "CUDA driver error: device not ready",
                    # and it killed the second run of this file at step 40.
                    # Each chunk is padded to its own longest row rather than
                    # the batch's, which is most of the saving.
                    for j0 in range(0, M, a.chunk):
                        j1 = min(j0 + a.chunk, M)
                        grp = ah[j0:j1]
                        w = len(grp) / M
                        L = max(len(r[0]) for r in grp)
                        AS = torch.full((len(grp), L), S_PAD_CLASS,
                                        dtype=torch.long)
                        ATH = torch.full((len(grp), L), TH_NULL_CLASS,
                                         dtype=torch.long)
                        ADT = torch.zeros((len(grp), L), dtype=torch.long)
                        for i, r in enumerate(grp):
                            AS[i, :len(r[0])] = torch.from_numpy(r[0])
                            ATH[i, :len(r[1])] = torch.from_numpy(r[1])
                            ADT[i, :len(r[2])] = torch.from_numpy(r[2])
                        acond = torch.tensor(
                            np.asarray(cond_all[akept[j0:j1]],
                                       dtype=np.float32)).to(dev)
                        nll = anchor_nll(model, (AS.to(dev), ATH.to(dev),
                                                 ADT.to(dev)), acond, False)
                        (a.lam * w * nll).backward()
                        nll_v += float(nll.detach()) * w
                        del nll, AS, ATH, ADT, acond
                        torch.cuda.empty_cache()
            gn_a = gradnorm(model)

            nkeep = float(len(keep))
            loss = 0.0
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
                             - mu_t) / sd_t)[:, tk]
                zmix = zt[row].clone()
                zmix[:, live] = zs[:, live]
                part = nn.functional.softplus(
                    -critic(zmix).squeeze(1)).sum() / nkeep
                part.backward()
                loss += float(part.detach())
                del Xs, zs, zmix, part
                torch.cuda.empty_cache()
            gn_t = gradnorm(model)
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                      a.clip_g))
            opt.step()
            loss = torch.tensor(loss)
        else:
            loss, gn, nll_v = torch.tensor(float("nan")), float("nan"), 0.0
            gn_a = gn_t = float("nan")

        cbuf.append(zt.detach())
        if step % 10 == 0 or step == 1:
            print(f"  step {step:>4}  rows {len(keep):>3}  "
                  f"critic auc {auc_c:.3f}  loss {float(loss):.3f}  "
                  f"nll {nll_v:6.3f}  anch {gn_a:.3e}  tot {gn_t:.3e}  "
                  f"{gpu_temp()}C  {time.time() - t0:.0f}s", flush=True)
        hist["steps"].append({"step": step, "n": len(keep),
                              "critic_auc": auc_c, "loss": float(loss),
                              "nll": nll_v, "grad_norm": gn,
                              "gn_anchor": gn_a, "gn_total": gn_t})
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
    # both endpoints are means of eval_draws draws; the draw sd on a single
    # 800 row draw is about 0.0133, measured in w4_drawvar
    se = float(np.sqrt(2) * 0.0133 / np.sqrt(a.eval_draws))
    d = endp - base
    met = d < -0.02 and d + 2 * se < 0
    fals = d > 0.02 or abs(d) < 2 * se
    print(f"\n  base {base:.4f}  endpoint {endp:.4f}  "
          f"difference {d:+.4f}  se {se:.4f}  {d / se:.1f} se")
    print(f"  2 se band [{d - 2 * se:+.4f}, {d + 2 * se:+.4f}]")
    print(f"  PREDICTION {'MET' if met else 'NOT MET'}    "
          f"FALSIFIER {'TRIGGERED' if fals else 'not met'}")
    if fals and d > 0.02:
        print("  training on this objective made the model worse.")
    elif fals:
        print("  this arm lands in the same plateau band as the seven before "
              "it. pathwise gradient into a classifier objective is not the "
              "missing piece.")
    hist["verdict"] = {"base": base, "end": endp, "diff": d, "se": se,
                       "met": bool(met), "falsified": bool(fals)}
    hist["thermal"] = therm
    with open(out_json, "w") as f:
        json.dump(hist, f, indent=2)
    print(f"\n  peak {therm['peak']}C, cooled {therm['cooled_s']:.0f}s, "
          f"wrote {out_json}")


if __name__ == "__main__":
    main()
