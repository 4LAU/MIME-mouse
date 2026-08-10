"""Rollout level training: match the feature distribution of what the model
actually generates.

Design registered in HANDOFF.md under "Rollout level training, registered
2026-08-10 before any training". Arms, held out features, thresholds and the
prediction are fixed there. This file implements that design and nothing else.

WHY

Every per step conditional in this model is exact, measured by teacher forcing
against real human histories, on all three heads, to three or four decimal
places. Five separate token level defects have been found, confirmed and priced
at nothing on the contract. Meanwhile repairing all eighteen FEATURE marginals
directly buys 0.075 of 0.107, and no token level edit reaches more than 0.012 of
it. The error is made by free running composition and by nothing else, so the
objective has to see free running composition.

THE ESTIMATOR

Relaxing the token sampling to get a differentiable rollout is refused on
memory: a rollout is up to 256 sequential trunk passes over a growing prefix
with no KV cache, and retaining activations for all of them does not fit in
8 GB at any useful batch size.

Instead, the trunk is causally masked, so ONE full sequence teacher forced pass
over a sequence reproduces exactly the logits that were used when that sequence
was sampled. Each step is therefore one no grad rollout plus one ordinary
forward and backward over the model's own emitted tokens. That is a score
function estimator costing one rollout plus one training pass.

Features are standardised by the human mean and standard deviation. With m and s
the batch mean and standard deviation of standardised feature k:

    L = sum_k m_k^2 + sum_k (log s_k)^2

Location and spread. Spread is where the defect is: mean shifts are at worst a
fifth of a human standard deviation while spread ratios run to 2.5. The per
trajectory weight is the exact score function coefficient for that loss, with
the batch mean as its control variate, so it is unbiased and needs no learned
baseline:

    w_i = sum_k [ 2 m_k (z_ik - m_k)
                  + (log s_k / s_k^2) ((z_ik - m_k)^2 - s_k^2) ]

The surrogate minimised is mean_i w_i.detach() * logp_i, where logp_i is the
mean log probability per DECIDED token. Per token rather than per trajectory
because a summed log probability is order 240 nats here and would swamp the
anchor; per token also removes the length bias a summed objective introduces.
Deterministic positions carry no decision and are excluded: the turn token at a
no motion event is substituted to NULL rather than sampled, and everything after
a row's first PAD is forced.

w is centred and scaled to unit standard deviation within the batch and clipped,
which is advantage normalisation. It changes the step size, not the direction.

The anchor is a teacher forced negative log likelihood on real human batches at
weight LAMBDA. It is the specific answer to how the GRPO pilot failed: that run
collapsed the model's variety. Here the variance term and the anchor both
penalise collapse rather than rewarding it. The second known failure, the
learned critic reaching 0.94 by round eight and outrunning the generator, cannot
occur because nothing in this objective is learned. It is a fixed statistic of a
fixed feature map.

THE GOODHART GUARD

This objective is stated over the same eighteen features the scorer reads, so
the arm is worthless without a held out set. Twelve features enter the loss. Six
never do, chosen to span the families rather than to be easy. If the trained
twelve come into line and the held out six do not, the model learned the loss
and not the movement, and the arm is a FAIL whatever the contract says.

SAFETY

Reads only training data. Never reads the protected eval sample. Writes its own
checkpoint under research/ and never touches training/event_ar_v2_s40000.pt or
training/candi_polar_flow_best.pt. Aborts if the GPU reaches KILL_C. No
DataLoader and no worker processes, so the memmap pickling hazard in
training/train_event_ar.py cannot arise here.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

os.environ.setdefault("EVENT_SNAP", "2.5")
for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
import scoring  # noqa: E402
from features import FEATURE_NAMES, extract_feature_matrix  # noqa: E402
from models.event_ar import (  # noqa: E402
    DT_MAX_MS, EventARModel, class_to_dt_ms, prefix_state,
)
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TH_NULL_CLASS, TICK_CLASS, dth_lattice_to_class, s2_to_class,
)

D = "training"
CKPT = f"{D}/event_ar_v2_s40000.pt"
OUT_JSON = "research/w4_rollout.json"
OUT_CKPT = "research/w4_rollout_pilot.pt"

# never enter the loss; the whole arm is read off these
HELD_OUT = ["max_acceleration", "velocity_skewness", "curvature_std",
            "num_direction_changes", "time_to_peak_velocity",
            "angular_velocity_std"]
TRAINED = [f for f in FEATURE_NAMES if f not in HELD_OUT]
KILL_C = 79          # the machine crashed on this workload on 2026-08-06
GATE_C = 75


def gpu_temp():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return -1


# ------------------------------------------------------------------ corpus

def load_human(rows, cap, s2a, dtha, dta, lens, cond_all):
    """Corpus rows as token streams, capped at the same length the rollout is,
    so truncation cannot separate the arms on its own."""
    out, kept = [], []
    for j in rows:
        L = min(int(lens[j]), cap)
        if L < 8:
            continue
        s2 = torch.from_numpy(np.asarray(s2a[j, :L]).astype(np.int64))
        dth = torch.from_numpy(np.asarray(dtha[j, :L]).astype(np.int64))
        s_c = s2_to_class(s2).numpy()
        if ((s_c > TICK_CLASS) & (s_c < S_PAD_CLASS)).sum() < 8:
            continue
        th_c = torch.where(s2 > 0, dth_lattice_to_class(dth),
                           torch.full_like(dth, TH_NULL_CLASS)).numpy()
        dt_c = np.round(np.asarray(dta[j, :L]).astype(np.float64)
                        ).clip(0, DT_MAX_MS).astype(np.int64)
        c = np.asarray(cond_all[j], dtype=np.float64)
        out.append((s_c, th_c, dt_c, float(np.arctan2(c[3], c[2]))))
        kept.append(int(j))
    # the kept ids are returned because rows are dropped here, so a caller that
    # pairs these streams with their conditioning must index by them and not by
    # a prefix of the request
    return out, np.asarray(kept, dtype=np.int64)


def to_paths(s_cls, th_cls, dt_cls, angles):
    """Token streams to continuous paths through the SERVED decoder. A hand
    rolled walk reads about 0.13 high because esp._decode snaps short steps to
    whole pixels and merges ticks, so it must not be reimplemented here."""
    paths, keep = [], []
    for i in range(len(angles)):
        s = s_cls[i]
        pad = np.flatnonzero(s >= S_PAD_CLASS)
        L = int(pad[0]) if len(pad) else len(s)
        if L < 8 or ((s[:L] > TICK_CLASS) & (s[:L] < S_PAD_CLASS)).sum() < 8:
            continue
        ms = class_to_dt_ms(torch.from_numpy(dt_cls[i][:L])).numpy().astype(np.float64)
        dz = (np.log(np.maximum(ms, 0.05)) - esp._DT_MEAN) / esp._DT_STD
        p = esp._decode(dz, s[:L], th_cls[i][:L], 0.0, 0.0, angles[i])
        if p is not None and len(p) >= 4:
            paths.append(np.asarray(p, dtype=np.float64))
            keep.append(i)
    return paths, np.asarray(keep, dtype=np.int64)


def feature_matrix(paths):
    X = extract_feature_matrix(paths)
    ok = np.all(np.isfinite(X), 1)
    return X, ok


# ----------------------------------------------------------------- logprob

def token_logprob(model, s, th, dt, cond, amp):
    """Mean log probability per DECIDED token, per row. Because the trunk is
    causally masked this reproduces exactly the logits used at sampling time.
    Positions that were not decisions are excluded: the turn token at a no
    motion event is a NULL substitution, and everything after a row's first PAD
    is forced."""
    with torch.amp.autocast("cuda", enabled=amp):
        s_lg, th_lg, dt_lg = model(*model.shift_inputs(s, th, dt),
                                   prefix_state(s, th, dt, cond), cond,
                                   s, th, dt)
    s_lg, th_lg, dt_lg = s_lg.float(), th_lg.float(), dt_lg.float()

    pad = s >= S_PAD_CLASS
    first_pad = torch.where(pad.any(1), pad.float().argmax(1),
                            torch.full_like(s[:, 0], s.shape[1] - 1))
    pos = torch.arange(s.shape[1], device=s.device).unsqueeze(0)
    live = pos <= first_pad.unsqueeze(1)          # includes the PAD decision
    motion = (s > TICK_CLASS) & (s < S_PAD_CLASS) & live

    lp = torch.log_softmax(s_lg, -1).gather(-1, s.unsqueeze(-1)).squeeze(-1) * live
    lp = lp + torch.log_softmax(dt_lg, -1).gather(
        -1, dt.unsqueeze(-1)).squeeze(-1) * live
    lp = lp + torch.log_softmax(th_lg, -1).gather(
        -1, th.unsqueeze(-1)).squeeze(-1) * motion
    n = (live.sum(1) * 2 + motion.sum(1)).clamp(min=1)
    return lp.sum(1) / n


def anchor_nll(model, batch, cond, amp):
    s, th, dt = batch
    return -token_logprob(model, s, th, dt, cond, amp).mean()


# ------------------------------------------------------------------- report

def dispersion(X, mu, sd):
    z = (X - mu) / sd
    return {f: float(z[:, k].std()) for k, f in enumerate(FEATURE_NAMES)}


def summarise(X, mu, sd):
    d = dispersion(X, mu, sd)
    z = (X - mu) / sd
    loc = {f: float(z[:, k].mean()) for k, f in enumerate(FEATURE_NAMES)}
    err = lambda names: float(np.mean([abs(np.log(max(d[f], 1e-6))) for f in names]))
    lerr = lambda names: float(np.mean([abs(loc[f]) for f in names]))
    return {"spread_err_trained": err(TRAINED), "spread_err_held": err(HELD_OUT),
            "loc_err_trained": lerr(TRAINED), "loc_err_held": lerr(HELD_OUT),
            "spread": d, "loc": loc}


# ---------------------------------------------------------------------- run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--cap", type=int, default=160)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--clip-w", type=float, default=5.0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-n", type=int, default=800)
    ap.add_argument("--human-n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--amp", action="store_true", default=True)
    a = ap.parse_args()

    t = gpu_temp()
    if t > GATE_C:
        print(f"  GPU at {t}C, above the {GATE_C}C launch gate. Not starting.")
        return
    dev = torch.device("cuda")
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    s2a = np.load(f"{D}/events_s2.npy", mmap_mode="r")
    dtha = np.load(f"{D}/events_dth.npy", mmap_mode="r")
    dta = np.load(f"{D}/events_dt.npy", mmap_mode="r")
    lens = np.load(f"{D}/events_len.npy")
    cond_all = np.load(f"{D}/events_cond.npy", mmap_mode="r")
    ok = np.flatnonzero(np.load(f"{D}/events_feat18_ok.npy"))
    perm = ok[rng.permutation(len(ok))]

    ref_rows = perm[:a.human_n]                       # the human reference
    pool_rows = perm[a.human_n:a.human_n + 400000]    # commands and anchors
    hum, _ = load_human(ref_rows, a.cap, s2a, dtha, dta, lens, cond_all)
    hp, _ = to_paths([r[0] for r in hum], [r[1] for r in hum],
                     [r[2] for r in hum], [r[3] for r in hum])
    HX, hok = feature_matrix(hp)
    HX = HX[hok]
    rng.shuffle(HX)
    mu, sd = HX.mean(0), HX.std(0)
    sd[sd == 0] = 1.0
    print(f"\n  human reference {len(HX)} rows, cap {a.cap}", flush=True)

    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    model = EventARModel(**ck["config"]).to(dev)
    model.load_state_dict(ck["model_state_dict"])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp)

    eval_rows = pool_rows[:a.eval_n]
    eval_cond = torch.tensor(np.asarray(cond_all[np.sort(eval_rows)],
                                        dtype=np.float32))
    eval_ang = np.arctan2(eval_cond[:, 3].numpy().astype(np.float64),
                          eval_cond[:, 2].numpy().astype(np.float64))
    train_rows = pool_rows[a.eval_n:]

    def evaluate(tag):
        model.eval()
        S, TH, DT = [], [], []
        for c0 in range(0, len(eval_cond), a.batch):
            c = eval_cond[c0:c0 + a.batch].to(dev)
            s, th, dt = model.sample(c, seq_len=a.cap)
            S.append(s.cpu().numpy()); TH.append(th.cpu().numpy())
            DT.append(dt.cpu().numpy())
        S = np.concatenate(S); TH = np.concatenate(TH); DT = np.concatenate(DT)
        paths, keep = to_paths(list(S), list(TH), list(DT), eval_ang)
        X, xok = feature_matrix(paths)
        X = X[xok]
        np.random.default_rng(a.seed).shuffle(X)
        r = scoring.score_features(X)
        g = scoring.gbm_cv_auc(X)
        sm = summarise(X, mu, sd)
        row = {"tag": tag, "auc_rf": float(r["auc_rf_oob"]),
               "auc_gbm": float(g["auc_gbm_cv"]),
               "collapse": bool(r["collapse_flag"]), "n": int(len(X)),
               **{k: v for k, v in sm.items() if not isinstance(v, dict)}}
        print(f"  EVAL {tag:<10} rf {row['auc_rf']:.4f}  gbm {row['auc_gbm']:.4f}"
              f"  spread err trained {row['spread_err_trained']:.4f}"
              f"  held {row['spread_err_held']:.4f}"
              f"  collapse {row['collapse']}", flush=True)
        row["detail"] = sm
        model.train()
        return row

    hist = {"config": vars(a), "held_out": HELD_OUT, "evals": [], "steps": []}
    hist["evals"].append(evaluate("base"))
    with open(OUT_JSON, "w") as f:
        json.dump(hist, f, indent=2)

    tk = [FEATURE_NAMES.index(f) for f in TRAINED]
    mu_t = torch.tensor(mu, dtype=torch.float32)
    sd_t = torch.tensor(sd, dtype=torch.float32)
    t0, peak = time.time(), 0
    model.train()

    for step in range(1, a.steps + 1):
        tnow = gpu_temp()
        peak = max(peak, tnow)
        if tnow >= KILL_C:
            print(f"  GPU hit {tnow}C, at or above the {KILL_C}C kill. Stopping.")
            break

        pick = rng.choice(len(train_rows), a.batch, replace=False)
        rows = np.sort(train_rows[pick])
        cond = torch.tensor(np.asarray(cond_all[rows], dtype=np.float32)).to(dev)
        ang = np.arctan2(cond[:, 3].cpu().numpy().astype(np.float64),
                         cond[:, 2].cpu().numpy().astype(np.float64))

        model.eval()
        s, th, dt = model.sample(cond, seq_len=a.cap)
        model.train()
        paths, keep = to_paths(list(s.cpu().numpy()), list(th.cpu().numpy()),
                               list(dt.cpu().numpy()), ang)
        if len(keep) < 16:
            continue
        X, xok = feature_matrix(paths)
        keep = keep[xok]
        z = (torch.tensor(X[xok], dtype=torch.float32) - mu_t) / sd_t
        zt = z[:, tk]

        m = zt.mean(0)
        sdev = zt.std(0).clamp(min=1e-4)
        loss_feat = float((m ** 2).sum() + (torch.log(sdev) ** 2).sum())
        c = zt - m
        w = (2 * m * c + (torch.log(sdev) / sdev ** 2) * (c ** 2 - sdev ** 2)).sum(1)
        w = (w - w.mean()) / w.std().clamp(min=1e-6)
        w = w.clamp(-a.clip_w, a.clip_w).to(dev)

        ki = torch.tensor(keep, device=dev)
        logp = token_logprob(model, s[ki], th[ki], dt[ki], cond[ki], a.amp)
        surrogate = (w * logp).mean()

        arows = np.sort(train_rows[rng.choice(len(train_rows), a.batch,
                                              replace=False)])
        ah, akept = load_human(arows, a.cap, s2a, dtha, dta, lens, cond_all)
        if len(ah) >= 8:
            L = max(len(r[0]) for r in ah)
            AS = torch.full((len(ah), L), S_PAD_CLASS, dtype=torch.long)
            ATH = torch.full((len(ah), L), TH_NULL_CLASS, dtype=torch.long)
            ADT = torch.zeros((len(ah), L), dtype=torch.long)
            for i, r in enumerate(ah):
                AS[i, :len(r[0])] = torch.from_numpy(r[0])
                ATH[i, :len(r[1])] = torch.from_numpy(r[1])
                ADT[i, :len(r[2])] = torch.from_numpy(r[2])
            acond = torch.tensor(np.asarray(cond_all[akept],
                                            dtype=np.float32)).to(dev)
            nll = anchor_nll(model, (AS.to(dev), ATH.to(dev), ADT.to(dev)),
                             acond, a.amp)
        else:
            nll = torch.zeros((), device=dev)

        loss = surrogate + a.lam * nll
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step % 10 == 0 or step == 1:
            print(f"  step {step:>4}  featloss {loss_feat:8.3f}  "
                  f"nll {float(nll.detach()):6.3f}  n {len(keep):>3}  "
                  f"{tnow}C  {(time.time() - t0) / step:5.1f} s/step",
                  flush=True)
        hist["steps"].append({"step": step, "feat_loss": loss_feat,
                              "nll": float(nll.detach()), "n": int(len(keep)),
                              "temp_c": tnow})

        if step % a.eval_every == 0:
            hist["evals"].append(evaluate(f"step{step}"))
            with open(OUT_JSON, "w") as f:
                json.dump(hist, f, indent=2)

    if not hist["evals"] or hist["evals"][-1]["tag"] == "base":
        hist["evals"].append(evaluate("final"))
    elif hist["evals"][-1]["tag"] != f"step{a.steps}":
        hist["evals"].append(evaluate("final"))

    b, e = hist["evals"][0], hist["evals"][-1]
    d_auc = b["auc_rf"] - e["auc_rf"]
    imp_t = b["spread_err_trained"] - e["spread_err_trained"]
    imp_h = b["spread_err_held"] - e["spread_err_held"]
    ratio = imp_h / imp_t if imp_t > 1e-9 else 0.0
    hist["summary"] = {"d_auc": d_auc, "improve_trained": imp_t,
                       "improve_held": imp_h, "held_over_trained": ratio,
                       "peak_temp_c": peak,
                       "wall_min": round((time.time() - t0) / 60, 1)}
    hist["summary"]["verdict"] = (
        "GOODHART" if imp_t > 0.02 and ratio < 0.5 else
        "NULL" if d_auc < 0.01 else
        "CONFIRMED" if d_auc >= 0.03 and ratio >= 0.5 and not e["collapse"] else
        "PARTIAL")
    torch.save({"config": ck["config"], "model_state_dict": model.state_dict()},
               OUT_CKPT)
    with open(OUT_JSON, "w") as f:
        json.dump(hist, f, indent=2)
    print(f"\n  base rf {b['auc_rf']:.4f} -> {e['auc_rf']:.4f}  (d {d_auc:+.4f})")
    print(f"  spread err trained {b['spread_err_trained']:.4f} -> "
          f"{e['spread_err_trained']:.4f}, held {b['spread_err_held']:.4f} -> "
          f"{e['spread_err_held']:.4f}, held over trained {ratio:.2f}")
    print(f"  VERDICT {hist['summary']['verdict']}   peak {peak}C   "
          f"{hist['summary']['wall_min']} min\n")


if __name__ == "__main__":
    main()
