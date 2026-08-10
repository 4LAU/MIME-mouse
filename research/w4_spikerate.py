"""Does the sub millisecond wait excess manufacture the feature displacement?

Registered 2026-08-10 before the run. Hypothesis, controls, prediction and
falsifier are all fixed below and are not edited after reading the output.

WHY THIS ARM

Two independent measurements point at one channel. Free running, the model
emits a wait of one millisecond or less at motion events 0.0507 of the time
against a human 0.0301, a factor of 1.68, and that survives the token round
trip so it is not a rounding artefact. Separately, the whole remaining gap is
a coherent displacement of the eighteen feature vector: no single feature
separates, repairing one marginal HURTS sixteen times out of eighteen, and
repairing all eighteen at once buys 0.0750 of the 0.1046. The displacement is
in SPREAD, not in mean. Mean acceleration reads 2.52 times the human spread,
std velocity 2.10, std jerk 2.06, while movement duration, the one quantity
the command pins down, sits at 0.98 and is exactly right.

Those two facts have an arithmetic bridge. Instantaneous velocity is step size
over wait, so an event with a near zero wait carries a velocity several times
anything else in its trajectory, and the contract differentiates that once more
for acceleration and twice more for jerk. Every feature reading near 2.5 is a
differentiated one. So a trajectory carrying three spike events and one carrying
none land in completely different places, and the model only has to get the RATE
of spikes wrong for the across trajectory spread of those features to double.

If that bridge carries the load, the whole remaining gap has ONE cause and it
is attackable at serving time. If it does not, the displacement is spread across
the representation and the only remaining move is a training arm whose objective
sees the model's own rollouts, which is expensive and has failed twice already
in this repo (the GRPO pilot, RL_PILOT.md, and the learned critic, w4_critic).
This arm is an hour of CPU and it decides which.

THE INTERVENTION

At motion events the wait classes 0 and 1 are the sub millisecond band, class 0
being 0.5 ms. Pooled over every trajectory in an arm, a fraction of those events
is converted to a draw from THAT SAME ARM'S wait distribution over classes 2 and
up, which scales the sub millisecond rate by a multiplier while changing nothing
else about the arm's own wait marginal. Multiplier above 1 runs the other way,
converting events from the upper band into draws from the arm's own sub
millisecond band.

This is a diagnostic and not a proposed fix. Choosing the multiplier that lands
on the human rate uses knowledge of the human rate, which serving does not have.
It is still not selection: every trajectory is edited and kept, none is scored
and discarded, and the count of trajectories out equals the count in.

ARMS

  generated, multiplier 1.0        the base, unedited
  generated, multiplier 0.8        part way down
  generated, multiplier 0.594      lands the rate on the human 0.0301
  generated, multiplier 0.4        past the human rate
  generated, multiplier 0.0        every spike removed
  generated, whole wait marginal   rank mapped onto the human corpus, the upper
                                   bound on what this channel can be worth
  human corpus, multiplier 1.0     the corpus floor
  human corpus, multiplier 1.68    the model's excess injected into real humans
  human corpus, multiplier 2.5     further, to see the dose keep going

CONTROLS, both of which must come back flat or the arm is uninterpretable

  rate held, spikes reshuffled     the same number of sub millisecond events is
                                   redrawn from the sub millisecond band, so the
                                   rate is untouched and only which events carry
                                   a spike changes. If this moves the score, the
                                   edit machinery is doing something on its own
                                   and no other row can be read.
  speed channel instead            the same number of motion events has its
                                   SPEED class redrawn from the arm's own speed
                                   marginal. Step size was measured clean, its
                                   autocorrelation matching a hand to three
                                   decimals at every lag out to thirty, so a
                                   perturbation of comparable size there must
                                   not help.

PREDICTION, fixed before the run

The generated dose curve falls as the multiplier drops, reaches its minimum at
or near 0.594, and rises again by multiplier 0.0, because removing every spike
overshoots into a population no hand produces either. The human corpus arm runs
the other way and rises at 1.68. Both null controls sit inside 0.01 of their
base.

READING

  CONFIRMED  the dose minimum sits at multiplier 1.0 or below, at least 0.03
             beneath the base, both null controls inside 0.01, and the human
             corpus arm rises at 1.68. The channel carries the gap and the next
             arm is a serving time controller, not a training run.
  PARTIAL    the minimum is 0.01 to 0.03 beneath the base. The channel is a
             contributor and not the cause.
  NULL       the whole sweep spans less than 0.01, or a null control moves as
             much as the treatment. The wait rate is not the route and the next
             arm is rollout level training.

Scored with research/autoloop/scoring.py, the contract, against its own
reference. Nothing is generated in this file: it reuses token streams already
sampled one per command with no selection. The protected eval sample is never
read and no model file is written.
"""
from __future__ import annotations

import json
import os
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
from features import extract_feature_matrix  # noqa: E402
from models.event_ar import DT_MAX_MS, class_to_dt_ms  # noqa: E402
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TH_NULL_CLASS, TICK_CLASS, dth_lattice_to_class, s2_to_class,
)

D = "training"
TOK = ("/tmp/claude-1000/-home-aaronadmin/"
       "059c9656-a421-4ab6-9053-614d1dc15765/scratchpad/gen_tokens.npz")
OUT = "research/w4_spikerate.json"
T = 256
SEED = 17
EDIT_SEEDS = (101, 102, 103)
SUB_MS_MAX_CLASS = 1          # classes 0 and 1 are 0.5 ms and 1 ms
N_HUMAN = 2600                # corpus rows for the human arm, disjoint from idx


# ---------------------------------------------------------------- load arms

def motion_mask(s):
    return (s > TICK_CLASS) & (s < S_PAD_CLASS)


z = np.load(TOK)
S_G, TH_G, DT_G, gen_idx, cond = z["s"], z["th"], z["dt"], z["idx"], z["cond"]
ang_g = np.arctan2(cond[:, 3].astype(np.float64), cond[:, 2].astype(np.float64))

gen = []
for j in range(len(cond)):
    s = S_G[j].astype(np.int64)
    pad = s >= S_PAD_CLASS
    L = int(pad.argmax()) if pad.any() else T
    if L < 8 or motion_mask(s[:L]).sum() < 8:
        continue
    gen.append([s[:L], TH_G[j, :L].astype(np.int64),
                DT_G[j, :L].astype(np.int64), float(ang_g[j])])

s2a = np.load(f"{D}/events_s2.npy", mmap_mode="r")
dtha = np.load(f"{D}/events_dth.npy", mmap_mode="r")
dta = np.load(f"{D}/events_dt.npy", mmap_mode="r")
lens = np.load(f"{D}/events_len.npy")
cond_all = np.load(f"{D}/events_cond.npy", mmap_mode="r")
ok = np.flatnonzero(np.load(f"{D}/events_feat18_ok.npy"))

# disjoint from the rows whose commands drove generation, so no trajectory
# appears on both sides of anything measured here
used = set(int(v) for v in gen_idx)
pool = np.array([v for v in ok[np.random.default_rng(SEED).permutation(len(ok))]
                 if int(v) not in used][:N_HUMAN * 3])

hum = []
for j in pool:
    L = min(int(lens[j]), T)
    if L < 8:
        continue
    s2 = torch.from_numpy(np.asarray(s2a[j, :L]).astype(np.int64))
    dth = torch.from_numpy(np.asarray(dtha[j, :L]).astype(np.int64))
    s_c = s2_to_class(s2).numpy()
    if motion_mask(s_c).sum() < 8:
        continue
    th_c = torch.where(s2 > 0, dth_lattice_to_class(dth),
                       torch.full_like(dth, TH_NULL_CLASS)).numpy()
    # the training target: raw milliseconds rounded to a whole class, which is
    # what the model was fit to emit and therefore the only fair comparison
    dt_c = np.round(np.asarray(dta[j, :L]).astype(np.float64)
                    ).clip(0, DT_MAX_MS).astype(np.int64)
    c = np.asarray(cond_all[j], dtype=np.float64)
    hum.append([s_c, th_c, dt_c, float(np.arctan2(c[3], c[2]))])
    if len(hum) >= N_HUMAN:
        break

print(f"\n  generated {len(gen)}, human corpus {len(hum)}", flush=True)


# ------------------------------------------------------------- edit helpers

def event_index(rows):
    """Every motion event in the arm as a flat (row, position) index."""
    r, p = [], []
    for i, (s, th, dt, a) in enumerate(rows):
        w = np.flatnonzero(motion_mask(s))
        r.append(np.full(len(w), i, dtype=np.int64))
        p.append(w)
    return np.concatenate(r), np.concatenate(p)


def bands(rows, ri, pi):
    d = np.array([rows[i][2][k] for i, k in zip(ri, pi)], dtype=np.int64)
    return d, d <= SUB_MS_MAX_CLASS


def copy_rows(rows):
    return [[s, th, dt.copy(), a] for s, th, dt, a in rows]


def scale_rate(rows, ri, pi, mult, rng):
    """Scale the arm's sub millisecond wait rate at motion events by `mult`,
    drawing every replacement from the arm's OWN wait marginal so nothing but
    the rate moves."""
    out = copy_rows(rows)
    d, lo = bands(rows, ri, pi)
    lo_pool, hi_pool = d[lo], d[~lo]
    n_lo = int(lo.sum())
    target = int(round(n_lo * mult))
    target = max(0, min(target, len(d)))
    if target < n_lo:
        pick = rng.choice(np.flatnonzero(lo), n_lo - target, replace=False)
        draw = rng.choice(hi_pool, len(pick), replace=True)
    elif target > n_lo:
        take = min(target - n_lo, len(hi_pool))
        pick = rng.choice(np.flatnonzero(~lo), take, replace=False)
        draw = rng.choice(lo_pool, len(pick), replace=True)
    else:
        return out, float(n_lo / len(d))
    for k, v in zip(pick, draw):
        out[ri[k]][2][pi[k]] = int(v)
    _, lo2 = bands(out, ri, pi)
    return out, float(lo2.mean())


def reshuffle_control(rows, ri, pi, rng):
    """Same count of sub millisecond events, redrawn from the sub millisecond
    band. The rate is untouched; only which events carry a spike changes."""
    out = copy_rows(rows)
    d, lo = bands(rows, ri, pi)
    idx_lo = np.flatnonzero(lo)
    draw = rng.choice(d[lo], len(idx_lo), replace=True)
    for k, v in zip(idx_lo, draw):
        out[ri[k]][2][pi[k]] = int(v)
    return out


def speed_control(rows, ri, pi, n, rng):
    """The same number of motion events has its SPEED class redrawn from the
    arm's own speed marginal. Step size was measured clean, so this must not
    help."""
    out = copy_rows(rows)
    out = [[s.copy(), th, dt, a] for s, th, dt, a in out]
    pool_s = np.array([rows[i][0][k] for i, k in zip(ri, pi)], dtype=np.int64)
    pick = rng.choice(len(ri), min(n, len(ri)), replace=False)
    draw = rng.choice(pool_s, len(pick), replace=True)
    for k, v in zip(pick, draw):
        out[ri[k]][0][pi[k]] = int(v)
    return out


def rank_map_waits(rows, ri, pi, donor_rows, rng):
    """Map the arm's motion event waits onto the donor's by rank, the upper
    bound on what this whole channel can be worth."""
    out = copy_rows(rows)
    d, _ = bands(rows, ri, pi)
    dri, dpi = event_index(donor_rows)
    dd, _ = bands(donor_rows, dri, dpi)
    donor = np.sort(rng.choice(dd, len(d), replace=len(dd) < len(d)))
    order = np.argsort(np.argsort(d, kind="stable"), kind="stable")
    new = donor[order]
    for k, v in zip(range(len(ri)), new):
        out[ri[k]][2][pi[k]] = int(v)
    return out


# ---------------------------------------------------------------- scoring

def score(rows, tag):
    paths = []
    for s, th, dt, a in rows:
        ms = class_to_dt_ms(torch.from_numpy(dt)).numpy().astype(np.float64)
        dz = (np.log(np.maximum(ms, 0.05)) - esp._DT_MEAN) / esp._DT_STD
        p = esp._decode(dz, s, th, 0.0, 0.0, a)
        if p is not None and len(p) >= 4:
            paths.append(np.asarray(p, dtype=np.float64))
    X = extract_feature_matrix(paths)
    X = X[np.all(np.isfinite(X), 1)]
    # the corpus is ordered by session, so an unshuffled prefix scores a narrow
    # band of people and reads 0.02 to 0.03 high
    np.random.default_rng(SEED).shuffle(X)
    r = scoring.score_features(X)
    print(f"  {tag:<38}{r['auc_rf_oob']:>9.4f}   n {r['n_per_class']}",
          flush=True)
    return {"auc": float(r["auc_rf_oob"]), "n": int(r["n_per_class"]),
            "collapse": bool(r["collapse_flag"])}


def mean_sd(vals):
    v = np.asarray(vals, dtype=np.float64)
    return float(v.mean()), float(v.std())


t0 = time.time()
gri, gpi = event_index(gen)
hri, hpi = event_index(hum)
gd, glo = bands(gen, gri, gpi)
hd, hlo = bands(hum, hri, hpi)
rate_g, rate_h = float(glo.mean()), float(hlo.mean())
print(f"  sub millisecond rate at motion events, generated {rate_g:.4f}, "
      f"human {rate_h:.4f}, factor {rate_g / rate_h:.3f}\n", flush=True)

res = {"rate_gen": rate_g, "rate_hum": rate_h, "ratio": rate_g / rate_h,
       "n_gen": len(gen), "n_hum": len(hum), "arms": {}}

print(f"  {'arm':<38}{'auc':>9}\n", flush=True)
res["arms"]["gen_base"] = score(gen, "generated, multiplier 1.0")
res["arms"]["hum_base"] = score(hum, "human corpus, multiplier 1.0")

for mult in (0.8, 0.594, 0.4, 0.0):
    aucs, rates = [], []
    for sd in EDIT_SEEDS:
        rows, r = scale_rate(gen, gri, gpi, mult, np.random.default_rng(sd))
        aucs.append(score(rows, f"generated, multiplier {mult}")["auc"])
        rates.append(r)
    m, s = mean_sd(aucs)
    res["arms"][f"gen_mult_{mult}"] = {"auc": m, "sd": s, "all": aucs,
                                       "rate": float(np.mean(rates))}
    print(f"    mean {m:.4f} sd {s:.4f} at rate {np.mean(rates):.4f}\n",
          flush=True)

for mult in (1.68, 2.5):
    aucs, rates = [], []
    for sd in EDIT_SEEDS:
        rows, r = scale_rate(hum, hri, hpi, mult, np.random.default_rng(sd))
        aucs.append(score(rows, f"human corpus, multiplier {mult}")["auc"])
        rates.append(r)
    m, s = mean_sd(aucs)
    res["arms"][f"hum_mult_{mult}"] = {"auc": m, "sd": s, "all": aucs,
                                       "rate": float(np.mean(rates))}
    print(f"    mean {m:.4f} sd {s:.4f} at rate {np.mean(rates):.4f}\n",
          flush=True)

aucs = [score(reshuffle_control(gen, gri, gpi, np.random.default_rng(sd)),
              "CONTROL rate held, spikes reshuffled")["auc"]
        for sd in EDIT_SEEDS]
m, s = mean_sd(aucs)
res["arms"]["gen_reshuffle"] = {"auc": m, "sd": s, "all": aucs}
print(f"    mean {m:.4f} sd {s:.4f}\n", flush=True)

n_touch = int(round(glo.sum() * 0.406))    # the count the 0.594 arm moves
aucs = [score(speed_control(gen, gri, gpi, n_touch, np.random.default_rng(sd)),
              "CONTROL speed channel instead")["auc"]
        for sd in EDIT_SEEDS]
m, s = mean_sd(aucs)
res["arms"]["gen_speed_control"] = {"auc": m, "sd": s, "all": aucs,
                                    "n_touched": n_touch}
print(f"    mean {m:.4f} sd {s:.4f}\n", flush=True)

aucs = [score(rank_map_waits(gen, gri, gpi, hum, np.random.default_rng(sd)),
              "generated, whole wait marginal mapped")["auc"]
        for sd in EDIT_SEEDS]
m, s = mean_sd(aucs)
res["arms"]["gen_rankmap"] = {"auc": m, "sd": s, "all": aucs}
print(f"    mean {m:.4f} sd {s:.4f}\n", flush=True)

base = res["arms"]["gen_base"]["auc"]
curve = {k: v["auc"] for k, v in res["arms"].items() if k.startswith("gen_mult")}
best = min(curve, key=curve.get)
drop = base - curve[best]
ctl = max(abs(res["arms"]["gen_reshuffle"]["auc"] - base),
          abs(res["arms"]["gen_speed_control"]["auc"] - base))
res["base"] = base
res["best_arm"] = best
res["best_drop"] = float(drop)
res["worst_control_move"] = float(ctl)
res["verdict"] = ("NULL" if drop < 0.01 or ctl >= drop else
                  "CONFIRMED" if drop >= 0.03 else "PARTIAL")
res["elapsed_s"] = round(time.time() - t0, 1)

print(f"\n  base {base:.4f}, best {best} at {curve[best]:.4f}, "
      f"drop {drop:.4f}, worst control move {ctl:.4f}")
print(f"  VERDICT {res['verdict']}   ({res['elapsed_s']:.0f} s)\n")

with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(f"  wrote {OUT}\n")
