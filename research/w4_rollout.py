"""Rollout level training: match the feature distribution of what the model
actually generates.

Design registered in HANDOFF.md under "Rollout level training, registered
2026-08-10 before any training". Arms, held out features, thresholds and the
prediction are fixed there. This file implements that design and nothing else.

THE SAMPLING BUDGET IS NOT A FREE PARAMETER, corrected 2026-08-10

prefix_state feeds the model its step index divided by the width of the buffer
being generated into, so the seq_len handed to sample is an input the model
conditions on at every step and not merely a stopping rule. Training buffers are
256 wide. Every arm run from this file before this correction sampled at 160,
which tells the model it is 31 percent through its buffer where training said 20
percent, a clock running about 1.6 times fast. That is worth 0.0444 on the base
checkpoint at 7.4 sampler draw standard deviations, measured in w4_budget.

So the width is now read off the checkpoint rather than taken from the command
line, and the human reference is truncated to the same width so the two sides of
every comparison are cut alike. There is no --cap flag any more, because a flag
is exactly how this went wrong. Every number this file printed before 2026-08-10
is on the fast clock and its absolute level does not transfer.

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

THE SECOND OBJECTIVE, --objective energy

w4_gapsplit measured what the moment objective above can ever buy. Matching all
eighteen marginals exactly is worth 0.0298 of a 0.0905 gap, the correlation
matrix another 0.0235, and 0.0372 survives both. Means and spreads are a subset
of the first of those three, so the moment objective has a ceiling near 0.03 and
the first pilot reached it in ninety steps.

The energy distance has no such ceiling. Between the generated batch z and a
human batch h it is

    E = 2 mean_ij ||z_i - h_j|| - mean_ik ||z_i - z_k|| - mean_jl ||h_j - h_l||

which is zero if and only if the two distributions are equal, at every order,
not just the first two. The last term does not depend on the model and is
dropped. The score function coefficient is the partial derivative of the
statistic with respect to including trajectory i,

    phi_i = (2/m) sum_j ||z_i - h_j|| - (2/n) sum_k ||z_i - z_k||

with the factor two on the second term because z_i appears twice in the
generated double sum. Everything else in the estimator is unchanged: same
rollout, same teacher forced pass, same advantage normalisation, same anchor.
It is still a fixed statistic of a fixed feature map, so the learned critic
failure cannot recur here either.

Its ceiling is the corpus floor, 0.5455, which is the whole gap.

PER TOKEN CREDIT, --credit token, added 2026-08-11

Both arms run so far gave every token of a rollout the same weight, because the
weight is one number for the trajectory and the surrogate multiplies it by the
average log probability over roughly 250 tokens. That is REINFORCE with no
return decomposition. It can move what the model does on average and it cannot
move which token in which context, which is the dependence the gap ladder says
survives. The energy arm's behaviour is what that predicts: it falls 0.030 in a
hundred steps, lands just past the rung where the marginals are matched, and
then sits still.

--credit token splits the same weight over the tokens and uses reward to go.
The split, the nine features it covers, the three it cannot and the measured
attribution error are all in research/w4_credit.py. Nothing about the objective,
the rollout, the anchor or the held out six changes. Where the split is empty
the two modes are the same computation, which is how the change is checked.

THE GOODHART GUARD

This objective is stated over the same eighteen features the scorer reads, so
the arm is worthless without a held out set. Twelve features enter the loss. Six
never do, chosen to span the families rather than to be easy. If the trained
twelve come into line and the held out six do not, the model learned the loss
and not the movement, and the arm is a FAIL whatever the contract says.

SAFETY

Reads only training data. Never reads the protected eval sample. Writes its own
checkpoint under research/ and never touches training/event_ar_v2_s40000.pt or
training/candi_polar_flow_best.pt. Pauses at COOL_C and only aborts at KILL_C.
No DataLoader and no worker processes, so the memmap pickling hazard in
training/train_event_ar.py cannot arise here.

CHANGES AFTER THE FIRST RUN

The 2026-08-10 pilot lost 160 of its 250 steps to the thermal abort, so the loop
now pauses to cool instead of waiting for the abort to catch it, and --init
continues from a saved checkpoint. Neither touches the objective.

One registered threshold is replaced, and the replacement is registered in
HANDOFF before the continuation runs. The original CONFIRMED condition required
scoring.py to raise no collapse flag. That flag is measured against the
reference set, which was recorded on different hardware and carries a tail on
acceleration and velocity that neither the model nor the training corpus has, so
it fires on real corpus humans and cannot discriminate anything. It is replaced
by a runaway check against the corpus the model is actually matched to: no
feature's spread ratio outside [0.5, 2.0]. The pilot's mean jerk overshot to
0.439, so this gate is not vacuous.
"""
from __future__ import annotations

import argparse
import collections
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
from features import (  # noqa: E402
    FEATURE_NAMES, extract_features, resample_trajectory,
)
from models.event_ar import (  # noqa: E402
    DT_MAX_MS, EventARModel, class_to_dt_ms, prefix_state,
)
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TH_NULL_CLASS, TICK_CLASS, dth_lattice_to_class, s2_to_class,
)
from w4_credit import (  # noqa: E402
    DECOMP, HZ, credit_terms, decode_indexed, energy_grad, moment_grad,
    token_advantage,
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
# The pilot walked from 62C to the kill in eighteen minutes and lost 160 of its
# 250 steps. Pausing at COOL_C until the card is back under RESUME_C makes the
# kill unreachable instead of leaving it to catch a run that is already lost.
COOL_C = 74
RESUME_C = 70
COOL_MAX_S = 300     # cannot cool in five minutes means something else is wrong


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


def dt_to_z(dt_cls):
    ms = class_to_dt_ms(torch.from_numpy(dt_cls)).numpy().astype(np.float64)
    return (np.log(np.maximum(ms, 0.05)) - esp._DT_MEAN) / esp._DT_STD


def decode_batch(s_cls, th_cls, dt_cls, angles, want_credit=False):
    """Token streams to contract features through the SERVED decoder, with each
    surviving row's per token credit terms carried alongside it.

    decode_indexed is a line for line mirror of esp._decode that also reports
    which token produced each path point. It must never become a
    reimplementation: a hand rolled walk reads about 0.13 high because
    esp._decode snaps short steps to whole pixels and merges ticks.
    w4_credit_check confirms the two agree exactly, and verify_mirror repeats
    that on the model's own output at launch.

    Rows are dropped for a short stream, a failed decode or a feature that does
    not come out finite. Returning the surviving features, their positions in
    the batch and their credit terms from one loop is what keeps the three
    aligned; building them in separate passes and intersecting afterwards is
    how they could ever disagree.
    """
    X, keep, cred = [], [], []
    for i in range(len(angles)):
        s = s_cls[i]
        pad = np.flatnonzero(s >= S_PAD_CLASS)
        L = int(pad[0]) if len(pad) else len(s)
        if L < 8 or ((s[:L] > TICK_CLASS) & (s[:L] < S_PAD_CLASS)).sum() < 8:
            continue
        p, tok = decode_indexed(dt_to_z(dt_cls[i][:L]), s[:L],
                                th_cls[i][:L], 0.0, 0.0, angles[i])
        if p is None or len(p) < 4:
            continue
        grid = resample_trajectory(p, HZ)
        f = extract_features(grid)
        if f is None or not np.all(np.isfinite(f)):
            continue
        X.append(f)
        keep.append(i)
        cred.append(credit_terms(p, grid, tok, L) if want_credit else None)
    return (np.asarray(X, dtype=np.float64).reshape(len(X), len(FEATURE_NAMES)),
            np.asarray(keep, dtype=np.int64), cred)


def verify_mirror(s_cls, th_cls, dt_cls, angles):
    """decode_indexed against esp._decode on the model's OWN output.

    The mirror was verified on corpus streams in w4_credit_check. This repeats
    it on generated ones, because those are the distribution the estimator will
    actually be built from and the model can reach token combinations the
    corpus never does.
    """
    n = bad = 0
    for i in range(len(angles)):
        s = s_cls[i]
        pad = np.flatnonzero(s >= S_PAD_CLASS)
        L = int(pad[0]) if len(pad) else len(s)
        if L < 8:
            continue
        dz = dt_to_z(dt_cls[i][:L])
        ref = esp._decode(dz, s[:L], th_cls[i][:L], 0.0, 0.0, angles[i])
        got, _ = decode_indexed(dz, s[:L], th_cls[i][:L], 0.0, 0.0, angles[i])
        if ref is None and got is None:
            continue
        n += 1
        if (ref is None) != (got is None) or len(ref) != len(got) or not (
                np.asarray(ref, dtype=np.float64)
                == np.asarray(got, dtype=np.float64)).all():
            bad += 1
    return n, bad


# ----------------------------------------------------------------- logprob

def token_logprob_pos(model, s, th, dt, cond, amp):
    """Log probability at each position, and the per row normaliser.

    Because the trunk is causally masked this reproduces exactly the logits
    used at sampling time. Positions that were not decisions are excluded: the
    turn token at a no motion event is a NULL substitution, and everything
    after a row's first PAD is forced. Those positions come back at exactly
    zero, so a per token weight applied to them cannot leak into the update.
    """
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
    return lp, n


def token_logprob(model, s, th, dt, cond, amp):
    """Mean log probability per DECIDED token, per row."""
    lp, n = token_logprob_pos(model, s, th, dt, cond, amp)
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
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--clip-w", type=float, default=5.0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-n", type=int, default=800)
    ap.add_argument("--eval-draws", type=int, default=2,
                    help="sampler draws averaged per evaluation. one draw "
                         "carries sd 0.0141 on a trained checkpoint, which is "
                         "half of what the first energy arm appeared to move")
    ap.add_argument("--human-n", type=int, default=4000)
    ap.add_argument("--human-ref", choices=("corpus", "sir"), default="corpus",
                    help="which human sample the objective steers toward. "
                         "corpus decodes rows of the training event corpus, "
                         "which is what every arm before 2026-08-11 used and "
                         "is a different population from the one the contract "
                         "scores against, by 0.0149 of AUC at 3.5 se. sir uses "
                         "data/human_ref_features_sir.npy, 4000 rows from the "
                         "same pool as validation, disjoint from eval by index "
                         "and from validation by feature match. sir also "
                         "matches the contract's structure, raw human against "
                         "decoded generated, where corpus cancels a decoder "
                         "artifact the contract does not cancel. Default is "
                         "left on corpus so reruns of earlier arms reproduce")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--objective", choices=("moments", "energy", "critic"),
                    default="moments",
                    help="moments matches means and spreads, energy matches the "
                         "whole joint through an energy distance, critic scores "
                         "each row by a capacity limited logistic classifier "
                         "refitted every step")
    ap.add_argument("--critic-c", type=float, default=0.05,
                    help="inverse L2 strength of the critic. 0.05 on the "
                         "ninety column lift is the capacity that reads the "
                         "raw gap at only 10.3 null sd, against 40.3 for the "
                         "energy distance, which is the measured reason it "
                         "cannot outrun the generator the way the 2026-07 "
                         "critic did when it reached 0.94 by round eight")
    ap.add_argument("--critic-kind", choices=("logistic", "forest"),
                    default="forest",
                    help="the score function estimator never differentiates the "
                         "weight, so the critic does not have to be smooth and a "
                         "forest is as usable as a logistic. Measured on the "
                         "transported sample, a forest at leaf 20 depth 8 reads "
                         "3.38 null sd against the logistic's 1.42 while "
                         "replaying the same sprint profile, 0.691 against "
                         "0.697 at step forty. An unconstrained forest reads "
                         "5.42 but sprints to 0.860 and is still climbing, "
                         "which is the 2026-07 failure")
    ap.add_argument("--critic-buf", type=int, default=16,
                    help="steps of generated rows the critic is fitted on. The "
                         "current batch is scored before it is added, so every "
                         "weight is out of sample")
    ap.add_argument("--energy-m", type=int, default=512,
                    help="human sample per step for the energy distance")
    ap.add_argument("--zbuf-steps", type=int, default=1,
                    help="how many recent steps of generated rows estimate the "
                         "generated side of the energy distance. 1 is the "
                         "batch itself, which is what every arm so far did")
    ap.add_argument("--credit", choices=("trajectory", "token"),
                    default="trajectory",
                    help="trajectory gives every token of a rollout the same "
                         "weight, which is what the first two arms did. token "
                         "splits that weight over the tokens and uses reward "
                         "to go. See w4_credit")
    ap.add_argument("--resmap", type=str, default=None,
                    help="a frozen conditional residual map from w4_resmap. Its "
                         "twelve columns are appended to the twelve standardised "
                         "trained features, so the energy distance is computed "
                         "in twenty four dimensions instead of twelve. Exists "
                         "because the distance in the plain feature space is "
                         "blind to the part of the gap that is left: see the "
                         "scrambling experiment in HANDOFF")
    ap.add_argument("--resmap-weight", type=float, default=4.0,
                    help="how hard the residual columns count. Swept on a fixed "
                         "sample: 0 is the plain arm, and past 4 the new "
                         "component stops improving while the old one keeps "
                         "degrading")
    ap.add_argument("--init", type=str, default=None,
                    help="checkpoint to continue from, default is the base model")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for the result and checkpoint filenames")
    ap.add_argument("--amp", action="store_true", default=True)
    a = ap.parse_args()
    if a.objective == "critic" and a.credit == "token":
        # the critic returns one logit per row and there is no per token
        # decomposition of it. energy_grad has no counterpart here
        raise SystemExit("--objective critic needs --credit trajectory")
    if a.resmap and a.credit == "token":
        # the per token decomposition indexes zt by DECOMP's column positions and
        # energy_grad returns a gradient the width of zt. Both assume zt is the
        # trained twelve and nothing else.
        raise SystemExit("--resmap needs --credit trajectory")

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
    # the buffer width the model was trained at. prefix_state divides the step
    # index by it, so sampling into any other width feeds the model a clock it
    # has never seen. See the docstring.
    cap = int(ck["config"]["max_seq_len"])
    a.cap = cap                      # recorded in the run config for the ledger

    s2a = np.load(f"{D}/events_s2.npy", mmap_mode="r")
    dtha = np.load(f"{D}/events_dth.npy", mmap_mode="r")
    dta = np.load(f"{D}/events_dt.npy", mmap_mode="r")
    lens = np.load(f"{D}/events_len.npy")
    cond_all = np.load(f"{D}/events_cond.npy", mmap_mode="r")
    ok = np.flatnonzero(np.load(f"{D}/events_feat18_ok.npy"))
    perm = ok[rng.permutation(len(ok))]

    ref_rows = perm[:a.human_n]                       # the human reference
    pool_rows = perm[a.human_n:a.human_n + 400000]    # commands and anchors
    if a.human_ref == "sir":
        # Features only, so nothing is decoded. The commands and anchors still
        # come from the corpus via pool_rows; only what the objective is aimed
        # at changes.
        HX = np.load("data/human_ref_features_sir.npy").astype(np.float64)
        HX = HX[np.isfinite(HX).all(1)]
    else:
        hum, _ = load_human(ref_rows, cap, s2a, dtha, dta, lens, cond_all)
        HX, _, _ = decode_batch([r[0] for r in hum], [r[1] for r in hum],
                                [r[2] for r in hum], [r[3] for r in hum])
    rng.shuffle(HX)
    mu, sd = HX.mean(0), HX.std(0)
    sd[sd == 0] = 1.0
    print(f"\n  human reference {a.human_ref}, {len(HX)} rows, "
          f"buffer width {cap}", flush=True)

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

    # a single draw of 2500 rows carries sd 0.0141 on a trained checkpoint,
    # measured in w4_seedvar and w4_budget. The first energy arm read its base
    # off the highest of five draws and its endpoint off the lowest of three,
    # which inflated the fall threefold. Averaging draws is the fix, and the
    # spread across them is reported so the error bar is never guessed again.
    therm = {"peak": 0, "cooled_s": 0.0}

    def thermal():
        """Temperature after any cooling pause, and the running peak. Both loops
        call this. Two evaluation draws of 2500 rows at buffer width 256 is
        twenty minutes of continuous sampling, which is now the hottest part of
        the run, and the training loop's gate never covered it."""
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
                        f"  GPU at or above the {KILL_C}C kill during the {tag} "
                        f"evaluation. Stopping. The checkpoint and the JSON on "
                        f"disk are the state of the previous evaluation.")
                c = eval_cond[c0:c0 + a.batch].to(dev)
                s, th, dt = model.sample(c, seq_len=cap)
                S.append(s.cpu().numpy()); TH.append(th.cpu().numpy())
                DT.append(dt.cpu().numpy())
            S = np.concatenate(S); TH = np.concatenate(TH); DT = np.concatenate(DT)
            X, _, _ = decode_batch(list(S), list(TH), list(DT), eval_ang)
            np.random.default_rng(a.seed).shuffle(X)
            r = scoring.score_features(X)
            g = scoring.gbm_cv_auc(X)
            sm = summarise(X, mu, sd)
            draws.append({"auc_rf": float(r["auc_rf_oob"]),
                          "auc_gbm": float(g["auc_gbm_cv"]),
                          "collapse": bool(r["collapse_flag"]),
                          "n": float(len(X)), **sm})

        mean = lambda k: float(np.mean([d[k] for d in draws]))
        aucs = [d["auc_rf"] for d in draws]
        row = {"tag": tag, "auc_rf": mean("auc_rf"), "auc_gbm": mean("auc_gbm"),
               "auc_rf_draws": aucs,
               "auc_rf_range": float(max(aucs) - min(aucs)),
               "collapse": any(d["collapse"] for d in draws),
               "n": int(mean("n")),
               **{k: mean(k) for k in ("spread_err_trained", "spread_err_held",
                                       "loc_err_trained", "loc_err_held")}}
        print(f"  EVAL {tag:<10} rf {row['auc_rf']:.4f}"
              f" (range {row['auc_rf_range']:.4f} over {len(aucs)})"
              f"  gbm {row['auc_gbm']:.4f}"
              f"  spread err trained {row['spread_err_trained']:.4f}"
              f"  held {row['spread_err_held']:.4f}"
              f"  collapse {row['collapse']}", flush=True)
        row["detail"] = {
            "spread": {f: float(np.mean([d["spread"][f] for d in draws]))
                       for f in FEATURE_NAMES},
            "loc": {f: float(np.mean([d["loc"][f] for d in draws]))
                    for f in FEATURE_NAMES}}
        model.train()
        return row

    mirror = None
    if a.credit == "token":
        model.eval()
        with torch.no_grad():
            ms, mth, mdt = model.sample(eval_cond[:a.batch].to(dev), seq_len=cap)
        model.train()
        n_m, bad_m = verify_mirror(list(ms.cpu().numpy()), list(mth.cpu().numpy()),
                                   list(mdt.cpu().numpy()), eval_ang[:a.batch])
        del ms, mth, mdt
        print(f"  token index mirror: {bad_m} disagreements with esp._decode "
              f"over {n_m} generated rows", flush=True)
        if bad_m:
            raise SystemExit(
                "  decode_indexed does not reproduce the served decoder on "
                "generated output. Per token credit would be attributed to the "
                "wrong tokens, so this run is refused.")
        mirror = {"rows": n_m, "disagreements": bad_m}

    hist = {"config": vars(a), "held_out": HELD_OUT, "mirror": mirror,
            "evals": [], "steps": []}
    hist["evals"].append(evaluate("base"))
    with open(out_json, "w") as f:
        json.dump(hist, f, indent=2)

    tk = [FEATURE_NAMES.index(f) for f in TRAINED]
    # the nine trained features that split over the resampled path, as columns
    # of zt and as raw standardisation scales. The other three trained features
    # stay in the remainder that every token sees: max_velocity is a max, and
    # path_efficiency and max_deviation read absolute geometry, which a rigid
    # downstream shift moves and per token credit would get wrong.
    dcols = [TRAINED.index(n) for n, *_ in DECOMP]
    sd_dec = np.asarray([sd[FEATURE_NAMES.index(n)] for n, *_ in DECOMP])
    mu_t = torch.tensor(mu, dtype=torch.float32)
    sd_t = torch.tensor(sd, dtype=torch.float32)
    # the human side of the energy distance, standardised the same way and cut
    # to the trained twelve so the held out six stay out of the objective
    ZH_t = torch.tensor(((HX - mu) / sd)[:, tk], dtype=torch.float32)
    # the residual map is fitted on corpus rows only and frozen. Appending its
    # columns leaves the held out six out of the objective, because every
    # conditional in it is a trained feature given the other eleven trained ones.
    resmap = None
    if a.resmap:
        import w4_resmap
        resmap = w4_resmap.load(a.resmap)
        if resmap.apply(HX[:4]).shape[1] != len(tk):
            raise SystemExit("the residual map does not have one column per "
                             "trained feature")
        ZH_t = torch.cat([ZH_t, torch.tensor(a.resmap_weight * resmap.apply(HX),
                                             dtype=torch.float32)], 1)
        print(f"  residual map on, {ZH_t.shape[1]} columns, "
              f"weight {a.resmap_weight}")
    # The critic objective. Measured 2026-08-11: on a generated sample already
    # transported to corpus marginals and corpus rank correlations, which is
    # where all five arms so far have stopped, no distance statistic can see the
    # remaining gap at a 1536 row batch. Energy reads -0.46 null sd, a gaussian
    # mmd reads between -0.52 and 0.45 across five bandwidths, energy on the
    # quadratic lift reads -0.48, and the largest of 256 random projections
    # reads 0.15. The contract's forest reads the same sample at 3.47 and a
    # logistic regression on the lift at 1.42. A distance test spreads its power
    # over every direction at once and a classifier concentrates it where the
    # difference is, so the plateau is the objective running out of finite
    # sample power rather than the optimiser failing.
    cbuf = collections.deque(maxlen=max(2, a.critic_buf))
    critic = LH = lmu = lsd = None
    if a.objective == "critic":
        iu = np.triu_indices(len(tk))

        def lift(Z):
            """The trained twelve, then every product including squares. The
            clip is load bearing for the logistic: mean_acceleration reaches 158
            sd on corpus rows and its square would own the statistic outright. A
            forest is invariant to all of it and takes the twelve raw."""
            if a.critic_kind == "forest":
                return np.clip(Z, -40.0, 40.0)
            C = np.clip(Z, -4.0, 4.0)
            return np.hstack([C, (C[:, :, None] * C[:, None, :])[:, iu[0],
                                                                 iu[1]]])

        LH = lift(((HX - mu) / sd)[:, tk])
        lmu, lsd = LH.mean(0), LH.std(0)
        lsd[lsd == 0] = 1.0
        LH = (LH - lmu) / lsd
        if a.critic_kind == "forest":
            from sklearn.ensemble import RandomForestClassifier
            critic = RandomForestClassifier(n_estimators=100, max_depth=8,
                                            min_samples_leaf=20, n_jobs=8,
                                            random_state=42)
        else:
            from sklearn.linear_model import LogisticRegression
            critic = LogisticRegression(max_iter=2000, C=a.critic_c)
        print(f"  critic on, kind {a.critic_kind}, {LH.shape[1]} columns, "
              f"fitted on {a.critic_buf} steps of generated rows")
    zbuf = collections.deque(maxlen=max(1, a.zbuf_steps))
    t0 = time.time()
    model.train()

    for step in range(1, a.steps + 1):
        tnow = thermal()
        if tnow >= KILL_C:
            print(f"  GPU hit {tnow}C, at or above the {KILL_C}C kill after "
                  f"{COOL_MAX_S}s of cooling. Stopping.")
            break

        pick = rng.choice(len(train_rows), a.batch, replace=False)
        rows = np.sort(train_rows[pick])
        cond = torch.tensor(np.asarray(cond_all[rows], dtype=np.float32)).to(dev)
        ang = np.arctan2(cond[:, 3].cpu().numpy().astype(np.float64),
                         cond[:, 2].cpu().numpy().astype(np.float64))

        model.eval()
        s, th, dt = model.sample(cond, seq_len=cap)
        model.train()
        X, keep, cred = decode_batch(list(s.cpu().numpy()), list(th.cpu().numpy()),
                                     list(dt.cpu().numpy()), ang,
                                     want_credit=a.credit == "token")
        if len(keep) < 16:
            continue
        z = (torch.tensor(X, dtype=torch.float32) - mu_t) / sd_t
        zt = z[:, tk]
        if resmap is not None:
            zt = torch.cat([zt, torch.tensor(a.resmap_weight * resmap.apply(X),
                                             dtype=torch.float32)], 1)

        if a.objective == "critic":
            lz = (lift(zt[:, :len(tk)].numpy().astype(np.float64)) - lmu) / lsd
            nbuf = sum(len(b) for b in cbuf)
            if nbuf < 2 * len(lz):
                # cold start. Two steps of rows before the first fit, and a zero
                # weight is a skipped update rather than a random one
                w = torch.zeros(len(lz), dtype=torch.float32)
                loss_feat = 0.0
            else:
                # the fit never sees the batch it scores, so every weight is out
                # of sample and none of it is the classifier memorising the rows
                # it is about to grade
                XG = np.vstack(list(cbuf))
                XH = LH[rng.choice(len(LH), min(len(XG), len(LH)),
                                   replace=False)]
                critic.fit(np.vstack([XG, XH]),
                           np.concatenate([np.ones(len(XG)),
                                           np.zeros(len(XH))]))
                if a.critic_kind == "forest":
                    # a forest has no decision_function. The log odds is the
                    # same quantity the logistic returns, and the clip only
                    # bites where every tree agrees, which is where the weight
                    # was going to be clamped by --clip-w anyway
                    p = np.clip(critic.predict_proba(lz)[:, 1], 1e-4, 1 - 1e-4)
                    d = np.log(p / (1 - p))
                else:
                    d = critic.decision_function(lz)
                w = torch.tensor(d, dtype=torch.float32)
                loss_feat = float(w.mean())
            cbuf.append(lz)
            grad_z = None
        elif a.objective == "energy":
            ht = ZH_t[torch.from_numpy(
                rng.choice(len(ZH_t), min(a.energy_m, len(ZH_t)), replace=False))]
            # the generated side of the statistic, which is the batch itself plus
            # the rows the last zbuf_steps - 1 steps produced. Those rows carry no
            # gradient and never did: they only estimate the expectation over a
            # second independent draw. With zbuf_steps 1 zpool is zt and every
            # line below is the arithmetic the first three arms ran.
            zbuf.append(zt)
            zpool = torch.cat(list(zbuf), 0) if len(zbuf) > 1 else zt
            d_zh = torch.cdist(zt, ht)
            d_zz = torch.cdist(zt, zpool)
            # the human to human term is constant in the model, so it is dropped
            # from the reported loss as well to keep the two comparable in sign
            loss_feat = float(2 * d_zh.mean() - d_zz.mean())
            # partial derivative of the population statistic with respect to the
            # law of trajectory i. The 2 on the second term is the symmetry of
            # E||Z - Z'||, not double counting inside one batch, so it survives
            # when the second draw is estimated from a larger pool
            w = 2 * d_zh.mean(1) - (2.0 / zpool.shape[0]) * d_zz.sum(1)
            grad_z = (energy_grad(zt, ht, zpool) if a.credit == "token"
                      else None)
        else:
            m = zt.mean(0)
            sdev = zt.std(0).clamp(min=1e-4)
            loss_feat = float((m ** 2).sum() + (torch.log(sdev) ** 2).sum())
            c = zt - m
            w = (2 * m * c
                 + (torch.log(sdev) / sdev ** 2) * (c ** 2 - sdev ** 2)).sum(1)
            grad_z = moment_grad(zt) if a.credit == "token" else None

        if a.credit == "token":
            # A is a weight per position rather than per row. Normalising it
            # over the live positions rather than over the rows is the one place
            # the two credit modes are not the same computation: a per token
            # weight has to be scaled by the spread of per token weights.
            A, livem = token_advantage(w.numpy(), grad_z[:, dcols].numpy(),
                                       sd_dec, cred, s.shape[1])
            fin = A[livem]
            A = (A - fin.mean()) / max(fin.std(), 1e-6)
            wt = torch.tensor(np.clip(A, -a.clip_w, a.clip_w),
                              dtype=torch.float32, device=dev)
        else:
            w = (w - w.mean()) / w.std().clamp(min=1e-6)
            wt = w.clamp(-a.clip_w, a.clip_w).to(dev)

        opt.zero_grad(set_to_none=True)
        ki = torch.tensor(keep, device=dev)
        lp_pos, lp_n = token_logprob_pos(model, s[ki], th[ki], dt[ki],
                                         cond[ki], a.amp)
        # with a per row weight these are the same expression, because the row
        # weight factors straight out of the sum over positions
        surrogate = (((wt * lp_pos).sum(1) if a.credit == "token"
                      else wt * lp_pos.sum(1)) / lp_n).mean()
        # The surrogate is backed off before the anchor graph is built. Both are
        # a full width teacher forced forward at this batch size, and holding
        # two of them alive at once does not fit in 8 GB now that the buffer is
        # 256 wide rather than 160. Gradients accumulate, so the update is the
        # same one the single combined backward produced.
        scaler.scale(surrogate).backward()
        del lp_pos, lp_n, surrogate, s, th, dt, ki

        arows = np.sort(train_rows[rng.choice(len(train_rows), a.batch,
                                              replace=False)])
        ah, akept = load_human(arows, cap, s2a, dtha, dta, lens, cond_all)
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
            scaler.scale(a.lam * nll).backward()
            nll_v = float(nll.detach())
            del nll
        else:
            nll_v = 0.0

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step % 10 == 0 or step == 1:
            print(f"  step {step:>4}  featloss {loss_feat:8.3f}  "
                  f"nll {nll_v:6.3f}  n {len(keep):>3}  "
                  f"{tnow}C  {(time.time() - t0) / step:5.1f} s/step",
                  flush=True)
        hist["steps"].append({"step": step, "feat_loss": loss_feat,
                              "nll": nll_v, "n": int(len(keep)),
                              "temp_c": tnow})

        if step % a.eval_every == 0:
            hist["evals"].append(evaluate(f"step{step}"))
            with open(out_json, "w") as f:
                json.dump(hist, f, indent=2)
            # the pilot lost 160 of its 250 steps to a thermal kill and left no
            # checkpoint behind. Saving at each eval means the file on disk is
            # always the state the last eval row describes. Both earlier arms
            # scored best at step 100 and then got worse, and neither left that
            # state recoverable, so each eval now also keeps its own copy.
            torch.save({"config": ck["config"],
                        "model_state_dict": model.state_dict()}, out_ckpt)
            torch.save({"config": ck["config"],
                        "model_state_dict": model.state_dict()},
                       out_ckpt.replace(".pt", f"_step{step}.pt"))

    if not hist["evals"] or hist["evals"][-1]["tag"] == "base":
        hist["evals"].append(evaluate("final"))
    elif hist["evals"][-1]["tag"] != f"step{a.steps}":
        hist["evals"].append(evaluate("final"))

    b, e = hist["evals"][0], hist["evals"][-1]
    d_auc = b["auc_rf"] - e["auc_rf"]
    imp_t = b["spread_err_trained"] - e["spread_err_trained"]
    imp_h = b["spread_err_held"] - e["spread_err_held"]
    ratio = imp_h / imp_t if imp_t > 1e-9 else 0.0
    # scoring.py's collapse_flag is measured against the reference set, which was
    # recorded on different hardware and carries a tail neither the model nor the
    # corpus has, so it fires on real corpus humans and cannot discriminate. The
    # runaway check asks the same question against the corpus the model is
    # actually being matched to. Registered as a replacement, see HANDOFF.
    runaway = sorted(k for k, v in e["detail"]["spread"].items()
                     if v < 0.5 or v > 2.0)
    hist["summary"] = {"d_auc": d_auc, "improve_trained": imp_t,
                       "improve_held": imp_h, "held_over_trained": ratio,
                       "runaway_spread": runaway,
                       "collapse_flag_vs_reference": bool(e["collapse"]),
                       "peak_temp_c": therm["peak"],
                       "cooldown_min": round(therm["cooled_s"] / 60, 1),
                       "wall_min": round((time.time() - t0) / 60, 1)}
    hist["summary"]["verdict"] = (
        "GOODHART" if imp_t > 0.02 and ratio < 0.5 else
        "NULL" if d_auc < 0.01 else
        "CONFIRMED" if d_auc >= 0.03 and ratio >= 0.5 and not runaway else
        "PARTIAL")
    torch.save({"config": ck["config"], "model_state_dict": model.state_dict()},
               out_ckpt)
    with open(out_json, "w") as f:
        json.dump(hist, f, indent=2)
    print(f"\n  base rf {b['auc_rf']:.4f} -> {e['auc_rf']:.4f}  (d {d_auc:+.4f})")
    print(f"  spread err trained {b['spread_err_trained']:.4f} -> "
          f"{e['spread_err_trained']:.4f}, held {b['spread_err_held']:.4f} -> "
          f"{e['spread_err_held']:.4f}, held over trained {ratio:.2f}")
    print(f"  runaway spread vs corpus {runaway or 'none'}")
    print(f"  VERDICT {hist['summary']['verdict']}   peak {therm['peak']}C   "
          f"{hist['summary']['cooldown_min']} min cooling of "
          f"{hist['summary']['wall_min']} min\n")


if __name__ == "__main__":
    main()
