"""Is the rollout gradient signal or noise? Registered before it ran.

THE QUESTION

Seven rollout objectives all land between 0.5900 and 0.5997, a band of 0.0097
against a measured draw noise of 0.0133. The critic arm showed its critic was
still finding separation at the final step, so that objective did not run out of
signal, and the score still did not move. Two explanations survive: the gradient
estimate is too noisy to use the signal that is demonstrably there, or the model
class cannot express the correction.

This separates them. Take one rollout step's worth of work, split the batch in
half, compute the REINFORCE gradient from each half independently, and read the
cosine similarity between the two. Each half is a self contained step: it forms
its own batch statistic, its own weights and its own surrogate, exactly as a real
step would with a different batch. If two independent estimates of the same
update point in unrelated directions, then every arm has been mostly diffusing
and seven objectives landing in one place is what diffusing looks like.

PREDICTION, fixed before the run: mean cosine below 0.05 for the surrogate.

FALSIFIER: cosine above about 0.2. That would say the gradient is informative,
the plateau is a real optimum of these objectives, and the estimator is not the
thing to fix.

THE CONTROL, and it is what makes the number readable. The same split is applied
to the anchor NLL, which is ordinary supervised teacher forcing on human token
streams and is known to be a usable training signal on this model at this batch
size. It sets the scale: whatever cosine a working gradient produces here is the
bar the surrogate has to be read against. Without it, "0.05" is a number with no
units.

WHAT THIS DOES NOT SETTLE. A low cosine does not prove the model could reach
0.50 with a better estimator, only that the current one is the binding
constraint. And this measures one objective. The moment objective is used
because its weights depend on nothing but the generated batch, so no human
reference enters the generated side and the split is clean; the energy and
critic objectives add a reference that both halves share, which would inflate
the agreement rather than deflate it.

Standardisation mu and sd are taken from corpus features rather than from
decoded human tokens. Both halves share them, so they cancel in a cosine, and
this avoids a decode of thousands of human rows for a quantity that cannot
affect the answer.

No model file is written and the protected eval sample is never read.
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

from models.event_ar import EventARModel  # noqa: E402
from models.event_stream_polar import S_PAD_CLASS, TH_NULL_CLASS  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402
from w4_rollout import (  # noqa: E402
    TRAINED, anchor_nll, decode_batch, gpu_temp, load_human,
    token_logprob_pos,
)

D = "training"
CKPT = "research/w4_rollout_pilot_zbuf_step100.pt"
OUT = "research/w4_gradsnr.json"
SEED = 17
HALF = 96          # each half is a full sized training batch
REPS = 6
CAP = 256
CLIP_W = 5.0
AMP = True
SCALE = 1024.0     # stands in for the GradScaler; identical on both halves so
                   # it cancels in a cosine, and it keeps fp16 grads off the floor
KILL_C = 79
COOL_C = 74
RESUME_C = 70

dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

ok = np.flatnonzero(np.load(f"{D}/events_feat18_ok.npy"))
perm = ok[rng.permutation(len(ok))]
pool = perm[4000:4000 + 400000]
train_rows = pool[2500:]          # the same rows the arms train on
cond_all = np.load(f"{D}/events_cond.npy", mmap_mode="r")
s2a = np.load(f"{D}/events_s2.npy", mmap_mode="r")
dtha = np.load(f"{D}/events_dth.npy", mmap_mode="r")
dta = np.load(f"{D}/events_dt.npy", mmap_mode="r")
lens = np.load(f"{D}/events_len.npy")

# standardisation only, see the docstring
Hall = np.load(f"{D}/events_feat18.npy", mmap_mode="r")
C = np.asarray(Hall[np.sort(rng.choice(ok, 20000, replace=False))],
               dtype=np.float64)
C = C[np.isfinite(C).all(1)]
mu, sd = C.mean(0), C.std(0)
sd[sd == 0] = 1.0
mu_t = torch.tensor(mu, dtype=torch.float32)
sd_t = torch.tensor(sd, dtype=torch.float32)
tk = [FEATURE_NAMES.index(f) for f in TRAINED]

ck = torch.load(CKPT, map_location=dev, weights_only=False)
model = EventARModel(**ck["config"]).to(dev)
model.load_state_dict(ck["model_state_dict"])
model.train()
params = [p for p in model.parameters() if p.requires_grad]


def gate():
    t = gpu_temp()
    if t >= COOL_C:
        c0 = time.time()
        while gpu_temp() > RESUME_C and time.time() - c0 < 300:
            time.sleep(10)
        t = gpu_temp()
    if t >= KILL_C:
        raise SystemExit(f"GPU {t}C, at or above the {KILL_C}C kill. Stopping.")
    return t


def flat_grad():
    g = torch.cat([(p.grad if p.grad is not None
                    else torch.zeros_like(p)).detach().flatten()
                   for p in params])
    return g.float().cpu()


def cosine(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def surrogate_grad(s, th, dt, cond, ang):
    """One self contained rollout step on the rows given, returning its gradient.

    This is the arithmetic of w4_rollout's moment branch, its weight
    normalisation and its surrogate, unchanged.
    """
    X, keep, _ = decode_batch(list(s.cpu().numpy()), list(th.cpu().numpy()),
                              list(dt.cpu().numpy()), ang)
    if len(keep) < 16:
        return None, 0
    z = (torch.tensor(X, dtype=torch.float32) - mu_t) / sd_t
    zt = z[:, tk]
    m = zt.mean(0)
    sdev = zt.std(0).clamp(min=1e-4)
    c = zt - m
    w = (2 * m * c
         + (torch.log(sdev) / sdev ** 2) * (c ** 2 - sdev ** 2)).sum(1)
    w = (w - w.mean()) / w.std().clamp(min=1e-6)
    wt = w.clamp(-CLIP_W, CLIP_W).to(dev)

    model.zero_grad(set_to_none=True)
    ki = torch.tensor(keep, device=dev)
    lp_pos, lp_n = token_logprob_pos(model, s[ki], th[ki], dt[ki],
                                     cond[ki], AMP)
    sur = (wt * lp_pos.sum(1) / lp_n).mean()
    (SCALE * sur).backward()
    g = flat_grad()
    model.zero_grad(set_to_none=True)
    del lp_pos, lp_n, sur
    return g, len(keep)


def anchor_grad(rows):
    """The supervised control, teacher forced on human streams."""
    ah, akept = load_human(np.sort(rows), CAP, s2a, dtha, dta, lens, cond_all)
    if len(ah) < 8:
        return None, 0
    L = max(len(r[0]) for r in ah)
    AS = torch.full((len(ah), L), S_PAD_CLASS, dtype=torch.long)
    ATH = torch.full((len(ah), L), TH_NULL_CLASS, dtype=torch.long)
    ADT = torch.zeros((len(ah), L), dtype=torch.long)
    for i, r in enumerate(ah):
        AS[i, :len(r[0])] = torch.from_numpy(r[0])
        ATH[i, :len(r[1])] = torch.from_numpy(r[1])
        ADT[i, :len(r[2])] = torch.from_numpy(r[2])
    acond = torch.tensor(np.asarray(cond_all[akept], dtype=np.float32)).to(dev)
    model.zero_grad(set_to_none=True)
    nll = anchor_nll(model, (AS.to(dev), ATH.to(dev), ADT.to(dev)), acond, AMP)
    (SCALE * nll).backward()
    g = flat_grad()
    model.zero_grad(set_to_none=True)
    del nll
    return g, len(ah)


print(f"\n  checkpoint {CKPT}")
print(f"  {REPS} repeats, halves of {HALF}, cap {CAP}, moment objective\n",
      flush=True)

sur_cos, anc_cos, rows_out = [], [], []
for rep in range(REPS):
    t = gate()
    pick = rng.choice(len(train_rows), 2 * HALF, replace=False)
    rws = np.sort(train_rows[pick])
    cond = torch.tensor(np.asarray(cond_all[rws], dtype=np.float32)).to(dev)
    ang = np.arctan2(cond[:, 3].cpu().numpy().astype(np.float64),
                     cond[:, 2].cpu().numpy().astype(np.float64))
    model.eval()
    # sampled in chunks of HALF because that is the batch the arms sample at and
    # 192 rows at once does not fit in 8 GB. The model is frozen, so chunking
    # changes which draw comes out and nothing else.
    ss, tt, dd = [], [], []
    with torch.no_grad():
        for c0 in range(0, len(cond), HALF):
            a, b, c = model.sample(cond[c0:c0 + HALF], seq_len=CAP)
            ss.append(a); tt.append(b); dd.append(c)
    s, th, dt = torch.cat(ss), torch.cat(tt), torch.cat(dd)
    del ss, tt, dd
    model.train()

    gs = []
    for h in (slice(0, HALF), slice(HALF, 2 * HALF)):
        g, n = surrogate_grad(s[h], th[h], dt[h], cond[h], ang[h])
        gs.append(g)
    del s, th, dt, cond
    torch.cuda.empty_cache()

    ar = rng.choice(len(train_rows), 2 * HALF, replace=False)
    ga = [anchor_grad(train_rows[ar[:HALF]])[0],
          anchor_grad(train_rows[ar[HALF:]])[0]]
    torch.cuda.empty_cache()

    cs = cosine(gs[0], gs[1]) if all(x is not None for x in gs) else float("nan")
    ca = cosine(ga[0], ga[1]) if all(x is not None for x in ga) else float("nan")
    sur_cos.append(cs)
    anc_cos.append(ca)
    rows_out.append({"rep": rep, "surrogate_cos": cs, "anchor_cos": ca})
    print(f"  rep {rep}  surrogate {cs:+.4f}   anchor control {ca:+.4f}   {t}C",
          flush=True)
    with open(OUT, "w") as f:
        json.dump({"checkpoint": CKPT, "half": HALF, "rows": rows_out}, f,
                  indent=2)

sv = np.array([x for x in sur_cos if np.isfinite(x)])
av = np.array([x for x in anc_cos if np.isfinite(x)])
print(f"\n  surrogate       mean {sv.mean():+.4f}  sd {sv.std(ddof=1):.4f}  "
      f"over {len(sv)}")
print(f"  anchor control  mean {av.mean():+.4f}  sd {av.std(ddof=1):.4f}  "
      f"over {len(av)}")
print(f"\n  ratio, surrogate over control  {sv.mean() / max(av.mean(), 1e-9):.3f}")
if sv.mean() < 0.05:
    print("  PREDICTION MET. the surrogate gradient is mostly noise and the "
          "estimator is the binding constraint.")
if sv.mean() > 0.2:
    print("  FALSIFIED. the gradient is informative, so the plateau is a real "
          "optimum of these objectives and the estimator is not the fix.")

with open(OUT, "w") as f:
    json.dump({"checkpoint": CKPT, "half": HALF, "reps": REPS, "rows": rows_out,
               "surrogate_mean": float(sv.mean()),
               "surrogate_sd": float(sv.std(ddof=1)),
               "anchor_mean": float(av.mean()),
               "anchor_sd": float(av.std(ddof=1)), "complete": True}, f,
              indent=2)
print(f"  wrote {OUT}")
