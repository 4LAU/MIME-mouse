"""Does the relaxed decode track the served one, and does its gradient carry
signal? Registered before it ran.

Two questions, in the order that can kill the arm cheapest.

PART ONE, FIDELITY. On one batch of sampled tokens, compare the eighteen
features the relaxed path produces against the eighteen the served decoder
produces from the same tokens. A feature whose relaxed value does not move with
the served value across a batch has a Jacobian pointing somewhere unrelated to
the thing being scored, and no amount of low variance rescues that.

  PREDICTION: correlation above 0.9 on at least nine of the trained twelve, and
  above 0.5 on all twelve.
  FALSIFIER: any of mean_velocity, std_velocity, movement_duration or
  path_efficiency below 0.5. Those four are near direct functions of the token
  values, so if the relaxed path cannot track them it is wrong rather than
  merely approximate, and the arm stops here.

PART TWO, SIGNAL. The same statistic w4_gradsnr2 used, on the same checkpoint
and batch size, so the numbers are directly comparable to what is already
measured:

    REINFORCE surrogate   block normalised pairwise cosine  -0.0071 se 0.0125
    supervised anchor NLL                                   +0.0942 se 0.0192

  PREDICTION: the pathwise gradient reads above +0.05, at least half the
  supervised control, and clears the REINFORCE surrogate by more than three
  standard errors.
  FALSIFIER: at or below +0.02. That would say a pathwise gradient through this
  decode is no better determined than the score function one, that the plateau
  is not an estimator problem after all, and that the remaining explanation is
  the model class.

WHAT NEITHER PART SETTLES. Agreement is not correctness. A pathwise gradient can
be perfectly consistent across batches and still be consistently biased, because
the relaxation drops the snap, the rounding and the tick merge, and those are
real parts of what the scorer sees. Part one bounds that bias by measuring it at
the features, which is where it matters, but it does not eliminate it. Only a
training run scored on the contract can do that, and it is the next thing, not
this thing.

No model file is written and the protected eval sample is never read.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import torch

os.environ.setdefault("EVENT_SNAP", "2.5")
for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.event_ar import EventARModel  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402
from w4_rollout import TRAINED, decode_batch, gpu_temp  # noqa: E402
from w4_softdec import soft_forward, straight_through  # noqa: E402

D = "training"
CKPT = "research/w4_rollout_pilot_zbuf_step100.pt"
OUT = "research/w4_softdec_check.json"
SEED = 17
BATCH = 96
K = 8
CAP = 256
CLIP_W = 5.0
KILL_C = 79

dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

ok = np.flatnonzero(np.load(f"{D}/events_feat18_ok.npy"))
perm = ok[rng.permutation(len(ok))]
train_rows = perm[4000:4000 + 400000][2500:]
cond_all = np.load(f"{D}/events_cond.npy", mmap_mode="r")

Hall = np.load(f"{D}/events_feat18.npy", mmap_mode="r")
C = np.asarray(Hall[np.sort(rng.choice(ok, 20000, replace=False))],
               dtype=np.float64)
C = C[np.isfinite(C).all(1)]
mu, sd = C.mean(0), C.std(0)
sd[sd == 0] = 1.0
mu_t = torch.tensor(mu, dtype=torch.float32, device=dev)
sd_t = torch.tensor(sd, dtype=torch.float32, device=dev)
tk = [FEATURE_NAMES.index(f) for f in TRAINED]

ck = torch.load(CKPT, map_location=dev, weights_only=False)
model = EventARModel(**ck["config"]).to(dev)
model.load_state_dict(ck["model_state_dict"])
model.eval()
names = [n for n, p in model.named_parameters() if p.requires_grad]
params = [p for p in model.parameters() if p.requires_grad]
sizes = [p.numel() for p in params]
offs = np.cumsum([0] + sizes)


def draw(rows):
    cond = torch.tensor(np.asarray(cond_all[np.sort(rows)],
                                   dtype=np.float32)).to(dev)
    ang = np.arctan2(cond[:, 3].cpu().numpy().astype(np.float64),
                     cond[:, 2].cpu().numpy().astype(np.float64))
    with torch.no_grad():
        s, th, dt = model.sample(cond, seq_len=CAP)
    Xh, keep, _ = decode_batch(list(s.cpu().numpy()), list(th.cpu().numpy()),
                               list(dt.cpu().numpy()), ang)
    return s, th, dt, cond, ang, Xh, keep


def flat_grad():
    g = torch.cat([(p.grad if p.grad is not None
                    else torch.zeros_like(p)).detach().flatten()
                   for p in params]).double().cpu()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return g


def blocknorm(g):
    o = g.clone()
    for i in range(len(sizes)):
        sl = slice(int(offs[i]), int(offs[i + 1]))
        n = o[sl].norm()
        if n > 0:
            o[sl] /= n
    return o


def cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))


def pairstats(G):
    v = np.array([cos(a, b) for a, b in itertools.combinations(G, 2)])
    return float(v.mean()), float(v.std(ddof=1) / len(v) ** 0.5)


# --------------------------------------------------------------- part one
print(f"\n  PART ONE, fidelity of the relaxed features\n", flush=True)
s, th, dt, cond, ang, Xh, keep = draw(train_rows[
    rng.choice(len(train_rows), BATCH, replace=False)])
angt = torch.tensor(ang, dtype=torch.float32, device=dev)
with torch.no_grad():
    Xs, _ = soft_forward(model, s, th, dt, cond, angt)
Xs = Xs[torch.tensor(keep, device=dev)].cpu().numpy().astype(np.float64)
print(f"  {len(keep)} of {BATCH} rows survived the served decode")

fid = {}
for i, n in enumerate(FEATURE_NAMES):
    a, b = Xh[:, i], Xs[:, i]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        r = float("nan")
    else:
        r = float(np.corrcoef(a, b)[0, 1])
    scale = float(np.mean(np.abs(b)) / max(np.mean(np.abs(a)), 1e-12))
    fid[n] = {"corr": r, "scale_soft_over_hard": scale,
              "trained": n in TRAINED}
    flag = "trained" if n in TRAINED else "held out"
    print(f"    {n:<26} {flag:<9}  corr {r:+.4f}   relaxed/served scale "
          f"{scale:7.3f}")

tr = {n: fid[n]["corr"] for n in TRAINED}
good = sum(1 for v in tr.values() if v > 0.9)
worst = min(tr.values())
core = ["mean_velocity", "std_velocity", "movement_duration", "path_efficiency"]
core_min = min(tr[n] for n in core)
p1_met = good >= 9 and worst > 0.5
p1_fals = core_min < 0.5
print(f"\n  {good} of 12 trained features above 0.9, worst trained "
      f"{worst:+.4f}, worst of the four core {core_min:+.4f}")
print(f"  PART ONE PREDICTION {'MET' if p1_met else 'NOT MET'}    "
      f"FALSIFIER {'TRIGGERED' if p1_fals else 'not met'}")

res = {"fidelity": fid, "part1_met": bool(p1_met), "part1_falsified": bool(p1_fals)}
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)

if p1_fals:
    print("\n  stopping before part two. the relaxed path cannot track features "
          "that are near direct functions of the token values, so it is wrong "
          "rather than approximate and its gradient is not worth measuring.")
    raise SystemExit(0)

# --------------------------------------------------------------- part two
print(f"\n  PART TWO, gradient agreement over {K} batches, "
      f"{K * (K - 1) // 2} pairs\n", flush=True)

G = []
for k in range(K):
    t = gpu_temp()
    if t >= KILL_C:
        raise SystemExit(f"GPU {t}C, stopping.")
    s, th, dt, cond, ang, Xh, keep = draw(train_rows[
        rng.choice(len(train_rows), BATCH, replace=False)])
    if len(keep) < 16:
        continue
    angt = torch.tensor(ang, dtype=torch.float32, device=dev)
    Xs, _ = soft_forward(model, s, th, dt, cond, angt)
    ki = torch.tensor(keep, device=dev)
    X = straight_through(torch.tensor(Xh, dtype=torch.float32, device=dev),
                         Xs[ki])
    z = (X - mu_t) / sd_t
    zt = z[:, tk]
    # the moment objective, the same one w4_gradsnr2 measured, now as a direct
    # loss on the features rather than as a weight on a log probability
    m = zt.mean(0)
    sdev = zt.std(0).clamp(min=1e-4)
    loss = (m ** 2).sum() + (torch.log(sdev) ** 2).sum()
    model.zero_grad(set_to_none=True)
    loss.backward()
    G.append(flat_grad())
    print(f"  batch {k}  loss {float(loss):.4f}  |g| {G[-1].norm():.4f}  "
          f"rows {len(keep)}  {t}C", flush=True)

r, se = pairstats(G)
rb, seb = pairstats([blocknorm(g) for g in G])
print(f"\n  pathwise      raw pairwise cos {r:+.4f} se {se:.4f}"
      f"    block normalised {rb:+.4f} se {seb:.4f}")
print(f"  for comparison, already measured on this checkpoint and batch size")
print(f"    REINFORCE surrogate   block  -0.0071 se 0.0125")
print(f"    supervised anchor     block  +0.0942 se 0.0192")
d = rb - (-0.0071)
dse = (seb ** 2 + 0.0125 ** 2) ** 0.5
print(f"\n  pathwise minus REINFORCE  {d:+.4f}  se {dse:.4f}  "
      f"({d / dse:.1f} se)")
p2_met = rb > 0.05 and d > 3 * dse
p2_fals = rb <= 0.02
print(f"  PART TWO PREDICTION {'MET' if p2_met else 'NOT MET'}    "
      f"FALSIFIER {'TRIGGERED' if p2_fals else 'not met'}")

print("\n  per tensor, blocks holding the most gradient")
share = sorted(((float(np.mean([float(g[slice(int(offs[i]), int(offs[i + 1]))]
                                      .norm()) ** 2 for g in G])), i, n)
                for i, n in enumerate(names)), reverse=True)
tot = sum(s_ for s_, _, _ in share)
for sq, i, n in share[:8]:
    sl = slice(int(offs[i]), int(offs[i + 1]))
    print(f"    {n:<44} {100 * sq / tot:5.1f}% of |g|^2   cos "
          f"{pairstats([g[sl] for g in G])[0]:+.4f}")

res.update({"pathwise_raw": r, "pathwise_raw_se": se,
            "pathwise_block": rb, "pathwise_block_se": seb,
            "vs_reinforce": d, "vs_reinforce_se": dse,
            "part2_met": bool(p2_met), "part2_falsified": bool(p2_fals),
            "n_batches": len(G)})
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(f"\n  wrote {OUT}")
