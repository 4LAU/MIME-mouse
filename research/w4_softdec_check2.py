"""Which features are exploding the pathwise gradient, and does dropping them
recover signal? Registered before it ran.

WHAT CAME BEFORE, AND WHAT IS NOT BEING RETRACTED

w4_softdec_check ran two registered parts and this is what they returned.

Part one, fidelity of the relaxed features against the served decoder on the
same tokens, trained twelve only:

    mean_velocity        +0.9957     path_efficiency      +0.9988
    std_velocity         +0.9738     max_deviation        +0.9998
    max_velocity         +0.9996     curvature_mean       +0.6277
    mean_acceleration    -0.9054     movement_duration    +1.0000
    std_acceleration     +0.9563     angular_velocity_mean +0.9806
    mean_jerk            -0.5572     std_jerk             +0.9779

Its prediction required nine above 0.9 and all twelve above 0.5. Nine cleared
0.9 and two came back negative, so the prediction was not met. Its falsifier,
any of the four core features below 0.5, was not met: the worst of those is
+0.9738.

Part two, gradient agreement, block normalised pairwise cosine -0.0088 se
0.0466 against a registered falsifier of 0.02 or below. **That falsifier is
met and it stays met.** Nothing here reinterprets it. The run below is a new
experiment with its own prediction, and if it succeeds the honest statement is
that the first construction failed and a second one worked, not that the first
result was misread.

WHAT PROMPTED A SECOND CONSTRUCTION

The eight gradient norms in that run were 728, 10260, 17458, 17494, 166842,
247093, 1.3e9 and 2.3e12. Six orders of magnitude between batches of the same
quantity at the same weights is not a converged measurement of anything, and a
cosine computed on vectors that a single entry can dominate carries little
information either way. So the second part answered its question about that
particular construction and left the general question open.

Two suspects, both visible in part one and both pointing at the same three
features. `mean_acceleration` and `mean_jerk` are anti correlated with the
served values, so their Jacobians push against the thing being scored rather
than merely approximating it. All three of those and `curvature_mean` are built
from second and third differences on a 125 Hz grid, where each division by the
step multiplies by 125, so a third difference carries about 125 cubed, and
curvature divides by a cubed speed on top of that.

THE RUN

  A  Attribution. On one batch, take the gradient of each trained feature's own
     term in the objective separately and report its norm. This says which
     features carry the explosion instead of assuming it.
  B  The same eight batch cosine as before, with the objective restricted to the
     trained features whose relaxed value tracks the served one above 0.9. That
     set is recomputed here rather than copied, so it cannot silently drift.

PREDICTION: on the restricted objective the block normalised pairwise cosine is
above +0.05, and the spread of gradient norms across batches falls below one
order of magnitude.

FALSIFIER: at or below +0.02 again. Two independent constructions of a pathwise
gradient through this decode failing the same way would say the problem is not
the choice of features, and that the next thing to doubt is the relaxation
itself or the model class, not the objective.

WHAT THIS STILL CANNOT SETTLE. Restricting the objective to nine features means
the three dropped ones are no longer trained at all. If they carry separation on
the contract, this buys gradient quality at the cost of coverage, and only a
scored training run shows which trade wins. It also remains true that agreement
is not correctness: the relaxation drops the snap, the rounding and the tick
merge, and a consistent gradient can be consistently biased.

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
OUT = "research/w4_softdec_check2.json"
SEED = 17
BATCH = 96
K = 8
CAP = 256
FID_MIN = 0.9
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
    return s, th, dt, cond, torch.tensor(ang, dtype=torch.float32,
                                         device=dev), Xh, keep


def zt_of(s, th, dt, cond, angt, Xh, keep, cols):
    Xs, _ = soft_forward(model, s, th, dt, cond, angt)
    X = straight_through(torch.tensor(Xh, dtype=torch.float32, device=dev),
                         Xs[torch.tensor(keep, device=dev)])
    return ((X - mu_t) / sd_t)[:, cols]


def moment_loss(zt):
    m = zt.mean(0)
    sdev = zt.std(0).clamp(min=1e-4)
    return (m ** 2).sum() + (torch.log(sdev) ** 2).sum()


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


# ---------------------------------------------------------- fidelity, fresh
print("\n  fidelity of the relaxed features, recomputed here\n", flush=True)
s, th, dt, cond, angt, Xh, keep = draw(
    train_rows[rng.choice(len(train_rows), BATCH, replace=False)])
with torch.no_grad():
    Xs0, _ = soft_forward(model, s, th, dt, cond, angt)
Xs0 = Xs0[torch.tensor(keep, device=dev)].cpu().numpy().astype(np.float64)

fid, faithful = {}, []
for n in TRAINED:
    i = FEATURE_NAMES.index(n)
    a, b = Xh[:, i], Xs0[:, i]
    r = (float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12
         and np.std(b) > 1e-12 else float("nan"))
    fid[n] = r
    if r > FID_MIN:
        faithful.append(n)
    print(f"    {n:<24} corr {r:+.4f}   "
          f"{'kept' if r > FID_MIN else 'DROPPED'}")
cols = [FEATURE_NAMES.index(n) for n in faithful]
print(f"\n  {len(faithful)} of {len(TRAINED)} trained features kept")

# ---------------------------------------------------------- A, attribution
print("\n  A. gradient norm of each trained feature's own term, one batch\n",
      flush=True)
attrib = {}
for n in TRAINED:
    c = [FEATURE_NAMES.index(n)]
    zt = zt_of(s, th, dt, cond, angt, Xh, keep, c)
    model.zero_grad(set_to_none=True)
    moment_loss(zt).backward()
    g = flat_grad()
    attrib[n] = float(g.norm())
    print(f"    {n:<24} |g| {attrib[n]:>18.4f}"
          f"   {'kept' if n in faithful else 'DROPPED'}")

# --------------------------------------------------------- B, the statistic
print(f"\n  B. gradient agreement on the restricted objective, {K} batches\n",
      flush=True)
G, norms = [], []
for k in range(K):
    t = gpu_temp()
    if t >= KILL_C:
        raise SystemExit(f"GPU {t}C, stopping.")
    s, th, dt, cond, angt, Xh, keep = draw(
        train_rows[rng.choice(len(train_rows), BATCH, replace=False)])
    if len(keep) < 16:
        continue
    zt = zt_of(s, th, dt, cond, angt, Xh, keep, cols)
    loss = moment_loss(zt)
    model.zero_grad(set_to_none=True)
    loss.backward()
    g = flat_grad()
    G.append(g)
    norms.append(float(g.norm()))
    print(f"  batch {k}  loss {float(loss.detach()):.4f}  |g| {norms[-1]:.4f}  "
          f"rows {len(keep)}  {t}C", flush=True)

r, se = pairstats(G)
rb, seb = pairstats([blocknorm(g) for g in G])
spread = max(norms) / max(min(norms), 1e-30)
print(f"\n  restricted pathwise   raw {r:+.4f} se {se:.4f}"
      f"    block normalised {rb:+.4f} se {seb:.4f}")
print(f"  gradient norm spread across batches  {spread:.1f}x")
print(f"\n  already measured on this checkpoint and batch size")
print(f"    REINFORCE surrogate        block  -0.0071 se 0.0125")
print(f"    supervised anchor          block  +0.0942 se 0.0192")
print(f"    pathwise, all twelve       block  -0.0088 se 0.0466")

met = rb > 0.05 and spread < 10
fals = rb <= 0.02
print(f"\n  PREDICTION {'MET' if met else 'NOT MET'}    "
      f"FALSIFIER {'TRIGGERED' if fals else 'not met'}")

print("\n  per tensor, blocks holding the most gradient")
share = sorted(((float(np.mean([float(g[slice(int(offs[i]), int(offs[i + 1]))]
                                      .norm()) ** 2 for g in G])), i, n)
                for i, n in enumerate(names)), reverse=True)
tot = sum(s_ for s_, _, _ in share)
for sq, i, n in share[:8]:
    sl = slice(int(offs[i]), int(offs[i + 1]))
    print(f"    {n:<44} {100 * sq / tot:5.1f}% of |g|^2   cos "
          f"{pairstats([g[sl] for g in G])[0]:+.4f}")

with open(OUT, "w") as f:
    json.dump({"fidelity": fid, "faithful": faithful, "attribution": attrib,
               "raw": r, "raw_se": se, "block": rb, "block_se": seb,
               "norms": norms, "norm_spread": spread,
               "met": bool(met), "falsified": bool(fals)}, f, indent=2)
print(f"\n  wrote {OUT}")
