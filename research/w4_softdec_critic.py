"""Does a differentiable critic carry gradient the way a per row target does?
Registered before it ran.

WHY THIS ARM EXISTS

Three explanations for the plateau are now ruled out. The relaxed decode
transmits gradient: a per row target pushed through it reads +0.0667 against
+0.0992 for ordinary supervised teacher forcing on the same weights, which is
0.6 se apart. So the model class, the pathwise estimator and the relaxation are
all fine. What is left is the shape of the objective, and `w4_shape` measured
that paired on identical batches:

    per row target        +0.0667 se 0.0233
    batch statistic       +0.0119 se 0.0075
    paired difference     +0.0548 se 0.0261     2.1 se, supported not established

A batch statistic condenses ninety odd trajectories into eighteen numbers, nine
means and nine standard deviations, before any gradient is taken. A per row
target keeps about eight hundred. Every arm in this programme has used a batch
statistic.

WHAT A PER ROW TARGET CANNOT BE

It cannot be trained on. Hitting each real trajectory's own eighteen features
is exactly what a model collapsed onto the conditional mean achieves, and the
contract scorer punishes that through the dispersion ratios. The lead is about
the shape of the credit, per trajectory rather than per batch, not about that
particular target.

A critic has that shape and does not have that loophole. It scores one
trajectory at a time, so credit is per row, and it is trained to separate
generated rows from human ones, so a model that collapsed onto the conditional
mean would be trivially caught by it. It is also the closest differentiable
relative of the contract scorer, which is itself a classifier two sample test on
these same features.

THE CONSTRUCTION

A small MLP on the nine trained features, standardised, is fit to separate the
current checkpoint's own generated rows from human rows, then frozen. The
generator objective is the usual non saturating form, summed over rows so that
credit is exactly per row:

    loss = sum_i softplus(-critic(z_i))

with z_i reaching the critic through the straight through join, so forward
values are the ones the served decoder produces and the Jacobian is the relaxed
decode's.

The critic is frozen for this measurement and that is deliberate. Gradient
agreement across batches only means anything if every batch is differentiating
the same function. A critic updated between batches would make the comparison
meaningless.

WHAT IS CONTAMINATED, STATED PLAINLY

The critic is fit on generated rows pooled from the same 64 cached batches whose
gradients are then measured, with a held out row split used only to read its
AUC. The alternative, fitting on half the batches and measuring on the other
half, costs half the gradients and about 40 percent more error on every number
below, which is worse than the contamination. The contamination is conservative
for what is being measured here: a critic that had memorised individual rows
would produce row specific gradients that cancel within a batch and lower the
cosine. What raises the cosine is the critic having found structure that
generated rows share, and that is the thing being tested.

PREDICTION: the paired difference, critic minus batch statistic, is at least 3
jackknife se above zero.

FALSIFIER: that difference is within 2 se of zero, or negative. Either would
say that per row credit through a critic does not recover what per row credit
through a direct target did, that the shape lead does not survive contact with
an objective that can actually be trained on, and that the next arm has to come
from somewhere else.

SECONDARY, NOT GATED: critic minus per row target, which says whether the
critic reaches the diagnostic ceiling or only part way to it. And the critic's
own held out AUC, which is a fact about how separable this checkpoint is on the
nine trained features and is worth having on its own.

WHAT THIS CANNOT SETTLE. Gradient agreement at fixed weights is not score. A
critic that yields a clean gradient direction can still be pointing somewhere
useless, and only training and then scoring can say. What this decides is
whether the next training arm is worth the GPU time.

No model file is overwritten and the protected eval sample is never read. The
critic is written to research/w4_softdec_critic.pt for the training arm to use.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("EVENT_SNAP", "2.5")
for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.event_ar import EventARModel  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402
from w4_rollout import TRAINED, gpu_temp  # noqa: E402
from w4_softdec import soft_forward, straight_through  # noqa: E402

D = "training"
CKPT = "research/w4_rollout_pilot_zbuf_step100.pt"
OUT = "research/w4_softdec_critic.json"
CRITIC_OUT = "research/w4_softdec_critic.pt"
SEED = 17
BATCH = 96
CAP = 256
TAU = 4.0
DROP = ("mean_acceleration", "mean_jerk", "curvature_mean")
HID = 64
CRIT_STEPS = 4000
CRIT_LR = 1e-3
CRIT_WD = 1e-4
KILL_C = 79
COOL_C = 74
RESUME_C = 70
SCR = os.environ.get("W4_CACHE", "/tmp/w4_cache")

dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

ok = np.flatnonzero(np.load(f"{D}/events_feat18_ok.npy"))
cond_all = np.load(f"{D}/events_cond.npy", mmap_mode="r")
Hall = np.load(f"{D}/events_feat18.npy", mmap_mode="r")
C = np.asarray(Hall[np.sort(rng.choice(ok, 20000, replace=False))],
               dtype=np.float64)
C = C[np.isfinite(C).all(1)]
mu, sd = C.mean(0), C.std(0)
sd[sd == 0] = 1.0
mu_t = torch.tensor(mu, dtype=torch.float32, device=dev)
sd_t = torch.tensor(sd, dtype=torch.float32, device=dev)
cols = [FEATURE_NAMES.index(n) for n in TRAINED if n not in DROP]
NF = len(cols)

ck = torch.load(CKPT, map_location=dev, weights_only=False)
model = EventARModel(**ck["config"]).to(dev)
model.load_state_dict(ck["model_state_dict"])
model.eval()
params = [p for p in model.parameters() if p.requires_grad]
sizes = [p.numel() for p in params]
offs = np.cumsum([0] + sizes)
peak = 0


def gate():
    global peak
    t = gpu_temp()
    peak = max(peak, t)
    if t >= COOL_C:
        import time
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
                   for p in params]).float().cpu()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return g


def blocknorm_(g):
    for i in range(len(sizes)):
        sl = slice(int(offs[i]), int(offs[i + 1]))
        n = g[sl].norm()
        if n > 0:
            g[sl] /= n
    return g


def cos(a, b):
    a, b = a.double(), b.double()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))


def jack_se(v, n):
    idx = list(itertools.combinations(range(n), 2))
    th = np.array([v[[j for j, (a, b) in enumerate(idx)
                      if a != i and b != i]].mean() for i in range(n)])
    return float(np.sqrt((n - 1) / n * ((th - th.mean()) ** 2).sum()))


# ---------------------------------------------------------------- the batches
# Only batches where every objective produces a gradient are used, so the three
# arms see byte identical tokens and the pair indices line up exactly.
paths = [f"{SCR}/softdec_tau_b{k}_{BATCH}_{CAP}.npz" for k in range(64)]
paths = [p for p in paths if os.path.exists(p)]
keepable = []
for p in paths:
    z = np.load(p)
    tgt = np.asarray(Hall[z["rows"][z["keep"]]], dtype=np.float64)
    if np.isfinite(tgt).all(1).sum() >= 16:
        keepable.append(p)
K = len(keepable)

# ------------------------------------------------------------- fit the critic
gen, condrows = [], []
for p in keepable:
    z = np.load(p)
    gen.append(np.asarray(z["Xh"], dtype=np.float64))
    condrows.append(z["rows"][z["keep"]])
gen = np.concatenate(gen, 0)
gen = gen[np.isfinite(gen).all(1)]
used = np.unique(np.concatenate(condrows))
# human rows disjoint from everything the generator was conditioned on, so the
# critic is never asked to separate a trajectory from its own conditioning
pool = np.setdiff1d(ok, used)
hum = np.asarray(Hall[np.sort(rng.choice(pool, len(gen), replace=False))],
                 dtype=np.float64)
hum = hum[np.isfinite(hum).all(1)]

Zg = ((gen - mu) / sd)[:, cols]
Zh = ((hum - mu) / sd)[:, cols]
X = np.concatenate([Zg, Zh], 0)
y = np.concatenate([np.zeros(len(Zg)), np.ones(len(Zh))])
perm = rng.permutation(len(X))
X, y = X[perm], y[perm]
ntr = int(0.8 * len(X))
Xtr = torch.tensor(X[:ntr], dtype=torch.float32, device=dev)
ytr = torch.tensor(y[:ntr], dtype=torch.float32, device=dev)
Xva = torch.tensor(X[ntr:], dtype=torch.float32, device=dev)
yva = y[ntr:]

critic = nn.Sequential(nn.Linear(NF, HID), nn.ReLU(),
                       nn.Linear(HID, HID), nn.ReLU(),
                       nn.Linear(HID, 1)).to(dev)
opt = torch.optim.Adam(critic.parameters(), lr=CRIT_LR, weight_decay=CRIT_WD)
bce = nn.BCEWithLogitsLoss()
best_auc, best_state = 0.0, None
for step in range(CRIT_STEPS):
    i = torch.randint(0, len(Xtr), (512,), device=dev)
    opt.zero_grad(set_to_none=True)
    bce(critic(Xtr[i]).squeeze(1), ytr[i]).backward()
    opt.step()
    if (step + 1) % 250 == 0:
        with torch.no_grad():
            sc = critic(Xva).squeeze(1).cpu().numpy()
        o = np.argsort(sc)
        r = np.empty(len(sc))
        r[o] = np.arange(1, len(sc) + 1)
        npos, nneg = yva.sum(), (1 - yva).sum()
        auc = (r[yva == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
        if auc > best_auc:
            best_auc = float(auc)
            best_state = {k: v.detach().clone()
                          for k, v in critic.state_dict().items()}
critic.load_state_dict(best_state)
for q in critic.parameters():
    q.requires_grad_(False)
critic.eval()

print(f"\n  checkpoint {CKPT}")
print(f"  critic on {NF} trained features, {len(Zg)} generated rows against "
      f"{len(Zh)} human")
print(f"  held out AUC {best_auc:.4f}   (0.5 is nothing to learn, 1.0 is "
      f"saturated)")
print(f"  {K} batches of {BATCH} at tau {TAU:g}, {K * (K - 1) // 2} pairs")
print(f"  three objectives on identical tokens, differences taken pair by "
      f"pair\n", flush=True)


def grad_of(path, which):
    z = np.load(path)
    rws, keep = z["rows"], z["keep"]
    s = torch.from_numpy(z["s"]).to(dev)
    th = torch.from_numpy(z["th"]).to(dev)
    dt = torch.from_numpy(z["dt"]).to(dev)
    cond = torch.tensor(np.asarray(cond_all[rws], dtype=np.float32)).to(dev)
    angt = torch.tensor(z["ang"], dtype=torch.float32, device=dev)
    Xht = torch.tensor(z["Xh"], dtype=torch.float32, device=dev)
    ki = torch.tensor(keep, device=dev)
    Xs, _ = soft_forward(model, s, th, dt, cond, angt, tau=TAU)
    zg = ((straight_through(Xht, Xs[ki]) - mu_t) / sd_t)
    if which == "critic":
        # a sum over rows, so credit is exactly per trajectory
        loss = torch.nn.functional.softplus(
            -critic(zg[:, cols]).squeeze(1)).sum()
    elif which == "per_row":
        tgt = np.asarray(Hall[rws[keep]], dtype=np.float64)
        fin = np.isfinite(tgt).all(1)
        fi = torch.tensor(np.flatnonzero(fin), device=dev)
        ztg = ((torch.tensor(tgt[fin], dtype=torch.float32, device=dev)
                - mu_t) / sd_t)[:, cols]
        loss = ((zg[fi][:, cols] - ztg) ** 2).sum()
    else:
        zt = zg[:, cols]
        loss = ((zt.mean(0)) ** 2).sum() + \
            (torch.log(zt.std(0).clamp(min=1e-4)) ** 2).sum()
    model.zero_grad(set_to_none=True)
    loss.backward()
    g = flat_grad()
    del Xs, zg, s, th, dt, cond
    torch.cuda.empty_cache()
    return g


V = {}
for which, oname in (("critic", "critic, per row"),
                     ("per_row", "per row target"),
                     ("batch", "batch statistic")):
    G = []
    for k, p in enumerate(keepable):
        gate()
        G.append(blocknorm_(grad_of(p, which)))
        if (k + 1) % 16 == 0:
            print(f"  {oname}  {k + 1} of {K}", flush=True)
    V[oname] = np.array([cos(a, b) for a, b in itertools.combinations(G, 2)])
    del G
    torch.cuda.empty_cache()
    m, j = float(V[oname].mean()), jack_se(V[oname], K)
    print(f"  {oname:<20} {m:+.4f}   jackknife se {j:.4f}   "
          f"{m / max(j, 1e-9):.1f} se from zero\n", flush=True)


def paired(a, b, label):
    d = V[a] - V[b]
    dm, dj = float(d.mean()), jack_se(d, K)
    print(f"  {label}")
    print(f"    {dm:+.4f}   jackknife se {dj:.4f}   {dm / max(dj, 1e-9):.1f} "
          f"se   2 se band [{dm - 2 * dj:+.4f}, {dm + 2 * dj:+.4f}]")
    return dm, dj


gm, gj = paired("critic, per row", "batch statistic",
                "GATE  critic minus batch statistic, paired")
cm, cj = paired("critic, per row", "per row target",
                "secondary  critic minus per row target, paired")

met = gm > 3 * gj
fals = gm < 0 or abs(gm) < 2 * gj
print(f"\n  PREDICTION {'MET' if met else 'NOT MET'}    "
      f"FALSIFIER {'TRIGGERED' if fals else 'not met'}")
if met:
    print("  per row credit through a critic recovers what per row credit "
          "through a direct target did, on an objective that a collapse onto "
          "the conditional mean cannot satisfy. the training arm is worth "
          "running.")
elif fals:
    print("  per row credit through a critic does not carry what a direct per "
          "row target carries. the shape lead does not survive contact with a "
          "trainable objective and the next arm has to come from elsewhere.")

res = {"K": K, "tau": TAU, "critic_auc": best_auc,
       "n_gen": int(len(Zg)), "n_hum": int(len(Zh)),
       "arms": {k: {"mean": float(v.mean()), "jack": jack_se(v, K)}
                for k, v in V.items()},
       "gate_critic_minus_batch": {"mean": gm, "jack": gj},
       "critic_minus_per_row": {"mean": cm, "jack": cj},
       "met": bool(met), "falsified": bool(fals), "peak_c": peak}
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
torch.save({"state_dict": critic.state_dict(), "cols": cols, "hid": HID,
            "mu": mu, "sd": sd, "auc": best_auc, "ckpt": CKPT}, CRITIC_OUT)
print(f"\n  peak {peak}C, wrote {OUT} and {CRITIC_OUT}")
