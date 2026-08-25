"""Is the objective shape effect real, measured paired on identical batches?
Registered before it ran.

THE CLAIM THIS TESTS, AND WHY IT NEEDS TESTING

The redirect coming out of today's work is that the fault is the shape of the
objective rather than the estimator, the relaxation or the model class. It
rests on comparing two numbers:

    per row target through the relaxed decode   +0.0818 se 0.0280   64 gradients
    moment objective, batch statistic           +0.0058 se 0.0190   24 gradients

Those were measured on different numbers of gradients, from overlapping but
not identical batch sets, in different runs. The difference is 2.2 se by
combining errors that were never meant to be combined. The whole redirect
rests on it, so it gets measured properly: both objectives, the same 64 cached
batches, the same tokens, the same pair indices, and the difference taken pair
by pair so that the batch draw noise the two share cancels instead of adding.

WHAT ELSE THIS RUN SETTLES

w4_softdec_clip established that gradient magnitudes are extremely heavy
tailed on both objectives, 2557x spread on the per row objective and 23242x on
the batch statistic one, with the largest member of each group of four holding
about 0.68 of the summed norm where equal members would hold 0.25. That much is
a direct measurement of a distribution and does not depend on any cosine.

What it did not establish is that fixing the tails fixes anything. Paired,
unit normalising the sub batches before summing gives +0.1123 se 0.0911 on the
per row objective, which is 1.2 se and proves nothing, and on the batch
statistic objective it gives -0.0190 se 0.0210, with clipping at -0.0010 se
0.0272. **On the objective that needs rescuing, an improvement above +0.055 is
excluded at two standard errors.** So clipping is not the fix, and rerunning
the existing arms with it would be wasted GPU time.

An earlier version of that conclusion was stated too strongly, on a gate that
asked whether a point estimate exceeded +0.20 without asking whether it was
distinguishable from the thing it was being compared to. That is the same
mistake as registering an absolute threshold on an unfamiliar statistic, made
twice in one day, and it is the reason this run's gate is written as a paired
difference in standard errors rather than as a level.

THE MECHANISM THE SHAPE EFFECT WOULD HAVE, if it is real

A batch statistic objective condenses ninety odd trajectories into eighteen
numbers, nine means and nine standard deviations, before any gradient is taken.
The per row objective keeps nine residuals per trajectory, about eight hundred
numbers from the same batch. If the shape effect is real it is not mysterious:
the objectives this programme has been using throw away almost all of the
information in a batch before differentiating, and the gradient noise is what
that discarding looks like from the outside.

PREDICTION: the paired difference, per row minus batch statistic, is above
+0.05 and at least 3 jackknife se clear of zero.

FALSIFIER: at or below +0.02, or within 2 se of zero. Either would say the
shape effect is not established, that the gap between +0.0818 and +0.0058 came
from comparing runs at different gradient counts rather than from the
objectives, and that today's redirect has no measurement under it.

WHAT THIS CANNOT SETTLE. Gradient agreement at fixed weights is not score. A
per row target is a diagnostic and not a training objective, because hitting
each real trajectory's own features is also what a model collapsed onto the
conditional mean would do, and the contract scorer punishes that through the
dispersion ratios. What this can do is say whether the shape of an objective
changes how much gradient information a batch yields, which decides what the
next objective should look like rather than what it should be.

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
from w4_rollout import TRAINED, gpu_temp  # noqa: E402
from w4_softdec import soft_forward, straight_through  # noqa: E402

D = "training"
CKPT = "research/w4_rollout_pilot_zbuf_step100.pt"
OUT = "research/w4_shape.json"
SEED = 17
BATCH = 96
CAP = 256
TAU = 4.0
DROP = ("mean_acceleration", "mean_jerk", "curvature_mean")
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


def grad_of(path, per_row):
    z = np.load(path)
    rws, keep = z["rows"], z["keep"]
    if per_row:
        tgt = np.asarray(Hall[rws[keep]], dtype=np.float64)
        if np.isfinite(tgt).all(1).sum() < 16:
            return None
    s = torch.from_numpy(z["s"]).to(dev)
    th = torch.from_numpy(z["th"]).to(dev)
    dt = torch.from_numpy(z["dt"]).to(dev)
    cond = torch.tensor(np.asarray(cond_all[rws], dtype=np.float32)).to(dev)
    angt = torch.tensor(z["ang"], dtype=torch.float32, device=dev)
    Xht = torch.tensor(z["Xh"], dtype=torch.float32, device=dev)
    ki = torch.tensor(keep, device=dev)
    Xs, _ = soft_forward(model, s, th, dt, cond, angt, tau=TAU)
    zg = ((straight_through(Xht, Xs[ki]) - mu_t) / sd_t)
    if per_row:
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


# Only batches where both objectives produce a gradient are used, so the two
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
print(f"\n  checkpoint {CKPT}")
print(f"  {K} batches of {BATCH} at tau {TAU:g}, {K * (K - 1) // 2} pairs")
print(f"  both objectives on identical tokens, difference taken pair by "
      f"pair\n", flush=True)

V = {}
for per_row, oname in ((True, "per row target"), (False, "batch statistic")):
    G = []
    for k, p in enumerate(keepable):
        gate()
        g = grad_of(p, per_row)
        G.append(blocknorm_(g))
        if (k + 1) % 16 == 0:
            print(f"  {oname}  {k + 1} of {K}", flush=True)
    V[oname] = np.array([cos(a, b) for a, b in itertools.combinations(G, 2)])
    del G
    torch.cuda.empty_cache()
    m, j = float(V[oname].mean()), jack_se(V[oname], K)
    print(f"  {oname:<20} {m:+.4f}   jackknife se {j:.4f}   "
          f"{m / max(j, 1e-9):.1f} se from zero\n", flush=True)

d = V["per row target"] - V["batch statistic"]
dm, dj = float(d.mean()), jack_se(d, K)
met = dm > 0.05 and dm > 3 * dj
fals = dm <= 0.02 or abs(dm) < 2 * dj
print(f"  paired difference, per row minus batch statistic")
print(f"    {dm:+.4f}   jackknife se {dj:.4f}   {dm / max(dj, 1e-9):.1f} se")
print(f"    2 se band [{dm - 2 * dj:+.4f}, {dm + 2 * dj:+.4f}]")
print(f"\n  PREDICTION {'MET' if met else 'NOT MET'}    "
      f"FALSIFIER {'TRIGGERED' if fals else 'not met'}")
if met:
    print("  the shape of the objective decides how much gradient a batch "
          "yields. a batch statistic condenses ninety trajectories into "
          "eighteen numbers before differentiating; a per row target keeps "
          "about eight hundred.")
elif fals:
    print("  the shape effect is not established. today's redirect has no "
          "measurement under it and must not be carried forward as fact.")

res = {"K": K, "tau": TAU,
       "per_row": {"mean": float(V["per row target"].mean()),
                   "jack": jack_se(V["per row target"], K)},
       "batch_stat": {"mean": float(V["batch statistic"].mean()),
                      "jack": jack_se(V["batch statistic"], K)},
       "paired_diff": {"mean": dm, "jack": dj},
       "met": bool(met), "falsified": bool(fals), "peak_c": peak}
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(f"\n  peak {peak}C, wrote {OUT}")
