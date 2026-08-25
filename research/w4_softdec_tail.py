"""Why does averaging four sub batches barely improve the gradient cosine?
Registered before it ran.

WHAT CAME BEFORE

w4_softdec_pair2 took 64 sub batch gradients of the per row objective through
the relaxed decode and read them two ways:

    batch 96,  64 gradients   +0.0818  jackknife se 0.0280   2.9 se from zero
    batch 384, 16 gradients   +0.0944  jackknife se 0.0411   2.3 se from zero

The first of those settles the question the arm was built to answer. Against
the other cells, with jackknife errors throughout:

    supervised anchor NLL, per row       +0.0992 se 0.0139
    per row target through the decode    +0.0818 se 0.0280   0.6 se from it
    moment objective, batch statistic    +0.0058 se 0.0190   2.2 se below it

A per row target through the relaxed decode is as well determined as ordinary
supervised teacher forcing, so **the relaxation transmits gradient and is not
the fault**. The batch statistic objective is, and every arm in this programme
has used one.

Its registered prediction and falsifier both missed, because both were written
about the batch 384 reading, and that reading is the thing this run is about.

THE ANOMALY

The 16 groups are sums of 4 disjoint sub batches. With errors that average, a
sub batch gradient is mu + e with e independent between sub batches, so a
group is 4mu + sum of four e, and the cosine should go from
|mu|^2 / (|mu|^2 + sigma^2) to 4|mu|^2 / (4|mu|^2 + sigma^2). Reading the
first as 0.0818 fixes |mu|^2 / sigma^2 at 0.089 and predicts 0.263 at four
times the batch. The measurement returned 0.094.

Averaging four times as much data bought almost nothing. That is not what
sampling noise does, and it is a different failure from having no mean
direction at all, with a different remedy.

THE SUSPECT, already recorded and not followed up

Every run in this arm reported a gradient norm spread across batches of two to
three orders of magnitude: 699x, 1543x, 2362x, 2504x, and 728 to 2.3e12 in the
first construction before the feature fixes. If the norms are that heavy
tailed, a sum of four sub batches is not an average of four things. It is
whichever member happens to be largest, plus three rounding errors. The
effective sample size of a mean stays near one however much data is fed in,
which would explain both the missing improvement and why every rollout arm
plateaued no matter how many steps it took.

THE RUN. The same 64 sub batch gradients, aggregated into the same 16 groups
three ways, so the only thing that varies is how a group combines its members.

  A  plain sum, which is what pair2 did and what an optimiser does
  B  unit normalised sum, each sub batch scaled to unit norm first, so every
     member contributes equally regardless of its magnitude
  C  norm clipped sum, each sub batch scaled down to the median norm if it
     exceeds it, which is ordinary gradient clipping applied per sub batch

Rule B is the diagnostic, because it removes magnitude entirely. Rule C is the
practical version, because it is what a training loop can actually do.

PREDICTION: rule B reads above +0.20 at an effective batch of 384, against
+0.0944 for rule A. That would confirm heavy tails as the reason averaging
fails, and would mean the plateau has a named cause with a known fix rather
than an unknown one.

FALSIFIER: rule B reads at or below +0.12, that is no better than rule A
within error. That would say the failure to average is not a magnitude effect,
that the sub batch errors are genuinely correlated with each other, and that
the direction to doubt next is the objective's dependence on the corpus
sample rather than the arithmetic of the average.

DIAGNOSTIC, NOT A GATE. The report also gives the norm distribution and the
share of each group's summed norm contributed by its largest member. Under
errors that average, four comparable members contribute about a quarter each.
A largest share near one is the tail hypothesis stated directly, without going
through any cosine.

WHAT THIS CANNOT SETTLE. Making the gradient average properly is not the same
as making it point somewhere useful. A well determined direction for a per row
target is still a direction toward each trajectory's own conditional mean, and
the contract scorer punishes exactly that collapse through the dispersion
ratios. That trade is a scored training run, and it is the next thing, not
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
from w4_rollout import TRAINED, gpu_temp  # noqa: E402
from w4_softdec import soft_forward, straight_through  # noqa: E402

D = "training"
CKPT = "research/w4_rollout_pilot_zbuf_step100.pt"
OUT = "research/w4_softdec_tail.json"
SEED = 17
BATCH = 96
CAP = 256
TAU = 4.0
GROUP = 4
K = 64
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


def paircos(G):
    return np.array([cos(a, b) for a, b in itertools.combinations(G, 2)])


def jack_se(v, n):
    idx = list(itertools.combinations(range(n), 2))
    th = np.array([v[[j for j, (a, b) in enumerate(idx)
                      if a != i and b != i]].mean() for i in range(n)])
    return float(np.sqrt((n - 1) / n * ((th - th.mean()) ** 2).sum()))


paths = [f"{SCR}/softdec_tau_b{k}_{BATCH}_{CAP}.npz" for k in range(K)]
paths = [p for p in paths if os.path.exists(p)]
if len(paths) < K:
    raise SystemExit(f"only {len(paths)} of {K} cached batches, "
                     "run w4_softdec_pair2.py first")
ngrp = len(paths) // GROUP

print(f"\n  checkpoint {CKPT}")
print(f"  {len(paths)} sub batches of {BATCH} at tau {TAU:g}, "
      f"{ngrp} groups of {GROUP}")
print(f"  three aggregation rules, only the arithmetic of the sum varies\n",
      flush=True)

# Gradients are accumulated straight into their group rather than stored, so
# three rules cost 48 vectors held rather than 64 plus 48. The per sub batch
# norms are kept as scalars for the tail report.
def grad_of(path):
    """One sub batch gradient, or None if too few rows survive."""
    z = np.load(path)
    rws, keep = z["rows"], z["keep"]
    tgt = np.asarray(Hall[rws[keep]], dtype=np.float64)
    fin = np.isfinite(tgt).all(1)
    if fin.sum() < 16:
        return None
    s = torch.from_numpy(z["s"]).to(dev)
    th = torch.from_numpy(z["th"]).to(dev)
    dt = torch.from_numpy(z["dt"]).to(dev)
    cond = torch.tensor(np.asarray(cond_all[rws], dtype=np.float32)).to(dev)
    angt = torch.tensor(z["ang"], dtype=torch.float32, device=dev)
    Xht = torch.tensor(z["Xh"], dtype=torch.float32, device=dev)
    ki = torch.tensor(keep, device=dev)
    fi = torch.tensor(np.flatnonzero(fin), device=dev)
    ztg = ((torch.tensor(tgt[fin], dtype=torch.float32, device=dev)
            - mu_t) / sd_t)[:, cols]
    Xs, _ = soft_forward(model, s, th, dt, cond, angt, tau=TAU)
    zg = ((straight_through(Xht, Xs[ki]) - mu_t) / sd_t)[fi][:, cols]
    loss = ((zg - ztg) ** 2).sum()
    model.zero_grad(set_to_none=True)
    loss.backward()
    g = flat_grad()
    del Xs, zg, s, th, dt, cond
    torch.cuda.empty_cache()
    return g


# Two passes. Rule C clips to the median norm, which is not known until every
# gradient has been taken, and holding all 64 raw gradients alongside the 48
# accumulators would need about 10 GB against a 14 GB box. Recomputing costs
# about seven minutes and no memory.
print("  pass one, norms only\n", flush=True)
norms, member = [], [[] for _ in range(ngrp)]
for k, path in enumerate(paths):
    gate()
    g = grad_of(path)
    if g is None:
        continue
    n = float(g.norm())
    j = len(norms) // GROUP
    if j < ngrp:
        member[j].append(n)
    norms.append(n)
    del g
    if (k + 1) % 16 == 0:
        print(f"  {k + 1} of {len(paths)}", flush=True)

med = float(np.median(norms))
print(f"\n  median norm {med:.1f}, pass two, accumulating three rules\n",
      flush=True)

acc = {r: [torch.zeros(int(offs[-1]), dtype=torch.float32)
           for _ in range(ngrp)] for r in ("plain", "unit", "clip")}
seen = 0
for k, path in enumerate(paths):
    gate()
    g = grad_of(path)
    if g is None:
        continue
    j = seen // GROUP
    seen += 1
    if j >= ngrp:
        del g
        continue
    n = max(float(g.norm()), 1e-30)
    acc["plain"][j] += g
    acc["unit"][j] += g / n
    acc["clip"][j] += g * min(1.0, med / n)
    del g
    if (k + 1) % 16 == 0:
        print(f"  {k + 1} of {len(paths)}", flush=True)

q = np.percentile(norms, [0, 25, 50, 75, 100])
print(f"\n  gradient norm across {len(norms)} sub batches")
print(f"    min {q[0]:.1f}   p25 {q[1]:.1f}   median {q[2]:.1f}   "
      f"p75 {q[3]:.1f}   max {q[4]:.1f}   spread {q[4] / max(q[0], 1e-9):.0f}x")
share = [max(m) / sum(m) for m in member if m]
print(f"    largest member's share of its group's summed norm: "
      f"mean {np.mean(share):.3f}, max {np.max(share):.3f} "
      f"(0.25 if four members contribute equally)")

res = {"norms": norms, "median_norm": med,
       "largest_share_mean": float(np.mean(share)),
       "largest_share_max": float(np.max(share)), "n_sub": len(norms),
       "n_groups": ngrp, "tau": TAU, "rules": {}}

print(f"\n  effective batch {BATCH * GROUP}, {ngrp} gradients per rule\n")
for r, label in (("plain", "A  plain sum"),
                 ("unit", "B  unit normalised sum"),
                 ("clip", "C  norm clipped sum")):
    v = paircos([blocknorm_(g) for g in acc[r]])
    m, j = float(v.mean()), jack_se(v, ngrp)
    print(f"  {label:<26} {m:+.4f}   jackknife se {j:.4f}   "
          f"{m / max(j, 1e-9):.1f} se from zero", flush=True)
    res["rules"][r] = {"mean": m, "jack": j}
    acc[r] = None

b = res["rules"]["unit"]["mean"]
met = b > 0.20
fals = b <= 0.12
print(f"\n  for reference, same objective and checkpoint, jackknife errors")
print(f"    batch 96, 64 gradients                   +0.0818 se 0.0280")
print(f"    batch 384 plain sum, from pair2          +0.0944 se 0.0411")
print(f"    what averaging errors would have given   +0.2628")
print(f"\n  PREDICTION {'MET' if met else 'NOT MET'}    "
      f"FALSIFIER {'TRIGGERED' if fals else 'not met'}")
if met:
    print("  heavy tails are why averaging fails. the plateau has a named "
          "cause and clipping or normalising per sub batch is the fix to try.")
elif fals:
    print("  magnitude is not the problem, so the sub batch errors are "
          "genuinely correlated. doubt the objective's dependence on the "
          "corpus sample next, not the arithmetic.")
else:
    print("  between the thresholds. do not read a direction into it.")

res.update({"met": bool(met), "falsified": bool(fals), "peak_c": peak})
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(f"\n  peak {peak}C, wrote {OUT}")
