"""Does clipping the per sub batch gradient rescue the objective every arm
actually used? Registered before it ran.

WHAT CAME BEFORE, AND WHAT IS AND IS NOT ESTABLISHED

w4_softdec_tail asked why summing four disjoint sub batches barely improved the
gradient cosine, when errors that average would have taken it from +0.0818 to
+0.2628. Three aggregation rules over the same 64 sub batch gradients, grouped
into the same 16 groups of four:

    A  plain sum              +0.0904  jackknife se 0.0406
    B  unit normalised sum    +0.2026  jackknife se 0.0935
    C  norm clipped sum       +0.1910  jackknife se 0.0850

**What is established** is the mechanism, and it does not depend on any cosine.
The gradient norm across 64 sub batches runs from 15593 to 39875216, a spread
of 2557x, with the 99th percentile at 255 times the median. Four of the 64 sub
batches hold 58 percent of the total gradient mass. Within each group of four,
the largest member contributes on average 70 percent of the summed norm, where
four comparable members would each contribute 25. A sum of four such things is
not an average of four things, it is the largest one plus three rounding
errors, so the effective sample size of a mean stays near one however much
data is fed in.

**What is not established** is the size of the cosine recovery. B beats A by
+0.1123, but that was quoted against a standard error of 0.1020 formed by
combining the two arms' errors as if they were independent, which is 1.1 se and
worth nothing. The two rules are computed from identical gradients in identical
groups, and differ only in how each group weights its members, so the
comparison must be made pair by pair. That error is the same shape as the one
w4_cosse caught earlier today, and it is being fixed the same way, by
measuring rather than assuming.

WHY THE SECOND QUESTION MATTERS MORE

The per row objective was a diagnostic. It answered whether the relaxed decode
can transmit gradient at all, and it does: +0.0818 se 0.0280 at batch 96,
which is 0.6 se from ordinary supervised teacher forcing. But a per row target
is not a thing to train on. Hitting each real trajectory's own features is also
what a model collapsed onto the conditional mean would do, and the contract
scorer punishes exactly that through the dispersion ratios.

The objective every arm in this programme actually used is the batch statistic
one: match the mean and the spread of each feature across the batch. It reads
+0.0058 se 0.0190, indistinguishable from zero. If its gradient is heavy tailed
in the same way, clipping is a one line change to every arm already written. If
it is not, then its problem is the shape of the objective rather than the
arithmetic of the average, and no amount of clipping will help it.

A note on what the batch statistic groups mean. A batch mean and a batch
standard deviation do not decompose across sub batches, so summing four of
these gradients is not the gradient at batch 384. It is the average of four
batch 96 gradients, which is exactly what a training loop with four gradient
accumulation steps computes, so it is the right thing to measure even though
it is not the same estimator the per row arm used.

PREDICTION, two parts, both registered.
  i   For the per row objective, the paired difference B minus A is above
      +0.05 and at least 2.5 jackknife se clear of zero.
  ii  For the batch statistic objective, rule B reads above +0.05 at four
      accumulated sub batches, against +0.0058 for a single batch. That would
      say the plateau has a named cause with a one line fix, and every rollout
      arm should be rerun with per sub batch clipping before anything else is
      tried.

FALSIFIER, two parts.
  i   The paired difference is at or below +0.02. The tail explanation would
      then fail on its own evidence and the pair2 anomaly would be unexplained,
      whatever the norm distribution looks like.
  ii  The batch statistic objective reads at or below +0.02 under rule B. The
      tail fix would then not transfer to the objective it needs to transfer
      to, and the failure would be located in the shape of the objective rather
      than in the arithmetic, which is a harder problem and a different one.

The two parts can land differently and that is the point of running both.

WHAT THIS CANNOT SETTLE. A gradient that averages properly is not a gradient
that points somewhere useful. Everything here is measured at fixed weights on
one checkpoint, and says nothing about whether following the fixed gradient
improves the contract score. That needs a scored training run and is the next
thing, not this thing.

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
OUT = "research/w4_softdec_clip.json"
SEED = 17
BATCH = 96
CAP = 256
TAU = 4.0
GROUP = 4
WARM = 16          # sub batches held to set the clip threshold from their median
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


def grad_of(path, per_row):
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
    if per_row:
        tgt = np.asarray(Hall[rws[keep]], dtype=np.float64)
        fin = np.isfinite(tgt).all(1)
        if fin.sum() < 16:
            return None
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


paths = [f"{SCR}/softdec_tau_b{k}_{BATCH}_{CAP}.npz" for k in range(64)]
paths = [p for p in paths if os.path.exists(p)]
ngrp = len(paths) // GROUP
print(f"\n  checkpoint {CKPT}")
print(f"  {len(paths)} sub batches of {BATCH} at tau {TAU:g}, "
      f"{ngrp} groups of {GROUP}")
print(f"  two objectives, three aggregation rules, paired per group\n",
      flush=True)

res = {"n_sub": len(paths), "n_groups": ngrp, "tau": TAU, "objectives": {}}

for per_row, oname in ((True, "per row target"),
                       (False, "batch statistic")):
    print(f"  --- {oname} ---", flush=True)
    acc = {r: [torch.zeros(int(offs[-1]), dtype=torch.float32)
               for _ in range(ngrp)] for r in ("plain", "unit", "clip")}
    # the first WARM gradients are held so the clip threshold can be their
    # median, which is what a training loop does with a warm up sample
    held, norms, seen = [], [], 0
    for k, path in enumerate(paths):
        gate()
        g = grad_of(path, per_row)
        if g is None:
            continue
        j = seen // GROUP
        seen += 1
        if j >= ngrp:
            del g
            continue
        n = max(float(g.norm()), 1e-30)
        norms.append(n)
        acc["plain"][j] += g
        acc["unit"][j] += g / n
        if len(held) < WARM:
            held.append((j, g))
        else:
            if len(held) == WARM:
                med = float(np.median([x for x in norms[:WARM]]))
                for jj, gg in held:
                    acc["clip"][jj] += gg * min(
                        1.0, med / max(float(gg.norm()), 1e-30))
                held.append(None)          # mark as flushed
            acc["clip"][j] += g * min(1.0, med / n)
            del g
        if (k + 1) % 16 == 0:
            print(f"  {k + 1} of {len(paths)}", flush=True)
    del held

    q = np.percentile(norms, [0, 50, 99, 100])
    share = [max(norms[i * GROUP:(i + 1) * GROUP])
             / sum(norms[i * GROUP:(i + 1) * GROUP]) for i in range(ngrp)]
    print(f"    norms  min {q[0]:.4g}  median {q[1]:.4g}  p99 {q[2]:.4g}  "
          f"max {q[3]:.4g}   spread {q[3] / max(q[0], 1e-12):.0f}x")
    print(f"    largest member's share of its group  mean {np.mean(share):.3f}"
          f"   (0.25 if four contribute equally)", flush=True)

    V = {}
    for r in ("plain", "unit", "clip"):
        V[r] = paircos([blocknorm_(g) for g in acc[r]])
        acc[r] = None
    o = {"norm_median": float(q[1]), "norm_spread": float(q[3] / max(q[0], 1e-12)),
         "largest_share": float(np.mean(share)), "rules": {}, "clip_at": med}
    for r, label in (("plain", "A  plain sum"),
                     ("unit", "B  unit normalised sum"),
                     ("clip", "C  norm clipped sum")):
        m, j = float(V[r].mean()), jack_se(V[r], ngrp)
        print(f"    {label:<26} {m:+.4f}   jackknife se {j:.4f}   "
              f"{m / max(j, 1e-9):.1f} se from zero", flush=True)
        o["rules"][r] = {"mean": m, "jack": j}
    # paired, group by group, because the three rules share their gradients
    for r in ("unit", "clip"):
        d = V[r] - V["plain"]
        dj = jack_se(d, ngrp)
        o["rules"][r]["vs_plain"] = {"mean": float(d.mean()), "jack": dj}
        print(f"    {r} minus plain, PAIRED   {d.mean():+.4f}   "
              f"jackknife se {dj:.4f}   {d.mean() / max(dj, 1e-9):.1f} se",
              flush=True)
    res["objectives"][oname] = o
    print("", flush=True)

pr = res["objectives"]["per row target"]
bs = res["objectives"]["batch statistic"]
d = pr["rules"]["unit"]["vs_plain"]
p1_met = d["mean"] > 0.05 and d["mean"] > 2.5 * d["jack"]
p1_fal = d["mean"] <= 0.02
p2_met = bs["rules"]["unit"]["mean"] > 0.05
p2_fal = bs["rules"]["unit"]["mean"] <= 0.02

print(f"  i   per row, paired B minus A {d['mean']:+.4f} se {d['jack']:.4f}"
      f"    PREDICTION {'MET' if p1_met else 'NOT MET'}   "
      f"FALSIFIER {'TRIGGERED' if p1_fal else 'not met'}")
print(f"  ii  batch statistic under B  {bs['rules']['unit']['mean']:+.4f} "
      f"se {bs['rules']['unit']['jack']:.4f}"
      f"    PREDICTION {'MET' if p2_met else 'NOT MET'}   "
      f"FALSIFIER {'TRIGGERED' if p2_fal else 'not met'}")
print(f"\n  for reference, single batch 96, jackknife errors")
print(f"    per row target        +0.0818 se 0.0280")
print(f"    batch statistic       +0.0058 se 0.0190")
print(f"    supervised anchor     +0.0992 se 0.0139")
if p2_met:
    print("\n  the fix transfers. every rollout arm should be rerun with per "
          "sub batch clipping before anything else is tried.")
elif p2_fal:
    print("\n  the fix does not transfer. the batch statistic objective's "
          "problem is its shape, not the arithmetic of its average, and "
          "clipping will not rescue the arms already run.")

res.update({"p1_met": bool(p1_met), "p1_falsified": bool(p1_fal),
            "p2_met": bool(p2_met), "p2_falsified": bool(p2_fal),
            "peak_c": peak})
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(f"\n  peak {peak}C, wrote {OUT}")
