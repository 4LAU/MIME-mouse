"""Does the gradient cosine read the shape of an objective or the size of its
residual? Registered before it ran.

WHAT FORCED THIS

The critic arm was built to be the trainable version of the per row lead, and it
failed its own falsifier. On identical batches:

    per row target        +0.0667 se 0.0233
    critic, per row       +0.0185 se 0.0147
    batch statistic       +0.0119 se 0.0075
    critic minus batch statistic, paired   +0.0066 se 0.0157   0.4 se

The critic gives credit one trajectory at a time, summed over rows, which is
exactly the shape the per row lead said was the thing that mattered. It carries
no more gradient than a batch statistic. So per row against per batch is not
what separates these objectives, and the mechanism written into the handoff
yesterday, that a batch statistic discards information before differentiating,
does not survive its own test.

THE ALTERNATIVE THAT FITS ALL FOUR NUMBERS

An objective's gradient is large and consistent when the model is far from
satisfying it, and small and noise dominated when the model is close. Read that
way:

  the per row target is unsatisfiable. Mean per row loss is 1.58 in units of
  feature variance and no model can ever match a specific trajectory it was not
  given. The residual is permanently large, so the gradient is permanently large.

  the moment objective is nearly satisfied. The generated mean is close to zero
  and the generated log standard deviation is close to zero in these units, so
  its gradient is a difference of large nearly cancelling terms divided by the
  batch size. What is left after the cancellation is mostly sampling noise.

  the critic separates at AUC 0.62, which is weak. Its logits sit near zero,
  where the non saturating loss has a nearly constant derivative and the
  direction it supplies is only as informative as a weak classifier's Jacobian.

If that is what is going on then the cosine is a measure of residual magnitude
and not of objective quality, a high cosine is not evidence that an objective is
trainable, and every cross objective comparison made in this workstream is
uninterpretable. That includes the session headline, REINFORCE against
supervised teacher forcing, which is also two objectives with very different
residuals.

THE TEST

Four arms, same 64 cached batches, same tokens, same pair indices, differences
taken pair by pair.

  A  moment objective at its true target, mean 0 and log sd 0. The incumbent.
  B  the same objective with the mean target moved to 1 standard deviation.
  C  the same objective with the mean target moved to 4 standard deviations.
  D  per row target against a randomly drawn human row rather than the row the
     generator was conditioned on.

A, B and C are the same shape and differ only in how far the model is from the
target. If the cosine is reading shape they stay together. If it is reading
residual they separate, and the arm nobody could train on is the one that looks
best.

D removes per row correspondence while keeping the residual large. If D matches
the per row target then correspondence contributes nothing and what looked like
per trajectory credit was a large pull toward the human feature distribution
wearing a per row costume.

PREDICTION: C minus A, paired, is at least 3 jackknife se above zero.

FALSIFIER: C minus A is within 2 se of zero. That would say residual magnitude
does not drive the cosine, the instrument is measuring something about the
objective rather than about the distance to it, and the critic result stands as
a genuine fact about critics rather than an artifact.

WHAT THIS CANNOT SETTLE. It cannot say which objective trains best, only
whether the statistic used to rank objectives all week ranks them by something
other than merit. A confirmation here is a retraction, not a lead.

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
OUT = "research/w4_residual.json"
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


paths = [f"{SCR}/softdec_tau_b{k}_{BATCH}_{CAP}.npz" for k in range(64)]
paths = [p for p in paths if os.path.exists(p)]
keepable = []
for p in paths:
    z = np.load(p)
    tgt = np.asarray(Hall[z["rows"][z["keep"]]], dtype=np.float64)
    if np.isfinite(tgt).all(1).sum() >= 16:
        keepable.append(p)
K = len(keepable)

# one fixed pool of random human rows, drawn once so every batch's D arm is
# scored against the same distribution and the arms stay comparable
pool = np.asarray(Hall[np.sort(rng.choice(ok, 40000, replace=False))],
                  dtype=np.float64)
pool = pool[np.isfinite(pool).all(1)]

print(f"\n  checkpoint {CKPT}")
print(f"  {K} batches of {BATCH} at tau {TAU:g}, {K * (K - 1) // 2} pairs")
print(f"  four arms on identical tokens, differences taken pair by pair\n",
      flush=True)

# how far the model actually is from each target, so the residual claim is a
# measured quantity rather than an assumption
diag = {"mean_abs_z": [], "log_sd": [], "per_row_own": [], "per_row_rand": []}


def grad_of(path, arm, ib):
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
    zt = zg[:, cols]
    if arm in ("A", "B", "C"):
        off = {"A": 0.0, "B": 1.0, "C": 4.0}[arm]
        loss = ((zt.mean(0) - off) ** 2).sum() + \
            (torch.log(zt.std(0).clamp(min=1e-4)) ** 2).sum()
        if arm == "A":
            diag["mean_abs_z"].append(float(zt.mean(0).abs().mean()))
            diag["log_sd"].append(
                float(torch.log(zt.std(0).clamp(min=1e-4)).abs().mean()))
    else:
        if arm == "own":
            tgt = np.asarray(Hall[rws[keep]], dtype=np.float64)
            fin = np.isfinite(tgt).all(1)
            tgt = tgt[fin]
            fi = torch.tensor(np.flatnonzero(fin), device=dev)
        else:
            # a different random human row per generated row, fixed per batch by
            # index so the arm is deterministic and reproducible
            g = np.random.default_rng(SEED + 1000 + ib)
            tgt = pool[g.choice(len(pool), zt.shape[0], replace=False)]
            fi = torch.arange(zt.shape[0], device=dev)
        ztg = ((torch.tensor(tgt, dtype=torch.float32, device=dev)
                - mu_t) / sd_t)[:, cols]
        r = zt[fi] - ztg
        loss = (r ** 2).sum()
        diag["per_row_own" if arm == "own" else "per_row_rand"].append(
            float((r ** 2).mean()))
    model.zero_grad(set_to_none=True)
    loss.backward()
    g = flat_grad()
    del Xs, zg, zt, s, th, dt, cond
    torch.cuda.empty_cache()
    return g


V = {}
ARMS = (("A", "moment, target 0"), ("B", "moment, target 1 sd"),
        ("C", "moment, target 4 sd"), ("own", "per row, own row"),
        ("rand", "per row, random row"))
for arm, oname in ARMS:
    G = []
    for k, p in enumerate(keepable):
        gate()
        G.append(blocknorm_(grad_of(p, arm, k)))
        if (k + 1) % 32 == 0:
            print(f"  {oname}  {k + 1} of {K}", flush=True)
    V[oname] = np.array([cos(a, b) for a, b in itertools.combinations(G, 2)])
    del G
    torch.cuda.empty_cache()
    m, j = float(V[oname].mean()), jack_se(V[oname], K)
    print(f"  {oname:<24} {m:+.4f}   jackknife se {j:.4f}   "
          f"{m / max(j, 1e-9):.1f} se from zero\n", flush=True)

print("  how far the model is from each target, measured")
print(f"    mean |z| per feature, generated      "
      f"{np.mean(diag['mean_abs_z']):.4f}")
print(f"    mean |log sd| per feature, generated "
      f"{np.mean(diag['log_sd']):.4f}")
print(f"    per row squared residual, own row    "
      f"{np.mean(diag['per_row_own']):.4f}")
print(f"    per row squared residual, random row "
      f"{np.mean(diag['per_row_rand']):.4f}\n")


def paired(a, b, label):
    d = V[a] - V[b]
    dm, dj = float(d.mean()), jack_se(d, K)
    print(f"  {label}")
    print(f"    {dm:+.4f}   jackknife se {dj:.4f}   {dm / max(dj, 1e-9):.1f} "
          f"se   2 se band [{dm - 2 * dj:+.4f}, {dm + 2 * dj:+.4f}]")
    return dm, dj


gm, gj = paired("moment, target 4 sd", "moment, target 0",
                "GATE  same shape, residual 4 sd against residual 0, paired")
b1 = paired("moment, target 1 sd", "moment, target 0",
            "secondary  residual 1 sd against residual 0, paired")
b2 = paired("per row, random row", "per row, own row",
            "secondary  random target against own row, paired")

met = gm > 3 * gj
fals = abs(gm) < 2 * gj
print(f"\n  PREDICTION {'MET' if met else 'NOT MET'}    "
      f"FALSIFIER {'TRIGGERED' if fals else 'not met'}")
if met:
    print("  the cosine reads how far the model is from the target, not the "
          "shape of the objective. every cross objective comparison in this "
          "workstream is uninterpretable and the per row lead is withdrawn.")
elif fals:
    print("  residual magnitude does not drive the cosine. the instrument is "
          "measuring the objective rather than the distance to it, and the "
          "critic result stands as a fact about critics.")

res = {"K": K, "tau": TAU,
       "arms": {k: {"mean": float(v.mean()), "jack": jack_se(v, K)}
                for k, v in V.items()},
       "diag": {k: float(np.mean(v)) for k, v in diag.items() if v},
       "gate_c_minus_a": {"mean": gm, "jack": gj},
       "b_minus_a": {"mean": b1[0], "jack": b1[1]},
       "rand_minus_own": {"mean": b2[0], "jack": b2[1]},
       "met": bool(met), "falsified": bool(fals), "peak_c": peak}
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(f"\n  peak {peak}C, wrote {OUT}")
