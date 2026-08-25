"""Does the pathwise gradient carry signal at any relaxation temperature?
Registered before it ran.

WHY A THIRD RUN, AND WHAT THE FIRST TWO LEFT OPEN

    construction                              block normalised pairwise cosine
    REINFORCE surrogate, for reference             -0.0071 se 0.0125
    supervised anchor NLL, for reference           +0.0942 se 0.0192
    pathwise, all twelve trained features          -0.0088 se 0.0466
    pathwise, nine faithful features               +0.0729 se 0.0503
    pathwise, nine, with the angle floor           +0.0232 se 0.0512

The last three are all one construction at eight batches, and their standard
error is 0.05. They cannot be told apart from each other or from zero, so the
verdict so far is not "the pathwise gradient fails", it is "eight batches cannot
say". Two things have to change at once for the answer to mean anything.

  PRECISION. Twenty four batches instead of eight gives 276 pairs instead of 28
  and a standard error near 0.017, which can separate zero from the supervised
  control's 0.094.

  THE ONE UNTESTED KNOB. tau, the softmax temperature of the straight through
  relaxation, has been fixed at 1.0 without justification. It is the parameter
  the construction turns on. A trained model puts almost all its probability on
  one class per position, so at tau 1 the backward signal comes almost entirely
  from the few positions where two classes are nearly tied, which is exactly the
  condition that produces a high variance gradient. Raising tau spreads the
  relaxed distribution and trades bias for variance. Declaring the arm dead
  without testing it would be a verdict on one arbitrary setting.

The sampled tokens do not depend on tau, so all three settings reuse the same
twenty four draws. The extra cost over a single setting is three backward passes
instead of one per batch, and the three arms are paired on identical data, which
removes batch composition from the comparison between them entirely.

PREDICTION: at least one tau reads above +0.05, and the cosine increases with
tau across the three settings.

FALSIFIER: all three at or below +0.03 with a standard error near 0.017. That
would say a pathwise gradient through this decode carries no more batch to batch
signal than the score function estimator it was built to replace, at any
temperature, and the straight through arm should stop. The remaining
explanations for the plateau would then be the relaxation's bias, which no
amount of variance reduction fixes, or the model class.

WHAT IT STILL CANNOT SETTLE. Agreement is not correctness. The relaxation drops
the snap, the rounding and the tick merge, and w4_softdec_check measured the
resulting bias at the features: nine of the trained twelve track the served
decoder above 0.9 and three do not and are excluded here. A gradient can be
consistent and consistently wrong, and only a scored training run separates
those.

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
OUT = "research/w4_softdec_tau.json"
SEED = 17
BATCH = 96
K = 24
CAP = 256
TAUS = (1.0, 2.0, 4.0)
# the nine trained features whose relaxed value tracks the served one above 0.9,
# measured by w4_softdec_check and reconfirmed by w4_softdec_check2. the three
# excluded are mean_acceleration (-0.905), mean_jerk (-0.557) and curvature_mean
# (+0.628, and the source of a 2e9 gradient norm on its own)
DROP = ("mean_acceleration", "mean_jerk", "curvature_mean")
KILL_C = 79
COOL_C = 74
RESUME_C = 70

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
cols = [FEATURE_NAMES.index(n) for n in TRAINED if n not in DROP]

ck = torch.load(CKPT, map_location=dev, weights_only=False)
model = EventARModel(**ck["config"]).to(dev)
model.load_state_dict(ck["model_state_dict"])
model.eval()
names = [n for n, p in model.named_parameters() if p.requires_grad]
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


def blocknorm(g):
    o = g.clone()
    for i in range(len(sizes)):
        sl = slice(int(offs[i]), int(offs[i + 1]))
        n = o[sl].norm()
        if n > 0:
            o[sl] /= n
    return o


def cos(a, b):
    # gradients are held in float32 so that twenty four of them fit in a 14 GB
    # box, and promoted per pair for the dot, because float32 returned 1.0064
    # for a pair of identical vectors in w4_gradsnr_check
    a, b = a.double(), b.double()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))


def pairstats(G):
    v = np.array([cos(a, b) for a, b in itertools.combinations(G, 2)])
    return float(v.mean()), float(v.std(ddof=1) / len(v) ** 0.5)


print(f"\n  checkpoint {CKPT}")
print(f"  {K} batches of {BATCH}, {K * (K - 1) // 2} pairs per tau, "
      f"tau in {TAUS}")
print(f"  objective on {len(cols)} features, dropping {', '.join(DROP)}\n",
      flush=True)

# Tokens are sampled once and cached, then each temperature is run as its own
# pass. Holding all three temperatures' gradients at once would be 72 vectors of
# 21.7M, which does not fit in a 14 GB box; this holds 24 at a time. The three
# arms still see byte identical batches, so they stay paired.
SCR = os.environ.get("W4_CACHE", "/tmp/w4_cache")
os.makedirs(SCR, exist_ok=True)

print("  sampling and caching the batches\n", flush=True)
batches = []
for k in range(K):
    t = gate()
    path = f"{SCR}/softdec_tau_b{k}_{BATCH}_{CAP}.npz"
    if os.path.exists(path):
        print(f"  batch {k:>2}  cached", flush=True)
        batches.append(path)
        continue
    rws = np.sort(train_rows[rng.choice(len(train_rows), BATCH, replace=False)])
    cond = torch.tensor(np.asarray(cond_all[rws], dtype=np.float32)).to(dev)
    ang = np.arctan2(cond[:, 3].cpu().numpy().astype(np.float64),
                     cond[:, 2].cpu().numpy().astype(np.float64))
    with torch.no_grad():
        s, th, dt = model.sample(cond, seq_len=CAP)
    sn, thn, dtn = (s.cpu().numpy(), th.cpu().numpy(), dt.cpu().numpy())
    Xh, keep, _ = decode_batch(list(sn), list(thn), list(dtn), ang)
    if len(keep) < 16:
        print(f"  batch {k:>2}  only {len(keep)} rows survived, skipped")
        continue
    np.savez(path, s=sn, th=thn, dt=dtn, rows=rws, ang=ang, Xh=Xh,
             keep=np.asarray(keep))
    batches.append(path)
    del s, th, dt, cond
    torch.cuda.empty_cache()
    print(f"  batch {k:>2}  rows {len(keep)}  {t}C", flush=True)

res = {"taus": {}, "n_batches": len(batches)}
print(f"\n  {len(batches)} batches cached, now one pass per temperature\n",
      flush=True)

for tau in TAUS:
    gate()
    G, norms = [], []
    for path in batches:
        z = np.load(path)
        s = torch.from_numpy(z["s"]).to(dev)
        th = torch.from_numpy(z["th"]).to(dev)
        dt = torch.from_numpy(z["dt"]).to(dev)
        cond = torch.tensor(np.asarray(cond_all[z["rows"]],
                                       dtype=np.float32)).to(dev)
        angt = torch.tensor(z["ang"], dtype=torch.float32, device=dev)
        Xht = torch.tensor(z["Xh"], dtype=torch.float32, device=dev)
        ki = torch.tensor(z["keep"], device=dev)
        Xs, _ = soft_forward(model, s, th, dt, cond, angt, tau=tau)
        zt = ((straight_through(Xht, Xs[ki]) - mu_t) / sd_t)[:, cols]
        m = zt.mean(0)
        sdev = zt.std(0).clamp(min=1e-4)
        model.zero_grad(set_to_none=True)
        ((m ** 2).sum() + (torch.log(sdev) ** 2).sum()).backward()
        g = flat_grad()
        G.append(g)
        norms.append(float(g.norm()))
        del s, th, dt, cond, Xs, zt
        torch.cuda.empty_cache()
    r, se = pairstats(G)
    rb, seb = pairstats([blocknorm(g) for g in G])
    sp = max(norms) / max(min(norms), 1e-30)
    res["taus"][str(tau)] = {"raw": r, "raw_se": se, "block": rb,
                             "block_se": seb, "norm_spread": sp,
                             "median_norm": float(np.median(norms))}
    print(f"  tau {tau:g}   raw {r:+.4f} se {se:.4f}    block {rb:+.4f} "
          f"se {seb:.4f}    norm spread {sp:.0f}x    median |g| "
          f"{np.median(norms):.2f}", flush=True)
    del G
    import gc
    gc.collect()

b = [res["taus"][str(t)]["block"] for t in TAUS]
print(f"\n  for reference, same checkpoint and batch size")
print(f"    REINFORCE surrogate   block  -0.0071 se 0.0125")
print(f"    supervised anchor     block  +0.0942 se 0.0192")
met = max(b) > 0.05 and b[0] < b[1] < b[2]
fals = all(v <= 0.03 for v in b)
print(f"\n  PREDICTION {'MET' if met else 'NOT MET'}    "
      f"FALSIFIER {'TRIGGERED' if fals else 'not met'}")
if met:
    print("  the pathwise gradient carries signal the score function one does "
          "not, and the training arm is worth building.")
if fals:
    print("  no temperature rescues it. the next thing to doubt is the "
          "relaxation's bias or the model class, not the estimator.")

best = TAUS[int(np.argmax(b))]
print(f"\n  per tensor at the best tau, {best:g}, recomputed from the cached "
      f"batches")
G = []
for path in batches:
    z = np.load(path)
    s = torch.from_numpy(z["s"]).to(dev)
    th = torch.from_numpy(z["th"]).to(dev)
    dt = torch.from_numpy(z["dt"]).to(dev)
    cond = torch.tensor(np.asarray(cond_all[z["rows"]],
                                   dtype=np.float32)).to(dev)
    Xs, _ = soft_forward(model, s, th, dt, cond,
                         torch.tensor(z["ang"], dtype=torch.float32,
                                      device=dev), tau=best)
    zt = ((straight_through(torch.tensor(z["Xh"], dtype=torch.float32,
                                         device=dev),
                            Xs[torch.tensor(z["keep"], device=dev)])
           - mu_t) / sd_t)[:, cols]
    m = zt.mean(0)
    sdev = zt.std(0).clamp(min=1e-4)
    model.zero_grad(set_to_none=True)
    ((m ** 2).sum() + (torch.log(sdev) ** 2).sum()).backward()
    G.append(flat_grad())
    del s, th, dt, cond, Xs, zt
    torch.cuda.empty_cache()

share = sorted(((float(np.mean([float(g[slice(int(offs[i]), int(offs[i + 1]))]
                                      .norm()) ** 2 for g in G])), i, n)
                for i, n in enumerate(names)), reverse=True)
tot = sum(s_ for s_, _, _ in share)
for sq, i, n in share[:8]:
    sl = slice(int(offs[i]), int(offs[i + 1]))
    print(f"    {n:<44} {100 * sq / tot:5.1f}% of |g|^2   cos "
          f"{pairstats([g[sl] for g in G])[0]:+.4f}")

res.update({"met": bool(met), "falsified": bool(fals), "best_tau": best,
            "peak_temp_c": peak, "dropped": list(DROP)})
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(f"\n  peak {peak}C, wrote {OUT}")
