"""Gradient signal to noise, second design. Registered before it ran.

WHY THERE IS A SECOND DESIGN

w4_gradsnr asked whether two half batch gradients point the same way and set a
threshold of 0.05. Its control killed the threshold. The anchor NLL, ordinary
supervised teacher forcing that demonstrably trains this model, produced cosines
of -0.36, -0.44, -0.56, +0.83, +0.30, -0.71: mean -0.16, sd 0.60. A gradient
known to work scatters across almost the whole range, so "below 0.05" cannot
mean "this gradient is unusable". It is the ordinary condition of minibatch
training, where the cosine between two batch gradients is about

    |mu|^2 / (|mu|^2 + sigma^2)

and sits near zero whenever per batch noise dominates the mean, which is nearly
always. SGD still works, because the mean is recovered by averaging over steps
and not within one.

w4_gradsnr_check found why the spread is so wide. 51.4% of the squared gradient
norm lives in cond_embed.0.weight, 1536 numbers out of 21.7 million. A cosine
dominated by one small block behaves like a cosine in a handful of dimensions,
so it lands near plus or minus one and tells you almost nothing per repeat. The
reader itself is sound: the same batch computed twice deterministically returns
1.000.

WHAT THIS RUN CHANGES

  1  The statistic is the mean over all pairs of K gradients, not one pair.
     K = 8 gives 28 pairs and an error bar small enough to compare two arms.
  2  Everything is measured against the anchor control on identical footing, and
     the number that gets read is the RATIO. Absolute cosine has no meaning here;
     "the rollout gradient is a quarter as well determined per batch as ordinary
     supervised training" does.
  3  A block normalised cosine is reported alongside the raw one. Each parameter
     tensor is scaled to unit norm before the dot, so cond_embed cannot decide
     the answer by itself. If the two disagree, the answer depends on which
     subspace you weight, and that is worth knowing on its own.
  4  Dot products and norms in float64. In float32 the identical vector control
     returned 1.0064, which is a 0.6% error and larger than some effects here.

PREDICTION, fixed before the run: the surrogate's mean pairwise cosine is below
half the anchor's, on both the raw and the block normalised statistic.

FALSIFIER: the surrogate reaches or beats the anchor. That would say the
rollout gradient is as well determined per batch as supervised training, the
estimator is not the binding constraint, and the plateau has to be explained by
the model class or by the objectives themselves.

WHAT IT STILL CANNOT SETTLE. It measures agreement, not correctness. A gradient
can be perfectly consistent across batches and still point somewhere useless,
because consistency says the batches agree with each other and nothing about
whether their common direction lowers the contract score. And it measures one
objective, the moment branch, chosen because its weights depend on nothing but
the generated batch.

No model file is written and the protected eval sample is never read.
"""
from __future__ import annotations

import itertools
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
OUT = "research/w4_gradsnr2.json"
SEED = 17
BATCH = 96
K = 8              # 28 pairs per arm
CAP = 256
CLIP_W = 5.0
AMP = False        # deterministic, and the check showed amp alone costs 0.37
                   # of same batch cosine, which would be charged to both arms
                   # but adds variance for nothing
SCALE = 1.0
KILL_C = 79
COOL_C = 74
RESUME_C = 70
COOL_MAX_S = 300

dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

ok = np.flatnonzero(np.load(f"{D}/events_feat18_ok.npy"))
perm = ok[rng.permutation(len(ok))]
train_rows = perm[4000:4000 + 400000][2500:]
cond_all = np.load(f"{D}/events_cond.npy", mmap_mode="r")
s2a = np.load(f"{D}/events_s2.npy", mmap_mode="r")
dtha = np.load(f"{D}/events_dth.npy", mmap_mode="r")
dta = np.load(f"{D}/events_dt.npy", mmap_mode="r")
lens = np.load(f"{D}/events_len.npy")

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
model.eval()      # dropout off: it is nuisance variance charged to both arms
names = [n for n, p in model.named_parameters() if p.requires_grad]
params = [p for p in model.parameters() if p.requires_grad]
sizes = [p.numel() for p in params]
offs = np.cumsum([0] + sizes)
peak, cooled_s = 0, 0.0


def gate():
    global peak, cooled_s
    t = gpu_temp()
    peak = max(peak, t)
    if t >= COOL_C:
        c0 = time.time()
        while gpu_temp() > RESUME_C and time.time() - c0 < COOL_MAX_S:
            time.sleep(10)
        cooled_s += time.time() - c0
        t = gpu_temp()
        peak = max(peak, t)
    if t >= KILL_C:
        raise SystemExit(f"GPU {t}C, at or above the {KILL_C}C kill. Stopping.")
    return t


def flat_grad():
    g = torch.cat([(p.grad if p.grad is not None
                    else torch.zeros_like(p)).detach().flatten()
                   for p in params]).double().cpu()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return g


def blocknorm(g):
    """Each parameter tensor rescaled to unit norm, so no single block can carry
    the cosine by itself."""
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
    v = [cos(a, b) for a, b in itertools.combinations(G, 2)]
    v = np.array(v)
    return float(v.mean()), float(v.std(ddof=1) / len(v) ** 0.5)


def surrogate_grad(rows):
    cond = torch.tensor(np.asarray(cond_all[np.sort(rows)],
                                   dtype=np.float32)).to(dev)
    ang = np.arctan2(cond[:, 3].cpu().numpy().astype(np.float64),
                     cond[:, 2].cpu().numpy().astype(np.float64))
    with torch.no_grad():
        s, th, dt = model.sample(cond, seq_len=CAP)
    X, keep, _ = decode_batch(list(s.cpu().numpy()), list(th.cpu().numpy()),
                              list(dt.cpu().numpy()), ang)
    if len(keep) < 16:
        return None
    z = (torch.tensor(X, dtype=torch.float32) - mu_t) / sd_t
    zt = z[:, tk]
    m = zt.mean(0)
    sdev = zt.std(0).clamp(min=1e-4)
    c = zt - m
    w = (2 * m * c
         + (torch.log(sdev) / sdev ** 2) * (c ** 2 - sdev ** 2)).sum(1)
    w = (w - w.mean()) / w.std().clamp(min=1e-6)
    wt = w.clamp(-CLIP_W, CLIP_W).to(dev)
    ki = torch.tensor(keep, device=dev)
    model.zero_grad(set_to_none=True)
    lp_pos, lp_n = token_logprob_pos(model, s[ki], th[ki], dt[ki],
                                     cond[ki], AMP)
    (SCALE * (wt * lp_pos.sum(1) / lp_n).mean()).backward()
    g = flat_grad()
    del s, th, dt, cond, lp_pos, lp_n
    torch.cuda.empty_cache()
    return g


def anchor_grad(rows):
    ah, akept = load_human(np.sort(rows), CAP, s2a, dtha, dta, lens, cond_all)
    if len(ah) < 8:
        return None
    L = max(len(r[0]) for r in ah)
    AS = torch.full((len(ah), L), S_PAD_CLASS, dtype=torch.long)
    ATH = torch.full((len(ah), L), TH_NULL_CLASS, dtype=torch.long)
    ADT = torch.zeros((len(ah), L), dtype=torch.long)
    for i, r in enumerate(ah):
        AS[i, :len(r[0])] = torch.from_numpy(r[0])
        ATH[i, :len(r[1])] = torch.from_numpy(r[1])
        ADT[i, :len(r[2])] = torch.from_numpy(r[2])
    cond = torch.tensor(np.asarray(cond_all[akept], dtype=np.float32)).to(dev)
    model.zero_grad(set_to_none=True)
    (SCALE * anchor_nll(model, (AS.to(dev), ATH.to(dev), ADT.to(dev)),
                        cond, AMP)).backward()
    return flat_grad()


print(f"\n  checkpoint {CKPT}")
print(f"  {K} batches of {BATCH} per arm, {K * (K - 1) // 2} pairs, cap {CAP}, "
      f"eval mode, amp off\n", flush=True)

# the gradients are cached because every question asked of them afterwards, per
# block, per subspace, is arithmetic on vectors that cost GPU to produce once
SCR = os.environ.get("W4_CACHE", "/tmp/w4_cache")
os.makedirs(SCR, exist_ok=True)
CACHE = f"{SCR}/gradsnr2_{K}x{BATCH}_{CAP}.npz"

GS, GA = [], []
if os.path.exists(CACHE):
    z = np.load(CACHE)
    GS = [torch.from_numpy(z[f"s{k}"]).double() for k in range(K)]
    GA = [torch.from_numpy(z[f"a{k}"]).double() for k in range(K)]
    print(f"  reusing cached gradients from {CACHE}", flush=True)
else:
    for k in range(K):
        t = gate()
        pick = rng.choice(len(train_rows), 2 * BATCH, replace=False)
        gs = surrogate_grad(train_rows[pick[:BATCH]])
        ga = anchor_grad(train_rows[pick[BATCH:]])
        if gs is not None:
            GS.append(gs)
        if ga is not None:
            GA.append(ga)
        print(f"  batch {k}  surrogate |g| {gs.norm():.3f}   anchor |g| "
              f"{ga.norm():.3f}   {t}C", flush=True)
    np.savez(CACHE,
             **{f"s{k}": g.float().numpy() for k, g in enumerate(GS)},
             **{f"a{k}": g.float().numpy() for k, g in enumerate(GA)})

res = {}
for tag, G in (("surrogate", GS), ("anchor", GA)):
    r, se = pairstats(G)
    rb, seb = pairstats([blocknorm(g) for g in G])
    res[tag] = {"raw": r, "raw_se": se, "block": rb, "block_se": seb,
                "n": len(G), "mean_norm": float(np.mean([float(g.norm())
                                                         for g in G]))}
    print(f"\n  {tag:<10} raw pairwise cos {r:+.4f} se {se:.4f}"
          f"    block normalised {rb:+.4f} se {seb:.4f}")
    # the cosine is |mu|^2/(|mu|^2+sigma^2), so this inverts it to the ratio of
    # signal power to per batch noise power, which is the quantity that says how
    # many batches must be averaged before the mean direction dominates
    if 0 < r < 1:
        print(f"             implied signal to noise power {r / (1 - r):.4f}, "
              f"so about {int(round((1 - r) / max(r, 1e-9)))} batches to average"
              f" before the mean direction dominates")

for stat in ("raw", "block"):
    s_, a_ = res["surrogate"][stat], res["anchor"][stat]
    se = (res["surrogate"][stat + "_se"] ** 2
          + res["anchor"][stat + "_se"] ** 2) ** 0.5
    print(f"\n  {stat:<6} surrogate minus anchor {s_ - a_:+.4f}  se {se:.4f}"
          f"   ratio {s_ / a_ if a_ else float('nan'):.3f}")
    res[stat + "_diff"] = s_ - a_
    res[stat + "_diff_se"] = se

met = all(res["surrogate"][s] < 0.5 * res["anchor"][s] for s in ("raw", "block"))
fals = any(res["surrogate"][s] >= res["anchor"][s] for s in ("raw", "block"))
print(f"\n  PREDICTION {'MET' if met else 'NOT MET'}"
      f"    FALSIFIER {'TRIGGERED on at least one statistic' if fals else 'not met'}")

# where the two arms differ by block, which is the actionable part
print("\n  per tensor mean pairwise cosine, blocks holding the most gradient")
share = []
for i, n in enumerate(names):
    sl = slice(int(offs[i]), int(offs[i + 1]))
    share.append((float(np.mean([float(g[sl].norm()) ** 2 for g in GA])), i, n))
share.sort(reverse=True)
tot = sum(s for s, _, _ in share)
blocks = []
for sq, i, n in share[:10]:
    sl = slice(int(offs[i]), int(offs[i + 1]))
    cs = pairstats([g[sl] for g in GS])[0]
    ca = pairstats([g[sl] for g in GA])[0]
    print(f"    {n:<44} {100 * sq / tot:5.1f}% of |g|^2   surrogate {cs:+.4f}"
          f"   anchor {ca:+.4f}")
    blocks.append({"name": n, "share": sq / tot, "surrogate": cs, "anchor": ca})
res["blocks"] = blocks
res["peak_temp_c"] = peak
res["cooldown_min"] = round(cooled_s / 60, 1)
print(f"\n  peak {peak}C, {cooled_s / 60:.1f} min cooling")

with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(f"  wrote {OUT}")
