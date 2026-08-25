"""Does the relaxed decode transmit a well determined gradient for a per row
target, measured at enough precision to call it? Registered before it ran.

WHAT CAME BEFORE

w4_softdec_pair filled the empty cell of the design at batch 96 and landed
between its own thresholds:

    per row target through the relaxed decode, tau 4   +0.0483

and w4_cosse then showed the standard errors this session has been quoting are
understated. The naive helper divides by the root of the pair count, but 276
pairs come from 24 gradients, so the pairs are not independent. The jackknife
over gradients inflates the error by 2.04x on the anchor, 1.70x on the per row
target and 0.84x on the batch statistic arm, which is the expected pattern:
pairs only become dependent when there is a shared mean direction for them to
be dependent about, so the null arm is unaffected and the arms with signal are.

Repriced with jackknife errors:

    anchor NLL, per row target        +0.0992  se 0.0139   clear of zero
    moment, batch statistic           +0.0058  se 0.0190   includes zero
    per row target, pathwise          +0.0483  se 0.0344   includes zero

**The headline of the session survives that correction.** The anchor beats the
batch statistic objective by +0.0934 with a jackknife se of 0.0235, 4.0 se
rather than the 4.4 se quoted from naive errors. The conclusion that the
rollout gradient carries no batch to batch signal while ordinary supervised
teacher forcing does is unchanged. What does not survive is the reading of the
per row cell: at 1.4 se from zero it is undecided, not positive.

WHY THIS DESIGN

Batch 192 with gradients through the trunk does not fit. It fails as
`dxgk: dxgkio_make_resident: Ioctl failed: -12`, which is the WSL paravirt
layer running out of the 8 GB of GPU memory and surfacing it as a driver error
rather than a clean torch OOM. So the batch cannot grow directly.

It can grow indirectly. The per row objective is a sum over rows, so its
gradient is exactly additive across sub batches: the gradient at 384 rows is
the sum of four gradients at 96 rows, with no approximation. The batch
statistic objective has no such property, because a mean and a standard
deviation over a batch do not decompose, which is why only the per row arm can
be measured this way. Since the pairwise cosine is about n|mu|^2/sigma^2 while
it is small, four times the batch should give about four times the cosine,
while the error only falls as the root of the gradient count. That is the
cheapest precision available here.

Both readings come out of one set of 64 sub batch gradients, grouped two ways,
so they cost one sampling run between them.

PREDICTION: at an effective batch of 384 the block normalised cosine is above
+0.12 and at least 2.5 jackknife se clear of zero. That would say the relaxed
decode does transmit a well determined gradient, which would put the
relaxation in the clear and move the whole failure onto the batch statistic
objective that every arm run so far has used.

FALSIFIER: the effective batch 384 reading is below +0.05 and within 2
jackknife se of zero. Four times the batch and a per row target is the most
favourable case this construction gets, so failing there ends the straight
through arm outright rather than leaving it open.

DIAGNOSTIC, NOT A GATE. The ratio of the 384 reading to the 96 reading should
be near 4 if the small cosine approximation holds. A ratio far from 4 does not
by itself decide anything, but it would say the two readings are not measuring
what the scaling argument assumes, and neither should then be quoted as a
signal share.

WHAT THIS STILL CANNOT SETTLE, whichever way it lands. A per row target is a
different thing to want than a matched distribution. Hitting each real
trajectory's own features would also be achieved by a model that has collapsed
onto the conditional mean, and the contract scorer punishes exactly that
through the dispersion ratios. This measures whether gradient reaches the
parameters, not whether following it would score. It also remains true that a
consistent gradient can be consistently biased, because the relaxation drops
the snap, the rounding and the tick merge.

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
OUT = "research/w4_softdec_pair2.json"
SEED = 17
BATCH = 96
CAP = 256
TAU = 4.0          # the best of the three the temperature sweep measured
GROUP = 4          # sub batches summed, giving an effective batch of 384
K = 64             # sub batches, so 16 groups
DROP = ("mean_acceleration", "mean_jerk", "curvature_mean")
KILL_C = 79
COOL_C = 74
RESUME_C = 70
SCR = os.environ.get("W4_CACHE", "/tmp/w4_cache")

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


def blocknorm_(g):
    """In place, because 64 of these are held at once and a copy would not
    fit."""
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


def report(v, n, tag):
    m, j = float(v.mean()), jack_se(v, n)
    naive = float(v.std(ddof=1) / len(v) ** 0.5)
    print(f"  {tag:<34} {m:+.4f}   jackknife se {j:.4f}   "
          f"(naive {naive:.4f})   {m / max(j, 1e-9):.1f} se from zero",
          flush=True)
    return {"mean": m, "jack": j, "naive": naive, "n": n}


os.makedirs(SCR, exist_ok=True)
print(f"\n  checkpoint {CKPT}")
print(f"  {K} sub batches of {BATCH} at tau {TAU:g}, summed {GROUP} at a time")
print(f"  giving {K // GROUP} gradients at an effective batch of "
      f"{BATCH * GROUP}")
print(f"  per row squared error on {len(cols)} features, "
      f"dropping {', '.join(DROP)}\n", flush=True)

# The first 24 slots reuse the temperature sweep's cache, so this arm and that
# one are paired on identical tokens where they overlap.
paths = []
for k in range(K):
    p = f"{SCR}/softdec_tau_b{k}_{BATCH}_{CAP}.npz"
    if not os.path.exists(p):
        t = gate()
        rws = np.sort(train_rows[rng.choice(len(train_rows), BATCH,
                                            replace=False)])
        cond = torch.tensor(np.asarray(cond_all[rws],
                                       dtype=np.float32)).to(dev)
        ang = np.arctan2(cond[:, 3].cpu().numpy().astype(np.float64),
                         cond[:, 2].cpu().numpy().astype(np.float64))
        with torch.no_grad():
            s, th, dt = model.sample(cond, seq_len=CAP)
        sn, thn, dtn = s.cpu().numpy(), th.cpu().numpy(), dt.cpu().numpy()
        Xh, keep, _ = decode_batch(list(sn), list(thn), list(dtn), ang)
        if len(keep) < 16:
            print(f"  batch {k:>2}  only {len(keep)} rows survived, skipped")
            continue
        np.savez(p, s=sn, th=thn, dt=dtn, rows=rws, ang=ang, Xh=Xh,
                 keep=np.asarray(keep))
        del s, th, dt, cond
        torch.cuda.empty_cache()
        print(f"  batch {k:>2}  sampled, rows {len(keep)}  {t}C", flush=True)
    paths.append(p)

print(f"\n  {len(paths)} sub batches ready, taking gradients\n", flush=True)

G, nrows = [], []
for k, path in enumerate(paths):
    gate()
    z = np.load(path)
    rws, keep = z["rows"], z["keep"]
    tgt = np.asarray(Hall[rws[keep]], dtype=np.float64)
    fin = np.isfinite(tgt).all(1)
    if fin.sum() < 16:
        continue
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
    # a sum over rows, not a mean, so that summing sub batch gradients gives
    # the gradient of the same objective at the larger batch exactly
    loss = ((zg - ztg) ** 2).sum()
    model.zero_grad(set_to_none=True)
    loss.backward()
    G.append(flat_grad())
    nrows.append(int(fin.sum()))
    del Xs, zg, s, th, dt, cond
    torch.cuda.empty_cache()
    if (k + 1) % 8 == 0:
        print(f"  {k + 1} of {len(paths)} gradients", flush=True)

n96 = len(G)
ngrp = n96 // GROUP
print(f"\n  {n96} sub batch gradients over {sum(nrows)} rows, "
      f"{ngrp} groups of {GROUP}\n", flush=True)

# group first, while the raw gradients are still raw
grp = [torch.stack(G[i * GROUP:(i + 1) * GROUP]).sum(0) for i in range(ngrp)]
vg = paircos([blocknorm_(g) for g in grp])
del grp

for g in G:
    blocknorm_(g)
v96 = paircos(G)
del G

res = {"n_sub": n96, "rows": int(sum(nrows)), "group": GROUP, "tau": TAU}
res["b96"] = report(v96, n96, f"batch {BATCH}, {n96} gradients")
res["b384"] = report(vg, ngrp, f"batch {BATCH * GROUP}, {ngrp} gradients")

ratio = res["b384"]["mean"] / res["b96"]["mean"] if res["b96"]["mean"] else \
    float("nan")
print(f"\n  scaling ratio {ratio:.2f}, expected near {GROUP} if the cosine is "
      f"a signal share")

m, j = res["b384"]["mean"], res["b384"]["jack"]
met = m > 0.12 and m > 2.5 * j
fals = m < 0.05 and abs(m) < 2 * j
print(f"\n  measured on this checkpoint, block normalised, jackknife errors")
print(f"    anchor NLL, per row target, batch 96     +0.0992 se 0.0139")
print(f"    moment, batch statistic, batch 96        +0.0058 se 0.0190")
print(f"    per row target, batch 96, 24 gradients   +0.0483 se 0.0344")
print(f"    per row target, batch {BATCH * GROUP:<3}, {ngrp} gradients   "
      f"{m:+.4f} se {j:.4f}")
print(f"\n  PREDICTION {'MET' if met else 'NOT MET'}    "
      f"FALSIFIER {'TRIGGERED' if fals else 'not met'}")
if met:
    print("  the relaxed decode carries a well determined gradient. the "
          "failure is the batch statistic objective, not the relaxation.")
elif fals:
    print("  four times the batch and a per row target is the most "
          "favourable case, and it still carries nothing. the straight "
          "through construction is the fault.")
else:
    print("  between the thresholds again. do not read a direction into it.")

res.update({"ratio": ratio, "met": bool(met), "falsified": bool(fals),
            "peak_c": peak})
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(f"\n  peak {peak}C, wrote {OUT}")
