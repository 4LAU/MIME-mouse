"""Does the relaxed decode transmit a well determined gradient for any
objective at all? Registered before it ran.

WHAT IS NOT BEING RETRACTED

w4_softdec_tau measured the block normalised pairwise gradient cosine of the
moment objective through the relaxed decode at three temperatures, 24 batches
of 96 each, 276 pairs per temperature:

    tau 1   block -0.0051 se 0.0147
    tau 2   block +0.0083 se 0.0193
    tau 4   block +0.0130 se 0.0213

Its falsifier was all three at or below +0.03, and **that falsifier is met and
it stays met**. The registered meaning stands: a pathwise gradient of the
moment objective through this decode carries no more batch to batch signal
than the score function estimator it was built to replace, and no temperature
rescues it. Nothing below reinterprets that. This is a new experiment with its
own prediction.

WHY THERE IS A NEXT EXPERIMENT AT ALL

Four numbers are now measured on the same checkpoint, the same batch size and
the same statistic:

                              score function        pathwise
    batch statistic           -0.0071 se 0.0125     -0.005 to +0.013
    per row target            +0.0942 se 0.0192     not measured

The supervised anchor is the only one that reads clearly positive, and it
differs from the three failures in two ways at once. It uses a different
estimator, and it uses a different objective: a target attached to each row
rather than a statistic of the batch. With those two changes confounded, the
three zeros cannot be attributed. The empty cell separates them.

THE OBJECTIVE THAT FILLS IT. Every generated trajectory is conditioned on the
start, end and duration of a real one, and that real trajectory's own eighteen
features sit at the same row of events_feat18.npy. So each generated row has a
natural per row target, and the loss is the squared error to it in z units on
the nine features whose relaxed value tracks the served one above 0.9. Same
relaxation, same straight through join at the features, same parameters, same
cached tokens as the temperature sweep. Only the shape of the objective moves.

PREDICTION: the block normalised pairwise cosine is above +0.05 at one of the
two temperatures. That would put the relaxation in the clear, because a decode
that transmits a well determined gradient for a per row target is not
structurally incapable of transmitting one, and would move the failure onto the
batch statistic objective, which every arm run so far has used.

FALSIFIER: both at or below +0.02. That would say the relaxed decode transmits
no usable gradient whatever the objective, which puts the fault in the straight
through construction rather than in the choice of objective, and ends the arm
without the ambiguity the temperature sweep left open.

WHAT THIS CANNOT SETTLE, whichever way it lands. A per row target is a
different thing to want than a matched distribution. Hitting each real
trajectory's own features would score well, but so would a model that has
merely learned the conditional mean, and the contract scorer punishes exactly
that through the dispersion ratios. So this measures whether gradient reaches
the parameters, not whether following it is a good idea. That second question
needs a scored training run and is not this.

It also remains true that a consistent gradient can be consistently biased. The
relaxation drops the snap, the rounding and the tick merge.

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
OUT = "research/w4_softdec_pair.json"
SEED = 17
BATCH = 96
CAP = 256
TAUS = (1.0, 4.0)
DROP = ("mean_acceleration", "mean_jerk", "curvature_mean")
KILL_C = 79
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
names = [n for n, p in model.named_parameters() if p.requires_grad]
params = [p for p in model.parameters() if p.requires_grad]
sizes = [p.numel() for p in params]
offs = np.cumsum([0] + sizes)


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
    a, b = a.double(), b.double()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))


def pairstats(G):
    v = np.array([cos(a, b) for a, b in itertools.combinations(G, 2)])
    return float(v.mean()), float(v.std(ddof=1) / len(v) ** 0.5)


# The temperature sweep cached its tokens. Reusing them means this arm and that
# one see byte identical batches, so the two are paired rather than merely
# comparable, and it costs no sampling.
batches = sorted(
    (f"{SCR}/softdec_tau_b{k}_{BATCH}_{CAP}.npz" for k in range(64)),
    key=lambda p: int(p.split("_b")[1].split("_")[0]))
batches = [p for p in batches if os.path.exists(p)]
if len(batches) < 8:
    raise SystemExit(f"only {len(batches)} cached batches in {SCR}, "
                     "run w4_softdec_tau.py first")

print(f"\n  checkpoint {CKPT}")
print(f"  {len(batches)} cached batches of {BATCH}, "
      f"{len(batches) * (len(batches) - 1) // 2} pairs per tau")
print(f"  per row squared error on {len(cols)} features, "
      f"dropping {', '.join(DROP)}\n", flush=True)

res = {"taus": {}, "n_batches": len(batches)}
for tau in TAUS:
    t = gpu_temp()
    if t >= KILL_C:
        raise SystemExit(f"GPU {t}C, at or above the {KILL_C}C kill.")
    G, norms, losses, nrows = [], [], [], 0
    for path in batches:
        z = np.load(path)
        rws, keep = z["rows"], z["keep"]
        # the target is the real trajectory whose start, end and duration
        # conditioned this row
        tgt = np.asarray(Hall[rws[keep]], dtype=np.float64)
        fin = np.isfinite(tgt).all(1)
        if fin.sum() < 16:
            continue
        s = torch.from_numpy(z["s"]).to(dev)
        th = torch.from_numpy(z["th"]).to(dev)
        dt = torch.from_numpy(z["dt"]).to(dev)
        cond = torch.tensor(np.asarray(cond_all[rws],
                                       dtype=np.float32)).to(dev)
        angt = torch.tensor(z["ang"], dtype=torch.float32, device=dev)
        Xht = torch.tensor(z["Xh"], dtype=torch.float32, device=dev)
        ki = torch.tensor(keep, device=dev)
        fi = torch.tensor(np.flatnonzero(fin), device=dev)
        ztg = torch.tensor(tgt[fin], dtype=torch.float32, device=dev)
        ztg = ((ztg - mu_t) / sd_t)[:, cols]

        Xs, _ = soft_forward(model, s, th, dt, cond, angt, tau=tau)
        zgen = ((straight_through(Xht, Xs[ki]) - mu_t) / sd_t)[fi][:, cols]
        loss = ((zgen - ztg) ** 2).mean()
        model.zero_grad(set_to_none=True)
        loss.backward()
        g = flat_grad()
        G.append(g)
        norms.append(float(g.norm()))
        losses.append(float(loss.detach()))
        nrows += int(fin.sum())
        del Xs, zgen, s, th, dt, cond
        torch.cuda.empty_cache()

    r, se = pairstats(G)
    rb, seb = pairstats([blocknorm(g) for g in G])
    spread = max(norms) / max(min(norms), 1e-30)
    print(f"  tau {tau:g}   raw {r:+.4f} se {se:.4f}    "
          f"block {rb:+.4f} se {seb:.4f}    "
          f"spread {spread:.0f}x   median |g| {np.median(norms):.2f}")
    print(f"           mean per row loss {np.mean(losses):.4f} "
          f"over {nrows} rows, {len(G)} batches", flush=True)
    res["taus"][str(tau)] = {"raw": r, "raw_se": se, "block": rb,
                             "block_se": seb, "norm_spread": spread,
                             "loss_mean": float(np.mean(losses)),
                             "norms": norms, "losses": losses}
    if tau == TAUS[-1]:
        share = sorted(((float(np.mean([float(g[slice(int(offs[i]),
                                                      int(offs[i + 1]))]
                                              .norm()) ** 2 for g in G])), i, n)
                        for i, n in enumerate(names)), reverse=True)
        tot = sum(s_ for s_, _, _ in share)
        blocks = []
        for sq, i, n in share[:8]:
            sl = slice(int(offs[i]), int(offs[i + 1]))
            c = pairstats([g[sl] for g in G])[0]
            blocks.append({"name": n, "share": 100 * sq / tot, "cos": c})
        res["blocks"] = blocks
    del G
    torch.cuda.empty_cache()

best = max(v["block"] for v in res["taus"].values())
met = best > 0.05
fals = best <= 0.02
print(f"\n  measured on this checkpoint and batch size, same statistic")
print(f"    REINFORCE surrogate, batch statistic    -0.0071 se 0.0125")
print(f"    pathwise, batch statistic, best tau     +0.0130 se 0.0213")
print(f"    supervised anchor NLL, per row target   +0.0942 se 0.0192")
print(f"    pathwise, per row target, best tau      {best:+.4f}")
print(f"\n  PREDICTION {'MET' if met else 'NOT MET'}    "
      f"FALSIFIER {'TRIGGERED' if fals else 'not met'}")
if met:
    print("  the relaxed decode does carry a well determined gradient. the "
          "failure is in the batch statistic objective, which every arm so "
          "far has used.")
elif fals:
    print("  the relaxed decode carries no usable gradient for either shape "
          "of objective. the straight through construction is the fault, not "
          "the objective.")
else:
    print("  between the two thresholds, so it settles neither. do not read a "
          "direction into it.")

if "blocks" in res:
    print(f"\n  per tensor at tau {TAUS[-1]:g}")
    for b in res["blocks"]:
        print(f"    {b['name']:<44} {b['share']:5.1f}% of |g|^2   "
              f"cos {b['cos']:+.4f}")

res.update({"best_block": best, "met": bool(met), "falsified": bool(fals)})
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(f"\n  wrote {OUT}")
