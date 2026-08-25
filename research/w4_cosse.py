"""Are the standard errors on every gradient cosine in this session wrong, and
if so by how much? Registered before it ran.

THE PROBLEM WITH THE NUMBERS ALREADY REPORTED

Every cosine measured in w4_gradsnr2, w4_softdec_check, w4_softdec_check2,
w4_softdec_tau and w4_softdec_pair used the same helper:

    v = [cos(a, b) for a, b in combinations(G, 2)]
    return v.mean(), v.std(ddof=1) / len(v) ** 0.5

With K gradients that is K(K-1)/2 pairs, and the standard error divides by the
square root of the pair count. That divisor is only correct when the pairs are
independent draws, and they are not: 24 gradients produce 276 pairs, but each
gradient appears in 23 of them, so one unlucky batch moves 23 pairs together.
The mean pairwise cosine is a U statistic, and its variance is set by the
number of gradients, not the number of pairs. Dividing by root 276 instead of
something nearer root 24 understates the error by a factor that could be as
large as root 23, about 4.8.

If the factor is that large, then the headline of this session is misreported.
The supervised anchor read +0.0942 with a quoted se of 0.0192, which was
described as clearly positive and 4.4 se clear of the REINFORCE surrogate. At
4.8 times the error that is +0.0942 se 0.092, which is one se from zero and
says nothing at all. The direction of every conclusion drawn from these
cosines depends on a divisor nobody checked.

WHAT IS MEASURED

The jackknife over gradients, which is the right estimator for a U statistic.
Drop gradient i, recompute the mean pairwise cosine over the remaining pairs,
repeat for all K, and take

    se_jack = sqrt((K - 1) / K * sum_i (theta_(i) - theta_bar) ** 2)

Three arms on the same 24 cached batches, so all three are paired and the
inflation factor is measured on each rather than assumed to be shared:

  A  the supervised anchor NLL, the arm whose positive reading everything else
     is compared against
  B  the batch statistic moment objective through the relaxed decode, tau 4
  C  the per row target through the relaxed decode, tau 4

For B and C the per pair cosines are kept, so the paired difference C minus B
is computed pair by pair rather than from two independent means. Those two
arms see byte identical tokens, so pairing removes the batch draw noise the
two share, and the difference is the quantity w4_softdec_pair left undecided
when it landed between its thresholds at +0.0483.

PREDICTION: the jackknife standard error is at least twice the naive one on
all three arms. That is a statement about the estimator, not about this data,
and the factor should be roughly common across arms.

FALSIFIER: the jackknife error is within 25 percent of the naive one. That
would say the pairs are close enough to independent for the reported errors to
stand, that the concern is unfounded, and that every number already reported
can be read as written.

WHAT THIS DOES NOT DO. It does not change any measured cosine. The point
estimates are unaffected; only their errors move. If the errors inflate, the
correct response is to say which past conclusions no longer follow, not to
rerun everything at once.

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
from models.event_stream_polar import S_PAD_CLASS, TH_NULL_CLASS  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402
from w4_rollout import (  # noqa: E402
    TRAINED, anchor_nll, gpu_temp, load_human,
)
from w4_softdec import soft_forward, straight_through  # noqa: E402

D = "training"
CKPT = "research/w4_rollout_pilot_zbuf_step100.pt"
OUT = "research/w4_cosse.json"
SEED = 17
BATCH = 96
CAP = 256
TAU = 4.0
DROP = ("mean_acceleration", "mean_jerk", "curvature_mean")
KILL_C = 79
SCR = os.environ.get("W4_CACHE", "/tmp/w4_cache")

dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

ok = np.flatnonzero(np.load(f"{D}/events_feat18_ok.npy"))
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


def paircos(G):
    return np.array([cos(a, b) for a, b in itertools.combinations(G, 2)])


def naive_se(v):
    return float(v.std(ddof=1) / len(v) ** 0.5)


def jack_se(v, K):
    """Jackknife over gradients, not over pairs. Drop gradient i, average the
    pairs that do not involve it."""
    idx = list(itertools.combinations(range(K), 2))
    th = []
    for i in range(K):
        keep = [j for j, (a, b) in enumerate(idx) if a != i and b != i]
        th.append(v[keep].mean())
    th = np.asarray(th)
    return float(np.sqrt((K - 1) / K * ((th - th.mean()) ** 2).sum()))


batches = [f"{SCR}/softdec_tau_b{k}_{BATCH}_{CAP}.npz" for k in range(64)]
batches = [p for p in batches if os.path.exists(p)]
if len(batches) < 8:
    raise SystemExit(f"only {len(batches)} cached batches in {SCR}")
K = len(batches)
print(f"\n  checkpoint {CKPT}")
print(f"  {K} cached batches of {BATCH}, {K * (K - 1) // 2} pairs, tau {TAU:g}")
print(f"  naive divisor root {K * (K - 1) // 2}, "
      f"jackknife over {K} gradients\n", flush=True)


def grads_anchor():
    G = []
    for path in batches:
        rws = np.sort(np.load(path)["rows"])
        ah, akept = load_human(rws, CAP, s2a, dtha, dta, lens, cond_all)
        if len(ah) < 8:
            continue
        L = max(len(r[0]) for r in ah)
        AS = torch.full((len(ah), L), S_PAD_CLASS, dtype=torch.long)
        ATH = torch.full((len(ah), L), TH_NULL_CLASS, dtype=torch.long)
        ADT = torch.zeros((len(ah), L), dtype=torch.long)
        for i, r in enumerate(ah):
            AS[i, :len(r[0])] = torch.from_numpy(r[0])
            ATH[i, :len(r[1])] = torch.from_numpy(r[1])
            ADT[i, :len(r[2])] = torch.from_numpy(r[2])
        cond = torch.tensor(np.asarray(cond_all[akept],
                                       dtype=np.float32)).to(dev)
        model.zero_grad(set_to_none=True)
        anchor_nll(model, (AS.to(dev), ATH.to(dev), ADT.to(dev)),
                   cond, False).backward()
        G.append(flat_grad())
    return G


def grads_soft(per_row):
    G = []
    for path in batches:
        z = np.load(path)
        rws, keep = z["rows"], z["keep"]
        s = torch.from_numpy(z["s"]).to(dev)
        th = torch.from_numpy(z["th"]).to(dev)
        dt = torch.from_numpy(z["dt"]).to(dev)
        cond = torch.tensor(np.asarray(cond_all[rws],
                                       dtype=np.float32)).to(dev)
        angt = torch.tensor(z["ang"], dtype=torch.float32, device=dev)
        Xht = torch.tensor(z["Xh"], dtype=torch.float32, device=dev)
        ki = torch.tensor(keep, device=dev)
        Xs, _ = soft_forward(model, s, th, dt, cond, angt, tau=TAU)
        zg = ((straight_through(Xht, Xs[ki]) - mu_t) / sd_t)
        if per_row:
            tgt = np.asarray(Hall[rws[keep]], dtype=np.float64)
            fin = np.isfinite(tgt).all(1)
            if fin.sum() < 16:
                continue
            fi = torch.tensor(np.flatnonzero(fin), device=dev)
            ztg = ((torch.tensor(tgt[fin], dtype=torch.float32, device=dev)
                    - mu_t) / sd_t)[:, cols]
            loss = ((zg[fi][:, cols] - ztg) ** 2).mean()
        else:
            zt = zg[:, cols]
            loss = ((zt.mean(0)) ** 2).sum() + \
                (torch.log(zt.std(0).clamp(min=1e-4)) ** 2).sum()
        model.zero_grad(set_to_none=True)
        loss.backward()
        G.append(flat_grad())
        del Xs, zg, s, th, dt, cond
        torch.cuda.empty_cache()
    return G


ARMS = (("anchor NLL, per row", grads_anchor),
        ("moment, batch statistic", lambda: grads_soft(False)),
        ("per row target", lambda: grads_soft(True)))

res, kept = {}, {}
for tag, fn in ARMS:
    t = gpu_temp()
    if t >= KILL_C:
        raise SystemExit(f"GPU {t}C, at or above the {KILL_C}C kill.")
    G = fn()
    vr = paircos(G)
    vb = paircos([blocknorm(g) for g in G])
    kept[tag] = {"raw": vr, "block": vb, "K": len(G)}
    row = {}
    for st, v in (("raw", vr), ("block", vb)):
        row[st] = {"mean": float(v.mean()), "naive": naive_se(v),
                   "jack": jack_se(v, len(G))}
        row[st]["factor"] = row[st]["jack"] / max(row[st]["naive"], 1e-12)
    res[tag] = row
    b = row["block"]
    print(f"  {tag:<26} block {b['mean']:+.4f}   "
          f"naive se {b['naive']:.4f}   jackknife se {b['jack']:.4f}   "
          f"inflation {b['factor']:.2f}x", flush=True)
    del G
    torch.cuda.empty_cache()

fac = [res[t]["block"]["factor"] for t, _ in ARMS]
met = all(f >= 2.0 for f in fac)
fals = all(abs(f - 1.0) <= 0.25 for f in fac)
print(f"\n  inflation factors {', '.join(f'{f:.2f}' for f in fac)}")
print(f"  PREDICTION {'MET' if met else 'NOT MET'}    "
      f"FALSIFIER {'TRIGGERED' if fals else 'not met'}")

print("\n  every block normalised cosine this session, with both errors")
for tag, _ in ARMS:
    b = res[tag]["block"]
    lo, hi = b["mean"] - 2 * b["jack"], b["mean"] + 2 * b["jack"]
    print(f"    {tag:<26} {b['mean']:+.4f}  naive +-{2 * b['naive']:.4f}  "
          f"jackknife +-{2 * b['jack']:.4f}   "
          f"{'clear of zero' if lo > 0 or hi < 0 else 'includes zero'}")

# the paired difference, per row target minus batch statistic, on identical
# tokens and therefore on identical pair indices
a = kept["per row target"]["block"]
b = kept["moment, batch statistic"]["block"]
if len(a) == len(b):
    d = a - b
    dj = jack_se(d, kept["per row target"]["K"])
    print(f"\n  paired, per row minus batch statistic  {d.mean():+.4f}  "
          f"naive se {naive_se(d):.4f}   jackknife se {dj:.4f}")
    print(f"  {'the two objectives differ' if abs(d.mean()) > 2 * dj else 'the two objectives are not distinguishable'}")
    res["paired_diff"] = {"mean": float(d.mean()), "naive": naive_se(d),
                          "jack": dj}

res.update({"met": bool(met), "falsified": bool(fals), "K": K, "tau": TAU})
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(f"\n  wrote {OUT}")
