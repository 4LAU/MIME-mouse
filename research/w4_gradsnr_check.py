"""Instrument check for w4_gradsnr. Does the gradient reader work at all?

w4_gradsnr's first two repeats returned a NEGATIVE cosine for the anchor NLL
control, about -0.36 and -0.44. Two disjoint batches of ordinary supervised
teacher forcing cannot be anti correlated on a model trained on that corpus, so
before any reading is taken from the surrogate number the reader itself has to
be shown to work.

Four controls, cheapest first, each one isolating a different suspect.

  1  SAME batch, model.eval(), amp off. The two gradients are computed from
     identical inputs by a deterministic network. Cosine must be 1.000. Anything
     else means flat_grad or the cosine is broken and nothing downstream counts.
  2  SAME batch, model.train(), amp on. Same as 1 but with dropout live and fp16
     autocast. Cosine should still be high. If 1 passes and 2 does not, the
     nondeterminism is dropout or fp16, not the reader.
  3  DISJOINT batches, model.eval(), amp off. This is the real question with
     every source of noise except batch composition removed.
  4  The dot product in float64. If 3 disagrees with its float64 twin then the
     cosine was being destroyed by cancellation in float32 across 21 million
     coordinates and the fix is the accumulator, not the experiment.

Nothing here trains, nothing is written except this script's own JSON, and the
protected eval sample is never touched.
"""
from __future__ import annotations

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
from w4_rollout import anchor_nll, load_human  # noqa: E402

D = "training"
CKPT = "research/w4_rollout_pilot_zbuf_step100.pt"
OUT = "research/w4_gradsnr_check.json"
SEED = 17
HALF = 96
CAP = 256
SCALE = 1024.0

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

ck = torch.load(CKPT, map_location=dev, weights_only=False)
model = EventARModel(**ck["config"]).to(dev)
model.load_state_dict(ck["model_state_dict"])
params = [p for p in model.parameters() if p.requires_grad]
print(f"  {sum(p.numel() for p in params) / 1e6:.1f}M parameters")


def flat_grad():
    return torch.cat([(p.grad if p.grad is not None
                       else torch.zeros_like(p)).detach().flatten()
                      for p in params]).float().cpu()


def cosine(a, b, dtype=torch.float32):
    a, b = a.to(dtype), b.to(dtype)
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def batch(rows):
    ah, akept = load_human(np.sort(rows), CAP, s2a, dtha, dta, lens, cond_all)
    L = max(len(r[0]) for r in ah)
    AS = torch.full((len(ah), L), S_PAD_CLASS, dtype=torch.long)
    ATH = torch.full((len(ah), L), TH_NULL_CLASS, dtype=torch.long)
    ADT = torch.zeros((len(ah), L), dtype=torch.long)
    for i, r in enumerate(ah):
        AS[i, :len(r[0])] = torch.from_numpy(r[0])
        ATH[i, :len(r[1])] = torch.from_numpy(r[1])
        ADT[i, :len(r[2])] = torch.from_numpy(r[2])
    cond = torch.tensor(np.asarray(cond_all[akept], dtype=np.float32)).to(dev)
    return (AS.to(dev), ATH.to(dev), ADT.to(dev)), cond


def grad_of(b, cond, amp):
    model.zero_grad(set_to_none=True)
    nll = anchor_nll(model, b, cond, amp)
    (SCALE * nll).backward()
    g = flat_grad()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return g, float(nll)


pick = rng.choice(len(train_rows), 2 * HALF, replace=False)
bA, cA = batch(train_rows[pick[:HALF]])
bB, cB = batch(train_rows[pick[HALF:]])

res = {}

model.eval()
g1, n1 = grad_of(bA, cA, False)
g2, n2 = grad_of(bA, cA, False)
res["same_eval_noamp"] = cosine(g1, g2)
print(f"\n  1  same batch, eval, no amp     cos {res['same_eval_noamp']:+.6f}"
      f"   nll {n1:.4f} vs {n2:.4f}   must be 1.000000")

model.train()
g3, _ = grad_of(bA, cA, True)
g4, _ = grad_of(bA, cA, True)
res["same_train_amp"] = cosine(g3, g4)
print(f"  2  same batch, train, amp       cos {res['same_train_amp']:+.6f}")

model.eval()
gA, na = grad_of(bA, cA, False)
gB, nb = grad_of(bB, cB, False)
res["disjoint_eval_noamp"] = cosine(gA, gB)
res["disjoint_eval_noamp_f64"] = cosine(gA, gB, torch.float64)
print(f"  3  disjoint, eval, no amp       cos {res['disjoint_eval_noamp']:+.6f}"
      f"   nll {na:.4f} vs {nb:.4f}")
print(f"  4  the same in float64          cos "
      f"{res['disjoint_eval_noamp_f64']:+.6f}")

# where does the disagreement live? a cosine over 21M coordinates hides which
# tensors are fighting, and a single dominant block can carry the whole sign.
print("\n  per tensor, disjoint batches, largest gradient blocks first")
o, rowsg = 0, []
for nme, p in model.named_parameters():
    if not p.requires_grad:
        continue
    k = p.numel()
    a, b = gA[o:o + k], gB[o:o + k]
    rowsg.append((float(a.norm()), nme, cosine(a, b), k))
    o += k
rowsg.sort(reverse=True)
for nrm, nme, c, k in rowsg[:12]:
    print(f"    {nme:<48} n {k:>9}  |g| {nrm:9.3f}  cos {c:+.4f}")
res["per_tensor_top"] = [{"name": n, "cos": c, "numel": k, "norm": g}
                         for g, n, c, k in rowsg[:12]]

tot = sum(g ** 2 for g, _, _, _ in rowsg) ** 0.5
top = rowsg[0]
print(f"\n  the single largest block is {top[1]}, carrying "
      f"{100 * top[0] ** 2 / tot ** 2:.1f}% of the squared gradient norm")

with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(f"  wrote {OUT}")
