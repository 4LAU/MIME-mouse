"""How much of the contract AUC could any marginal matching objective ever
remove? Registered before it ran.

WHAT FORCED THIS

`w4_residual` established that the gradient cosine reads how far the model is
from its target rather than the shape of the objective. Same objective, same
batch statistic, same tokens, only the target moved:

    moment objective, target 0        +0.0119 se 0.0075
    moment objective, target 1 sd     +0.1542 se 0.0482
    moment objective, target 4 sd     +0.1998 se 0.0518
    paired, 4 sd against 0            +0.1879 se 0.0526     3.6 se

A batch statistic reaches three times the per row target's cosine when its
target is moved away from where the model already sits, so the per row lead is
withdrawn and so is the mechanism written up for it. The same run also showed a
per row target against a randomly drawn human row scoring higher than against
the row the generator was actually conditioned on, +0.0904 against +0.0667,
which is the residual ordering again and not a correspondence effect.

WHAT THAT LEAVES, AND WHY IT POINTS HERE

The measured distances are small. Across the nine trained features the generated
batch mean sits 0.10 standard deviations from the human mean and the generated
log standard deviation sits 0.26 from the human one. Every rollout arm in this
programme has optimised exactly those quantities. If they are nearly matched
already then the arms were not failing to optimise, there was little there to
optimise, and the separation the contract scorer sees lives somewhere the
objective never looks.

That is a claim about headroom and it can be measured directly, without any
training, by asking what the scorer would say if the marginals were matched
perfectly.

THE TEST

Take the generated features this checkpoint actually produces and transform them
per feature so that their marginal distribution matches the human one, then
rescore. The transform is a cheat no model can perform, which is the point: it
is an upper bound on what any objective controlling those marginals could
achieve, reached for free.

  0  baseline, untouched
  1  affine match on the twelve trained features. Mean and standard deviation
     set exactly to the human corpus. This is the ceiling for every arm this
     programme has run.
  2  quantile match on the twelve trained features. The whole marginal, not
     just its first two moments.
  3  quantile match on all eighteen. The ceiling for any marginal only method,
     including ones that reach the six held out features.

The match target is the training corpus, `training/events_feat18.npy`, not the
reference the scorer holds. A model could in principle be trained against the
training corpus, and whether matching it transfers to a fresh human sample is
part of what is being measured rather than something to assume away.

PREDICTION: baseline minus arm 1, paired across folds, is below +0.02 and its
two se band excludes +0.05. Perfect moment matching of the trained features buys
almost nothing, and the seven arm plateau is a headroom fact rather than an
optimisation failure.

FALSIFIER: baseline minus arm 1 is above +0.05 by at least 2 se. That would say
moment matching has real headroom the arms failed to reach, that the problem is
optimisation after all, and that the residual result explains why the gradient
is noisy without excusing the arms from having failed.

WHAT THIS CANNOT SETTLE. It bounds what marginal matching can do; it says
nothing about whether a model can be built that matches the joint structure. A
confirmed prediction redirects the programme away from every objective it has
tried so far and does not by itself supply the replacement.

Uses `research/autoloop/scoring.py`, the contract scorer. The protected eval
sample is never read and no model file is written. CPU only.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

from features import FEATURE_NAMES  # noqa: E402
from scoring import score_features  # noqa: E402
from w4_rollout import TRAINED  # noqa: E402

D = "training"
OUT = "research/w4_marginal.json"
SEED = 17
BATCH = 96
CAP = 256
NFOLD = 5
SCR = os.environ.get("W4_CACHE", "/tmp/w4_cache")

rng = np.random.default_rng(SEED)
tcols = [FEATURE_NAMES.index(n) for n in TRAINED]
acols = list(range(18))

ok = np.flatnonzero(np.load(f"{D}/events_feat18_ok.npy"))
Hall = np.load(f"{D}/events_feat18.npy", mmap_mode="r")
HT = np.asarray(Hall[np.sort(rng.choice(ok, 40000, replace=False))],
                dtype=np.float64)
HT = HT[np.isfinite(HT).all(1)]

gen = []
for k in range(64):
    p = f"{SCR}/softdec_tau_b{k}_{BATCH}_{CAP}.npz"
    if os.path.exists(p):
        gen.append(np.asarray(np.load(p)["Xh"], dtype=np.float64))
gen = np.concatenate(gen, 0)
gen = gen[np.isfinite(gen).all(1)]
gen = gen[rng.permutation(len(gen))]


def affine(g, cols):
    """Set each column's mean and standard deviation to the human corpus."""
    out = g.copy()
    for c in cols:
        s = g[:, c].std()
        if s > 0:
            out[:, c] = (g[:, c] - g[:, c].mean()) / s * HT[:, c].std() \
                + HT[:, c].mean()
    return out


def quantile(g, cols):
    """Replace each column by the human value at the same rank, so the whole
    marginal matches and not only its first two moments."""
    out = g.copy()
    n = len(g)
    for c in cols:
        hs = np.sort(HT[:, c])
        r = np.empty(n)
        r[np.argsort(g[:, c])] = np.arange(n)
        u = (r + 0.5) / n
        out[:, c] = np.interp(u, (np.arange(len(hs)) + 0.5) / len(hs), hs)
    return out


ARMS = (("baseline", lambda g: g),
        ("affine, 12 trained", lambda g: affine(g, tcols)),
        ("quantile, 12 trained", lambda g: quantile(g, tcols)),
        ("quantile, all 18", lambda g: quantile(g, acols)))

print(f"\n  {len(gen)} generated rows from the served decoder, "
      f"{NFOLD} disjoint folds")
print(f"  match target is the training corpus, {len(HT)} human rows")
print(f"  scored against the contract reference\n", flush=True)

folds = np.array_split(np.arange(len(gen)), NFOLD)
S = {}
for name, fn in ARMS:
    per = [score_features(fn(gen[f]))["auc_rf_oob"] for f in folds]
    S[name] = np.array(per)
    print(f"  {name:<22} {S[name].mean():.4f}   fold sd "
          f"{S[name].std(ddof=1):.4f}   "
          f"[{S[name].min():.4f}, {S[name].max():.4f}]", flush=True)

full = {name: score_features(fn(gen))["auc_rf_oob"] for name, fn in ARMS}
print(f"\n  on all {len(gen)} rows at once")
for name, _ in ARMS:
    print(f"    {name:<22} {full[name]:.4f}")


def paired(a, b, label):
    d = S[a] - S[b]
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(NFOLD))
    print(f"\n  {label}")
    print(f"    {m:+.4f}   se {se:.4f}   {m / max(se, 1e-9):.1f} se   "
          f"2 se band [{m - 2 * se:+.4f}, {m + 2 * se:+.4f}]")
    return m, se


gm, gse = paired("baseline", "affine, 12 trained",
                 "GATE  what perfect moment matching of the trained twelve buys")
q1 = paired("baseline", "quantile, 12 trained",
            "secondary  the whole marginal of the trained twelve")
q2 = paired("baseline", "quantile, all 18",
            "secondary  every marginal, the ceiling for any marginal method")

met = gm < 0.02 and gm + 2 * gse < 0.05
fals = gm > 0.05 and gm - 2 * gse > 0.05
print(f"\n  PREDICTION {'MET' if met else 'NOT MET'}    "
      f"FALSIFIER {'TRIGGERED' if fals else 'not met'}")
if met:
    print("  matching the trained feature moments perfectly, for free, removes "
          "almost none of the separation. the plateau is headroom and not "
          "optimisation, and every objective this programme has run was capped "
          "before it started.")
elif fals:
    print("  moment matching has real headroom the arms did not reach. the "
          "problem is optimisation and the arms failed at it.")

res = {"n_gen": int(len(gen)), "nfold": NFOLD,
       "folds": {k: v.tolist() for k, v in S.items()},
       "full": full,
       "gate_baseline_minus_affine12": {"mean": gm, "se": gse},
       "baseline_minus_quantile12": {"mean": q1[0], "se": q1[1]},
       "baseline_minus_quantile18": {"mean": q2[0], "se": q2[1]},
       "met": bool(met), "falsified": bool(fals)}
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(f"\n  wrote {OUT}")
