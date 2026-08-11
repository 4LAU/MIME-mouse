"""Split the remaining gap into marginal, second order, and beyond.

NOT PRE REGISTERED. Run exploratory on 2026-08-10 while w4_rollout was training,
to find out whether that objective had a ceiling and where it sat. It returns a
decomposition, not a verdict, and it is recorded as such. The marginal stage
reproduces w4_joint, which WAS pre registered on 2026-08-07, so that stage is a
replication and only the second order stage and the residual are new.

WHY

The contract scorer reads eighteen numbers and nothing else. So every objective
this workstream can state is a statement about the distribution of those
eighteen, and the only question that matters is which part of that distribution
carries the separation. w4_rollout matches the first two moments of twelve of
them. If moments are most of it, that arm has room. If they are not, that arm
has a ceiling and a longer run cannot help.

THE LADDER

Each rung imposes more of the corpus distribution on the generated features and
scores the result through the ordinary contract. The rungs are nested, so the
differences between them partition the gap.

  base   generated features untouched
  A      every marginal rank mapped onto the corpus, dependence untouched
  B      A plus the corpus correlation matrix, a Gaussian copula transport
  C      the corpus itself, the floor this training data can buy

A tells us what perfect marginal correction is worth. B minus A is what second
order dependence is worth. B minus C is what survives both, which is the part no
objective built from means, spreads and correlations can reach.

THE CONTROL IS INSIDE THE LADDER

A and B carry the corpus marginals EXACTLY, by construction. C is the corpus. So
rows A, B and C have the same eighteen marginals and any separation between them
is dependence and nothing else. That is a control the arm cannot fail to report.

Transports are applied to the generated set and scored against the fixed
reference inside scoring.py, so the two sides of every test are different
trajectories. A transport scored against a copy of its own source would break
the out of bag forest and read near zero rather than near a half.

CPU only. Reuses trajectories already sampled one per command with no selection.
The protected eval sample is never read and no model file is written.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from scipy.special import ndtri

os.environ.setdefault("EVENT_SNAP", "2.5")
for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
import scoring  # noqa: E402
from features import FEATURE_NAMES, extract_feature_matrix  # noqa: E402
from models.event_ar import class_to_dt_ms  # noqa: E402
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TH_NULL_CLASS, TICK_CLASS, dth_lattice_to_class, s2_to_class,
)

D = "training"
TOK = ("/tmp/claude-1000/-home-aaronadmin/"
       "059c9656-a421-4ab6-9053-614d1dc15765/scratchpad/gen_tokens.npz")
OUT = "research/w4_gapsplit.json"
CAP = 160                 # the length w4_rollout evaluates at
SEEDS = (17, 23, 31)      # corpus subsample and scoring shuffle
N_HUMAN = 3000


def motion(s):
    return (s > TICK_CLASS) & (s < S_PAD_CLASS)


def feats(rows):
    """Token streams to features through the SERVED decoder. A hand rolled walk
    reads about 0.13 high, so esp._decode is the only route allowed."""
    paths = []
    for s, th, dt, a in rows:
        ms = class_to_dt_ms(torch.from_numpy(dt)).numpy().astype(np.float64)
        dz = (np.log(np.maximum(ms, 0.05)) - esp._DT_MEAN) / esp._DT_STD
        p = esp._decode(dz, s, th, 0.0, 0.0, a)
        if p is not None and len(p) >= 4:
            paths.append(np.asarray(p, dtype=np.float64))
    X = extract_feature_matrix(paths)
    return X[np.all(np.isfinite(X), 1)]


def load_gen():
    z = np.load(TOK)
    S, TH, DT, idx, cond = z["s"], z["th"], z["dt"], z["idx"], z["cond"]
    ang = np.arctan2(cond[:, 3].astype(np.float64), cond[:, 2].astype(np.float64))
    rows = []
    for j in range(len(cond)):
        s = S[j].astype(np.int64)
        pad = s >= S_PAD_CLASS
        L = min(int(pad.argmax()) if pad.any() else len(s), CAP)
        if L < 8 or motion(s[:L]).sum() < 8:
            continue
        rows.append([s[:L], TH[j, :L].astype(np.int64),
                     DT[j, :L].astype(np.int64), float(ang[j])])
    return rows, set(int(v) for v in idx)


def load_hum(seed, used):
    s2a = np.load(f"{D}/events_s2.npy", mmap_mode="r")
    dtha = np.load(f"{D}/events_dth.npy", mmap_mode="r")
    dta = np.load(f"{D}/events_dt.npy", mmap_mode="r")
    lens = np.load(f"{D}/events_len.npy")
    cond_all = np.load(f"{D}/events_cond.npy", mmap_mode="r")
    ok = np.flatnonzero(np.load(f"{D}/events_feat18_ok.npy"))
    pool = [v for v in ok[np.random.default_rng(seed).permutation(len(ok))]
            if int(v) not in used][:N_HUMAN * 3]
    rows = []
    for j in pool:
        L = min(int(lens[j]), CAP)
        if L < 8:
            continue
        s2 = torch.from_numpy(np.asarray(s2a[j, :L]).astype(np.int64))
        dth = torch.from_numpy(np.asarray(dtha[j, :L]).astype(np.int64))
        s_c = s2_to_class(s2).numpy()
        if motion(s_c).sum() < 8:
            continue
        th_c = torch.where(s2 > 0, dth_lattice_to_class(dth),
                           torch.full_like(dth, TH_NULL_CLASS)).numpy()
        dt_c = np.round(np.asarray(dta[j, :L]).astype(np.float64)
                        ).clip(0, 150).astype(np.int64)
        c = np.asarray(cond_all[j], dtype=np.float64)
        rows.append([s_c, th_c, dt_c, float(np.arctan2(c[3], c[2]))])
        if len(rows) >= N_HUMAN:
            break
    return rows


def to_normal(X):
    """Per column, ranks pushed onto a standard normal. The copula of X."""
    r = np.argsort(np.argsort(X, 0), 0) + 1.0
    return ndtri(r / (len(X) + 1.0))


def from_normal(Z, target):
    """Per column, normal scores pushed back onto the target's own values, so
    the result carries the target marginal exactly."""
    out = np.empty_like(Z)
    for k in range(Z.shape[1]):
        q = (np.argsort(np.argsort(Z[:, k])) + 1.0) / (len(Z) + 1.0)
        out[:, k] = np.quantile(target[:, k], q)
    return out


def score(X, seed, imp=None):
    X = np.ascontiguousarray(X, dtype=np.float64)
    X = X[np.all(np.isfinite(X), 1)]
    # the corpus is ordered by session, so an unshuffled prefix scores a narrow
    # band of people and reads 0.02 to 0.03 high
    np.random.default_rng(seed).shuffle(X)
    r = scoring.score_features(X)
    # which features the forest reaches for at each rung. At rung B the
    # marginals and the correlation matrix are already human, so whatever it
    # still uses is where the dependence beyond second order lives
    if imp is not None:
        imp.append([float(r["importances"][f]) for f in FEATURE_NAMES])
    return float(r["auc_rf_oob"])


def mean_se(v):
    v = np.asarray(v, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


gen_rows, used = load_gen()
G = feats(gen_rows)
runs = {"base": [], "A_marginal": [], "B_plus_corr": [], "C_corpus": []}
imps = {k: [] for k in runs}
corr_err = []

for seed in SEEDS:
    H = feats(load_hum(seed, used))
    ZG, ZH = to_normal(G), to_normal(H)
    cg = np.cov(ZG, rowvar=False) + 1e-6 * np.eye(len(FEATURE_NAMES))
    ch = np.cov(ZH, rowvar=False) + 1e-6 * np.eye(len(FEATURE_NAMES))
    # whiten the generated copula, recolour it with the corpus one
    ZB = (ZG - ZG.mean(0)) @ np.linalg.inv(np.linalg.cholesky(cg)).T \
        @ np.linalg.cholesky(ch).T
    runs["base"].append(score(G, seed, imps["base"]))
    runs["A_marginal"].append(score(from_normal(ZG, H), seed, imps["A_marginal"]))
    runs["B_plus_corr"].append(score(from_normal(ZB, H), seed, imps["B_plus_corr"]))
    runs["C_corpus"].append(score(H, seed, imps["C_corpus"]))
    corr_err.append(np.corrcoef(ZG, rowvar=False) - np.corrcoef(ZH, rowvar=False))
    print(f"  seed {seed}  base {runs['base'][-1]:.4f}  A {runs['A_marginal'][-1]:.4f}"
          f"  B {runs['B_plus_corr'][-1]:.4f}  C {runs['C_corpus'][-1]:.4f}",
          flush=True)

out = {k: dict(zip(("mean", "se"), mean_se(v))) for k, v in runs.items()}
out["n_gen"] = int(len(G))
out["seeds"] = list(SEEDS)
out["cap"] = CAP
b, a = out["base"]["mean"], out["A_marginal"]["mean"]
bb, c = out["B_plus_corr"]["mean"], out["C_corpus"]["mean"]
out["split"] = {"marginal": b - a, "second_order": a - bb,
                "beyond_second_order": bb - c, "whole_gap": b - c}

E = np.abs(np.mean(corr_err, axis=0))
iu = np.triu_indices(len(FEATURE_NAMES), 1)
worst = np.argsort(-E[iu])[:10]
out["importances"] = {k: dict(zip(FEATURE_NAMES,
                                  np.mean(v, axis=0).round(4).tolist()))
                      for k, v in imps.items()}
out["worst_correlation_errors"] = [
    {"a": FEATURE_NAMES[iu[0][w]], "b": FEATURE_NAMES[iu[1][w]],
     "abs_error": float(E[iu[0][w], iu[1][w]])} for w in worst]

print()
for k in ("base", "A_marginal", "B_plus_corr", "C_corpus"):
    print(f"  {k:<16}{out[k]['mean']:>8.4f}  se {out[k]['se']:.4f}")
print()
for k, v in out["split"].items():
    print(f"  {k:<22}{v:+.4f}")
print("\n  what the forest reaches for at each rung, top five")
for k in ("base", "A_marginal", "B_plus_corr", "C_corpus"):
    top = sorted(out["importances"][k].items(), key=lambda kv: -kv[1])[:5]
    print(f"    {k:<14}" + "  ".join(f"{n} {v:.3f}" for n, v in top))
print("\n  largest correlation errors, generated against corpus")
for r in out["worst_correlation_errors"][:6]:
    print(f"    {r['a']:<24}{r['b']:<24}{r['abs_error']:.3f}")

with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  wrote {OUT}")
