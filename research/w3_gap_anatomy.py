"""What is the 0.21 made of? Three read-only diagnostics, no training, no GPU.

The target is one trajectory from A to B, no candidate selection, scoring 0.50.
The best single-trajectory arm on record is the one-shot event_polar_4m_fc_v2
path with the additive arrival correction applied: 0.7283 on the contract
scorer, 0.7085 under research/degeneracy_panel.py's control. So roughly 0.21
separates it from chance, and every measurement so far has been one number,
which says nothing about where that 0.21 lives.

  D1  which of the 18 features carries it
      Per-feature detectability (RF on that feature alone) and leave-one-out
      damage (RF on the other 17), plus how each feature is wrong: mean shift
      in units of the human spread, and spread ratio.

  D2  marginals or the joint
      An oracle probe. Stretch each synthetic feature, independently, onto the
      human marginal for that feature, then rescore. If the AUC collapses to
      chance, the whole gap is that individual quantities are miscalibrated,
      which a generator can be made to fix. If it survives, the detector is
      reading combinations (fast AND straight, say) that no per-feature fix
      touches, and that is an architecture problem. This is a bound on where
      the work is, not a method anyone would ship.

  D3  bias or variance
      Also an oracle. Score all 32 candidates per request in pool_s42_k32.npz
      with a detector, keep the least detectable one per request, and read the
      resulting set. If best-of-32 approaches chance, the model already emits
      near-human paths sometimes and its problem is spread; if it stays near
      0.70, every candidate is wrong the same way and the model must change.
      Selection is disqualified as an answer to the single-trajectory target.
      It is used here only as an instrument, and only as an upper bound.

On the scoring recipe. D1 and D2 need the RF fitted on subsets of columns and
against reference matrices other than the default, which scoring.score_features
cannot express (its dispersion battery indexes all 18 names, and column subsets
are not zero-paddable: with 17 constant columns the sqrt(18) per-split feature
sample would usually contain no usable feature at all). So this file has its own
rf_auc, built from scoring's own constants, and _check_recipe asserts it
reproduces score_features to 1e-12 on the full 18 columns before any diagnostic
runs. Numbers from rf_auc on subsets are diagnostics and never ledger numbers.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_gap_anatomy.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

import scoring  # noqa: E402  (metric contract, imported never edited)
from degeneracy_panel import (JITTER_PX, build_reference,  # noqa: E402
                              features_with_jitter)
from features import FEATURE_NAMES, extract_features, resample_trajectory  # noqa: E402
from w3_fallback_arrival import correct_additive  # noqa: E402

CACHE = R / "research" / "w3_landing_cache.pkl"
POOL = R / "pool_s42_k32.npz"
POOL_CACHE = Path("/tmp/claude-1000/-home-aaronadmin/gap_anatomy_pool_X.npz")
OUT = R / "research" / "w3_gap_anatomy_results.json"

# research/w3_arrival_tax_control_results.json, the arm this anatomizes.
ARM_CONTRACT = 0.7283
ARM_CONTROL = 0.7085


def rf_auc(synth, human, seed=scoring.RF_SEED):
    """scoring.score_features' RF recipe, on arbitrary column subsets and an
    arbitrary reference matrix. Balance to min(n), human 0 / synth 1, OOB
    decision function AUC. _check_recipe pins this to score_features."""
    synth = np.asarray(synth, dtype=np.float64)
    human = np.asarray(human, dtype=np.float64)
    if synth.ndim == 1:
        synth, human = synth[:, None], human[:, None]
    n = min(len(human), len(synth))
    X = np.vstack([human[:n], synth[:n]])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    clf = RandomForestClassifier(n_estimators=scoring.RF_N_ESTIMATORS,
                                 oob_score=True, n_jobs=-1, random_state=seed)
    clf.fit(X, y)
    return float(roc_auc_score(y, clf.oob_decision_function_[:, 1]))


def _check_recipe(X_arm):
    """rf_auc must be score_features' recipe, not a lookalike."""
    contract = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    mine = rf_auc(X_arm, contract)
    theirs = scoring.score_features(X_arm)["auc_rf_oob"]
    print(f"[recipe] rf_auc {mine:.12f} vs score_features {theirs:.12f} "
          f"(delta {abs(mine - theirs):.2e})")
    assert abs(mine - theirs) < 1e-12, "local RF recipe has drifted from the contract"


def build_arm(specs, trajs):
    """The single-trajectory arm: one-shot path, additive arrival correction,
    which is what makes it land on the requested pixel."""
    out = []
    for spec, tr in zip(specs, trajs):
        sx, sy, ex, ey = (int(v) for v in spec)
        a = np.asarray(tr, dtype=np.float64)
        out.append(correct_additive(a, sx, sy, ex, ey) if len(a) >= 3 else a)
    return out


def quantile_match(synth, human):
    """Map each column of synth onto that column's human marginal, one column
    at a time, so every marginal matches exactly and only the dependence
    between columns is left as it was."""
    out = np.empty_like(synth)
    for j in range(synth.shape[1]):
        hs = np.sort(human[:, j])
        order = np.argsort(np.argsort(synth[:, j]))       # ranks, ties broken
        q = order / max(len(synth) - 1, 1)
        out[:, j] = np.interp(q, np.linspace(0.0, 1.0, len(hs)), hs)
    return out


def d1_where(X, ref, label, out):
    """Per-feature detectability, leave-one-out damage, and how each feature is
    wrong. Two reads of the same 18 columns, never mixed with each other."""
    full = rf_auc(X, ref)
    rows = []
    for j, name in enumerate(FEATURE_NAMES):
        alone = rf_auc(X[:, [j]], ref[:, [j]])
        keep = [k for k in range(len(FEATURE_NAMES)) if k != j]
        without = rf_auc(X[:, keep], ref[:, keep])
        h_sd = float(np.std(ref[:, j]))
        rows.append({
            "feature": name,
            "auc_alone": alone,
            "auc_without": without,
            "loo_drop": full - without,
            "mean_shift_sd": float((X[:, j].mean() - ref[:, j].mean())
                                   / max(h_sd, 1e-12)),
            "sd_ratio": float(np.std(X[:, j]) / max(h_sd, 1e-12)),
        })
    rows.sort(key=lambda r: -r["auc_alone"])

    print(f"\nD1  where the gap lives, {label} reading (all 18 together: "
          f"{full:.4f})")
    print(f"{'feature':<24}{'alone':>8}{'drop if cut':>13}{'mean off':>10}"
          f"{'spread':>9}")
    for r in rows:
        print(f"{r['feature']:<24}{r['auc_alone']:>8.4f}"
              f"{r['loo_drop']:>+13.4f}{r['mean_shift_sd']:>+10.2f}"
              f"{r['sd_ratio']:>9.2f}")
    out[f"d1_{label}"] = {"auc_all": full, "features": rows}
    return full


def d2_marginals(X, ref_fit, ref_score, label, out):
    """Oracle: force every marginal to match, then rescore against a reference
    the mapping never saw. Baseline uses the same held-out reference so the two
    numbers are comparable to each other, not to the ledger."""
    base = rf_auc(X, ref_score)
    matched = rf_auc(quantile_match(X, ref_fit), ref_score)
    print(f"\nD2  marginals or the joint, {label} reading")
    print(f"  as generated, vs held-out human half        {base:.4f}")
    print(f"  every marginal forced onto the human one    {matched:.4f}")
    print(f"  gap above chance closed by marginals alone  "
          f"{(base - matched) / max(base - 0.5, 1e-9):.0%}")
    out[f"d2_{label}"] = {"auc_base": base, "auc_marginals_matched": matched,
                          "fraction_closed": float((base - matched)
                                                   / max(base - 0.5, 1e-9))}


def pool_features(pool_path):
    """Corrected-candidate features for the K=32 pool, cached: the correction
    plus extraction over ~64k candidates is the slow part of D3."""
    if POOL_CACHE.exists():
        d = np.load(POOL_CACHE)
        print(f"[d3] reusing cached pool features {POOL_CACHE}")
        return d["X"], d["owner"]
    # allow_pickle: this repo's own poolgen output (object-dtype trajs array)
    d = np.load(pool_path, allow_pickle=True)
    trajs, owner = d["trajs"], d["owner_idx"].astype(int)
    tgt = np.round(d["specs"]).astype(int)
    X = np.full((len(trajs), len(FEATURE_NAMES)), np.nan)
    t0 = time.time()
    for ci in range(len(trajs)):
        t = trajs[ci]
        if t is None or len(t) < 3:
            continue
        sx, sy, ex, ey = tgt[owner[ci]]
        f = extract_features(resample_trajectory(
            correct_additive(np.asarray(t), sx, sy, ex, ey)))
        if f is not None and np.all(np.isfinite(f)):
            X[ci] = f
    print(f"[d3] corrected and extracted {len(trajs):,} candidates in "
          f"{time.time()-t0:.0f}s")
    POOL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(POOL_CACHE, X=X, owner=owner)
    return X, owner


def d3_bias_or_variance(pool_path, ref_fit, seed, out):
    """Oracle: a detector picks the least detectable of each request's 32
    candidates. Upper bound on what any selector could do, so it bounds how
    much of the gap is spread rather than a consistent flaw."""
    X, owner = pool_features(pool_path)
    ok = np.all(np.isfinite(X), axis=1)
    specs = {}
    for ci in np.flatnonzero(ok):
        specs.setdefault(int(owner[ci]), []).append(ci)
    specs = {s: np.asarray(v) for s, v in specs.items() if len(v) >= 2}
    print(f"[d3] {ok.sum():,} valid candidates over {len(specs)} requests, "
          f"{np.mean([len(v) for v in specs.values()]):.1f} per request")

    rng = np.random.default_rng(seed)
    typical = np.array([rng.choice(v) for v in specs.values()])

    # the picker is fitted against a human half the final scorer does not use,
    # and the final read is the contract scorer against its own reference
    judge = RandomForestClassifier(n_estimators=scoring.RF_N_ESTIMATORS,
                                   n_jobs=-1, random_state=seed)
    judge.fit(np.vstack([ref_fit, X[typical]]),
              np.concatenate([np.zeros(len(ref_fit)), np.ones(len(typical))]))
    p = judge.predict_proba(X)[:, 1]

    best = np.array([v[np.argmin(p[v])] for v in specs.values()])
    worst = np.array([v[np.argmax(p[v])] for v in specs.values()])
    within = float(np.mean([p[v].max() - p[v].min() for v in specs.values()]))

    res = {}
    for name, rows in (("typical (one at random)", typical),
                       ("oracle best of 32", best),
                       ("oracle worst of 32", worst)):
        res[name] = float(scoring.score_features(X[rows])["auc_rf_oob"])
    res["within_request_score_spread"] = within
    res["across_request_score_spread"] = float(np.std(p[typical]))

    print("\nD3  bias or variance, contract scorer")
    for name in ("typical (one at random)", "oracle best of 32",
                 "oracle worst of 32"):
        print(f"  {name:<26}{res[name]:.4f}")
    print(f"  detector score spread within one request  {within:.3f}")
    print(f"  ... across requests                       "
          f"{res['across_request_score_spread']:.3f}")
    out["d3"] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(POOL))
    ap.add_argument("--n-ref", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-d3", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    # pickle.load: this repo's own artifact from the 2026-07-20 landing-price
    # run on this machine, never third-party input.
    with open(CACHE, "rb") as fh:
        specs, trajs = pickle.load(fh)
    arm = build_arm(specs, trajs)
    print(f"[arm] {len(arm)} one-shot paths, additive arrival correction "
          f"(recorded {ARM_CONTRACT:.4f} contract, {ARM_CONTROL:.4f} control)")

    X_plain = features_with_jitter(arm, 0.0, args.seed)
    X_nudge = features_with_jitter(arm, JITTER_PX, args.seed)
    keep = (np.all(np.isfinite(X_plain), axis=1)
            & np.all(np.isfinite(X_nudge), axis=1))
    X_plain, X_nudge = X_plain[keep], X_nudge[keep]
    print(f"[arm] {keep.sum()} paths survive extraction in both readings")

    contract_ref = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    ref_plain = build_reference(0.0, args.n_ref, args.seed)["X"]
    ref_nudge = build_reference(JITTER_PX, args.n_ref, args.seed)["X"]
    _check_recipe(X_plain)

    out = {"n_arm": int(keep.sum()), "n_ref": len(ref_plain),
           "seed": args.seed, "recorded": {"contract": ARM_CONTRACT,
                                           "control": ARM_CONTROL}}

    d1_where(X_plain, contract_ref, "contract", out)
    d1_where(X_nudge, ref_nudge, "control", out)

    # split the rebuilt reference: the quantile map is fitted on one half and
    # the rescore happens against the other, so a match cannot be memorisation
    h = len(ref_plain) // 2
    d2_marginals(X_plain, ref_plain[:h], ref_plain[h:], "contract", out)
    d2_marginals(X_nudge, ref_nudge[:h], ref_nudge[h:], "control", out)

    if not args.skip_d3:
        d3_bias_or_variance(args.pool, contract_ref, args.seed, out)

    out["wall_sec"] = time.time() - t0
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[anatomy] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
