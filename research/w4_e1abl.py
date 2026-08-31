"""w4_e1abl. AMENDMENT 63 BRANCH C, registered in step0_prereg.md before
this file existed and, more importantly, WRITTEN BEFORE AMENDMENT 62 HAD
PRODUCED A SINGLE NUMBER. A62's generation was on its second of twelve
seeds when this was written. Writing the branch C reader now is the only
way to be sure its design was not shaped by the result that decides
whether it runs at all.

THE QUESTION. A61 and A62 read IMPORTANCE SHARES, which describe how a
forest fits, not what produces an AUC gap. A feature can track the
ordering without causing it. This asks the causal shaped version that is
actually available: drop each of the eighteen features in turn and see
whether dropping num_direction_changes shrinks the k0 minus zero band
gap by more than dropping anything else does.

THIS IS NOT THE CONTRACT SCORER AND EVERY NUMBER HERE IS A NON CONTRACT
DIAGNOSTIC AUC. scoring.score_features loads the eighteen column anchor
itself, so a seventeen column matrix cannot be handed to it, and
substituting a column rather than dropping it would edit a contract
matrix and score it, which AMENDMENT 49 forbids. So the forest is built
here to the identical recipe and gated on reproducing the contract AUC
EXACTLY on all eighteen columns. Nothing this file prints may be
compared to the 0.5795 headline or to any contract AUC anywhere else.

CPU only, no torch, no generation. Twelve fresh seeds 64 to 75.
Diagnostic only, never a training signal, never a serve candidate, no
selection, one trajectory per row.
"""
import json
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import scoring   # noqa: E402
import ledger    # noqa: E402
from features import FEATURE_NAMES   # noqa: E402

SEEDS = list(range(64, 76))
K = 20
PERM_SEED = 3208
N = 2000
MAX_T = 256
KMAX = 4
TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
ANCHOR = "data/human_val_features_grpo.npy"
FLOOR = "research/w4_e1floor_F_{a}_s{s}.npy"
ARM = "research/w4_e1feat_F_{a}_s{s}.npy"
ZERO = "L3_FULL"
K0 = "k0"
BAND = ("17 to 32", 17, 32)
FULL = ("full", 0, MAX_T)
TARGET = "num_direction_changes"
GATE_JSON = "research/w4_e1ndc.json"
CACHE = "research/w4_e1abl_cache.json"
CAL_TOL = 1e-9
SEED_BAR = 10          # of twelve, registered


def contract_auc(synth, anchor, cols=None):
    """the contract recipe, reproduced. cols=None means all eighteen."""
    n_use = min(len(anchor), len(synth))
    h = anchor[:n_use]
    s = np.asarray(synth, dtype=np.float64)[:n_use]
    if cols is not None:
        h, s = h[:, cols], s[:, cols]
    X = np.vstack([h, s])
    y = np.concatenate([np.zeros(n_use), np.ones(n_use)])
    clf = RandomForestClassifier(n_estimators=100, oob_score=True,
                                 n_jobs=-1, random_state=42)
    clf.fit(X, y)
    return float(roc_auc_score(y, clf.oob_decision_function_[:, 1]))


def perm(m, k):
    return m[np.random.default_rng(PERM_SEED + k).permutation(len(m))]


def paired(v):
    d = np.asarray(v, float)
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return m, se, (m / se if se > 0 else float("inf"))


def main():
    # BRANCH GUARD. This file may only run if AMENDMENT 62 returned
    # CONFIRMED. The registration fixed one branch per outcome and the
    # guard is here so the branch cannot be entered by hand.
    try:
        j62 = json.load(open(GATE_JSON))
        call = j62["decision"]
    except FileNotFoundError:
        print("  AMENDMENT 62 has not run. Branch C does not open.")
        return
    if not call.startswith("CONFIRMED"):
        print(f"  AMENDMENT 62 returned: {call}")
        print("  Branch C runs only on CONFIRMED. Registered branches K"
              " and U cover the other outcomes. Nothing is read here.")
        return

    print(f"  AMENDMENT 63 BRANCH C. Twelve fresh seeds {SEEDS[0]} to"
          f" {SEEDS[-1]}, K={K} permutations, PERM_SEED {PERM_SEED}.")
    print(f"  EVERY AUC BELOW IS A NON CONTRACT DIAGNOSTIC AUC. It may not"
          f" be compared to the headline or to any contract number.\n",
          flush=True)

    anchor = np.load(ANCHOR)
    lengths = np.load("training/events_len.npy")
    NT = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(NT, min(NT, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(NT), trained)
    elig = held[lengths[held] > KMAX]

    mats, per_ok, per_len = {}, {}, {}
    for s in SEEDS:
        m = {ZERO: np.load(FLOOR.format(a=ZERO, s=s)),
             K0: np.load(ARM.format(a=K0, s=s))}
        ok = np.ones(len(m[ZERO]), dtype=bool)
        for v in m.values():
            ok &= np.isfinite(v).all(1)
        pick = np.sort(np.random.default_rng(1000 + s).choice(elig, N,
                                                              replace=False))
        mats[s], per_ok[s] = m, ok
        per_len[s] = np.minimum(lengths[pick], MAX_T).astype(np.int64)

    def rows(s, lo, hi):
        return np.flatnonzero(per_ok[s] & (per_len[s] >= lo)
                              & (per_len[s] <= hi))

    # CALIBRATION GATE, registered as exact. One permutation, every seed,
    # both row sets, both arms, all eighteen columns.
    print("  CALIBRATION GATE, my forest against scoring.score_features,"
          " all eighteen columns, tolerance 1e-9:")
    worst = 0.0
    for band in (BAND, FULL):
        for s in SEEDS:
            ix = rows(s, band[1], band[2])
            for a in (ZERO, K0):
                m0 = perm(mats[s][a][ix], 0)
                mine = contract_auc(m0, anchor)
                theirs = scoring.score_features(m0)["auc_rf_oob"]
                worst = max(worst, abs(mine - theirs))
    ok_cal = worst <= CAL_TOL
    print(f"    worst absolute difference {worst:.3e}"
          f"   {'ok, GATE PASSED' if ok_cal else 'MISS, BRANCH C IS VOID (registered)'}",
          flush=True)
    if not ok_cal:
        print("    My forest is not the contract scorer's forest, so the"
              " ablation ranking means nothing. Reporting the mismatch"
              " rather than loosening the tolerance.")
        return

    # CHECKPOINT. Three consecutive segfaults inside libc, at different
    # addresses and different elapsed times, with 12 GB free and no OOM,
    # killed this run before any ablation finished. I do not know the
    # cause and am not going to invent one. Every completed cell is
    # written to disk so a retry resumes instead of restarting, which
    # makes the run finish without needing to know why it dies. The cache
    # is keyed by row set and dropped feature only, so it can never carry
    # a value computed under different settings into a different design.
    try:
        cache = json.load(open(CACHE))
    except FileNotFoundError:
        cache = {}

    def gaps(band, cols, tag):
        """per seed mean of (k0 minus zero) over K permutations."""
        key = f"{band[0]}|{tag}"
        if key in cache:
            return np.asarray(cache[key])
        out = []
        for s in SEEDS:
            ix = rows(s, band[1], band[2])
            d = []
            for k in range(K):
                z = contract_auc(perm(mats[s][ZERO][ix], k), anchor, cols)
                a = contract_auc(perm(mats[s][K0][ix], k), anchor, cols)
                d.append(a - z)
            out.append(float(np.mean(d)))
        cache[key] = out
        with open(CACHE, "w") as fh:
            json.dump(cache, fh)
        print(f"      cell done: {key}", flush=True)
        return np.asarray(out)

    results, ranks = {}, {}
    for band in (BAND, FULL):
        base = gaps(band, None, 'ALL')
        bm, bse, bt = paired(base.tolist())
        print(f"\n  {band[0]} row set. Baseline gap on all eighteen"
              f" columns: {bm:+.4f} se {bse:.4f} t {bt:+.2f}")
        shrink = {}
        for j, name in enumerate(FEATURE_NAMES):
            cols = [c for c in range(len(FEATURE_NAMES)) if c != j]
            g = gaps(band, cols, name)
            shrink[name] = np.abs(base) - np.abs(g)
        order = sorted(FEATURE_NAMES, key=lambda n: -float(shrink[n].mean()))
        print(f"    {'dropped feature':<26}{'shrink':>9}{'se':>8}{'t':>8}"
              f"  seeds first")
        # per seed winner, the registered count
        stacked = np.vstack([shrink[n] for n in FEATURE_NAMES])
        winner = [FEATURE_NAMES[int(np.argmax(stacked[:, i]))]
                  for i in range(len(SEEDS))]
        n_first = sum(1 for w in winner if w == TARGET)
        for name in order:
            m, se, t = paired(shrink[name].tolist())
            cnt = sum(1 for w in winner if w == name)
            print(f"    {name:<26}{m:+9.4f}{se:8.4f}{t:+8.2f}  {cnt:>2}/12",
                  flush=True)
        results[band[0]] = dict(
            baseline=dict(mean=bm, se=bse, t=bt),
            shrink={n: dict(zip(("mean", "se", "t"),
                                paired(shrink[n].tolist())))
                    for n in FEATURE_NAMES},
            winner_per_seed=winner, n_first_target=n_first,
            order=order)
        ranks[band[0]] = n_first

    n_band = ranks[BAND[0]]
    if n_band >= SEED_BAR:
        call63 = (f"SUPPORTED, {TARGET} shrinks the band gap more than any"
                  f" other single feature in {n_band} of 12 seeds")
    else:
        call63 = (f"NOT SUPPORTED, {TARGET} leads in only {n_band} of 12"
                  f" seeds against a registered bar of {SEED_BAR}. The"
                  f" feature tracks the ordering without producing it,"
                  f" which is weaker than A61 and A62 alone suggest.")
    print(f"\n  REGISTERED DECISION (band {BAND[0]}): {call63}")
    print("  There is no partial credit for ranking second (registered).")

    res = dict(k=K, seeds=SEEDS, target=TARGET, seed_bar=SEED_BAR,
               calibration_worst=worst, results=results, call=call63,
               non_contract_auc=True)
    with open("research/w4_e1abl.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1abl.json")
    print("  one trajectory per row, no selection, diagnostic only,"
          " never a training signal, no serve decision")

    rid = ledger.append_row(
        "w4_e1abl",
        {"seeds": SEEDS, "k": K, "perm_seed": PERM_SEED, "zero": ZERO,
         "arm": K0, "band": BAND[0], "target": TARGET,
         "reference": "A63 branch C, ablation on the gap, non contract AUC"},
        "ok",
        metrics={"calibration_worst": worst,
                 "n_first_band": n_band,
                 "n_first_full": ranks[FULL[0]]},
        artifacts=["research/w4_e1abl.json"],
        notes=f"AMENDMENT 63 BRANCH C. Leave one feature out ablation on"
              f" the k0 minus zero gap, twelve fresh seeds. {TARGET} leads"
              f" in {n_band} of 12 seeds in the band against a registered"
              f" bar of {SEED_BAR}. NON CONTRACT diagnostic AUC, never"
              f" comparable to the headline. Diagnostic only.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
