"""w4_e1disp. AMENDMENT 50, registered in step0_prereg.md before this file
existed.

A49 tried to settle the spread question by rescaling feature matrices and
died twice, in both directions, because the eighteen contract features are
deterministic functions of one trajectory and a rescaled vector leaves the
surface real vectors live on. The classifier finds that departure long
before it finds anything about spread. The full failure is in the prereg.

This amendment obeys the constraint that failure imposed: every row scored
here is a real, unmodified feature vector, and dispersion is changed only by
choosing WHICH rows are in the set.

READ 1, secondary and explicitly NOT blind, costs no new scoring: the slope
of AUC on spread ratio across the five committed arms, within each seed.

READ 2, primary and blind: how much AUC a spread difference of this size
actually buys among real human rows, measured by building same sized
subsets of the human rows that differ only in dispersion.

CPU only, no generation. Diagnostic only, never a training signal, never a
serve candidate, no selection of trajectories.
"""
import json
import sys

import numpy as np

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import scoring   # noqa: E402
import ledger    # noqa: E402

SEEDS = list(range(40, 52))
K = 20
PERM_SEED = 3208
SUBSET_SEED = 7700
NSUB = 1200
NBINS = 10
KMAX = 4
TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
ANCHOR = "data/human_val_features_grpo.npy"
ARMS = {"HUM": "research/w4_e1floor_F_L0_RAW_s{s}.npy",
        "H02": "research/w4_e1feat_F_h02_s{s}.npy",
        "F25": "research/w4_e1feat_F_hf25_s{s}.npy",
        "F50": "research/w4_e1feat_F_hf50_s{s}.npy",
        "F75": "research/w4_e1feat_F_hf75_s{s}.npy"}
GAP = 0.118   # the A48 HUM to H02 spread gap this predicts a contribution for


def row_lengths(seed, lengths, held):
    """The exact w4_qladder pick rule, so row i here is row i there."""
    elig = held[lengths[held] > KMAX]
    pick = np.sort(np.random.default_rng(1000 + seed)
                   .choice(elig, 2000, replace=False))
    return lengths[pick]


def auc_mean(m):
    v = np.array([scoring.score_features(
        m[np.random.default_rng(PERM_SEED + k).permutation(len(m))])["auc_rf_oob"]
        for k in range(K)])
    return float(v.mean()), float(v.std(ddof=1))


def spread_ratio(m, asd):
    return float(np.mean(m.std(0) / asd))


def radius(m):
    """Mean absolute z score per row, standardised on this matrix itself."""
    z = (m - m.mean(0)) / m.std(0)
    return np.abs(z).mean(1)


def paired_t(d):
    d = np.asarray(d, dtype=float)
    mn = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return mn, se, (mn / se if se > 0 else float("inf"))


def main():
    A = np.load(ANCHOR)
    A = A[np.isfinite(A).all(1)]
    asd = A.std(0)

    lengths = np.load("training/events_len.npy")
    N = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(N, min(N, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(N), trained)

    prev = json.load(open("research/w4_e1frac.json"))["per_seed"]

    print(f"  every value is the mean of K={K} permutations,"
          f" {len(SEEDS)} seeds\n", flush=True)

    # ---------- READ 1: free, not blind ----------
    R = {a: {} for a in ARMS}
    hum = {}
    for s in SEEDS:
        mats = {a: np.load(t.format(s=s)) for a, t in ARMS.items()}
        ok = np.ones(len(mats["HUM"]), dtype=bool)
        for m in mats.values():
            ok &= np.isfinite(m).all(1)
        for a, m in mats.items():
            R[a][s] = spread_ratio(m[ok], asd)
        hum[s] = mats["HUM"][ok]

    print("  spread ratio R, mean over seeds (1.0 means it matches the"
          " anchor's spread):")
    print("    " + "   ".join(
        f"{a} {np.mean([R[a][s] for s in SEEDS]):.3f}" for a in ARMS))

    slopes = []
    for s in SEEDS:
        x = np.array([R[a][s] for a in ARMS])
        y = np.array([prev[a][str(s)] for a in ARMS])
        slopes.append(np.polyfit(x, y, 1)[0])
    sm, sse, st = paired_t(slopes)
    contrib1 = sm * GAP
    print(f"\n  READ 1 (SECONDARY, NOT BLIND, no new scoring): slope of AUC"
          f" on spread ratio across the five arms")
    print(f"    {sm:+.4f} AUC per unit of R  se {sse:.4f}  t {st:+.2f}")
    print(f"    over the {GAP:.3f} HUM to H02 spread gap that predicts"
          f" {contrib1:+.4f}, against the observed residual +0.0195")
    print(f"    the confound needs this NEGATIVE and large;"
          f" it is {'NEGATIVE' if sm < 0 else 'POSITIVE'}")

    # ---------- READ 2: primary, blind ----------
    print(f"\n  READ 2 (PRIMARY, BLIND): subsets of {NSUB} real unmodified"
          f" human rows differing only in dispersion", flush=True)
    sets = ["NARROW", "MID", "MID2", "WIDE"]
    auc = {k: {} for k in sets}
    rr = {k: {} for k in sets}
    ln = {"NARROW": {}, "WIDE": {}}
    for s in SEEDS:
        M = hum[s]
        Ls = row_lengths(s, lengths, held)
        ok = np.ones(len(np.load(ARMS["HUM"].format(s=s))), dtype=bool)
        for a, t in ARMS.items():
            ok &= np.isfinite(np.load(t.format(s=s))).all(1)
        Ls = Ls[ok]
        rng = np.random.default_rng(SUBSET_SEED + s)
        idx = {}
        # stratified on length, per the pre run correction: equal counts per
        # length bin and the radius computed within the bin, so the
        # dispersion selection cannot move the length distribution
        edges = np.quantile(Ls, np.linspace(0, 1, NBINS + 1))
        edges[-1] += 1
        b = np.clip(np.digitize(Ls, edges) - 1, 0, NBINS - 1)
        per = NSUB // NBINS
        nar, wid = [], []
        for k in range(NBINS):
            g = np.where(b == k)[0]
            r = radius(M[g])
            nar.append(g[np.argsort(r)[:per]])
            wid.append(rng.choice(g, per, replace=False, p=r / r.sum()))
        idx["NARROW"] = np.concatenate(nar)
        idx["WIDE"] = np.concatenate(wid)
        idx["MID"] = rng.choice(len(M), NSUB, replace=False)
        idx["MID2"] = rng.choice(len(M), NSUB, replace=False)
        for k in sets:
            sub = M[idx[k]]
            rr[k][s] = spread_ratio(sub, asd)
            auc[k][s], _ = auc_mean(sub)
        for k in ("NARROW", "WIDE"):
            ln[k][s] = float(Ls[idx[k]].mean())
        print(f"  seed {s}   " + "   ".join(
            f"{k} R {rr[k][s]:.3f} auc {auc[k][s]:.4f}" for k in sets)
            + f"   len {ln['NARROW'][s]:.1f}/{ln['WIDE'][s]:.1f}", flush=True)

    # GATE 1
    g1 = all(rr["NARROW"][s] < rr["MID"][s] < rr["WIDE"][s] for s in SEEDS)
    rep, repse, _ = paired_t([auc["MID"][s] - auc["MID2"][s] for s in SEEDS])
    g1b = abs(rep) <= 0.010
    print(f"\n  GATE 1: MID between NARROW and WIDE on R on every seed:"
          f" {'ok' if g1 else 'FAILED'}")
    print(f"          two independent uniform draws agree, {rep:+.4f}"
          f" (bar 0.010): {'ok' if g1b else 'FAILED'}")

    # GATE 2
    dl = [(ln["WIDE"][s] - ln["NARROW"][s]) / ln["NARROW"][s] for s in SEEDS]
    dlm = float(np.mean(dl))
    g2 = abs(dlm) <= 0.05
    print(f"  GATE 2: mean row length differs between WIDE and NARROW by"
          f" {dlm * 100:+.1f} percent (bar 5): {'ok' if g2 else 'FAILED'}")

    dauc, dse, dt = paired_t([auc["WIDE"][s] - auc["NARROW"][s]
                              for s in SEEDS])
    dR, dRse, _ = paired_t([rr["WIDE"][s] - rr["NARROW"][s] for s in SEEDS])
    slope2 = dauc / dR
    contrib2 = slope2 * GAP
    lo = (dauc - 1.96 * dse) / dR * GAP
    hi = (dauc + 1.96 * dse) / dR * GAP

    if abs(contrib2) >= 0.010 and contrib2 < 0:
        v = "CONFOUND REAL"
    elif abs(contrib2) >= 0.010 and contrib2 > 0:
        v = "WRONG SIGN"
    elif abs(contrib2) < 0.005:
        v = "CONFOUND TOO SMALL"
    else:
        v = "BETWEEN"

    print(f"\n  WIDE minus NARROW: AUC {dauc:+.4f} se {dse:.4f} t {dt:+.2f},"
          f" R {dR:+.3f}")
    print(f"  slope among real human rows {slope2:+.4f} AUC per unit of R")
    print(f"  predicted contribution over the {GAP:.3f} gap"
          f" {contrib2:+.4f}, 95 percent interval"
          f" [{min(lo, hi):+.4f}, {max(lo, hi):+.4f}]")
    print(f"  against the observed HUM to H02 residual of +0.0195")
    print(f"\n  READ 2: {v}")
    if not (g1 and g1b):
        print("  GATE 1 FAILED, the subsets are not ordered by dispersion,"
              " the read above is NOT interpretable (registered).")
        v = "VOID, gate 1"
    elif not g2:
        print("  GATE 2 FAILED, the dispersion selection also moved the row"
              " length distribution, which A47 showed moves the anchor"
              " comparison on its own. The read is CONFOUNDED BY LENGTH and"
              " is reported as such, not interpreted (registered).")
        v = "CONFOUNDED BY LENGTH"
    elif v == "CONFOUND TOO SMALL":
        print("  a spread difference the size of the HUM to H02 gap cannot"
              " carry the residual. The residuals against HUM stand as"
              " model error.")
    elif v == "WRONG SIGN":
        print("  spread differences of this size make a set MORE detectable,"
              " not less, so the A48 explanation is backwards and the"
              " residuals are if anything understated.")
    elif v == "CONFOUND REAL":
        print("  spread alone can carry a large part of the residual. Every"
              " residual against HUM since A41 has to be restated in spread"
              " corrected terms.")

    res = {"k": K, "seeds": SEEDS, "nsub": NSUB, "gap": GAP,
           "read1": {"slope": sm, "se": sse, "t": st, "contribution": contrib1,
                     "blind": False,
                     "R": {a: {str(s): R[a][s] for s in SEEDS} for a in ARMS}},
           "read2": {"verdict": v, "slope": slope2, "contribution": contrib2,
                     "ci": [min(lo, hi), max(lo, hi)],
                     "dauc": dauc, "dauc_se": dse, "dauc_t": dt, "dR": dR,
                     "gate1": bool(g1 and g1b), "gate2": bool(g2),
                     "length_gap_pct": dlm * 100, "mid_repeat": rep,
                     "auc": {k: {str(s): auc[k][s] for s in SEEDS}
                             for k in sets},
                     "R": {k: {str(s): rr[k][s] for s in SEEDS}
                           for k in sets}}}
    with open("research/w4_e1disp.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1disp.json")
    print(f"  all subsets are {NSUB} rows, not the 2000 the rest of the study"
          f" uses, so only the slope within this amendment may be read")
    print("  real unmodified rows throughout, no trajectory produced,"
          " diagnostic only, never a training signal, no selection")

    rid = ledger.append_row(
        "w4_e1disp",
        {"seeds": SEEDS, "n": NSUB, "k": K, "perm_seed": PERM_SEED,
         "subset_seed": SUBSET_SEED,
         "construction": "same size subsets of real human rows by dispersion",
         "reference": "w4_e1floor L0_RAW, corpus human rows"},
        "ok",
        metrics={"read1_slope": sm, "read1_t": st,
                 "read1_contribution": contrib1, "read2_slope": slope2,
                 "read2_contribution": contrib2, "read2_dauc": dauc,
                 "read2_dauc_t": dt, "gate1": int(g1 and g1b),
                 "gate2": int(g2), "length_gap_pct": dlm * 100},
        artifacts=["research/w4_e1disp.json"],
        notes=f"AMENDMENT 50 spread confound, real unmodified rows only."
              f" READ 2 (blind, primary) {v}: a spread gap of {GAP:.3f}"
              f" predicts {contrib2:+.4f} against an observed residual of"
              f" +0.0195. READ 1 (not blind) slope {sm:+.4f} t {st:+.2f}."
              f" Replaces AMENDMENT 49, whose rescaling method failed in"
              f" both directions. Diagnostic only, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
