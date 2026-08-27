"""w4_e1ceil. AMENDMENT 44, registered in step0_prereg.md before this
file existed.

The conditioning ceiling test. h0np1 draws event 1 nonparametrically
from the empirical p(e1 | e0, cond), so it carries no model error. If
it scores like H0P1 the pair head is already at the ceiling of what
its conditioning allows and only more conditioning helps. If it scores
below, the modelling route A43 appeared to close is open again.

h0np1 copies real human event 1 triples from OTHER rows, so it is a
BOUND and never a servable model. Scored the A40 way, K 20 row
permutations, paired on identical rows. CPU only, no generation,
diagnostic, never a training signal, no selection.
"""
import json
import sys

import numpy as np

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import scoring   # noqa: E402
import ledger    # noqa: E402

SEEDS = [40, 41, 42, 43, 44, 45]
K = 20
PERM_SEED = 3208
ARMS = {"HUM": "research/w4_e1floor_F_L0_RAW_s{s}.npy",
        "H0P1": "research/w4_e1feat_F_h0p1_s{s}.npy",
        "H0NP1": "research/w4_e1feat_F_h0np1_s{s}.npy",
        "H02": "research/w4_e1feat_F_h02_s{s}.npy"}


def auc_mean(path):
    m = np.load(path)
    m = m[np.isfinite(m).all(1)]
    v = np.array([scoring.score_features(
        m[np.random.default_rng(PERM_SEED + k).permutation(len(m))])["auc_rf_oob"]
        for k in range(K)])
    return float(v.mean()), float(v.std(ddof=1))


def verdict(mean, t):
    if abs(mean) < 0.010 and abs(t) < 2.0:
        return "AT THE FLOOR"
    if mean >= 0.010 and t >= 3.0:
        return "ABOVE THE FLOOR"
    return "BETWEEN"


def step_verdict(mean, t):
    if abs(mean) >= 0.005 and abs(t) >= 3.0:
        return "REAL"
    if abs(t) < 2.0:
        return "NULL"
    return "BETWEEN"


def paired(a, b, per):
    d = np.array([per[a][s] - per[b][s] for s in SEEDS])
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return m, se, (m / se if se > 0 else float("inf")), [float(x) for x in d]


def main():
    print(f"  every arm value is the mean of K={K} permutations\n", flush=True)
    per, sds = {a: {} for a in ARMS}, {a: {} for a in ARMS}
    for s in SEEDS:
        for a, tmpl in ARMS.items():
            per[a][s], sds[a][s] = auc_mean(tmpl.format(s=s))
        print("  seed %d: " % s + "  ".join(
            f"{a} {per[a][s]:.4f}" for a in ARMS), flush=True)

    res = {"k": K, "arm_is_a_bound_not_a_model": True,
           "per_seed": {a: {str(s): per[a][s] for s in SEEDS} for a in ARMS},
           "per_seed_sd": {a: {str(s): sds[a][s] for s in SEEDS} for a in ARMS}}

    m, se, t, d = paired("H0NP1", "H0P1", per)
    v = step_verdict(m, t)
    res["read1"] = dict(mean=m, se=se, t=t, verdict=v, per_seed=d)
    print(f"\n  READ 1 (PRIMARY): H0NP1 minus H0P1  mean {m:+.4f}  se {se:.4f}"
          f"  t {t:+.2f}   {v}")
    if v == "NULL":
        print("  THE PAIR HEAD IS AT THE CEILING of what (e0, cond) allows.")
        print("  No better model of p(e1 | e0, cond) helps. Only more"
              " conditioning does.")
    elif v == "REAL" and m < 0:
        print("  THE MODELLING ROUTE IS OPEN: a zero model error draw on the"
              " same conditioning beats the head.")
    elif v == "REAL" and m > 0:
        print("  THE READ IS VOID: the nonparametric draw is WORSE than the"
              " head, so the neighbourhood is too coarse to be a ceiling."
              " This is not a finding about the head (registered).")
    else:
        print("  BETWEEN, neither branch is supported at the registered bars.")

    m2, se2, t2, d2 = paired("H0NP1", "HUM", per)
    res["read2"] = dict(mean=m2, se=se2, t=t2, verdict=verdict(m2, t2),
                        per_seed=d2)
    print(f"\n  READ 2 (THE CEILING): H0NP1 minus HUM  mean {m2:+.4f}"
          f"  se {se2:.4f}  t {t2:+.2f}   {res['read2']['verdict']}")
    print(f"  a perfect conditional on (e0, cond) still sits {m2:+.4f} from"
          f" real human data")

    m3, se3, t3, d3 = paired("H0NP1", "H02", per)
    res["read3"] = dict(mean=m3, se=se3, t=t3, verdict=step_verdict(m3, t3),
                        per_seed=d3)
    print(f"\n  READ 3: H0NP1 minus H02  mean {m3:+.4f}  se {se3:.4f}"
          f"  t {t3:+.2f}   {res['read3']['verdict']}")

    with open("research/w4_e1ceil.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1ceil.json")
    print("  h0np1 copies real human event 1 triples from other rows: a BOUND,"
          " never a servable model, no serve decision may cite it")
    print("  one trajectory per row, no selection, diagnostic only,"
          " never a training signal, no headline from this")

    rid = ledger.append_row(
        "w4_e1ceil",
        {"seeds": SEEDS, "n": 2000, "k": K, "perm_seed": PERM_SEED,
         "arm": "h0np1", "reference": "w4_e1floor L0_RAW, corpus human rows"},
        "ok",
        metrics={"read1_mean": m, "read1_t": t, "read2_mean": m2,
                 "read2_t": t2, "read3_mean": m3, "read3_t": t3},
        artifacts=["research/w4_e1ceil.json"],
        notes=f"AMENDMENT 44 conditioning ceiling test. READ 1 H0NP1 minus"
              f" H0P1 {m:+.4f} t {t:+.2f} ({v}). Ceiling above human"
              f" {m2:+.4f} t {t2:+.2f}. h0np1 copies real human e1 from other"
              f" rows: a bound, never a servable model. Diagnostic only,"
              f" registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
