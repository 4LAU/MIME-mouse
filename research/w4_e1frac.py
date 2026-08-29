"""w4_e1frac. AMENDMENT 48, registered in step0_prereg.md before this file
existed.

A47 killed front loading and could not separate two remaining shapes: is the
residual spread uniformly across the movement, or is it concentrated in the
events the model generates LAST. This reads that directly.

The arms force the first ceil(f L) events of EACH row, a fixed fraction of
that row rather than a fixed count, for f = 0.25, 0.50, 0.75. Two reasons,
both of which fix a fault A47 recorded against itself. No row can saturate,
because ceil(f L) is strictly below L for every f below 1 once L is 5 or
more and eligibility is L above 4, so there is no confound to filter and the
contract scorer's fixed human anchor still matches the synthetic length
distribution. And the share of the movement the model still generates is
1 minus f for every row, which is exactly the quantity the uniform
hypothesis is about, instead of varying row by row as it does under a fixed
count.

UNIFORM says the residual is proportional to that share. TAIL says forcing
more of the opening leaves it roughly unchanged. The primary read is the
PAIRED DIFFERENCE between the observed residual at f = 0.75 and what
proportionality predicts from each seed's own baseline, never a ratio: see
the A47 lesson in the prereg.

Scored the A40 way, K 20 row permutations, paired on identical rows, twelve
seeds per the A46 row draw rule. CPU only, no generation. Diagnostic only,
never a training signal, no selection, no serve decision.
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
KMAX = 4
TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
FRACS = [0.25, 0.50, 0.75]
PRIMARY_F = 0.75
ARMS = {"HUM": "research/w4_e1floor_F_L0_RAW_s{s}.npy",
        "H02": "research/w4_e1feat_F_h02_s{s}.npy",
        "F25": "research/w4_e1feat_F_hf25_s{s}.npy",
        "F50": "research/w4_e1feat_F_hf50_s{s}.npy",
        "F75": "research/w4_e1feat_F_hf75_s{s}.npy"}
ARM_OF = {0.25: "F25", 0.50: "F50", 0.75: "F75"}


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


def verdict(mean, t):
    if abs(mean) < 0.010 and abs(t) < 2.0:
        return "AT THE FLOOR"
    if mean >= 0.010 and t >= 3.0:
        return "ABOVE THE FLOOR"
    return "BETWEEN"


def paired(a, b):
    d = np.array([a[s] - b[s] for s in SEEDS])
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return m, se, (m / se if se > 0 else float("inf")), [float(x) for x in d]


def main():
    lengths = np.load("training/events_len.npy")
    N = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(N, min(N, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(N), trained)

    # the share of events the model still generates, averaged over rows,
    # exactly as the arms construct it
    g = {}
    for s in SEEDS:
        Ls = row_lengths(s, lengths, held).astype(np.float64)
        g.setdefault("H02", []).append(float((1.0 - 2.0 / Ls).mean()))
        for f in FRACS:
            g.setdefault(f, []).append(
                float((1.0 - np.ceil(f * Ls) / Ls).mean()))
    # keys stay strings: ledger.stable_hash sorts them and cannot
    # order a mix of float and str
    g = {str(k): float(np.mean(v)) for k, v in g.items()}

    print(f"  every arm value is the mean of K={K} permutations,"
          f" {len(SEEDS)} seeds\n", flush=True)
    print("  share of events the model still generates, averaged over rows:")
    print(f"    H02 {g['H02']:.3f}   " + "   ".join(
        f"f={f:.2f} {g[str(f)]:.3f}" for f in FRACS) + "\n", flush=True)

    per = {a: {} for a in ARMS}
    for s in SEEDS:
        mats = {a: np.load(t.format(s=s)) for a, t in ARMS.items()}
        ok = np.ones(len(mats["HUM"]), dtype=bool)
        for m in mats.values():
            ok &= np.isfinite(m).all(1)
        for a, m in mats.items():
            per[a][s], _ = auc_mean(m[ok])
        print(f"  seed {s} rows {int(ok.sum())}   "
              + "  ".join(f"{a} {per[a][s]:.4f}" for a in ARMS), flush=True)

    res = {"k": K, "seeds": SEEDS, "fracs": FRACS, "generated_share": g,
           "per_seed": {a: {str(s): per[a][s] for s in SEEDS} for a in ARMS}}

    # residual against HUM at every rung
    r = {}
    for name in ["H02"] + [ARM_OF[f] for f in FRACS]:
        m, se, t, d = paired(per[name], per["HUM"])
        r[name] = dict(mean=m, se=se, t=t, verdict=verdict(m, t), per_seed=d)
    res["ladder"] = r

    print("\n  THE LADDER, residual against HUM, with what proportionality"
          " predicts from the H02 baseline:")
    print(f"    H02 (baseline)  {r['H02']['mean']:+.4f}  se"
          f" {r['H02']['se']:.4f}  t {r['H02']['t']:+.2f}")
    for f in FRACS:
        a = ARM_OF[f]
        pred = r["H02"]["mean"] * g[str(f)] / g["H02"]
        print(f"    f={f:.2f}         {r[a]['mean']:+.4f}  se"
              f" {r[a]['se']:.4f}  t {r[a]['t']:+.2f}"
              f"    uniform predicts {pred:+.4f}   {r[a]['verdict']}")

    print("\n  VALIDITY GATE (read before anything else): the ladder must not"
          " RISE as more of the movement is forced")
    order = ["H02"] + [ARM_OF[f] for f in FRACS]
    gate_ok, gate_rows = True, []
    for lo, hi in zip(order, order[1:]):
        m, se, t, d = paired(per[hi], per[lo])
        bad = m >= 0.005 and t >= 3.0
        gate_ok = gate_ok and not bad
        gate_rows.append(dict(step=f"{hi} minus {lo}", mean=m, se=se, t=t,
                              violates=bool(bad)))
        print(f"    {hi} minus {lo}  {m:+.4f}  se {se:.4f}  t {t:+.2f}"
              f"   {'VIOLATION' if bad else 'ok'}")
    res["gate"] = {"passed": bool(gate_ok), "steps": gate_rows}
    if not gate_ok:
        print("  GATE FAILED: forcing more of the real movement made the"
              " trajectory LESS human, so the arms are mislabelled. The read"
              " is VOID and the correct action is to find the labelling bug"
              " (registered).")
        with open("research/w4_e1frac.json", "w") as fh:
            json.dump(res, fh, indent=1)
        return
    print("  GATE PASSED.")

    # PRIMARY: paired excess over the proportional prediction, per seed,
    # each seed using its OWN baseline. Registered as a difference.
    scale = g[str(PRIMARY_F)] / g["H02"]
    obs = {s: per[ARM_OF[PRIMARY_F]][s] - per["HUM"][s] for s in SEEDS}
    uni = {s: (per["H02"][s] - per["HUM"][s]) * scale for s in SEEDS}
    m, se, t, d = paired(obs, uni)
    if abs(t) < 2.0:
        shape = "UNIFORM"
        gloss = ("the residual is proportional to the share of the movement"
                 " the model generates, so the defect is spread evenly")
    elif m >= 0.005 and t >= 3.0:
        shape = "TAIL LOCATED"
        gloss = ("more residual survives than proportionality allows, so the"
                 " defect sits in the events the model generates last")
    elif m <= -0.005 and t <= -3.0:
        shape = "FRONT WEIGHTED"
        gloss = ("less survives than proportionality allows, which A47 makes"
                 " unlikely and which is evidence of a fault, not a finding")
    else:
        shape = "BETWEEN"
        gloss = "neither branch is supported at the registered bars"
    res["read1"] = dict(shape=shape, mean=m, se=se, t=t, scale=scale,
                        per_seed=d)
    print(f"\n  READ 1 (PRIMARY): observed r(0.75) minus the proportional"
          f" prediction  {m:+.4f}  se {se:.4f}  t {t:+.2f}")
    print(f"  {shape}: {gloss}")

    res["read2"] = r[ARM_OF[PRIMARY_F]]
    print(f"\n  READ 2: r(0.75) {r[ARM_OF[PRIMARY_F]]['mean']:+.4f}"
          f"  se {r[ARM_OF[PRIMARY_F]]['se']:.4f}"
          f"  t {r[ARM_OF[PRIMARY_F]]['t']:+.2f}"
          f"   {r[ARM_OF[PRIMARY_F]]['verdict']}")
    print("  that is the last quarter of each movement alone, with the first"
          " three quarters real")

    with open("research/w4_e1frac.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1frac.json")
    print("  one trajectory per row, no selection, diagnostic only,"
          " never a training signal, no headline from this")

    rid = ledger.append_row(
        "w4_e1frac",
        {"seeds": SEEDS, "n": 2000, "k": K, "perm_seed": PERM_SEED,
         "fracs": FRACS, "generated_share": g,
         "reference": "w4_e1floor L0_RAW, corpus human rows"},
        "ok",
        metrics={"read1_mean": m, "read1_se": se, "read1_t": t,
                 "gate_passed": int(gate_ok),
                 "r_h02": r["H02"]["mean"], "r_f25": r["F25"]["mean"],
                 "r_f50": r["F50"]["mean"], "r_f75": r["F75"]["mean"]},
        artifacts=["research/w4_e1frac.json"],
        notes=f"AMENDMENT 48 fraction ladder. Gate PASSED. Ladder"
              f" {r['H02']['mean']:+.4f}, {r['F25']['mean']:+.4f},"
              f" {r['F50']['mean']:+.4f}, {r['F75']['mean']:+.4f}."
              f" READ 1 {shape}, excess over proportional {m:+.4f} t {t:+.2f}."
              f" Diagnostic only, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
