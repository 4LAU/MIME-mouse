"""w4_e1len2. AMENDMENT 59, registered in step0_prereg.md before this
file existed and disclosed there as motivated by A58's open question.

A58 established that splitting rows by human length reverses the sign of
every arm's residual and could not say why. This tests one account: the
scorer never filters the anchor, so restricting the synthetic rows to a
length band leaves a length mismatch that no arm can avoid. READ 1 asks
whether the forest actually leans on the duration channel inside a band.
READ 2 asks whether the arm that reads lower is the one whose durations
sit closer to the anchor. Either can refute the account.

CPU only, no torch, no generation, the same twelve seeds A56 and A58
used, so this is a control and NOT a replication. Diagnostic only, never
a training signal, never a serve candidate, no selection.
"""
import json
import sys

import numpy as np

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import scoring   # noqa: E402
import ledger    # noqa: E402
from features import FEATURE_NAMES   # noqa: E402

SEEDS = list(range(52, 64))
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
ARMS = ["k0", "h016"]
DUR = FEATURE_NAMES.index("movement_duration")
BANDS = [("5 to 15", 5, 15), ("17 to 32", 17, 32),
         ("33 to 64", 33, 64), ("65 to 256", 65, 256)]
FULL = ("full", 0, MAX_T)
# exact per seed gate values, see AMENDMENT 59 GATE CORRECTION
GATE_FULL = 0.510328     # A56 research/w4_e1len.json auc_full L3_FULL 52
GATE_5_16 = 0.8499       # A58 e1flip.log C2 seed 52
A58_GAP = {"full": +0.0592, "C1": +0.0294, "17 to 32": -0.0192,
           "33 to 64": -0.0407, "65 to 256": -0.0167, "5 to 15": -0.0838}
GATE_SEED = 52
GATE_TOL = 1e-4


def scored(m):
    """mean AUC and mean importances over K row permutations."""
    aucs, imps = [], []
    for k in range(K):
        r = scoring.score_features(
            m[np.random.default_rng(PERM_SEED + k).permutation(len(m))])
        aucs.append(r["auc_rf_oob"])
        imps.append([r["importances"][n] for n in FEATURE_NAMES])
    return float(np.mean(aucs)), np.asarray(imps).mean(0)


def paired(v):
    d = np.asarray(v, float)
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return m, se, (m / se if se > 0 else float("inf"))


def bar(m, t, floor):
    if abs(m) >= floor and abs(t) >= 3.0:
        return "REAL"
    if abs(t) < 2.0:
        return "NULL"
    return "BETWEEN"


def w1(a, b):
    """1 Wasserstein distance between two equal length samples."""
    return float(np.mean(np.abs(np.sort(a) - np.sort(b))))


def main():
    print(f"  every value is the mean of K={K} permutations,"
          f" {len(SEEDS)} seeds {SEEDS[0]} to {SEEDS[-1]}, the SAME twelve"
          f" A56 and A58 used, so this is a control and NOT a"
          f" replication\n", flush=True)

    anchor = np.load(ANCHOR)
    lengths = np.load("training/events_len.npy")
    NT = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(NT, min(NT, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(NT), trained)
    elig = held[lengths[held] > KMAX]

    mats, per_ok, per_len = {}, {}, {}
    for s in SEEDS:
        m = {ZERO: np.load(FLOOR.format(a=ZERO, s=s))}
        for a in ARMS:
            m[a] = np.load(ARM.format(a=a, s=s))
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

    # GATE
    print(f"  GATE (recompute A58's zero levels on seed {GATE_SEED} through"
          f" this code path):", flush=True)
    gf, _ = scored(mats[GATE_SEED][ZERO][rows(GATE_SEED, *FULL[1:])])
    gs, _ = scored(mats[GATE_SEED][ZERO][rows(GATE_SEED, 5, 16)])
    ok_f = abs(gf - GATE_FULL) <= GATE_TOL
    ok_s = abs(gs - GATE_5_16) <= GATE_TOL
    print(f"    full row set  {gf:.6f}  against A56's {GATE_FULL:.6f}"
          f"   {'ok' if ok_f else 'MISS'}", flush=True)
    print(f"    band 5 to 16  {gs:.6f}  against A58's {GATE_5_16:.6f}"
          f"   {'ok' if ok_s else 'MISS'}", flush=True)
    gate = bool(ok_f and ok_s)
    print(f"    {'GATE PASSED' if gate else 'GATE FAILED, every read is VOID (registered)'}",
          flush=True)
    if not gate:
        return

    # per seed, per row set: zero level, zero importances, arm levels, W1
    print(f"\n  scoring {len(BANDS) + 1} row sets on {len(SEEDS)} seeds",
          flush=True)
    lv, imp, dw = {}, {}, {}
    for name, lo, hi in [FULL] + BANDS:
        lv[name] = {"zero": [], "k0": [], "h016": []}
        imp[name], dw[name] = [], {"zero": [], "k0": [], "h016": []}
        for s in SEEDS:
            ix = rows(s, lo, hi)
            a, i = scored(mats[s][ZERO][ix])
            lv[name]["zero"].append(a)
            imp[name].append(i[DUR])
            for x in ARMS:
                av, _ = scored(mats[s][x][ix])
                lv[name][x].append(av)
            n_use = min(len(anchor), len(ix))
            ha = anchor[:n_use, DUR]
            dw[name]["zero"].append(w1(mats[s][ZERO][ix][:n_use, DUR], ha))
            for x in ARMS:
                dw[name][x].append(w1(mats[s][x][ix][:n_use, DUR], ha))
        print(f"    {name:<10} rows {len(rows(SEEDS[0], lo, hi)):5d}"
              f"   zero {np.mean(lv[name]['zero']):.4f}"
              f"   k0 {np.mean(lv[name]['k0']):.4f}"
              f"   duration importance {np.mean(imp[name]):.4f}", flush=True)

    print(f"\n  READ 1 (PRIMARY), the forest's movement_duration importance"
          f" inside a band minus its importance on the full row set,"
          f" measured on the zero:")
    print(f"    {'band':<12}{'mean':>9}{'se':>8}{'t':>8}  bar"
          f"        band share   full share")
    r1 = {}
    for name, lo, hi in BANDS:
        m, se, t = paired([imp[name][i] - imp["full"][i]
                           for i in range(len(SEEDS))])
        b = bar(m, t, 0.02)
        r1[name] = dict(mean=m, se=se, t=t, bar=b,
                        band=float(np.mean(imp[name])),
                        full=float(np.mean(imp["full"])))
        print(f"    {name:<12}{m:+9.4f}{se:8.4f}{t:+8.2f}  {b:<9}"
              f"   {np.mean(imp[name]):.4f}      {np.mean(imp['full']):.4f}",
              flush=True)
    hits1 = sum(1 for v in r1.values() if v["bar"] == "REAL" and v["mean"] > 0)
    print(f"    REAL and positive in {hits1} of {len(BANDS)} bands."
          f" The account needs the forest to lean on the channel inside a"
          f" band, so {'this half holds' if hits1 == len(BANDS) else 'this half is not clean'}.")

    print(f"\n  READ 2, the duration distance to the anchor, zero minus k0,"
          f" in seconds. POSITIVE means k0 sits closer to the anchor,"
          f" which the account requires:")
    print(f"    {'band':<12}{'mean':>9}{'se':>8}{'t':>8}  bar"
          f"        W1 zero    W1 k0")
    r2 = {}
    for name, lo, hi in [FULL] + BANDS:
        m, se, t = paired([dw[name]["zero"][i] - dw[name]["k0"][i]
                           for i in range(len(SEEDS))])
        b = bar(m, t, 0.002)
        r2[name] = dict(mean=m, se=se, t=t, bar=b,
                        w1_zero=float(np.mean(dw[name]["zero"])),
                        w1_k0=float(np.mean(dw[name]["k0"])))
        print(f"    {name:<12}{m:+9.4f}{se:8.4f}{t:+8.2f}  {b:<9}"
              f"   {np.mean(dw[name]['zero']):.4f}   {np.mean(dw[name]['k0']):.4f}",
              flush=True)
    neg = [n for n, _, _ in BANDS if r2[n]["mean"] < 0]
    print(f"    negative in {len(neg)} of {len(BANDS)} bands"
          + (f" ({', '.join(neg)}), which REFUTES the account as registered"
             if neg else ", so no band refutes the account"))

    print(f"\n  READ 3, DESCRIPTIVE, no verdict. The AUC gap k0 minus zero"
          f" beside the duration gap, five row sets:")
    print(f"    {'row set':<12}{'AUC gap':>10}{'duration gap':>15}  signs agree")
    agree = 0
    for name in ["full"] + [n for n, _, _ in BANDS]:
        g = float(np.mean(lv[name]["k0"]) - np.mean(lv[name]["zero"]))
        d = r2[name]["mean"]
        ok = (g < 0) == (d > 0)
        agree += int(ok)
        print(f"    {name:<12}{g:+10.4f}{d:+15.4f}  {'yes' if ok else 'no'}")
    print(f"    {agree} of 5 agree. Descriptive, no verdict, as registered.")

    res = dict(k=K, seeds=SEEDS, gate=gate, read1=r1, read2=r2,
               levels={n: {a: float(np.mean(v)) for a, v in d.items()}
                       for n, d in lv.items()},
               dur_importance={n: float(np.mean(v)) for n, v in imp.items()},
               read3_agree=agree)
    with open("research/w4_e1len2.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1len2.json")
    print("  one trajectory per row, no selection, diagnostic only,"
          " never a training signal, no serve decision")

    rid = ledger.append_row(
        "w4_e1len2",
        {"seeds": SEEDS, "k": K, "perm_seed": PERM_SEED, "zero": ZERO,
         "bands": [b[0] for b in BANDS],
         "reference": "A58 open question, the length mismatch account"},
        "ok" if gate else "failed",
        metrics={"gate": int(gate), "read1_real_positive": hits1,
                 "read2_negative_bands": len(neg), "read3_agree": agree,
                 "dur_imp_full": float(np.mean(imp["full"])),
                 "dur_imp_short": float(np.mean(imp["5 to 15"]))},
        artifacts=["research/w4_e1len2.json"],
        notes=f"AMENDMENT 59 the length mismatch account for A58's sign"
              f" reversal. READ 1 REAL and positive in {hits1} of 4 bands,"
              f" READ 2 negative in {len(neg)} of 4, READ 3 {agree} of 5"
              f" signs agree. Correlational, refutable, registered in"
              f" advance. Diagnostic only.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
