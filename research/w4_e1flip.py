"""w4_e1flip. AMENDMENT 58, registered in step0_prereg.md before this
file existed, and registered as motivated by AMENDMENT 56's numbers.

A56 read k0 at +0.0592 above the zero on the full row set and between
-0.0167 and -0.0407 below it inside each of three length bins, on the
same twelve seeds. It listed four candidate causes and separated none.
C1 gives the full row population at the bins' row count. C2 gives the
rows the bins drop. C3 pools every row of 17 events or more without
stratifying, which is the read that decides between narrowness and
exclusion.

CPU only, no torch, no generation, no new seeds. Diagnostic only, never
a training signal, never a serve candidate, no selection of
trajectories.
"""
import json
import sys

import numpy as np

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import scoring   # noqa: E402
import ledger    # noqa: E402

SEEDS = list(range(52, 64))
K = 20
PERM_SEED = 3208
N = 2000
MAX_T = 256
KMAX = 4
TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
FLOOR = "research/w4_e1floor_F_{a}_s{s}.npy"
ARM = "research/w4_e1feat_F_{a}_s{s}.npy"
ZERO = "L3_FULL"
ARMS = ["k0", "h016"]
BINS = [(17, 32), (33, 64), (65, 256)]
SHORT = (5, 16)
POOLED = (17, 256)
# A56 on these same twelve seeds, so they are memory, not evidence
A56_FULL = {"k0": 0.0592, "h016": 0.0193}
A56_BINS = {"k0": {(17, 32): -0.0192, (33, 64): -0.0407, (65, 256): -0.0167},
            "h016": {(17, 32): -0.0060, (33, 64): -0.0157, (65, 256): -0.0002}}


def auc_mean(m):
    v = np.array([scoring.score_features(
        m[np.random.default_rng(PERM_SEED + k).permutation(len(m))])["auc_rf_oob"]
        for k in range(K)])
    return float(v.mean())


def paired(vals):
    d = np.asarray(vals, dtype=float)
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return m, se, (m / se if se > 0 else float("inf"))


def bar(m, t):
    if abs(m) >= 0.004 and abs(t) >= 3.0:
        return "REAL"
    if abs(t) < 2.0:
        return "NULL"
    return "BETWEEN"


def row_lengths(seed, lengths, elig):
    pick = np.sort(np.random.default_rng(1000 + seed).choice(elig, N,
                                                             replace=False))
    return np.minimum(lengths[pick], MAX_T).astype(np.int64)


def main():
    print(f"  every value is the mean of K={K} permutations,"
          f" {len(SEEDS)} seeds {SEEDS[0]} to {SEEDS[-1]}, the SAME twelve"
          f" A56 used, so this is a control and NOT a replication\n",
          flush=True)

    lengths = np.load("training/events_len.npy")
    NT = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(NT, min(NT, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(NT), trained)
    elig = held[lengths[held] > KMAX]

    mats, per_ok, per_len, bin_n = {}, {}, {}, {}
    for s in SEEDS:
        m = {ZERO: np.load(FLOOR.format(a=ZERO, s=s))}
        for a in ARMS:
            m[a] = np.load(ARM.format(a=a, s=s))
        ok = np.ones(len(m[ZERO]), dtype=bool)
        for v in m.values():
            ok &= np.isfinite(v).all(1)
        mats[s], per_ok[s], per_len[s] = m, ok, row_lengths(s, lengths, elig)
        # the row count A56's matched bins used on this seed
        bin_n[s] = min(int((ok & (per_len[s] >= lo)
                            & (per_len[s] <= hi)).sum()) for lo, hi in BINS)

    # C1, row count alone: the whole row population at the bins' row count
    print(f"  C1, ROW COUNT ALONE. A random subsample of every finite row,"
          f" no length condition, sized to that seed's matched bin count.",
          flush=True)
    c1 = {a: [] for a in ARMS}
    c1z, c1lv = [], {a: [] for a in ARMS}
    for s in SEEDS:
        ix = np.flatnonzero(per_ok[s])
        take = np.sort(np.random.default_rng(77000 + s)
                       .choice(ix, bin_n[s], replace=False))
        z = auc_mean(mats[s][ZERO][take])
        c1z.append(z)
        for a in ARMS:
            v = auc_mean(mats[s][a][take])
            c1lv[a].append(v)
            c1[a].append(v - z)
        print(f"    seed {s} rows {bin_n[s]}   {ZERO} {z:.4f}   "
              + "  ".join(f"{a} {c1lv[a][-1]:.4f}" for a in ARMS), flush=True)

    # C2, the rows the bins drop, and the hard identity gate on h016
    print(f"\n  C2, THE EXCLUDED ROWS, human length {SHORT[0]} to {SHORT[1]}."
          f" h016 forces sixteen events so its term MUST be exactly 0.0000.",
          flush=True)
    c2 = {a: [] for a in ARMS}
    c2z, c2lv, c2n = [], {a: [] for a in ARMS}, []
    for s in SEEDS:
        ix = np.flatnonzero(per_ok[s] & (per_len[s] >= SHORT[0])
                            & (per_len[s] <= SHORT[1]))
        c2n.append(len(ix))
        z = auc_mean(mats[s][ZERO][ix])
        c2z.append(z)
        for a in ARMS:
            v = auc_mean(mats[s][a][ix])
            c2lv[a].append(v)
            c2[a].append(v - z)
        print(f"    seed {s} rows {len(ix)}   {ZERO} {z:.4f}   "
              + "  ".join(f"{a} {c2lv[a][-1]:.4f}" for a in ARMS), flush=True)
    m2, se2, t2 = paired(c2["h016"])
    gate = bool(abs(m2) < 1e-12 and abs(se2) < 1e-12)
    print(f"    HARD GATE, h016 minus the zero on this band:"
          f" {m2:+.6f}  {'exactly zero, ok' if gate else 'NONZERO, every read here is VOID (registered)'}",
          flush=True)

    # C3, the decisive read: every row of 17 or more, pooled, unstratified
    print(f"\n  C3, THE DECISIVE READ. Every finite row of {POOLED[0]} events"
          f" or more, pooled and scored together, NOT stratified.",
          flush=True)
    c3 = {a: [] for a in ARMS}
    c3z, c3lv, c3n = [], {a: [] for a in ARMS}, []
    for s in SEEDS:
        ix = np.flatnonzero(per_ok[s] & (per_len[s] >= POOLED[0])
                            & (per_len[s] <= POOLED[1]))
        c3n.append(len(ix))
        z = auc_mean(mats[s][ZERO][ix])
        c3z.append(z)
        for a in ARMS:
            v = auc_mean(mats[s][a][ix])
            c3lv[a].append(v)
            c3[a].append(v - z)
        print(f"    seed {s} rows {len(ix)}   {ZERO} {z:.4f}   "
              + "  ".join(f"{a} {c3lv[a][-1]:.4f}" for a in ARMS), flush=True)

    # the full row set on the same seeds, for the sign the controls explain
    print(f"\n  FULL, every finite row, carried from A56 and recomputed here"
          f" so the controls sit beside it on identical code:", flush=True)
    cf = {a: [] for a in ARMS}
    cfz, cflv = [], {a: [] for a in ARMS}
    for s in SEEDS:
        ix = np.flatnonzero(per_ok[s])
        z = auc_mean(mats[s][ZERO][ix])
        cfz.append(z)
        for a in ARMS:
            v = auc_mean(mats[s][a][ix])
            cflv[a].append(v)
            cf[a].append(v - z)
    print(f"    {len(SEEDS)} seeds, {ZERO} level {np.mean(cfz):.4f}",
          flush=True)

    print(f"\n  THE TERMS. Every one is a paired seed by seed difference"
          f" from {ZERO}, twelve seeds.")
    print(f"    {'read':<26}{'rows':>6}{'arm':>7}"
          f"{'mean':>10}{'se':>8}{'t':>8}  bar        level")
    out, sets = {}, [("FULL, unstratified", cf, cfz, cflv,
                      float(np.mean([int(per_ok[s].sum()) for s in SEEDS]))),
                     ("C1 subsample, all lengths", c1, c1z, c1lv,
                      float(np.mean([bin_n[s] for s in SEEDS]))),
                     (f"C2 length {SHORT[0]} to {SHORT[1]}", c2, c2z, c2lv,
                      float(np.mean(c2n))),
                     (f"C3 length {POOLED[0]} and up", c3, c3z, c3lv,
                      float(np.mean(c3n)))]
    for name, d, zs, lv, nrows in sets:
        out[name] = dict(rows=nrows, zero_level=float(np.mean(zs)), arms={})
        for a in ARMS:
            m, se, t = paired(d[a])
            b = bar(m, t)
            out[name]["arms"][a] = dict(mean=m, se=se, t=t, bar=b,
                                        level=float(np.mean(lv[a])))
            print(f"    {name:<26}{nrows:6.0f}{a:>7}"
                  f"{m:+10.4f}{se:8.4f}{t:+8.2f}  {b:<9}"
                  f"  {np.mean(lv[a]):.4f}", flush=True)

    print(f"\n  A56's bins, for the sign the controls are explaining:")
    for a in ARMS:
        print(f"    {a:<6}" + "  ".join(
            f"{lo} to {hi} {A56_BINS[a][(lo, hi)]:+.4f}" for lo, hi in BINS))

    print(f"\n  THE REGISTERED DECISION RULE, applied:")
    k1 = out["C1 subsample, all lengths"]["arms"]["k0"]
    k3 = out[f"C3 length {POOLED[0]} and up"]["arms"]["k0"]
    if k1["mean"] > 0 and k1["bar"] == "REAL":
        print(f"    C1 is positive and clears the bar, so ROW COUNT ALONE"
              f" does not produce the flip. Candidate (b) is out.")
    else:
        print(f"    C1 is {k1['bar']} at {k1['mean']:+.4f}, so row count"
              f" alone is NOT ruled out as a cause. Candidate (b) stands.")
    if k3["mean"] > 0 and k3["bar"] != "NULL":
        print(f"    C3 is POSITIVE at {k3['mean']:+.4f} {k3['bar']}. Pooling"
              f" rows of 17 or more restores the sign, so the flip belongs"
              f" to NARROWNESS: stratifying is what does it, candidate (d).")
    elif k3["mean"] < 0 and k3["bar"] == "REAL":
        print(f"    C3 is NEGATIVE and clears the bar at {k3['mean']:+.4f}."
              f" Excluding short rows is by itself enough to flip the sign,"
              f" so the model's detectability lives in the SHORT rows."
              f" Candidate (c) survives.")
    else:
        print(f"    C3 is {k3['bar']} at {k3['mean']:+.4f}. Nothing is"
              f" concluded and the question stays open, as registered.")

    res = dict(k=K, seeds=SEEDS, hard_gate=gate, reads=out,
               a56_full=A56_FULL, a56_bins={a: {f"{lo}-{hi}": v for
                                                (lo, hi), v in A56_BINS[a].items()}
                                            for a in ARMS},
               bin_n={str(s): bin_n[s] for s in SEEDS})
    with open("research/w4_e1flip.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1flip.json")
    print("  one trajectory per row, no selection, diagnostic only,"
          " never a training signal, no serve decision")

    rid = ledger.append_row(
        "w4_e1flip",
        {"seeds": SEEDS, "k": K, "perm_seed": PERM_SEED, "arms": ARMS,
         "zero": ZERO, "reference": "A56 sign flip control, same twelve seeds"},
        "ok" if gate else "failed",
        metrics={"hard_gate": int(gate),
                 "c1_k0": k1["mean"], "c1_k0_t": k1["t"],
                 "c3_k0": k3["mean"], "c3_k0_t": k3["t"],
                 "c2_k0": out[f"C2 length {SHORT[0]} to {SHORT[1]}"]["arms"]["k0"]["mean"],
                 "full_k0": out["FULL, unstratified"]["arms"]["k0"]["mean"]},
        artifacts=["research/w4_e1flip.json"],
        notes=f"AMENDMENT 58 control for A56's sign flip. C1 {k1['mean']:+.4f}"
              f" t {k1['t']:+.2f}, C3 {k3['mean']:+.4f} t {k3['t']:+.2f},"
              f" full {out['FULL, unstratified']['arms']['k0']['mean']:+.4f}."
              f" Same twelve seeds as A56, a control and not a replication."
              f" Diagnostic only, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
