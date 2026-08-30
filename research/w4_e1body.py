"""w4_e1body. AMENDMENT 54, registered in step0_prereg.md before this file
existed.

A53 left +0.0143 that forcing the human's first sixteen events cannot
reach. By construction it lives entirely in rows longer than sixteen
events, because a shorter row renders the full human round trip and
matches the zero exactly. This asks whether that residual is just the
model's per event error accumulating over however many events it still
generates, or whether the body of a movement is harder than its opening
in some way the count does not explain.

CPU only, no model generation. Diagnostic only, never a training signal,
never a serve candidate, no selection of trajectories.
"""
import json
import os
import shutil
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
N = 2000
MAX_T = 256
KMAX = 4
TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
FLOOR = "research/w4_e1floor_F_{a}_s{s}.npy"
ARM = "research/w4_e1feat_F_{a}_s{s}.npy"
ZERO = "L3_FULL"

# READ 1
LONG_MIN = 32                      # rows STRICTLY longer than this
PREFIX = [2, 4, 8, 16, 32]
ANCHOR = 2                         # proportionality is anchored here
VOTERS = [16, 32]                  # only these two decide the verdict
# READ 2
BINS = [(17, 32), (33, 64), (65, 256)]
CACHED_H016_S40 = "/home/aaronadmin/w4_arms/h016_s40_cached.npy"


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


def load(seed, arms):
    return {a: np.load(FLOOR.format(a=a, s=seed) if a.startswith("L")
                       else ARM.format(a=a, s=seed)) for a in arms}


def integrity_gate():
    """h016 on seed 40 was regenerated alongside h032 with the new arm in
    ALL_ARMS. torch.manual_seed is set per batch inside the arm loop, so
    adding an entry must not move any other arm. Bit identical or void.
    The cached copy is restored either way."""
    live = ARM.format(a="h016", s=40)
    a = np.load(CACHED_H016_S40)
    b = np.load(live)
    same = a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    print("\n  INTEGRITY GATE (adding h032 to ALL_ARMS must not move h016):")
    if same:
        print(f"    seed 40 h016 regenerated bit identical to the cache,"
              f" {a.shape[0]} rows: ok")
    else:
        n_diff = int((~np.isclose(a, b, equal_nan=True)).any(1).sum())
        print(f"    seed 40 h016 DIFFERS from the cache on {n_diff} rows:"
              f" GATE FAILED, every read is VOID (registered)")
    shutil.copyfile(CACHED_H016_S40, live)
    print(f"    cached copy restored to {live}")
    return bool(same)


def main():
    lengths = np.load("training/events_len.npy")
    NT = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(NT, min(NT, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(NT), trained)
    elig = held[lengths[held] > KMAX]
    L = {s: row_lengths(s, lengths, elig) for s in SEEDS}

    print(f"  every value is the mean of K={K} permutations, "
          f"{len(SEEDS)} seeds\n", flush=True)

    gate0 = integrity_gate()

    # ---------------- READ 1, the prefix ladder on long rows -------------
    arms = [ZERO] + [f"h0{k:02d}" if k >= 10 else f"h0{k}" for k in PREFIX]
    names = {k: (f"h0{k:02d}" if k >= 10 else f"h0{k}") for k in PREFIX}
    per = {a: {} for a in arms}
    g = {}
    print(f"\n  READ 1 rows: strictly longer than {LONG_MIN} events,"
          f" finite in all {len(arms)} matrices", flush=True)
    for s in SEEDS:
        mats = load(s, arms)
        ok = L[s] > LONG_MIN
        for m in mats.values():
            ok &= np.isfinite(m).all(1)
        g[s] = float(L[s][ok].mean())
        for a in arms:
            per[a][s] = auc_mean(mats[a][ok])
        print(f"    seed {s} rows {int(ok.sum())} mean length {g[s]:.1f}   "
              + "  ".join(f"{a} {per[a][s]:.4f}" for a in arms), flush=True)

    print(f"\n  VALIDITY GATE (the ladder must not RISE as more real events"
          f" are forced):", flush=True)
    gate1, gsteps = True, {}
    for lo, hi in zip(PREFIX, PREFIX[1:]):
        m, se, t = paired([per[names[hi]][s] - per[names[lo]][s]
                           for s in SEEDS])
        bad = (m >= 0.005 and t >= 3.0)
        gate1 = gate1 and not bad
        gsteps[f"h{lo}->h{hi}"] = dict(mean=m, se=se, t=t)
        print(f"    h0{lo:<3} to h0{hi:<3} {m:+.4f}  se {se:.4f}  t {t:+.2f}"
              f"   {'RISES' if bad else 'ok'}", flush=True)
    print(f"  {'GATE PASSED' if gate1 else 'GATE FAILED, READ 1 is VOID'}")

    resid = {k: {s: per[names[k]][s] - per[ZERO][s] for s in SEEDS}
             for k in PREFIX}
    print(f"\n  THE LADDER on long rows, against {ZERO}:")
    lad = {}
    for k in PREFIX:
        m, se, t = paired([resid[k][s] for s in SEEDS])
        gk = float(np.mean([g[s] - k for s in SEEDS]))
        lad[k] = dict(mean=m, se=se, t=t, free_events=gk)
        print(f"    h0{k:<3} {m:+.4f} se {se:.4f} t {t:+6.2f}"
              f"   mean free events {gk:.1f}")

    print(f"\n  PRIMARY READ, does plain accumulation explain the ladder."
          f"\n  D(k) = residual(k) minus residual({ANCHOR}) times"
          f" g(k)/g({ANCHOR}), paired over the seeds:")
    D = {}
    for k in PREFIX:
        if k == ANCHOR:
            continue
        d = [resid[k][s] - resid[ANCHOR][s] * (g[s] - k) / (g[s] - ANCHOR)
             for s in SEEDS]
        m, se, t = paired(d)
        D[k] = dict(mean=m, se=se, t=t, bar=bar(m, t))
        vote = "" if k in VOTERS else "   (does not vote)"
        print(f"    D({k:>2})  {m:+.4f}  se {se:.4f}  t {t:+.2f}"
              f"   {bar(m, t):<7}{vote}")

    b16, b32 = D[16], D[32]
    if not gate0 or not gate1:
        v1 = "VOID, a gate failed"
    elif b16["bar"] == "NULL" and b32["bar"] == "NULL":
        v1 = "ACCUMULATION IS ENOUGH"
    elif (b16["bar"] == "REAL" and b32["bar"] == "REAL"
          and b16["mean"] < 0 and b32["mean"] < 0):
        v1 = "FALLS FASTER THAN ACCUMULATION"
    elif (b16["bar"] == "REAL" and b32["bar"] == "REAL"
          and b16["mean"] > 0 and b32["mean"] > 0):
        v1 = "FALLS SLOWER THAN ACCUMULATION"
    else:
        v1 = "UNRESOLVED"
    print(f"    VERDICT {v1}")

    # ---------------- READ 2, the length stratification -------------------
    pops = np.array([[int(((L[s] >= lo) & (L[s] <= hi)).sum())
                      for lo, hi in BINS] for s in SEEDS])
    matched = int(pops.min())
    print(f"\n  READ 2, the h016 residual by row length."
          f"\n  bin populations over the seeds, min {matched}, which is the"
          f" matched row count:")
    for j, (lo, hi) in enumerate(BINS):
        print(f"    {lo:>3} to {hi:<3}  min {pops[:, j].min()}"
              f"  max {pops[:, j].max()}")

    bres = {j: {} for j in range(len(BINS))}
    bfree = {j: [] for j in range(len(BINS))}
    bn = {j: [] for j in range(len(BINS))}
    for s in SEEDS:
        mats = load(s, [ZERO, "h016"])
        fin = np.isfinite(mats[ZERO]).all(1) & np.isfinite(mats["h016"]).all(1)
        for j, (lo, hi) in enumerate(BINS):
            idx = np.flatnonzero((L[s] >= lo) & (L[s] <= hi))
            take = np.random.default_rng(7000 + 100 * s + j).choice(
                idx, matched, replace=False)
            take = np.sort(take[fin[take]])
            bn[j].append(len(take))
            bfree[j].append(float((L[s][take] - 16).mean()))
            bres[j][s] = (auc_mean(mats["h016"][take])
                          - auc_mean(mats[ZERO][take]))
        print(f"    seed {s} done", flush=True)

    print(f"\n    bin        rows   mean free events   h016 minus {ZERO}")
    binout = {}
    for j, (lo, hi) in enumerate(BINS):
        m, se, t = paired([bres[j][s] for s in SEEDS])
        binout[f"{lo}-{hi}"] = dict(mean=m, se=se, t=t,
                                    rows=int(np.mean(bn[j])),
                                    free_events=float(np.mean(bfree[j])))
        print(f"    {lo:>3} to {hi:<4}{int(np.mean(bn[j])):5d}"
              f"        {np.mean(bfree[j]):7.1f}"
              f"        {m:+.4f} se {se:.4f} t {t:+6.2f}")
    dm, dse, dt = paired([bres[len(BINS) - 1][s] - bres[0][s] for s in SEEDS])
    if dm >= 0.005 and dt >= 3.0:
        v2 = "GROWS WITH LENGTH"
    elif abs(dm) < 0.005 or abs(dt) < 2.0:
        v2 = "FLAT IN LENGTH"
    else:
        v2 = "BETWEEN"
    print(f"    longest bin minus shortest bin {dm:+.4f} se {dse:.4f}"
          f" t {dt:+.2f}   {v2}")

    res = dict(k=K, seeds=SEEDS, integrity_gate=bool(gate0),
               validity_gate=bool(gate1), validity_steps=gsteps,
               ladder={str(k): lad[k] for k in PREFIX},
               D={str(k): D[k] for k in D}, verdict1=v1,
               matched_rows=matched, bins=binout,
               bin_difference=dict(mean=dm, se=dse, t=dt), verdict2=v2,
               auc={a: {str(s): per[a][s] for s in SEEDS} for a in arms})
    with open("research/w4_e1body.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1body.json")
    print("  one trajectory per row, no selection, diagnostic only,"
          " never a training signal, no serve decision")

    rid = ledger.append_row(
        "w4_e1body",
        {"seeds": SEEDS, "n": N, "k": K, "perm_seed": PERM_SEED,
         "long_min": LONG_MIN, "prefix": PREFIX, "anchor": ANCHOR,
         "bins": [list(b) for b in BINS], "matched_rows": matched,
         "zero": ZERO, "reference": "A53 ladder, A52 corrected zero"},
        "ok" if (gate0 and gate1) else "failed",
        metrics={"integrity_gate": int(gate0), "validity_gate": int(gate1),
                 "d16": D[16]["mean"], "d16_t": D[16]["t"],
                 "d32": D[32]["mean"], "d32_t": D[32]["t"],
                 "bin_difference": dm, "bin_difference_t": dt},
        artifacts=["research/w4_e1body.json"],
        notes=f"AMENDMENT 54 the body residual. READ 1 {v1}, D(16)"
              f" {D[16]['mean']:+.4f} t {D[16]['t']:+.2f}, D(32)"
              f" {D[32]['mean']:+.4f} t {D[32]['t']:+.2f}. READ 2 {v2},"
              f" longest minus shortest bin {dm:+.4f} t {dt:+.2f}."
              f" Diagnostic only, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
