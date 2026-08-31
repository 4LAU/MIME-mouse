"""w4_e1len. AMENDMENT 56, registered in step0_prereg.md before this
file existed.

A52 established L3_FULL, the full human token round trip, as the zero the
arms can reach. A54's READ 2 then found the h016 arm sitting BELOW that
zero on rows of 17 to 32 and 33 to 64 events. Those bin levels were not a
registered read and I have seen them, so this asks the question on twelve
seeds that have never been scored, and on the same generation budget asks
where by row length the free running model's gap actually lives.

CPU only apart from the arms, which are generated elsewhere. Diagnostic
only, never a training signal, never a serve candidate, no selection of
trajectories.
"""
import json
import os
import sys

import numpy as np

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import w4_e1floor as fl   # noqa: E402  (pins the decoder env at import)
import scoring            # noqa: E402
import ledger             # noqa: E402
from features import FEATURE_NAMES   # noqa: E402

SEEDS = list(range(52, 64))
GATE_A_SEED = 45
K = 20
PERM_SEED = 3208
N = 2000
MAX_T = 256
KMAX = 4
TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
GATE_B_MIN_ROWS = 100
ANCHOR = "data/human_val_features_grpo.npy"
FLOOR = "research/w4_e1floor_F_{a}_s{s}.npy"
ARM = "research/w4_e1feat_F_{a}_s{s}.npy"
FLOOR_ARMS = ["L0_RAW", "L1_DT", "L2_POLAR", "L3_FULL", "L3B_NOSNAP"]
ZERO = "L3_FULL"
ARMS = ["k0", "h016"]
BINS = [(17, 32), (33, 64), (65, 256)]
DUR = FEATURE_NAMES.index("movement_duration")
# A53 measured these against L3_FULL on seeds 40 to 51, full row set
A53_FULL = {"k0": 0.0533, "h016": 0.0143}
# A54 READ 2, h016 by bin on seeds 40 to 51 at a matched 452 rows
A54_BINS = {(17, 32): -0.0048, (33, 64): -0.0117, (65, 256): +0.0032}


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
          f" {len(SEEDS)} fresh seeds {SEEDS[0]} to {SEEDS[-1]}\n", flush=True)

    lengths = np.load("training/events_len.npy")
    NT = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(NT, min(NT, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(NT), trained)
    elig = held[lengths[held] > KMAX]
    arrs = [np.load(f"training/events_{k}.npy", mmap_mode="r")
            for k in ("s2", "dth", "dt", "dx", "dy", "cond")]

    # GATE A, the registered stop
    print(f"  GATE A (rebuild the floor on seed {GATE_A_SEED} and compare"
          f" to the committed cache):", flush=True)
    _, rebuilt = fl.build(GATE_A_SEED, lengths, *arrs, elig)
    ref = np.load(FLOOR.format(a=ZERO, s=GATE_A_SEED))
    gate_a = bool(np.array_equal(np.nan_to_num(rebuilt[ZERO], nan=-9e18),
                                 np.nan_to_num(ref, nan=-9e18)))
    print(f"    {ZERO} {'bit exact, ok' if gate_a else 'DIFFERS, every read is VOID (registered)'}",
          flush=True)
    if not gate_a:
        d = np.abs(np.nan_to_num(rebuilt[ZERO]) - np.nan_to_num(ref))
        print(f"    rows differing {int((d > 0).any(1).sum())}/{N},"
              f" max abs {d.max():.3e}")
        return

    # build the floor on the twelve fresh seeds
    for s in SEEDS:
        paths = {a: FLOOR.format(a=a, s=s) for a in FLOOR_ARMS}
        if all(os.path.exists(v) for v in paths.values()):
            continue
        _, built = fl.build(s, lengths, *arrs, elig)
        for a, v in paths.items():
            np.save(v, built[a])
        print(f"  built floor seed {s}: "
              + "  ".join(f"{a} {int(np.isfinite(built[a]).all(1).sum())}"
                          for a in FLOOR_ARMS), flush=True)

    # GATE B, A52's corrected identity on rows the forced arm renders whole
    print(f"\n  GATE B (rows with L under 16: h016 forces the whole row, so"
          f" it must equal {ZERO} element for element):", flush=True)
    gate_b, tot_rows, bad_rows = True, 0, 0
    for s in SEEDS:
        L = row_lengths(s, lengths, elig)
        z = np.load(FLOOR.format(a=ZERO, s=s))
        h = np.load(ARM.format(a="h016", s=s))
        ix = np.flatnonzero(L < 16)
        eq = np.array_equal(np.nan_to_num(z[ix], nan=-9e18),
                            np.nan_to_num(h[ix], nan=-9e18))
        nbad = int((np.nan_to_num(z[ix], nan=-9e18)
                    != np.nan_to_num(h[ix], nan=-9e18)).any(1).sum())
        tot_rows += len(ix)
        bad_rows += nbad
        gate_b = gate_b and eq
        if not eq or s == SEEDS[0]:
            print(f"    seed {s}: {len(ix)} rows, {nbad} differing", flush=True)
    print(f"    {tot_rows} rows over twelve seeds, {bad_rows} differing"
          f"   {'GATE PASSED' if gate_b else 'GATE FAILED, every read is VOID (registered)'}")
    if tot_rows / len(SEEDS) < GATE_B_MIN_ROWS:
        print(f"    UNDERPOWERED, under {GATE_B_MIN_ROWS} rows per seed")
    if not gate_b:
        return

    # per seed matrices, finite intersection, bin membership
    per_len, per_ok, mats = {}, {}, {}
    for s in SEEDS:
        per_len[s] = row_lengths(s, lengths, elig)
        m = {ZERO: np.load(FLOOR.format(a=ZERO, s=s))}
        for a in ARMS:
            m[a] = np.load(ARM.format(a=a, s=s))
        ok = np.ones(N, dtype=bool)
        for v in m.values():
            ok &= np.isfinite(v).all(1)
        per_ok[s], mats[s] = ok, m

    # GATE C, the signs on the full row set
    print(f"\n  GATE C (full row set, both arms must sit ABOVE the zero):",
          flush=True)
    full = {a: {} for a in [ZERO] + ARMS}
    for s in SEEDS:
        ok = per_ok[s]
        for a in [ZERO] + ARMS:
            full[a][s] = auc_mean(mats[s][a][ok])
        print(f"    seed {s} rows {int(ok.sum())}   "
              + "  ".join(f"{a} {full[a][s]:.4f}" for a in [ZERO] + ARMS),
              flush=True)
    gate_c, gc = True, {}
    for a in ARMS:
        m, se, t = paired([full[a][s] - full[ZERO][s] for s in SEEDS])
        gc[a] = dict(mean=m, se=se, t=t, a53=A53_FULL[a])
        gate_c = gate_c and m > 0
        print(f"    {a:<6} minus {ZERO}  {m:+.4f} se {se:.4f} t {t:+.2f}"
              f"   A53 on seeds 40 to 51 {A53_FULL[a]:+.4f}"
              f"   {'ok' if m > 0 else 'SIGN FLIPPED'}", flush=True)
    print(f"  {'GATE PASSED' if gate_c else 'GATE FAILED, every read is VOID (registered)'}")
    if not gate_c:
        return

    # the matched bin draws, fixed before any bin AUC is computed
    take = {}
    for s in SEEDS:
        ok, L = per_ok[s], per_len[s]
        idx = [np.flatnonzero(ok & (L >= lo) & (L <= hi)) for lo, hi in BINS]
        matched = min(len(i) for i in idx)
        take[s] = [np.sort(np.random.default_rng(9000 + 100 * s + j)
                           .choice(i, matched, replace=False))
                   for j, i in enumerate(idx)]
        print(f"    seed {s} matched {matched} rows per bin", flush=True)

    print(f"\n  READ 1 (PRIMARY) and READ 2, each arm against the zero"
          f" inside each row length bin:", flush=True)
    binauc = {a: {j: {} for j in range(len(BINS))} for a in [ZERO] + ARMS}
    for s in SEEDS:
        for j in range(len(BINS)):
            ix = take[s][j]
            for a in [ZERO] + ARMS:
                binauc[a][j][s] = auc_mean(mats[s][a][ix])
    reads = {a: {} for a in ARMS}
    for a in ARMS:
        head = "READ 1 (PRIMARY)" if a == "h016" else "READ 2"
        print(f"    {a}   {head}")
        for j, (lo, hi) in enumerate(BINS):
            m, se, t = paired([binauc[a][j][s] - binauc[ZERO][j][s]
                               for s in SEEDS])
            b = bar(m, t)
            ref54 = (f"   A54 {A54_BINS[(lo, hi)]:+.4f}" if a == "h016" else "")
            reads[a][f"{lo}-{hi}"] = dict(mean=m, se=se, t=t, bar=b)
            print(f"      {lo:>3} to {hi:<3}  {m:+.4f} se {se:.4f}"
                  f" t {t:+6.2f}  {b:<8}{ref54}", flush=True)
        g, gse, gt = paired([binauc[a][2][s] - binauc[ZERO][2][s]
                             - (binauc[a][0][s] - binauc[ZERO][0][s])
                             for s in SEEDS])
        reads[a]["longest_minus_shortest"] = dict(mean=g, se=gse, t=gt,
                                                  bar=bar(g, gt))
        print(f"      longest minus shortest {g:+.4f} se {gse:.4f}"
              f" t {gt:+.2f}   {bar(g, gt)}", flush=True)
    v1 = reads["h016"][f"{BINS[0][0]}-{BINS[0][1]}"]
    print(f"    THE REGISTERED QUESTION is the {BINS[0][0]} to {BINS[0][1]}"
          f" bin for h016: {v1['mean']:+.4f}, {v1['bar']},"
          f" sign {'NEGATIVE' if v1['mean'] < 0 else 'POSITIVE'}")

    print(f"\n  READ 3 (MECHANISM, no verdict power), movement_duration"
          f" against the zero, seconds:", flush=True)
    anchor = np.load(ANCHOR)
    a_dur = float(anchor[np.isfinite(anchor).all(1)][:, DUR].mean())
    r3 = {a: {} for a in ARMS}
    z_dur = {}
    for j, (lo, hi) in enumerate(BINS):
        zd = paired([float(mats[s][ZERO][take[s][j]][:, DUR].mean())
                     for s in SEEDS])
        z_dur[f"{lo}-{hi}"] = zd[0]
        line = f"    {lo:>3} to {hi:<3}  zero {zd[0]:7.4f}"
        for a in ARMS:
            m, se, t = paired([float(mats[s][a][take[s][j]][:, DUR].mean())
                               - float(mats[s][ZERO][take[s][j]][:, DUR].mean())
                               for s in SEEDS])
            toward = abs(zd[0] + m - a_dur) < abs(zd[0] - a_dur)
            r3[a][f"{lo}-{hi}"] = dict(mean=m, se=se, t=t, toward_anchor=bool(toward))
            line += f"   {a} {m:+.4f} t {t:+5.1f} {'toward' if toward else 'away'}"
        print(line, flush=True)
    print(f"    anchor mean movement_duration {a_dur:.4f}s")

    print(f"\n  READ 4 (MECHANISM, no verdict power), the {BINS[0][0]} to"
          f" {BINS[0][1]} bin restricted to the half of rows whose h016"
          f" duration is closest to the zero's:", flush=True)
    m4 = {}
    for s in SEEDS:
        ix = take[s][0]
        d = np.abs(mats[s]["h016"][ix][:, DUR] - mats[s][ZERO][ix][:, DUR])
        keep = ix[np.argsort(d, kind="stable")[:len(ix) // 2]]
        m4[s] = (auc_mean(mats[s]["h016"][np.sort(keep)])
                 - auc_mean(mats[s][ZERO][np.sort(keep)]))
    m, se, t = paired([m4[s] for s in SEEDS])
    r4 = dict(mean=m, se=se, t=t, bar=bar(m, t))
    print(f"    h016 minus the zero on the duration matched half"
          f"  {m:+.4f} se {se:.4f} t {t:+.2f}   {r4['bar']}", flush=True)
    print(f"    REGISTERED CAVEAT: this is a row filter, and A54 established"
          f" that a row filter moves the anchor and costs sensitivity, so a"
          f" null here is weak evidence and is called weak.")

    res = dict(k=K, seeds=SEEDS, bins=[list(b) for b in BINS],
               gate_a=gate_a, gate_b=dict(passed=gate_b, rows=tot_rows,
                                          differing=bad_rows),
               gate_c=dict(passed=gate_c, arms=gc),
               read1=reads["h016"], read2=reads["k0"],
               primary=v1, read3=dict(arms=r3, zero=z_dur, anchor=a_dur),
               read4=r4,
               auc_full={a: {str(s): full[a][s] for s in SEEDS}
                         for a in [ZERO] + ARMS},
               auc_bins={a: {f"{BINS[j][0]}-{BINS[j][1]}":
                             {str(s): binauc[a][j][s] for s in SEEDS}
                             for j in range(len(BINS))}
                         for a in [ZERO] + ARMS})
    with open("research/w4_e1len.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1len.json")
    print("  one trajectory per row, no selection, diagnostic only,"
          " never a training signal, no serve decision")

    rid = ledger.append_row(
        "w4_e1len",
        {"seeds": SEEDS, "k": K, "perm_seed": PERM_SEED, "zero": ZERO,
         "arms": ARMS, "bins": [list(b) for b in BINS],
         "reference": "A52 zero, A54 bin observation, fresh seeds"},
        "ok" if (gate_a and gate_b and gate_c) else "failed",
        metrics={"gate_a": int(gate_a), "gate_b": int(gate_b),
                 "gate_c": int(gate_c),
                 "h016_short": v1["mean"], "h016_short_t": v1["t"],
                 "k0_short": reads["k0"][f"{BINS[0][0]}-{BINS[0][1]}"]["mean"],
                 "k0_growth": reads["k0"]["longest_minus_shortest"]["mean"],
                 "h016_growth": reads["h016"]["longest_minus_shortest"]["mean"],
                 "read4": r4["mean"]},
        artifacts=["research/w4_e1len.json"],
        notes=f"AMENDMENT 56 the residual by row length on twelve fresh"
              f" seeds. h016 on rows {BINS[0][0]} to {BINS[0][1]}"
              f" {v1['mean']:+.4f} t {v1['t']:+.2f}, {v1['bar']}, which is"
              f" the registered question of whether L3_FULL is a floor"
              f" within subsets. Diagnostic only, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
