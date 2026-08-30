"""w4_e1chan2. AMENDMENT 55, registered in step0_prereg.md before this
file existed.

A42 decomposed the cost inside event 1, with a real human event 0 held
fixed, as a nested ladder: what the real speed buys, then the angle on
top, then the duration on top of both. On six seeds the speed was REAL,
the angle NULL, and the duration NULL at t -1.71 while carrying a third
of the whole event 1 cost. A42 recorded then that the duration channel
was underpowered rather than absent and deserved more seeds. This is
that read, at twelve.

The increments are arm minus arm on identical rows, so A52's
displacement of the zero cancels out of them and cannot move the primary
read. It moves only the descriptive per arm read, restated here against
L3_FULL for the first time.

CPU only, no model generation. Diagnostic only, never a training signal,
never a serve candidate, no selection of trajectories.
"""
import json
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
ARM = "research/w4_e1feat_F_{a}_s{s}.npy"
FLOOR = "research/w4_e1floor_F_{a}_s{s}.npy"
ZERO = "L3_FULL"
LADDER = ["h0p1", "h0s1p", "h0st1p", "h02"]
ALL = [ZERO, "h01"] + LADDER
# the three nested channel increments, in A42's fixed order
STEPS = [("speed", "h0s1p", "h0p1"),
         ("angle", "h0st1p", "h0s1p"),
         ("duration", "h02", "h0st1p"),
         ("total", "h02", "h0p1")]
A42_STEPS = {"speed": -0.0114, "angle": -0.0015, "duration": -0.0065,
             "total": -0.0194}
# A42 READ 3, each arm against L0_RAW on six seeds
A42_READ3 = {"h01": 0.0252, "h0p1": 0.0304, "h0s1p": 0.0190,
             "h0st1p": 0.0175, "h02": 0.0110}
GATE_ARM = "/home/aaronadmin/w4_arms/h0s1p_s45_cached.npy"
P1_CACHED = "/home/aaronadmin/w4_arms/h0p1_s{s}_cached.npy"
P1_SEEDS = [46, 47, 48, 49]


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
    if abs(m) >= 0.005 and abs(t) >= 3.0:
        return "REAL"
    if abs(t) < 2.0:
        return "NULL"
    return "BETWEEN"


def same(a_path, b_path):
    a, b = np.load(a_path), np.load(b_path)
    return (a.shape == b.shape and np.array_equal(a, b, equal_nan=True),
            a.shape[0])


def gates():
    """The registered stop: h0s1p on seed 45 was regenerated with the
    current code path and must match the cache bit for bit. Then the
    disclosed second check on the four h0p1 matrices that already
    existed; those are NOT restored, because the registration says every
    seed of the ladder comes from one run of one code path."""
    print("\n  INTEGRITY GATE (registered stop):")
    live = ARM.format(a="h0s1p", s=45)
    ok, n = same(GATE_ARM, live)
    print(f"    seed 45 h0s1p regenerated, {n} rows:"
          f" {'bit identical to the cache, ok' if ok else 'DIFFERS, every read is VOID (registered)'}")
    shutil.copyfile(GATE_ARM, live)
    print(f"    cached copy restored to {live}")

    print("\n  SECOND CHECK, disclosed in the registration addendum,"
          " the four h0p1 matrices that already existed:")
    p1 = {}
    for s in P1_SEEDS:
        m, n = same(P1_CACHED.format(s=s), ARM.format(a="h0p1", s=s))
        p1[str(s)] = bool(m)
        print(f"    seed {s} h0p1 {'bit identical' if m else 'DIFFERS from the earlier cache'}")
    print("    the regenerated matrices are kept either way, as registered")
    return bool(ok), p1


def main():
    print(f"  every value is the mean of K={K} permutations,"
          f" {len(SEEDS)} seeds\n", flush=True)
    gate0, p1check = gates()

    per = {a: {} for a in ALL}
    print(f"\n  rows: finite in all {len(ALL)} matrices", flush=True)
    for s in SEEDS:
        mats = {a: np.load(FLOOR.format(a=a, s=s) if a.startswith("L")
                           else ARM.format(a=a, s=s)) for a in ALL}
        ok = np.ones(len(mats[ZERO]), dtype=bool)
        for m in mats.values():
            ok &= np.isfinite(m).all(1)
        for a in ALL:
            per[a][s] = auc_mean(mats[a][ok])
        print(f"    seed {s} rows {int(ok.sum())}   "
              + "  ".join(f"{a} {per[a][s]:.4f}" for a in ALL), flush=True)

    print(f"\n  VALIDITY GATE (handing the model more of the true event 1"
          f" cannot make the arm less human):", flush=True)
    gate1, gsteps = True, {}
    for lo, hi in zip(LADDER, LADDER[1:]):
        m, se, t = paired([per[hi][s] - per[lo][s] for s in SEEDS])
        bad = (m >= 0.005 and t >= 3.0)
        gate1 = gate1 and not bad
        gsteps[f"{lo}->{hi}"] = dict(mean=m, se=se, t=t)
        print(f"    {lo:>7} to {hi:<7} {m:+.4f}  se {se:.4f}  t {t:+.2f}"
              f"   {'RISES' if bad else 'ok'}", flush=True)
    print(f"  {'GATE PASSED' if gate1 else 'GATE FAILED, every read is VOID (registered)'}")

    print(f"\n  READ 1 (PRIMARY), the nested channel increments,"
          f" twelve seeds:")
    print(f"    {'channel':<9}{'twelve seeds':>26}{'A42 on six':>14}")
    inc = {}
    for name, hi, lo in STEPS:
        m, se, t = paired([per[hi][s] - per[lo][s] for s in SEEDS])
        b = "VOID" if not (gate0 and gate1) else bar(m, t)
        inc[name] = dict(mean=m, se=se, t=t, bar=b, a42=A42_STEPS[name])
        print(f"    {name:<9}{m:+.4f} se {se:.4f} t {t:+6.2f}  {b:<8}"
              f"{A42_STEPS[name]:+.4f}")
    vd = inc["duration"]["bar"]
    print(f"    THE REGISTERED QUESTION is the DURATION increment and only"
          f" it: {vd}")
    kept = {n: ("kept" if inc[n]["bar"] == b42 else "changed")
            for n, b42 in (("speed", "REAL"), ("angle", "NULL"))}
    print(f"    A42's other two verdicts: speed {kept['speed']},"
          f" angle {kept['angle']}")

    m2, se2, t2 = paired([per["h0p1"][s] - per["h01"][s] for s in SEEDS])
    b2 = "VOID" if not (gate0 and gate1) else bar(m2, t2)
    print(f"\n  READ 2, the pair head against no pair head:"
          f"\n    h0p1 minus h01  {m2:+.4f} se {se2:.4f} t {t2:+.2f}"
          f"   {b2}    A42 on six seeds +0.0052 t +1.96")

    print(f"\n  READ 3, descriptive, each arm against the corrected zero:")
    print(f"    {'arm':<8}{'vs L3_FULL, twelve seeds':>28}"
          f"{'A42 vs L0_RAW, six':>21}")
    r3 = {}
    for a in ["h01"] + LADDER:
        m, se, t = paired([per[a][s] - per[ZERO][s] for s in SEEDS])
        r3[a] = dict(mean=m, se=se, t=t, a42=A42_READ3[a])
        print(f"    {a:<8}{m:+.4f} se {se:.4f} t {t:+6.2f}"
              f"          {A42_READ3[a]:+.4f}")
    shift = float(np.mean([r3[a]["mean"] - A42_READ3[a]
                           for a in ["h01"] + LADDER]))
    print(f"    mean shift against A42's recorded values {shift:+.4f},"
          f" A52's displacement of the zero is +0.0054")

    res = dict(k=K, seeds=SEEDS, integrity_gate=bool(gate0),
               h0p1_recheck=p1check, validity_gate=bool(gate1),
               validity_steps=gsteps, increments=inc, duration_verdict=vd,
               read2=dict(mean=m2, se=se2, t=t2, bar=b2),
               read3=r3, read3_mean_shift=shift,
               auc={a: {str(s): per[a][s] for s in SEEDS} for a in ALL})
    with open("research/w4_e1chan2.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1chan2.json")
    print("  one trajectory per row, no selection, diagnostic only,"
          " never a training signal, no serve decision")

    rid = ledger.append_row(
        "w4_e1chan2",
        {"seeds": SEEDS, "k": K, "perm_seed": PERM_SEED, "ladder": LADDER,
         "zero": ZERO, "reference": "A42 channel ladder, A52 corrected zero"},
        "ok" if (gate0 and gate1) else "failed",
        metrics={"integrity_gate": int(gate0), "validity_gate": int(gate1),
                 "speed": inc["speed"]["mean"], "speed_t": inc["speed"]["t"],
                 "angle": inc["angle"]["mean"], "angle_t": inc["angle"]["t"],
                 "duration": inc["duration"]["mean"],
                 "duration_t": inc["duration"]["t"],
                 "total": inc["total"]["mean"], "read2": m2, "read2_t": t2},
        artifacts=["research/w4_e1chan2.json"],
        notes=f"AMENDMENT 55 the event 1 channel ladder at twelve seeds."
              f" Duration increment {inc['duration']['mean']:+.4f} t"
              f" {inc['duration']['t']:+.2f}, {vd}. Speed"
              f" {inc['speed']['mean']:+.4f} t {inc['speed']['t']:+.2f},"
              f" angle {inc['angle']['mean']:+.4f} t"
              f" {inc['angle']['t']:+.2f}. Diagnostic only, registered in"
              f" advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
