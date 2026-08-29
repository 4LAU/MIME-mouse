"""w4_e1spread. AMENDMENT 49 REVISION, registered in step0_prereg.md before
this file was rewritten.

A48 left an explanation on the table: the human corpus rows are much narrower
than the scoring anchor, so a residual measured as arm minus HUM might be a
spread difference rather than model error. The first version of this file
tested that by WIDENING the human rows up to each arm's spread. That
construction is invalid and its run was killed: most of these columns are
bounded below at zero with a long right tail, so widening the deviations
sends a large share of the left tail negative and the classifier separates
impossible rows at 0.9997. The failure is recorded in full in the prereg.

This version runs the same question in the direction that stays inside the
data. Each arm's spread is SHRUNK down to the human rows' spread, per column,
holding that column's mean fixed. A contraction toward the mean leaves every
value between the mean and where it already was, so it cannot produce a value
the arm did not already bracket. Per column scaling leaves every Pearson
correlation unchanged exactly.

If the residual against HUM is a spread artefact it collapses once both sides
carry the same spread. If it survives, it is model error.

The plain HUM and raw arm values are reused from w4_e1frac.json, which is
only valid if the rows and permutations match, so gate 0 recomputes one of
them and checks. CPU only, no generation. This constructs feature matrices
directly and never produces a trajectory: diagnostic only, never a training
signal, never a serve candidate, no selection.
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
PRIMARY = "H02"
GATE2_SLACK = 25
ANCHOR = "data/human_val_features_grpo.npy"
ARMS = {"HUM": "research/w4_e1floor_F_L0_RAW_s{s}.npy",
        "H02": "research/w4_e1feat_F_h02_s{s}.npy",
        "F25": "research/w4_e1feat_F_hf25_s{s}.npy",
        "F50": "research/w4_e1feat_F_hf50_s{s}.npy",
        "F75": "research/w4_e1feat_F_hf75_s{s}.npy"}
TARGETS = ["H02", "F25", "F50", "F75"]


def shrink(X, c):
    """Contract each column's deviations toward its own mean by c.

    Written as X + (c - 1)(X - mu) rather than the algebraically identical
    mu + c(X - mu) because only this form is BIT EXACT at c = 1, which is
    what gate 1 rests on. The plain form differs from X by up to 7.5e-09 on
    1998 of 2000 rows, measured before this file was run.
    """
    mu = X.mean(0)
    return X + (c - 1.0) * (X - mu)


def factors(target, ref):
    """Per column c = min(1, sd_ref / sd_target). Clipped at 1 so the
    transform is a pure contraction and can never widen a column."""
    return np.minimum(1.0, ref.std(0) / target.std(0))


def auc_mean(m):
    v = np.array([scoring.score_features(
        m[np.random.default_rng(PERM_SEED + k).permutation(len(m))])["auc_rf_oob"]
        for k in range(K)])
    return float(v.mean()), float(v.std(ddof=1))


def out_of_range(m, lo, hi):
    return int(((m < lo) | (m > hi)).any(1).sum())


def step_verdict(mean, t):
    if abs(mean) >= 0.005 and abs(t) >= 3.0:
        return "REAL"
    if abs(t) < 2.0:
        return "NULL"
    return "BETWEEN"


def paired(a, b):
    d = np.array([a[s] - b[s] for s in SEEDS])
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return m, se, (m / se if se > 0 else float("inf")), [float(x) for x in d]


def main():
    prev = json.load(open("research/w4_e1frac.json"))["per_seed"]
    raw = {a: {s: prev[a][str(s)] for s in SEEDS} for a in ARMS}

    A = np.load(ANCHOR)
    A = A[np.isfinite(A).all(1)]
    lo, hi = A.min(0), A.max(0)

    mats = {}
    for s in SEEDS:
        ms = {a: np.load(t.format(s=s)) for a, t in ARMS.items()}
        ok = np.ones(len(ms["HUM"]), dtype=bool)
        for m in ms.values():
            ok &= np.isfinite(m).all(1)
        mats[s] = {a: m[ok] for a, m in ms.items()}

    print(f"  every arm value is the mean of K={K} permutations,"
          f" {len(SEEDS)} seeds\n", flush=True)

    # GATE 0, reuse. Recompute one committed value on the same rows.
    chk, _ = auc_mean(mats[SEEDS[0]]["HUM"])
    d0 = abs(chk - raw["HUM"][SEEDS[0]])
    gate0 = d0 < 1e-12
    print(f"  GATE 0 (reuse of w4_e1frac.json): recomputed HUM seed"
          f" {SEEDS[0]} {chk:.10f} against committed"
          f" {raw['HUM'][SEEDS[0]]:.10f}, difference {d0:.2e}"
          f"   {'ok' if gate0 else 'FAILED'}", flush=True)
    if not gate0:
        print("  GATE 0 FAILED: the rows or permutations do not match the"
              " committed run, so nothing here can be paired against it."
              " Every number below is VOID.", flush=True)
        return

    # GATE 1, identity. c = 1 must return the input bit for bit.
    one = np.ones(mats[SEEDS[0]]["H02"].shape[1])
    gate1 = all(np.array_equal(shrink(mats[s][a], one), mats[s][a])
                for s in SEEDS for a in TARGETS)
    print(f"  GATE 1 (identity at c = 1, bit exact):"
          f" {'ok' if gate1 else 'FAILED'}", flush=True)
    if not gate1:
        print("  GATE 1 FAILED: the construction is not the identity at"
              " c = 1, so it is wrong. Every number below is VOID.",
              flush=True)
        return

    # build the shrunk arms and the calibration control
    shrunk, ctrl, cinfo, g2 = {a: {} for a in TARGETS}, {}, {}, []
    for s in SEEDS:
        M = mats[s]
        for a in TARGETS:
            c = factors(M[a], M["HUM"])
            S = shrink(M[a], c)
            shrunk[a][s] = S
            cinfo.setdefault(a, []).append(
                (float(c.min()), float(c.max()), int((c >= 1.0).sum())))
            g2.append((a, s, out_of_range(M[a], lo, hi),
                       out_of_range(S, lo, hi)))
        # the shrink operation applied to HUM itself, same factors as H02
        ctrl[s] = shrink(M["HUM"], factors(M[PRIMARY], M["HUM"]))

    # GATE 2, support. The shrink must not push rows outside the anchor.
    print("\n  GATE 2 (no row pushed outside the anchor's observed range,"
          f" bar is raw plus {GATE2_SLACK}):")
    bad = {}
    for a in TARGETS:
        rows = [(s, r, t) for (aa, s, r, t) in g2 if aa == a]
        worst = max(t - r for _, r, t in rows)
        fails = [s for s, r, t in rows if t > r + GATE2_SLACK]
        bad[a] = fails
        print(f"    {a}: raw {min(r for _, r, _ in rows)} to"
              f" {max(r for _, r, _ in rows)} rows,"
              f" shrunk {min(t for _, _, t in rows)} to"
              f" {max(t for _, _, t in rows)},"
              f" worst increase {worst}"
              f"   {'ok' if not fails else 'FAILED on seeds ' + str(fails)}")
    live = [a for a in TARGETS if not bad[a]]
    if not live:
        print("  GATE 2 FAILED ON EVERY ARM. Every number below is VOID.")
        return
    if len(live) < len(TARGETS):
        print(f"  VOID ARMS, reported not dropped: "
              f"{[a for a in TARGETS if bad[a]]}")

    print("\n  shrink factors and clipped columns per arm (seed 40):")
    for a in TARGETS:
        mn, mx, nc = cinfo[a][0]
        print(f"    {a}: c from {mn:.2f} to {mx:.2f},"
              f" {nc} of {mats[SEEDS[0]][a].shape[1]} columns clipped at 1")

    print("\n  scoring the shrunk arms and the control", flush=True)
    per = {a: {} for a in TARGETS}
    cper = {}
    for s in SEEDS:
        for a in TARGETS:
            per[a][s], _ = auc_mean(shrunk[a][s])
        cper[s], _ = auc_mean(ctrl[s])
        print(f"  seed {s}   HUM {raw['HUM'][s]:.4f}   "
              + "  ".join(f"{a} {raw[a][s]:.4f}>{per[a][s]:.4f}"
                          for a in TARGETS)
              + f"   ctrl {cper[s]:.4f}", flush=True)

    res = {"k": K, "seeds": SEEDS, "gate0_diff": d0, "gate1": bool(gate1),
           "gate2_void_arms": [a for a in TARGETS if bad[a]],
           "shrink_factors_seed40": {a: cinfo[a][0] for a in TARGETS},
           "raw": {a: {str(s): raw[a][s] for s in SEEDS} for a in ARMS},
           "shrunk": {a: {str(s): per[a][s] for s in SEEDS}
                      for a in TARGETS},
           "control": {str(s): cper[s] for s in SEEDS}}

    # SUPPORTING READ first: it can void the primary reads.
    cm, cse, ct, cd = paired(cper, raw["HUM"])
    res["supporting"] = dict(mean=cm, se=cse, t=ct,
                             verdict=step_verdict(cm, ct), per_seed=cd)
    print(f"\n  SUPPORTING READ (calibration), the shrink operation applied"
          f" to HUM itself:")
    print(f"    {cm:+.4f}  se {cse:.4f}  t {ct:+.2f}"
          f"   {step_verdict(cm, ct)}")
    print("    this is what the transform alone does to an AUC, no arm"
          " involved")

    print("\n  PRIMARY READ, shrunk arm minus plain HUM, against the raw"
          " residual:")
    reads = {}
    for a in live:
        rm, rse, rt, _ = paired(raw[a], raw["HUM"])
        sm, sse, st, sd = paired(per[a], raw["HUM"])
        if abs(st) < 2.0:
            v = "COLLAPSES"
        elif abs(sm) >= 0.005 and abs(st) >= 3.0 and abs(sm - rm) <= 0.005:
            v = "SURVIVES"
        else:
            v = "PARTIAL"
        reads[a] = dict(raw_mean=rm, raw_t=rt, mean=sm, se=sse, t=st,
                        verdict=v, per_seed=sd)
        print(f"    {a}: raw {rm:+.4f} (t {rt:+.2f})   shrunk {sm:+.4f}"
              f"  se {sse:.4f}  t {st:+.2f}   {v}")
    res["primary"] = reads

    # FAULT BRANCH
    lift = float(np.mean([per[a][s] - raw[a][s] for a in live for s in SEEDS]))
    res["shrink_lift"] = lift
    faulted = lift > 0.010
    res["fault_branch"] = bool(faulted)
    print(f"\n  FAULT BRANCH: mean shrunk minus raw over the live arms"
          f" {lift:+.4f}")
    if faulted:
        print("    ABOVE 0.010: the shrink is adding detectability on its"
              " own, so every primary read above is DESCRIPTIVE and decides"
              " nothing (registered).")
    elif abs(cm) > abs(reads[PRIMARY]["raw_mean"] - reads[PRIMARY]["mean"]):
        print("    the calibration control moves the AUC by more than the"
              " shift the primary read resolves, so the reads are"
              " DESCRIPTIVE (registered).")
    else:
        print("    clear, the primary reads stand as registered.")

    v = reads.get(PRIMARY, {}).get("verdict", "VOID")
    print(f"\n  {PRIMARY} IS THE TEST CASE: {v}")
    if v == "COLLAPSES":
        print("    the residual against HUM is a spread artefact. Every"
              " residual measured against the raw corpus since A41 has to be"
              " restated in spread corrected terms.")
    elif v == "SURVIVES":
        print("    equalising spread leaves the residual intact, so it is"
              " model error and the reference is sound. A48's negative rung"
              " stays an open puzzle.")
    else:
        print("    spread carries part of it, neither branch at the bars.")

    with open("research/w4_e1spread.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1spread.json")
    print("  feature matrices only, no trajectory, diagnostic only, never a"
          " training signal, no selection, no headline from this")

    rid = ledger.append_row(
        "w4_e1spread",
        {"seeds": SEEDS, "n": 2000, "k": K, "perm_seed": PERM_SEED,
         "construction": "per column contraction of each arm to HUM spread",
         "reference": "w4_e1floor L0_RAW, corpus human rows"},
        "ok",
        metrics={"gate0_diff": d0, "gate1": int(gate1),
                 "n_void_arms": len(TARGETS) - len(live),
                 "shrink_lift": lift, "fault_branch": int(faulted),
                 "control_mean": cm, "control_t": ct,
                 "primary_mean": reads.get(PRIMARY, {}).get("mean", 0.0),
                 "primary_t": reads.get(PRIMARY, {}).get("t", 0.0)},
        artifacts=["research/w4_e1spread.json"],
        notes=f"AMENDMENT 49 REVISION spread confound control. Arms shrunk"
              f" to the human rows' spread per column. {PRIMARY} primary"
              f" {reads.get(PRIMARY, {}).get('mean', 0.0):+.4f}"
              f" t {reads.get(PRIMARY, {}).get('t', 0.0):+.2f} ({v}),"
              f" raw was {reads.get(PRIMARY, {}).get('raw_mean', 0.0):+.4f}."
              f" The widening version of this test was invalid and was"
              f" killed, see the prereg. Diagnostic only, registered in"
              f" advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
