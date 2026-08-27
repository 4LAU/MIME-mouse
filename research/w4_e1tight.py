"""w4_e1tight. AMENDMENT 45, registered in step0_prereg.md before this
file existed.

A44 drew event 1 nonparametrically from one neighbourhood and the
primary read came out on the branch registered as void. A45 turns that
single point into a ladder: the same donor construction at four match
tightnesses, from the single nearest donor to a donor drawn with the
conditioning ignored entirely. The validity gate is read first and the
ladder must not run backwards.

Every h0np1* arm copies real human event 1 triples from OTHER rows, so
each is a BOUND and never a servable model. Scored the A40 way, K 20
row permutations, paired on identical rows. CPU only, no generation,
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
        "H02": "research/w4_e1feat_F_h02_s{s}.npy",
        "H0P1": "research/w4_e1feat_F_h0p1_s{s}.npy",
        "K1": "research/w4_e1feat_F_h0np1k1_s{s}.npy",
        "K8": "research/w4_e1feat_F_h0np1k8_s{s}.npy",
        "K64": "research/w4_e1feat_F_h0np1_s{s}.npy",
        "MARG": "research/w4_e1feat_F_h0np1m_s{s}.npy"}
LADDER = ["K1", "K8", "K64", "MARG"]


def auc_mean(path):
    m = np.load(path)
    m = m[np.isfinite(m).all(1)]
    v = np.array([scoring.score_features(
        m[np.random.default_rng(PERM_SEED + k).permutation(len(m))])["auc_rf_oob"]
        for k in range(K)])
    return float(v.mean()), float(v.std(ddof=1))


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


def read(name, a, b, per, res, note=""):
    m, se, t, d = paired(a, b, per)
    v = step_verdict(m, t)
    res[name] = dict(pair=f"{a} minus {b}", mean=m, se=se, t=t, verdict=v,
                     per_seed=d)
    print(f"\n  {name}: {a} minus {b}  mean {m:+.4f}  se {se:.4f}"
          f"  t {t:+.2f}   {v}")
    if note:
        print(f"  {note}")
    return m, se, t, v


def main():
    print(f"  every arm value is the mean of K={K} permutations\n", flush=True)
    per, sds = {a: {} for a in ARMS}, {a: {} for a in ARMS}
    for s in SEEDS:
        for a, tmpl in ARMS.items():
            per[a][s], sds[a][s] = auc_mean(tmpl.format(s=s))
        print("  seed %d: " % s + "  ".join(
            f"{a} {per[a][s]:.4f}" for a in ARMS), flush=True)

    res = {"k": K, "arms_are_bounds_not_models": True,
           "per_seed": {a: {str(s): per[a][s] for s in SEEDS} for a in ARMS},
           "per_seed_sd": {a: {str(s): sds[a][s] for s in SEEDS} for a in ARMS}}

    print("\n  THE LADDER, mean over seeds, tightest match first:")
    for a in LADDER:
        mu = float(np.mean([per[a][s] for s in SEEDS]))
        print(f"    {a:5s} {mu:.4f}   vs H02 {mu - np.mean([per['H02'][s] for s in SEEDS]):+.4f}")

    print("\n  VALIDITY GATE (read before anything else): K1 minus K64 must"
          " not be REAL and POSITIVE")
    mg, seg, tg, vg = read("gate", "K1", "K64", per, res)
    ok = not (vg == "REAL" and mg > 0)
    res["gate_passed"] = bool(ok)
    if ok:
        print("  GATE PASSED: a tighter match does not score worse than a"
              " looser one. The reads below stand.")
    else:
        print("  GATE FAILED: the ladder runs backwards. Every read below is"
              " VOID and the nonparametric route is abandoned, not respun"
              " (registered).")

    m1, se1, t1, v1 = read(
        "read1", "K1", "MARG", per, res,
        "does conditioning tightness buy anything at the contract")
    if ok:
        if v1 == "REAL" and m1 < 0:
            print("  TIGHTER CONDITIONING HELPS: the contract can tell a well"
                  " matched real e1 from a badly matched one.")
        elif v1 == "NULL":
            print("  CONDITIONING TIGHTNESS BUYS NOTHING: the contract"
                  " relevant content of e1 is not a usable function of"
                  " (e0, cond).")

    m2, se2, t2, v2 = read("read2", "K1", "H0P1", per, res)
    if ok:
        if v2 == "REAL" and m2 > 0:
            print("  the nearest real donor still loses to the head, and with"
                  " the gate passed that is about the conditioning, not the"
                  " neighbourhood")
        elif v2 in ("NULL",) or m2 < 0:
            print("  the A44 instrument is VALID at K 1 and the original"
                  " ceiling question is answered in this run")

    m3, se3, t3, v3 = read(
        "read3", "K1", "H02", per, res,
        "how far the nearest real human donor sits from the row's own e1")
    if ok and v3 == "REAL" and m3 > 0:
        print("  even the nearest real human e1 from a tightly matched"
              " situation cannot stand in for the row's own e1")

    with open("research/w4_e1tight.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1tight.json")
    print("  every h0np1 arm copies real human event 1 triples from other"
          " rows: a BOUND, never a servable model, no serve decision may"
          " cite one")
    print("  one trajectory per row, no selection, diagnostic only,"
          " never a training signal, no headline from this")

    rid = ledger.append_row(
        "w4_e1tight",
        {"seeds": SEEDS, "n": 2000, "k": K, "perm_seed": PERM_SEED,
         "arms": LADDER, "reference": "w4_e1floor L0_RAW, corpus human rows"},
        "ok",
        metrics={"gate_mean": mg, "gate_t": tg, "gate_passed": bool(ok),
                 "read1_mean": m1, "read1_t": t1, "read2_mean": m2,
                 "read2_t": t2, "read3_mean": m3, "read3_t": t3},
        artifacts=["research/w4_e1tight.json"],
        notes=f"AMENDMENT 45 conditioning tightness ladder. Gate K1 minus K64"
              f" {mg:+.4f} t {tg:+.2f}, passed={ok}. READ 1 K1 minus MARG"
              f" {m1:+.4f} t {t1:+.2f} ({v1}). READ 2 K1 minus H0P1"
              f" {m2:+.4f} t {t2:+.2f} ({v2}). READ 3 K1 minus H02"
              f" {m3:+.4f} t {t3:+.2f} ({v3}). All h0np1 arms copy real human"
              f" e1 from other rows: bounds, never servable models."
              f" Diagnostic only, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
