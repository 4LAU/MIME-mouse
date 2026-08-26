"""w4_e1lad. AMENDMENT 41, registered in step0_prereg.md before this
file existed.

The human prefix ladder, scored the AMENDMENT 40 way: K 20 row
permutations per arm per seed, paired against the same real human
reference on identical rows. Answers where the prefix stops paying.
CPU only, no generation, diagnostic, never a training signal, no
selection, no serve decision.
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
RUNGS = ["K0", "H01", "H02", "H03", "H04"]
ARMS = {"HUM": "research/w4_e1floor_F_L0_RAW_s{s}.npy",
        "K0": "research/w4_e1feat_F_k0_s{s}.npy",
        "H01": "research/w4_e1feat_F_h01_s{s}.npy",
        "H02": "research/w4_e1feat_F_h02_s{s}.npy",
        "H03": "research/w4_e1feat_F_h03_s{s}.npy",
        "H04": "research/w4_e1feat_F_h04_s{s}.npy"}

# AMENDMENT 41 integrity gate. Single permutation contract AUCs from the
# earlier qladder runs of these same arms and seeds. Generation is
# deterministic given the seed, so the regenerated arms must reproduce
# these to four decimals.
STORED = {
    "K0":  {40: 0.5827, 41: 0.5661, 42: 0.5638, 43: 0.5890, 44: 0.5814, 45: 0.5635},
    "H01": {40: 0.5459, 41: 0.5509, 42: 0.5599, 43: 0.5520, 44: 0.5545, 45: 0.5533},
    "H03": {40: 0.5414, 41: 0.5337, 42: 0.5455, 43: 0.5520, 44: 0.5307, 45: 0.5301},
    "H04": {40: 0.5516, 41: 0.5444, 42: 0.5465, 43: 0.5372, 44: 0.5417, 45: 0.5346},
}
ARM_KEY = {"K0": "k0", "H01": "h01", "H03": "h03", "H04": "h04"}


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
    print("  AMENDMENT 41 integrity gate, regenerated arms against the"
          " stored single permutation values:", flush=True)
    bad = []
    for rung, cells in STORED.items():
        for s, want in cells.items():
            j = json.load(open(f"research/w4_qladder_lad_s{s}.json"))
            got = round(float(j["arms"][ARM_KEY[rung]]["contract"]), 4)
            if got != round(want, 4):
                bad.append((rung, s, want, got))
    if bad:
        for rung, s, want, got in bad:
            print(f"  MISMATCH {rung} seed {s}: stored {want:.4f} regenerated {got:.4f}")
        print(f"  GATE FAILED {len(bad)} of 24, generation path has drifted,"
              f" no read taken (registered)")
        ledger.append_row(
            "w4_e1lad", {"seeds": SEEDS, "k": K}, "failed",
            metrics={"gate_bad": len(bad)},
            notes="AMENDMENT 41 integrity gate failed: regenerated prefix"
                  " ladder arms do not reproduce their stored contract AUCs,"
                  " no read taken (registered).", tier=1)
        ledger.regenerate_leaderboard()
        sys.exit(2)
    print("  gate passed 24 of 24\n", flush=True)

    print(f"  every arm value is the mean of K={K} permutations\n", flush=True)
    per, sds = {a: {} for a in ARMS}, {a: {} for a in ARMS}
    for s in SEEDS:
        for a, tmpl in ARMS.items():
            per[a][s], sds[a][s] = auc_mean(tmpl.format(s=s))
        print("  seed %d: " % s + "  ".join(
            f"{a} {per[a][s]:.4f}" for a in ["HUM"] + RUNGS), flush=True)

    res = {"k": K, "gate": "passed 24 of 24",
           "per_seed": {a: {str(s): per[a][s] for s in SEEDS} for a in ARMS},
           "per_seed_sd": {a: {str(s): sds[a][s] for s in SEEDS} for a in ARMS}}

    print("\n  READ 1 (PRIMARY), the ladder profile, each rung minus HUM:")
    res["read1"] = {}
    for r in RUNGS:
        m, se, t, d = paired(r, "HUM", per)
        v = verdict(m, t)
        res["read1"][r] = dict(mean=m, se=se, t=t, verdict=v, per_seed=d)
        print(f"  {r:>4} minus HUM  mean {m:+.4f}  se {se:.4f}  t {t:+.2f}"
              f"   {v}")

    print("\n  READ 2, rung to rung increments:")
    res["read2"] = {}
    for a, b in zip(RUNGS[1:], RUNGS[:-1]):
        m, se, t, d = paired(a, b, per)
        v = step_verdict(m, t)
        res["read2"][f"{a}_minus_{b}"] = dict(mean=m, se=se, t=t, verdict=v,
                                              per_seed=d)
        print(f"  {a:>4} minus {b:<4} mean {m:+.4f}  se {se:.4f}"
              f"  t {t:+.2f}   {v}")

    m, se, t, d = paired("H04", "HUM", per)
    res["read3"] = dict(mean=m, se=se, t=t, verdict=verdict(m, t), per_seed=d)
    print(f"\n  READ 3, the residual after four correct human events:"
          f" H04 minus HUM mean {m:+.4f} se {se:.4f} t {t:+.2f}"
          f"   {res['read3']['verdict']}")

    with open("research/w4_e1lad.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1lad.json")
    print("  one trajectory per row, no selection, diagnostic only,"
          " never a training signal, no headline from this")

    first_null = next((k for k, v in res["read2"].items()
                       if v["verdict"] == "NULL"), "none")
    rid = ledger.append_row(
        "w4_e1lad",
        {"seeds": SEEDS, "n": 2000, "k": K, "perm_seed": PERM_SEED,
         "rungs": RUNGS, "reference": "w4_e1floor L0_RAW, corpus human rows"},
        "ok",
        metrics={f"read1_{r}_mean": res["read1"][r]["mean"] for r in RUNGS}
        | {"read3_mean": m, "read3_t": t},
        artifacts=["research/w4_e1lad.json"],
        notes=f"AMENDMENT 41 human prefix ladder, permutation averaged,"
              f" paired on identical rows. Gate passed 24 of 24. First null"
              f" increment {first_null}. Residual after four human events"
              f" {m:+.4f} t {t:+.2f} ({res['read3']['verdict']}). Diagnostic"
              f" only, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
