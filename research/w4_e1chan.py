"""w4_e1chan. AMENDMENT 42, registered in step0_prereg.md before this
file existed.

The channel ladder inside event 1. Every arm is handed the real human
event 0, so event 0 is held fixed and only event 1 varies: H0P1 draws
it from the pair head with nothing given, H0S1P with the real speed
given, H0ST1P with the real speed and angle given, H02 is the real
event 1 entirely. H01 is the same real event 0 with the AR
continuation running on and no pair head at all.

Scored the AMENDMENT 40 way, K 20 row permutations, paired on
identical rows. CPU only, no generation, diagnostic, never a training
signal, no selection, no serve decision.
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
RUNGS = ["H0P1", "H0S1P", "H0ST1P", "H02"]
ARMS = {"HUM": "research/w4_e1floor_F_L0_RAW_s{s}.npy",
        "H01": "research/w4_e1feat_F_h01_s{s}.npy",
        "H0P1": "research/w4_e1feat_F_h0p1_s{s}.npy",
        "H0S1P": "research/w4_e1feat_F_h0s1p_s{s}.npy",
        "H0ST1P": "research/w4_e1feat_F_h0st1p_s{s}.npy",
        "H02": "research/w4_e1feat_F_h02_s{s}.npy"}

# AMENDMENT 42 integrity gate. Single permutation contract AUCs from the
# earlier qladder ec runs of these same arms and seeds.
STORED = {
    "H0S1P":  {40: 0.5420, 41: 0.5615, 42: 0.5615, 43: 0.5421, 44: 0.5506, 45: 0.5421},
    "H0ST1P": {40: 0.5341, 41: 0.5487, 42: 0.5713, 43: 0.5366, 44: 0.5537, 45: 0.5378},
}
ARM_KEY = {"H0S1P": "h0s1p", "H0ST1P": "h0st1p"}
# What each step of the ladder hands the pair head.
STEP_LABEL = {"H0S1P_minus_H0P1": "the real speed",
              "H0ST1P_minus_H0S1P": "the real angle on top",
              "H02_minus_H0ST1P": "the real duration on top"}


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
    print("  AMENDMENT 42 integrity gate, regenerated arms against the"
          " stored single permutation values:", flush=True)
    bad = []
    for rung, cells in STORED.items():
        for s, want in cells.items():
            j = json.load(open(f"research/w4_qladder_chan_s{s}.json"))
            got = round(float(j["arms"][ARM_KEY[rung]]["contract"]), 4)
            if got != round(want, 4):
                bad.append((rung, s, want, got))
    if bad:
        for rung, s, want, got in bad:
            print(f"  MISMATCH {rung} seed {s}: stored {want:.4f} regenerated {got:.4f}")
        print(f"  GATE FAILED {len(bad)} of 12, generation path has drifted,"
              f" no read taken (registered)")
        ledger.append_row(
            "w4_e1chan", {"seeds": SEEDS, "k": K}, "failed",
            metrics={"gate_bad": len(bad)},
            notes="AMENDMENT 42 integrity gate failed: regenerated channel"
                  " arms do not reproduce their stored contract AUCs, no"
                  " read taken (registered).", tier=1)
        ledger.regenerate_leaderboard()
        sys.exit(2)
    print("  gate passed 12 of 12\n", flush=True)

    print(f"  every arm value is the mean of K={K} permutations\n", flush=True)
    per, sds = {a: {} for a in ARMS}, {a: {} for a in ARMS}
    order = ["HUM", "H01"] + RUNGS
    for s in SEEDS:
        for a, tmpl in ARMS.items():
            per[a][s], sds[a][s] = auc_mean(tmpl.format(s=s))
        print("  seed %d: " % s + "  ".join(
            f"{a} {per[a][s]:.4f}" for a in order), flush=True)

    res = {"k": K, "gate": "passed 12 of 12",
           "per_seed": {a: {str(s): per[a][s] for s in SEEDS} for a in ARMS},
           "per_seed_sd": {a: {str(s): sds[a][s] for s in SEEDS} for a in ARMS}}

    print("\n  READ 1 (PRIMARY), what each channel of event 1 buys:")
    res["read1"] = {}
    for a, b in zip(RUNGS[1:], RUNGS[:-1]):
        key = f"{a}_minus_{b}"
        m, se, t, d = paired(a, b, per)
        v = step_verdict(m, t)
        res["read1"][key] = dict(mean=m, se=se, t=t, verdict=v, per_seed=d,
                                 gives=STEP_LABEL[key])
        print(f"  {a:>7} minus {b:<7} ({STEP_LABEL[key]:>22})"
              f"  mean {m:+.4f}  se {se:.4f}  t {t:+.2f}   {v}")
    m, se, t, d = paired("H02", "H0P1", per)
    res["read1"]["H02_minus_H0P1"] = dict(mean=m, se=se, t=t,
                                          verdict=step_verdict(m, t), per_seed=d,
                                          gives="all three channels")
    print(f"  {'H02':>7} minus {'H0P1':<7} ({'all three channels':>22})"
          f"  mean {m:+.4f}  se {se:.4f}  t {t:+.2f}"
          f"   {res['read1']['H02_minus_H0P1']['verdict']}")
    tot = m
    if tot != 0:
        print("\n  share of the whole event 1 distance closed by each channel: "
              + "  ".join(f"{STEP_LABEL[k]} {res['read1'][k]['mean'] / tot:+.0%}"
                          for k in STEP_LABEL))
        res["read1_shares"] = {k: float(res["read1"][k]["mean"] / tot)
                               for k in STEP_LABEL}

    m, se, t, d = paired("H0P1", "H01", per)
    res["read2"] = dict(mean=m, se=se, t=t, verdict=step_verdict(m, t),
                        per_seed=d)
    print(f"\n  READ 2, the pair head against no pair head: H0P1 minus H01"
          f"  mean {m:+.4f}  se {se:.4f}  t {t:+.2f}   {res['read2']['verdict']}")
    if m > 0 and res["read2"]["verdict"] == "REAL":
        print("  the pair head is WORSE than letting the AR continuation run on")

    print("\n  READ 3, each arm minus HUM on the floor bars:")
    res["read3"] = {}
    for r in ["H01"] + RUNGS:
        m, se, t, d = paired(r, "HUM", per)
        res["read3"][r] = dict(mean=m, se=se, t=t, verdict=verdict(m, t),
                               per_seed=d)
        print(f"  {r:>7} minus HUM  mean {m:+.4f}  se {se:.4f}  t {t:+.2f}"
              f"   {res['read3'][r]['verdict']}")

    with open("research/w4_e1chan.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1chan.json")
    print("  one trajectory per row, no selection, diagnostic only,"
          " never a training signal, no headline from this")

    reals = [STEP_LABEL[k] for k in STEP_LABEL
             if res["read1"][k]["verdict"] == "REAL"]
    rid = ledger.append_row(
        "w4_e1chan",
        {"seeds": SEEDS, "n": 2000, "k": K, "perm_seed": PERM_SEED,
         "rungs": RUNGS, "reference": "w4_e1floor L0_RAW, corpus human rows"},
        "ok",
        metrics={f"read1_{k}_mean": res["read1"][k]["mean"] for k in res["read1"]}
        | {"read2_mean": res["read2"]["mean"], "read2_t": res["read2"]["t"]},
        artifacts=["research/w4_e1chan.json"],
        notes=f"AMENDMENT 42 channel ladder inside event 1, permutation"
              f" averaged, paired on identical rows. Gate passed 12 of 12."
              f" REAL channels: {', '.join(reals) if reals else 'none'}."
              f" READ 2 pair head minus free AR {res['read2']['mean']:+.4f}"
              f" t {res['read2']['t']:+.2f} ({res['read2']['verdict']})."
              f" Diagnostic only, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
