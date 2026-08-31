"""w4_e1rez2. AMENDMENT 57, registered in step0_prereg.md before this
file existed and before any AMENDMENT 56 number existed.

A53 assembled the opening forcing ladder against the corrected zero on
seeds 40 to 51 and released FIRST EVENT DOMINATES on a share of 0.509
whose jackknife standard error was 0.057, so the verdict word sat on its
own boundary. A53 also withdrew a six seed story that the first TWO
events dominate, leaving the h01 to h02 step at +0.0079 with no verdict.
This restates the whole ladder on twelve seeds nobody has scored, taking
k0 and h016 from A56's generation on the same seeds.

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

SEEDS = list(range(52, 64))
K = 20
PERM_SEED = 3208
ARM = "research/w4_e1feat_F_{a}_s{s}.npy"
FLOOR = "research/w4_e1floor_F_{a}_s{s}.npy"
ZERO = "L3_FULL"
LADDER = ["k0", "h01", "h02", "h016"]
ALL = [ZERO] + LADDER
STEPS = [("first event", "k0", "h01"),
         ("second event", "h01", "h02"),
         ("events 2 to 15", "h02", "h016"),
         ("total", "k0", "h016")]
# A53 on seeds 40 to 51, against L3_FULL
A53_STEPS = {"first event": 0.0199, "second event": 0.0079,
             "events 2 to 15": 0.0112, "total": 0.0390}
A53_LEVELS = {"k0": 0.0533, "h01": 0.0335, "h02": 0.0255, "h016": 0.0143}
A53_SHARE = 0.509
A53_SHARE_SE = 0.057
GATE_SEED = 52
GATE_CACHE = "/home/aaronadmin/w4_arms/h016_s52_a56.npy"


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


def jackknife(num, den):
    """se of the ratio of the means, leaving one seed out at a time."""
    n = len(num)
    full = [sum(num[:i] + num[i + 1:]) / sum(den[:i] + den[i + 1:])
            for i in range(n)]
    mj = float(np.mean(full))
    return float(np.sqrt((n - 1) / n * sum((f - mj) ** 2 for f in full)))


def main():
    print(f"  every value is the mean of K={K} permutations,"
          f" {len(SEEDS)} seeds {SEEDS[0]} to {SEEDS[-1]},"
          f" none scored before A56\n", flush=True)

    print(f"  INTEGRITY GATE (registered stop): h016 on seed {GATE_SEED} was"
          f" produced twice, in two separate runs of the same code:",
          flush=True)
    live = ARM.format(a="h016", s=GATE_SEED)
    a, b = np.load(GATE_CACHE), np.load(live)
    gate0 = bool(a.shape == b.shape and np.array_equal(a, b, equal_nan=True))
    print(f"    {a.shape[0]} rows:"
          f" {'bit identical across runs, ok' if gate0 else 'DIFFERS, every read is VOID (registered)'}",
          flush=True)
    shutil.copyfile(GATE_CACHE, live)
    print(f"    the A56 copy is restored to {live}")
    if not gate0:
        d = np.abs(np.nan_to_num(a) - np.nan_to_num(b))
        print(f"    rows differing {int((d > 0).any(1).sum())},"
              f" max abs {d.max():.3e}")
        return

    per = {x: {} for x in ALL}
    print(f"\n  rows: finite in all {len(ALL)} matrices", flush=True)
    for s in SEEDS:
        mats = {x: np.load(FLOOR.format(a=x, s=s) if x.startswith("L")
                           else ARM.format(a=x, s=s)) for x in ALL}
        ok = np.ones(len(mats[ZERO]), dtype=bool)
        for m in mats.values():
            ok &= np.isfinite(m).all(1)
        for x in ALL:
            per[x][s] = auc_mean(mats[x][ok])
        print(f"    seed {s} rows {int(ok.sum())}   "
              + "  ".join(f"{x} {per[x][s]:.4f}" for x in ALL), flush=True)

    print(f"\n  VALIDITY GATE (forcing more of the opening cannot make the"
          f" arm less human):", flush=True)
    gate1, gsteps = True, {}
    for lo, hi in zip(LADDER, LADDER[1:]):
        m, se, t = paired([per[hi][s] - per[lo][s] for s in SEEDS])
        bad = (m >= 0.005 and t >= 3.0)
        gate1 = gate1 and not bad
        gsteps[f"{lo}->{hi}"] = dict(mean=m, se=se, t=t)
        print(f"    {lo:>4} to {hi:<5} {m:+.4f}  se {se:.4f}  t {t:+.2f}"
              f"   {'RISES' if bad else 'ok'}", flush=True)
    print(f"  {'GATE PASSED' if gate1 else 'GATE FAILED, every read is VOID (registered)'}")
    if not gate1:
        return

    print(f"\n  READ 1 (PRIMARY) is the SECOND event's step and only it."
          f"  READ 2 is the replication.")
    print(f"    {'step':<16}{'twelve fresh seeds':>28}{'A53 on 40 to 51':>18}")
    inc = {}
    for name, hi, lo in STEPS:
        m, se, t = paired([per[hi][s] - per[lo][s] for s in SEEDS])
        b = bar(m, t)
        inc[name] = dict(mean=m, se=se, t=t, bar=b, a53=A53_STEPS[name])
        print(f"    {name:<16}{m:+.4f} se {se:.4f} t {t:+6.2f}  {b:<8}"
              f"{A53_STEPS[name]:+.4f}", flush=True)
    v = inc["second event"]["bar"]
    print(f"    THE REGISTERED QUESTION is the SECOND event's step: {v}")
    print(f"      {'REAL and positive means the opening cost is a TWO event phenomenon' if (v == 'REAL' and inc['second event']['mean'] > 0) else 'NULL means the first event stands alone' if v == 'NULL' else 'BETWEEN, no claim'}")

    print(f"\n  READ 3, the share, DESCRIPTIVE ONLY, carries no verdict"
          f" (A47 rule, and A53's own experience):")
    num = [per["k0"][s] - per["h01"][s] for s in SEEDS]
    den = [per["k0"][s] - per["h016"][s] for s in SEEDS]
    share = float(sum(num) / sum(den))
    sj = jackknife(num, den)
    print(f"    (k0 minus h01) / (k0 minus h016) = {share:.3f}"
          f"  jackknife se {sj:.3f}"
          f"   A53 {A53_SHARE:.3f} se {A53_SHARE_SE:.3f}", flush=True)
    print(f"    the half boundary is {'inside' if abs(share - 0.5) < 2 * sj else 'outside'}"
          f" two jackknife standard errors of this share")

    print(f"\n  READ 4, descriptive, each arm against the zero:")
    print(f"    {'arm':<6}{'twelve fresh seeds':>26}{'A53 on 40 to 51':>18}")
    r4 = {}
    for x in LADDER:
        m, se, t = paired([per[x][s] - per[ZERO][s] for s in SEEDS])
        r4[x] = dict(mean=m, se=se, t=t, a53=A53_LEVELS[x])
        print(f"    {x:<6}{m:+.4f} se {se:.4f} t {t:+6.2f}"
              f"        {A53_LEVELS[x]:+.4f}", flush=True)

    res = dict(k=K, seeds=SEEDS, integrity_gate=bool(gate0),
               validity_gate=bool(gate1), validity_steps=gsteps,
               steps=inc, primary=inc["second event"],
               share=dict(value=share, jackknife_se=sj, a53=A53_SHARE,
                          a53_se=A53_SHARE_SE),
               levels=r4,
               auc={x: {str(s): per[x][s] for s in SEEDS} for x in ALL})
    with open("research/w4_e1rez2.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1rez2.json")
    print("  one trajectory per row, no selection, diagnostic only,"
          " never a training signal, no serve decision")

    rid = ledger.append_row(
        "w4_e1rez2",
        {"seeds": SEEDS, "k": K, "perm_seed": PERM_SEED, "ladder": LADDER,
         "zero": ZERO, "reference": "A53 opening ladder, fresh seeds"},
        "ok" if (gate0 and gate1) else "failed",
        metrics={"integrity_gate": int(gate0), "validity_gate": int(gate1),
                 "second_event": inc["second event"]["mean"],
                 "second_event_t": inc["second event"]["t"],
                 "first_event": inc["first event"]["mean"],
                 "first_event_t": inc["first event"]["t"],
                 "total": inc["total"]["mean"],
                 "share": share, "share_jackknife_se": sj},
        artifacts=["research/w4_e1rez2.json"],
        notes=f"AMENDMENT 57 A53's opening ladder on twelve unscored seeds."
              f" Second event step {inc['second event']['mean']:+.4f} t"
              f" {inc['second event']['t']:+.2f}, {v}. First event"
              f" {inc['first event']['mean']:+.4f} t"
              f" {inc['first event']['t']:+.2f}, total"
              f" {inc['total']['mean']:+.4f}, share {share:.3f} se {sj:.3f}."
              f" Diagnostic only, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
