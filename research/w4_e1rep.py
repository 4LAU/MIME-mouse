"""w4_e1rep. AMENDMENT 46, registered in step0_prereg.md before this
file existed.

A45 split the pair head's gap above the human floor into three terms.
Two of the three were BETWEEN rather than REAL on six seeds. This
replicates the decomposition on six new seeds, 46 to 51, and combines
to twelve. No new instrument and no new arm: h0np1k1 is reused exactly
as committed.

The validity gate is read first and the three terms measured on the
NEW seeds alone must keep the sign they had on the old ones. READ 4 is
an arithmetic identity and a check on the code, not a finding.

Scored the A40 way, K 20 row permutations, paired on identical rows.
CPU only, no generation, diagnostic, never a training signal, no
selection.
"""
import json
import sys

import numpy as np

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import scoring   # noqa: E402
import ledger    # noqa: E402

OLD = [40, 41, 42, 43, 44, 45]
NEW = [46, 47, 48, 49, 50, 51]
ALL = OLD + NEW
K = 20
PERM_SEED = 3208
ARMS = {"HUM": "research/w4_e1floor_F_L0_RAW_s{s}.npy",
        "H02": "research/w4_e1feat_F_h02_s{s}.npy",
        "K1": "research/w4_e1feat_F_h0np1k1_s{s}.npy",
        "H0P1": "research/w4_e1feat_F_h0p1_s{s}.npy"}
TERMS = [("read1", "H0P1", "K1", "modelling p(e1 | e0, cond)"),
         ("read2", "K1", "H02", "content not a function of (e0, cond)"),
         ("read3", "H02", "HUM", "the A41 continuation residual")]


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


def paired(a, b, per, seeds):
    d = np.array([per[a][s] - per[b][s] for s in seeds])
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return m, se, (m / se if se > 0 else float("inf")), [float(x) for x in d]


def main():
    print(f"  every arm value is the mean of K={K} permutations\n", flush=True)
    per, sds = {a: {} for a in ARMS}, {a: {} for a in ARMS}
    for s in ALL:
        for a, tmpl in ARMS.items():
            per[a][s], sds[a][s] = auc_mean(tmpl.format(s=s))
        tag = "old" if s in OLD else "NEW"
        print(f"  seed {s} ({tag}): " + "  ".join(
            f"{a} {per[a][s]:.4f}" for a in ARMS), flush=True)

    res = {"k": K, "old_seeds": OLD, "new_seeds": NEW,
           "per_seed": {a: {str(s): per[a][s] for s in ALL} for a in ARMS},
           "per_seed_sd": {a: {str(s): sds[a][s] for s in ALL} for a in ARMS}}

    print("\n  VALIDITY GATE (read first): every term must keep its sign on"
          " the six NEW seeds alone")
    gate_ok = True
    res["gate"] = {}
    for name, hi, lo, _ in TERMS:
        mo = paired(hi, lo, per, OLD)[0]
        mn, sen, tn, _ = paired(hi, lo, per, NEW)
        keeps = (mo > 0) == (mn > 0)
        gate_ok &= keeps
        res["gate"][name] = dict(old=mo, new=mn, new_se=sen, sign_kept=bool(keeps))
        print(f"    {hi:5s} minus {lo:5s}  old {mo:+.4f}   new {mn:+.4f}"
              f" se {sen:.4f}   sign {'KEPT' if keeps else 'FLIPPED'}")
    res["gate_passed"] = bool(gate_ok)
    if gate_ok:
        print("  GATE PASSED: the decomposition is stable under row"
              " resampling. The combined reads stand.")
    else:
        print("  GATE FAILED: a term flipped sign on fresh rows. The combined"
              " reads are VOID and the instability is the result (registered).")

    print(f"\n  THE DECOMPOSITION ON {len(ALL)} SEEDS")
    for a in ("HUM", "H02", "K1", "H0P1"):
        print(f"    {a:5s} {np.mean([per[a][s] for s in ALL]):.4f}")
    tot = 0.0
    for name, hi, lo, what in TERMS:
        m, se, t, d = paired(hi, lo, per, ALL)
        v = step_verdict(m, t)
        was = paired(hi, lo, per, OLD)[0]
        res[name] = dict(pair=f"{hi} minus {lo}", what=what, mean=m, se=se, t=t,
                         verdict=v, a45_value=was, moved=m - was, per_seed=d)
        tot += m
        print(f"\n  {name.upper()}: {hi} minus {lo}  mean {m:+.4f}  se {se:.4f}"
              f"  t {t:+.2f}   {v}")
        print(f"    {what}")
        print(f"    A45 had {was:+.4f} on six seeds, moved {m - was:+.4f}")

    total, se_t, t_t, _ = paired("H0P1", "HUM", per, ALL)
    gap = abs(tot - total)
    res["read4_identity"] = dict(sum_of_terms=tot, total=total, gap=gap,
                                 holds=bool(gap < 1e-9))
    print(f"\n  READ 4 (code check, not a finding): terms sum to {tot:+.6f},"
          f" H0P1 minus HUM is {total:+.6f}, gap {gap:.2e}"
          f"   {'HOLDS' if gap < 1e-9 else 'BROKEN, everything above is void'}")
    print(f"  total gap H0P1 minus HUM  mean {total:+.4f}  se {se_t:.4f}"
          f"  t {t_t:+.2f}")

    with open("research/w4_e1rep.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1rep.json")
    print("  K1 copies real human event 1 triples from other rows: a BOUND,"
          " never a servable model, no serve decision may cite it")
    print("  twelve seeds are twelve row samples from one corpus against one"
          " model: this reduces row sampling noise and nothing else")
    print("  one trajectory per row, no selection, diagnostic only,"
          " never a training signal, no headline from this")

    rid = ledger.append_row(
        "w4_e1rep",
        {"seeds": ALL, "n": 2000, "k": K, "perm_seed": PERM_SEED,
         "arms": list(ARMS), "reference": "w4_e1floor L0_RAW, corpus human rows"},
        "ok",
        metrics={"gate_passed": bool(gate_ok), "identity_holds": bool(gap < 1e-9),
                 "read1_mean": res["read1"]["mean"], "read1_t": res["read1"]["t"],
                 "read2_mean": res["read2"]["mean"], "read2_t": res["read2"]["t"],
                 "read3_mean": res["read3"]["mean"], "read3_t": res["read3"]["t"],
                 "total_mean": total, "total_t": t_t},
        artifacts=["research/w4_e1rep.json"],
        notes=f"AMENDMENT 46 replication of the A45 decomposition on twelve"
              f" seeds. Gate passed={gate_ok}. Modelling term"
              f" {res['read1']['mean']:+.4f} t {res['read1']['t']:+.2f}"
              f" ({res['read1']['verdict']}). Missing conditioning term"
              f" {res['read2']['mean']:+.4f} t {res['read2']['t']:+.2f}"
              f" ({res['read2']['verdict']}). Continuation residual"
              f" {res['read3']['mean']:+.4f} t {res['read3']['t']:+.2f}"
              f" ({res['read3']['verdict']}). Total {total:+.4f} t {t_t:+.2f}."
              f" Diagnostic only, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
