"""w4_e1cont. AMENDMENT 47, registered in step0_prereg.md before this file
existed.

A46 put the whole surviving opening gap in the continuation: H02 minus HUM
is +0.0195 se 0.0038 t +5.18 on twelve seeds, and both opening terms are
undecided. H02 hands the model the first two human events and lets it
generate everything after them, which on a median 39 event row is almost the
entire trajectory. This amendment asks WHERE inside that stretch the defect
lives, by forcing the first k human events for k = 2, 4, 8, 16 and reading
the residual against the same HUM reference at each rung.

THE SATURATION CONFOUND. Row eligibility is length above 4, so a row can be
shorter than the forced prefix. models/event_ar.py forces the pad class at
those trailing positions and a forced pad terminates the row exactly as a
generated one would, so a saturated row renders the full human token round
trip with no model continuation left in it to be wrong. Saturated rows
therefore push the residual DOWN and mimic a defect concentrated at the
front, which is one of the two verdicts on offer. The PRIMARY ladder is
therefore restricted to rows of length above 16, where every rung is
saturation free by construction and all four rungs sit on identical rows.
The unfiltered ladder is reported beside it and decides nothing.

Scored the A40 way, K 20 row permutations, paired on identical rows, twelve
seeds per the A46 row draw rule. CPU only, no generation. Diagnostic only,
never a training signal, no selection, no serve decision.
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
KMAX = 4
TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
MIN_LEN = 16          # the primary ladder keeps rows LONGER than this
RUNGS = [2, 4, 8, 16]
ARMS = {"HUM": "research/w4_e1floor_F_L0_RAW_s{s}.npy",
        "H02": "research/w4_e1feat_F_h02_s{s}.npy",
        "H04": "research/w4_e1feat_F_h04_s{s}.npy",
        "H08": "research/w4_e1feat_F_h08_s{s}.npy",
        "H016": "research/w4_e1feat_F_h016_s{s}.npy"}
ARM_OF = {2: "H02", 4: "H04", 8: "H08", 16: "H016"}


def row_lengths(seed, lengths, held):
    """The exact w4_qladder pick rule, so row i here is row i there."""
    elig = held[lengths[held] > KMAX]
    pick = np.sort(np.random.default_rng(1000 + seed)
                   .choice(elig, 2000, replace=False))
    return lengths[pick]


def auc_mean(m):
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


def paired(vals_a, vals_b):
    d = np.array([vals_a[s] - vals_b[s] for s in SEEDS])
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return m, se, (m / se if se > 0 else float("inf")), [float(x) for x in d]


def main():
    lengths = np.load("training/events_len.npy")
    N = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(N, min(N, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(N), trained)

    print(f"  every arm value is the mean of K={K} permutations,"
          f" {len(SEEDS)} seeds\n", flush=True)

    # saturation on the unfiltered pool, the reason the primary ladder filters
    L0 = row_lengths(SEEDS[0], lengths, held)
    sat = {k: float((L0 <= k).mean()) for k in RUNGS}
    print("  saturated fraction of the unfiltered rows, seed %d:" % SEEDS[0])
    print("   " + "  ".join(f"k={k} {100 * sat[k]:.1f}%" for k in RUNGS))
    print(f"  median row length {int(np.median(L0))}\n", flush=True)

    pri = {a: {} for a in ARMS}
    sec = {a: {} for a in ARMS}
    kept = {}
    for s in SEEDS:
        mats = {a: np.load(t.format(s=s)) for a, t in ARMS.items()}
        ok = np.ones(len(mats["HUM"]), dtype=bool)
        for m in mats.values():
            ok &= np.isfinite(m).all(1)
        long_ = row_lengths(s, lengths, held) > MIN_LEN
        keep_p = ok & long_
        kept[s] = (int(ok.sum()), int(keep_p.sum()))
        for a, m in mats.items():
            sec[a][s], _ = auc_mean(m[ok])
            pri[a][s], _ = auc_mean(m[keep_p])
        print(f"  seed {s} rows {kept[s][1]} of {kept[s][0]}   "
              + "  ".join(f"{a} {pri[a][s]:.4f}" for a in ARMS), flush=True)

    res = {"k": K, "seeds": SEEDS, "min_len": MIN_LEN, "saturation": sat,
           "rows_kept": {str(s): kept[s] for s in SEEDS},
           "per_seed_primary": {a: {str(s): pri[a][s] for s in SEEDS}
                                for a in ARMS},
           "per_seed_secondary": {a: {str(s): sec[a][s] for s in SEEDS}
                                  for a in ARMS}}

    def residuals(per):
        out = {}
        for k in RUNGS:
            m, se, t, d = paired(per[ARM_OF[k]], per["HUM"])
            out[k] = dict(mean=m, se=se, t=t, verdict=verdict(m, t), per_seed=d)
        return out

    rp, rs = residuals(pri), residuals(sec)
    res["primary"], res["secondary"] = rp, rs

    print("\n  THE PRIMARY LADDER, rows longer than %d, residual against HUM:"
          % MIN_LEN)
    for k in RUNGS:
        r = rp[k]
        print(f"    r({k:2d})  {r['mean']:+.4f}  se {r['se']:.4f}"
              f"  t {r['t']:+.2f}   {r['verdict']}")

    print("\n  VALIDITY GATE (read before anything else): the primary ladder"
          " must not RISE with k")
    gate_ok, gate_rows = True, []
    for i in range(1, len(RUNGS)):
        lo, hi = RUNGS[i - 1], RUNGS[i]
        m, se, t, d = paired(pri[ARM_OF[hi]], pri[ARM_OF[lo]])
        bad = m >= 0.005 and t >= 3.0
        gate_ok = gate_ok and not bad
        gate_rows.append(dict(step=f"r({hi}) minus r({lo})", mean=m, se=se,
                             t=t, violates=bool(bad)))
        print(f"    r({hi}) minus r({lo})  {m:+.4f}  se {se:.4f}  t {t:+.2f}"
              f"   {'VIOLATION' if bad else 'ok'}")
    res["gate"] = {"passed": bool(gate_ok), "steps": gate_rows}
    if not gate_ok:
        print("  GATE FAILED: forcing more human events made the trajectory"
              " LESS human, so the arms are mislabelled. The ladder is VOID"
              " and the correct action is to find the labelling bug"
              " (registered).")
        with open("research/w4_e1cont.json", "w") as fh:
            json.dump(res, fh, indent=1)
        print("  wrote research/w4_e1cont.json")
        return
    print("  GATE PASSED, the ladder is monotone non increasing.")

    r2, r8, r16 = rp[2]["mean"], rp[8]["mean"], rp[16]["mean"]
    f8 = r8 / r2 if r2 else float("nan")
    f16 = r16 / r2 if r2 else float("nan")
    if f8 <= 0.50:
        shape = "FRONT LOADED"
        gloss = ("the defect is concentrated in the first handful of"
                 " generated events")
    elif 0.48 <= f16 <= 0.72:
        shape = "SPREAD"
        gloss = ("the defect is spread evenly over the movement, the residual"
                 " tracks the fraction of events the model still generates")
    else:
        shape = "OTHER"
        gloss = ("the residual barely responds to forcing the opening, so the"
                 " defect lives in the tail" if f16 > 0.72 else
                 "neither registered shape fits")
    res["read1"] = dict(shape=shape, frac_at_8=float(f8), frac_at_16=float(f16))
    print(f"\n  READ 1 (PRIMARY), THE SHAPE: r(8) is {f8:.2f} of r(2),"
          f" r(16) is {f16:.2f} of r(2)")
    print(f"  {shape}: {gloss}")

    res["read2"] = rp[16]
    print(f"\n  READ 2: r(16) {rp[16]['mean']:+.4f}  se {rp[16]['se']:.4f}"
          f"  t {rp[16]['t']:+.2f}   {rp[16]['verdict']}")
    print("  after sixteen forced human events the model's continuation of the"
          f" rest still sits {rp[16]['mean']:+.4f} from real human data")

    m3, se3, t3, d3 = paired(
        {s: pri["H02"][s] - pri["H016"][s] for s in SEEDS},
        {s: 0.0 for s in SEEDS})
    res["read3"] = dict(mean=m3, se=se3, t=t3, verdict=step_verdict(m3, t3),
                        per_seed=d3)
    print(f"\n  READ 3: r(2) minus r(16)  {m3:+.4f}  se {se3:.4f}  t {t3:+.2f}"
          f"   {res['read3']['verdict']}")
    print("  that is the share of the continuation residual living in"
          " events 2 through 16")

    print("\n  SECONDARY, all rows, descriptive only, decides nothing:")
    for k in RUNGS:
        r = rs[k]
        print(f"    r({k:2d})  {r['mean']:+.4f}  se {r['se']:.4f}"
              f"  t {r['t']:+.2f}    saturated {100 * sat[k]:.1f}%")
    fs16 = rs[16]["mean"] / rs[2]["mean"] if rs[2]["mean"] else float("nan")
    faster = bool(fs16 < f16)
    res["secondary_falls_faster"] = faster
    print(f"  r(16)/r(2) is {fs16:.2f} unfiltered against {f16:.2f} filtered:"
          f" the unfiltered ladder falls"
          f" {'FASTER, as saturation predicts' if faster else 'no faster'}")

    with open("research/w4_e1cont.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1cont.json")
    print("  one trajectory per row, no selection, diagnostic only,"
          " never a training signal, no headline from this")

    rid = ledger.append_row(
        "w4_e1cont",
        {"seeds": SEEDS, "n": 2000, "k": K, "perm_seed": PERM_SEED,
         "rungs": RUNGS, "min_len": MIN_LEN,
         "reference": "w4_e1floor L0_RAW, corpus human rows"},
        "ok",
        metrics={f"r{k}_mean": rp[k]["mean"] for k in RUNGS}
        | {f"r{k}_t": rp[k]["t"] for k in RUNGS}
        | {"gate_passed": int(gate_ok), "frac_at_8": float(f8),
           "frac_at_16": float(f16), "read3_mean": m3, "read3_t": t3},
        artifacts=["research/w4_e1cont.json"],
        notes=f"AMENDMENT 47 deep continuation ladder. Gate PASSED. Primary"
              f" ladder on rows longer than {MIN_LEN}, saturation free:"
              f" r(2) {rp[2]['mean']:+.4f}, r(4) {rp[4]['mean']:+.4f},"
              f" r(8) {rp[8]['mean']:+.4f}, r(16) {rp[16]['mean']:+.4f}."
              f" READ 1 {shape}. Diagnostic only, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
