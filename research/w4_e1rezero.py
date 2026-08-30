"""w4_e1rezero. AMENDMENT 53, registered in step0_prereg.md before this file
existed.

A52 moved the zero: every arm is a token round trip of the human events, so
the raw corpus featurisation is a reference no arm can produce, and L3_FULL
is the one they can. This assembles the forcing ladder against it, from the
free running arm down to sixteen forced events, on identical rows.

The question is the one A19 through A34 circled: how much of the free
running arm's detectability is bought back by forcing the FIRST real event,
against the rest of the prefix.

CPU only, no model generation. Diagnostic only, never a training signal,
never a serve candidate, no selection of trajectories.
"""
import json
import sys

import numpy as np

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import scoring   # noqa: E402
import ledger    # noqa: E402

SEEDS6 = list(range(40, 46))     # what the k0, h01 and h03 caches hold
SEEDS12 = list(range(40, 52))
K = 20
PERM_SEED = 3208
N = 2000
MAX_T = 256
KMAX = 4
TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
MIN_LEN = 16                     # A47's filter for the third read
FLOOR = "research/w4_e1floor_F_{a}_s{s}.npy"
ARM = "research/w4_e1feat_F_{a}_s{s}.npy"
ZERO = "L3_FULL"
RAW = "L0_RAW"
LADDER = ["k0", "h01", "h02", "h03", "h04", "h08", "h016"]
A47_ARMS = ["h02", "h04", "h08", "h016"]
# A47's recorded primary ladder against L0_RAW, rows longer than 16
A47_RECORDED = {"h02": 0.0142, "h04": 0.0169, "h08": 0.0127, "h016": 0.0097}


def auc_mean(m):
    v = np.array([scoring.score_features(
        m[np.random.default_rng(PERM_SEED + k).permutation(len(m))])["auc_rf_oob"]
        for k in range(K)])
    return float(v.mean())


def paired(a, b, seeds):
    d = np.array([a[s] - b[s] for s in seeds])
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return m, se, (m / se if se > 0 else float("inf"))


def row_lengths(seed, lengths, elig):
    pick = np.sort(np.random.default_rng(1000 + seed).choice(elig, N,
                                                             replace=False))
    return np.minimum(lengths[pick], MAX_T).astype(np.int64)


def load(seed, arms):
    out = {}
    for a in arms:
        out[a] = np.load(FLOOR.format(a=a, s=seed) if a.startswith("L")
                         else ARM.format(a=a, s=seed))
    return out


def main():
    lengths = np.load("training/events_len.npy")
    NT = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(NT, min(NT, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(NT), trained)
    elig = held[lengths[held] > KMAX]

    print(f"  every value is the mean of K={K} permutations\n", flush=True)
    print("  BLIND: k0, h01, h03 against either zero, and the whole ladder"
          " assembled on identical rows.")
    print("  SEEN: h02, h04, h08, h016 against L3_FULL on twelve seeds"
          " (A52). The A48 fraction ladder is not re read here.\n", flush=True)

    arms = [RAW, ZERO] + LADDER
    per = {a: {} for a in arms}
    for s in SEEDS6:
        mats = load(s, arms)
        ok = np.ones(N, dtype=bool)
        for m in mats.values():
            ok &= np.isfinite(m).all(1)
        for a in arms:
            per[a][s] = auc_mean(mats[a][ok])
        print(f"  seed {s} rows {int(ok.sum())}   "
              + "  ".join(f"{a} {per[a][s]:.4f}" for a in arms), flush=True)

    # VALIDITY GATE, read before anything else
    print(f"\n  VALIDITY GATE (the ladder must not RISE as more real events"
          f" are forced):", flush=True)
    gate_ok, gate = True, {}
    for lo, hi in zip(LADDER, LADDER[1:]):
        m, se, t = paired(per[hi], per[lo], SEEDS6)
        bad = (m >= 0.005 and t >= 3.0)
        gate_ok = gate_ok and not bad
        gate[f"{lo}->{hi}"] = dict(mean=m, se=se, t=t)
        print(f"    {lo:>4} to {hi:<5} {m:+.4f}  se {se:.4f}  t {t:+.2f}"
              f"   {'RISES' if bad else 'ok'}", flush=True)
    print(f"  {'GATE PASSED' if gate_ok else 'GATE FAILED, every read below is VOID (registered)'}")

    # the ladder itself, against both zeros
    print(f"\n  THE LADDER, six seeds, paired on identical rows:")
    print(f"    {'arm':<6}{'vs L3_FULL (the reachable zero)':>34}"
          f"{'vs L0_RAW (as the record has it)':>36}")
    lad = {}
    for a in LADDER:
        nm, nse, nt = paired(per[a], per[ZERO], SEEDS6)
        om, ose, ot = paired(per[a], per[RAW], SEEDS6)
        lad[a] = dict(vs_zero=dict(mean=nm, se=nse, t=nt),
                      vs_raw=dict(mean=om, se=ose, t=ot))
        print(f"    {a:<6}{nm:+.4f} se {nse:.4f} t {nt:+6.2f}"
              f"        {om:+.4f} se {ose:.4f} t {ot:+6.2f}")

    # PRIMARY READ
    first, fse, ft = paired(per["k0"], per["h01"], SEEDS6)
    tot, tse, tt = paired(per["k0"], per["h016"], SEEDS6)
    share = first / tot if tot != 0 else float("nan")
    underpowered = (abs(first) < 0.015 or abs(ft) < 4.0)
    if first >= 0.010 and ft >= 3.0 and share >= 0.5:
        v = "FIRST EVENT DOMINATES"
    elif share < 0.25:
        v = "SPREAD ACROSS THE PREFIX"
    else:
        v = "BETWEEN"
    if not gate_ok:
        v = "VOID, gate failed"
    print(f"\n  PRIMARY READ, what the first forced event buys:")
    print(f"    k0 minus h01   {first:+.4f}  se {fse:.4f}  t {ft:+.2f}")
    print(f"    k0 minus h016  {tot:+.4f}  se {tse:.4f}  t {tt:+.2f}")
    print(f"    share {share:.2f}   {v}")
    if underpowered:
        print(f"    UNDERPOWERED by the registered trigger (|mean| under"
              f" 0.015 or |t| under 4 on six seeds). The verdict above is"
              f" NOT quoted until seeds 46 to 51 exist for k0, h01 and h03.")

    # SECOND READ, step sizes
    print(f"\n  SECOND READ, the step sizes, descriptive:")
    steps = {}
    for lo, hi in zip(LADDER, LADDER[1:]):
        m, se, t = paired(per[lo], per[hi], SEEDS6)
        steps[f"{lo}->{hi}"] = dict(mean=m, se=se, t=t)
        print(f"    {lo:>4} to {hi:<5} buys {m:+.4f}  se {se:.4f}"
              f"  t {t:+.2f}")

    # THIRD READ, A47's filtered ladder restated
    print(f"\n  THIRD READ, A47's primary ladder (rows longer than"
          f" {MIN_LEN}) restated against the reachable zero, twelve seeds:")
    fper = {a: {} for a in [ZERO] + A47_ARMS}
    for s in SEEDS12:
        L = row_lengths(s, lengths, elig)
        mats = load(s, [ZERO] + A47_ARMS)
        ok = L > MIN_LEN
        for m in mats.values():
            ok &= np.isfinite(m).all(1)
        for a in [ZERO] + A47_ARMS:
            fper[a][s] = auc_mean(mats[a][ok])
        print(f"    seed {s} rows {int(ok.sum())}", flush=True)
    print(f"    {'arm':<6}{'vs L3_FULL restated':>24}"
          f"{'A47 recorded vs L0_RAW':>26}")
    a47 = {}
    for a in A47_ARMS:
        m, se, t = paired(fper[a], fper[ZERO], SEEDS12)
        a47[a] = dict(mean=m, se=se, t=t, recorded=A47_RECORDED[a])
        print(f"    {a:<6}{m:+.4f} se {se:.4f} t {t:+6.2f}"
              f"          {A47_RECORDED[a]:+.4f}")

    res = dict(k=K, seeds6=SEEDS6, seeds12=SEEDS12, gate=gate,
               gate_passed=bool(gate_ok), ladder=lad, steps=steps,
               primary=dict(first=first, first_se=fse, first_t=ft,
                            total=tot, total_se=tse, total_t=tt,
                            share=share, verdict=v,
                            underpowered=bool(underpowered)),
               a47_restated=a47,
               auc6={a: {str(s): per[a][s] for s in SEEDS6} for a in arms},
               auc12={a: {str(s): fper[a][s] for s in SEEDS12}
                      for a in [ZERO] + A47_ARMS})
    with open("research/w4_e1rezero.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1rezero.json")
    print("  one trajectory per row, no selection, diagnostic only,"
          " never a training signal, no serve decision")

    rid = ledger.append_row(
        "w4_e1rezero",
        {"seeds6": SEEDS6, "seeds12": SEEDS12, "n": N, "k": K,
         "perm_seed": PERM_SEED, "ladder": LADDER, "zero": ZERO,
         "min_len": MIN_LEN,
         "reference": "A52 corrected zero, w4_e1zero"},
        "ok" if gate_ok else "failed",
        metrics={"gate_passed": int(gate_ok), "first_event": first,
                 "first_event_t": ft, "total_drop": tot, "share": share,
                 "underpowered": int(underpowered)},
        artifacts=["research/w4_e1rezero.json"],
        notes=f"AMENDMENT 53 forcing ladder against the A52 zero. First"
              f" forced event buys {first:+.4f} t {ft:+.2f} of a total"
              f" {tot:+.4f}, share {share:.2f}, {v}."
              f"{' UNDERPOWERED on six seeds.' if underpowered else ''}"
              f" Diagnostic only, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
