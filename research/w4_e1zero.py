"""w4_e1zero. AMENDMENT 52, registered in step0_prereg.md before this file
existed.

Every residual from A41 to A48 has the form "arm minus HUM", where HUM is
the RAW corpus featurisation and every arm is the token round trip of the
human events through esp._decode. Those are not the same trajectory even
when the model generates nothing, so the arms cannot reach that zero. A36
built the round trip arm (L3_FULL) and its validity gate stopped the read
before any rung was quoted; A37 later showed that gate was wrong.

This measures the offset paired over twelve seeds, restates the A47 and A48
ladders against the zero the arms can actually reach, and puts the A51 width
correction on top of the RESIDUAL width gap rather than the raw one.

CPU only, no model generation, no GPU. Diagnostic only, never a training
signal, never a serve candidate, no selection of trajectories.
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

SEEDS = list(range(40, 52))
CACHED = list(range(40, 46))     # the six seeds A36 already built
K = 20
PERM_SEED = 3208
N = 2000
MAX_T = 256
KMAX = 4
TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
GATE1_MIN_ROWS = 100
ANCHOR = "data/human_val_features_grpo.npy"
FLOOR = "research/w4_e1floor_F_{a}_s{s}.npy"
ARM = "research/w4_e1feat_F_{a}_s{s}.npy"
FLOOR_ARMS = ["L0_RAW", "L1_DT", "L2_POLAR", "L3_FULL", "L3B_NOSNAP"]
MODEL_ARMS = ["h02", "h04", "h08", "h016", "hf25", "hf50", "hf75"]
ZERO = "L3_FULL"
RAW = "L0_RAW"
SCALE = "research/w4_e1scale.json"
# the run whose registered gate failed on its own row filter, kept on the
# ledger; this rerun corrects the filter and nothing else
CONFIRMS = "w4_e1zero_2026-08-29T200529+0000_92d2813f"


def auc_mean(m):
    v = np.array([scoring.score_features(
        m[np.random.default_rng(PERM_SEED + k).permutation(len(m))])["auc_rf_oob"]
        for k in range(K)])
    return float(v.mean())


def paired(a, b, seeds=SEEDS):
    d = np.array([a[s] - b[s] for s in seeds])
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return m, se, (m / se if se > 0 else float("inf"))


def iqr(m):
    return np.percentile(m, 75, 0) - np.percentile(m, 25, 0)


def row_lengths(seed, lengths, elig):
    pick = np.sort(np.random.default_rng(1000 + seed).choice(elig, N,
                                                             replace=False))
    return np.minimum(lengths[pick], MAX_T).astype(np.int64)


def width_buys(gap, xs, ys):
    """What the A51 ladder says a width gap buys, by linear interpolation.

    Registered as a band with the chord as an upper bound, on the ground
    that the A51 curve is convex. The post hoc correction in the prereg
    withdraws that: the ladder's four interval slopes are -1.054, +0.553,
    +0.346 and +0.585, which is not monotone, and its two bottom rungs are
    indistinguishable from zero, so no shape can be asserted below s100.
    This is a point estimate on an unresolved curve, not a bound, and it
    carries no error bar from A51.
    """
    if gap > xs[-1]:
        return None, "out of range, no extrapolation (registered)"
    if gap <= 0:
        return 0.0, "no width excess over the zero"
    j = int(np.searchsorted(xs, gap))
    a, b = xs[j - 1], xs[j]
    v = ys[j - 1] + (ys[j] - ys[j - 1]) * (gap - a) / (b - a)
    return float(v), f"interpolated between {a:.4f} and {b:.4f}"


def main():
    lengths = np.load("training/events_len.npy")
    NT = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(NT, min(NT, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(NT), trained)
    elig = held[lengths[held] > KMAX]
    arrs = [np.load(f"training/events_{k}.npy", mmap_mode="r")
            for k in ("s2", "dth", "dt", "dx", "dy", "cond")]

    print(f"  every value is the mean of K={K} permutations,"
          f" {len(SEEDS)} seeds\n", flush=True)

    # GATE 2 first, since it also tells us whether the build we are about to
    # run for six new seeds reproduces the six that are already committed.
    print("  GATE 2 (rebuild seed 40 and compare to the committed caches):",
          flush=True)
    _, rebuilt = fl.build(CACHED[0], lengths, *arrs, elig)
    g2 = {}
    for a in FLOOR_ARMS:
        ref = np.load(FLOOR.format(a=a, s=CACHED[0]))
        same = np.array_equal(np.nan_to_num(rebuilt[a], nan=-9e18),
                              np.nan_to_num(ref, nan=-9e18))
        g2[a] = bool(same)
        extra = ""
        if not same:
            d = np.abs(np.nan_to_num(rebuilt[a]) - np.nan_to_num(ref))
            extra = (f"   rows differing {int((d > 0).any(1).sum())}/{N},"
                     f" max abs {d.max():.3e}")
        print(f"    {a:<11} {'bit exact' if same else 'DIFFERS'}{extra}",
              flush=True)
    if not g2[RAW]:
        print("    GATE 2 FAILED on L0_RAW: the build does not reproduce the"
              " committed raw matrix, so everything below is VOID.")
        return
    if not g2[ZERO]:
        print("    the decoder carries randomness. Z is one draw of it; the"
              " paired read over twelve seeds still measures the mean offset"
              " and is reported as such (registered).")

    # build the six seeds A36 never ran
    for s in SEEDS:
        paths = {a: FLOOR.format(a=a, s=s) for a in FLOOR_ARMS}
        if all(os.path.exists(v) for v in paths.values()):
            continue
        _, built = fl.build(s, lengths, *arrs, elig)
        for a, v in paths.items():
            np.save(v, built[a])
        print(f"  built seed {s}: "
              + "  ".join(f"{a} {int(np.isfinite(built[a]).all(1).sum())}"
                          for a in FLOOR_ARMS), flush=True)

    # GATE 1, bit exactness of the zero against an arm that generates nothing
    print("\n  GATE 1 (h016 on rows shorter than 16 IS the full human round"
          " trip, so it must equal L3_FULL exactly):", flush=True)
    tot, bad, per_seed_rows = 0, 0, []
    for s in CACHED:
        L = row_lengths(s, lengths, elig)
        z = np.load(FLOOR.format(a=ZERO, s=s))
        h = np.load(ARM.format(a="h016", s=s))
        # strictly under 16: such a row has at least one trailing forced
        # position on the pad class, which terminates the sequence and makes
        # it a pure human round trip. A row of exactly 16 has no such
        # position and the model may append an event, so it is not entitled
        # to match. See the post hoc gate correction in the prereg.
        sel = (L < 16) & np.isfinite(z).all(1) & np.isfinite(h).all(1)
        n = int(sel.sum())
        nb = int((~np.isclose(z[sel], h[sel], rtol=0, atol=0)).any(1).sum())
        tot += n
        bad += nb
        per_seed_rows.append(n)
        print(f"    seed {s}: {n} rows shorter than 16,"
              f" {nb} differ from L3_FULL", flush=True)
    underpowered = min(per_seed_rows) < GATE1_MIN_ROWS
    g1 = (bad == 0)
    if underpowered:
        print(f"    GATE 1 UNDERPOWERED: the smallest seed offers"
              f" {min(per_seed_rows)} rows, under the registered"
              f" {GATE1_MIN_ROWS}. Reported as underpowered, not as passed.")
    print(f"    {tot} rows checked, {bad} differ:"
          f" {'ok' if g1 else 'FAILED'}", flush=True)
    if not g1:
        print("    GATE 1 FAILED: L3_FULL is not the arms' zero. The primary"
              " read is VOID (registered) and the correct action is to find"
              " the path difference, not to interpret Z.")

    # score everything on the finite intersection, paired per seed
    ARMS = FLOOR_ARMS + MODEL_ARMS
    per = {a: {} for a in ARMS}
    mats_by_seed = {}
    for s in SEEDS:
        mats = {a: np.load(FLOOR.format(a=a, s=s)) for a in FLOOR_ARMS}
        mats.update({a: np.load(ARM.format(a=a, s=s)) for a in MODEL_ARMS})
        ok = np.ones(N, dtype=bool)
        for m in mats.values():
            ok &= np.isfinite(m).all(1)
        mats = {a: m[ok] for a, m in mats.items()}
        mats_by_seed[s] = mats
        for a in ARMS:
            per[a][s] = auc_mean(mats[a])
        print(f"  seed {s} rows {int(ok.sum())}   "
              + "  ".join(f"{a} {per[a][s]:.4f}" for a in ARMS), flush=True)

    # PRIMARY
    zm, zse, zt = paired(per[ZERO], per[RAW])
    if abs(zm) >= 0.005 and abs(zt) >= 3.0:
        zv = "ZERO DISPLACED"
    elif abs(zm) < 0.002 and abs(zt) < 2.0:
        zv = "ZERO CLEAN"
    else:
        zv = "BETWEEN"
    if not g1:
        zv = "VOID, gate 1 failed"
    print(f"\n  PRIMARY READ: the token round trip minus the raw corpus rows,"
          f" paired over twelve seeds")
    print(f"    Z {zm:+.4f}  se {zse:.4f}  t {zt:+.2f}   {zv}")

    # READ 2, the restated ladders
    print(f"\n  READ 2, the ladders restated against the zero the arms can"
          f" actually reach:")
    print(f"    {'arm':<6}{'vs L0_RAW (as recorded)':>26}"
          f"{'vs L3_FULL (restated)':>26}")
    rest = {}
    for a in MODEL_ARMS:
        om, ose, ot = paired(per[a], per[RAW])
        nm, nse, nt = paired(per[a], per[ZERO])
        rest[a] = dict(old=dict(mean=om, se=ose, t=ot),
                       new=dict(mean=nm, se=nse, t=nt))
        print(f"    {a:<6}{om:+.4f} se {ose:.4f} t {ot:+6.2f}"
              f"   {nm:+.4f} se {nse:.4f} t {nt:+6.2f}")
    neg = [a for a in MODEL_ARMS
           if rest[a]["new"]["mean"] <= -0.005 and rest[a]["new"]["t"] <= -3.0]
    print(f"    {'NEGATIVE RUNG SURVIVES: ' + ', '.join(neg) if neg else 'no rung is significantly negative'}")

    # READ 3, the A51 width correction on the RESIDUAL width gap
    sc = json.load(open(SCALE))
    moved = [FEATURE_NAMES.index(f) for f in sc["moved_features"]]
    xs = [0.0] + [sc["ladder"][k]["iqr_gain"] for k in ("s10", "s33",
                                                        "s100", "s200")]
    ys = [0.0] + [sc["ladder"][k]["mean"] for k in ("s10", "s33",
                                                    "s100", "s200")]
    A = np.load(ANCHOR)
    aiqr = iqr(A[np.isfinite(A).all(1)])

    def gap_vs(a, b):
        v = [float(np.mean((iqr(mats_by_seed[s][a])
                            - iqr(mats_by_seed[s][b]))[moved] / aiqr[moved]))
             for s in SEEDS]
        return float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))

    rt_gap, rt_se = gap_vs(ZERO, RAW)
    h02_gap, _ = gap_vs("h02", RAW)
    share = rt_gap / h02_gap if h02_gap != 0 else float("nan")
    print(f"\n  READ 3, the width correction, in A51 units on the eight"
          f" columns its scaling moved:")
    print(f"    the round trip alone widens the raw rows by {rt_gap:+.4f}"
          f" se {rt_se:.4f}, against the h02 gap of {h02_gap:+.4f}"
          f"   ({share:.0%} of it)")
    print(f"    {'arm':<6}{'gap vs L3_FULL':>16}{'width buys':>14}"
          f"{'model error left':>20}   (point estimates, no A51 error bar)")
    corr = {}
    for a in MODEL_ARMS:
        g, gse = gap_vs(a, ZERO)
        buys, note = width_buys(g, xs, ys)
        r = rest[a]["new"]["mean"]
        if buys is None:
            corr[a] = dict(gap=g, note=note)
            print(f"    {a:<6}{g:+16.4f}   {note}")
            continue
        corr[a] = dict(gap=g, gap_se=gse, buys=buys, err=r - buys, note=note)
        print(f"    {a:<6}{g:+16.4f}{buys:+14.4f}{r - buys:+20.4f}")

    # READ 4, descriptive
    print(f"\n  READ 4, the A36 rungs A37 unblocked, against L0_RAW:")
    rungs = {}
    for a in ["L1_DT", "L2_POLAR", "L3_FULL", "L3B_NOSNAP"]:
        m, se, t = paired(per[a], per[RAW])
        rungs[a] = dict(mean=m, se=se, t=t)
        print(f"    {a:<11}{m:+.4f}  se {se:.4f}  t {t:+.2f}")

    res = dict(k=K, seeds=SEEDS, gate1=bool(g1), gate1_rows=tot,
               gate1_bad=bad, gate1_underpowered=bool(underpowered),
               gate2={a: g2[a] for a in FLOOR_ARMS},
               primary=dict(mean=zm, se=zse, t=zt, verdict=zv),
               ladders=rest, negative_rungs=neg,
               roundtrip_width_gap=rt_gap, h02_width_gap=h02_gap,
               width=corr, rungs=rungs,
               auc={a: {str(s): per[a][s] for s in SEEDS} for a in ARMS})
    with open("research/w4_e1zero.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1zero.json")
    print("  real human events only in every floor arm, no selection,"
          " diagnostic only, never a training signal")

    rid = ledger.append_row(
        "w4_e1zero",
        {"seeds": SEEDS, "n": N, "k": K, "perm_seed": PERM_SEED,
         "zero": ZERO, "raw": RAW, "arms": ARMS,
         "reference": "w4_e1floor A36 build, extended to twelve seeds"},
        "ok" if g1 else "failed",
        metrics={"gate1": int(g1), "gate1_rows": tot, "gate1_bad": bad,
                 "z_mean": zm, "z_t": zt,
                 "roundtrip_width_gap": rt_gap,
                 "n_negative_rungs": len(neg)},
        artifacts=["research/w4_e1zero.json"],
        confirms_run_id=CONFIRMS,
        notes=f"AMENDMENT 52 the zero. Token round trip minus raw corpus"
              f" rows {zm:+.4f} t {zt:+.2f}, {zv}. Ladders restated against"
              f" L3_FULL; negative rungs surviving: {', '.join(neg) or 'none'}."
              f" Diagnostic only, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
