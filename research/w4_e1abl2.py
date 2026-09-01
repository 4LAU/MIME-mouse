"""w4_e1abl2. AMENDMENT 64, registered in step0_prereg.md before this
file existed and written while the generation for seeds 76 to 87 was on
its first seed, so before any number it reads could exist.

A63 branch C found that dropping num_direction_changes closes about 41
per cent of the k0 minus zero gap inside the 17 to 32 band, leading in
10 of 12 seeds against a registered bar of 10. It ran on seeds 64 to 75,
which AMENDMENT 62 had already read, so it is a second instrument on the
same rows and not a replication. This is the replication.

THE PRIMARY IS THE SIZE, NOT THE COUNT, and the registration says why
without reference to which one looked better: if no feature were
special, each of eighteen would lead about one seed in eighteen, so 10
of 12 and 12 of 12 are both far from the null and barely separate. The
count keeps branch C's bar of 10 deliberately, because moving a bar in
either direction after a result landed exactly on it is the thing being
avoided.

EVERY AUC HERE IS A NON CONTRACT DIAGNOSTIC AUC. The contract scorer
loads the eighteen column anchor itself so it cannot take a seventeen
column matrix, and A49 forbids editing a contract matrix, so the forest
is built here to the identical recipe and gated on reproducing the
contract AUC EXACTLY. Nothing this file prints may be compared to the
0.5795 headline or to any contract number.

CPU only, no torch. Diagnostic only, never a training signal, never a
serve candidate, no selection, one trajectory per row.
"""
import json
import os
import re
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import scoring   # noqa: E402
import ledger    # noqa: E402
from features import FEATURE_NAMES   # noqa: E402

SEEDS = list(range(76, 88))
K = 20
PERM_SEED = 3208
N = 2000
MAX_T = 256
KMAX = 4
TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
ANCHOR = "data/human_val_features_grpo.npy"
FLOOR = "research/w4_e1floor_F_{a}_s{s}.npy"
ARM = "research/w4_e1feat_F_{a}_s{s}.npy"
ZERO = "L3_FULL"
K0 = "k0"
BAND = ("17 to 32", 17, 32)
FULL = ("full", 0, MAX_T)
TARGET = "num_direction_changes"
CACHE = "research/w4_e1abl2_cache.json"
CAL_TOL = 1e-9

# registered bars, fixed before the generation existed
REF_SHRINK = 0.0095          # A63 branch C, band
REF_HALFWIDTH = 0.0039       # three times branch C's se of 0.0013
T_BAR = 3.0
SEED_BAR = 10                # of twelve, branch C's bar kept deliberately
REF_ZERO = {"full": 0.521129, "17 to 32": 0.754585}   # A62 fresh seeds
GATE_W = 0.015
GEN_LOG = "/home/aaronadmin/w4_arms/qladder_a64.log"
BANNER = ("[event_stream_polar] ckpt=event_polar_best.pt epoch=22"
          " steps=100 temp=1.0 th_temp=None order=gumbel round=True bestof=1")
CKPT_MD5 = "91326a29750789f3167055324ef377c5"


def gate_b():
    """banner identical to A57's, checkpoint unmoved, no cached arm."""
    txt = open(GEN_LOG).read()
    banners = [ln.strip() for ln in txt.splitlines()
               if ln.startswith("[event_stream_polar]")]
    md5s = re.findall(r"^([0-9a-f]{32})  training/", txt, re.M)
    ok_b = bool(banners) and all(b == BANNER for b in banners)
    ok_m = bool(md5s) and all(m == CKPT_MD5 for m in md5s)
    print(f"  INTEGRITY GATE B (registered): the fresh generation must"
          f" match A56, A57 and A62 exactly:")
    print(f"    banner lines {len(banners)}  "
          f"{'all identical to A57' if ok_b else 'MISMATCH'}")
    print(f"    md5 lines {len(md5s)}  "
          f"{'checkpoint unmoved' if ok_m else 'CHECKPOINT MOVED'}")
    print(f"  GATE B {'PASSED' if ok_b and ok_m else 'FAILED, every read below is VOID (registered)'}",
          flush=True)
    return ok_b and ok_m


def contract_auc(synth, anchor, cols=None):
    """the contract recipe, reproduced. cols=None means all eighteen."""
    n_use = min(len(anchor), len(synth))
    h = anchor[:n_use]
    s = np.asarray(synth, dtype=np.float64)[:n_use]
    if cols is not None:
        h, s = h[:, cols], s[:, cols]
    X = np.vstack([h, s])
    y = np.concatenate([np.zeros(n_use), np.ones(n_use)])
    clf = RandomForestClassifier(n_estimators=100, oob_score=True,
                                 n_jobs=-1, random_state=42)
    clf.fit(X, y)
    return float(roc_auc_score(y, clf.oob_decision_function_[:, 1]))


def perm(m, k):
    return m[np.random.default_rng(PERM_SEED + k).permutation(len(m))]


def paired(v):
    d = np.asarray(v, float)
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return m, se, (m / se if se > 0 else float("inf"))


def main():
    print(f"  AMENDMENT 64. Twelve seeds {SEEDS[0]} to {SEEDS[-1]}, NEVER"
          f" GENERATED AND NEVER SCORED. K={K}, PERM_SEED {PERM_SEED}.")
    print(f"  EVERY AUC BELOW IS A NON CONTRACT DIAGNOSTIC AUC. It may not"
          f" be compared to the headline or to any contract number.\n",
          flush=True)

    if not gate_b():
        return
    print()

    anchor = np.load(ANCHOR)
    lengths = np.load("training/events_len.npy")
    NT = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(NT, min(NT, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(NT), trained)
    elig = held[lengths[held] > KMAX]

    mats, per_ok, per_len = {}, {}, {}
    for s in SEEDS:
        m = {ZERO: np.load(FLOOR.format(a=ZERO, s=s)),
             K0: np.load(ARM.format(a=K0, s=s))}
        ok = np.ones(len(m[ZERO]), dtype=bool)
        for v in m.values():
            ok &= np.isfinite(v).all(1)
        pick = np.sort(np.random.default_rng(1000 + s).choice(elig, N,
                                                              replace=False))
        mats[s], per_ok[s] = m, ok
        per_len[s] = np.minimum(lengths[pick], MAX_T).astype(np.int64)

    def rows(s, lo, hi):
        return np.flatnonzero(per_ok[s] & (per_len[s] >= lo)
                              & (per_len[s] <= hi))

    try:
        cache = json.load(open(CACHE))
    except FileNotFoundError:
        cache = {}

    def cell(key, fn):
        if key in cache:
            return np.asarray(cache[key])
        v = fn()
        cache[key] = [float(x) for x in v]
        with open(CACHE, "w") as fh:
            json.dump(cache, fh)
        print(f"      cell done: {key}", flush=True)
        return np.asarray(cache[key])

    def levels(band, arm):
        return cell(f"lvl|{band[0]}|{arm}", lambda: [
            float(np.mean([contract_auc(perm(mats[s][arm][rows(s, band[1],
                                                               band[2])], k),
                                        anchor)
                           for k in range(K)])) for s in SEEDS])

    # SANITY GATE, registered
    print("  SANITY GATE. The zero's twelve seed mean must land within"
          f" {GATE_W} of A62's fresh seed value on both row sets:")
    gate_ok = True
    for band in (BAND, FULL):
        got = float(levels(band, ZERO).mean())
        ref = REF_ZERO[band[0]]
        ok = abs(got - ref) <= GATE_W
        gate_ok = gate_ok and ok
        print(f"    {band[0]:<10} A62 {ref:.6f}  fresh {got:.6f}"
              f"  diff {got - ref:+.6f}   {'ok' if ok else 'MISS'}")
    print(f"  SANITY GATE {'PASSED' if gate_ok else 'FAILED, every read below is VOID (registered)'}",
          flush=True)
    if not gate_ok:
        return

    # ORDERING, A62's READ 0 repeated on a third seed set
    print("\n  ORDERING, the precondition on a THIRD independent seed set."
          " The band term must be NEGATIVE and the full row set term"
          " POSITIVE:")
    order_ok = True
    base = {}
    for band in (BAND, FULL):
        d = levels(band, K0) - levels(band, ZERO)
        m, se, t = paired(d.tolist())
        base[band[0]] = d
        want_neg = band is BAND
        ok = (m < 0) if want_neg else (m > 0)
        order_ok = order_ok and ok
        print(f"    {band[0]:<10} k0 minus zero {m:+.4f} se {se:.4f}"
              f" t {t:+6.2f}   {int((d < 0).sum())} of 12 negative"
              f"   {'ok' if ok else 'FAILS, this failure IS the result (registered)'}")
    if not order_ok:
        print("    The ordering did not replicate. The ablation question"
              " does not arise and that is the finding.")
        return

    # CALIBRATION GATE, registered as exact
    print("\n  CALIBRATION GATE, my forest against scoring.score_features,"
          " all eighteen columns, tolerance 1e-9:")
    worst = 0.0
    for band in (BAND, FULL):
        for s in SEEDS:
            ix = rows(s, band[1], band[2])
            for a in (ZERO, K0):
                m0 = perm(mats[s][a][ix], 0)
                worst = max(worst, abs(contract_auc(m0, anchor)
                                       - scoring.score_features(m0)["auc_rf_oob"]))
    ok_cal = worst <= CAL_TOL
    print(f"    worst absolute difference {worst:.3e}"
          f"   {'ok, GATE PASSED' if ok_cal else 'MISS, AMENDMENT 64 IS VOID (registered)'}",
          flush=True)
    if not ok_cal:
        print("    My forest is not the contract scorer's forest, so the"
              " ablation ranking means nothing. Reporting the mismatch"
              " rather than loosening the tolerance.")
        return

    def gaps(band, cols, tag):
        def fn():
            out = []
            for s in SEEDS:
                ix = rows(s, band[1], band[2])
                d = [contract_auc(perm(mats[s][K0][ix], k), anchor, cols)
                     - contract_auc(perm(mats[s][ZERO][ix], k), anchor, cols)
                     for k in range(K)]
                out.append(float(np.mean(d)))
            return out
        return cell(f"{band[0]}|{tag}", fn)

    results = {}
    for band in (BAND, FULL):
        b0 = gaps(band, None, "ALL")
        bm, bse, bt = paired(b0.tolist())
        print(f"\n  {band[0]} row set. Baseline gap on all eighteen"
              f" columns: {bm:+.4f} se {bse:.4f} t {bt:+.2f}", flush=True)
        shrink = {}
        for j, name in enumerate(FEATURE_NAMES):
            cols = [c for c in range(len(FEATURE_NAMES)) if c != j]
            shrink[name] = np.abs(b0) - np.abs(gaps(band, cols, name))
        stacked = np.vstack([shrink[n] for n in FEATURE_NAMES])
        winner = [FEATURE_NAMES[int(np.argmax(stacked[:, i]))]
                  for i in range(len(SEEDS))]
        order = sorted(FEATURE_NAMES, key=lambda n: -float(shrink[n].mean()))
        print(f"    {'dropped feature':<26}{'shrink':>9}{'se':>8}{'t':>8}"
              f"  seeds first")
        for name in order:
            m, se, t = paired(shrink[name].tolist())
            cnt = sum(1 for w in winner if w == name)
            print(f"    {name:<26}{m:+9.4f}{se:8.4f}{t:+8.2f}  {cnt:>2}/12",
                  flush=True)
        results[band[0]] = dict(
            baseline=dict(mean=bm, se=bse, t=bt),
            shrink={n: dict(zip(("mean", "se", "t"),
                                paired(shrink[n].tolist())))
                    for n in FEATURE_NAMES},
            winner_per_seed=winner,
            n_first_target=sum(1 for w in winner if w == TARGET),
            order=order)

    b = results[BAND[0]]
    tm = b["shrink"][TARGET]["mean"]
    tt = b["shrink"][TARGET]["t"]
    lo, hi = REF_SHRINK - REF_HALFWIDTH, REF_SHRINK + REF_HALFWIDTH
    read1 = (tm > 0 and abs(tt) >= T_BAR and lo <= tm <= hi)
    read2 = b["n_first_target"] >= SEED_BAR

    print(f"\n  READ 1 (PRIMARY), THE SIZE. {TARGET} shrink in the band:")
    print(f"    {tm:+.4f} t {tt:+.2f}, registered interval"
          f" {lo:+.4f} to {hi:+.4f} around A63's {REF_SHRINK:+.4f}")
    print(f"    {'REPLICATED' if read1 else 'FAILED TO REPLICATE'}")
    if tm > hi:
        print("    NOTE: the shrink is LARGER than the interval. The"
              " registration calls that a failure to replicate too, and"
              " says so because a bigger number is the tempting one.")
    print(f"\n  READ 2, SECONDARY, THE COUNT. {TARGET} leads in"
          f" {b['n_first_target']} of 12, bar {SEED_BAR}:"
          f" {'PASS' if read2 else 'FAIL'}")

    if read1 and read2:
        verdict = "REPLICATED, the size and the count both hold on untouched seeds"
    elif not read1:
        verdict = ("NOT REPLICATED, the size failed. A passing count cannot"
                   " rescue a failing size (registered)")
    else:
        verdict = ("PARTIAL, the size holds and the count does not. PARTIAL"
                   " IS NOT REPLICATION (registered): the effect is the"
                   " right size and less consistent per seed than A63 found")
    print(f"\n  THE REGISTERED VERDICT: {verdict}", flush=True)

    frac = tm / abs(b["baseline"]["mean"])
    print(f"\n  READ 3, DESCRIPTIVE, NO VERDICT and no bar because it is a"
          f" ratio (A47). Fraction of the band gap closed:"
          f" {frac * 100:.0f} per cent, against A63's 41 per cent.")
    print(f"  READ 4, DESCRIPTIVE, NO VERDICT. The two tables above are"
          f" the comparison to A63 branch C: its band runner up was"
          f" mean_jerk at +0.0026 and its full row set leader was"
          f" mean_acceleration at +0.0116 with {TARGET} second at"
          f" +0.0058.")

    res = dict(k=K, seeds=SEEDS, target=TARGET, gate_b=True,
               sanity_gate=gate_ok, ordering_ok=order_ok,
               calibration_worst=worst, results=results,
               read1=bool(read1), read2=bool(read2), verdict=verdict,
               fraction_closed=frac, non_contract_auc=True)
    with open("research/w4_e1abl2.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1abl2.json")
    print("  one trajectory per row, no selection, diagnostic only,"
          " never a training signal, no serve decision")

    rid = ledger.append_row(
        "w4_e1abl2",
        {"seeds": SEEDS, "k": K, "perm_seed": PERM_SEED, "zero": ZERO,
         "arm": K0, "band": BAND[0], "target": TARGET,
         "reference": "A64, A63 branch C replicated on untouched seeds"},
        "ok",
        metrics={"calibration_worst": worst,
                 "band_shrink": tm, "band_shrink_t": tt,
                 "n_first_band": b["n_first_target"],
                 "read1": int(read1), "read2": int(read2),
                 "replicated": int(read1 and read2)},
        artifacts=["research/w4_e1abl2.json"],
        notes=f"AMENDMENT 64. A63 branch C's ablation on twelve untouched"
              f" seeds 76 to 87. {verdict}. Band shrink {tm:+.4f} t"
              f" {tt:+.2f} against A63's {REF_SHRINK:+.4f},"
              f" {b['n_first_target']} of 12 seeds first. NON CONTRACT"
              f" diagnostic AUC, never comparable to the headline."
              f" Diagnostic only, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
