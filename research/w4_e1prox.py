"""w4_e1prox. AMENDMENT 65, registered in step0_prereg.md before this
file existed and written while the q0 generation for seeds 76 to 87 was
still on its first seed, so before any number it reads could exist.

Every amendment from 56 to 64 measures k0, the FREE RUNNING arm with no
forcing at any position. The served configuration (AMENDMENT 19) is q0:
event 0 drawn from training/w4_firsthead_q.pt at temperatures 1, 1, 1,
the autoregressive model continuing from position 1, and the spec log
duration from the DUR_EMPIRICAL pin. Read off research/w4_qladder.py
line 138, lines 266 to 270 and line 29 before the registration was
written. Nobody has checked the diagnostic picture transfers from k0 to
the arm we actually serve. If it does not, nine amendments describe an
arm nobody serves.

The floors and the k0 arms for these twelve seeds already exist, so this
run adds only q0 and reuses everything else. That makes a stronger gate
available than A64 had: the contract scorer is deterministic across
processes at fixed row order, so the k0 baselines must come back BIT
IDENTICAL to A64's, not merely close.

A PASS DOES NOT MAKE A56 TO A64 MEASUREMENTS OF q0. It means the sign
pattern and the named channel are shared between the two arms. Those
numbers stay k0 numbers.

EVERY AUC HERE IS A NON CONTRACT DIAGNOSTIC AUC where a column is
dropped: the contract scorer loads the eighteen column anchor itself so
it cannot take a seventeen column matrix, and A49 forbids editing a
contract matrix, so the forest is built here to the identical recipe and
gated on reproducing the contract AUC EXACTLY. Nothing this file prints
may be compared to the 0.5795 headline or to any contract number.

CPU only, no torch. Diagnostic only, never a training signal, never a
serve candidate, no selection, one trajectory per row.
"""
import json
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
Q0 = "q0"
BAND = ("17 to 32", 17, 32)
FULL = ("full", 0, MAX_T)
TARGET = "num_direction_changes"
CACHE = "research/w4_e1prox_cache.json"

# registered bars and references, fixed before the generation existed
T_BAR = 3.0
CAL_TOL = 1e-9
REPRO_TOL = 1e-9
REF_K0 = {"17 to 32": -0.020558143564370003,   # AMENDMENT 64, same seeds
          "full": 0.05912182409436803}
REF_ABL_K0 = 0.0094          # A64 band shrink, for READ 4 side by side
GEN_LOG = "/home/aaronadmin/w4_arms/qladder_a65.log"
BANNER = ("[event_stream_polar] ckpt=event_polar_best.pt epoch=22"
          " steps=100 temp=1.0 th_temp=None order=gumbel round=True bestof=1")
CKPT_MD5 = "91326a29750789f3167055324ef377c5"


def gate_b():
    """banner identical to A57's and the checkpoint unmoved."""
    txt = open(GEN_LOG).read()
    banners = [ln.strip() for ln in txt.splitlines()
               if ln.startswith("[event_stream_polar]")]
    md5s = re.findall(r"^([0-9a-f]{32})  training/", txt, re.M)
    ok_b = bool(banners) and all(b == BANNER for b in banners)
    ok_m = bool(md5s) and all(m == CKPT_MD5 for m in md5s)
    print("  INTEGRITY GATE B (registered): the q0 generation must match"
          " A56, A57, A62 and A64 exactly:")
    print(f"    banner lines {len(banners)}  "
          f"{'all identical to A57' if ok_b else 'MISMATCH'}")
    print(f"    md5 lines {len(md5s)}  "
          f"{'checkpoint unmoved' if ok_m else 'CHECKPOINT MOVED'}")
    if not ok_b:
        for b in sorted(set(banners))[:3]:
            print(f"      got: {b}")
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
    print(f"  AMENDMENT 65. Seeds {SEEDS[0]} to {SEEDS[-1]}. The question is"
          f" whether the arm A56 to A64 measured, k0, stands in for the arm"
          f" we serve, q0.")
    print("  EVERY AUC BELOW IS A NON CONTRACT DIAGNOSTIC AUC. It may not"
          " be compared to the headline or to any contract number.\n",
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

    # Two row masks, disclosed in the prereg under AMENDMENT 65 EXECUTION
    # FAULT. q0 has one non finite row in five seeds. The joint mask over
    # all three arms is what every q0 read uses, because READ 2 is a
    # paired difference and pairs must share rows. The A64 mask (zero and
    # k0 only) is what the reproduction gate uses, because bit identity
    # with A64 is only a meaningful demand on A64's own rows.
    mats, per_ok, per_ok64, per_len = {}, {}, {}, {}
    for s in SEEDS:
        m = {ZERO: np.load(FLOOR.format(a=ZERO, s=s)),
             K0: np.load(ARM.format(a=K0, s=s)),
             Q0: np.load(ARM.format(a=Q0, s=s))}
        ok64 = np.isfinite(m[ZERO]).all(1) & np.isfinite(m[K0]).all(1)
        ok = ok64 & np.isfinite(m[Q0]).all(1)
        pick = np.sort(np.random.default_rng(1000 + s).choice(elig, N,
                                                              replace=False))
        mats[s], per_ok[s], per_ok64[s] = m, ok, ok64
        per_len[s] = np.minimum(lengths[pick], MAX_T).astype(np.int64)

    def rows(s, lo, hi, a64=False):
        mask = per_ok64[s] if a64 else per_ok[s]
        return np.flatnonzero(mask & (per_len[s] >= lo)
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

    def gap(band, arm, cols=None, tag="ALL", a64=False):
        """per seed mean over K permutations of (arm minus zero)."""
        def fn():
            out = []
            for s in SEEDS:
                ix = rows(s, band[1], band[2], a64)
                d = [contract_auc(perm(mats[s][arm][ix], k), anchor, cols)
                     - contract_auc(perm(mats[s][ZERO][ix], k), anchor, cols)
                     for k in range(K)]
                out.append(float(np.mean(d)))
            return out
        return cell(f"gap|{band[0]}|{arm}|{tag}|{'a64mask' if a64 else 'joint'}", fn)

    # REPRODUCTION GATE, registered as bit identical
    print("  REPRODUCTION GATE (registered). The contract scorer is"
          " deterministic across processes at fixed row order and these are"
          " AMENDMENT 64's own seeds, floors, k0 features, ROW MASK and"
          " permutations, so the k0 baselines must come back BIT IDENTICAL,"
          f" tolerance {REPRO_TOL:.0e}:")
    repro_ok = True
    for band in (BAND, FULL):
        d = gap(band, K0, a64=True)
        got = float(d.mean())
        ref = REF_K0[band[0]]
        ok = abs(got - ref) <= REPRO_TOL
        repro_ok = repro_ok and ok
        print(f"    {band[0]:<10} A64 {ref:+.15f}")
        print(f"    {'':<10} now {got:+.15f}  diff {abs(got - ref):.3e}"
              f"   {'ok' if ok else 'MISMATCH'}")
    print(f"  REPRODUCTION GATE {'PASSED' if repro_ok else 'FAILED'}",
          flush=True)
    if not repro_ok:
        print("    The pipeline moved between A64 and here, so no q0 number"
              " in this run would be interpretable. Reporting the mismatch"
              " rather than loosening the tolerance (registered).")
        return
    # the k0 gap every q0 read is paired against: same quantity, on the
    # joint mask, five rows fewer than A64 (disclosed)
    k0gap = {band[0]: gap(band, K0) for band in (BAND, FULL)}

    # CALIBRATION GATE, registered as exact
    print("\n  CALIBRATION GATE, my forest against scoring.score_features,"
          " all eighteen columns, tolerance 1e-9:")
    worst = 0.0
    for band in (BAND, FULL):
        for s in SEEDS:
            ix = rows(s, band[1], band[2])
            for a in (ZERO, K0, Q0):
                m0 = perm(mats[s][a][ix], 0)
                worst = max(worst, abs(contract_auc(m0, anchor)
                                       - scoring.score_features(m0)["auc_rf_oob"]))
    ok_cal = worst <= CAL_TOL
    print(f"    worst absolute difference {worst:.3e}"
          f"   {'ok, GATE PASSED' if ok_cal else 'MISS, AMENDMENT 65 IS VOID (registered)'}",
          flush=True)
    if not ok_cal:
        print("    My forest is not the contract scorer's forest, so the"
              " ablation number means nothing. Reporting the mismatch"
              " rather than loosening the tolerance.")
        return

    # READ 1, PRIMARY, the sign pattern on the served arm
    print("\n  READ 1 (PRIMARY), THE SIGN PATTERN ON THE SERVED ARM."
          " q0 minus L3_FULL. The band term must be NEGATIVE at t -3.0 and"
          " the full row set term POSITIVE at t +3.0:")
    q0gap, read1 = {}, True
    for band in (BAND, FULL):
        d = gap(band, Q0)
        q0gap[band[0]] = d
        m, se, t = paired(d.tolist())
        want_neg = band is BAND
        ok = (t <= -T_BAR) if want_neg else (t >= T_BAR)
        read1 = read1 and ok
        km = float(k0gap[band[0]].mean())
        print(f"    {band[0]:<10} q0 {m:+.4f} se {se:.4f} t {t:+6.2f}"
              f"   {int((d < 0).sum())} of 12 negative"
              f"   k0 was {km:+.4f}   {'ok' if ok else 'MISS'}")
    print(f"  READ 1 {'PASSES' if read1 else 'FAILS. That failure IS the result (registered)'}",
          flush=True)

    # READ 2, SECONDARY, the magnitude, a paired DIFFERENCE not a ratio
    print("\n  READ 2, SECONDARY, THE MAGNITUDE. Paired per seed"
          " (q0 minus zero) minus (k0 minus zero), same rows, same"
          " permutations. A DIFFERENCE, not a ratio (A47). This labels the"
          " verdict, it does not change it:")
    read2 = {}
    for band in (BAND, FULL):
        d = q0gap[band[0]] - k0gap[band[0]]
        m, se, t = paired(d.tolist())
        lab = "NULL, k0 stands in quantitatively" if abs(t) < T_BAR else \
              "REAL, the proxy is directional only"
        read2[band[0]] = dict(mean=m, se=se, t=t, null=bool(abs(t) < T_BAR))
        print(f"    {band[0]:<10} q0 minus k0 {m:+.4f} se {se:.4f}"
              f" t {t:+6.2f}   {lab}")

    # READ 3, PRIMARY 2, the named channel on the served arm
    print("\n  READ 3 (PRIMARY 2), THE CHANNEL. The A61 and A62 instrument"
          f" on q0: {TARGET} importance share, q0 minus the zero. The band"
          " term must be NEGATIVE at t -3.0:")
    j = FEATURE_NAMES.index(TARGET)

    def share(band, arm):
        def fn():
            out = []
            for s in SEEDS:
                ix = rows(s, band[1], band[2])
                v = [scoring.score_features(perm(mats[s][arm][ix], k))
                     ["importances"][TARGET] for k in range(K)]
                out.append(float(np.mean(v)))
            return out
        return cell(f"imp|{band[0]}|{arm}", fn)

    read3, imp = True, {}
    for band in (BAND, FULL):
        d = share(band, Q0) - share(band, ZERO)
        m, se, t = paired(d.tolist())
        imp[band[0]] = dict(mean=m, se=se, t=t)
        if band is BAND:
            read3 = t <= -T_BAR
            note = "ok" if read3 else "MISS"
        else:
            note = "descriptive, A62 read +0.0037 on k0"
        print(f"    {band[0]:<10} {m:+.4f} se {se:.4f} t {t:+6.2f}   {note}")
    print(f"  READ 3 {'PASSES' if read3 else 'FAILS'}", flush=True)

    # THE REGISTERED VERDICT
    if read1 and read3:
        verdict = ("PROXY HOLDS, the sign pattern and the named channel are"
                   " shared between the free running arm and the served arm")
    elif read1:
        verdict = ("CHANNEL DOES NOT TRANSFER, the sign pattern holds on the"
                   " served arm and num_direction_changes does not")
    else:
        verdict = ("PROXY FAILS, the sign pattern A56 to A64 is built on does"
                   " not reproduce on the served arm")
    qual = ("quantitatively as well as directionally"
            if read2[BAND[0]]["null"] else "directionally only, so every"
            " magnitude quoted from A56 to A64 is a k0 magnitude and must"
            " be said that way")
    print(f"\n  THE REGISTERED VERDICT: {verdict}")
    print(f"  READ 2 labels it: {qual}")
    print("  A PASS DOES NOT MAKE A56 TO A64 MEASUREMENTS OF q0"
          " (registered). Those numbers stay k0 numbers.", flush=True)

    # READ 4, DESCRIPTIVE, the single registered ablation contrast
    print("\n  READ 4, DESCRIPTIVE, NO VERDICT AND NO BAR. The single"
          f" registered ablation contrast: {TARGET} dropped, band shrink on"
          " q0. The eighteen feature table is deliberately NOT read here"
          " (registered): one contrast cannot be picked from, eighteen can.")
    cols = [c for c in range(len(FEATURE_NAMES)) if c != j]
    ab = np.abs(q0gap[BAND[0]]) - np.abs(gap(BAND, Q0, cols, TARGET))
    am, ase, at = paired(ab.tolist())
    print(f"    q0 band shrink {am:+.4f} se {ase:.4f} t {at:+.2f}"
          f"   A64 read {REF_ABL_K0:+.4f} on k0")

    res = dict(k=K, seeds=SEEDS, target=TARGET, gate_b=True,
               reproduction_gate=repro_ok, calibration_worst=worst,
               k0_gap={b: float(k0gap[b].mean()) for b in k0gap},
               q0_gap={b: dict(zip(("mean", "se", "t"),
                                   paired(q0gap[b].tolist()))) for b in q0gap},
               read1=bool(read1), read2=read2,
               read3=bool(read3), importance=imp,
               ablation_q0_band=dict(mean=am, se=ase, t=at),
               verdict=verdict, non_contract_auc=True)
    with open("research/w4_e1prox.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1prox.json")
    print("  one trajectory per row, no selection, diagnostic only,"
          " never a training signal, no serve decision")

    rid = ledger.append_row(
        "w4_e1prox",
        {"seeds": SEEDS, "k": K, "perm_seed": PERM_SEED, "zero": ZERO,
         "arms": [K0, Q0], "band": BAND[0], "target": TARGET,
         "reference": "A65, does k0 stand in for the served q0"},
        "ok",
        metrics={"calibration_worst": worst,
                 "q0_band": float(q0gap[BAND[0]].mean()),
                 "q0_full": float(q0gap[FULL[0]].mean()),
                 "read1": int(read1), "read3": int(read3),
                 "read2_band_null": int(read2[BAND[0]]["null"]),
                 "imp_band_t": imp[BAND[0]]["t"],
                 "ablation_q0_band": am},
        artifacts=["research/w4_e1prox.json"],
        notes=f"AMENDMENT 65. Does the free running arm k0 stand in for the"
              f" served arm q0. {verdict}. q0 band"
              f" {float(q0gap[BAND[0]].mean()):+.4f}, full"
              f" {float(q0gap[FULL[0]].mean()):+.4f}, k0 baselines"
              f" reproduced bit identical to A64. NON CONTRACT diagnostic"
              f" AUC, never comparable to the headline. Diagnostic only,"
              f" registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
