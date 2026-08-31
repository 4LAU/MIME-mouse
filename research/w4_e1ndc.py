"""w4_e1ndc. AMENDMENT 62, registered in step0_prereg.md before this file
existed and before seed 64 had ever been generated.

A61 found that the forest leans on num_direction_changes LESS when
separating k0 from the anchor than when separating the L3_FULL floor
from it inside a length band, and MORE on the full row set. The sign
reversed with the AUC ordering, which is what an account of that
ordering has to do. A61 could not confirm it: seeds 52 to 63 had been
read five times by then, and A61's outcome says so in those words.

This reads seeds 64 to 75, which have never been generated and never
been scored. READ 0 checks that the AUC ordering itself replicates,
because if it does not the channel question does not arise. READ 1 is
the registered decision and its three outcomes are written down in the
prereg, including that UNDECIDED IS NOT CONFIRMATION.

CPU only, no torch, no generation. Diagnostic only, never a training
signal, never a serve candidate, no selection.
"""
import json
import re
import sys

import numpy as np

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import scoring   # noqa: E402
import ledger    # noqa: E402
from features import FEATURE_NAMES   # noqa: E402

SEEDS = list(range(64, 76))
K = 20
PERM_SEED = 3208
N = 2000
MAX_T = 256
KMAX = 4
TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
FLOOR = "research/w4_e1floor_F_{a}_s{s}.npy"
ARMF = "research/w4_e1feat_F_{a}_s{s}.npy"
ZERO = "L3_FULL"
ARM = "k0"
BAND = ("17 to 32", 17, 32)
FULL = ("full", 0, MAX_T)
OTHER = [("5 to 15", 5, 15), ("33 to 64", 33, 64), ("65 to 256", 65, 256)]
PRIMARY_FEATURE = "num_direction_changes"
SECOND_FEATURE = "movement_duration"
# A61 on seeds 52 to 63, research/w4_e1chan4.json
REF_ZERO = {"full": 0.517344, "17 to 32": 0.753872}
GATE_W = 0.015
REF_GAP = {"full": +0.0592, "17 to 32": -0.0192}
A61_SHIFT = {"full": +0.0038, "17 to 32": -0.0348,
             "5 to 15": -0.0352, "33 to 64": -0.0387, "65 to 256": -0.0064}
MEAN_BAR = 0.010
T_BAR = 3.0
# AMENDMENT 62 ADDENDUM, INTEGRITY GATE B. Registered as a write up
# check; enforced here instead, which is stricter and not weaker.
GEN_LOG = "/home/aaronadmin/w4_arms/qladder_a62.log"
BANNER = ("[event_stream_polar] ckpt=event_polar_best.pt epoch=22"
          " steps=100 temp=1.0 th_temp=None order=gumbel round=True"
          " bestof=1")
CKPT_MD5 = "91326a29750789f3167055324ef377c5"


def gate_b():
    """banner identical to A57's, checkpoint unmoved, no cached arm."""
    import os
    txt = open(GEN_LOG).read()
    banners = [ln.strip() for ln in txt.splitlines()
               if ln.startswith("[event_stream_polar]")]
    md5s = re.findall(r"^([0-9a-f]{32})  training/", txt, re.M)
    ok_b = bool(banners) and all(b == BANNER for b in banners)
    ok_m = bool(md5s) and all(m == CKPT_MD5 for m in md5s)
    print(f"    banner lines {len(banners)}"
          f"   {'all identical to A57' if ok_b else 'DIFFER FROM A57'}")
    print(f"    md5 lines {len(md5s)}"
          f"   {'checkpoint unmoved' if ok_m else 'CHECKPOINT MOVED'}")
    if not ok_b:
        for b in sorted(set(banners))[:3]:
            print(f"      got: {b}")
    return ok_b and ok_m


def scored(m):
    aucs, imps = [], []
    for k in range(K):
        r = scoring.score_features(
            m[np.random.default_rng(PERM_SEED + k).permutation(len(m))])
        aucs.append(r["auc_rf_oob"])
        imps.append([r["importances"][n] for n in FEATURE_NAMES])
    return float(np.mean(aucs)), np.asarray(imps).mean(0)


def paired(v):
    d = np.asarray(v, float)
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return m, se, (m / se if se > 0 else float("inf"))


def bar(m, t):
    if abs(m) >= MEAN_BAR and abs(t) >= T_BAR:
        return "REAL"
    if abs(t) < 2.0:
        return "NULL"
    return "BETWEEN"


def main():
    print(f"  every value is the mean of K={K} permutations,"
          f" {len(SEEDS)} seeds {SEEDS[0]} to {SEEDS[-1]},"
          f" NEVER GENERATED AND NEVER SCORED BEFORE THIS RUN.")
    print(f"  A61's lead is on trial. UNDECIDED IS NOT CONFIRMATION"
          f" (registered).\n", flush=True)

    lengths = np.load("training/events_len.npy")
    NT = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(NT, min(NT, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(NT), trained)
    elig = held[lengths[held] > KMAX]

    mats, per_ok, per_len = {}, {}, {}
    for s in SEEDS:
        m = {ZERO: np.load(FLOOR.format(a=ZERO, s=s)),
             ARM: np.load(ARMF.format(a=ARM, s=s))}
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

    cache = {}

    def get(band, which):
        key = (band[0], which)
        if key not in cache:
            a, im = [], []
            for s in SEEDS:
                x, y = scored(mats[s][which][rows(s, band[1], band[2])])
                a.append(x)
                im.append(y)
            cache[key] = (np.asarray(a), np.asarray(im))
        return cache[key]

    print(f"  INTEGRITY GATE B (AMENDMENT 62 ADDENDUM), the fresh"
          f" generation must match A56's settings exactly:")
    gb = gate_b()
    print(f"  {'GATE B PASSED' if gb else 'GATE B FAILED, every read below is VOID (registered)'}",
          flush=True)
    if not gb:
        return

    print(f"\n  GATE (sanity on the generation, NOT a test of the claim)."
          f" The zero's twelve seed mean must land within {GATE_W} of"
          f" A61's on both row sets:")
    gate = True
    for band in (BAND, FULL):
        want = REF_ZERO[band[0]]
        got = float(get(band, ZERO)[0].mean())
        ok = abs(got - want) <= GATE_W
        gate = gate and ok
        print(f"    {band[0]:<10} A61 {want:.6f}  fresh {got:.6f}"
              f"  diff {got - want:+.6f}   {'ok' if ok else 'MISS'}",
              flush=True)
    print(f"  {'GATE PASSED' if gate else 'GATE FAILED, every read below is VOID (registered)'}",
          flush=True)
    if not gate:
        return

    print(f"\n  READ 0, THE PRECONDITION. Does the AUC ordering itself"
          f" replicate on fresh seeds? The band term must be NEGATIVE and"
          f" the full row set term POSITIVE.")
    r0 = {}
    for band in (BAND, FULL):
        z, a = get(band, ZERO)[0], get(band, ARM)[0]
        m, se, t = paired(a - z)
        neg = int((a - z < 0).sum())
        want = "negative" if REF_GAP[band[0]] < 0 else "positive"
        ok = (m < 0) == (REF_GAP[band[0]] < 0) and abs(t) >= T_BAR
        r0[band[0]] = dict(mean=m, se=se, t=t, n_neg=neg, ok=bool(ok),
                           a61=REF_GAP[band[0]])
        print(f"    {band[0]:<10} k0 minus zero {m:+.4f} se {se:.4f}"
              f" t {t:+6.2f}   {neg} of {len(SEEDS)} seeds negative"
              f"   A61 {REF_GAP[band[0]]:+.4f}"
              f"   {'ok, ' + want + ' as required' if ok else 'DOES NOT REPLICATE'}",
              flush=True)
    r0_ok = all(v["ok"] for v in r0.values())
    if not r0_ok:
        print(f"    THE ORDERING DOES NOT REPLICATE ON FRESH SEEDS."
              f" The channel question does not arise and THAT is the"
              f" result of this amendment (registered).")

    def shift(band, feat):
        j = FEATURE_NAMES.index(feat)
        return get(band, ARM)[1][:, j] - get(band, ZERO)[1][:, j]

    print(f"\n  READ 1 (PRIMARY), {PRIMARY_FEATURE} alone, importance share,"
          f" k0 minus the zero:")
    r1 = {}
    for band in (BAND, FULL):
        m, se, t = paired(shift(band, PRIMARY_FEATURE))
        r1[band[0]] = dict(mean=m, se=se, t=t, bar=bar(m, t),
                           a61=A61_SHIFT[band[0]])
        print(f"    {band[0]:<10}{m:+.4f} se {se:.4f} t {t:+6.2f}"
              f"  {r1[band[0]]['bar']:<8} A61 {A61_SHIFT[band[0]]:+.4f}",
              flush=True)
    b, f = r1[BAND[0]], r1[FULL[0]]
    band_ok = (b["bar"] == "REAL" and b["mean"] < 0)
    if not band_ok:
        call = "KILLED, the band shift is not REAL and negative"
    elif f["mean"] < 0 and abs(f["t"]) >= T_BAR:
        call = "KILLED, the full row set shift is negative and clears t"
    elif f["mean"] > 0 and abs(f["t"]) >= T_BAR:
        call = "CONFIRMED, the sign reverses with the ordering on fresh seeds"
    else:
        call = ("UNDECIDED, the band shift is real and the reversal did not"
                " replicate. UNDECIDED IS NOT CONFIRMATION (registered)")
    print(f"    THE REGISTERED DECISION: {call}", flush=True)

    print(f"\n  READ 2, {SECOND_FEATURE}, which A61 DISQUALIFIED for having"
          f" the same sign in both regimes:")
    r2 = {}
    for band in (BAND, FULL):
        m, se, t = paired(shift(band, SECOND_FEATURE))
        r2[band[0]] = dict(mean=m, se=se, t=t, bar=bar(m, t))
        print(f"    {band[0]:<10}{m:+.4f} se {se:.4f} t {t:+6.2f}"
              f"  {r2[band[0]]['bar']}", flush=True)
    same = np.sign(r2[BAND[0]]["mean"]) == np.sign(r2[FULL[0]]["mean"])
    print(f"    {'DISQUALIFIED again, same sign in both regimes' if same else 'IT REVERSES ON FRESH SEEDS, so A61 disqualified it on a seed artifact and A59 duration account is shakier than I have treated it'}")

    print(f"\n  READ 3, DESCRIPTIVE, NO VERDICT. Every feature in the"
          f" {BAND[0]} band, fresh seeds beside A61's leaders:")
    allsh = get(BAND, ARM)[1] - get(BAND, ZERO)[1]
    rows3 = []
    for j, name in enumerate(FEATURE_NAMES):
        m, se, t = paired(allsh[:, j])
        rows3.append((m, name, se, t))
    r3 = {}
    print(f"    {'feature':<28}{'mean':>8}{'se':>8}{'t':>8}  bar")
    for m, name, se, t in sorted(rows3, reverse=True):
        r3[name] = dict(mean=m, se=se, t=t, bar=bar(m, t))
        print(f"    {name:<28}{m:+.4f}  {se:.4f}  {t:+6.2f}  {bar(m, t)}",
              flush=True)
    lead = max(rows3, key=lambda x: abs(x[0]))[1]
    print(f"    largest absolute shift: {lead}"
          f"   (A61's was {PRIMARY_FEATURE})")

    print(f"\n  READ 4, DESCRIPTIVE, NO VERDICT, no relation registered"
          f" (A50). {PRIMARY_FEATURE} in the other bands:")
    r4 = {}
    for band in OTHER:
        m, se, t = paired(shift(band, PRIMARY_FEATURE))
        r4[band[0]] = dict(mean=m, se=se, t=t, bar=bar(m, t),
                           a61=A61_SHIFT[band[0]])
        print(f"    {band[0]:<11}{m:+.4f} se {se:.4f} t {t:+6.2f}"
              f"  {r4[band[0]]['bar']:<8} A61 {A61_SHIFT[band[0]]:+.4f}",
              flush=True)

    res = dict(k=K, seeds=SEEDS, gate=bool(gate), read0=r0,
               read0_ok=bool(r0_ok), read1=r1, decision=call,
               read2=r2, read2_disqualified=bool(same),
               read3=r3, read3_leader=lead, read4=r4,
               auc={b[0]: {w: get(b, w)[0].tolist() for w in (ZERO, ARM)}
                    for b in (BAND, FULL)})
    with open("research/w4_e1ndc.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1ndc.json")
    print("  one trajectory per row, no selection, diagnostic only,"
          " never a training signal, no serve decision")

    rid = ledger.append_row(
        "w4_e1ndc",
        {"seeds": SEEDS, "k": K, "perm_seed": PERM_SEED, "zero": ZERO,
         "arm": ARM, "band": BAND[0], "feature": PRIMARY_FEATURE,
         "reference": "A61 fresh seed test"},
        "ok" if (gate and r0_ok) else "failed",
        metrics={"gate": int(gate), "read0_ok": int(r0_ok),
                 "band_shift": b["mean"], "band_t": b["t"],
                 "full_shift": f["mean"], "full_t": f["t"],
                 "confirmed": int(call.startswith("CONFIRMED"))},
        artifacts=["research/w4_e1ndc.json"],
        notes=f"AMENDMENT 62 fresh seed test of A61's channel."
              f" {call}. Band {b['mean']:+.4f} t {b['t']:+.2f},"
              f" full {f['mean']:+.4f} t {f['t']:+.2f}."
              f" Seeds 64 to 75, never scored before."
              f" Diagnostic only, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
