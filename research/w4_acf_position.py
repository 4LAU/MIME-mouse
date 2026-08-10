"""Is the human speed lag 2 bump a new fact, or w4_position seen from another
angle, or just tick scheduling.

Zero GPU. Human corpus only. Registered in HANDOFF.md under "Is the speed lag 2
bump a new fact, or w4_position wearing a different hat" plus its amendment, and
the branches there are read off the numbers this prints.

Loading matches research/w4_seqstats.py exactly, same files, same seed, same
tokenize and detokenize round trip, so the whole corpus row here is comparable
to the `human` row there.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_acf_position.py --n 20000
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.event_stream_polar import (  # noqa: E402
    TICK_CLASS, class_to_speed, s2_to_class,
)

# Each third must hold at least eight events for a lag 2 product to mean
# anything inside it, so 24 is the floor. Registered before the run.
MIN_LEN = 24
NTHIRD = 3
BOOT = 400


def contribs(v, nthird=NTHIRD, maxlag=2):
    """Per sequence numerator and denominator of the pooled autocorrelation,
    split by position third.

    Standardisation uses the WHOLE sequence, not the third, so that a third with
    a lower local spread is not silently rescaled to look like the others. The
    product z[t] * z[t+k] is assigned to the third of its LEFT index t.

    Returns (num, den), each shaped (nthird, maxlag), or None if the sequence is
    too short or flat.
    """
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    if n < MIN_LEN:
        return None
    sd = v.std()
    if sd < 1e-12:
        return None
    z = (v - v.mean()) / sd
    num = np.zeros((nthird, maxlag))
    den = np.zeros((nthird, maxlag))
    third = np.minimum(nthird - 1, (nthird * np.arange(n)) // n)
    for k in range(1, maxlag + 1):
        prod = z[:-k] * z[k:]
        t = third[:n - k]
        for j in range(nthird):
            m = t == j
            num[j, k - 1] = prod[m].sum()
            den[j, k - 1] = m.sum()
    return num, den


def pooled(num, den):
    """(S, nthird, maxlag) stacks -> pooled acf, summing over sequences."""
    return num.sum(0) / np.maximum(den.sum(0), 1)


def boot_se(num, den, rng, stat, nboot=BOOT):
    """Bootstrap over SEQUENCES. Positions inside one trajectory are not
    independent, so resampling positions would understate every error bar here
    by a large factor."""
    s = len(num)
    vals = []
    for _ in range(nboot):
        idx = rng.integers(0, s, s)
        vals.append(stat(pooled(num[idx], den[idx])))
    return np.std(vals, axis=0, ddof=1)


def whole(seqs, rng, maxlag=6):
    """Whole sequence pooled acf on the same filtered set, with a bootstrap se
    on the bump, so the length filter cannot be blamed for any difference."""
    num, den = [], []
    for v in seqs:
        r = contribs(v, nthird=1, maxlag=maxlag)
        if r is None:
            continue
        num.append(r[0])
        den.append(r[1])
    num = np.array(num)
    den = np.array(den)
    ac = pooled(num, den)[0]
    se = boot_se(num, den, rng, lambda p: p[0, 1] - p[0, 0])
    return ac, float(se), len(num)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="research/w4_acf_position.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    lengths = np.load("training/events_len.npy")
    pick = np.sort(rng.choice(len(lengths), min(args.n * 3, len(lengths)),
                              replace=False))
    s2 = np.load("training/events_s2.npy", mmap_mode="r")[pick]
    L = np.minimum(lengths[pick], 256)

    cls = s2_to_class(torch.from_numpy(s2.astype(np.int64))).numpy()

    tok, raw, motion_only = [], [], []
    ticks, total = 0, 0
    for i in range(len(L)):
        n = int(L[i])
        if n < MIN_LEN:
            continue
        c = cls[i, :n]
        sv = class_to_speed(torch.from_numpy(c.astype(np.int64))).numpy()
        rv = np.sqrt(np.maximum(s2[i, :n].astype(np.float64), 0.0))
        is_tick = c <= TICK_CLASS
        ticks += int(is_tick.sum())
        total += n
        tok.append(sv)
        raw.append(rv)
        motion_only.append(sv[~is_tick])
        if len(tok) >= args.n:
            break

    print(f"\n  human corpus, {len(tok)} trajectories at least {MIN_LEN} events")
    print(f"  tick share {ticks / max(total, 1):.4f} of all events\n")

    out = {"n_traj": len(tok), "tick_share": ticks / max(total, 1),
           "min_len": MIN_LEN, "nboot": BOOT}

    # whole sequence, tokenized, the row comparable to w4_seqstats
    ac, se, ns = whole(tok, np.random.default_rng(args.seed + 1))
    print("  WHOLE SEQUENCE, tokenized speeds")
    print("    lag      " + "".join(f"{k:>9d}" for k in range(1, 7)))
    print("    acf      " + "".join(f"{a:>9.4f}" for a in ac))
    print(f"    bump ac2 minus ac1  {ac[1] - ac[0]:+.4f}  se {se:.4f}\n")
    out["whole_tok"] = {"acf": ac.tolist(), "bump": float(ac[1] - ac[0]),
                        "bump_se": se, "n": ns}

    # tokenization control
    acr, ser, _ = whole(raw, np.random.default_rng(args.seed + 2))
    dif = (ac[1] - ac[0]) - (acr[1] - acr[0])
    dse = float(np.hypot(se, ser))
    print("  TOKENIZATION CONTROL, raw sqrt(s2) speeds")
    print("    acf      " + "".join(f"{a:>9.4f}" for a in acr))
    print(f"    bump                {acr[1] - acr[0]:+.4f}  se {ser:.4f}")
    print(f"    tokenized minus raw {dif:+.4f}  se {dse:.4f}"
          f"   {'ARTEFACT' if abs(dif) > 2 * dse else 'clean'}\n")
    out["whole_raw"] = {"acf": acr.tolist(), "bump": float(acr[1] - acr[0]),
                        "bump_se": ser}
    out["tok_minus_raw"] = {"diff": float(dif), "se": dse,
                            "artefact": bool(abs(dif) > 2 * dse)}

    # tick control, the amendment
    acm, sem_, nm = whole(motion_only, np.random.default_rng(args.seed + 3))
    survives = (acm[1] - acm[0]) > 2 * sem_
    print("  TICK CONTROL, motion events only, ticks dropped")
    print("    acf      " + "".join(f"{a:>9.4f}" for a in acm))
    print(f"    bump                {acm[1] - acm[0]:+.4f}  se {sem_:.4f}"
          f"   {'SURVIVES' if survives else 'DISAPPEARS'}\n")
    out["motion_only"] = {"acf": acm.tolist(), "bump": float(acm[1] - acm[0]),
                          "bump_se": sem_, "n": nm, "survives": bool(survives)}

    # position resolved, the registered branches
    num, den = [], []
    for v in tok:
        r = contribs(v)
        if r is None:
            continue
        num.append(r[0])
        den.append(r[1])
    num = np.array(num)
    den = np.array(den)
    p = pooled(num, den)
    brng = np.random.default_rng(args.seed + 4)
    bse = boot_se(num, den, brng, lambda q: q[:, 1] - q[:, 0])

    print("  POSITION RESOLVED, tokenized speeds, thirds of each trajectory")
    print(f"    {'third':<10}{'ac1':>9}{'ac2':>9}{'bump':>10}{'se':>9}")
    names = ["first", "middle", "last"]
    for j in range(NTHIRD):
        print(f"    {names[j]:<10}{p[j, 0]:>9.4f}{p[j, 1]:>9.4f}"
              f"{p[j, 1] - p[j, 0]:>+10.4f}{bse[j]:>9.4f}")
    out["by_third"] = [{"third": names[j], "ac1": float(p[j, 0]),
                        "ac2": float(p[j, 1]),
                        "bump": float(p[j, 1] - p[j, 0]),
                        "bump_se": float(bse[j])} for j in range(NTHIRD)]

    # the last minus first contrast, bootstrapped as a PAIRED difference so the
    # sequence level correlation between the two thirds is carried, not ignored
    dlf = boot_se(num, den, np.random.default_rng(args.seed + 5),
                  lambda q: (q[2, 1] - q[2, 0]) - (q[0, 1] - q[0, 0]))
    contrast = (p[2, 1] - p[2, 0]) - (p[0, 1] - p[0, 0])
    print(f"\n    last minus first bump  {contrast:+.4f}  se {float(dlf):.4f}")
    out["last_minus_first"] = {"diff": float(contrast), "se": float(dlf)}

    b = [p[j, 1] - p[j, 0] for j in range(NTHIRD)]
    first_zero = abs(b[0]) <= 2 * bse[0]
    all_pos = all(b[j] > 2 * bse[j] for j in range(NTHIRD))
    flat = abs(contrast) <= 2 * float(dlf)
    if contrast > 2 * float(dlf) and first_zero:
        verdict = "LATE CONCENTRATED, same fact as w4_position, thread closes"
    elif all_pos and flat:
        verdict = "UNIFORM, separate structural fact, thread stays alive"
    elif contrast < -2 * float(dlf):
        verdict = "EARLY CONCENTRATED, contradicts w4_position"
    else:
        verdict = "MIXED, report the curve, no verdict, no build"
    print(f"\n    VERDICT  {verdict}\n")
    out["verdict"] = verdict

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}\n")


if __name__ == "__main__":
    main()
