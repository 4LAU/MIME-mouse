"""w4_step0_tune. Choose the bin count for w4_step0's per position statistic.

Registered in /home/aaronadmin/w4_arms/step0_prereg.md AMENDMENT 4.

16 bins was inherited from `w4_poskl`, where the statistic served a SLOPE over
40 positions and never had to resolve a LEVEL at one position. Total variation
noise grows with the bin count and the signal need not, so 16 is unlikely to be
the resolving optimum.

TUNED ON A PLANT AND NOTHING ELSE. Human against human, with a known planted
difference of the size `w4_step0`'s AMENDMENT 3 calibration settled on. THE
MODEL ARM IS NEVER READ HERE. Choosing bins to make the model's own excess look
bigger would be fitting the instrument to the answer.

CPU only. Nothing generated.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

os.environ.setdefault("EVENT_CHOICE_TEMP", "10")
for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.event_ar import N_DT_CLASSES                          # noqa: E402
from models.event_stream_polar import (S_PAD_CLASS, TH_BINS,      # noqa: E402
                                       TH_NULL_CLASS)
from w4_detcap import CORPUS, corpus_tokens                       # noqa: E402
from w4_poskl import channels, cuts_from, tv                      # noqa: E402
from w4_step0 import bin_shift, plant_at, prep                    # noqa: E402

KMAX = 20
NROW = 3111                 # matches the KMAX 20 model arm the first run had
GRID = (4, 6, 8, 12, 16, 24)
SEEDS = (301, 302, 303, 304, 305, 306)
CH = ("s", "th", "dt")
MOD = {"s": S_PAD_CLASS, "th": TH_BINS, "dt": N_DT_CLASSES}
# The AMENDMENT 3 calibration, copied so this file cannot drift from it.
PLANT = {"s": 0.05, "th": 0.02, "dt": 0.05}


def cuts_for(ref, nbin):
    out = {}
    for name in CH:
        v = ref[name]
        live = v[v < TH_NULL_CLASS] if name == "th" else v.reshape(-1)
        out[name] = cuts_from(live, nbin)
    return out


def dig(v, name, cuts):
    if name == "th":
        b = np.digitize(v, cuts[name])
        return np.where(v >= TH_NULL_CLASS, len(cuts[name]) + 1, b)
    return np.digitize(v, cuts[name])


def main():
    rng = np.random.default_rng(7700)
    Lall = np.load(CORPUS / "events_len.npy")
    elig = np.flatnonzero(Lall >= KMAX)
    elig = rng.choice(elig, 300000, replace=False)

    ref_ids = rng.choice(elig, 20000, replace=False)
    rs, rth, rdt, _, rL = corpus_tokens(ref_ids)
    rs, rth, rdt, _ = prep(rs, rth, rdt, rL, KMAX)
    ref = channels(rs, rth, rdt)

    def pair(seed):
        ids = rng.choice(elig, 2 * NROW, replace=False)
        a, b, c, _, L = corpus_tokens(ids)
        a, b, c, _ = prep(a, b, c, L, KMAX)
        o = np.random.default_rng(seed).permutation(len(a))
        a, b, c = a[o], b[o], c[o]
        h = len(a) // 2
        return (a[:h], b[:h], c[:h]), (a[h:], b[h:], c[h:])

    pairs = [pair(sd) for sd in SEEDS]

    print("w4_step0_tune. bin count chosen on a PLANT. the model arm is never "
          "read here.", flush=True)
    print(f"  rows per side {NROW // 2 * 2 // 2}, {len(SEEDS)} seeds, "
          f"KMAX {KMAX}", flush=True)

    out = {}
    for name in CH:
        print(f"\n  {name}   plant rate {PLANT[name]}")
        print(f"  {'bins':>6}{'shift':>7}{'planted':>10}{'null sd':>10}"
              f"{'z':>8}{'':>6}")
        # THE PLANT IS HELD FIXED ACROSS THE GRID. First attempt derived the
        # shift from each candidate's own cut spacing, so 4 bins got a 121 class
        # shift on direction and 16 bins got 5, and the apparent 5x "gain" at 6
        # bins was a bigger perturbation rather than a better statistic. The
        # shift is now taken once from the registered 16 bin baseline and reused.
        base_cuts = cuts_for(ref, 16)
        sh = bin_shift(base_cuts[name])
        rows = []
        for nbin in GRID:
            cuts = cuts_for(ref, nbin)
            nullv, plantv = [], []
            for (A, B) in pairs:
                Ach = channels(*A)
                Bch = channels(*B)
                ks = np.arange(5, KMAX)
                pa, pb, pc = plant_at(*B, np.random.default_rng(hash((nbin, name))
                                                                % 2**31),
                                      PLANT[name], ks, name, sh, MOD[name])
                Pch = channels(pa, pb, pc)
                nb = nbin + 3
                da = dig(Ach[name], name, cuts)
                db = dig(Bch[name], name, cuts)
                dp = dig(Pch[name], name, cuts)
                nullv.append(np.mean([tv(da[:, k], db[:, k], nb)
                                      for k in ks]))
                plantv.append(np.mean([tv(da[:, k], dp[:, k], nb)
                                       for k in ks]))
            nullv, plantv = np.array(nullv), np.array(plantv)
            eff = float(np.mean(plantv - nullv))
            sd = float(np.std(nullv, ddof=1))
            z = eff / max(sd, 1e-9)
            rows.append(dict(nbin=nbin, shift=int(sh), planted=eff, sd=sd, z=z))
            print(f"  {nbin:>6}{sh:>7}{eff:>10.4f}{sd:>10.4f}{z:>8.2f}",
                  flush=True)
        best = max(rows, key=lambda r: r["z"])
        base = [r for r in rows if r["nbin"] == 16][0]
        gain = best["z"] / max(base["z"], 1e-9)
        keep = best["nbin"] if gain > 1.2 else 16
        print(f"    best {best['nbin']} bins, z {best['z']:.2f} against "
              f"16 bins z {base['z']:.2f}, gain {gain:.2f}x  -> "
              f"{'ADOPT ' + str(keep) if keep != 16 else 'KEEP 16, null result'}")
        out[name] = dict(rows=rows, best=best["nbin"], gain=gain, chosen=keep)

    json.dump(out, open("research/w4_step0_tune.json", "w"), indent=1)
    print("\n  wrote research/w4_step0_tune.json")
    print(f"  CHOSEN {[(n, out[n]['chosen']) for n in CH]}")


if __name__ == "__main__":
    main()
