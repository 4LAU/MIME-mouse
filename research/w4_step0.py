"""w4_step0. How much of the defect is already there at the FIRST event?

Registered in /home/aaronadmin/w4_arms/step0_prereg.md with two amendments,
both written before this file existed.

`w4_poskl` found the divergence FLAT in position and concluded compounding is
not supported. A flat slope is consistent with two mechanisms it cannot separate:
the defect is present at event 0 and never grows, or the defect is absent at
event 0 and equilibrates within two or three events. `w4_poskl`'s power control
was a constant heading drift, which is the slow shape, so it has no power against
the fast one.

Reading a LEVEL rather than a slope means the query condition mismatch between
the two populations has to be removed, which `w4_poskl` sidestepped by reading a
slope. Removed here by exact per cell subsampling on the condition vector.

CPU only. Reads streams that already exist. Generates nothing, trains nothing.
"""
from __future__ import annotations

import argparse
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
from w4_detcap import CORPUS, corpus_tokens, model_tokens         # noqa: E402
from w4_poskl import (NBIN, channels, cuts_from, digitize,        # noqa: E402
                      make_cuts, tv)

NCELL_Q = 4        # quantile bins on cond dims 0 and 1, human defined
NCELL_A = 8        # equal width bins on the condition angle
NCELL = NCELL_Q * NCELL_Q * NCELL_A
DRAWS = 8          # independent matched human draws, TV averaged over them
NULL_SEEDS = (201, 202, 203, 204, 205, 206, 207, 208)
CH = ("s", "th", "dt", "cumhead")


# ------------------------------------------------------------------ cells ---
def cell_cuts(cond):
    """Cut points from the HUMAN reference only, so cells never depend on the
    model. Dims 0 and 1 by quantile, the angle by equal width because it is
    already uniform on a circle and quantiles there would be meaningless."""
    return [cuts_from(cond[:, 0], NCELL_Q), cuts_from(cond[:, 1], NCELL_Q)]


def cells(cond, cuts):
    a = np.digitize(cond[:, 0], cuts[0])
    b = np.digitize(cond[:, 1], cuts[1])
    ang = np.arctan2(cond[:, 3], cond[:, 2])
    c = np.clip(((ang + np.pi) / (2 * np.pi) * NCELL_A).astype(np.int64),
                0, NCELL_A - 1)
    return (a * NCELL_Q + b) * NCELL_A + c


def draw_matched(pool_by_cell, want, rng, mult=1):
    """Draw `mult * want[c]` ids from each cell without replacement.

    Returns the ids and the per cell shortfall. Nothing is silently truncated:
    a cell that cannot supply is reported and counted against the contamination
    gate, which is what the registered N4 replacement asks for."""
    out, short = [], np.zeros(NCELL, dtype=np.int64)
    for c in np.flatnonzero(want):
        need = int(want[c]) * mult
        have = pool_by_cell[c]
        if len(have) < need:
            short[c] = need - len(have)
            out.append(rng.permutation(have))
            continue
        out.append(rng.choice(have, need, replace=False))
    return np.concatenate(out) if out else np.zeros(0, np.int64), short


# ------------------------------------------------------------------- read ---
def prep(s, th, dt, L, kmax):
    keep = np.asarray(L) >= kmax
    return s[keep, :kmax], th[keep, :kmax], dt[keep, :kmax], keep


def curve(A, B, kmax):
    """TV per position between two already digitized channel matrices."""
    return np.array([tv(A[:, k], B[:, k], NBIN + 3) for k in range(kmax)])


def arm_curves(hs, ms, cuts, kmax):
    hb, mb = digitize(hs, cuts), digitize(ms, cuts)
    return {n: curve(hb[n], mb[n], kmax) for n in CH}


# --------------------------------------------------------------- planting ---
# AMENDMENT 3. The first run's plant moved the direction class by +1, which
# changes 16 quantile BIN for only 11.4 percent of live tokens, measured on
# 20,000 corpus rows. The direction distribution is concentrated enough that the
# 16 quantile cut points collapse to 11 distinct values with spacings of 2 and 3
# classes near the mode. So the plant now moves by a whole BIN SPACING and its
# rate is CALIBRATED against the model's own measured excess.


def bin_shift(cuts):
    """A shift large enough to cross a bin where the bins are narrowest, which
    is where most of the mass sits."""
    d = np.diff(cuts)
    return int(max(1, np.ceil(np.median(d))))


def plant_at(s, th, dt, rng, p, ks, ch, shift, mod):
    """Perturb ONE channel at the given positions ONLY. Used twice per channel,
    once at k = 0 alone and once at k >= 5 alone, so the instrument has to
    distinguish the two by construction or it cannot answer the question."""
    s, th, dt = s.copy(), th.copy(), dt.copy()
    tgt = {"s": s, "th": th, "dt": dt}[ch]
    sub = tgt[:, ks]
    live = sub < TH_NULL_CLASS if ch == "th" else np.ones(sub.shape, bool)
    bump = (rng.random(sub.shape) < p) & live
    tgt[:, ks] = np.where(bump, (sub + shift) % mod, sub)
    return s, th, dt


# ------------------------------------------------------------------- main ---
def run(kmax, mstreams, hpool_n, args):
    rng = np.random.default_rng(7300)

    ms, mth, mdt, mcond, mL = [], [], [], [], []
    for f in mstreams:
        a, b, c, d, L = model_tokens(f)
        ms.append(a); mth.append(b); mdt.append(c); mcond.append(d); mL.append(L)
    ms, mth, mdt = np.concatenate(ms), np.concatenate(mth), np.concatenate(mdt)
    mcond, mL = np.concatenate(mcond), np.concatenate(mL)
    ms, mth, mdt, keep = prep(ms, mth, mdt, mL, kmax)
    mcond = mcond[keep]

    Lall = np.load(CORPUS / "events_len.npy")
    Call = np.load(CORPUS / "events_cond.npy")
    elig = np.flatnonzero(Lall >= kmax)
    if len(elig) > hpool_n:
        elig = rng.choice(elig, hpool_n, replace=False)
    hc = np.asarray(Call[elig], dtype=np.float32)

    ccuts = cell_cuts(hc)
    hcell, mcell = cells(hc, ccuts), cells(mcond, ccuts)
    want = np.bincount(mcell, minlength=NCELL)
    pool = [elig[hcell == c] for c in range(NCELL)]

    print(f"\n  KMAX {kmax}   model rows {len(ms)}   human pool {len(elig)}   "
          f"cells used {int((want > 0).sum())} of {NCELL}", flush=True)

    # Channel cut points come from a plain human sample, never from the model
    # and never from the matched arm, so the alphabet is fixed before any
    # comparison is made.
    ref = corpus_tokens(rng.choice(elig, min(20000, len(elig)), replace=False))
    rs, rth, rdt, _, rL = ref
    rs, rth, rdt, _ = prep(rs, rth, rdt, rL, kmax)
    cuts = make_cuts(channels(rs, rth, rdt))

    def human_arm(ids):
        a, b, c, _, L = corpus_tokens(ids)
        a, b, c, _ = prep(a, b, c, L, kmax)
        return channels(a, b, c)

    mch = channels(ms, mth, mdt)

    # ---- ARM. model against DRAWS independent matched human draws ----------
    acc, shorts = [], np.zeros(NCELL, dtype=np.int64)
    for d in range(DRAWS):
        ids, sh = draw_matched(pool, want, np.random.default_rng(7400 + d))
        shorts = np.maximum(shorts, sh)
        acc.append(arm_curves(human_arm(ids), mch, cuts, kmax))
    model_tv = {n: np.mean([a[n] for a in acc], 0) for n in CH}

    short_frac = float(shorts.sum()) / max(int(want.sum()), 1)
    print(f"  cell supply shortfall {short_frac * 100:.2f} percent of model rows",
          flush=True)

    # ---- NULL. two disjoint human sets, SAME per cell counts --------------
    def null_curves(seed, plant=None):
        ids, _ = draw_matched(pool, want, np.random.default_rng(seed), mult=2)
        o = np.random.default_rng(seed + 50).permutation(len(ids))
        ids = ids[o]
        h = len(ids) // 2
        A, B = human_arm(ids[:h]), human_arm(ids[h:])
        if plant is not None:
            pr, ks_, pch, sh, mod = plant
            a, b, c = plant_at(B["s"], B["th"], B["dt"],
                               np.random.default_rng(seed + 90), pr, ks_,
                               pch, sh, mod)
            B = channels(a, b, c)
        return arm_curves(A, B, cuts, kmax)

    nacc = [null_curves(sd) for sd in NULL_SEEDS]
    null_tv = {n: np.mean([a[n] for a in nacc], 0) for n in CH}
    null_sd = {n: np.std([a[n] for a in nacc], 0) for n in CH}

    def share(mtv, ntv):
        out = {}
        for n in CH:
            c = mtv[n] - ntv[n]
            plateau = float(np.mean(c[5:]))
            out[n] = dict(k0=float(c[0]), plateau=plateau,
                          share0=float(c[0] / plateau) if abs(plateau) > 1e-9
                          else float("nan"),
                          early=[float(v) for v in c[:9]],
                          kstar=int(np.argmax(c >= 0.9 * plateau))
                          if plateau > 0 else -1)
        return out

    res = share(model_tv, null_tv)

    print(f"\n  null corrected TV by position, k = 0..8 then the k>=5 plateau")
    print(f"  {'ch':>8}" + "".join(f"{k:>8}" for k in range(9)) +
          f"{'plateau':>10}{'share0':>9}{'k*':>4}")
    for n in CH:
        r = res[n]
        print(f"  {n:>8}" + "".join(f"{v:>8.4f}" for v in r["early"]) +
              f"{r['plateau']:>10.4f}{r['share0']:>9.2f}{r['kstar']:>4}")

    # ---- N1. does the pipeline itself make k = 0 special? ------------------
    n1 = {}
    for n in CH:
        base = float(np.mean(null_tv[n][5:]))
        sd = float(np.mean(null_sd[n][5:])) / np.sqrt(len(NULL_SEEDS))
        z = (null_tv[n][0] - base) / max(sd, 1e-9)
        n1[n] = dict(tvnull0=float(null_tv[n][0]), tvnull_plateau=base,
                     z=float(z), ok=bool(abs(z) <= 3.0))
    print(f"\n  N1 NULL. is k = 0 special for the pipeline alone")
    print(f"  {'ch':>8}{'TVnull(0)':>12}{'plateau':>10}{'z':>8}{'':>6}")
    for n in CH:
        print(f"  {n:>8}{n1[n]['tvnull0']:>12.4f}{n1[n]['tvnull_plateau']:>10.4f}"
              f"{n1[n]['z']:>8.2f}{'  ok' if n1[n]['ok'] else '  FAIL':>6}")

    # ---- N2. power, both directions ---------------------------------------
    ks0 = np.array([0])
    kslate = np.arange(5, kmax)
    MOD = {"s": S_PAD_CLASS, "th": TH_BINS, "dt": N_DT_CLASSES}
    PCH = {"s": "s", "th": "th", "dt": "dt", "cumhead": "th"}
    shifts = {c: bin_shift(cuts[c]) for c in ("s", "th", "dt")}

    # ---- calibrate the plant rate against the MODEL's own excess ----------
    # Registered in AMENDMENT 3. One coarse grid on one null seed pair, then
    # the chosen setting is run on four seeds like any other reading.
    cal = {}
    for n in CH:
        pc = PCH[n]
        target = abs(res[n]["plateau"])
        best = None
        for pr in (0.02, 0.05, 0.10, 0.20, 0.40):
            c = null_curves(NULL_SEEDS[0], plant=(pr, kslate, pc, shifts[pc],
                                                  MOD[pc]))
            got = float(np.mean(c[n][5:] - null_tv[n][5:]))
            d = abs(np.log(max(got, 1e-6) / max(target, 1e-6)))
            if best is None or d < best[0]:
                best = (d, pr, got)
        cal[n] = dict(p=best[1], achieved=best[2], target=target,
                      shift=shifts[pc], channel=pc,
                      within2x=bool(best[2] > 0 and 0.5 <= best[2] / max(target, 1e-9) <= 2.0))
    print(f"\n  PLANT CALIBRATION against the model's own plateau excess")
    print(f"  {'ch':>8}{'via':>6}{'shift':>7}{'p':>7}{'target':>9}{'achieved':>10}{'':>9}")
    for n in CH:
        c = cal[n]
        print(f"  {n:>8}{c['channel']:>6}{c['shift']:>7}{c['p']:>7.2f}"
              f"{c['target']:>9.4f}{c['achieved']:>10.4f}"
              f"{'  within 2x' if c['within2x'] else '  NOT CALIBRATABLE':>9}")

    # ---- N2. power, both directions, at the calibrated setting -------------
    pw = {}
    for tag, ks in (("k0_only", ks0), ("late_only", kslate)):
        pw[tag] = {}
        for n in CH:
            c = cal[n]
            pacc = [null_curves(sd, plant=(c["p"], ks, c["channel"], c["shift"],
                                           MOD[c["channel"]]))
                    for sd in NULL_SEEDS[:4]]
            ptv = np.mean([a[n] for a in pacc], 0) - null_tv[n]
            pw[tag][n] = dict(k0=float(ptv[0]), plateau=float(np.mean(ptv[5:])))

    scale = {n: float(np.mean(null_sd[n])) for n in CH}
    powered = {}
    for n in CH:
        a, b = pw["k0_only"][n], pw["late_only"][n]
        sees_k0 = a["k0"] > 3.0 * scale[n] and a["plateau"] < 0.3 * a["k0"]
        sees_late = b["plateau"] > 3.0 * scale[n] and b["k0"] < 0.3 * b["plateau"]
        powered[n] = bool(sees_k0 and sees_late and cal[n]["within2x"])
    print(f"\n  N2 POWER at the calibrated plant, null sd printed for scale")
    print(f"  {'ch':>8}{'sd':>8}{'k0plant k0':>12}{'k0plant plat':>14}"
          f"{'late k0':>10}{'late plat':>11}{'':>8}")
    for n in CH:
        a, b = pw["k0_only"][n], pw["late_only"][n]
        print(f"  {n:>8}{scale[n]:>8.4f}{a['k0']:>12.4f}{a['plateau']:>14.4f}"
              f"{b['k0']:>10.4f}{b['plateau']:>11.4f}"
              f"{'  powered' if powered[n] else '  UNPOWERED':>8}")

    return dict(kmax=kmax, n_model=int(len(ms)), share=res, n1=n1,
                calibration=cal,
                power={k: {n: v[n] for n in CH} for k, v in pw.items()},
                powered=powered, short_frac=short_frac,
                model_tv={n: model_tv[n].tolist() for n in CH},
                null_tv={n: null_tv[n].tolist() for n in CH})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="+",
                    default=["research/w4_texcover_streams_s0.npz",
                             "research/w4_texcover_streams_s1.npz"])
    ap.add_argument("--hpool", type=int, default=400000)
    ap.add_argument("--out", default="research/w4_step0_results.json")
    args = ap.parse_args()

    print("w4_step0. is the defect there at the first event? CPU only, "
          "nothing generated.", flush=True)
    out = {}
    for kmax in (40, 20):
        out[str(kmax)] = run(kmax, args.streams, args.hpool, args)

    # ---- verdict, by the registered rule -----------------------------------
    def verdict(r):
        ok = [n for n in CH if r["powered"][n] and r["n1"][n]["ok"]]
        if not ok:
            return "UNREADABLE", ok
        sh = [r["share"][n]["share0"] for n in ok]
        hi = sum(v >= 0.7 for v in sh)
        lo = sum(v <= 0.3 for v in sh)
        if hi > len(ok) / 2:
            return "PRESENT AT ONE", ok
        if lo > len(ok) / 2:
            return "BUILDS EARLY", ok
        return "MIXED", ok

    v40, ok40 = verdict(out["40"])
    v20, ok20 = verdict(out["20"])
    print(f"\n  VERDICT  KMAX 40  {v40}   on powered readable channels {ok40}")
    print(f"  VERDICT  KMAX 20  {v20}   on powered readable channels {ok20}")
    final = v40 if v40 == v20 else "MIXED, the two row budgets disagree"
    print(f"  FINAL    {final}")
    if max(out["40"]["short_frac"], out["20"]["short_frac"]) > 0.05:
        print("  CONTAMINATED. more than 5 percent of model rows sit in cells "
              "the human pool could not supply.")

    out["verdict"] = dict(kmax40=v40, kmax20=v20, final=final,
                          readable40=ok40, readable20=ok20)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
