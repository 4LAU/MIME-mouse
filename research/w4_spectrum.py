"""Does human motion carry frequency content the model does not?

Nothing in this repository's research history has ever looked at a spectrum.
Every "band" in the record is a distance band or a duration band. The programme
came at this as a statistics problem and built statistics: means, spreads,
pairwise rank correlations, autocorrelation at lag one and lag two. Dozens of
them agree between human and model. The frequency domain is orthogonal to all of
them, which is exactly the property that would let a deficit survive every test
run so far.

The fact this is chasing. `w4_seqstats` recorded that human speed carries a lag 2
autocorrelation ABOVE its lag 1, 0.6220 against 0.5952, while `event_ar_v1`
reads a flat 1.010 ratio. A monotonically decaying autocorrelation is what a
diffusion or a smooth drift produces. Human speed does not decay monotonically,
and a non monotone autocorrelation is the signature of an OSCILLATORY component
riding on the smooth motion. `w4_coupletok` reaches the same place from the other
side on `event_ar_v2`: turn alternation -0.3394 against -0.3929, speed lag 1
0.8064 against 0.7866, model smoother both times.

Why cross entropy would be blind to it. The objective is a sum of per step
surprises, so a component is worth what it contributes to predicting the NEXT
token. A small amplitude oscillation is nearly worthless by that measure. Its
contribution to any statistic computed over the WHOLE trajectory is a different
size, because a coherent component is phase locked and its contributions add
linearly across steps while incoherent noise of the same per step amplitude adds
in quadrature. Over sixty steps that is roughly a factor of eight between what
the training objective values and what the detector aggregates.

Run on the 125 Hz RESAMPLED path, not the event stream. The event stream has a
non uniform time base because every event carries its own dt, so its lag axis is
events rather than milliseconds and it cannot express a period in real time. The
resampled path is also what the contract's feature extractor actually sees.

Reading order.

  1. the human against human floor column FIRST. Two disjoint halves of real
     human data through the identical pipeline. Any ratio inside that band is
     the instrument's own noise and is not a finding.
  2. the STANDARDISED table, not the raw one. A deficit that shows up in raw
     power and vanishes once each row is z scored is an amplitude difference,
     which `w4_redundancy` already covers, and does not count here.
  3. narrow against broad. A deficit concentrated in a few adjacent bins is the
     coherent component this is looking for. A deficit spread evenly across the
     whole band is ordinary over or under dispersion and is NOT this mechanism.

DIAGNOSTIC ONLY. `scoring.py` is untouched and is not in this path. Nothing here
is a contract score. One trajectory per row, no selection, no reranking. Rows
come from the 2,528,855 trajectories the `default_rng(123)` training subset never
selected.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_spectrum.py \
        --ckpt event_ar_v2_s40000.pt --n 12000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

os.environ.setdefault("EVENT_CHOICE_TEMP", "10")
os.environ.setdefault("EVENT_SNAP", "2.5")
os.environ.setdefault("EVENT_DUR_STD", "1.0")
os.environ.setdefault("DUR_EMPIRICAL", "1")
os.environ.setdefault("EVENT_BESTOF", "1")
os.environ.setdefault("EVENT_SIR", "1")
os.environ.setdefault("EVENT_ORDER", "gumbel")
os.environ.setdefault("EVENT_STEPS", "100")
os.environ.setdefault("EVENT_CFG_W", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
from features import resample_trajectory  # noqa: E402
from models.event_ar import (  # noqa: E402
    DT_MAX_MS, EventARModel, dt_ms_to_class,
)
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TH_NULL_CLASS, dth_lattice_to_class, s2_to_class,
)

TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
MAX_T = 256
HZ = 125.0


def signals(path: np.ndarray) -> dict | None:
    """Speed and signed heading change of a trajectory, on the contract's own
    125 Hz resampled grid. Both arms go through the identical resampler, so the
    interpolation is not a difference between them.

    Signed turn is atan2 of a resampled displacement, which is noisy wherever
    that displacement is near zero. That is true on both arms identically and it
    is also what the feature extractor sees, so it is left in rather than
    cleaned, but it is the reason speed is the primary channel here.
    """
    rs = resample_trajectory([tuple(r) for r in path], hz=HZ)
    if len(rs) < 8:
        return None
    a = np.asarray(rs, dtype=np.float64)
    dx, dy = np.diff(a[:, 0]), np.diff(a[:, 1])
    sp = np.hypot(dx, dy) * HZ
    head = np.arctan2(dy, dx)
    st = np.diff(head)
    st = (st + np.pi) % (2.0 * np.pi) - np.pi
    return {"speed": sp, "turn": st}


def windows(x: np.ndarray, w: int) -> np.ndarray | None:
    """One centred window per trajectory, never several. Taking every window
    would let long trajectories contribute more of them and turn a duration
    difference between the arms into a spectral one."""
    if len(x) < w:
        return None
    o = (len(x) - w) // 2
    return x[o:o + w]


def psd(rows: list, w: int, standardise: bool) -> np.ndarray | None:
    """Mean periodogram over rows. Mean removed and a Hann window applied before
    the transform, because the velocity profile of a movement is a large low
    frequency component and its leakage would otherwise sit on top of the high
    band this is trying to read.

    standardise divides each row by its own standard deviation first, which
    removes amplitude and leaves only the shape of the spectrum."""
    win = np.hanning(w)
    norm = (win * win).sum()
    acc = []
    for x in rows:
        seg = windows(x, w)
        if seg is None:
            continue
        seg = seg - seg.mean()
        if standardise:
            sd = seg.std()
            if sd < 1e-9:
                continue
            seg = seg / sd
        acc.append(np.abs(np.fft.rfft(seg * win)) ** 2 / norm)
    if len(acc) < 50:
        return None
    return np.asarray(acc)


def acf(rows: list, lag: int) -> float:
    """Pooled autocorrelation at one lag, computed inside each row and averaged,
    so that between trajectory variation the cond vector already supplies does
    not enter."""
    v = []
    for x in rows:
        if len(x) < lag + 8:
            continue
        a, b = x[:-lag], x[lag:]
        a = a - a.mean()
        b = b - b.mean()
        d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
        if d > 0:
            v.append(float((a * b).sum() / d))
    return float(np.mean(v)) if len(v) >= 50 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--w", type=int, default=64,
                    help="window length in 125 Hz samples; sets the frequency "
                         "resolution at 125/w Hz per bin")
    ap.add_argument("--out", default="research/w4_spectrum.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    lengths = np.load("training/events_len.npy")
    N = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(N, min(N, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(N), trained)
    pick = np.sort(np.random.default_rng(args.seed)
                   .choice(held, args.n, replace=False))
    print(f"  corpus {N:,}, never seen {len(held):,}, drew {args.n:,}",
          flush=True)

    s2 = np.load("training/events_s2.npy", mmap_mode="r")[pick]
    dth = np.load("training/events_dth.npy", mmap_mode="r")[pick]
    dt_ms = np.load("training/events_dt.npy", mmap_mode="r")[pick].astype(np.float64)
    conds = np.load("training/events_cond.npy")[pick]
    L = np.minimum(lengths[pick], MAX_T).astype(np.int64)

    keep = L >= 12
    s2, dth, dt_ms, conds, L = (s2[keep], dth[keep], dt_ms[keep],
                                conds[keep], L[keep])
    B = len(L)
    print(f"  {B:,} rows at least 12 events, median length "
          f"{int(np.median(L))}\n", flush=True)

    real_s = np.full((B, MAX_T), S_PAD_CLASS, dtype=np.int64)
    real_th = np.full((B, MAX_T), TH_NULL_CLASS, dtype=np.int64)
    real_dt = np.zeros((B, MAX_T), dtype=np.float64)
    sc = s2_to_class(torch.from_numpy(np.asarray(s2, dtype=np.int64))).numpy()
    tc = np.where(np.asarray(s2) > 0,
                  dth_lattice_to_class(
                      torch.from_numpy(np.asarray(dth, dtype=np.int64))).numpy(),
                  TH_NULL_CLASS)
    for i in range(B):
        n = int(L[i])
        real_s[i, :n] = sc[i, :n]
        real_th[i, :n] = tc[i, :n]
        real_dt[i, :n] = dt_ms[i, :n]

    cond_t = torch.from_numpy(conds[:, :4].astype(np.float32))
    angs = np.arctan2(conds[:, 3].astype(np.float64),
                      conds[:, 2].astype(np.float64))

    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    print(f"  {args.ckpt} step {ck.get('step')} "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params\n",
          flush=True)

    print("  generating, free running exactly as served", flush=True)
    g_s = np.full((B, MAX_T), S_PAD_CLASS, dtype=np.int64)
    g_th = np.full((B, MAX_T), TH_NULL_CLASS, dtype=np.int64)
    g_dt = np.zeros((B, MAX_T), dtype=np.float64)
    for c0 in range(0, B, args.batch):
        sl = slice(c0, min(c0 + args.batch, B))
        s_o, th_o, dt_o = model.sample(cond_t[sl].to(dev),
                                       temperature=args.temp)
        w = s_o.shape[1]
        g_s[sl, :w] = s_o.cpu().numpy()
        g_th[sl, :w] = th_o.cpu().numpy()
        from models.event_ar import class_to_dt_ms
        g_dt[sl, :w] = class_to_dt_ms(dt_o.cpu()).numpy()

    def collect(s_arr, th_arr, dtms_arr):
        """Decode through the SERVING decoder, resample at 125 Hz, keep the two
        channels. Identical path on both arms."""
        out = {"speed": [], "turn": []}
        for i in range(B):
            dz = ((np.log(np.maximum(dtms_arr[i], 0.05)) - esp._DT_MEAN)
                  / esp._DT_STD)
            p = esp._decode(dz, s_arr[i], th_arr[i], 0.0, 0.0, float(angs[i]))
            if p is None or len(p) < 8:
                continue
            sg = signals(np.asarray(p, dtype=np.float64))
            if sg is None:
                continue
            out["speed"].append(sg["speed"])
            out["turn"].append(sg["turn"])
        return out

    hum = collect(real_s, real_th, real_dt)
    gen = collect(g_s, g_th, g_dt)
    nh, ng = len(hum["speed"]), len(gen["speed"])
    print(f"  {nh:,} human and {ng:,} generated rows decoded and resampled\n",
          flush=True)

    freqs = np.fft.rfftfreq(args.w, d=1.0 / HZ)
    out = {"ckpt": args.ckpt, "w": args.w, "hz": HZ, "n_human": nh,
           "n_model": ng, "diagnostic_only": True,
           "freqs_hz": freqs.tolist(), "channels": {}}

    print(f"  window {args.w} samples = {args.w / HZ * 1000:.0f} ms, "
          f"resolution {HZ / args.w:.2f} Hz per bin, one centred window per row")

    for ch in ("speed", "turn"):
        for std in (False, True):
            tag = "standardised" if std else "raw"
            h_all = psd(hum[ch], args.w, std)
            g_all = psd(gen[ch], args.w, std)
            if h_all is None or g_all is None:
                print(f"\n  {ch} {tag}: too few rows reach the window length")
                continue
            # human split in half is the instrument's own floor
            fa, fb = h_all[0::2].mean(0), h_all[1::2].mean(0)
            hm, gm = h_all.mean(0), g_all.mean(0)
            hse = h_all.std(0, ddof=1) / math.sqrt(len(h_all))
            gse = g_all.std(0, ddof=1) / math.sqrt(len(g_all))

            print(f"\n  {ch}, {tag}, {len(h_all):,} human windows "
                  f"{len(g_all):,} model windows")
            print(f"  {'freq Hz':>9}{'human':>12}{'model':>12}"
                  f"{'ratio':>9}{'floor':>9}{'se ratio':>10}")
            rows = []
            for k in range(1, len(freqs)):
                r = gm[k] / hm[k] if hm[k] > 0 else float("nan")
                fl = fa[k] / fb[k] if fb[k] > 0 else float("nan")
                # ratio's own uncertainty from the two arms' standard errors
                rse = abs(r) * math.sqrt((gse[k] / gm[k]) ** 2
                                         + (hse[k] / hm[k]) ** 2) \
                    if gm[k] > 0 and hm[k] > 0 else float("nan")
                rows.append({"hz": float(freqs[k]), "human": float(hm[k]),
                             "model": float(gm[k]), "ratio": float(r),
                             "floor": float(fl), "ratio_se": float(rse)})
                print(f"  {freqs[k]:>9.2f}{hm[k]:>12.4g}{gm[k]:>12.4g}"
                      f"{r:>9.3f}{fl:>9.3f}{rse:>10.4f}", flush=True)
            out["channels"][f"{ch}_{tag}"] = rows

    # the non monotone signature, restated on the 125 Hz time axis
    print("\n  autocorrelation of the resampled signal, time axis not events")
    print(f"  {'channel':<10}{'lag1':>10}{'lag2':>10}{'lag3':>10}{'r2/r1':>10}")
    out["acf"] = {}
    for ch in ("speed", "turn"):
        for lab, rows_ in (("human", hum[ch]), ("model", gen[ch])):
            a1, a2, a3 = acf(rows_, 1), acf(rows_, 2), acf(rows_, 3)
            ratio = a2 / a1 if abs(a1) > 1e-6 else float("nan")
            out["acf"][f"{ch}_{lab}"] = {"lag1": a1, "lag2": a2, "lag3": a3,
                                         "ratio21": ratio}
            print(f"  {ch + ' ' + lab:<10}{a1:>10.4f}{a2:>10.4f}{a3:>10.4f}"
                  f"{ratio:>10.4f}", flush=True)

    # The premise check. `w4_seqstats` found human speed lag 2 ABOVE lag 1 on
    # the EVENT INDEX, restricted to motion events. That series has two
    # differences from the resampled one above: a non uniform time base, and the
    # tick events spliced out. Splicing zeros out of a series distorts its lag
    # structure on its own, so the ratio is recomputed here both ways to find
    # out whether the non monotone signature is a real oscillation or an
    # artefact of the exclusion.
    print("\n  premise check, event index axis rather than time")
    print(f"  {'series':<24}{'lag1':>10}{'lag2':>10}{'lag3':>10}{'r2/r1':>10}")
    out["acf_event_axis"] = {}
    for lab, s_arr in (("human", real_s), ("model", g_s)):
        motion_only, all_events = [], []
        for i in range(B):
            s_i = s_arr[i]
            end = int(np.argmax(s_i >= S_PAD_CLASS)) if (s_i >= S_PAD_CLASS).any() \
                else len(s_i)
            if end < 16:
                continue
            from models.event_stream_polar import TICK_CLASS, class_to_speed
            sp = class_to_speed(torch.from_numpy(s_i[:end])).numpy()
            all_events.append(sp)
            m = s_i[:end] > TICK_CLASS
            if m.sum() >= 16:
                motion_only.append(sp[m])
        for tag, rows_ in (("motion events only", motion_only),
                           ("every event", all_events)):
            a1, a2, a3 = acf(rows_, 1), acf(rows_, 2), acf(rows_, 3)
            r = a2 / a1 if abs(a1) > 1e-6 else float("nan")
            out["acf_event_axis"][f"{lab}_{tag.replace(' ', '_')}"] = {
                "lag1": a1, "lag2": a2, "lag3": a3, "ratio21": r}
            print(f"  {lab + ' ' + tag:<24}{a1:>10.4f}{a2:>10.4f}{a3:>10.4f}"
                  f"{r:>10.4f}", flush=True)
    print("  a ratio above 1 is the non monotone signature. if it appears on")
    print("  motion events only and NOT on every event, the signature is the")
    print("  tick exclusion splicing the series, not an oscillation, and the")
    print("  premise of this run does not hold.")

    json.dump(out, open(args.out, "w"), indent=2)
    print("\n  DIAGNOSTIC ONLY, never a contract score. one trajectory per row,")
    print("  no selection, no reranking.")
    print("  read the floor column FIRST. it is two disjoint halves of real")
    print("  human data through the identical pipeline, so any ratio inside")
    print("  that band is the instrument's own noise and is not a finding.")
    print("  read the STANDARDISED table, not the raw one. a deficit that")
    print("  vanishes under standardisation is an amplitude difference and")
    print("  w4_redundancy already covers those.")
    print("  a deficit in a few adjacent bins is the coherent component this")
    print("  is looking for. a deficit spread evenly across the whole band is")
    print("  ordinary over or under dispersion and is NOT that mechanism.")


if __name__ == "__main__":
    main()
