"""Is the wrong spectral shape made by WHEN events fire or by WHAT SPEED they carry?

PRE REGISTERED in HANDOFF.md under "Is the wrong spectral shape made by WHEN the
events fire or by WHAT SPEED they carry, 2026-08-05". Criteria there are binding
and were fixed before this file existed.

The question. `w4_spectrum` found the model's standardised speed spectrum tilted
high, and the temperature sweep established that the contract reads the SHAPE of
that spectrum and tolerates a two times excess in its total energy. The record's
standing verdict is `w4_sharpness`'s "it has the wrong distribution", which
indicts the whole model and names no component.

A trajectory reaches the contract as a list of events, and each event carries two
separable things: WHEN it fires, its dt, and WHAT SPEED it carries. The contract
never sees events, it sees a uniform 125 Hz resampling of them, so event timing
is not merely part of the output, it is the clock deciding how much of each speed
value survives onto the measured grid. A long dt is interpolated across many grid
samples and contributes almost nothing above a few Hz. A short dt lands on
adjacent samples and contributes its full high frequency content. A model with
perfect speeds and wrong timing produces the wrong shape, and so does the
reverse. Every measurement in this repo mixes the two.

The arms. Same held out rows, same instrument as `w4_spectrum`, one centred
window per row.

    A   human dt exact       human speed    the real held out trajectory
    Aq  human dt quantised   human speed    A with dt rounded to whole ms
    C   model dt             model speed    free running, exactly as served
    E   model dt             human speed    s and th forced, model chooses dt
    D   model dt             human speed    offline recombination, no GPU

E is the decisive arm. See HANDOFF "SECOND AMENDMENT" for why the originally
registered arm B, human clock forced onto model speeds, is not well posed for
this checkpoint. Its emit order is `s_th_dt`: speed first, then direction
conditioned on speed, then dt conditioned on both. Forcing dt, the LAST channel,
substitutes a clock the speed was never conditioned on and breaks the within
step coupling at every event. Measured at n=800 it made things far worse than
free running, +0.52 against +0.20, which is the signature of a broken pairing
and not of a clean separation.

The only internally consistent forcing is a PREFIX of the emit order. Arm E
forces (s, th) and leaves dt to the model, so every quantity the dt head
conditions on is real human data and the head is measured on distribution and in
isolation. It also forces the speed stream's PAD, so arm E terminates exactly
where the human row does and its retention must match arm A.

The decomposition is then by elimination. E differs from C only in that its
speeds and directions are real. If E is clean, the dt head is innocent and the
speed head carries the tilt. If E keeps the tilt, the dt head carries it even
when everything it conditions on is real. The complementary arm, model speeds on
a forced human clock, is NOT available for this checkpoint at all, and that is
stated rather than faked.

Aq is NOT in the pre registration and is new. It exists because the dt alphabet
is whole milliseconds while recorded human dt looked finer. Measured, the round
trip error is 0.006 ms mean absolute, 0.7 percent of one 125 Hz sample, so the
confound does not exist. Aq is kept because it costs nothing and closes the
question, and no reading depends on it.

D is the UNCONDITIONED counterpart of E: the same human speeds on a model clock
that never saw them. E minus D is the value of conditioning. D remains a chimera
and is subordinate. It may not overturn E.

DIAGNOSTIC ONLY. `scoring.py` is untouched and is not in this path. Nothing here
is a contract score. One trajectory per row, no selection, no reranking. Rows
come from the 2,528,855 trajectories the `default_rng(123)` training subset never
selected.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_timing.py \
        --ckpt event_ar_v2_s40000.pt --n 20000
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
    EventARModel, class_to_dt_ms, dt_ms_to_class, prefix_state,
)
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TH_NULL_CLASS, dth_lattice_to_class, s2_to_class,
)

TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
MAX_T = 256
HZ = 125.0

# The pre registered band and statistic. Fixed before any number existed.
BAND_LO_HZ = 11.0
BAND_HI_HZ = 41.5
# AMENDED thresholds, see HANDOFF "AMENDMENT, made before the GPU arms ran".
# The original absolute thresholds were tied to a statistic that failed its own
# self test. The INTENT they encoded, two thirds removed and one quarter
# removed, is preserved and now expressed as a fraction of arm C's own excess,
# which is what "how much of the defect did B remove" always meant.
FRAC_TIMING_MAX = 1.0 / 3.0   # B keeps less than a third, timing dominant
FRAC_VALUES_MIN = 3.0 / 4.0   # B keeps more than three quarters, values dominant
RETENTION_GATE = 0.20

# Arm G decision rule, registered here BEFORE the arm was ever run, expressed as
# a fraction of arm E's excess so it cannot be retuned once the number is in.
# Arm G asks the same conditional question as arm E, p(dt | real s, real th),
# but with a fully real dt history instead of the model's own. The gap between
# them is exactly the autoregressive drift.
G_HEAD_MIN = 0.50    # G keeps at least half of E, the dt conditional is tilted
G_DRIFT_MAX = 0.25   # G keeps under a quarter of E, the excess was drift    # relative retention difference that invalidates B


def signals(path: np.ndarray) -> dict | None:
    """Speed and signed heading change on the contract's own 125 Hz grid.
    Identical resampler on every arm, so interpolation is never a difference
    between them."""
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
    """Mean periodogram over rows, mean removed and Hann windowed. standardise
    divides each row by its own standard deviation first, which removes
    amplitude and leaves only the shape."""
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


def null_sd(ref_psd: np.ndarray, n_arm: int, freqs: np.ndarray,
            draws: int = 400, seed: int = 11) -> float:
    """Standard deviation of the band statistic under the null that the arm is
    drawn from the same population as the reference.

    Built by resampling the REFERENCE arm twice with replacement, once at the
    arm's own sample size and once at the reference's, and taking the band mean
    ratio. That is the null for exactly the comparison being made, including its
    dependence on both sample sizes.

    This replaces the split half floor, which the pre registration used as a
    SUBTRACTED OFFSET and which cannot serve as one. A single odd or even split
    is one draw from a distribution with a standard deviation of 0.03 to 0.05 at
    these sample sizes, so subtracting it injects that entire error into the
    statistic. Demonstrated by the reference arm failing its own self test:
    compared against itself the ratio is exactly 1 by construction, so the
    statistic must be 0, and with the floor subtracted it read +0.117. The floor
    is an error bar and is now used as one."""
    sel = (freqs >= BAND_LO_HZ) & (freqs <= BAND_HI_HZ)
    rng = np.random.default_rng(seed)
    n_ref, vals = len(ref_psd), []
    for _ in range(draws):
        a = ref_psd[rng.integers(0, n_ref, n_arm)].mean(0)
        b = ref_psd[rng.integers(0, n_ref, n_ref)].mean(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            vals.append(float((a[sel] / b[sel]).mean()))
    return float(np.std(vals, ddof=1))


def band_stat(arm_psd: np.ndarray, ref_psd: np.ndarray,
              freqs: np.ndarray) -> dict:
    """E, the AMENDED statistic: mean standardised power ratio over the band,
    minus one. The null is 1.0 by construction, so a self comparison gives
    exactly 0 and the statistic passes its own self test. Uncertainty comes
    from `null_sd`, not from an offset."""
    sel = (freqs >= BAND_LO_HZ) & (freqs <= BAND_HI_HZ)
    rm, am = ref_psd.mean(0), arm_psd.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = am / rm
    sd = null_sd(ref_psd, len(arm_psd), freqs)
    e = float(ratio[sel].mean() - 1.0)
    return {
        "mean_ratio": float(ratio[sel].mean()),
        "E": e,
        "null_sd": sd,
        "sigma": e / sd if sd > 0 else float("nan"),
        "n_bins": int(sel.sum()),
        "n_windows": int(len(arm_psd)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--w", type=int, default=64)
    ap.add_argument("--no-gpu-arms", action="store_true",
                    help="run only A, Aq and the token side statistics, which "
                         "need no model at all")
    ap.add_argument("--control-only", action="store_true",
                    help="run ONLY the arm F seam control against A and Aq. "
                         "One generation pass rather than three, which is a "
                         "third of the thermal load. Use when C and E are "
                         "already measured and only the control is missing.")
    ap.add_argument("--drift-only", action="store_true",
                    help="skip the arm F generation pass and run ONLY arm G, "
                         "the drift control. Arm G is a single teacher forced "
                         "forward pass rather than 256 sequential steps, so "
                         "this is the cheapest and coolest mode there is. Use "
                         "it once arm F has already passed, since regenerating "
                         "a control that passed buys nothing and costs the "
                         "sustained load that got a previous run killed.")
    ap.add_argument("--e-ref", type=float, default=None,
                    help="arm E's excess against Aq, for scoring arm G when "
                         "arm E is not itself in this run. Run 1 at n=20000 "
                         "measured +0.3864. Only ever pass a number this repo "
                         "has already recorded.")
    ap.add_argument("--out", default="research/w4_timing.json")
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

    # The dt alphabet is whole milliseconds. Round tripping the human clock
    # through it is exactly what the model is able to express and no more.
    real_dt_cls = dt_ms_to_class(torch.from_numpy(real_dt)).numpy()
    quant_dt = class_to_dt_ms(torch.from_numpy(real_dt_cls)).numpy().astype(np.float64)
    quant_dt[real_dt == 0.0] = 0.0     # padding stays padding
    live = real_dt > 0
    err = (quant_dt - real_dt)[live]
    print("  dt quantisation, human clock through the model's whole ms alphabet")
    print(f"    events {live.sum():,}  mean |error| {np.abs(err).mean():.4f} ms"
          f"  rms {np.sqrt((err ** 2).mean()):.4f} ms")
    print(f"    that is {np.sqrt((err ** 2).mean()) / (1000.0 / HZ) * 100:.1f}% "
          f"of one 125 Hz sample, which is {1000.0 / HZ:.1f} ms wide")
    print(f"    human dt: min {real_dt[live].min():.3f} median "
          f"{np.median(real_dt[live]):.3f} max {real_dt[live].max():.3f} ms\n",
          flush=True)

    cond_t = torch.from_numpy(conds[:, :4].astype(np.float32))
    angs = np.arctan2(conds[:, 3].astype(np.float64),
                      conds[:, 2].astype(np.float64))

    arms: dict[str, tuple] = {
        "A_human_exact_dt": (real_s, real_th, real_dt),
        "Aq_human_quantised_dt": (real_s, real_th, quant_dt),
    }

    if not args.no_gpu_arms:
        ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                        weights_only=False)
        model = EventARModel(**ck["config"]).to(dev).eval()
        model.load_state_dict(ck["model_state_dict"])
        print(f"  {args.ckpt} step {ck.get('step')} "
              f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params\n",
              flush=True)

        f_s_t = torch.from_numpy(real_s)
        f_th_t = torch.from_numpy(real_th)
        f_dt_t = torch.from_numpy(real_dt_cls)

        # This checkpoint emits s, then th CONDITIONED ON s, then dt
        # CONDITIONED ON s AND th. The only forcing that keeps a step internally
        # consistent is therefore a PREFIX of that order. Forcing dt, the last
        # channel, would substitute a clock the speed was never conditioned on
        # and would break the within step coupling at every event. Arm E forces
        # the prefix (s, th) and leaves dt to the model, so every quantity the
        # dt head conditions on is real human data and the head is measured on
        # distribution and in isolation.
        pos = np.arange(MAX_T)[None, :]

        def run(channels: tuple):
            """channels lists which of (s, th, dt) are forced from the human."""
            o_s = np.full((B, MAX_T), S_PAD_CLASS, dtype=np.int64)
            o_th = np.full((B, MAX_T), TH_NULL_CLASS, dtype=np.int64)
            o_dt = np.zeros((B, MAX_T), dtype=np.float64)
            for c0 in range(0, B, args.batch):
                sl = slice(c0, min(c0 + args.batch, B))
                force = None
                if channels:
                    nb = f_s_t[sl].shape[0]
                    m = torch.zeros(nb, MAX_T, 3, dtype=torch.bool)
                    # every position including the PAD, so the row terminates
                    # exactly where the human row does and retention matches A
                    for c in channels:
                        m[:, :, c] = True
                    force = (f_s_t[sl].to(dev), f_th_t[sl].to(dev),
                             f_dt_t[sl].to(dev), m.to(dev))
                s_o, th_o, dt_o = model.sample(cond_t[sl].to(dev),
                                               temperature=args.temp,
                                               force=force)
                w = s_o.shape[1]
                o_s[sl, :w] = s_o.cpu().numpy()
                o_th[sl, :w] = th_o.cpu().numpy()
                o_dt[sl, :w] = class_to_dt_ms(dt_o.cpu()).numpy()
            return o_s, o_th, o_dt

        # Arm F, the seam control, runs FIRST because nothing downstream is
        # worth reading if it fails. Forcing all three channels routes real
        # human tokens through the sampler and the serving decoder and must
        # reproduce arm Aq exactly. If it does not, the forcing machinery or the
        # decode path is introducing the effect and arm E means nothing. This is
        # the check the repo rule requires before a generated artefact is read
        # as a model property, and it is run on the real seam rather than a
        # stub.
        if not args.drift_only:
            print("  arm F, ALL THREE channels forced, the seam control",
                  flush=True)
            f_s, f_th, f_dt = run((0, 1, 2))
            arms["F_control_all_forced"] = (f_s, f_th, f_dt)

        # Arm G, the DRIFT control, and the one that decides whether arm E is
        # readable. Arm F cannot answer this: with all three channels forced it
        # has no free channel, so it proves the plumbing and the decode path are
        # exact and nothing more. The live risk in arm E is different. There the
        # model conditions on real human s and th but on ITS OWN previously
        # generated dt, a hybrid history it never saw in training, so arm E's
        # excess could be exposure drift rather than a defective dt head.
        #
        # Arm G removes drift entirely. One fully teacher forced forward pass
        # over the real sequences, dt sampled ONCE per position from
        # p(dt_i | real history, real s_i, th_i). Every quantity conditioned on
        # is real, including the dt history. No autoregression, so no drift can
        # accumulate. It is also a single forward pass rather than 256
        # sequential steps, which is why it is thermally cheap.
        #
        #   G tilted -> the dt conditional is genuinely wrong, drift excluded
        #   G clean  -> arm E's excess is drift and the dt head reading falls
        print("  arm G, one step dt resample under full teacher forcing, the "
              "drift control", flush=True)
        g_dt_cls = np.zeros((B, MAX_T), dtype=np.int64)
        with torch.no_grad():
            for c0 in range(0, B, args.batch):
                sl = slice(c0, min(c0 + args.batch, B))
                s_b = f_s_t[sl].to(dev)
                th_b = f_th_t[sl].to(dev)
                dt_b = f_dt_t[sl].to(dev)
                cnd = cond_t[sl].to(dev)
                s_p, th_p, dt_p = model.shift_inputs(s_b, th_b, dt_b)
                st = prefix_state(s_b, th_b, dt_b, cnd)
                x = model.trunk(s_p, th_p, dt_p, st, cnd)
                logits = model.dt_logits(x, s_b, th_b)
                p = torch.softmax(logits / args.temp, dim=-1)
                flat = p.reshape(-1, p.shape[-1])
                samp = torch.multinomial(flat, 1).squeeze(-1)
                g_dt_cls[sl] = samp.view(p.shape[0], p.shape[1]).cpu().numpy()
        g_dt_ms = class_to_dt_ms(torch.from_numpy(g_dt_cls)).numpy().astype(np.float64)
        g_dt_ms[real_dt == 0.0] = 0.0     # padding stays padding
        arms["G_teacherforced_dt"] = (real_s, real_th, g_dt_ms)

        if not (args.control_only or args.drift_only):
            print("  arm C, free running exactly as served", flush=True)
            c_s, c_th, c_dt = run(())
            print("  arm E, human speeds and directions forced, model choosing "
                  "its own clock conditioned on them", flush=True)
            e_s, e_th, e_dt = run((0, 1))
            arms["C_model_free"] = (c_s, c_th, c_dt)
            arms["E_humanspeed_modeldt"] = (e_s, e_th, e_dt)

        # D, offline. Human tokens on the model's clock. Only rows where the
        # model produced at least as many events as the human row needs, so no
        # human sequence is truncated to fit.
            c_len = np.where((c_s >= S_PAD_CLASS).any(1),
                             np.argmax(c_s >= S_PAD_CLASS, axis=1), MAX_T)
            ok_d = c_len >= L
            d_dt = np.zeros((B, MAX_T), dtype=np.float64)
            d_dt[ok_d] = c_dt[ok_d]
            for i in np.flatnonzero(ok_d):
                d_dt[i, int(L[i]):] = 0.0
            d_s = np.where(ok_d[:, None], real_s, S_PAD_CLASS)
            d_th = np.where(ok_d[:, None], real_th, TH_NULL_CLASS)
            arms["D_modeldt_humanspeed"] = (d_s, d_th, d_dt)
            print(f"  arm D, {ok_d.sum():,} of {B:,} rows usable "
                  f"({ok_d.mean() * 100:.1f}%), model row long enough to carry "
                  f"the human sequence\n", flush=True)

    def collect(s_arr, th_arr, dtms_arr):
        """Decode through the SERVING decoder, resample at 125 Hz. Identical
        path on every arm."""
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

    freqs = np.fft.rfftfreq(args.w, d=1.0 / HZ)
    out = {"ckpt": args.ckpt, "w": args.w, "hz": HZ, "n_rows": int(B),
           "temp": args.temp, "diagnostic_only": True,
           "pre_registered": "HANDOFF.md 2026-08-05",
           "band_hz": [BAND_LO_HZ, BAND_HI_HZ],
           "freqs_hz": freqs.tolist(), "arms": {}, "retention": {}}

    decoded, psds = {}, {}
    for name, (sa, ta, da) in arms.items():
        d = collect(sa, ta, da)
        decoded[name] = d
        n_win = sum(1 for x in d["speed"] if len(x) >= args.w)
        out["retention"][name] = {"decoded": len(d["speed"]),
                                  "windowed": n_win,
                                  "rate": n_win / B}
        print(f"  {name:<24} decoded {len(d['speed']):>6,}  "
              f"windows {n_win:>6,}  retention {n_win / B * 100:5.1f}%",
              flush=True)
        psds[name] = {(ch, std): psd(d[ch], args.w, std)
                      for ch in ("speed", "turn") for std in (False, True)}

    # VALIDITY GATE, checked before any ratio is read.
    print()
    gate_ok = True
    if "F_control_all_forced" in arms:
        # The seam control, read before anything else. Real human tokens routed
        # through the sampler and the serving decoder must come back as arm Aq.
        e_f = None
        rp_ = psds["Aq_human_quantised_dt"][("speed", True)]
        if rp_ is not None and psds["F_control_all_forced"][("speed", True)] is not None:
            st_f = band_stat(psds["F_control_all_forced"][("speed", True)],
                             rp_, freqs)
            e_f = st_f["E"]
            seam_ok = abs(st_f["sigma"]) < 3.0
            print(f"  SEAM CONTROL   arm F, all three channels forced, against "
                  f"Aq: E {e_f:+.4f} ({st_f['sigma']:+.1f} sd)")
            print(f"  must be indistinguishable from zero, "
                  f"{'PASS' if seam_ok else 'FAIL, arm E is not readable'}")
            out["seam_control"] = {**st_f, "pass": bool(seam_ok)}
            gate_ok = gate_ok and seam_ok
    if "G_teacherforced_dt" in arms:
        # The DRIFT control. Arm F proves the plumbing is exact but cannot speak
        # to this, because forcing all three channels leaves no free channel to
        # disturb. Arm G leaves dt free while making everything it conditions on
        # real, including the dt history, so it is arm E with the drift removed.
        rp_ = psds["Aq_human_quantised_dt"][("speed", True)]
        if rp_ is not None and psds["G_teacherforced_dt"][("speed", True)] is not None:
            st_g = band_stat(psds["G_teacherforced_dt"][("speed", True)],
                             rp_, freqs)
            rg = out["retention"]["G_teacherforced_dt"]["rate"]
            ra = out["retention"]["A_human_exact_dt"]["rate"]
            print(f"  DRIFT CONTROL  arm G, one step dt resample under full "
                  f"teacher forcing, against Aq: E {st_g['E']:+.4f} "
                  f"({st_g['sigma']:+.1f} sd)")
            print(f"                 retention {rg * 100:.1f}% against arm A "
                  f"{ra * 100:.1f}%")
            out["drift_control"] = st_g
            # Arm E's excess, either measured in this same run or supplied from
            # the run that measured it. The rule below was registered above.
            e_e = None
            if "E_humanspeed_modeldt" in psds and \
                    psds["E_humanspeed_modeldt"][("speed", True)] is not None:
                e_e = band_stat(psds["E_humanspeed_modeldt"][("speed", True)],
                                rp_, freqs)["E"]
                src = "measured in this run"
            elif args.e_ref is not None:
                e_e, src = args.e_ref, "supplied via --e-ref"
            if e_e is not None and e_e > 0:
                frac = st_g["E"] / e_e
                if frac >= G_HEAD_MIN:
                    verdict = ("DT CONDITIONAL is genuinely tilted, drift "
                               "excluded, arm E's reading STANDS")
                elif frac <= G_DRIFT_MAX:
                    verdict = ("arm E's excess is AUTOREGRESSIVE DRIFT, the "
                               "dt head reading FALLS")
                else:
                    verdict = ("MIXED, neither the head nor drift alone "
                               "accounts for arm E, no clean reading")
                print(f"                 arm E is {e_e:+.4f} ({src}), G keeps "
                      f"{frac * 100:.1f}% of it")
                print(f"                 -> {verdict}")
                out["drift_control"].update(
                    {"E_arm_E": e_e, "E_source": src, "frac_of_E": frac,
                     "verdict": verdict})
    if "E_humanspeed_modeldt" in arms:
        # Arm E forces the speed stream including its PAD, so it terminates
        # exactly where the human row does and its retention must match arm A,
        # not arm C. A mismatch here means the forcing did not take.
        re_ = out["retention"]["E_humanspeed_modeldt"]["rate"]
        ra = out["retention"]["A_human_exact_dt"]["rate"]
        rel = abs(re_ - ra) / max(ra, 1e-9)
        gate_ok = rel <= RETENTION_GATE
        print(f"  VALIDITY GATE  arm E retention {re_ * 100:.1f}% against arm A "
              f"{ra * 100:.1f}%, relative difference {rel * 100:.1f}%")
        print(f"  gate is {RETENTION_GATE * 100:.0f}%, "
              f"{'PASS' if gate_ok else 'FAIL, no verdict is reported'}")
        out["validity_gate"] = {"rate_E": re_, "rate_A": ra, "rel_diff": rel,
                                "pass": bool(gate_ok)}

    # The registered statistic, every arm against both references.
    print("\n  E = mean standardised speed power ratio over "
          f"{BAND_LO_HZ:.0f} to {BAND_HI_HZ:.0f} Hz, minus one. null sd is a "
          "400 draw bootstrap of the reference against itself at the same two "
          "sample sizes. a self comparison must read E exactly 0.")
    for ref in ("A_human_exact_dt", "Aq_human_quantised_dt"):
        rp = psds[ref][("speed", True)]
        if rp is None:
            continue
        print(f"\n  against {ref}")
        print(f"  {'arm':<24}{'mean ratio':>12}{'E':>9}{'null sd':>10}"
              f"{'sigma':>8}{'windows':>10}")
        out["arms"].setdefault(ref, {})
        for name in arms:
            ap_ = psds[name][("speed", True)]
            if ap_ is None:
                continue
            st = band_stat(ap_, rp, freqs)
            out["arms"][ref][name] = st
            print(f"  {name:<24}{st['mean_ratio']:>12.4f}{st['E']:>+9.4f}"
                  f"{st['null_sd']:>10.4f}{st['sigma']:>8.1f}"
                  f"{st['n_windows']:>10,}")

    # Per bin detail for the speed channel, standardised, against Aq.
    ref = "Aq_human_quantised_dt"
    rp = psds[ref][("speed", True)]
    if rp is not None and len(arms) > 2:
        rm = rp.mean(0)
        print(f"\n  per bin standardised speed power, ratio against {ref}")
        hdr = "".join(f"{n.split('_')[0]:>10}" for n in arms)
        print(f"  {'freq Hz':>9}{hdr}")
        for k in range(1, len(freqs)):
            cells = ""
            for name in arms:
                ap_ = psds[name][("speed", True)]
                cells += (f"{ap_.mean(0)[k] / rm[k]:>10.3f}"
                          if ap_ is not None and rm[k] > 0 else f"{'nan':>10}")
            print(f"  {freqs[k]:>9.2f}{cells}", flush=True)

    # VERDICT, against the pre registered thresholds. Read against Aq, which is
    # the quantisation matched reference, and reported against A as registered.
    print()
    if "E_humanspeed_modeldt" in arms and gate_ok:
        # Elimination. Arm E differs from arm C only in that its speeds and
        # directions are real. If E is clean the dt head is innocent and the
        # speed head carries the tilt. If E keeps the tilt, the dt head carries
        # it even when everything it conditions on is real human data.
        for ref in ("A_human_exact_dt", "Aq_human_quantised_dt"):
            e_e = out["arms"][ref]["E_humanspeed_modeldt"]["E"]
            e_c = out["arms"][ref]["C_model_free"]["E"]
            sig = out["arms"][ref]["E_humanspeed_modeldt"]["sigma"]
            kept = e_e / e_c if abs(e_c) > 1e-9 else float("nan")
            if kept < FRAC_TIMING_MAX:
                v = "SPEED HEAD carries it, dt head innocent"
            elif kept > FRAC_VALUES_MIN:
                v = "DT HEAD carries it"
            else:
                v = "SPLIT, both heads carry it"
            if kept > 1.0:
                v += ", AND E EXCEEDS C"
            print(f"  vs {ref:<24} E_E {e_e:+.4f} ({sig:+.1f} sd)  "
                  f"E_C {e_c:+.4f}  E keeps {kept * 100:5.1f}% of C   -> {v}")
            if kept > 1.0:
                print("    E above C is OUTSIDE the range the registered "
                      "thresholds assumed, which was 0 to E_C. Handing the "
                      "model real speeds made the tilt WORSE than leaving it "
                      "alone. The dt head reading stands but the fraction is "
                      "not a share of a fixed budget and must not be quoted as "
                      "one. Compare against arm D, the same speeds on a clock "
                      "that never saw them, to price what conditioning buys.")
            out.setdefault("verdict", {})[ref] = {
                "E_E": e_e, "E_C": e_c, "fraction_kept": kept,
                "sigma_E": sig, "verdict": v}
        if out["arms"]["Aq_human_quantised_dt"]["E_humanspeed_modeldt"]["E"] < -0.05:
            print("  SANITY CHECK FAILED: real human speeds through the model's "
                  "own clock produced materially LESS high frequency energy "
                  "than a human. The forcing is broken and no verdict holds.")

    json.dump(out, open(args.out, "w"), indent=2)
    print("\n  DIAGNOSTIC ONLY, never a contract score. one trajectory per row,")
    print("  no selection, no reranking. no serving change follows from this.")
    print("  read the VALIDITY GATE first. arm E forces the speed stream's PAD")
    print("  so it ends where the human row ends, and its retention must match")
    print("  arm A. a mismatch means the forcing did not take.")
    print("  arm E is the decisive arm and the reading is by elimination. E")
    print("  differs from C only in that its speeds and directions are real, so")
    print("  a clean E puts the tilt in the speed head and a tilted E puts it in")
    print("  the dt head. the complementary arm, model speeds on a forced human")
    print("  clock, is NOT available for an s_th_dt checkpoint. see HANDOFF.")
    print("  Aq minus A is the price of the whole ms dt alphabet and measures")
    print("  as nothing. that confound does not exist.")
    print("  arm D is E without the conditioning. E minus D is what conditioning")
    print("  buys. D is a chimera, is subordinate, and may not overturn E.")


if __name__ == "__main__":
    main()
