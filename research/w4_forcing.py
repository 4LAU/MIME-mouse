"""Does `w4_timing` arm E's forcing carry an estimator artefact too?

PRE REGISTERED in HANDOFF, thresholds fixed before this file existed:

    VALIDITY           |E(SC vs SA)| within 2 null sd of zero, or NO verdict
    artefact >= 0.25   arm E's +0.3864 is dominated by construction, the
                       "real speeds make it worse" reading FALLS
    artefact <= 0.08   construction is minor, arm E's reading STANDS
    in between         subtract, report both numbers and the residual
    BOUNDARY           within one null sd of either threshold, the call is
                       REFUSED and the in between case is reported instead

Why this run exists. `w4_artefact` established that the arm G one step resample
manufactures its own spectral signature on data with no defect in it, and that
the shape match, not the scalar, is what settled it. The rule that followed is
that no generated artefact may be read as a model property until the identical
pipeline has run on data whose truth is known by construction. Arm E has never
had that check. Arm F checked the plumbing, which is a different thing.

Why arm E is suspect. It forces human s and th at every position while the model
supplies its own dt. In the model's generative order dt_i influences s_i+1 and
th_i+1. Under forcing those come from the human and were produced alongside the
HUMAN's dt_i, not the model's, so every position after the first sits in a
context the joint never produces: real speeds paired with a clock that did not
generate them. That is the same class of mismatch that produced arm G's entire
signature.

The control.

    arm SA   model generated free running, the synthetic reference. On these rows
             the model IS the true generator, exactly, by construction.
    arm SE   the SAME rows' s and th forced back in, model supplying its own dt,
             exactly as arm E does to human rows
    arm SC   a second independent free running generation, the NULL

Nothing here contains a model defect, so SE against SA is pure construction and
SC against SA is the pipeline's own noise floor.

The per bin shape is read too and is not subordinate. If SE reproduces arm E's
monotone rotation climbing to roughly 2.4 at 62.5 Hz, arm E's shape is the
estimator's shape and the scalar hardly matters.

DIAGNOSTIC ONLY, never a contract score. One trajectory per row, no selection, no
reranking. No serving change follows and no build is authorised either way. Phase
conditioning and the spectral loss term remain NOT AUTHORISED.

Safety. Reads `training/events_*.npy` and one checkpoint. Touches no evaluation
data, no scoring code, and never `training/candi_polar_flow_best.pt`. Checks the
GPU between generation passes and waits rather than running them back to back;
the run that was killed at 84C did three passes without pausing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set BEFORE experiments.event_stream_polar is imported, because it reads these
# at import time. w4_timing sets the same three; importing it later cannot undo
# an esp that has already read the environment.
os.environ.setdefault("EVENT_STEPS", "100")
os.environ.setdefault("EVENT_CFG_W", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import experiments.event_stream_polar as esp  # noqa: E402
from models.event_ar import EventARModel, class_to_dt_ms  # noqa: E402
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TH_NULL_CLASS,
)
from research.w4_timing import (  # noqa: E402
    BAND_HI_HZ, BAND_LO_HZ, HZ, MAX_T, N_TRAIN_DEFAULT, TRAIN_PICK_SEED,
    band_stat, psd, signals,
)

ARTEFACT_DOMINATES = 0.25
ARTEFACT_MINOR = 0.08
ARM_E_EXCESS = 0.3864          # w4_timing arm E against Aq, n=20000
ARM_C_EXCESS = 0.1825          # w4_timing arm C against Aq, n=20000
LAUNCH_GATE_C = 75


def gpu_temp():
    try:
        o = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        return int(o.stdout.strip().split("\n")[0])
    except Exception:
        return None


def cooldown(label):
    """Wait for the GPU to fall back under the launch gate before the next
    generation pass. The run that was killed at 84C ran three passes back to
    back; this is the fix for that, in the script rather than in my memory."""
    t = gpu_temp()
    if t is None:
        print(f"  [{label}] no GPU reading, not gating", flush=True)
        return
    if t < LAUNCH_GATE_C:
        print(f"  [{label}] GPU {t}C, under the {LAUNCH_GATE_C}C gate, "
              f"continuing", flush=True)
        return
    print(f"  [{label}] GPU {t}C, waiting for it to fall under "
          f"{LAUNCH_GATE_C}C", flush=True)
    while True:
        time.sleep(20)
        t = gpu_temp()
        if t is None or t < LAUNCH_GATE_C:
            print(f"  [{label}] GPU {t}C, continuing", flush=True)
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--w", type=int, default=64)
    ap.add_argument("--no-null", action="store_true",
                    help="skip arm SC, the second independent generation. Saves "
                         "one generation pass. The validity gate then cannot be "
                         "evaluated and the script says so rather than passing "
                         "it silently.")
    ap.add_argument("--out", default="research/w4_forcing.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    lengths = np.load("training/events_len.npy")
    N = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(N, min(N, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(N), trained)
    pick = np.sort(np.random.default_rng(args.seed)
                   .choice(held, args.n, replace=False))
    conds = np.load("training/events_cond.npy")[pick]
    L = np.minimum(lengths[pick], MAX_T).astype(np.int64)
    conds, L = conds[L >= 12], L[L >= 12]
    B = len(L)
    print(f"  corpus {N:,}, never seen {len(held):,}, drew {args.n:,}")
    print(f"  {B:,} conditioning vectors, the same rows w4_timing used\n",
          flush=True)

    angs = np.arctan2(conds[:, 3], conds[:, 2])
    cond_t = torch.from_numpy(conds[:, :4].astype(np.float32))

    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    print(f"  {args.ckpt} step {ck.get('step')} "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params\n",
          flush=True)

    def generate(force_streams=None):
        """One generation pass. force_streams, when given, is (s, th) class
        arrays whose values are forced at every position while dt stays free,
        which is exactly what arm E does."""
        o_s = np.full((B, MAX_T), S_PAD_CLASS, dtype=np.int64)
        o_th = np.full((B, MAX_T), TH_NULL_CLASS, dtype=np.int64)
        # ms, not classes, and filled by class_to_dt_ms exactly as w4_timing's
        # run() does. Do not zero the PAD tail here: w4_timing's generated arms
        # do not, and the whole point of this control is that the pipeline is
        # identical on both sides.
        o_dt = np.zeros((B, MAX_T), dtype=np.float64)
        f_s = f_th = None
        if force_streams is not None:
            f_s = torch.from_numpy(force_streams[0])
            f_th = torch.from_numpy(force_streams[1])
        with torch.no_grad():
            for c0 in range(0, B, args.batch):
                sl = slice(c0, min(c0 + args.batch, B))
                force = None
                if force_streams is not None:
                    nb = f_s[sl].shape[0]
                    m = torch.zeros(nb, MAX_T, 3, dtype=torch.bool)
                    m[:, :, 0] = True      # s forced
                    m[:, :, 1] = True      # th forced
                    force = (f_s[sl].to(dev), f_th[sl].to(dev),
                             torch.zeros(nb, MAX_T, dtype=torch.long).to(dev),
                             m.to(dev))
                s_o, th_o, dt_o = model.sample(cond_t[sl].to(dev),
                                               temperature=args.temp,
                                               force=force)
                w = s_o.shape[1]
                o_s[sl, :w] = s_o.cpu().numpy()
                o_th[sl, :w] = th_o.cpu().numpy()
                o_dt[sl, :w] = class_to_dt_ms(dt_o.cpu()).numpy()
        return o_s, o_th, o_dt

    arms = {}
    torch.manual_seed(args.seed + 17)
    print("  arm SA, model generated free running, the synthetic reference",
          flush=True)
    arms["SA_reference"] = generate()

    cooldown("after SA")
    print("  arm SE, the SAME rows' s and th forced back in, model supplying "
          "its own dt", flush=True)
    sa_s, sa_th, _ = arms["SA_reference"]
    arms["SE_forced_sth"] = generate((sa_s, sa_th))

    if not args.no_null:
        cooldown("after SE")
        print("  arm SC, a second independent free running generation, the NULL",
              flush=True)
        torch.manual_seed(args.seed + 991)
        arms["SC_null_second_draw"] = generate()

    def collect(s_arr, th_arr, dtms_arr):
        out = {"speed": []}
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
        return out

    freqs = np.fft.rfftfreq(args.w, d=1.0 / HZ)
    out = {"ckpt": args.ckpt, "w": args.w, "hz": HZ, "n_rows": int(B),
           "temp": args.temp, "diagnostic_only": True,
           "pre_registered": "HANDOFF.md 2026-08-05",
           "band_hz": [BAND_LO_HZ, BAND_HI_HZ], "freqs_hz": freqs.tolist(),
           "arm_E_excess_being_priced": ARM_E_EXCESS,
           "arm_C_excess": ARM_C_EXCESS,
           "thresholds": {"dominates": ARTEFACT_DOMINATES,
                          "minor": ARTEFACT_MINOR},
           "retention": {}}

    print()
    psds = {}
    for name, (sa, ta, da) in arms.items():
        d = collect(sa, ta, da)
        n_win = sum(1 for x in d["speed"] if len(x) >= args.w)
        out["retention"][name] = {"decoded": len(d["speed"]),
                                  "windowed": n_win, "rate": n_win / B}
        print(f"  {name:<24} decoded {len(d['speed']):>6,}  "
              f"windows {n_win:>6,}  retention {n_win / B * 100:5.1f}%",
              flush=True)
        psds[name] = psd(d["speed"], args.w, True)

    ref = psds["SA_reference"]
    print(f"\n  SELF CHECK   arm SA against itself: "
          f"E {band_stat(ref, ref, freqs)['E']:+.4f}, must be exactly 0")

    gate_ok, null_stat = None, None
    if "SC_null_second_draw" in psds:
        null_stat = band_stat(psds["SC_null_second_draw"], ref, freqs)
        gate_ok = abs(null_stat["sigma"]) <= 2.0
        out["validity_gate"] = {**null_stat, "pass": bool(gate_ok)}
        print(f"  VALIDITY     arm SC, a second independent draw, against SA: "
              f"E {null_stat['E']:+.4f} ({null_stat['sigma']:+.1f} sd)")
        print(f"               must be within 2 sd of zero, "
              f"{'PASS' if gate_ok else 'FAIL, no verdict is reported'}")
    else:
        print("  VALIDITY     arm SC was skipped, so the gate CANNOT be "
              "evaluated. The verdict below is reported without it.")
        out["validity_gate"] = {"evaluated": False}

    stat = band_stat(psds["SE_forced_sth"], ref, freqs)
    out["artefact"] = stat
    print(f"\n  ARTEFACT     arm SE against arm SA: E {stat['E']:+.4f} "
          f"({stat['sigma']:+.1f} sd), null sd {stat['null_sd']:.4f}")
    print("               there is no model defect here by construction, so "
          "this is pure estimator")

    a, sd = stat["E"], stat["null_sd"]
    resid = ARM_E_EXCESS - a
    if gate_ok is False:
        verdict = ("VALIDITY GATE FAILED, two independent model generations do "
                   "not agree, so no verdict is reported.")
    else:
        margin = min(abs(a - ARTEFACT_DOMINATES), abs(a - ARTEFACT_MINOR))
        if margin < sd:
            verdict = (f"BOUNDARY, the nearest threshold is {margin:.4f} away "
                       f"against a null sd of {sd:.4f}, so the threshold call "
                       f"is REFUSED. Reported as the in between case: artefact "
                       f"{a:+.4f}, residual {resid:+.4f}, both quoted, neither "
                       f"alone.")
        elif a >= ARTEFACT_DOMINATES:
            verdict = (f"artefact {a:+.4f} >= {ARTEFACT_DOMINATES:.2f}. Arm E's "
                       f"{ARM_E_EXCESS:+.4f} is DOMINATED BY CONSTRUCTION and "
                       f"the 'real speeds make it worse' reading FALLS.")
        elif a <= ARTEFACT_MINOR:
            verdict = (f"artefact {a:+.4f} <= {ARTEFACT_MINOR:.2f}. Construction "
                       f"is minor and arm E's reading STANDS, residual "
                       f"{resid:+.4f}.")
        else:
            verdict = (f"artefact {a:+.4f} is between {ARTEFACT_MINOR:.2f} and "
                       f"{ARTEFACT_DOMINATES:.2f}. Residual {resid:+.4f}. Both "
                       f"numbers reported, neither alone.")
    out["verdict"] = verdict
    out["residual_after_subtraction"] = round(resid, 4)
    print(f"\n  -> {verdict}")

    print("\n  per bin standardised speed power, SE against SA. arm E's own "
          "shape climbs monotonically to about 2.4 at 62.5 Hz; if this does "
          "the same the shape is the estimator's, not the model's")
    am = psds["SE_forced_sth"].mean(0)
    rm = ref.mean(0)
    print(f"    {'freq Hz':>9}  {'ratio':>7}")
    for f_, r_ in zip(freqs, am / rm):
        print(f"    {f_:>9.2f}  {r_:>7.3f}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.out}")
    print("""
  DIAGNOSTIC ONLY, never a contract score. no serving change follows and
  no build is authorised either way. phase conditioning and the spectral
  loss term remain NOT AUTHORISED.
  read the VALIDITY gate first. a threshold call landing within one null
  sd of a threshold is REFUSED, not rounded, and the in between case is
  reported instead.
  the per bin shape is NOT subordinate. w4_artefact was decided by shape
  and not by its scalar.""")


if __name__ == "__main__":
    main()
