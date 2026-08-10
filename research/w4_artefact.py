"""How much of `w4_timing` arm G's excess is the estimator rather than the model?

PRE REGISTERED in HANDOFF, thresholds fixed before this file existed:

    artefact >= 0.10   arm G's +0.1221 is explained by construction, the dt head
                       is exonerated, and the mid band bump is an artefact
    artefact <= 0.04   the artefact is small, arm G's excess is mostly real, and
                       the withdrawn attribution is reinstated
    in between         subtract and report both numbers with the residual

The problem. `w4_timing` arm G resamples one interval per position under full
teacher forcing and reads a +0.1221 spectral excess against human. That was
written up as a defect in the dt conditional. `w4_dtcal` then found the
conditional calibrated on every instrument at 1,034,517 positions: PIT flat to
KS 0.0018, unbiased to +0.000 ms, uniform inside every slice of the previous
interval, the interval two back and the real speed, and serially independent in
its PIT residuals to within 0.0027. So the attribution was withdrawn.

The reason it had to be withdrawn is an estimator bias that arm G shares with a
diagnostic already deleted from `w4_dtcal` for exactly this reason. A human
interval feeds forward into the conditional of its own successor and a resampled
one does not:

    human   Cov(dt_i, dt_i+1) = E[Cov(dt_i, mu_i+1(dt_i) | H_i)] + Cov(mu_i, mu_i+1)
    arm G   Cov(dt_i, dt_i+1) =                                    Cov(mu_i, mu_i+1)

The first term survives in the human and vanishes in arm G, and it vanishes for a
perfect model exactly as readily as for a bad one. Under correlation is extra high
frequency power. Arm G's number is therefore a defect plus an artefact of unknown
size, and nothing measured so far separates them.

THE CONTROL. Generate sequences FROM the model. On its own samples the model IS
the true conditional, exactly, by construction, with no estimation error and no
approximation. Then run the identical arm G pipeline on them. There is no defect
left for it to find, so any excess that appears is PURE artefact.

    arm SA   model generated, its own intervals, the synthetic reference
    arm SG   the same rows, intervals resampled one step under full teacher
             forcing, exactly as arm G does to human rows

Both arms are decoded through the serving decoder and windowed by the same code
as every other spectral run in this repo, imported rather than copied so the
comparison cannot drift.

VALIDITY GATE. The generation temperature and the resampling temperature must be
identical, or the model is not the true conditional of its own samples and the
control means nothing. Both are `--temp` and there is deliberately no way to set
them apart.

DIAGNOSTIC ONLY, never a contract score. One trajectory per row, no selection, no
reranking. No serving change follows and no build is authorised either way. Phase
conditioning and the spectral loss term remain NOT AUTHORISED.

Safety. Reads `training/events_*.npy` and one checkpoint. Touches no evaluation
data, no scoring code, and never `training/candi_polar_flow_best.pt`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments.event_stream_polar as esp  # noqa: E402
from models.event_ar import (  # noqa: E402
    DT_PAD_CLASS, EventARModel, N_DT_VALS, class_to_dt_ms, prefix_state,
)
from models.event_stream_polar import S_PAD_CLASS, TH_NULL_CLASS  # noqa: E402
from research.w4_timing import (  # noqa: E402
    BAND_HI_HZ, BAND_LO_HZ, HZ, MAX_T, N_TRAIN_DEFAULT, TRAIN_PICK_SEED,
    band_stat, psd, signals,
)

# Registered in HANDOFF before this file existed.
ARTEFACT_EXPLAINS = 0.10
ARTEFACT_NEGLIGIBLE = 0.04
ARM_G_EXCESS = 0.1221          # w4_timing arm G against Aq, n=20000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--temp", type=float, default=1.0,
                    help="used for BOTH the generation and the resample. There "
                         "is deliberately no way to set them apart: if they "
                         "differ the model is not the true conditional of its "
                         "own samples and the control means nothing.")
    ap.add_argument("--w", type=int, default=64)
    ap.add_argument("--out", default="research/w4_artefact.json")
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
    keep = L >= 12
    conds, L = conds[keep], L[keep]
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

    # ---- arm SA, the synthetic reference ------------------------------
    # Free running exactly as arm C runs. These rows ARE the ground truth for
    # this control, and the model is their exact conditional by construction.
    print("  arm SA, model generated, the synthetic reference", flush=True)
    sa_s = np.full((B, MAX_T), S_PAD_CLASS, dtype=np.int64)
    sa_th = np.full((B, MAX_T), TH_NULL_CLASS, dtype=np.int64)
    sa_dtc = np.zeros((B, MAX_T), dtype=np.int64)
    with torch.no_grad():
        for c0 in range(0, B, args.batch):
            sl = slice(c0, min(c0 + args.batch, B))
            s_o, th_o, dt_o = model.sample(cond_t[sl].to(dev),
                                           temperature=args.temp)
            w = s_o.shape[1]
            sa_s[sl, :w] = s_o.cpu().numpy()
            sa_th[sl, :w] = th_o.cpu().numpy()
            sa_dtc[sl, :w] = dt_o.cpu().numpy()

    # ---- arm SG, the same rows one step resampled ----------------------
    # Identical to w4_timing arm G, applied to model rows instead of human ones.
    print("  arm SG, the same rows, intervals resampled one step under full "
          "teacher forcing", flush=True)
    sg_dtc = np.zeros((B, MAX_T), dtype=np.int64)
    s_t = torch.from_numpy(sa_s)
    th_t = torch.from_numpy(sa_th)
    dt_t = torch.from_numpy(sa_dtc)
    with torch.no_grad():
        for c0 in range(0, B, args.batch):
            sl = slice(c0, min(c0 + args.batch, B))
            s_b, th_b = s_t[sl].to(dev), th_t[sl].to(dev)
            dt_b, cnd = dt_t[sl].to(dev), cond_t[sl].to(dev)
            s_p, th_p, dt_p = model.shift_inputs(s_b, th_b, dt_b)
            st = prefix_state(s_b, th_b, dt_b, cnd)
            x = model.trunk(s_p, th_p, dt_p, st, cnd)
            p = torch.softmax(model.dt_logits(x, s_b, th_b).double()
                              / args.temp, dim=-1)
            # Same renormalisation w4_dtcal established: the PAD class decodes
            # to a clamped 150 ms and is not a real interval. Measured at nil,
            # applied anyway so the two arms cannot differ by it.
            p = p[..., :N_DT_VALS]
            p = p / p.sum(-1, keepdim=True).clamp(min=1e-12)
            samp = torch.multinomial(p.reshape(-1, N_DT_VALS), 1).squeeze(-1)
            sg_dtc[sl] = samp.view(p.shape[0], p.shape[1]).cpu().numpy()

    sa_dt = class_to_dt_ms(torch.from_numpy(sa_dtc)).numpy().astype(np.float64)
    sg_dt = class_to_dt_ms(torch.from_numpy(sg_dtc)).numpy().astype(np.float64)
    dead = sa_s >= S_PAD_CLASS
    sa_dt[dead] = 0.0
    sg_dt[dead] = 0.0          # padding stays padding on BOTH arms

    def collect(s_arr, th_arr, dtms_arr):
        """Decode through the SERVING decoder, resample at 125 Hz. Same path as
        every other spectral run, and `signals` is imported not copied."""
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
    arms = {"SA_model_own_dt": (sa_s, sa_th, sa_dt),
            "SG_model_resampled_dt": (sa_s, sa_th, sg_dt)}
    out = {"ckpt": args.ckpt, "w": args.w, "hz": HZ, "n_rows": int(B),
           "temp": args.temp, "diagnostic_only": True,
           "pre_registered": "HANDOFF.md 2026-08-05",
           "band_hz": [BAND_LO_HZ, BAND_HI_HZ], "freqs_hz": freqs.tolist(),
           "arm_G_excess_being_priced": ARM_G_EXCESS,
           "thresholds": {"explains": ARTEFACT_EXPLAINS,
                          "negligible": ARTEFACT_NEGLIGIBLE},
           "retention": {}}

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

    ref = psds["SA_model_own_dt"]
    self_stat = band_stat(ref, ref, freqs)
    stat = band_stat(psds["SG_model_resampled_dt"], ref, freqs)
    out["self_check"] = self_stat
    out["artefact"] = stat

    print(f"\n  SELF CHECK   arm SA against itself: E {self_stat['E']:+.4f}, "
          f"must be exactly 0, "
          f"{'PASS' if abs(self_stat['E']) < 1e-9 else 'FAIL'}")
    print(f"\n  ARTEFACT     arm SG against arm SA: E {stat['E']:+.4f} "
          f"({stat['sigma']:+.1f} sd), null sd {stat['null_sd']:.4f}")
    print("               there is no model defect here by construction, so "
          "this is pure estimator")

    a = stat["E"]
    if a >= ARTEFACT_EXPLAINS:
        verdict = (f"artefact {a:+.4f} >= {ARTEFACT_EXPLAINS:.2f}, so arm G's "
                   f"{ARM_G_EXCESS:+.4f} IS EXPLAINED BY CONSTRUCTION. The dt "
                   f"head is exonerated and the mid band bump is an artefact.")
    elif a <= ARTEFACT_NEGLIGIBLE:
        verdict = (f"artefact {a:+.4f} <= {ARTEFACT_NEGLIGIBLE:.2f}, so arm G's "
                   f"{ARM_G_EXCESS:+.4f} is mostly REAL. The withdrawn "
                   f"attribution to the dt head is REINSTATED.")
    else:
        verdict = (f"artefact {a:+.4f} is between {ARTEFACT_NEGLIGIBLE:.2f} and "
                   f"{ARTEFACT_EXPLAINS:.2f}. Residual after subtraction is "
                   f"{ARM_G_EXCESS - a:+.4f}, and BOTH numbers are reported. "
                   f"Neither the artefact nor the defect alone accounts for "
                   f"arm G.")
    # A verdict that flips on a margin smaller than the null sd is spurious
    # precision and must say so out loud. The thresholds were registered before
    # the null sd for this arm was known, and nothing licenses reading a
    # threshold to a resolution the measurement does not have.
    sd = stat["null_sd"]
    margin = min(abs(a - ARTEFACT_EXPLAINS), abs(a - ARTEFACT_NEGLIGIBLE))
    if margin < sd:
        verdict += (
            f"\n     BOUNDARY WARNING. The nearest threshold is {margin:.4f} "
            f"away and the null sd is {sd:.4f}, so this call rests on "
            f"{margin / sd:.2f} of one standard deviation and is NOT a clean "
            f"pass. Read it as the registered IN BETWEEN case: the artefact is "
            f"real at {a:+.4f}, the residual is {ARM_G_EXCESS - a:+.4f}, and "
            f"BOTH numbers are reported. At {stat['sigma']:+.1f} sd the "
            f"artefact is not separable from zero, and it is equally not "
            f"separable from twice its own size.")
    out["verdict"] = verdict
    out["boundary_margin_in_null_sd"] = round(margin / sd, 3) if sd > 0 else None
    out["residual_after_subtraction"] = round(ARM_G_EXCESS - a, 4)
    print(f"\n  -> {verdict}")

    print(f"\n  per bin standardised speed power, SG against SA")
    rm, am = ref.mean(0), psds["SG_model_resampled_dt"].mean(0)
    print(f"    {'freq Hz':>9}  {'ratio':>7}")
    for f_, r_ in zip(freqs, am / rm):
        print(f"    {f_:>9.2f}  {r_:>7.3f}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.out}")
    print("""
  DIAGNOSTIC ONLY, never a contract score. one trajectory per row, no
  selection, no reranking. no serving change follows and no build is
  authorised either way. phase conditioning and the spectral loss term
  remain NOT AUTHORISED.
  the generation temperature and the resample temperature are the SAME
  argument on purpose. if they ever differ the model is not the true
  conditional of its own samples and this control means nothing.""")


if __name__ == "__main__":
    main()
