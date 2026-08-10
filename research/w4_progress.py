"""Does the model condition on PROGRESS through the movement?

PRE REGISTERED in HANDOFF, thresholds fixed before this file existed. Two of the
three are carried over unchanged from w4_condtex and w4_launch, because the
statistics and the sample size are identical.

w4_position measured texture against elapsed fraction of the movement and found
the human curve climbing twice as fast as the model's. w4_condtex sliced by the
history's own recent texture and found nothing. Progress is the covariate that
matches the defect and it has never been sliced.

It is a FAIR covariate because the model is told the answer. prepare_events.py
builds events_cond.npy as [log_dist, log_dur, cos, sin] with log_dur the log of the
total duration in SECONDS, and events_dt.npy is the same durations in MILLISECONDS,
so

    progress(t) = (sum of dt before t) / (exp(cond[1]) * 1000)

is a function of the history and the conditioning, both of which the model holds at
every step. Slicing a PIT by a quantity that depends on the FUTURE proves nothing,
because the conditional was never required to be uniform across slices of the
future. Slicing by a quantity the model was handed is a real test.

Two co primaries on the speed head, across progress quintiles:

    D    = shape(top quintile) - shape(bottom quintile)
    Trng = max tilt across quintiles - min tilt across quintiles

    D >= 0.06     or  Trng >= 0.02     the conditional depends on progress WRONGLY
    D <= 0.02     and Trng <= 0.008    the model conditions on progress correctly
    otherwise     MIXED
    BOUNDARY      within one bootstrap sd of a threshold that call is REFUSED

TWO CONTROL SCHEMES AND THE VERDICT REQUIRES BOTH. Within POSITION bands, progress
varies because duration varies, so a slope could be short movements versus long
ones. Within DURATION bands, progress varies because position varies, so a slope
could be position, which w4_launch showed is flat. A real dependence appears in
both. If the schemes disagree in sign, or only one clears its threshold, the
registered answer is CONFOUNDED and that is what gets reported.

Position 0 is excluded. Progress is identically zero there and w4_launch already
showed that position is anomalous on its own.

VALIDITY ARM in both schemes. Tokens drawn from the model's own predictive law,
sliced by the same progress quintiles from the same real histories. Flat within two
bootstrap sd or the run is reported as failed rather than interpreted.

DESCRIPTIVE PANEL, linked to the shape reading by arithmetic. What humans did,
mean |s(t) - s(t-1)| over real tokens, against what the model expected, the same
quantity under its predicted law. If the human curve climbs and the model's does
not, the PIT shape MUST read above 1 at high progress. If the two disagree the
estimator is wrong and neither is reported. The panel decides nothing on its own.

DIAGNOSTIC ONLY, never a contract score. No serving change follows and no build is
authorised by any outcome. Phase conditioning and the spectral loss term remain
NOT AUTHORISED.

Safety. Reads training/events_*.npy and one checkpoint. Touches no evaluation
data, no scoring code, and never training/candi_polar_flow_best.pt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("EVENT_STEPS", "100")
os.environ.setdefault("EVENT_CFG_W", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import experiments.event_stream_polar as esp  # noqa: E402
from models.event_ar import (  # noqa: E402
    EventARModel, dt_ms_to_class, prefix_state,
)
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TH_NULL_CLASS, dth_lattice_to_class, s2_to_class,
)
from research.w4_timing import (  # noqa: E402
    MAX_T, N_TRAIN_DEFAULT, TRAIN_PICK_SEED,
)
from research.w4_launch import (  # noqa: E402
    N_REAL, pit_of, pit_shape, renorm, self_sample,
)

WRONG_D = 0.06          # w4_condtex, unchanged
RIGHT_D = 0.02          # w4_condtex, unchanged
WRONG_TILT = 0.02       # w4_launch, unchanged
RIGHT_TILT = 0.008      # w4_launch, unchanged
VALID_SD = 2.0
N_QUINT = 5
MIN_CELL = 200
POS_BANDS = [(1, 4), (4, 8), (8, 12), (12, 20), (20, 32), (32, MAX_T)]
N_DUR_BANDS = 5


def cell_stats(u, cov, edges, mask):
    """Per quintile shape and signed tilt inside one control cell."""
    q = np.digitize(cov[mask], edges)
    uu = u[mask]
    sh, ti, cnt = [], [], []
    for k in range(N_QUINT):
        m = q == k
        cnt.append(int(m.sum()))
        if m.sum() >= MIN_CELL:
            sh.append(pit_shape(uu[m]))
            ti.append(float(uu[m].mean() - 0.5))
        else:
            sh.append(None)
            ti.append(None)
    return sh, ti, cnt


def spread(vals):
    """Top minus bottom for shape, max minus min for tilt. None if incomplete."""
    ok = [v for v in vals if v is not None]
    if len(ok) < 2:
        return None, None
    d = (vals[-1] - vals[0]) if (vals[-1] is not None and vals[0] is not None) \
        else None
    return d, float(max(ok) - min(ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--out", default="research/w4_progress.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    rng = np.random.default_rng(args.seed + 7)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed + 113)

    lengths = np.load("training/events_len.npy")
    N = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(N, min(N, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(N), trained)
    pick = np.sort(np.random.default_rng(args.seed)
                   .choice(held, args.n, replace=False))

    s2 = np.load("training/events_s2.npy", mmap_mode="r")[pick]
    dth = np.load("training/events_dth.npy", mmap_mode="r")[pick]
    dt_raw = np.load("training/events_dt.npy", mmap_mode="r")[pick].astype(np.float64)
    conds = np.load("training/events_cond.npy")[pick]
    L = np.minimum(lengths[pick], MAX_T).astype(np.int64)
    keep = L >= 12
    s2, dth, dt_raw, conds, L = (s2[keep], dth[keep], dt_raw[keep],
                                 conds[keep], L[keep])
    B = len(L)
    print(f"  corpus {N:,}, never seen {len(held):,}, drew {args.n:,}")
    print(f"  {B:,} rows at least 12 events, the same rows w4_condtex used\n",
          flush=True)

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
        real_dt[i, :n] = dt_raw[i, :n]
    real_dt_cls = dt_ms_to_class(torch.from_numpy(real_dt)).numpy()

    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    print(f"  {args.ckpt} step {ck.get('step')} "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params\n",
          flush=True)

    s_t = torch.from_numpy(real_s)
    th_t = torch.from_numpy(real_th)
    dt_t = torch.from_numpy(real_dt_cls)
    cond_t = torch.from_numpy(conds[:, :4].astype(np.float32))

    heads = ("s", "th", "dt")
    pit = {h: np.zeros((B, MAX_T), dtype=np.float64) for h in heads}
    vpit = {h: np.zeros((B, MAX_T), dtype=np.float64) for h in heads}
    exp_jump = np.zeros((B, MAX_T), dtype=np.float64)
    live = np.zeros((B, MAX_T), dtype=bool)
    live_th = np.zeros((B, MAX_T), dtype=bool)
    s_grid = torch.arange(N_REAL["s"], device=dev).float()

    print("  one teacher forced forward pass, identical to w4_condtex's",
          flush=True)
    with torch.no_grad():
        for c0 in range(0, B, args.batch):
            sl = slice(c0, min(c0 + args.batch, B))
            s_b, th_b, dt_b = s_t[sl].to(dev), th_t[sl].to(dev), dt_t[sl].to(dev)
            cnd = cond_t[sl].to(dev)
            s_p, th_p, dt_p = model.shift_inputs(s_b, th_b, dt_b)
            st = prefix_state(s_b, th_b, dt_b, cnd)
            x = model.trunk(s_p, th_p, dt_p, st, cnd)
            lg = {"s": model.s_head(x),
                  "th": model.th_logits(x, s_b),
                  "dt": model.dt_logits(x, s_b, th_b)}
            true = {"s": s_b, "th": th_b, "dt": dt_b}
            for h in heads:
                p = renorm(torch.softmax(lg[h], -1), N_REAL[h])
                pit[h][sl] = pit_of(p, true[h], rng, dev).double().cpu().numpy()
                vpit[h][sl] = pit_of(p, self_sample(p, gen), rng,
                                     dev).double().cpu().numpy()
                if h == "s":
                    # E|s(t) - s(t-1)| under the model's own law, against the
                    # REAL previous class. The descriptive panel's model curve.
                    prev = torch.roll(s_b, 1, dims=1).clamp(max=N_REAL["s"] - 1)
                    d = (s_grid.view(1, 1, -1) - prev.unsqueeze(-1).float()).abs()
                    exp_jump[sl] = (p * d).sum(-1).double().cpu().numpy()
            live[sl] = (s_b < S_PAD_CLASS).cpu().numpy()
            live_th[sl] = ((s_b < S_PAD_CLASS) & (th_b < TH_NULL_CLASS)
                           ).cpu().numpy()

    # progress(t) = elapsed before t over total duration, both from what the
    # model holds. cond[1] is log seconds, dt is milliseconds.
    total_ms = np.exp(conds[:, 1].astype(np.float64)) * 1000.0
    elapsed = np.concatenate([np.zeros((B, 1)),
                              np.cumsum(np.where(live, real_dt, 0.0), 1)[:, :-1]], 1)
    prog = elapsed / total_ms[:, None]
    have = live.copy()
    have[:, 0] = False          # progress is identically zero at position 0
    have &= np.isfinite(prog)

    real_jump = np.abs(np.diff(real_s.astype(np.float64), axis=1))
    real_jump = np.concatenate([np.zeros((B, 1)), real_jump], 1)

    print(f"  {live.sum():,} live positions, {have.sum():,} of them with a "
          f"defined progress\n", flush=True)

    # Control cells. Scheme A bands by position index, scheme B by total
    # duration, and each kills the confound the other cannot see.
    cells = {"by_position": [], "by_duration": []}
    for lo, hi in POS_BANDS:
        m = np.zeros_like(have)
        m[:, lo:hi] = have[:, lo:hi]
        cells["by_position"].append((f"{lo} to {hi - 1}", m))
    dur_edges = np.quantile(conds[:, 1].astype(np.float64),
                            [k / N_DUR_BANDS for k in range(1, N_DUR_BANDS)])
    dq = np.digitize(conds[:, 1].astype(np.float64), dur_edges)
    for k in range(N_DUR_BANDS):
        m = have & (dq == k)[:, None]
        cells["by_duration"].append((f"dur Q{k + 1}", m))
    edges = {sch: [np.quantile(prog[m], [j / N_QUINT for j in range(1, N_QUINT)])
                   for _, m in cl] for sch, cl in cells.items()}

    def pooled(store, head, scheme, rows):
        """Count weighted D and tilt range across the cells of one scheme."""
        lv = live_th if head == "th" else live
        U, PR, LV = store[head][rows], prog[rows], lv[rows]
        per, nD, dD, nT, dT = [], 0.0, 0.0, 0.0, 0.0
        for ci, (lab, m) in enumerate(cells[scheme]):
            mm = m[rows] & LV
            sh, ti, cnt = cell_stats(U, PR, edges[scheme][ci], mm)
            d, _ = spread(sh)
            _, tr = spread(ti)
            per.append({"cell": lab, "D": d, "tilt_range": tr,
                        "shapes": sh, "tilts": ti, "n": cnt})
            w = cnt[0] + cnt[-1]
            if d is not None:
                nD += d * w
                dD += w
            if tr is not None:
                nT += tr * w
                dT += w
        return ((nD / dD if dD > 0 else None), (nT / dT if dT > 0 else None), per)

    all_rows = np.arange(B)
    out = {"ckpt": args.ckpt, "n_rows": int(B), "seed": args.seed,
           "diagnostic_only": True, "pre_registered": "HANDOFF.md 2026-08-05",
           "thresholds": {"wrong_D": WRONG_D, "right_D": RIGHT_D,
                          "wrong_tilt": WRONG_TILT, "right_tilt": RIGHT_TILT},
           "n_live": int(live.sum()), "n_with_progress": int(have.sum()),
           "validity": {}, "real": {}}

    print("  VALIDITY arm, both schemes. tokens from the model's OWN law sliced by")
    print("  the SAME progress quintiles from the SAME real histories\n")
    print(f"    {'scheme':>14} {'head':>5} {'D':>9} {'sd':>8} {'in sd':>7}")
    vfail = []
    for sch in cells:
        for h in heads:
            d, _, _ = pooled(vpit, h, sch, all_rows)
            vb = []
            for _ in range(max(20, args.draws // 4)):
                rs = rng.integers(0, B, B)
                x, _, _ = pooled(vpit, h, sch, rs)
                if x is not None:
                    vb.append(x)
            sd = float(np.std(vb)) if len(vb) > 10 else float("nan")
            z = abs(d) / sd if (d is not None and sd > 0) else float("nan")
            out["validity"][f"{sch}:{h}"] = {"D": d, "bootstrap_sd": sd, "z": z}
            print(f"    {sch:>14} {h:>5} " +
                  (f"{d:>+9.4f}" if d is not None else f"{'nan':>9}") +
                  f" {sd:>8.4f} {z:>7.1f}")
            if d is None or not (z <= VALID_SD):
                vfail.append(f"{sch}:{h}")
    if vfail:
        print(f"\n  VALIDITY ARM FAILED on {', '.join(vfail)}. the slicing "
              f"manufactures a slope, so nothing below would mean anything.")
        out["verdict"] = f"FAILED, validity arm slopes on {','.join(vfail)}"
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        return
    print("\n  validity arm PASSES everywhere, the slicing is sound\n")

    res = {}
    for sch in cells:
        d, tr, per = pooled(pit, "s", sch, all_rows)
        boot_d, boot_t = [], []
        for _ in range(args.draws):
            rs = rng.integers(0, B, B)
            a, b, _ = pooled(pit, "s", sch, rs)
            if a is not None:
                boot_d.append(a)
            if b is not None:
                boot_t.append(b)
        sd_d = float(np.std(boot_d)) if len(boot_d) > 20 else float("nan")
        sd_t = float(np.std(boot_t)) if len(boot_t) > 20 else float("nan")
        res[sch] = {"D": d, "D_sd": sd_d, "tilt_range": tr,
                    "tilt_range_sd": sd_t, "per_cell": per}
        out["real"][sch] = res[sch]

        print(f"  speed head, control by {sch.replace('by_', '')}. "
              f"D {d:+.4f} sd {sd_d:.4f}, tilt range {tr:.4f} sd {sd_t:.4f}")
        print(f"      {'cell':>12} " +
              " ".join(f"{'Q' + str(k + 1):>8}" for k in range(N_QUINT)) +
              f" {'D':>9}")
        for c in per:
            print(f"      {c['cell']:>12} " +
                  " ".join(f"{v:>8.3f}" if v is not None else f"{'nan':>8}"
                           for v in c["shapes"]) +
                  (f" {c['D']:>+9.4f}" if c["D"] is not None else f" {'nan':>9}"))
        print()

    # Descriptive panel, pooled over position bands so progress is the only axis.
    pos_all = np.zeros_like(have)
    for _, m in cells["by_position"]:
        pos_all |= m
    qq = np.digitize(prog[pos_all], np.quantile(prog[pos_all],
                                                [j / N_QUINT for j in range(1, N_QUINT)]))
    rj, ej = real_jump[pos_all], exp_jump[pos_all]
    panel = []
    for k in range(N_QUINT):
        m = qq == k
        panel.append({"n": int(m.sum()),
                      "human": float(rj[m].mean()) if m.any() else None,
                      "model_expected": float(ej[m].mean()) if m.any() else None})
    out["panel_step_change"] = panel
    print("  DESCRIPTIVE panel, mean |s(t) - s(t-1)| in speed classes.")
    print("  what humans DID against what the model EXPECTED, by progress\n")
    print(f"      {'quintile':>10} {'n':>12} {'human':>9} {'model':>9} {'ratio':>8}")
    for k, p in enumerate(panel):
        r = (p["model_expected"] / p["human"]) if p["human"] else float("nan")
        print(f"      {'Q' + str(k + 1):>10} {p['n']:>12,} "
              f"{p['human']:>9.4f} {p['model_expected']:>9.4f} {r:>8.4f}")

    A, Bs = res["by_position"], res["by_duration"]
    same_sign = (A["D"] is not None and Bs["D"] is not None
                 and A["D"] * Bs["D"] > 0)

    def call(v, sd, hi, lo):
        """Two sided, on the MAGNITUDE of the departure from flat.

        The registration wrote these thresholds one sided, because regression to
        the mean predicts a POSITIVE D and that was the shape being looked for. A
        one sided rule cannot tell "flat" from "sloping the other way", and it
        called a D of -0.0301 at 6.3 sd from zero "conditions correctly". See the
        CORRECTION in HANDOFF. Flat means near zero in either direction, so the
        rule reads |D|.
        """
        if v is None:
            return "none"
        a = abs(v)
        if min(abs(a - hi), abs(a - lo)) < sd:
            return "boundary"
        return "wrong" if a >= hi else ("right" if a <= lo else "mixed")

    cd = {s: call(res[s]["D"], res[s]["D_sd"], WRONG_D, RIGHT_D) for s in res}
    ct = {s: call(res[s]["tilt_range"], res[s]["tilt_range_sd"],
                  WRONG_TILT, RIGHT_TILT) for s in res}
    out["calls"] = {"D": cd, "tilt_range": ct, "same_sign_D": same_sign}

    wrong_both = all(cd[s] == "wrong" for s in res) or \
        all(ct[s] == "wrong" for s in res)
    right_both = all(cd[s] == "right" for s in res) and \
        all(ct[s] == "right" for s in res)
    if wrong_both and same_sign:
        verdict = ("THE CONDITIONAL DEPENDS ON PROGRESS WRONGLY, in BOTH control "
                   "schemes and with the same sign. The model does not use how far "
                   "through the movement it is, which is w4_position's under "
                   "modulation seen one step at a time and is addressable in "
                   "training.")
    elif right_both:
        verdict = ("THE MODEL CONDITIONS ON PROGRESS CORRECTLY, in both control "
                   "schemes and on both co primaries. Progress joins recent "
                   "texture and exposure as a dead explanation, and the defect is "
                   "somewhere no one step instrument has reached.")
    elif any(v == "boundary" for v in list(cd.values()) + list(ct.values())):
        verdict = (f"BOUNDARY, the threshold call is REFUSED. D calls {cd}, tilt "
                   f"calls {ct}. Reported as the in between case.")
    else:
        verdict = (f"CONFOUNDED or MIXED. D calls {cd}, tilt calls {ct}, same sign "
                   f"{same_sign}. The registered rule requires BOTH control "
                   f"schemes to agree, so no dependence on progress is claimed.")
    out["verdict"] = verdict
    print(f"\n  -> {verdict}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.out}")
    print("""
  DIAGNOSTIC ONLY, never a contract score. no serving change follows and
  no build is authorised by any outcome. phase conditioning and the
  spectral loss term remain NOT AUTHORISED.
  the descriptive panel decides nothing. it is linked to the shape reading
  by arithmetic, so if the two disagree the estimator is wrong.""")


if __name__ == "__main__":
    main()
