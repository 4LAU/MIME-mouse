"""w4_pairdep. AMENDMENT 30, registered in step0_prereg.md before this
file existed.

Exchangeability test of q1g0(e1 | e0, cond) on its own validation rows:
if the model conditional equals the true conditional per row, a model
draw and the human e1 are interchangeable, so E|d1 - h| = E|d1 - d2|
per channel. f > 0 means the model misses row specific location; f < 0
means over dispersion. Pure measurement, no training, no contract run,
no scorer.
"""
import json
import sys

import numpy as np
import torch

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp                       # noqa: E402
from models.event_stream_polar import (N_TH_CLASSES, S_PAD_CLASS,  # noqa: E402
                                       TH_NULL_CLASS, TICK_CLASS)
from w4_pairq import (N_VAL, P1_PATH, VAL_ROWS_SEED, Pair1,        # noqa: E402
                      pair_tokens, splits)
import ledger                                                      # noqa: E402

SEED_D1, SEED_D2 = 30001, 30002


def main():
    dev = esp._DEVICE
    lengths, trained, held = splits()
    val = np.sort(np.random.default_rng(VAL_ROWS_SEED).choice(held, N_VAL, replace=False))
    val = val[lengths[val] >= 2]
    s0, th0, d0, s1, th1, d1 = (x[val] for x in pair_tokens())
    cond = np.load("training/events_cond.npy")[val, :4].astype(np.float32)
    print(f"  val rows {len(val):,}", flush=True)

    pk = torch.load(P1_PATH, map_location=dev, weights_only=False)
    pair = Pair1(**pk["config"]).to(dev).eval()
    pair.load_state_dict(pk["model_state_dict"])
    print(f"  q1g0 best epoch {pk['best']['epoch']}", flush=True)

    C = torch.from_numpy(cond).to(dev)
    S0, TH0, D0 = (torch.from_numpy(np.asarray(x, dtype=np.int64)).to(dev)
                   for x in (s0, th0, d0))

    def draw(seed):
        out_s, out_th, out_dt = [], [], []
        with torch.no_grad():
            for c0 in range(0, len(val), 65536):
                torch.manual_seed(seed + c0)
                sl = slice(c0, c0 + 65536)
                ss, tt, dd = pair.sample(C[sl], S0[sl], TH0[sl], D0[sl], 1.0, 1.0, 1.0)
                out_s.append(ss.cpu().numpy()); out_th.append(tt.cpu().numpy())
                out_dt.append(dd.cpu().numpy())
        return (np.concatenate(out_s), np.concatenate(out_th), np.concatenate(out_dt))

    a_s, a_th, a_dt = draw(SEED_D1)
    b_s, b_th, b_dt = draw(SEED_D2)

    res = {"n_rows": int(len(val)), "seeds": [SEED_D1, SEED_D2],
           "best_epoch": int(pk["best"]["epoch"])}

    def ftest(h, x, y, name, circular=False):
        h = h.astype(np.float64); x = x.astype(np.float64); y = y.astype(np.float64)
        dh, dd = np.abs(x - h), np.abs(x - y)
        if circular:
            dh = np.minimum(dh, N_TH_CLASSES - dh)
            dd = np.minimum(dd, N_TH_CLASSES - dd)
        per_row = dh - dd
        m, se = per_row.mean(), per_row.std(ddof=1) / np.sqrt(len(per_row))
        base = dd.mean()
        f, fse = m / base, se / base
        if f >= 0.02 and abs(f) >= 2 * fse:
            v = "MISLOCATED"
        elif f <= -0.02 and abs(f) >= 2 * fse:
            v = "OVERDISPERSED"
        elif abs(f) <= 0.02:
            v = "CALIBRATED PER ROW"
        else:
            v = "BETWEEN"
        print(f"  {name}: E|d1-h| {dh.mean():.4f}  E|d1-d2| {base:.4f}  "
              f"delta {m:+.4f} se {se:.4f}  f {f:+.4f} fse {fse:.4f}  {v}")
        return dict(e_dh=float(dh.mean()), e_dd=float(base), delta=float(m),
                    delta_se=float(se), f=float(f), f_se=float(fse), verdict=v)

    print("\n  READ 1 (PRIMARY), exchangeability per channel:")
    res["read1_s"] = ftest(s1, a_s, b_s, "s ")
    res["read1_dt"] = ftest(d1, a_dt, b_dt, "dt")

    motion_h = (s1 > TICK_CLASS) & (s1 < S_PAD_CLASS)
    keep = motion_h & (a_th != TH_NULL_CLASS) & (b_th != TH_NULL_CLASS)
    print(f"\n  READ 2, th on motion rows all three, retained "
          f"{keep.mean():.3f} of rows:")
    res["read2_th"] = ftest(th1[keep], a_th[keep], b_th[keep], "th", circular=True)
    res["read2_retained"] = float(keep.mean())

    from scipy.stats import spearmanr
    print("\n  READ 3 (informational), dependence:")
    for nm, x0, xh, xd in (("s", s0, s1, a_s), ("dt", d0, d1, a_dt)):
        rh = spearmanr(x0, xh).statistic
        rd = spearmanr(x0, xd).statistic
        print(f"  rho({nm}0,{nm}1): human {rh:+.4f}  draw {rd:+.4f}")
        res[f"read3_rho_{nm}"] = dict(human=float(rh), draw=float(rd))
        edges = np.unique(np.quantile(x0, np.linspace(0, 1, 11)))
        bins = np.clip(np.digitize(x0, edges[1:-1]), 0, len(edges) - 2)
        sh = [xh[bins == b].std(ddof=1) for b in range(len(edges) - 1)]
        sd = [xd[bins == b].std(ddof=1) for b in range(len(edges) - 1)]
        print(f"    cond std {nm}1 by {nm}0 decile, human: "
              + " ".join(f"{v:.2f}" for v in sh))
        print(f"    cond std {nm}1 by {nm}0 decile, draw:  "
              + " ".join(f"{v:.2f}" for v in sd))
        res[f"read3_condstd_{nm}"] = dict(human=[float(v) for v in sh],
                                          draw=[float(v) for v in sd])

    print("\n  READ 4 (informational), marginals:")
    for nm, xh, xd in (("s1", s1, a_s), ("dt1", d1, a_dt)):
        print(f"  {nm}: human mean {xh.mean():.3f} std {xh.std(ddof=1):.3f}  "
              f"draw mean {xd.mean():.3f} std {xd.std(ddof=1):.3f}")
        res[f"read4_{nm}"] = dict(h_mean=float(xh.mean()), h_std=float(xh.std(ddof=1)),
                                  d_mean=float(xd.mean()), d_std=float(xd.std(ddof=1)))

    with open("research/w4_pairdep.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_pairdep.json")

    rid = ledger.append_row(
        "w4_pairdep",
        {"n_rows": res["n_rows"], "draw_seeds": [SEED_D1, SEED_D2],
         "population": "Pair1 val rows rng(2025) 200k of held",
         "model": "training/w4_pairq1.pt best epoch " + str(res["best_epoch"])},
        "ok",
        metrics={"f_s": res["read1_s"]["f"], "f_s_se": res["read1_s"]["f_se"],
                 "f_dt": res["read1_dt"]["f"], "f_dt_se": res["read1_dt"]["f_se"],
                 "f_th": res["read2_th"]["f"],
                 "rho_dt_human": res["read3_rho_dt"]["human"],
                 "rho_dt_draw": res["read3_rho_dt"]["draw"]},
        artifacts=["research/w4_pairdep.json"],
        notes=f"AMENDMENT 30 exchangeability measurement of q1g0's e1"
              f" conditional. s {res['read1_s']['verdict']},"
              f" dt {res['read1_dt']['verdict']},"
              f" th {res['read2_th']['verdict']}. No contract run, no"
              f" scorer, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
