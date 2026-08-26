"""w4_e1why. AMENDMENT 43, registered in step0_prereg.md before this
file existed.

A42 pinned the remaining event 1 cost to the speed channel. This reads
the pair head itself to say WHY: is the drawn s1 marginal wrong, is
its dependence on event 0 wrong, or is neither and the damage is in
the joint. No decoding, no trajectory generation, no contract scoring.
Diagnostic only, never a training signal, no selection.
"""
import json
import sys

import numpy as np
import torch

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp                        # noqa: E402
from models.event_stream_polar import TICK_CLASS, S_PAD_CLASS       # noqa: E402
from w4_pairq import Pair1, P1_PATH, pair_tokens, splits            # noqa: E402
import ledger                                                       # noqa: E402

SEEDS = [40, 41, 42, 43, 44, 45]
N = 2000
KMAX = 4
CTRL_SEED = 4301
POOL_SEED = 4302
N_POOL = 200_000
S_BINS = 12          # quantile bins of s0 for the MI and the curve


def mi_nats(x, y, bx, by):
    """Plug in mutual information of two binned integer vectors."""
    j = np.zeros((bx, by))
    np.add.at(j, (x, y), 1.0)
    j /= j.sum()
    px, py = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
    nz = j > 0
    return float((j[nz] * np.log(j[nz] / (px @ py)[nz])).sum())


def tv(a, b, nb):
    ha = np.bincount(a, minlength=nb).astype(np.float64)
    hb = np.bincount(b, minlength=nb).astype(np.float64)
    return float(0.5 * np.abs(ha / ha.sum() - hb / hb.sum()).sum())


def main():
    dev = esp._DEVICE
    lengths, trained, held = splits()
    elig = held[lengths[held] > KMAX]
    s0a, th0a, dt0a, s1a, th1a, dt1a = pair_tokens()
    cond_all = np.load("training/events_cond.npy")[:, :4].astype(np.float32)

    pk = torch.load(P1_PATH, map_location=dev, weights_only=False)
    pair = Pair1(**pk["config"]).to(dev).eval()
    pair.load_state_dict(pk["model_state_dict"])
    print(f"  pair head {P1_PATH}, {sum(p.numel() for p in pair.parameters())/1e6:.2f}M"
          f" params, device {dev}", flush=True)

    # s0 quantile bin edges from a disjoint pool, so binning never sees the
    # rows it is applied to.
    pool = np.sort(np.random.default_rng(POOL_SEED).choice(elig, N_POOL, replace=False))
    edges = np.unique(np.quantile(s0a[pool], np.linspace(0, 1, S_BINS + 1)[1:-1]))
    nbx = len(edges) + 1
    y_edges = np.unique(np.quantile(s1a[pool], np.linspace(0, 1, S_BINS + 1)[1:-1]))
    nby = len(y_edges) + 1
    binx = lambda v: np.digitize(v, edges)
    biny = lambda v: np.digitize(v, y_edges)
    mi_pool = mi_nats(binx(s0a[pool]), biny(s1a[pool]), nbx, nby)
    print(f"  disjoint pool of {N_POOL:,} rows: empirical I(s0 ; s1)"
          f" {mi_pool:.4f} nats over {nbx} by {nby} quantile bins", flush=True)

    res = {"seeds": SEEDS, "n": N, "mi_pool_nats": mi_pool,
           "s_bins": [int(nbx), int(nby)], "n_pool": N_POOL}
    per = {k: {} for k in ("tv_model", "tv_ctrl", "mi_real", "mi_drawn",
                           "mi_ctrl", "ce_real", "ce_shuf", "mean_real",
                           "mean_drawn", "sd_real", "sd_drawn")}
    curves = {}
    for s in SEEDS:
        pick = np.sort(np.random.default_rng(1000 + s).choice(elig, N, replace=False))
        banned = set(pick.tolist())
        ctrl_pool = np.array([i for i in elig if i not in banned])
        ctrl = np.sort(np.random.default_rng(CTRL_SEED + s)
                       .choice(ctrl_pool, N, replace=False))

        c = torch.from_numpy(cond_all[pick]).to(dev)
        S0 = torch.from_numpy(s0a[pick].astype(np.int64)).to(dev)
        TH0 = torch.from_numpy(th0a[pick].astype(np.int64)).to(dev)
        DT0 = torch.from_numpy(dt0a[pick].astype(np.int64)).to(dev)
        S1 = torch.from_numpy(s1a[pick].astype(np.int64)).to(dev)

        torch.manual_seed(s * 100003 + 13)
        with torch.no_grad():
            drawn = pair.sample(c, S0, TH0, DT0, 1.0, 1.0, 1.0)[0].cpu().numpy()

            h = pair.trunk(c, S0, TH0, DT0)
            lp = torch.log_softmax(pair.s_head(h), -1)
            ce_real = float(-lp.gather(1, S1[:, None]).mean())

            g = np.random.default_rng(CTRL_SEED * 7 + s).permutation(N)
            gi = torch.from_numpy(g).to(dev)
            hs = pair.trunk(c, S0[gi], TH0[gi], DT0[gi])
            lps = torch.log_softmax(pair.s_head(hs), -1)
            ce_shuf = float(-lps.gather(1, S1[:, None]).mean())

        real, cr = s1a[pick], s1a[ctrl]
        per["tv_model"][s] = tv(real, drawn, S_PAD_CLASS + 1)
        per["tv_ctrl"][s] = tv(real, cr, S_PAD_CLASS + 1)
        per["mi_real"][s] = mi_nats(binx(s0a[pick]), biny(real), nbx, nby)
        per["mi_drawn"][s] = mi_nats(binx(s0a[pick]), biny(drawn), nbx, nby)
        per["mi_ctrl"][s] = mi_nats(binx(s0a[ctrl]), biny(cr), nbx, nby)
        per["ce_real"][s], per["ce_shuf"][s] = ce_real, ce_shuf
        mo = (real > TICK_CLASS) & (real < S_PAD_CLASS)
        md = (drawn > TICK_CLASS) & (drawn < S_PAD_CLASS)
        per["mean_real"][s], per["sd_real"][s] = float(real[mo].mean()), float(real[mo].std())
        per["mean_drawn"][s], per["sd_drawn"][s] = float(drawn[md].mean()), float(drawn[md].std())

        b = binx(s0a[pick])
        curves[s] = [(float(real[b == k].mean()) if (b == k).sum() else float("nan"),
                      float(drawn[b == k].mean()) if (b == k).sum() else float("nan"))
                     for k in range(nbx)]
        print(f"  seed {s}: TV model {per['tv_model'][s]:.4f} control"
              f" {per['tv_ctrl'][s]:.4f}   MI real {per['mi_real'][s]:.4f}"
              f" drawn {per['mi_drawn'][s]:.4f} control {per['mi_ctrl'][s]:.4f}"
              f"   CE real {ce_real:.4f} shuffled {ce_shuf:.4f}", flush=True)

    res["per_seed"] = {k: {str(s): v for s, v in d.items()} for k, d in per.items()}

    def stat(vals):
        a = np.array([vals[s] for s in SEEDS])
        return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))

    tm, tms = stat(per["tv_model"])
    tc, tcs = stat(per["tv_ctrl"])
    flag1 = tm > tc + 3 * tcs
    print(f"\n  READ 1 (MARGINAL): TV against real {tm:.4f} se {tms:.4f},"
          f" real versus real control {tc:.4f} se {tcs:.4f}")
    print(f"  bar is control plus 3 se = {tc + 3 * tcs:.4f}."
          f"  {'FLAGGED' if flag1 else 'NOT FLAGGED'}")
    print(f"  moving class mean real {stat(per['mean_real'])[0]:.2f} drawn"
          f" {stat(per['mean_drawn'])[0]:.2f}, sd real"
          f" {stat(per['sd_real'])[0]:.2f} drawn {stat(per['sd_drawn'])[0]:.2f}")
    res["read1"] = dict(tv_model=tm, tv_model_se=tms, tv_ctrl=tc, tv_ctrl_se=tcs,
                        bar=tc + 3 * tcs, flagged=bool(flag1),
                        mean_real=stat(per["mean_real"])[0],
                        mean_drawn=stat(per["mean_drawn"])[0],
                        sd_real=stat(per["sd_real"])[0],
                        sd_drawn=stat(per["sd_drawn"])[0])

    d = np.array([per["mi_real"][s] - per["mi_drawn"][s] for s in SEEDS])
    dm = float(d.mean()); dse = float(d.std(ddof=1) / np.sqrt(len(d)))
    flag2 = dm > 3 * dse
    print(f"\n  READ 2 (COUPLING): I(s0 ; s1) real {stat(per['mi_real'])[0]:.4f},"
          f" drawn {stat(per['mi_drawn'])[0]:.4f}, real versus real control"
          f" {stat(per['mi_ctrl'])[0]:.4f}")
    print(f"  paired real minus drawn {dm:+.4f} se {dse:.4f}"
          f"  ({dm / dse if dse else float('inf'):+.2f} se)."
          f"  {'FLAGGED' if flag2 else 'NOT FLAGGED'}")
    res["read2"] = dict(mi_real=stat(per["mi_real"])[0],
                        mi_drawn=stat(per["mi_drawn"])[0],
                        mi_ctrl=stat(per["mi_ctrl"])[0],
                        paired_gap=dm, paired_se=dse, flagged=bool(flag2))

    r = np.array([per["ce_shuf"][s] - per["ce_real"][s] for s in SEEDS])
    rm = float(r.mean()); rse = float(r.std(ddof=1) / np.sqrt(len(r)))
    share = rm / mi_pool if mi_pool > 0 else float("nan")
    print(f"\n  READ 3 (ABLATION): CE on the real s1 given the real event 0"
          f" {stat(per['ce_real'])[0]:.4f} nats, given event 0 shuffled"
          f" {stat(per['ce_shuf'])[0]:.4f} nats")
    print(f"  the head gets {rm:.4f} nats se {rse:.4f} out of event 0, against"
          f" an available lower bound of {mi_pool:.4f} nats, a share of {share:.2f}")
    res["read3"] = dict(ce_real=stat(per["ce_real"])[0],
                        ce_shuf=stat(per["ce_shuf"])[0],
                        gain=rm, gain_se=rse, mi_pool=mi_pool, share=share,
                        below_0_05=bool(rm < 0.05), below_half=bool(share < 0.5))

    C = np.array([[curves[s][k] for k in range(nbx)] for s in SEEDS])
    mreal, mdrawn = np.nanmean(C[:, :, 0], 0), np.nanmean(C[:, :, 1], 0)
    dev_ = mdrawn - mreal
    k = int(np.nanargmax(np.abs(dev_)))
    print(f"\n  READ 4 (WHERE): mean s1 class by s0 bin, drawn minus real")
    print("  " + "  ".join(f"b{i}{dev_[i]:+.1f}" for i in range(nbx)))
    print(f"  largest deviation in bin {k} of {nbx} (s0 bins run slow to fast),"
          f" {dev_[k]:+.2f} classes, drawn"
          f" {'too fast' if dev_[k] > 0 else 'too slow'} there")
    res["read4"] = dict(mean_real=[float(x) for x in mreal],
                        mean_drawn=[float(x) for x in mdrawn],
                        deviation=[float(x) for x in dev_],
                        worst_bin=k, worst_dev=float(dev_[k]), n_bins=int(nbx))

    with open("research/w4_e1why.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1why.json")
    print("  diagnostic only, never a training signal, no selection,"
          " the contract scorer was not called")

    rid = ledger.append_row(
        "w4_e1why",
        {"seeds": SEEDS, "n": N, "s_bins": [int(nbx), int(nby)],
         "n_pool": N_POOL, "model": P1_PATH},
        "ok",
        metrics={"tv_model": tm, "tv_ctrl": tc, "mi_gap": dm,
                 "ablation_gain_nats": rm, "mi_pool_nats": mi_pool,
                 "ablation_share": share},
        artifacts=["research/w4_e1why.json"],
        notes=f"AMENDMENT 43 pair head speed diagnostic. MARGINAL"
              f" {'FLAGGED' if flag1 else 'not flagged'}, COUPLING"
              f" {'FLAGGED' if flag2 else 'not flagged'}. Head takes {rm:.4f}"
              f" nats out of event 0 against a {mi_pool:.4f} nat lower bound"
              f" (share {share:.2f}). Diagnostic only, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
