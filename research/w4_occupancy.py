"""Do the model's own states differ from real states, where, and in what?
Diagnostic only, never a training signal, never selection.

The likelihood side is exhausted as a locator. `w4_pit` says the served
conditional has no global dispersion error at real states and about 0.02
to 0.05 nats per token of state dependent error. `w4_ownent` says the
model's own uncertainty climbs steadily along its own rollouts at
temperature one. Neither can say WHERE the chain goes wrong, because off
the real manifold there is no ground truth conditional to compare to.

This asks the question that needs no ground truth. Take the state the
model conditions on, at every step, on real trajectories and on its own
rollouts, and run the contract scorer's own instrument on those states:
a random forest, out of bag, real steps against generated steps. A
disjoint real against real split is the floor. Split by position to see
whether the mismatch is there from the first step or grows, and read the
forest's importances to see which coordinate of the state carries it.

ONE ROW PER TRAJECTORY, always. Every trajectory contributes a single
step to any forest, chosen at random inside the position band being read.
With many steps per trajectory the forest does not learn a distribution
of states at all: the static condition is nearly unique per trajectory
and constant down its rows, so it memorises which trajectory a row came
from, and since a trajectory is wholly on one side that classifies
perfectly. Measured on the first build, real against real read 0.9915
that way. One row per trajectory makes that impossible.

THE GENERATED SIDE IS CONDITIONED ON THE OTHER REAL POOL. The first build
gave the rollouts the same conditions as the real rows they were scored
against. That is fatal here, because the two condition numbers are in the
feature vector and are near unique per trajectory, so every condition value
appeared exactly twice, once on each side. Out of bag a row's own copy is
held out while its twin from the other class is not, so the forest votes for
the wrong class and the AUC reads BELOW 0.5. Measured on a control with the
dynamic part held constant, as it nearly is at the first step, that reads
0.0000 against an honest 0.5097; with two live dynamic columns it reads
0.4360. Conditioning the rollouts on the disjoint pool removes it and leaves
the comparison exactly parallel to the floor.

The state is what the trunk actually sees, written in observable terms:
the last four events (speed, signed turn, inter event time), the tick
count in the last eight, the progress made so far against the commanded
distance, the step index, and the two static condition numbers. No
contract feature is used, nothing is resampled to 125 Hz, and no path is
rendered, so this cannot be confused with the scorer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "autoloop"))
from models.event_ar import (DT_MAX_MS, EventARModel, class_to_dt_ms,  # noqa: E402
                             dt_ms_to_class)
from models.event_stream_polar import (S_PAD_CLASS, TH_BINS,  # noqa: E402
                                       TH_NULL_CLASS, class_to_speed,
                                       dth_lattice_to_class, s2_to_class)
from w4_firsthead import FirstHead, Q_PATH  # noqa: E402
import ledger  # noqa: E402

TRAIN_PICK_SEED = 123
N_TRAIN = 1_500_000
MAX_T = 256
K = 4                       # events of history in the state
POS = [(0, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 256)]
NAMES = ([f"logspeed_{i}" for i in range(K)] + [f"turn_{i}" for i in range(K)]
         + [f"dt_{i}" for i in range(K)]
         + ["tick8", "progress", "step", "log_dist", "log_dur"])


def to_arrays(s_cls, th_cls, dt_cls):
    sp = class_to_speed(torch.as_tensor(np.minimum(s_cls, S_PAD_CLASS - 1))).numpy()
    sp = np.where(s_cls == 0, 0.0, sp)
    turn = np.where(th_cls < TH_BINS,
                    ((th_cls + TH_BINS // 2) % TH_BINS - TH_BINS // 2) / (TH_BINS / 2.0), 0.0)
    dt = class_to_dt_ms(torch.as_tensor(np.minimum(dt_cls, DT_MAX_MS))).numpy()
    return sp.astype(np.float32), turn.astype(np.float32), dt.astype(np.float32)


def states(s_cls, th_cls, dt_cls, cond, max_steps):
    """Every alive step as a row, with the trajectory it came from; the
    callers below then take at most one row per trajectory."""
    sp, turn, dt = to_arrays(s_cls, th_cls, dt_cls)
    alive = s_cls < S_PAD_CLASS
    n = alive.sum(1)
    B, T = s_cls.shape
    rows, pos, who = [], [], []
    cum = np.cumsum(sp, 1)
    for b in range(B):
        m = min(int(n[b]), max_steps)
        for t in range(m):
            h = [np.log1p(sp[b, t - 1 - i]) if t - 1 - i >= 0 else -1.0 for i in range(K)]
            g = [turn[b, t - 1 - i] if t - 1 - i >= 0 else 0.0 for i in range(K)]
            d = [dt[b, t - 1 - i] if t - 1 - i >= 0 else -1.0 for i in range(K)]
            lo = max(0, t - 8)
            tick8 = float((s_cls[b, lo:t] == 0).sum()) if t > lo else 0.0
            prog = float(cum[b, t - 1] / np.exp(cond[b, 0])) if t > 0 else 0.0
            rows.append(h + g + d + [tick8, prog, float(t),
                                     float(cond[b, 0]), float(cond[b, 1])])
            pos.append(t); who.append(b)
    return (np.asarray(rows, np.float32), np.asarray(pos, np.int64),
            np.asarray(who, np.int64))


def one_per_traj(X, pos, who, lo, hi, rng):
    """One row per trajectory, drawn at random among that trajectory's steps
    inside [lo, hi). Trajectories with no step in the band contribute none."""
    m = (pos >= lo) & (pos < hi)
    Xm, wm = X[m], who[m]
    order = rng.permutation(len(Xm))
    Xm, wm = Xm[order], wm[order]
    _, first = np.unique(wm, return_index=True)
    return Xm[first]


def rare_rates(s_cls, th_cls, dt_cls):
    """Per trajectory rates of the events a person makes and a model may not.
    A PAUSE is a run of no motion events whose total time is at least 40 ms,
    the definition w4_submove fixed (shorter still runs are the lattice, not
    a decision). A REVERSAL is a heading change past a quarter turn."""
    sp, turn, dt = to_arrays(s_cls, th_cls, dt_cls)
    alive = s_cls < S_PAD_CLASS
    n_traj = len(s_cls)
    pauses = np.zeros(n_traj); stills = np.zeros(n_traj); rev = np.zeros(n_traj)
    motion = np.zeros(n_traj); events = np.zeros(n_traj)
    for b in range(n_traj):
        m = int(alive[b].sum())
        tick = (s_cls[b, :m] == 0)
        events[b] = m
        mo = ~tick
        motion[b] = mo.sum()
        rev[b] = (np.abs(turn[b, :m][mo]) > 0.5).sum()
        i = 0
        while i < m:
            if tick[i]:
                j = i
                while j < m and tick[j]:
                    j += 1
                stills[b] += 1
                if dt[b, i:j].sum() >= 40.0:
                    pauses[b] += 1
                i = j
            else:
                i += 1
    return dict(events=round(float(events.mean()), 3),
                motion=round(float(motion.mean()), 3),
                still_runs=round(float(stills.mean()), 4),
                pauses_40ms=round(float(pauses.mean()), 4),
                pause_share_rows=round(float((pauses > 0).mean()), 4),
                reversals=round(float(rev.mean()), 4),
                reversal_share=round(float(rev.sum() / max(motion.sum(), 1)), 5))


def oob(X0, X1, seed, imp=False):
    X = np.concatenate([X0, X1]); y = np.r_[np.zeros(len(X0)), np.ones(len(X1))]
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, oob_score=True,
                                n_jobs=12, random_state=seed)
    rf.fit(X, y)
    a = float(roc_auc_score(y, rf.oob_decision_function_[:, 1]))
    return (a, rf.feature_importances_) if imp else a


@torch.no_grad()
def rollout(model, q, cond, temps, dev, seed):
    torch.manual_seed(seed)
    s, th, dt = model.sample(cond.to(dev), temperature=temps[0], th_temperature=temps[1],
                             dt_temperature=temps[2], force=None)
    n = s.shape[1]
    S = np.full((cond.shape[0], MAX_T), S_PAD_CLASS, np.int64)
    H = np.full((cond.shape[0], MAX_T), TH_NULL_CLASS, np.int64)
    D = np.zeros((cond.shape[0], MAX_T), np.int64)
    S[:, :n], H[:, :n], D[:, :n] = s.cpu().numpy(), th.cpu().numpy(), dt.cpu().numpy()
    return S, H, D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_hm_mlp.pt")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/home/aaronadmin/w4_arms/occupancy.json")
    a = ap.parse_args()
    t0 = time.time()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    lengths = np.load("training/events_len.npy")
    cond_all = np.load("training/events_cond.npy")
    N = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(N, min(N, N_TRAIN), replace=False))
    held = np.setdiff1d(np.arange(N), trained)
    elig = held[lengths[held] >= 5]
    pick = np.random.default_rng(1000 + a.seed).choice(elig, 2 * a.n, replace=False)
    A, Bp = np.sort(pick[:a.n]), np.sort(pick[a.n:])       # disjoint real pools
    s2 = np.load("training/events_s2.npy", mmap_mode="r")
    dth = np.load("training/events_dth.npy", mmap_mode="r")
    dtn = np.load("training/events_dt.npy", mmap_mode="r")

    def real_tokens(idx):
        n = np.minimum(lengths[idx], MAX_T).astype(int)
        S = np.full((len(idx), MAX_T), S_PAD_CLASS, np.int64)
        H = np.full((len(idx), MAX_T), TH_NULL_CLASS, np.int64)
        D = np.zeros((len(idx), MAX_T), np.int64)
        sc = s2_to_class(torch.as_tensor(np.asarray(s2[idx], np.int64))).numpy()
        tc = np.where(np.asarray(s2[idx]) > 0,
                      dth_lattice_to_class(torch.as_tensor(np.asarray(dth[idx], np.int64))).numpy(),
                      TH_NULL_CLASS)
        dc = dt_ms_to_class(torch.as_tensor(np.asarray(dtn[idx], np.float64))).numpy()
        for i in range(len(idx)):
            S[i, :n[i]] = sc[i, :n[i]]; H[i, :n[i]] = tc[i, :n[i]]
            D[i, :n[i]] = dc[i, :n[i]].clip(0, DT_MAX_MS)
        return S, H, D

    ck = torch.load(f"training/{a.ckpt}", map_location=dev, weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    cG = torch.as_tensor(cond_all[Bp][:, :4].astype(np.float32))
    gen = {"served": [], "pure": []}
    for c0 in range(0, a.n, a.batch):
        cb = cG[c0:c0 + a.batch]
        for arm, temps in (("served", (0.95, 0.90, 1.00)), ("pure", (1.0, 1.0, 1.0))):
            gen[arm].append(rollout(model, None, cb, temps, dev, a.seed * 100003 + c0))
        print(f"    generated {min(a.n, c0 + a.batch)}/{a.n} {time.time() - t0:.0f}s", flush=True)
    G = {arm: [np.concatenate([x[k] for x in v]) for k in range(3)] for arm, v in gen.items()}

    XA, pA, wA = states(*real_tokens(A), cond_all[A], a.max_steps)
    XB, pB, wB = states(*real_tokens(Bp), cond_all[Bp], a.max_steps)
    XS, pS, wS = states(*G["served"], cond_all[Bp], a.max_steps)
    XP, pP, wP = states(*G["pure"], cond_all[Bp], a.max_steps)
    print(f"  alive steps real {len(XA)} realB {len(XB)} served {len(XS)} pure {len(XP)}",
          flush=True)
    rng = np.random.default_rng(a.seed * 7919 + 11)
    out = dict(args=vars(a), alive_steps=dict(realA=len(XA), realB=len(XB),
                                              served=len(XS), pure=len(XP)))

    def band(lo, hi):
        return (one_per_traj(XA, pA, wA, lo, hi, rng), one_per_traj(XB, pB, wB, lo, hi, rng),
                one_per_traj(XS, pS, wS, lo, hi, rng), one_per_traj(XP, pP, wP, lo, hi, rng))

    aA, aB, aS, aP = band(0, MAX_T)
    # The floor's importances are taken too, and they are the only way to
    # read the served ones. `log_dist` and `log_dur` are raw condition
    # columns, and the real side is pool A while the generated side carries
    # pool B's conditions, so those two columns separate the pools by
    # construction and top the served list whether or not the model is wrong
    # about anything. The floor is realA against realB and carries exactly the
    # same condition mismatch, so the excess of a served importance over the
    # floor's is the part that is about the model.
    auc_f, imp_f = oob(aA, aB, a.seed, imp=True)
    auc_s, imp_s = oob(aA, aS, a.seed, imp=True)
    auc_p, imp_p = oob(aA, aP, a.seed, imp=True)
    out["rows_per_side"] = dict(realA=len(aA), realB=len(aB), served=len(aS), pure=len(aP))
    out["auc"] = dict(floor_real_vs_real=round(auc_f, 4), served=round(auc_s, 4),
                      pure=round(auc_p, 4))
    out["importance_served"] = {n: round(float(v), 4) for n, v in
                                sorted(zip(NAMES, imp_s), key=lambda x: -x[1])}
    out["importance_floor"] = {n: round(float(v), 4) for n, v in zip(NAMES, imp_f)}
    out["importance_pure"] = {n: round(float(v), 4) for n, v in zip(NAMES, imp_p)}
    out["importance_excess_served"] = {
        n: round(float(v), 4) for n, v in
        sorted(zip(NAMES, imp_s - imp_f), key=lambda x: -x[1])}
    out["importance_excess_pure"] = {
        n: round(float(v), 4) for n, v in
        sorted(zip(NAMES, imp_p - imp_f), key=lambda x: -x[1])}
    by = {}
    for lo, hi in POS:
        bA, bB, bS, bP = band(lo, hi)
        if min(len(bA), len(bB), len(bS), len(bP)) < 300:
            continue
        by[f"{lo}-{hi - 1}"] = dict(
            n=int(len(bA)), floor=round(oob(bA, bB, a.seed), 4),
            served=round(oob(bA, bS, a.seed), 4), pure=round(oob(bA, bP, a.seed), 4))
    out["by_position"] = by
    out["rare"] = dict(
        real=rare_rates(*real_tokens(A)), realB=rare_rates(*real_tokens(Bp)),
        served=rare_rates(*G["served"]), pure=rare_rates(*G["pure"]))
    print(f"\n  STATE OCCUPANCY, {a.ckpt}, {a.n} rows a side, one step per trajectory")
    print(f"    real vs real floor {auc_f:.4f}   served {auc_s:.4f}   pure {auc_p:.4f}"
          f"   ({len(aA)} rows a side)")
    print("    by position (floor / served / pure)")
    for k, v in by.items():
        print(f"      {k:8s} n {v['n']:6d}   {v['floor']:.4f} / {v['served']:.4f} / {v['pure']:.4f}")
    print("    per trajectory rates (real / realB / served / pure)")
    for k in out["rare"]["real"]:
        r = out["rare"]
        print(f"      {k:18s} {r['real'][k]:8.4f} / {r['realB'][k]:8.4f} / "
              f"{r['served'][k]:8.4f} / {r['pure'][k]:8.4f}")
    print("    importances, served against real, top ten")
    print(f"      {'feature':14s} {'served':>8}{'floor':>8}{'excess':>8}"
          f"{'pure ex':>9}")
    for n, v in list(out["importance_served"].items())[:10]:
        print(f"      {n:14s} {v:8.4f}{out['importance_floor'][n]:8.4f}"
              f"{v - out['importance_floor'][n]:8.4f}"
              f"{out['importance_excess_pure'][n]:9.4f}")
    print("    largest excess over the floor, served, top ten")
    for n, v in list(out["importance_excess_served"].items())[:10]:
        print(f"      {n:14s} {v:+.4f}")
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"  wrote {a.out}, {time.time() - t0:.0f}s")
    rid = ledger.append_row(
        "w4_occupancy",
        {"ckpt": a.ckpt, "n": a.n, "seed": a.seed, "max_steps": a.max_steps},
        "ok",
        metrics={"floor": round(auc_f, 4), "served": round(auc_s, 4),
                 "pure": round(auc_p, 4),
                 "reversals_real": out["rare"]["real"]["reversals"],
                 "reversals_served": out["rare"]["served"]["reversals"]},
        artifacts=[a.out],
        notes="DIAGNOSTIC, never a training signal and never selection. State"
              " occupancy, one row per trajectory, generated side conditioned"
              " on the DISJOINT real pool so no condition value appears on"
              " both sides of a forest.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid['run_id']}")


if __name__ == "__main__":
    main()
