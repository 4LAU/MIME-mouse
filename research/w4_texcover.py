"""Does the model's TOKEN HISTORY leave the human manifold?

Registered in /home/aaronadmin/w4_arms/texcover_prereg.md. Read that first, and
in particular read the two corrections at the top of it, because this file
exists only because the geometric half of this question was already answered
and I had recorded it as unanswered.

WHAT IS ALREADY KNOWN. HANDOFF's "THE RESULT. DIFFUSE" measured the nearest
human state to a model state in the six dimensional `prefix_state` and got a
ratio of 1.032 against the human to human control. Model states are on the
human manifold GEOMETRICALLY. That section then concluded, by elimination
rather than by measurement, that the whole 0.32 nat excess is carried by the
only other thing the trunk reads, the TOKEN HISTORY, and it never put the token
history into the distance.

WHAT THIS MEASURES. The same statistic with the recent token history appended
to the key, swept over history depth k in {0, 1, 2, 4, 8}. k = 0 reproduces the
recorded number. Each event adds four coordinates, log1p(speed), cos(dtheta),
sin(dtheta), dt in milliseconds.

THE STATISTIC. R = median nearest reference distance for MODEL queries divided
by median nearest reference distance for HELD OUT HUMAN queries, after
whitening by the human reference covariance so a unit is a unit of human
variation.

THE THREE CONTROLS, each fixing a way this measurement has already failed once:

  leakage    reference trajectories and human query trajectories are drawn from
             DISJOINT index sets, so a human query's own adjacent position
             cannot be its own nearest neighbour. The earlier version of this
             read 2.949 through exactly that hole and was caught only because
             the number was too good. Here there is no exclusion logic to have
             a bug in.
  depth      generated sequences choose their own lengths, so the two query
             pools can differ in their mix over fractional position and that
             alone moves distances. R is reported per fractional position
             quintile as well as pooled, and a pooled R not reproduced within
             its quintiles is a confound and not a result.
  seed       two rollout seeds, and every claim must hold on both.

POSITIONS. Only positions with index >= KMAX are used, for every k including
k = 0, so the position population is identical across the sweep and no k needs
a padded history. This drops the opening of every trajectory, which is the part
most pinned by the conditioning, so it makes the test harder rather than
easier.

CPU only. No GPU, no training, no model load. Reads the rollout dumps that
`w4_texcover_gen.py` wrote.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.neighbors import NearestNeighbors

# Brute force nearest neighbour over two million references chunks its query
# rows to fit working_memory. At the 1 GB default that is 67 rows per chunk,
# which pays the partial sort's fixed cost three hundred times per pool. Wider
# chunks are the same arithmetic in fewer passes. Held to 2 GB rather than
# wider because this VM is capped at 14 GB and the record contains three
# separate WSL deaths caused by a resident array nobody had budgeted for.
sklearn.set_config(working_memory=2048)

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.event_ar import DT_MAX_MS, prefix_state  # noqa: E402
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TH_NULL_CLASS, class_to_dtheta, class_to_speed,
    dth_lattice_to_class, s2_to_class,
)

CORPUS = Path("/home/aaronadmin/mts_data")
TBUF = 256          # the trunk's positional buffer, so prefix_state's
                    # index/T coordinate matches training and sampling alike
KMAX = 8
KS = (0, 1, 2, 4, 8)


def _hist_feats(s_cls, th_cls, dt_cls):
    """Per event physical values the history key is built from, (B, T, 4).

    Speed is logged because the speed vocabulary is log spaced, so that is the
    scale the model's own classes are uniform in. dtheta is split into cos and
    sin so the wrap at pi is not a discontinuity in the distance. A tick
    decodes to speed 0 and dtheta 0 and is therefore already distinguishable
    from every motion event, whose minimum speed is 1, with no indicator
    coordinate needed.
    """
    sp = class_to_speed(s_cls.clamp(max=S_PAD_CLASS))
    dth = class_to_dtheta(th_cls.clamp(max=TH_NULL_CLASS))
    dt = dt_cls.float().clamp(0, DT_MAX_MS)
    return torch.stack([torch.log1p(sp), torch.cos(dth), torch.sin(dth), dt],
                       dim=-1)


def _keys(s_cls, th_cls, dt_cls, cond, lengths, rng, cap):
    """(N, 6 + 4*KMAX) keys plus fractional position, for valid positions.

    Column layout: the six prefix_state coordinates, then the most recent event
    (lag 1), then lag 2, and so on out to lag KMAX. So the key for depth k is
    exactly the first 6 + 4*k columns, and the sweep is a slice.
    """
    B = s_cls.shape[0]
    st = prefix_state(s_cls, th_cls, dt_cls, cond)          # (B, T, 6)
    hf = _hist_feats(s_cls, th_cls, dt_cls)                 # (B, T, 4)

    lags = []
    for lag in range(1, KMAX + 1):
        z = torch.zeros(B, lag, 4)
        lags.append(torch.cat([z, hf[:, :-lag]], dim=1))
    key = torch.cat([st] + lags, dim=-1).numpy()            # (B, T, 6 + 4*KMAX)

    T = key.shape[1]
    idx = np.arange(T)[None, :]
    L = lengths[:, None]
    ok = (idx >= KMAX) & (idx < L)
    frac = np.where(L > 0, idx / np.maximum(L, 1), 0.0)

    flat = key[ok]
    fr = np.broadcast_to(frac, ok.shape)[ok]
    if cap is not None and len(flat) > cap:
        sel = rng.choice(len(flat), cap, replace=False)
        flat, fr = flat[sel], fr[sel]
    return flat.astype(np.float32), fr.astype(np.float32)


def human_keys(traj_ids, cap, rng, chunk=4000):
    """Corpus rows -> keys, converted exactly the way ARDataset converts them."""
    s2m = np.load(CORPUS / "events_s2.npy", mmap_mode="r")
    dthm = np.load(CORPUS / "events_dth.npy", mmap_mode="r")
    dtm = np.load(CORPUS / "events_dt.npy", mmap_mode="r")
    Lall = np.load(CORPUS / "events_len.npy")
    Call = np.load(CORPUS / "events_cond.npy")

    traj_ids = np.sort(traj_ids)
    out_k, out_f, got = [], [], 0
    for c0 in range(0, len(traj_ids), chunk):
        ids = traj_ids[c0:c0 + chunk]
        L = np.minimum(Lall[ids], TBUF)
        s2 = torch.from_numpy(np.asarray(s2m[ids, :TBUF], dtype=np.int64))
        dth = torch.from_numpy(np.asarray(dthm[ids, :TBUF], dtype=np.int64))
        dt_ms = torch.from_numpy(np.asarray(dtm[ids, :TBUF], dtype=np.float32))

        valid = torch.from_numpy(
            (np.arange(TBUF)[None, :] < L[:, None]))
        s_cls = torch.where(valid, s2_to_class(s2),
                            torch.full_like(s2, S_PAD_CLASS))
        th_cls = torch.where(valid & (s2 > 0), dth_lattice_to_class(dth),
                             torch.full_like(dth, TH_NULL_CLASS))
        dt_cls = torch.where(valid, torch.round(dt_ms).long().clamp(0, DT_MAX_MS),
                             torch.zeros_like(s2))
        cond = torch.from_numpy(np.asarray(Call[ids], dtype=np.float32))

        k, f = _keys(s_cls, th_cls, dt_cls, cond, L, rng, None)
        out_k.append(k)
        out_f.append(f)
        got += len(k)
        if got >= cap * 1.15:
            break
    K = np.concatenate(out_k)
    F = np.concatenate(out_f)
    if len(K) > cap:
        sel = rng.choice(len(K), cap, replace=False)
        K, F = K[sel], F[sel]
    return K, F


def model_keys(path, cap, rng):
    z = np.load(path)
    s = torch.from_numpy(z["s"].astype(np.int64))
    th = torch.from_numpy(z["th"].astype(np.int64))
    dt = torch.from_numpy(z["dt"].astype(np.int64))
    cond = torch.from_numpy(z["cond"].astype(np.float32))
    pad = (s >= S_PAD_CLASS).numpy()
    T = s.shape[1]
    L = np.where(pad.any(1), pad.argmax(1), T)
    # PAD positions carry no event, and dt PAD would poison the elapsed clock,
    # so blank the tail before any state is computed.
    valid = torch.from_numpy(np.arange(T)[None, :] < L[:, None])
    dt = torch.where(valid, dt.clamp(0, DT_MAX_MS), torch.zeros_like(dt))
    return _keys(s, th, dt, cond, L, rng, cap)


def whiten_fit(ref):
    mu = ref.mean(0)
    X = ref - mu
    C = (X.T @ X) / max(len(X) - 1, 1)
    w, V = np.linalg.eigh(C.astype(np.float64))
    w = np.maximum(w, 1e-8 * float(w.max()))
    W = (V / np.sqrt(w)).astype(np.float32)
    return mu, W


def rand_ref_median(ref_w, qry_w, rng, m=1000):
    """Median distance from a query to a RANDOM reference point.

    The concentration diagnostic registered in the prereg addendum. Nearest
    neighbour distances concentrate as dimension rises, so R is squeezed toward
    1 by the geometry alone. NN / D_rand says how much resolution is left: at
    1.0 the nearest reference is no nearer than a random one and the ratio is
    measuring nothing. The addendum's binding rule is that any k whose human
    control exceeds 0.90 here is UNINFORMATIVE and may not be read in either
    direction.
    """
    sel = rng.choice(len(ref_w), min(m, len(ref_w)), replace=False)
    R = ref_w[sel]
    q2 = (qry_w * qry_w).sum(1)[:, None]
    r2 = (R * R).sum(1)[None, :]
    d = np.sqrt(np.maximum(q2 + r2 - 2.0 * (qry_w @ R.T), 0.0))
    return float(np.median(d))


def overlap_p(d_model, d_human):
    """P(D_model > D_human) for independent draws, ties counted half.

    Scale free, so unlike R it does not inherit the compression that rising
    dimension imposes on every distance ratio. 0.5 means the two nearest
    neighbour distance distributions are indistinguishable.
    """
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(d_model, d_human, alternative="two-sided").statistic
    return float(u / (len(d_model) * len(d_human)))


NRANK = 32          # DESCRIPTIVE ONLY, carries no registered bar. Ranks beyond
                    # the first say how many human neighbours a lookup expert
                    # would actually find at a model visited state, which is
                    # the first engineering question of the arm this screens
                    # for. It is reported so the follow up is priced; it is not
                    # a result and no prediction is read off it.


def nn_fit(ref_w):
    """One index per key width, queried by every pool, so the model and the
    human control are answered by the identical reference structure."""
    return NearestNeighbors(n_neighbors=NRANK, algorithm="brute",
                            metric="euclidean", n_jobs=-1).fit(ref_w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="*", default=[])
    ap.add_argument("--n-ref", type=int, default=2_000_000)
    ap.add_argument("--n-qry", type=int, default=20_000)
    ap.add_argument("--ref-traj", type=int, default=90_000)
    ap.add_argument("--hq-traj", type=int, default=4_000)
    ap.add_argument("--null-traj", type=int, default=4_000,
                    help="THE CALIBRATION ARM. A third disjoint set of human "
                         "trajectories entered as if it were a model pool. It "
                         "must read R near 1.000 and P near 0.500 at every k, "
                         "and if it does not the instrument is broken and no "
                         "model number may be read. Exactly the role c_vs_c "
                         "plays for the contract scorer.")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="research/w4_texcover_results.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n_traj = len(np.load(CORPUS / "events_len.npy"))
    perm = rng.permutation(n_traj)
    a, b = args.ref_traj, args.ref_traj + args.hq_traj
    ref_ids = perm[:a]
    hq_ids = perm[a:b]
    nl_ids = perm[b:b + args.null_traj]
    assert len(np.intersect1d(ref_ids, hq_ids)) == 0
    assert len(np.intersect1d(ref_ids, nl_ids)) == 0
    assert len(np.intersect1d(hq_ids, nl_ids)) == 0

    print(f"  corpus {n_traj} trajectories, reference from {len(ref_ids)}, "
          f"human control from {len(hq_ids)}, null arm from {len(nl_ids)}, "
          f"all three disjoint", flush=True)

    REF, _ = human_keys(ref_ids, args.n_ref, rng)
    HQ, HQF = human_keys(hq_ids, args.n_qry, rng)
    print(f"  reference {REF.shape}   human query {HQ.shape}", flush=True)

    pools = {"human": (HQ, HQF)}
    if args.null_traj:
        pools["model_NULL"] = human_keys(nl_ids, args.n_qry, rng)
    for sp in args.streams:
        tag = Path(sp).stem.split("_")[-1]
        pools[f"model_{tag}"] = model_keys(sp, args.n_qry, rng)
        print(f"  {tag} query {pools['model_' + tag][0].shape}", flush=True)

    res = {}
    for k in KS:
        d = 6 + 4 * k
        mu, W = whiten_fit(REF[:, :d])
        rw = ((REF[:, :d] - mu) @ W).astype(np.float32)
        index = nn_fit(rw)
        row, dists = {}, {}
        for name, (Q, F) in pools.items():
            qw = ((Q[:, :d] - mu) @ W).astype(np.float32)
            dk = index.kneighbors(qw, return_distance=True)[0]
            dist = dk[:, 0]
            dists[name] = dist
            qs = np.quantile(F, [0.2, 0.4, 0.6, 0.8])
            bins = np.digitize(F, qs)
            row[name] = dict(
                median=float(np.median(dist)),
                q=[float(np.median(dist[bins == b])) for b in range(5)],
                d_rand=rand_ref_median(rw, qw, rng),
                rank=[float(np.median(dk[:, j - 1])) for j in (1, 8, 32)],
                n=int(len(dist)))
            row[name]["resolution"] = row[name]["median"] / row[name]["d_rand"]
            print(f"    k={k:<2} {name:<10} median {row[name]['median']:.4f}"
                  f"  nn/rand {row[name]['resolution']:.3f}", flush=True)
        for name in row:
            if name != "human":
                row[name]["P"] = overlap_p(dists[name], dists["human"])
        row["informative"] = bool(row["human"]["resolution"] <= 0.90)
        res[f"k{k}"] = row

    mtags = [n for n in pools if n.startswith("model_")]
    print(f"\n  {'k':>3}{'dim':>5}{'human':>9}{'nn/rand':>9}{'info':>6}",
          end="")
    for m in mtags:
        print(f"{m:>11}{'R':>8}{'P':>8}", end="")
    print()
    for k in KS:
        r = res[f"k{k}"]
        print(f"  {k:>3}{6 + 4 * k:>5}{r['human']['median']:>9.4f}"
              f"{r['human']['resolution']:>9.3f}"
              f"{('yes' if r['informative'] else 'NO'):>6}", end="")
        for m in mtags:
            R = r[m]["median"] / r["human"]["median"]
            r[m]["R"] = R
            print(f"{r[m]['median']:>11.4f}{R:>8.3f}{r[m]['P']:>8.4f}", end="")
        print()
    print("  info=NO means the human control's nearest reference is within 10")
    print("  percent of a random one: that k is in the concentration regime")
    print("  and its R and P are UNINFORMATIVE, per the prereg addendum.")

    print(f"\n  R by fractional position quintile")
    for m in mtags:
        print(f"  {m}")
        print(f"    {'k':>3}" + "".join(f"{'Q' + str(i + 1):>9}"
                                        for i in range(5)))
        for k in KS:
            r = res[f"k{k}"]
            rq = [r[m]["q"][i] / r["human"]["q"][i] for i in range(5)]
            res[f"k{k}"][m]["R_q"] = rq
            print(f"    {k:>3}" + "".join(f"{v:>9.3f}" for v in rq))

    shown = next((m for m in mtags if m != "model_NULL"), mtags[0])
    print(f"\n  median distance to the 1st, 8th and 32nd nearest human "
          f"reference\n  DESCRIPTIVE, no bar is read off this")
    print(f"    {'k':>3}" + "".join(f"{p + r:>12}" for p in ("hum", "mdl")
                                    for r in (" 1", " 8", "32")))
    for k in KS:
        r = res[f"k{k}"]
        cells = r["human"]["rank"] + r[shown]["rank"]
        print(f"    {k:>3}" + "".join(f"{v:>12.4f}" for v in cells))

    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\n  -> {args.out}")


if __name__ == "__main__":
    main()
