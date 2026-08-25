"""What is the retrieval expert WORTH, in nats, before anyone builds it?

Registered in /home/aaronadmin/w4_arms/expertgap_prereg.md INCLUDING its
amendment. Read that first. In particular read "Why this arm and not the three I
killed on the way to it", because the three dead candidates are each a plausible
day of GPU and the reasons they died are the reason this file has the shape it
has.

THE QUESTION. `w4_texcover` established today that the model's own free running
states, out to eight events of emitted token history, sit inside the human
corpus. So a human expert is queryable at every state the model visits. A DAgger
style arm would train the model toward that expert at its own states, and it
would be the first thing in the record to satisfy all three constraints RESUME
lists for a surviving training arm. This file does NOT build it. It prices it,
because seven consecutive arms in this record have found a large, overwhelmingly
significant per step defect and priced it at approximately nothing.

THE ESTIMATOR, per the amendment, is a difference in differences.

    G_model(arm) = mean over queries, over K retrieved human neighbours j, of
                   -log p_model(a_j | s), with a_j that neighbour's own next
                   token triple and the model's chain evaluated AT a_j's own
                   s_cur and th_cur rather than at its own sample.

    G_ref2(arm)  = the same targets scored by a purely human predictor, the
                   smoothed empirical of neighbours retrieved from a DISJOINT
                   human reference split R2 at the identical query state.

    EXCESS = [G_model(MODEL) - G_model(HUMAN)] - [G_ref2(MODEL) - G_ref2(HUMAN)]

Zero under the null that the model's conditional is no worse at its own states
than at human ones. A retrieved target that is simply higher entropy at model
states moves both brackets together and cancels. Both brackets are printed, so a
disagreement between them is visible rather than differenced away.

    >= 0.05 nats  MATERIAL, the DAgger arm is authorised and gets its own
                  registration before any training step runs
    <= 0.01 nats  NEGLIGIBLE, the "evaluate at the model's own visited states"
                  family closes for AUC purposes
    otherwise     MIXED, both numbers reported, nothing authorised

Read on the largest of the three heads, since a training arm may target one.

THE LEAKAGE GUARD. A human query state retrieves its own trajectory's adjacent
position unless prevented, which would drive G(HUMAN) toward zero and manufacture
the entire result. The registration demands per neighbour exclusion by trajectory
id. This implements something strictly stronger: the reference splits and the
human query set are DISJOINT TRAJECTORY SETS, so there is no exclusion logic to
have a bug in, exactly as `w4_texcover` argued. The per neighbour check is then
run anyway as a runtime ASSERTION, so the guard is verified rather than reasoned
about.

THE AUC CONVERSION IS SECONDARY. `w4_arcurve`'s 0.1904 AUC per nat was fitted on
teacher forced held out likelihood over human data, and EXCESS is occupancy
weighted at model states. Converting assumes the exchange rate is the same in
both measures and there is no evidence for that. The verdict is read in NATS.

Safety. Reads the corpus, one event AR checkpoint and the two rollout dumps.
Touches no evaluation data, never `data/human_eval_features.npy`, never
`evaluate.py`, never scoring code, and never writes any checkpoint. Its MD5 check
on `training/candi_polar_flow_best.pt` is done by the runner, not here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.neighbors import NearestNeighbors

# Same reasoning as w4_texcover: brute force over a million references chunks
# query rows to fit working_memory, and this VM is capped at 14 GB with three
# recorded WSL deaths behind it.
sklearn.set_config(working_memory=2048)

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.event_ar import (  # noqa: E402
    DT_MAX_MS, N_DT_CLASSES, EventARModel, prefix_state,
)
from models.event_stream_polar import (  # noqa: E402
    N_S_CLASSES, N_TH_CLASSES, S_PAD_CLASS, TH_NULL_CLASS,
    dth_lattice_to_class, s2_to_class,
)
from w4_texcover import KMAX, TBUF, _keys, CORPUS  # noqa: E402

KS = (16, 64, 256)          # bracketed both sides of 64. The 1024 rung is
                            # DROPPED for cost and that is logged in the
                            # amendment, not silently.
ALPHAS = (0.1, 0.5, 2.0)    # smoothing of the R2 empirical predictor. It sits
                            # inside the control, not the primary, but it is a
                            # dispersion parameter and the record's rule is to
                            # bracket both sides of one.
HEADS = ("s", "th", "dt")


def _corpus_tokens(ids):
    """Corpus rows -> class streams, converted exactly as ARDataset converts."""
    s2m = np.load(CORPUS / "events_s2.npy", mmap_mode="r")
    dthm = np.load(CORPUS / "events_dth.npy", mmap_mode="r")
    dtm = np.load(CORPUS / "events_dt.npy", mmap_mode="r")
    Lall = np.load(CORPUS / "events_len.npy")
    Call = np.load(CORPUS / "events_cond.npy")

    ids = np.sort(ids)
    L = np.minimum(Lall[ids], TBUF)
    s2 = torch.from_numpy(np.asarray(s2m[ids, :TBUF], dtype=np.int64))
    dth = torch.from_numpy(np.asarray(dthm[ids, :TBUF], dtype=np.int64))
    dt_ms = torch.from_numpy(np.asarray(dtm[ids, :TBUF], dtype=np.float32))
    valid = torch.from_numpy(np.arange(TBUF)[None, :] < L[:, None])

    s_cls = torch.where(valid, s2_to_class(s2), torch.full_like(s2, S_PAD_CLASS))
    th_cls = torch.where(valid & (s2 > 0), dth_lattice_to_class(dth),
                         torch.full_like(dth, TH_NULL_CLASS))
    dt_cls = torch.where(valid, torch.round(dt_ms).long().clamp(0, DT_MAX_MS),
                         torch.zeros_like(s2))
    cond = torch.from_numpy(np.asarray(Call[ids], dtype=np.float32))
    return s_cls, th_cls, dt_cls, cond, L, ids


def _model_tokens(path):
    z = np.load(path)
    s = torch.from_numpy(z["s"].astype(np.int64))
    th = torch.from_numpy(z["th"].astype(np.int64))
    dt = torch.from_numpy(z["dt"].astype(np.int64))
    cond = torch.from_numpy(z["cond"].astype(np.float32))
    pad = (s >= S_PAD_CLASS).numpy()
    T = s.shape[1]
    L = np.where(pad.any(1), pad.argmax(1), T)
    valid = torch.from_numpy(np.arange(T)[None, :] < L[:, None])
    dt = torch.where(valid, dt.clamp(0, DT_MAX_MS), torch.zeros_like(dt))
    # Rollout rows are not corpus rows, so they carry a trajectory id that can
    # never collide with a reference id. -1 makes the leakage assertion below
    # meaningful for this arm too rather than vacuous.
    return s, th, dt, cond, L, np.full(len(s), -1, dtype=np.int64)


def positions(s_cls, th_cls, dt_cls, cond, L, ids, cap, rng):
    """Valid (row, t) positions with their key, target triple and trajectory id.

    The key is w4_texcover's, unchanged and imported rather than reimplemented,
    so this arm's notion of "the same state" is identical to the one that
    established coverage. At key position t the state describes everything before
    event t, so the TARGET is event t itself.
    """
    key, frac = _keys(s_cls, th_cls, dt_cls, cond, L, rng, None)
    T = s_cls.shape[1]
    idx = np.arange(T)[None, :]
    ok = (idx >= KMAX) & (idx < L[:, None])
    rows = np.broadcast_to(np.arange(len(L))[:, None], ok.shape)[ok]
    ts = np.broadcast_to(idx, ok.shape)[ok]
    tgt = np.stack([s_cls.numpy()[ok], th_cls.numpy()[ok], dt_cls.numpy()[ok]], 1)
    traj = ids[rows]
    # THE DIRECTION HEAD IS SUPERVISED ON MOVING EVENTS ONLY, exactly as
    # w4_arfit does it. Where there is no motion the target is TH_NULL_CLASS and
    # the head was never trained to emit it, so it scores about 40 nats there.
    # The first run of this arm averaged th over ALL positions, which put 8.3
    # percent of the mass on a domain the head does not serve, inflated the
    # model's direction NLL from 0.86 to 4.15 and reversed the sign of the
    # arm's headline. Carry the mask with the positions so that cannot recur.
    move = ((tgt[:, 0] > 0) & (tgt[:, 0] < S_PAD_CLASS))

    if cap is not None and len(key) > cap:
        sel = rng.choice(len(key), cap, replace=False)
        key, frac, rows, ts, tgt, traj, move = (a[sel] for a in
                                               (key, frac, rows, ts, tgt, traj,
                                                move))
    return dict(key=key, frac=frac, row=rows, t=ts, tgt=tgt, traj=traj,
                move=move)


def head_mean(a, move):
    """Per head mean of a (Q, 3) array, with the th head on MOVING events only.

    Every per head aggregate in this arm goes through here. A plain .mean(0) on
    the th column is the bug that voided the first run.
    """
    assert a.shape[1] == 3 and len(a) == len(move), (a.shape, move.shape)
    return [float(a[:, 0].mean()), float(a[move, 1].mean()),
            float(a[:, 2].mean())]


@torch.no_grad()
def trunk_at(model, s_cls, th_cls, dt_cls, cond, rows, ts, dev, batch=48):
    """Trunk output x at the requested (row, t) positions, teacher forced.

    The trunk is causally masked, so one full sequence pass reproduces exactly
    the hidden state that was live when position t was emitted. That is the same
    property the rollout objective's estimator leans on.
    """
    state = prefix_state(s_cls, th_cls, dt_cls, cond)
    sp, tp, dp = EventARModel.shift_inputs(s_cls, th_cls, dt_cls)
    order = np.argsort(rows, kind="stable")
    out = None
    for c0 in range(0, len(s_cls), batch):
        sl = slice(c0, min(c0 + batch, len(s_cls)))
        x = model.trunk(sp[sl].to(dev), tp[sl].to(dev), dp[sl].to(dev),
                        state[sl].to(dev), cond[sl].to(dev))
        take = order[(rows[order] >= sl.start) & (rows[order] < sl.stop)]
        if len(take) == 0:
            continue
        got = x[rows[take] - sl.start, ts[take]].float().cpu()
        if out is None:
            out = torch.zeros(len(rows), got.shape[-1])
        out[take] = got
    return out


@torch.no_grad()
def model_logprob(model, X, cand, dev, chunk=200_000):
    """-log p_model of each candidate triple at each query state, (Q, K, 3).

    The chain is s, then th given s, then dt given s and th, and every candidate
    is scored at ITS OWN s_cur and th_cur. Scoring at the model's own sample
    instead would measure the model against itself, which is the distillation
    failure the record already closed twice.
    """
    Q, K, _ = cand.shape
    D = X.shape[-1]
    qi = np.repeat(np.arange(Q), K)
    sa = cand[:, :, 0].reshape(-1)
    ta = cand[:, :, 1].reshape(-1)
    da = cand[:, :, 2].reshape(-1)
    out = torch.zeros(Q * K, 3)

    for c0 in range(0, Q * K, chunk):
        sl = slice(c0, min(c0 + chunk, Q * K))
        x = X[qi[sl]].to(dev)
        s_c = torch.from_numpy(sa[sl]).to(dev).clamp(max=N_S_CLASSES - 1)
        t_c = torch.from_numpy(ta[sl]).to(dev).clamp(max=N_TH_CLASSES - 1)
        d_c = torch.from_numpy(da[sl]).to(dev).clamp(max=N_DT_CLASSES - 1)

        ls = torch.log_softmax(model.s_head(x).float(), -1)
        lt = torch.log_softmax(model.th_logits(x, s_c).float(), -1)
        ld = torch.log_softmax(model.dt_logits(x, s_c, t_c).float(), -1)
        out[sl, 0] = -ls.gather(1, s_c[:, None]).squeeze(1).cpu()
        out[sl, 1] = -lt.gather(1, t_c[:, None]).squeeze(1).cpu()
        out[sl, 2] = -ld.gather(1, d_c[:, None]).squeeze(1).cpu()
    return out.reshape(Q, K, 3).numpy()


def ref2_logprob(tgt_cand, ref2_cand, alpha):
    """-log p of each R1 target under the smoothed empirical of R2's neighbours.

    Per head and unconditional within the step. That handicaps R2 relative to the
    model, which has the within step coupling available to it, and the handicap
    is uniform so it cancels in the difference in differences. What does not
    cancel is a difference BETWEEN ARMS in coupling strength, which the
    registration names as this estimator's unguarded limitation.
    """
    Q, K, _ = tgt_cand.shape
    out = np.zeros((Q, K, 3), dtype=np.float32)
    for h, V in enumerate((N_S_CLASSES, N_TH_CLASSES, N_DT_CLASSES)):
        col = np.clip(ref2_cand[:, :, h], 0, V - 1)
        cnt = np.zeros((Q, V), dtype=np.float64)
        np.add.at(cnt, (np.repeat(np.arange(Q), col.shape[1]), col.reshape(-1)), 1.0)
        p = (cnt + alpha) / (cnt.sum(1, keepdims=True) + alpha * V)
        tc = np.clip(tgt_cand[:, :, h], 0, V - 1)
        out[:, :, h] = -np.log(np.take_along_axis(p, tc, axis=1))
    return out


def build_reference(ids, cap, rng, chunk=4000):
    """Reference keys, targets and trajectory ids, built in trajectory chunks.

    Chunked because `_keys` materialises (B, T, 38) plus eight lag copies, and
    forty five thousand trajectories at once is several gigabytes against a
    14 GB VM with three recorded WSL deaths behind it. `w4_texcover` chunked at
    this same width for the same reason.
    """
    parts, got = [], 0
    for c0 in range(0, len(ids), chunk):
        tk = _corpus_tokens(ids[c0:c0 + chunk])
        parts.append(positions(*tk, None, rng))
        got += len(parts[-1]["key"])
        if got >= cap * 1.15:
            break
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if len(out["key"]) > cap:
        sel = rng.choice(len(out["key"]), cap, replace=False)
        out = {k: v[sel] for k, v in out.items()}
    return out


def marginal_nll(ref_tgt, own_tgt, alpha, move):
    """-log p of each query's own token under the reference GLOBAL marginal.

    V0a's comparison point. Retrieval that cannot beat this carries no state
    information and the expert does not exist. The th column is scored on
    MOVING events only, on both sides, for the reason in positions().
    """
    out = np.zeros(3)
    for h, V in enumerate((N_S_CLASSES, N_TH_CLASSES, N_DT_CLASSES)):
        rt = ref_tgt[:, h] if h != 1 else ref_tgt[
            (ref_tgt[:, 0] > 0) & (ref_tgt[:, 0] < S_PAD_CLASS), h]
        ot = own_tgt[:, h] if h != 1 else own_tgt[move, h]
        cnt = np.bincount(np.clip(rt, 0, V - 1), minlength=V)
        p = (cnt + alpha) / (cnt.sum() + alpha * V)
        out[h] = float(-np.log(p[np.clip(ot, 0, V - 1)]).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="*", default=[])
    ap.add_argument("--ckpt", default="event_ar_hm_mlp.pt")
    ap.add_argument("--n-ref", type=int, default=1_000_000)
    ap.add_argument("--n-qry", type=int, default=10_000)
    ap.add_argument("--ref-traj", type=int, default=45_000)
    ap.add_argument("--hq-traj", type=int, default=1_500)
    ap.add_argument("--h2-traj", type=int, default=1_500,
                    help="THE NULL ARM, added by the second amendment. A fourth "
                         "disjoint human set entered through the model pool "
                         "path. Its EXCESS is the instrument's own width and no "
                         "model EXCESS that fails to exceed it may be read.")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="research/w4_expertgap_results.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    n_traj = len(np.load(CORPUS / "events_len.npy"))
    perm = rng.permutation(n_traj)
    a = args.ref_traj
    r1_ids, r2_ids = perm[:a], perm[a:2 * a]
    b = 2 * a + args.hq_traj
    hq_ids = perm[2 * a:b]
    h2_ids = perm[b:b + args.h2_traj]
    sets = dict(R1=r1_ids, R2=r2_ids, HQ=hq_ids, H2=h2_ids)
    for u in sets:
        for v in sets:
            if u < v:
                assert len(np.intersect1d(sets[u], sets[v])) == 0, f"{u} {v}"
    print(f"  corpus {n_traj} trajectories. R1 {len(r1_ids)}, R2 {len(r2_ids)}, "
          f"human query {len(hq_ids)}, NULL arm {len(h2_ids)}, all disjoint",
          flush=True)

    ck = torch.load(f"training/{args.ckpt}", map_location=dev, weights_only=False)
    cfg = ck["config"]
    assert cfg.get("emit_order", "s_th_dt") == "s_th_dt", \
        "this scorer implements the s then th then dt chain only"
    model = EventARModel(**cfg).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    print(f"  {args.ckpt} step {ck.get('step')} on {dev}", flush=True)

    refs = {}
    for tag, ids in (("R1", r1_ids), ("R2", r2_ids)):
        refs[tag] = build_reference(ids, args.n_ref, rng)
        print(f"  {tag} reference {refs[tag]['key'].shape}", flush=True)

    pools = {}
    tk = _corpus_tokens(hq_ids)
    pools["HUMAN"] = (positions(*tk, args.n_qry, rng), tk)
    if args.h2_traj:
        tk = _corpus_tokens(h2_ids)
        pools["HUMAN_NULL"] = (positions(*tk, args.n_qry, rng), tk)
    for sp in args.streams:
        tag = Path(sp).stem.split("_")[-1]
        tk = _model_tokens(sp)
        pools[f"MODEL_{tag}"] = (positions(*tk, args.n_qry, rng), tk)
    for nm, (P, _) in pools.items():
        print(f"  {nm} query {P['key'].shape}", flush=True)

    # One whitening, fitted on R1 and applied to everything, so every arm is
    # measured against the identical reference geometry.
    d = 6 + 4 * KMAX
    mu = refs["R1"]["key"][:, :d].mean(0)
    Xc = refs["R1"]["key"][:, :d] - mu
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    w, V = np.linalg.eigh(C.astype(np.float64))
    W = (V / np.sqrt(np.maximum(w, 1e-8 * float(w.max())))).astype(np.float32)
    idx = {t: NearestNeighbors(n_neighbors=max(KS), algorithm="brute",
                               n_jobs=-1).fit((refs[t]["key"][:, :d] - mu) @ W)
           for t in ("R1", "R2")}
    print(f"  whitened dim {d}, indices built", flush=True)

    res, res_v0 = {}, {}
    for nm, (P, tk) in pools.items():
        qw = ((P["key"][:, :d] - mu) @ W).astype(np.float32)
        nb = {t: idx[t].kneighbors(qw, return_distance=False)
              for t in ("R1", "R2")}
        # THE LEAKAGE GUARD, verified rather than reasoned about.
        for t in ("R1", "R2"):
            share = refs[t]["traj"][nb[t]] == P["traj"][:, None]
            assert not share.any(), f"{nm}: retrieved own trajectory from {t}"
        cand1 = refs["R1"]["tgt"][nb["R1"]]              # (Q, Kmax, 3) targets
        cand2 = refs["R2"]["tgt"][nb["R2"]]              # (Q, Kmax, 3) predictor

        X = trunk_at(model, tk[0], tk[1], tk[2], tk[3], P["row"], P["t"], dev)
        nll_m = model_logprob(model, X, cand1, dev)      # (Q, Kmax, 3)
        print(f"  {nm} scored, trunk {tuple(X.shape)}", flush=True)

        # V0, the premise gate, third amendment. Only meaningful where the
        # query's own next token is a real human one, so model arms are skipped.
        if not nm.startswith("MODEL_"):
            own = P["tgt"][:, None, :]                   # (Q, 1, 3)
            mv = P["move"]
            v0 = {"move_frac": float(mv.mean()),
                  "model": head_mean(model_logprob(model, X, own, dev)[:, 0], mv)}
            for K in KS:
                v0[f"expert_K{K}"] = head_mean(
                    ref2_logprob(own, cand1[:, :K], ALPHAS[1])[:, 0], mv)
            v0["marginal"] = marginal_nll(refs["R1"]["tgt"], P["tgt"],
                                          ALPHAS[1], mv).tolist()
            res_v0[nm] = v0

        row = {}
        # These score the NEIGHBOURS' tokens, so the direction mask is the
        # CANDIDATE's own motion, not the query's. Same reason as positions().
        cmv = (cand1[:, :, 0] > 0) & (cand1[:, :, 0] < S_PAD_CLASS)
        for K in KS:
            def hcol(a, K=K):
                """(Q, K, 3) -> per head mean, th over moving candidates only."""
                m = cmv[:, :K]
                return [float(a[:, :K, 0].mean()), float(a[:, :K, 1][m].mean()),
                        float(a[:, :K, 2].mean())]

            # Per query means, th averaged over that query's moving candidates.
            w = cmv[:, :K].astype(np.float64)
            q = np.stack([nll_m[:, :K, 0].mean(1),
                          (nll_m[:, :K, 1] * w).sum(1) / np.maximum(w.sum(1), 1),
                          nll_m[:, :K, 2].mean(1)], 1)
            keep = w.sum(1) > 0        # no moving candidate, nothing on th
            q = q[keep]
            lo, hi = np.quantile(q, [0.05, 0.95], axis=0)
            cell = {
                "model": hcol(nll_m),
                "model_med": [float(np.median(q[:, h])) for h in range(3)],
                # ten percent trimmed. The direction head's mean is carried by
                # rare extreme disagreements, so a verdict that flips between
                # this and the mean is a tail finding and is REFUSED.
                "model_trim": [float(q[(q[:, h] >= lo[h]) & (q[:, h] <= hi[h]), h]
                                     .mean()) for h in range(3)],
            }
            for al in ALPHAS:
                cell[f"ref2_a{al}"] = hcol(ref2_logprob(cand1[:, :K],
                                                        cand2[:, :K], al))
            cell["by_q"] = q.tolist()
            # frac is filtered by the SAME mask, so the quintile binning below
            # cannot silently misalign with by_q.
            cell["frac"] = P["frac"][keep].tolist()
            row[f"K{K}"] = cell
        res[nm] = row

    # HUMAN_NULL is differenced by the identical arithmetic as a model arm. It
    # IS the instrument's width and is listed first so it is read first.
    mtags = ([n for n in res if n == "HUMAN_NULL"]
             + [n for n in res if n.startswith("MODEL_")])
    out = {"arms": res, "excess": {}, "v0": res_v0}
    UNIF = [float(np.log(v)) for v in (N_S_CLASSES, N_TH_CLASSES, N_DT_CLASSES)]

    print(f"\n  V0, THE PREMISE GATE. Scored on each query's OWN next token, so")
    print(f"  it is only defined where that token is a real human one. Read")
    print(f"  BEFORE anything else. V0a: expert must beat marginal, or the")
    print(f"  retrieval carries no state information and the arm is VOID.")
    print(f"  V0b: expert must beat the model, or there is nothing to distil")
    print(f"  and the DAgger family closes here.")
    print(f"  {'arm':<11}{'predictor':>12}" + "".join(f"{h:>10}" for h in HEADS))
    for nm, v0 in res_v0.items():
        for lbl in ["marginal", "model"] + [f"expert_K{K}" for K in KS]:
            print(f"  {nm:<11}{lbl:>12}" +
                  "".join(f"{v:>10.4f}" for v in v0[lbl]))
        best = [min(v0[f"expert_K{K}"][h] for K in KS) for h in range(3)]
        print(f"  {nm:<11}{'V0a edge':>12}" + "".join(
            f"{v0['marginal'][h] - best[h]:>10.4f}" for h in range(3)))
        print(f"  {nm:<11}{'V0b edge':>12}" + "".join(
            f"{v0['model'][h] - best[h]:>10.4f}" for h in range(3)))
    print(f"  edges are POSITIVE when the expert wins. A head whose V0b edge is")
    print(f"  not positive is DROPPED and its primary is not read.")

    print(f"\n  LEVELS, nats per token. G_model is the model against retrieved")
    print(f"  human targets. G_ref2 is a purely human predictor on the same")
    print(f"  targets, so it is the instrument's floor and V1 is read off it.")
    print(f"  uniform bound" + "".join(f"{v:>10.4f}" for v in UNIF))
    print(f"  {'arm':<11}{'K':>5}{'stat':>7}" +
          "".join(f"{h:>10}" for h in HEADS))
    for nm in res:
        for K in KS:
            c = res[nm][f"K{K}"]
            for lbl, key in (("mean", "model"), ("med", "model_med"),
                             ("trim", "model_trim"),
                             ("ref2", f"ref2_a{ALPHAS[1]}")):
                print(f"  {nm:<11}{K:>5}{lbl:>7}" +
                      "".join(f"{v:>10.4f}" for v in c[key]))

    print(f"\n  EXCESS, difference in differences against the HUMAN arm, nats")
    print(f"  raw = model bracket only. dd = corrected by the R2 human predictor.")
    for m in mtags:
        print(f"  {m}")
        print(f"    {'K':>5}{'alpha':>7}" +
              "".join(f"{h + '_raw':>10}{h + '_dd':>10}" for h in HEADS))
        for K in KS:
            gm = np.array(res[m][f"K{K}"]["model"])
            gh = np.array(res["HUMAN"][f"K{K}"]["model"])
            raw = gm - gh
            trim = (np.array(res[m][f"K{K}"]["model_trim"])
                    - np.array(res["HUMAN"][f"K{K}"]["model_trim"]))
            out["excess"][f"{m}_K{K}_trim"] = trim.tolist()
            print(f"    {K:>5}{'trim':>7}" + "".join(
                f"{trim[h]:>10.4f}{'':>10}" for h in range(3)))
            for al in ALPHAS:
                r2m = np.array(res[m][f"K{K}"][f"ref2_a{al}"])
                r2h = np.array(res["HUMAN"][f"K{K}"][f"ref2_a{al}"])
                dd = raw - (r2m - r2h)
                out["excess"][f"{m}_K{K}_a{al}"] = dict(
                    raw=raw.tolist(), dd=dd.tolist())
                print(f"    {K:>5}{al:>7}" + "".join(
                    f"{raw[h]:>10.4f}{dd[h]:>10.4f}" for h in range(3)))

    print(f"\n  EXCESS by fractional position quintile, K={max(KS)}, "
          f"raw bracket only")
    KM = f"K{max(KS)}"
    hf = np.array(res["HUMAN"][KM]["frac"])
    hq = np.array(res["HUMAN"][KM]["by_q"])
    edges = np.quantile(hf, [0.2, 0.4, 0.6, 0.8])
    hb = np.digitize(hf, edges)
    for m in mtags:
        mf = np.array(res[m][KM]["frac"])
        mq = np.array(res[m][KM]["by_q"])
        assert len(mf) == len(mq) and len(hf) == len(hq)
        mb = np.digitize(mf, edges)
        # PER HEAD. The first run of this arm averaged over the head axis here,
        # which pools three heads whose excesses have opposite signs and is
        # uninterpretable. V3 was not performed on that run and its printed
        # quintile table must not be read.
        cells = [[float(mq[mb == b, h].mean() - hq[hb == b, h].mean())
                  for b in range(5)] for h in range(3)]
        out["excess"][f"{m}_quintile"] = cells
        for h, nm_h in enumerate(HEADS):
            print(f"    {m + ' ' + nm_h:<16}" +
                  "".join(f"{v:>10.4f}" for v in cells[h]))
    print("  A pooled EXCESS not reproduced within its quintiles is a position "
          "confound\n  and is not read as a pooled number.")

    for nm in res:                       # keep the JSON small enough to read
        for K in KS:
            res[nm][f"K{K}"].pop("by_q", None)
            res[nm][f"K{K}"].pop("frac", None)
        res[nm].pop("frac", None)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n  -> {args.out}")


if __name__ == "__main__":
    main()
