"""Fine tune the timing head against a learned critic.

PRE REGISTERED in HANDOFF.md 2026-08-10, "## Fine tuning the temporal head
against the critic". Arms, controls, thresholds and the prediction were fixed
before this file was written.

`w4_critic_ablate` localised the tell. Blinding the millisecond waits costs the
critic 0.134 of a 0.231 excess where blinding speed costs 0.006 and blinding
turn 0.009, and time is strongly super additive with shape. `w4_timing` reached
the same verdict on 2026-08-05 from a spectral readout. So the target is the
timing head and nothing else.

`dt_head` and `dt_norm` are the only parameters that move. The trunk, the speed
head, the direction head and every embedding stay frozen, so the speed and
direction conditionals are untouched by construction. That is a weight fine
tune of a head that already exists and is not the FiLM rewrite of `th_head` and
`dt_head` that sits on the NOT AUTHORISED list.

A critic cannot be differentiated through a sampled integer. Feeding it a soft
mixture of embeddings would let the generator win by being blurry, since blur
is off the manifold the critic was trained on. This uses the straight through
estimator instead: the critic only ever sees hard tokens in its forward pass
and the gradient reaches the timing head as if the token had been soft.

Serving does not change. One trajectory per command, no candidates, no
selection. The critic is a training signal and is discarded at the end.

Safety. Reads the training corpus and one checkpoint. Writes a new checkpoint
under a new name and never touches `event_ar_v2_s40000.pt` or
`candi_polar_flow_best.pt`. Adjudicates through `research/autoloop/scoring.py`
against `data/human_val_features_grpo.npy`, never the protected eval set.
Paces on GPU temperature through `w4_latent`.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python -u research/w4_advtime.py
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("EVENT_SNAP", "2.5")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
import scoring  # noqa: E402
from features import extract_feature_matrix  # noqa: E402
from models.event_ar import (DT_MAX_MS, EventARModel,  # noqa: E402
                            class_to_dt_ms, prefix_state)
from models.event_stream_polar import S_PAD_CLASS  # noqa: E402
from w4_critic import Critic, human_rows, pack  # noqa: E402
from w4_latent import gpu_c, throttle  # noqa: E402

DATA = Path("training")
T_MAX = 256
N_DT_CLASSES = DT_MAX_MS + 1

# AMENDED 2026-08-10 after a smoke test at 400 rows, before the real run. The
# registered guard was the served 0.6446, which is wrong for this measurement
# path: serving runs the full pipeline while this file calls `sample` and
# `esp._decode` directly, and `w4_prefix` measured that path at 0.6337 on its
# unforced arm. The guard is that number with the tolerance widened to the run
# to run contract noise `w4_prefix` records. See HANDOFF for the amendment.
PATH_REF = 0.6337
BASE_TOL = 0.035

WIN_AT = 0.025      # tuned beats its own paired baseline by this or more
PARTIAL_AT = 0.008  # the size of the snap 5.0 effect
NLL_MAX_RISE = 0.15  # above this the head is destroyed, not repaired
COUNT_MAX_SHIFT = 0.10  # event count moving this much means the state carried
                        # the intervention into the geometry


def load_model(dev, ckpt):
    ck = torch.load(DATA / ckpt, map_location=dev, weights_only=False)
    m = EventARModel(**ck["config"]).to(dev).eval()
    m.load_state_dict(ck["model_state_dict"])
    return m, ck


def tune_params(model):
    """The timing head and its norm, and nothing else."""
    return list(model.dt_head.parameters()) + list(model.dt_norm.parameters())


def freeze_all_but_timing(model):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in tune_params(model):
        p.requires_grad_(True)


def batch_tensors(rows, cond, dev):
    """A list of (s, th, dt) plus its commands, on the device."""
    s, th, dt, m = pack(rows, T_MAX)
    return (torch.from_numpy(s).to(dev), torch.from_numpy(th).to(dev),
            torch.from_numpy(dt).to(dev), torch.from_numpy(m).to(dev),
            torch.from_numpy(cond).to(dev))


def dt_logits_of(model, s, th, dt, cond):
    """Teacher forced timing logits over a whole token stream.

    The trunk is frozen, so it runs under no_grad and only the timing head
    carries a graph. That is what makes a 256 wide sequence affordable.
    """
    with torch.no_grad():
        state = prefix_state(s, th, dt, cond)
        s_prev, th_prev, dt_prev = model.shift_inputs(s, th, dt)
        x = model.trunk(s_prev, th_prev, dt_prev, state, cond)
        ctx = x + model._s_emb(s) + model._th_emb(th)
    return model.dt_head(model.dt_norm(ctx))


def sample_pool(model, cond, batch, dev, seed, min_len):
    """Generate on real commands through the call serving makes. Nothing is
    scored, filtered or selected here; every row that decodes is kept."""
    rows, keep, t0 = [], [], time.time()
    with torch.no_grad():
        for c0 in range(0, len(cond), batch):
            throttle()
            torch.manual_seed(seed + c0)
            blk = torch.tensor(cond[c0:c0 + batch], dtype=torch.float32,
                               device=dev)
            s_c, th_c, dt_c = model.sample(blk, temperature=1.0)
            s_np, th_np = s_c.cpu().numpy(), th_c.cpu().numpy()
            dt_np = dt_c.cpu().numpy()
            pad = s_np >= S_PAD_CLASS
            for j in range(s_np.shape[0]):
                L = int(pad[j].argmax()) if pad[j].any() else s_np.shape[1]
                if L >= min_len:
                    rows.append((s_np[j, :L], th_np[j, :L], dt_np[j, :L]))
                    keep.append(c0 + j)
    print(f"      sampled {len(rows)}/{len(cond)} in {time.time() - t0:.0f}s, "
          f"{gpu_c()}C", flush=True)
    return rows, np.asarray(keep, dtype=np.int64)


def critic_round(critic, opt, hum, hcond, gen, gcond, dev, bs, passes, seed):
    """Train the persistent critic further on this round's generation.

    Human is class one. Returns the held out AUC after the passes, which is how
    far apart the two distributions still are.
    """
    lossf = nn.BCEWithLogitsLoss()
    n = min(len(hum[0]), len(gen[0]))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(400, n // 6)
    vi, ti = perm[:n_val], perm[n_val:]
    critic.train()
    for _ in range(passes):
        order = rng.permutation(len(ti))
        for c0 in range(0, len(ti) - bs + 1, bs):
            if (c0 // bs) % 20 == 0:
                throttle()
            k = ti[order[c0:c0 + bs]]
            lg = torch.cat([critic(*[t[k] for t in gen], gcond[k]),
                            critic(*[t[k] for t in hum], hcond[k])])
            y = torch.cat([torch.zeros(len(k)), torch.ones(len(k))]).to(dev)
            opt.zero_grad()
            lossf(lg, y).backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt.step()
    critic.eval()
    with torch.no_grad():
        pg = torch.cat([critic(*[t[vi[c0:c0 + 256]] for t in gen],
                               gcond[vi[c0:c0 + 256]]).cpu()
                        for c0 in range(0, len(vi), 256)])
        ph = torch.cat([critic(*[t[vi[c0:c0 + 256]] for t in hum],
                               hcond[vi[c0:c0 + 256]]).cpu()
                        for c0 in range(0, len(vi), 256)])
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.zeros(len(vi)), np.ones(len(vi))])
    return float(roc_auc_score(y, torch.cat([pg, ph]).numpy()))


def gen_round(model, critic, opt, gen, gcond, hum, hcond, dev, bs, passes,
              anchor, seed):
    """Push the timing head towards a critic that calls the row human.

    The straight through term is the whole trick. `onehot + p - p.detach()`
    is exactly the sampled token in the forward pass, so the critic sees the
    hard stream it was trained on, while the backward pass differentiates the
    timing distribution that produced it.
    """
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    n, nh = len(gen[0]), len(hum[0])
    adv_tot = anc_tot = nb = 0.0
    model.eval()
    critic.eval()
    for _ in range(passes):
        order = rng.permutation(n)
        for c0 in range(0, n - bs + 1, bs):
            if (c0 // bs) % 20 == 0:
                throttle()
            k = order[c0:c0 + bs]
            gs, gth, gdt, gm = [t[k] for t in gen]
            logits = dt_logits_of(model, gs, gth, gdt, gcond[k])
            C = logits.shape[-1]
            p = F.softmax(logits, dim=-1)
            oh = F.one_hot(gdt.clamp(max=C - 1), C).float()
            st = oh + p - p.detach()
            dt_vec = st @ critic.dt_emb.weight[:C]
            lg = critic(gs, gth, gdt, gm, gcond[k], dt_vec=dt_vec)
            adv = lossf(lg, torch.ones_like(lg))

            j = rng.choice(nh, bs, replace=False)
            hs, hth, hdt, hm = [t[j] for t in hum]
            hl = dt_logits_of(model, hs, hth, hdt, hcond[j])
            anc = F.cross_entropy(hl[hm], hdt[hm].clamp(max=hl.shape[-1] - 1))

            opt.zero_grad()
            (adv + anchor * anc).backward()
            nn.utils.clip_grad_norm_(tune_params(model), 1.0)
            opt.step()
            adv_tot += float(adv)
            anc_tot += float(anc)
            nb += 1
    return adv_tot / max(nb, 1), anc_tot / max(nb, 1)


@torch.no_grad()
def real_nll(model, hum, hcond, dev, bs=128):
    """Cross entropy of the real timing token on real sequences. The registered
    guard against buying the critic by abandoning the data."""
    tot = cnt = 0.0
    for c0 in range(0, len(hum[0]), bs):
        sl = slice(c0, c0 + bs)
        hs, hth, hdt, hm = [t[sl] for t in hum]
        lg = dt_logits_of(model, hs, hth, hdt, hcond[sl])
        tot += float(F.cross_entropy(
            lg[hm], hdt[hm].clamp(max=lg.shape[-1] - 1), reduction="sum"))
        cnt += float(hm.sum())
    return tot / max(cnt, 1)


def decode_paths(rows, conds):
    """Token streams to pixel paths through the serving decoder."""
    paths = []
    for (s, th, dt), c in zip(rows, conds):
        d = class_to_dt_ms(torch.from_numpy(dt.astype(np.int64))).numpy()
        dz = (np.log(np.maximum(d, 0.05)) - esp._DT_MEAN) / esp._DT_STD
        ang = float(np.arctan2(float(c[3]), float(c[2])))
        p = esp._decode(dz, s, th, 0.0, 0.0, ang)
        if p is not None and len(p) >= 4:
            paths.append(np.asarray(p, dtype=np.float64))
    return paths


def adjudicate(model, cond, batch, dev, seed, min_len):
    """One trajectory per command, no selection, scored by the contract."""
    rows, keep = sample_pool(model, cond, batch, dev, seed, min_len)
    paths = decode_paths(rows, cond[keep])
    X = extract_feature_matrix(paths)
    X = X[np.all(np.isfinite(X), 1)]
    auc = float(scoring.score_features(X)["auc_rf_oob"])
    n_ev = float(np.mean([len(r[0]) for r in rows]))
    dur = float(np.mean([float(np.sum(r[2])) for r in rows]))
    return auc, int(len(X)), n_ev, dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--gen", type=int, default=1536)
    ap.add_argument("--hum", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--c-passes", type=int, default=2)
    ap.add_argument("--g-passes", type=int, default=2)
    ap.add_argument("--c-lr", type=float, default=3e-4)
    ap.add_argument("--g-lr", type=float, default=3e-5)
    ap.add_argument("--anchor", type=float, default=1.0)
    ap.add_argument("--adj-n", type=int, default=4000)
    ap.add_argument("--adj-seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--min-len", type=int, default=8)
    ap.add_argument("--warm", type=int, default=3000)
    ap.add_argument("--warm-passes", type=int, default=6)
    ap.add_argument("--dead-critic", action="store_true",
                    help="registered control: freeze the critic at random "
                         "initialisation so its signal is worth nothing")
    ap.add_argument("--out", default="research/w4_advtime.json")
    ap.add_argument("--save", default="training/event_ar_advtime.pt")
    args = ap.parse_args()

    dev = esp._DEVICE
    arr = {k: np.load(DATA / f"events_{k}.npy", mmap_mode="r")
           for k in ("s2", "dth", "dt", "len", "cond")}
    ok = np.flatnonzero(np.load(DATA / "events_feat18_ok.npy"))

    # The corpus is ordered by session, so every draw is a random permutation
    # and never a prefix or a positional half.
    rng = np.random.default_rng(args.seed)
    ok = ok[rng.permutation(len(ok))]
    need = (args.hum + args.warm + args.rounds * args.gen
            + args.adj_seeds * args.adj_n)
    if need > len(ok):
        raise SystemExit(f"need {need} rows, corpus has {len(ok)}")
    hi, rest = ok[:args.hum], ok[args.hum:]
    wi, rest = rest[:args.warm], rest[args.warm:]
    round_idx = [rest[i * args.gen:(i + 1) * args.gen]
                 for i in range(args.rounds)]
    adj_base = rest[args.rounds * args.gen:]
    adj_idx = [adj_base[i * args.adj_n:(i + 1) * args.adj_n]
               for i in range(args.adj_seeds)]

    hall = human_rows(arr, hi, args.min_len)
    hkeep = [j for j, r in enumerate(hall) if r is not None]
    hrows = [hall[j] for j in hkeep]
    hcond_np = np.asarray(arr["cond"][hi], dtype=np.float32)[hkeep]
    hum = batch_tensors(hrows, hcond_np, dev)
    hum, hcond = hum[:4], hum[4]
    print(f"\n  {len(hrows)} human rows held for the critic and the anchor")

    model, ck = load_model(dev, args.ckpt)
    freeze_all_but_timing(model)
    base_state = copy.deepcopy(
        {"dt_head": model.dt_head.state_dict(),
         "dt_norm": model.dt_norm.state_dict()})
    n_tune = sum(p.numel() for p in tune_params(model))
    print(f"  {args.ckpt} step {ck.get('step')}, tuning {n_tune/1e3:.1f}k "
          f"of {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    torch.manual_seed(args.seed + 1)
    critic = Critic().to(dev)
    copt = torch.optim.AdamW(critic.parameters(), lr=args.c_lr,
                             weight_decay=0.01)
    gopt = torch.optim.AdamW(tune_params(model), lr=args.g_lr,
                             weight_decay=0.0)
    if args.dead_critic:
        for p in critic.parameters():
            p.requires_grad_(False)
        print("  CONTROL: critic frozen at random initialisation")

    nll0 = real_nll(model, hum, hcond, dev)
    print(f"  real timing NLL before any update {nll0:.4f}\n")

    # Warm the critic on the untouched model before any generator step. The
    # smoke test showed a critic that has seen one round of a few hundred rows
    # reads 0.54, so without this the first generator updates push against
    # noise rather than against an opponent.
    warm_auc = 0.5
    if not args.dead_critic and args.warm > 0:
        wc_np = np.asarray(arr["cond"][wi], dtype=np.float32)
        print(f"  warming the critic on {args.warm} rows")
        wrows, wkeep = sample_pool(model, wc_np, args.batch, dev,
                                   args.seed + 500, args.min_len)
        wg = batch_tensors(wrows, wc_np[wkeep], dev)
        warm_auc = critic_round(critic, copt, hum, hcond, wg[:4], wg[4], dev,
                                args.bs, args.warm_passes, args.seed + 5)
        print(f"  warmed critic held out AUC {warm_auc:.4f}\n", flush=True)

    hist = []
    for r in range(args.rounds):
        idx = round_idx[r]
        gc_np = np.asarray(arr["cond"][idx], dtype=np.float32)
        print(f"  round {r + 1}/{args.rounds}")
        grows, gkeep = sample_pool(model, gc_np, args.batch, dev,
                                   args.seed + 1000 * (r + 1), args.min_len)
        gen = batch_tensors(grows, gc_np[gkeep], dev)
        gen, gcond = gen[:4], gen[4]

        if args.dead_critic:
            auc = 0.5
        else:
            auc = critic_round(critic, copt, hum, hcond, gen, gcond, dev,
                               args.bs, args.c_passes, args.seed + 2 * r + 7)
        adv, anc = gen_round(model, critic, gopt, gen, gcond, hum, hcond, dev,
                             args.bs, args.g_passes, args.anchor,
                             args.seed + 2 * r + 8)
        nll = real_nll(model, hum, hcond, dev)
        hist.append({"round": r + 1, "critic_auc": auc, "adv": adv,
                     "anchor_ce": anc, "real_nll": nll, "gpu_c": gpu_c()})
        print(f"      critic held out AUC {auc:.4f}, adversarial loss "
              f"{adv:.4f}, real timing NLL {nll:.4f} "
              f"({100 * (nll / nll0 - 1):+.1f}%)\n", flush=True)

    tuned_state = copy.deepcopy(
        {"dt_head": model.dt_head.state_dict(),
         "dt_norm": model.dt_norm.state_dict()})
    if not args.dead_critic:
        torch.save({"config": ck["config"],
                    "model_state_dict": model.state_dict(),
                    "step": ck.get("step"), "dt_mean": ck.get("dt_mean"),
                    "dt_std": ck.get("dt_std"), "parent": args.ckpt,
                    "tuned": ["dt_head", "dt_norm"]}, args.save)
        print(f"  wrote {args.save}\n")

    def wear(state):
        model.dt_head.load_state_dict(state["dt_head"])
        model.dt_norm.load_state_dict(state["dt_norm"])

    # Baseline and tuned run on the same commands with the same generation
    # seed, so the pair differs only by the head and most of the run to run
    # spread cancels.
    print(f"  {'seed':>7}{'base':>9}{'tuned':>9}{'delta':>9}"
          f"{'base ev':>9}{'tune ev':>9}{'base ms':>9}{'tune ms':>9}")
    adj = []
    for i, idx in enumerate(adj_idx):
        c_np = np.asarray(arr["cond"][idx], dtype=np.float32)
        s = args.seed + 90000 + 7 * i
        wear(base_state)
        b = adjudicate(model, c_np, args.batch, dev, s, args.min_len)
        wear(tuned_state)
        t = adjudicate(model, c_np, args.batch, dev, s, args.min_len)
        adj.append({"seed": s, "base_auc": b[0], "tuned_auc": t[0],
                    "delta": t[0] - b[0], "base_n": b[1], "tuned_n": t[1],
                    "base_events": b[2], "tuned_events": t[2],
                    "base_dur_ms": b[3], "tuned_dur_ms": t[3]})
        print(f"  {s:>7}{b[0]:>9.4f}{t[0]:>9.4f}{t[0] - b[0]:>9.4f}"
              f"{b[2]:>9.1f}{t[2]:>9.1f}{b[3]:>9.0f}{t[3]:>9.0f}", flush=True)

    base = float(np.mean([a["base_auc"] for a in adj]))
    tuned = float(np.mean([a["tuned_auc"] for a in adj]))
    delta = tuned - base
    spread = float(np.std([a["delta"] for a in adj], ddof=1)) \
        if len(adj) > 1 else float("nan")
    nll_rise = hist[-1]["real_nll"] / nll0 - 1.0
    ev_shift = float(np.mean([a["tuned_events"] / a["base_events"] - 1
                              for a in adj]))

    if abs(base - PATH_REF) > BASE_TOL:
        verdict = (f"VOID. The baseline arm reads {base:.4f} against the "
                   f"same path {PATH_REF}, outside the registered {BASE_TOL}, "
                   f"so this run is not reproducing the model it set out to "
                   f"fix and nothing may be attributed to the fine tune.")
    elif nll_rise > NLL_MAX_RISE:
        verdict = (f"VOID. Real timing NLL rose {100 * nll_rise:.1f}%, past "
                   f"the registered {100 * NLL_MAX_RISE:.0f}%. The head bought "
                   f"the critic by abandoning the data, so any movement in the "
                   f"score is a collapse rather than a repair.")
    elif -delta >= WIN_AT:
        verdict = (f"WIN. Tuned beats its own paired baseline by "
                   f"{-delta:.4f}, past the registered {WIN_AT}.")
    elif -delta >= PARTIAL_AT:
        verdict = (f"PARTIAL. Tuned beats its own paired baseline by "
                   f"{-delta:.4f}, past the registered {PARTIAL_AT} but short "
                   f"of {WIN_AT}.")
    else:
        verdict = (f"NULL. Tuned moves the paired baseline by {delta:+.4f}, "
                   f"short of the registered {PARTIAL_AT}.")

    if abs(ev_shift) > COUNT_MAX_SHIFT:
        verdict += (f" CONFOUNDED: mean event count moved "
                    f"{100 * ev_shift:+.1f}%, past the registered "
                    f"{100 * COUNT_MAX_SHIFT:.0f}%, so the timing pressure "
                    f"reached the geometry through the state.")

    print(f"\n  base {base:.4f}  tuned {tuned:.4f}  delta {delta:+.4f} "
          f"(seed spread {spread:.4f})")
    print(f"  real timing NLL {nll0:.4f} to {hist[-1]['real_nll']:.4f} "
          f"({100 * nll_rise:+.1f}%), event count {100 * ev_shift:+.1f}%")
    print(f"\n  VERDICT  {verdict}\n")

    json.dump({"ckpt": args.ckpt, "dead_critic": args.dead_critic,
               "rounds": args.rounds, "gen_per_round": args.gen,
               "anchor": args.anchor, "g_lr": args.g_lr, "seed": args.seed,
               "nll0": nll0, "warm_auc": warm_auc, "nll_rise": nll_rise, "event_shift": ev_shift,
               "base": base, "tuned": tuned, "delta": delta,
               "seed_spread": spread, "hist": hist, "adj": adj,
               "verdict": verdict, "gpu_c": gpu_c()},
              open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}\n")


if __name__ == "__main__":
    main()
