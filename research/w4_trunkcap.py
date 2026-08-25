"""Does the trunk carry the CROSS STEP joint, or only the marginals it was fed?

Registered in /home/aaronadmin/w4_arms/trunkcap_prereg.md before this file
existed. Read that first; this docstring states the design, not the case for it.

WHY THIS EXISTS. `research/w4_featmap.py` re run on the coupled checkpoint showed
the detector reads DEPENDENCE BETWEEN features rather than any feature's own
content: the eighteen column forest lost 0.0207 at 8.8 se while the five group
alone AUCs netted -0.0012 and two groups became MORE separable. The within step
conditionals are now exhausted, `s` has nothing before it, `th` given `s` was
0.2311 and is built and shipped, `dt` given both was 0.0024. The dependence never
probed this way is ACROSS steps.

THE STRUCTURE BEING QUESTIONED. `models/event_ar.py`:

    x = s_embed(s_prev) + th_embed(th_prev) + dt_embed(dt_prev)
        + pos_embed(i) + state_proj(state)

The previous step's three tokens enter as a SUM. Their combination is never
supplied. That is the same shape as the head defect one level up, and it is NOT
the same argument. The head defect was provable, `th_head` is linear and x could
not see s(t), so no training budget could ever have removed it. Here ten
nonlinear layers follow the sum and a superposition of three embeddings in 384
dimensions is very probably injective over the 540 token values in play, so the
trunk CAN form any joint its loss rewards. Whether it does is empirical.

THE INVERSION THAT MAKES A GAIN MEAN SOMETHING. In `w4_headcap` the correction
saw the CURRENT step's speed, which x provably could not know. Here it sees only
HISTORY, which x was already handed. Nothing new enters, so any gain is the
trunk's own representation being lossy about information it received.

THE ARMS. Every context stream is coarsened to 8 quantile bins BEFORE either arm
sees it. The additive arm gets one embedding per binned stream. The joint arm
gets one embedding of the tuple code whose output width equals the sum of the
additive arm's, so the MLP reading them is the same shape in both. Same
information, same width, differing only in whether an interaction among the
streams can be expressed at all.

    FAMILY A   bins of s(t-1), th(t-1), dt(t-1)             512 tuple codes
    FAMILY B   bins of s(t-1), s(t-2)                        64 tuple codes

    X_add     MLP([LN(x), e_1(b1), ... e_k(bk)])
    X_joint   MLP([LN(x), e_j(code)])
    X_shuf    X_joint with the code stream permuted, so it keeps every parameter
              and every dimension and carries no information about the token it
              sits beside. The joint table is far larger than the additive
              tables, so this is the control that prices those parameters.
    R_depth   MLP(LN(x)) with no context, the depth control
    P_ctl     THE POSITIVE CONTROL, th head only. Context is the CURRENT step's
              speed, binned the same way. That is `w4_headcap`'s R_full at 8 bin
              resolution, and it is known to be worth 0.2342 nats at full
              resolution. A null anywhere below is worth nothing unless this arm
              fires, because a harness that cannot find structure it is known to
              contain has not measured anything.

    JOINT RESIDUE    = add loss - joint loss
    MARGINAL RESIDUE = R_depth loss - add loss

Coarsening throws resolution away from both arms equally, so the contrast stays
valid while its magnitude is a LOWER BOUND.

THRESHOLDS, from the newly measured in family exchange rate of -0.0673 AUC per
nat rather than the -0.1904 that `w4_arcurve` fitted along a training trajectory.
0.05 MATERIAL, 0.02 to 0.05 PARTIAL, below 0.02 not the bottleneck. A family
whose shuf control reaches 0.005 is unreadable.

THE LIMITATION THAT TRAVELS WITH A NULL. x is the output of a trunk trained WITH
this input structure, so it has had every incentive to precompute whatever it
needed, which biases this AGAINST finding a residue. A positive reading is
strong. A null says the residue is small GIVEN a trunk fitted to this input, and
must be reported in those words.

DIAGNOSTIC ONLY. No checkpoint is written, no serving path changes, the frozen
model is never stepped.

Usage:
    cd /mnt/c/Users/aaron/Code/mouse-trajectory-synthesis
    NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW AVX512DQ \
        AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_trunkcap.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

for p in (".", "research", "research/autoloop", "training"):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.event_ar import (  # noqa: E402
    N_DT_CLASSES, DT_BOS_CLASS, EventARModel, S_BOS_CLASS, TH_BOS_CLASS,
    prefix_state,
)
from models.event_stream_polar import (  # noqa: E402
    N_S_CLASSES, N_TH_CLASSES, S_PAD_CLASS, TICK_CLASS,
)
from train_event_ar import ARDataset  # noqa: E402
from w4_rollout import GATE_C, gpu_temp  # noqa: E402

TRAIN_PICK_SEED = 123          # must match w4_headcap, same held out pool
N_TRAIN_DEFAULT = 1_500_000

AUC_PER_NAT = 0.0673           # in family rate from the coupled head grid,
                               # se 0.0047. NOT the 0.1904 arcurve rate, which
                               # this workstream now treats as an upper bound.
MATERIAL_NATS = 0.05
PARTIAL_NATS = 0.02
SHUF_BAR = 0.005               # a family whose width control reaches this is
                               # unreadable, whatever its joint residue says
POSCTL_BAR = 0.05              # the th positive control must clear this or the
                               # whole run is unreadable. w4_headcap bought
                               # 0.2342 with the same information at full
                               # resolution; 8 bins should keep most of it.

N_BINS = 8

# (family, streams in order). Every stream is binned to N_BINS before use, so
# the additive and joint arms of a family see identical information.
FAMILIES = {
    "A": ["s_prev", "th_prev", "dt_prev"],
    "B": ["s_prev", "s_prev2"],
}


class Correction(nn.Module):
    """A correction added to the frozen head's logits, output layer zero
    initialised so at step 0 the arm is exactly the checkpoint.

    mode "none"  : MLP(LN(x)), the depth control
    mode "add"   : MLP([LN(x), e_1(b1) ... e_k(bk)]), one table per stream
    mode "joint" : MLP([LN(x), e_j(code)]), one table over the tuple, emitting
                   width_k*d so the MLP input width matches "add" exactly

    vocabs carries the class count per stream, so the same class serves both the
    8 bin arms and the full resolution positive control without a second shape.
    """

    def __init__(self, d, n_out, mode, vocabs, width_k, hidden):
        super().__init__()
        self.mode = mode
        self.norm = nn.LayerNorm(d)
        if mode == "none":
            n_in = d
        elif mode == "add":
            self.emb = nn.ModuleList([nn.Embedding(v, d) for v in vocabs])
            n_in = d * (1 + len(vocabs))
        else:
            self.emb = nn.ModuleList([nn.Embedding(vocabs[0], d * width_k)])
            n_in = d * (1 + width_k)
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_out))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, ctx):
        parts = [self.norm(x)]
        if self.mode == "add":
            parts += [e(c) for e, c in zip(self.emb, ctx)]
        elif self.mode == "joint":
            parts.append(self.emb[0](ctx[0]))
        return self.net(torch.cat(parts, dim=-1))


@torch.no_grad()
def build_cache(model, ds, dev, batch, amp):
    """Cache the frozen trunk output per supervised token, the three targets,
    and the HISTORY streams the corrections are allowed to condition on.

    s_prev/th_prev/dt_prev are exactly what the trunk was fed at this position,
    from the model's own shift_inputs. s_prev2 is that shifted once more, so it
    is the speed two steps back with BOS covering the first two positions.
    """
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0,
                    pin_memory=True, drop_last=False)
    keys = ("x", "s", "th", "dt", "s_prev", "th_prev", "dt_prev", "s_prev2",
            "motion", "row")
    out = {k: [] for k in keys}
    base = 0
    for bi, b in enumerate(dl):
        s_cls, th_cls, dt_cls, n_sup, cond = (t.to(dev, non_blocking=True)
                                              for t in b)
        B, T = s_cls.shape
        s_prev, th_prev, dt_prev = model.shift_inputs(s_cls, th_cls, dt_cls)
        s_prev2 = torch.cat([torch.full((B, 1), S_BOS_CLASS, device=dev,
                                        dtype=torch.long), s_prev[:, :-1]], 1)
        state = prefix_state(s_cls, th_cls, dt_cls, cond)
        with torch.amp.autocast("cuda", enabled=amp):
            x = model.trunk(s_prev, th_prev, dt_prev, state, cond)

        ar = torch.arange(T, device=dev).unsqueeze(0)
        keep = (ar < n_sup.unsqueeze(1)).reshape(-1)
        motion = ((s_cls > TICK_CLASS) & (s_cls < S_PAD_CLASS)).reshape(-1)
        rid = (base + torch.arange(B, device=dev)).unsqueeze(1).expand(B, T)
        out["x"].append(x.reshape(-1, x.shape[-1])[keep].half().cpu())
        for k, v in (("s", s_cls), ("th", th_cls), ("dt", dt_cls),
                     ("s_prev", s_prev), ("th_prev", th_prev),
                     ("dt_prev", dt_prev), ("s_prev2", s_prev2)):
            out[k].append(v.reshape(-1)[keep].int().cpu())
        out["motion"].append(motion[keep].cpu())
        out["row"].append(rid.reshape(-1)[keep].int().cpu())
        base += B
        if bi % 100 == 0:
            print(f"    cache batch {bi}", flush=True)

    c = {k: torch.cat(v) for k, v in out.items()}
    for k in ("s", "th", "dt", "s_prev", "th_prev", "dt_prev", "s_prev2"):
        c[k] = c[k].long()
    c["n_rows"] = base
    return c


def bin_stream(v: torch.Tensor, k: int) -> torch.Tensor:
    """Coarsen a class stream to k quantile bins.

    Quantiles rather than a uniform split because the class distributions are
    very lopsided: th is TH_NULL for every tick token and s piles up at the low
    classes. This is an unsupervised transform of the input marginal, it never
    touches a label, and it is applied identically to the additive and joint
    arms so the contrast between them is unaffected by the choice. Ties collapse
    edges and leave some bins empty, which is harmless; the code range stays
    fixed at k so the joint table size is known in advance.
    """
    a = v.numpy()
    edges = np.quantile(a, np.linspace(0.0, 1.0, k + 1)[1:-1])
    return torch.from_numpy(
        np.searchsorted(edges, a, side="right").astype(np.int64))


@torch.no_grad()
def frozen_logits(model, ch, x, s, th):
    """The checkpoint's own head, recomputed. The class streams here are the
    CURRENT step's, which is what the frozen heads condition on; the correction
    contexts are history and are kept entirely separate."""
    if ch == "s":
        return model.s_head(x)
    if ch == "th":
        return model.th_logits(x.unsqueeze(1), s.unsqueeze(1)).squeeze(1)
    return model.dt_logits(x.unsqueeze(1), s.unsqueeze(1),
                           th.unsqueeze(1)).squeeze(1)


def frozen_loss(model, ch, cache, sel, split, part, target, dev):
    """The frozen head's loss on one split, so every gain below is paired on
    identical tokens rather than quoted against a corpus number."""
    j_all = torch.nonzero(sel & (split == part), as_tuple=True)[0]
    tot, n = 0.0, 0
    for c0 in range(0, len(j_all), 8192):
        j = j_all[c0:c0 + 8192]
        z = frozen_logits(model, ch, cache["x"][j].to(dev).float(),
                          cache["s"][j].to(dev), cache["th"][j].to(dev))
        ce = F.cross_entropy(z.float(), cache[target][j].to(dev),
                             reduction="none")
        tot += float(ce.sum()); n += len(j)
    return tot / max(n, 1)


def fit_arm(corr, model, ch, cache, sel, split, dev, a, ctx_keys, target,
            label):
    corr = corr.to(dev)
    opt = torch.optim.AdamW(corr.parameters(), lr=a.lr, weight_decay=0.0)
    idx = {k: torch.nonzero(sel & (split == v), as_tuple=True)[0]
           for k, v in (("fit", 0), ("dev", 1), ("eval", 2))}

    def z_of(j):
        return frozen_logits(model, ch, cache["x"][j].to(dev).float(),
                             cache["s"][j].to(dev),
                             cache["th"][j].to(dev)).float()

    def run(part, want_vec=False):
        corr.eval()
        tot, n, vec = 0.0, 0, []
        with torch.no_grad():
            for c0 in range(0, len(idx[part]), 8192):
                j = idx[part][c0:c0 + 8192]
                x = cache["x"][j].to(dev).float()
                ctx = [cache[k][j].to(dev) for k in ctx_keys]
                ce = F.cross_entropy(z_of(j) + corr(x, ctx),
                                     cache[target][j].to(dev),
                                     reduction="none")
                tot += float(ce.sum()); n += len(j)
                if want_vec:
                    vec.append(ce.cpu())
        return tot / max(n, 1), (torch.cat(vec) if want_vec else None)

    d0, _ = run("dev")
    best, best_state = d0, {k: v.detach().clone()
                            for k, v in corr.state_dict().items()}
    for ep in range(a.epochs):
        corr.train()
        perm = idx["fit"][torch.randperm(len(idx["fit"]))]
        for c0 in range(0, len(perm), a.fit_batch):
            j = perm[c0:c0 + a.fit_batch]
            x = cache["x"][j].to(dev).float()
            ctx = [cache[k][j].to(dev) for k in ctx_keys]
            loss = F.cross_entropy(z_of(j) + corr(x, ctx),
                                   cache[target][j].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        d, _ = run("dev")
        star = ""
        if d < best:
            best, star = d, "  *"
            best_state = {k: v.detach().clone()
                          for k, v in corr.state_dict().items()}
        print(f"    {label:<18} epoch {ep + 1}/{a.epochs}  dev {d:.4f}{star}",
              flush=True)
    corr.load_state_dict(best_state)
    ev, vec = run("eval", want_vec=True)
    return ev, vec, idx["eval"], d0


def clustered(diff, rows):
    """One reading per trajectory. Tokens inside a trajectory are not
    independent, so the standard error is over trajectories."""
    order = torch.argsort(rows)
    d, r = diff[order].double(), rows[order]
    starts = torch.nonzero(
        torch.cat([torch.ones(1, dtype=torch.bool), r[1:] != r[:-1]]),
        as_tuple=True)[0]
    ends = torch.cat([starts[1:], torch.tensor([len(d)])])
    cs = torch.cat([torch.zeros(1, dtype=torch.float64), d.cumsum(0)])
    sums = cs[ends] - cs[starts]
    cnts = (ends - starts).double()
    per = sums / cnts
    w = cnts / cnts.sum()
    mean = float((per * w).sum())
    se = float(((w * w * (per - mean) ** 2).sum()) ** 0.5)
    return mean, se, len(starts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fit-batch", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="research/w4_trunkcap.json")
    ap.add_argument("--channels", default="s,th,dt")
    ap.add_argument("--families", default="A,B")
    a = ap.parse_args()

    t = gpu_temp()
    if t > GATE_C:
        print(f"  GPU at {t}C, above the {GATE_C}C launch gate. Not starting.")
        return
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)

    s2 = np.load("training/events_s2.npy", mmap_mode="r")
    dth = np.load("training/events_dth.npy", mmap_mode="r")
    dtms = np.load("training/events_dt.npy", mmap_mode="r")
    lengths = np.load("training/events_len.npy")
    cond = np.load("training/events_cond.npy")
    N = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(N, min(N, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(N), trained)
    idx = np.sort(np.random.default_rng(a.seed)
                  .choice(held, min(a.n, len(held)), replace=False))

    ck = torch.load(f"training/{a.ckpt}", map_location=dev, weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)
    d_model = ck["config"]["d_model"]
    print(f"\n  {a.ckpt}, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
          f"FROZEN, no step is ever taken")
    print(f"  {len(idx):,} held out trajectories, never seen in training")
    print(f"  emit order {model.emit_order}, d_model {d_model}")
    print(f"  BOS classes s {S_BOS_CLASS} th {TH_BOS_CLASS} dt {DT_BOS_CLASS}\n",
          flush=True)

    ds = ARDataset(s2[idx], dth[idx], dtms[idx], lengths[idx], cond[idx],
                   ck["config"]["max_seq_len"])
    cache = build_cache(model, ds, dev, a.batch, dev.type == "cuda")
    model.layers = None
    torch.cuda.empty_cache()
    n_tok = len(cache["row"])
    print(f"\n  cached {n_tok:,} supervised tokens from "
          f"{cache['n_rows']:,} trajectories", flush=True)

    # Bin every history stream once, then build the per family tuple codes and
    # their permuted twins. The permutation is drawn once and shared across
    # families and channels, so the width control is the same control everywhere.
    print(f"  binning history streams to {N_BINS} quantile bins")
    binned = {}
    for k in ("s_prev", "th_prev", "dt_prev", "s_prev2"):
        binned[k] = bin_stream(cache[k], N_BINS)
        occ = int(binned[k].bincount(minlength=N_BINS).gt(0).sum())
        print(f"    {k:<9} {occ}/{N_BINS} bins occupied")
    perm = torch.from_numpy(
        np.random.default_rng(a.seed + 7).permutation(n_tok))
    # The positive control's stream: the CURRENT step's speed, binned the same
    # way. Only ever handed to the th head, where it is a legitimate conditioner
    # the frozen head already receives additively. It would be the label itself
    # for the s head and is never offered there.
    cache["P_b0"] = bin_stream(cache["s"], N_BINS)

    fams = [f.strip() for f in a.families.split(",")]
    for fam in fams:
        streams = FAMILIES[fam]
        code = torch.zeros(n_tok, dtype=torch.long)
        for st in streams:
            code = code * N_BINS + binned[st]
        cache[f"{fam}_code"] = code
        cache[f"{fam}_code_shuf"] = code[perm].contiguous()
        for i, st in enumerate(streams):
            cache[f"{fam}_b{i}"] = binned[st]
        print(f"    family {fam}: {len(streams)} streams, "
              f"{N_BINS ** len(streams)} codes, "
              f"{int(code.bincount(minlength=N_BINS ** len(streams)).gt(0).sum())}"
              f" occupied", flush=True)

    rs = np.random.default_rng(a.seed + 1)
    lab = torch.from_numpy(
        rs.choice([0, 1, 2], size=cache["n_rows"], p=[0.5, 0.25, 0.25]))
    split = lab[cache["row"].long()]
    all_tok = torch.ones(n_tok, dtype=torch.bool)

    chans = {
        "s": dict(sel=all_tok, target="s", n_out=N_S_CLASSES),
        "th": dict(sel=cache["motion"], target="th", n_out=N_TH_CLASSES),
        "dt": dict(sel=all_tok, target="dt", n_out=N_DT_CLASSES),
    }
    want = [c.strip() for c in a.channels.split(",")]
    chans = {k: v for k, v in chans.items() if k in want}

    res = {}
    for ch, sp in chans.items():
        fz = frozen_loss(model, ch, cache, sp["sel"], split, 2, sp["target"],
                         dev)
        print(f"\n  === {ch} head, FROZEN {fz:.4f} nats on the eval split ===",
              flush=True)

        # (name, mode, vocabs, width_k, cache keys)
        arms = [("R_depth", "none", [], 0, [])]
        if ch == "th":
            # Two positive controls on the SAME information, the current step's
            # speed, differing only in resolution. P_full replicates
            # w4_headcap's R_full and says whether the harness works at all.
            # P_ctl is that same arm coarsened to 8 bins, so P_ctl / P_full is
            # the measured price of the coarsening every history arm pays.
            arms.append(("P_full", "add", [N_S_CLASSES], 1, ["s"]))
            arms.append(("P_ctl", "add", [N_BINS], 1, ["P_b0"]))
            # A_fullres closes the one excuse a null would otherwise leave.
            # The whole previous step at FULL class resolution, no binning at
            # all, additive. If the trunk had dropped anything about the step it
            # was just handed, this arm recovers it with no coarsening to blame.
            # Vocabularies are +1 because the shifted streams carry a BOS class
            # above the ordinary range.
            arms.append(("A_fullres", "add",
                         [S_BOS_CLASS + 1, TH_BOS_CLASS + 1, DT_BOS_CLASS + 1],
                         3, ["s_prev", "th_prev", "dt_prev"]))
        for fam in fams:
            k = len(FAMILIES[fam])
            arms += [
                (f"{fam}_add", "add", [N_BINS] * k, k,
                 [f"{fam}_b{i}" for i in range(k)]),
                (f"{fam}_joint", "joint", [N_BINS ** k], k, [f"{fam}_code"]),
                (f"{fam}_shuf", "joint", [N_BINS ** k], k,
                 [f"{fam}_code_shuf"]),
            ]

        ev, vecs, ei = {}, {}, None
        for name, mode, vocabs, wk, keys in arms:
            # This run fits three times as many arms as w4_headcap, which itself
            # peaked at 78C. The machine crashed on this workload on 2026-08-06,
            # so the standing rule for it is a 79C ceiling, and the natural
            # place to hold is between arms where nothing is in flight.
            while gpu_temp() > 79:
                print(f"    {gpu_temp()}C, holding before {ch}/{name}",
                      flush=True)
                time.sleep(30)
            corr = Correction(d_model, sp["n_out"], mode, vocabs, wk, a.hidden)
            e, v, ei, d0 = fit_arm(corr, model, ch, cache, sp["sel"], split,
                                   dev, a, keys, sp["target"], f"{ch}/{name}")
            ev[name], vecs[name] = e, v
            print(f"    {ch}/{name:<10} eval {e:.4f}   gain over frozen "
                  f"{fz - e:+.4f}   (dev at init {d0:.4f})", flush=True)
            del corr
            torch.cuda.empty_cache()

        rows = cache["row"][ei]
        contr = {}
        for fam in fams:
            m, se, nc = clustered(vecs[f"{fam}_add"] - vecs[f"{fam}_joint"],
                                  rows)
            contr[f"{fam}_joint_residue"] = dict(mean=m, se=se, clusters=nc)
            m, se, nc = clustered(vecs[f"{fam}_shuf"] - vecs[f"{fam}_joint"],
                                  rows)
            contr[f"{fam}_joint_vs_shuf"] = dict(mean=m, se=se, clusters=nc)
            m, se, nc = clustered(vecs["R_depth"] - vecs[f"{fam}_add"], rows)
            contr[f"{fam}_marginal_residue"] = dict(mean=m, se=se, clusters=nc)
        res[ch] = dict(frozen=fz, eval_loss=ev,
                       gain={k: fz - v for k, v in ev.items()},
                       contrasts=contr, n_eval_tokens=int(len(ei)))
        for lbl, v in contr.items():
            print(f"    {ch} {lbl:<22} {v['mean']:+.4f} se {v['se']:.4f}",
                  flush=True)

    # ---- summary -----------------------------------------------------------
    print("\n  nats bought over the frozen head, positive is better")
    hdr = f"  {'channel':<9}{'frozen':>9}{'R_depth':>10}"
    for fam in fams:
        hdr += f"{fam + '_add':>10}{fam + '_joint':>10}{fam + '_shuf':>10}"
    print(hdr)
    for ch in [c for c in ("s", "th", "dt") if c in res]:
        g = res[ch]["gain"]
        line = f"  {ch:<9}{res[ch]['frozen']:>9.4f}{g['R_depth']:>+10.4f}"
        for fam in fams:
            line += (f"{g[f'{fam}_add']:>+10.4f}{g[f'{fam}_joint']:>+10.4f}"
                     f"{g[f'{fam}_shuf']:>+10.4f}")
        print(line)

    have = list(res)

    def summed(key):
        m = sum(res[c]["contrasts"][key]["mean"] for c in have)
        se = sum(res[c]["contrasts"][key]["se"] ** 2 for c in have) ** 0.5
        return m, se

    print("\n  summed over the heads scored")
    summary, worst_shuf, best_fam, best_val = {}, 0.0, None, -1.0
    for fam in fams:
        jr, jse = summed(f"{fam}_joint_residue")
        vs, vse = summed(f"{fam}_joint_vs_shuf")
        mr, mse = summed(f"{fam}_marginal_residue")
        # the conservative reading: a control can only take the claim down
        take, tse = (jr, jse) if jr <= vs else (vs, vse)
        shuf_gain = sum(res[c]["gain"][f"{fam}_shuf"] for c in have)
        worst_shuf = max(worst_shuf, shuf_gain)
        summary[fam] = dict(joint_residue=jr, joint_residue_se=jse,
                            joint_vs_shuf=vs, joint_vs_shuf_se=vse,
                            marginal_residue=mr, marginal_residue_se=mse,
                            taken=take, taken_se=tse, shuf_gain=shuf_gain)
        if take > best_val:
            best_val, best_fam = take, fam
        print(f"    family {fam}  joint residue vs add   {jr:+.4f} se {jse:.4f}")
        print(f"              joint residue vs shuf  {vs:+.4f} se {vse:.4f}")
        print(f"              taken, the smaller     {take:+.4f} se {tse:.4f}")
        print(f"              marginal residue       {mr:+.4f} se {mse:.4f}")
        print(f"              shuf gain over frozen  {shuf_gain:+.4f}"
              f"   (unreadable at or above {SHUF_BAR})")

    posctl = (res["th"]["gain"]["P_full"] if "th" in res else float("nan"))
    posbin = (res["th"]["gain"]["P_ctl"] if "th" in res else float("nan"))
    coarse_keep = posbin / posctl if posctl > 1e-6 else float("nan")
    print(f"\n  POSITIVE CONTROLS, th given the CURRENT step's speed")
    print(f"    full resolution, {N_S_CLASSES} classes   {posctl:+.4f} nats"
          f"   (w4_headcap R_full bought +0.2342)")
    print(f"    coarsened to {N_BINS} bins           {posbin:+.4f} nats")
    print(f"    fraction surviving the coarsening  {coarse_keep:.2f}")
    print(f"    -> every history arm below pays that same price, so divide by "
          f"{coarse_keep:.2f} for a full resolution equivalent")
    print(f"    the harness is readable only if the full arm clears "
          f"{POSCTL_BAR}")

    take = summary[best_fam]["taken"]
    print(f"\n  largest family is {best_fam} at {take:+.4f} nats"
          f"   ({take / coarse_keep:+.4f} full resolution equivalent)"
          if coarse_keep > 1e-6 else
          f"\n  largest family is {best_fam} at {take:+.4f} nats")
    if "th" in res and "A_fullres" in res["th"]["gain"]:
        fr = res["th"]["gain"]["A_fullres"]
        print(f"  the whole previous step at FULL resolution, th head, buys "
              f"{fr:+.4f} nats")
        print(f"    against a depth control of "
              f"{res['th']['gain']['R_depth']:+.4f} on the same head, so no "
              f"coarsening can be blamed for the reading above")
    print(f"  predicted contract AUC movement {AUC_PER_NAT * take:+.4f}"
          f"   (in family rate 0.0673 per nat, se 0.0047)")
    print(f"  the scorer resolves about 0.005 with five paired seeds")

    if not (posctl >= POSCTL_BAR):
        verdict = (f"UNREADABLE. The positive control bought {posctl:+.4f} "
                   f"nats, below the {POSCTL_BAR} bar, on information that "
                   "`w4_headcap` priced at +0.2342 in the same style of "
                   "harness. This run has not measured the trunk; it has "
                   "measured a fitting failure. Nothing below may be claimed "
                   "in either direction.")
    elif summary[best_fam]["shuf_gain"] >= SHUF_BAR:
        verdict = (f"NOT READABLE. Family {best_fam}'s width control bought "
                   f"{summary[best_fam]['shuf_gain']:+.4f} nats, at or above "
                   f"the {SHUF_BAR} bar, so the joint arm's gain cannot be "
                   "separated from the parameters its larger table adds. No "
                   "branch may be claimed.")
    elif take >= MATERIAL_NATS:
        verdict = (f"MATERIAL at {take:+.4f} nats. The trunk does NOT carry the "
                   f"cross step joint in family {best_fam}, and supplying it "
                   "explicitly is worth building. Note this is a LOWER BOUND, "
                   "since both arms were coarsened to 8 bins.")
    elif take >= PARTIAL_NATS:
        verdict = (f"PARTIAL at {take:+.4f} nats. Real but below the funding "
                   "bar on its own, and at the in family exchange rate worth "
                   f"{AUC_PER_NAT * take:.4f} of AUC, under what the scorer "
                   "resolves. Report it; do not fund a training run on it.")
    else:
        verdict = (f"NOT THE BOTTLENECK at {take:+.4f} nats. The frozen trunk "
                   "already carries the cross step joint structure these "
                   "corrections could add. Read the registered limitation: x "
                   "comes from a trunk trained WITH this input structure, so "
                   "it had every incentive to precompute what it needed, which "
                   "biases this against finding a residue. The residue is "
                   "small GIVEN a trunk fitted to this input; that does not "
                   "exclude a jointly trained model with a different input "
                   "structure doing better. This is a null about the trunk, "
                   "not a null about cross step dependence.")
    print(f"\n  VERDICT\n  {verdict}\n")

    json.dump(dict(ckpt=a.ckpt, config=vars(a), n_bins=N_BINS,
                   families={k: v for k, v in FAMILIES.items() if k in fams},
                   n_tokens=int(n_tok), n_trajectories=int(cache["n_rows"]),
                   channels=res, summary=summary, best_family=best_fam,
                   positive_control=posctl, positive_control_binned=posbin,
                   coarsening_keep_fraction=coarse_keep,
                   taken_nats_full_res_equivalent=(
                       take / coarse_keep if coarse_keep > 1e-6 else None),
                   taken_nats=take, predicted_auc=AUC_PER_NAT * take,
                   verdict=verdict, peak_c=gpu_temp(),
                   elapsed_s=round(time.time() - t0, 1)),
              open(a.out, "w"), indent=2)
    print(f"  {gpu_temp()}C, {time.time() - t0:.0f}s, wrote {a.out}")


if __name__ == "__main__":
    main()
