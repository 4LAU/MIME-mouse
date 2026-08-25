"""What does the WITHIN STEP head restriction cost, in nats? Registered before running.

WHY THIS EXISTS. On 2026-08-13 the rollout gradient was measured absent rather
than noisy, which closed the score function family at any affordable batch and
with it the record's own designated next arm. Five other roads were already
shut: the representation is not the ceiling (`w4_token_ceiling` round trips real
human tokens to 0.5118 against a floor of 0.467 to 0.497), more data is excluded
(`w4_arfit`), more capacity is excluded at any payable price (`w4_arbench`,
`w4_arcurve`), token space edits price at 0.012 of 0.107 across five separate
defects, and feature conditioning is closed. What HANDOFF named as the single
surviving lever is "the representation and the factorisation itself, meaning
what the tokens are and what the model is asked to predict at each step".

This file does not build a factorisation. It PRICES one, in the currency the
exchange rate is denominated in, before any GPU hours are spent on training.

THE RESTRICTION BEING PRICED. `models/event_ar.py` factorises each step as
p(s) p(th | s) p(dt | s, th) and implements the two conditional heads as

    th_logits = th_head(th_norm(x + s_ctx_embed(s)))

so the emitted speed reaches the direction head as a vector ADDED to the trunk
state, followed by one LayerNorm and one linear map. Because the head is linear,
the effect of s on the direction logits is an additive shift in logit space,
modulated only by the single scalar 1/sd(x + e_s) the norm contributes. The
model therefore cannot represent an effect of the emitted speed on the direction
distribution that DEPENDS ON THE HISTORY. It can only apply one fixed logit
offset per speed class, everywhere, rescaled by one number.

The trunk state x is causal and is computed BEFORE the current step's tokens
exist, so x cannot know s(t). That is what makes the separation below clean:
every route from s(t) to the direction logits passes through the additive term.

An interleaved factorisation, one sequence position per CHANNEL rather than per
event, removes the restriction completely: the whole trunk, at full depth and
with full attention, computes p(th | s, history). It is an exact refactorisation
of the same joint, so its held out likelihood is directly comparable, summed
over the same tokens. It costs three times the sequence length, so nine times
the attention, and it is a real training run. This file decides whether that run
is worth funding.

THIS IS NOT A REOPENING OF THE CLOSED ADDITIVE CONDITIONING SUSPECT, and the
distinction matters because the record warns about exactly this move.
`w4_coupletok` closed additive within event conditioning on Spearman rank
correlations that scattered around one, ratios 0.86 to 1.12 with every sign
matching, and that closure STANDS. The FiLM rewrite of `th_head` and `dt_head`
remains NOT AUTHORISED and nothing here authorises it. Two things separate this
from that closure. First, the instrument: Spearman is a monotone rank statistic
over all events in a trajectory, so it is dominated by the typical ninety
percent, while this measures the actual training objective in nats, which is
where the exchange rate lives. Second, the intervention: FiLM gates a head, and
what is being priced here is moving the conditioning INTO THE TRUNK by changing
what a token is, which is the lever HANDOFF named and a strictly more expressive
change. If this reads null, FiLM is not rescued by it.

THE DESIGN, and it is residual on purpose. An earlier version of this file
fitted four head shapes from scratch on cached trunk outputs and compared them
to each other. Its smoke test killed it: fitted with a realistic budget the
control shape reached 4.99 nats against the checkpoint's own 1.16, because the
frozen head was fitted jointly with the trunk over sixty million tokens and no
affordable refit reproduces that. Comparing three badly fitted heads to each
other measures the fitting budget, not the architecture.

So every arm here is a CORRECTION ADDED TO THE FROZEN HEAD, with the
correction's output layer initialised to exactly zero. Every arm therefore
starts at the checkpoint's own loss by construction, and what is fitted is only
the improvement. The validity check is no longer a hurdle to clear, it is an
identity at step zero.

    R_same    frozen + linear(LN(x + e))     same function class as the frozen
                                             head. Should buy about nothing. If
                                             it buys a lot the frozen head is
                                             simply not converged and every
                                             other row below is contaminated.
    R_depth   frozen + MLP(LN(x))            what more head depth buys on the
                                             SAME information. x cannot see
                                             s(t), so this correction cannot
                                             use the current step at all.
    R_full    frozen + MLP([LN(x), e])       what depth PLUS a non additive use
                                             of the current step buys.
    R_shuf    frozen + MLP([LN(x), e(perm)]) THE WIDTH CONTROL. Identical to
                                             R_full in every dimension and
                                             parameter count, but the context
                                             stream is a fixed permutation of
                                             itself, so it carries the same
                                             marginal and no information about
                                             the token it sits beside. If
                                             R_full's gain were bought by the
                                             wider first layer rather than by
                                             knowing s(t), this arm buys it too.

    INTERACTION = R_depth loss - R_full loss

Both R_depth and R_shuf are valid baselines for that contrast and they price
different confounds, depth in one case and depth plus width in the other. The
verdict runs on whichever of the two gives the SMALLER interaction, so a
control can only ever take the claim down. R_shuf was added after the n 8000
smoke test returned a th interaction five times the registered prediction; a
number that far outside its own forecast gets a control before it gets quoted.

That difference is the quantity. It is what the additive restriction costs, net
of head depth, because both arms have identical depth and differ only in whether
the correction may see s(t).

THRESHOLDS, from the exchange rate and not from taste. `w4_arcurve` measured
0.1904 contract AUC per nat, r 0.953, residual sd 0.0131, and the distance from
`event_ar_v2_s40000` to 0.50 is 0.65 to 0.80 nats. The record's own standing bar
is that anything worth less than 0.05 nats is worth less than 0.01 AUC.
Interaction is summed over the th and dt heads, since both carry the restriction.

    >= 0.05 nats    MATERIAL. The additive restriction is a real bottleneck and
                    the interleaved factorisation is worth its training run.
    0.02 to 0.05    PARTIAL. Report it, do not fund a training run on it alone.
    < 0.02 nats     NOT THE BOTTLENECK, given a trunk trained for it. Do not
                    build the interleaved factorisation. Look at the trunk.

    R_same >= 0.02  NOT READABLE, whatever the other rows say. The frozen head
                    is not at an optimum of its own function class and the
                    contrast is measuring slack rather than architecture.

THE LIMITATION THAT TRAVELS WITH A NULL, registered here so it cannot be added
afterwards. x is the output of a trunk that was TRAINED with the restricted head
attached, so the trunk has had every incentive to precompute whatever the
restricted head could not do for itself. That biases this measurement AGAINST
finding an interaction. A positive reading is therefore strong and a null
reading is suggestive rather than conclusive: it says the restriction costs
little GIVEN a trunk fitted to it, which does not exclude that a jointly trained
interleaved model does better. Any null must be reported in those words.

PREDICTION, ON THE RECORD, BEFORE THE RUN. PARTIAL, and specifically an
interaction near 0.023 nats. The reasoning is arithmetic rather than taste.
`w4_couple` localised the conditioning defect to the top decile of surprise,
which carries between 0.71 and 0.92 of the covariance, and the record quotes the
excess reaching 0.226 nats in that top decile. A defect confined to about ten
percent of tokens at that per token size is 0.10 * 0.226 = 0.023 nats when
averaged over all tokens, which lands in the PARTIAL band and below the funding
bar. Second prediction: the depth control R_depth is under 0.01 nats, because a
ten layer trunk read by a linear head is the standard arrangement and is rarely
depth limited at the readout. Third: R_same is under 0.005. This file's record
on predictions is 5 of 13 and its own standing advice is not to trust the guess.

DIAGNOSTIC ONLY. No checkpoint is written, no serving path changes, and the
frozen model is never stepped.

Usage:
    cd /mnt/c/Users/aaron/Code/mouse-trajectory-synthesis
    NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW AVX512DQ \
        AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_headcap.py
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
    EventARModel, N_DT_CLASSES, prefix_state,
)
from models.event_stream_polar import (  # noqa: E402
    N_S_CLASSES, N_TH_CLASSES, S_PAD_CLASS, TICK_CLASS,
)
from train_event_ar import ARDataset  # noqa: E402
from w4_rollout import GATE_C, gpu_temp  # noqa: E402

TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000

AUC_PER_NAT = 0.1904          # w4_arcurve, r 0.953, residual sd 0.0131
MATERIAL_NATS = 0.05
PARTIAL_NATS = 0.02
SLACK_NATS = 0.02             # R_same above this makes nothing readable


class Correction(nn.Module):
    """A correction added to the frozen head's logits. The output layer is zero
    initialised, so at step 0 this module is exactly the identity and the arm's
    loss is exactly the checkpoint's own."""

    def __init__(self, d, n_out, ctx_sizes, hidden, deep):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(v, d) for v in ctx_sizes])
        self.norm = nn.LayerNorm(d)
        self.deep = deep
        if deep:
            self.net = nn.Sequential(
                nn.Linear(d * (1 + len(ctx_sizes)), hidden),
                nn.GELU(),
                nn.Linear(hidden, n_out),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        else:
            # same function class as the frozen head: sum the context in, one
            # norm, one linear map
            self.net = nn.Linear(d, n_out)
            nn.init.zeros_(self.net.weight)
            nn.init.zeros_(self.net.bias)

    def forward(self, x, ctx):
        if self.deep:
            parts = [self.norm(x)] + [e(c) for e, c in zip(self.emb, ctx)]
            return self.net(torch.cat(parts, dim=-1))
        for e, c in zip(self.emb, ctx):
            x = x + e(c)
        return self.net(self.norm(x))


@torch.no_grad()
def build_cache(model, ds, dev, batch, amp):
    """Cache only the frozen trunk output per supervised token. The frozen heads
    are three small modules and are kept live instead of cached, because their
    logits are 540 floats a token and caching them costs more memory than the
    trunk state they are computed from."""
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0,
                    pin_memory=True, drop_last=False)
    out = {k: [] for k in ("x", "s", "th", "dt", "motion", "row")}
    base = 0
    for bi, b in enumerate(dl):
        s_cls, th_cls, dt_cls, n_sup, cond = (t.to(dev, non_blocking=True)
                                              for t in b)
        B, T = s_cls.shape
        s_prev, th_prev, dt_prev = model.shift_inputs(s_cls, th_cls, dt_cls)
        state = prefix_state(s_cls, th_cls, dt_cls, cond)
        with torch.amp.autocast("cuda", enabled=amp):
            x = model.trunk(s_prev, th_prev, dt_prev, state, cond)

        ar = torch.arange(T, device=dev).unsqueeze(0)
        keep = (ar < n_sup.unsqueeze(1)).reshape(-1)
        motion = ((s_cls > TICK_CLASS) & (s_cls < S_PAD_CLASS)).reshape(-1)
        rid = (base + torch.arange(B, device=dev)).unsqueeze(1).expand(B, T)
        out["x"].append(x.reshape(-1, x.shape[-1])[keep].half().cpu())
        for k, v in (("s", s_cls), ("th", th_cls), ("dt", dt_cls)):
            out[k].append(v.reshape(-1)[keep].short().cpu())
        out["motion"].append(motion[keep].cpu())
        out["row"].append(rid.reshape(-1)[keep].int().cpu())
        base += B
        if bi % 50 == 0:
            print(f"    cache batch {bi}", flush=True)

    c = {k: torch.cat(v) for k, v in out.items()}
    for k in ("s", "th", "dt"):
        c[k] = c[k].long()
    c["n_rows"] = base
    return c


@torch.no_grad()
def frozen_logits(model, ch, x, s, th):
    """The checkpoint's own head, recomputed. x is (n, d) so the class streams
    are unsqueezed to the (n, 1) the head helpers expect and squeezed back."""
    if ch == "s":
        return model.s_head(x)
    if ch == "th":
        return model.th_logits(x.unsqueeze(1), s.unsqueeze(1)).squeeze(1)
    return model.dt_logits(x.unsqueeze(1), s.unsqueeze(1),
                           th.unsqueeze(1)).squeeze(1)


def frozen_loss(model, ch, cache, sel, split, part, target, dev):
    """The frozen head's loss on ONE split. Every gain below is quoted against
    the eval split's own frozen number, never against a whole corpus number, so
    the difference is a paired comparison on identical tokens."""
    j_all = torch.nonzero(sel & (split == part), as_tuple=True)[0]
    tot, n, vec = 0.0, 0, []
    for c0 in range(0, len(j_all), 8192):
        j = j_all[c0:c0 + 8192]
        z = frozen_logits(model, ch, cache["x"][j].to(dev).float(),
                          cache["s"][j].to(dev), cache["th"][j].to(dev))
        ce = F.cross_entropy(z.float(), cache[target][j].to(dev),
                             reduction="none")
        tot += float(ce.sum()); n += len(j); vec.append(ce.cpu())
    return tot / max(n, 1), torch.cat(vec) if vec else torch.zeros(0)


# Speed bands for the mechanism test, registered in
# /home/aaronadmin/w4_arms/headcap_mechanism_prereg.md before this ran.
# The claim is that the admissible turn angles form a comb whose spacing is
# set by the emitted speed and whose phase is set by the current heading, so
# the interaction must be concentrated where that comb is coarse.
BANDS = [
    ("LOW  s2<=5",   1, 24),    # at most 8 admissible lattice headings
    ("MID  5<s2<=16", 25, 40),  # 8 to 12 admissible headings
    ("HIGH s2>16",   41, 129),  # effectively dense, no constraint
]


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
        print(f"    {label:<16} epoch {ep + 1}/{a.epochs}  dev {d:.4f}{star}",
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
    ap.add_argument("--out", default="research/w4_headcap.json")
    ap.add_argument("--channels", default="s,th,dt",
                    help="restrict to a subset, for a cheap rerun")
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
    print(f"  emit order {model.emit_order}, d_model {d_model}\n", flush=True)

    ds = ARDataset(s2[idx], dth[idx], dtms[idx], lengths[idx], cond[idx],
                   ck["config"]["max_seq_len"])
    cache = build_cache(model, ds, dev, a.batch, dev.type == "cuda")
    model.layers = None          # the trunk is done; the heads stay live
    torch.cuda.empty_cache()
    n_tok = len(cache["row"])
    print(f"\n  cached {n_tok:,} supervised tokens from "
          f"{cache['n_rows']:,} trajectories", flush=True)

    # The width control. R_full's first layer is twice as wide as R_depth's,
    # so a gain could in principle be bought by parameters rather than by
    # knowing s(t). These streams have the identical marginal distribution and
    # the identical embedding and layer shapes, and carry no information about
    # the token they sit beside, so R_shuf prices the width and nothing else.
    perm = torch.from_numpy(
        np.random.default_rng(a.seed + 7).permutation(n_tok))
    for k in ("s", "th"):
        cache[k + "_shuf"] = cache[k][perm].contiguous()

    rs = np.random.default_rng(a.seed + 1)
    lab = torch.from_numpy(
        rs.choice([0, 1, 2], size=cache["n_rows"], p=[0.5, 0.25, 0.25]))
    split = lab[cache["row"].long()]
    all_tok = torch.ones(n_tok, dtype=torch.bool)

    chans = {
        "s": dict(sel=all_tok, target="s", n_out=N_S_CLASSES,
                  ctx=[], sizes=[]),
        "th": dict(sel=cache["motion"], target="th",
                   n_out=N_TH_CLASSES, ctx=["s"], sizes=[N_S_CLASSES]),
        "dt": dict(sel=all_tok, target="dt", n_out=N_DT_CLASSES,
                   ctx=["s", "th"], sizes=[N_S_CLASSES, N_TH_CLASSES]),
    }

    want = [c.strip() for c in a.channels.split(",")]
    chans = {k: v for k, v in chans.items() if k in want}
    res = {}
    for ch, sp in chans.items():
        fz, fz_vec = frozen_loss(model, ch, cache, sp["sel"], split, 2,
                                 sp["target"], dev)
        print(f"  === {ch} head, FROZEN {fz:.4f} nats on the eval split ===",
              flush=True)
        arms = {
            "R_depth": (Correction(d_model, sp["n_out"], [], a.hidden, True),
                        []),
        }
        if sp["ctx"]:
            arms["R_same"] = (Correction(d_model, sp["n_out"], sp["sizes"],
                                         a.hidden, False), sp["ctx"])
            arms["R_full"] = (Correction(d_model, sp["n_out"], sp["sizes"],
                                         a.hidden, True), sp["ctx"])
            arms["R_shuf"] = (Correction(d_model, sp["n_out"], sp["sizes"],
                                         a.hidden, True),
                              [k + "_shuf" for k in sp["ctx"]])
        ev, vecs, ei = {}, {}, None
        for name, (corr, keys) in arms.items():
            e, v, ei, d0 = fit_arm(corr, model, ch, cache, sp["sel"], split,
                                   dev, a, keys, sp["target"], f"{ch}/{name}")
            ev[name], vecs[name] = e, v
            print(f"    {ch}/{name:<8} eval {e:.4f}   gain over frozen "
                  f"{fz - e:+.4f}   (dev at init {d0:.4f})", flush=True)
            del corr
            torch.cuda.empty_cache()

        rows = cache["row"][ei]
        contr = {}
        if sp["ctx"]:
            m, se, nc = clustered(vecs["R_depth"] - vecs["R_full"], rows)
            contr["interaction"] = dict(mean=m, se=se, clusters=nc)
            m, se, nc = clustered(vecs["R_shuf"] - vecs["R_full"], rows)
            contr["width_matched"] = dict(mean=m, se=se, clusters=nc)

            # THE MECHANISM TEST. Split the same eval tokens by the speed the
            # step actually realised. The comb account says the interaction
            # lives where the admissible set is small; a flat profile refutes
            # it and the account must be withdrawn.
            s_ev = cache["s"][ei]
            diff = vecs["R_depth"] - vecs["R_full"]
            print(f"    {ch} interaction by realised speed band")
            print(f"      {'band':<15}{'tokens':>9}{'frozen':>9}"
                  f"{'interaction':>13}{'se':>8}")
            bands = {}
            for name, lo, hi in BANDS:
                b = (s_ev >= lo) & (s_ev <= hi)
                if int(b.sum()) < 100:
                    continue
                bm, bse, _ = clustered(diff[b], rows[b])
                bands[name] = dict(mean=bm, se=bse, n=int(b.sum()),
                                   frozen=float(fz_vec[b].mean()))
                print(f"      {name:<15}{int(b.sum()):>9,}"
                      f"{float(fz_vec[b].mean()):>9.4f}{bm:>13.4f}"
                      f"{bse:>8.4f}", flush=True)
            contr["bands"] = bands
        res[ch] = dict(frozen=fz, eval_loss=ev,
                       gain={k: fz - v for k, v in ev.items()},
                       contrasts=contr, n_eval_tokens=int(len(ei)))
        for lbl, v in contr.items():
            if lbl == "bands":
                continue
            print(f"    {ch} {lbl:<12} {v['mean']:+.4f} se {v['se']:.4f}",
                  flush=True)
        print(flush=True)

    print("  nats bought over the frozen head, positive is better")
    print(f"  {'channel':<9}{'frozen':>9}{'R_same':>10}{'R_depth':>10}"
          f"{'R_shuf':>10}{'R_full':>10}{'INTERACTION':>14}")
    for ch in [c for c in ("s", "th", "dt") if c in res]:
        g = res[ch]["gain"]
        c = res[ch]["contrasts"]
        def f(k, src=g):
            return f"{src[k]:+.4f}" if k in src else "     .   "
        i = (f"{c['interaction']['mean']:+.4f}" if "interaction" in c
             else "     .   ")
        print(f"  {ch:<9}{res[ch]['frozen']:>9.4f}{f('R_same'):>10}"
              f"{f('R_depth'):>10}{f('R_shuf'):>10}{f('R_full'):>10}{i:>14}")

    have = [c for c in ("th", "dt") if c in res]

    def summed(key):
        m = sum(res[c]["contrasts"][key]["mean"] for c in have)
        se = sum(res[c]["contrasts"][key]["se"] ** 2 for c in have) ** 0.5
        return m, se

    raw, rse = summed("interaction")
    wid, wse = summed("width_matched")
    # The verdict runs on whichever contrast is smaller. A width control can
    # only ever take the claim down, never prop it up.
    inter, ise = (raw, rse) if raw <= wid else (wid, wse)
    slack = max(res[c]["gain"]["R_same"] for c in have)
    depth = res["s"]["gain"]["R_depth"] if "s" in res else float("nan")

    print(f"\n  interaction vs R_depth              {raw:+.4f} se {rse:.4f}")
    print(f"  interaction vs R_shuf, width matched {wid:+.4f} se {wse:.4f}")
    print(f"  taken forward, the conservative one  {inter:+.4f} se {ise:.4f}")
    print(f"  predicted contract AUC              "
          f"{AUC_PER_NAT * inter:+.4f}  "
          f"({AUC_PER_NAT} per nat, fit residual sd 0.0131)")
    print(f"  head depth alone, s channel         {depth:+.4f}")
    print(f"  slack in the frozen head, R_same    {slack:+.4f}"
          f"   (readable below {SLACK_NATS})")

    if slack >= SLACK_NATS:
        verdict = (f"NOT READABLE. R_same buys {slack:+.4f} nats, at or above "
                   f"the {SLACK_NATS} bar, so the frozen head is not at an "
                   "optimum of its own function class on held out data and "
                   "every contrast above is measuring slack rather than "
                   "architecture. No branch may be claimed.")
    elif inter >= MATERIAL_NATS:
        verdict = ("MATERIAL. The additive within step restriction costs at "
                   f"{inter:.4f} nats net of head depth, at or above the 0.05 "
                   "bar. The arm that buys it is R_full, which is one MLP on "
                   "the concatenation in place of the additive embedding, so "
                   "the fix is in the HEAD and needs no extra sequence length "
                   "and no extra attention. Fund that, not the interleaved "
                   "factorisation.")
    elif inter >= PARTIAL_NATS:
        verdict = (f"PARTIAL. The restriction is real at {inter:.4f} nats and "
                   "is below the funding bar on its own. Report it; do not "
                   "fund a training run on this alone.")
    else:
        verdict = (f"NOT THE BOTTLENECK at {inter:.4f} nats, given a trunk "
                   "trained for it. Do not build the interleaved "
                   "factorisation on this evidence. Read the registered "
                   "limitation: x comes from a trunk fitted WITH the "
                   "restricted head, which biases this measurement against "
                   "finding an interaction, so this is suggestive rather than "
                   "conclusive.")
    print(f"\n  VERDICT\n  {verdict}\n")

    json.dump(dict(ckpt=a.ckpt, config=vars(a), n_tokens=int(n_tok),
                   n_trajectories=int(cache["n_rows"]), channels=res,
                   interaction_total=inter, interaction_total_se=ise,
                   interaction_vs_depth=raw, interaction_vs_depth_se=rse,
                   interaction_vs_shuf=wid, interaction_vs_shuf_se=wse,
                   predicted_auc=AUC_PER_NAT * inter, depth_control=depth,
                   slack=slack, verdict=verdict, peak_c=gpu_temp(),
                   elapsed_s=round(time.time() - t0, 1)),
              open(a.out, "w"), indent=2)
    print(f"  {gpu_temp()}C, {time.time() - t0:.0f}s, wrote {a.out}")


if __name__ == "__main__":
    main()
