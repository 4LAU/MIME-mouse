"""Check the forward KL anchor, and check the claim it is built on. CPU, a minute.

The first three checks are `w4_klanchor_check`'s, rerun because the expression
changed underneath them: zero between identical models, masks agreeing with
`token_logprob_pos` by count, and a perturbed model giving a strictly positive
term that flows gradient to the live model and none to the base.

The fourth is the one this file exists for and it is not a plumbing check, it is
a check on the DESIGN CLAIM. `w4_kldir`'s whole argument is that forward KL
charges for abandoned mass and reverse KL does not, so only forward can restrain
collapse. That is a statement about the two expressions and it is falsifiable
without a GPU, so it should be falsified or confirmed before any GPU time is
spent on it.

The test simulates collapse directly. SHARPENING a model's logits by a factor
above one is exactly the failure mode: probability mass leaves the tails of every
conditional and concentrates on the mode, which is what the collapse flag reports
in feature space. Under the design claim, sharpening should make forward KL grow
much faster than reverse KL, because forward is weighted by the base's
probabilities, which stay put over the abandoned region, while reverse is
weighted by the live model's, which are vanishing there.

    if forward grows faster than reverse under sharpening   claim holds
    if the two grow alike                                   the claim is wrong
                                                            and w4_kldir should
                                                            not be run

The second outcome would save a GPU run rather than cost one, which is the
point of putting the check here instead of reading it off the arm afterwards.

REVERSE is also checked to reproduce `w4_klanchor.token_kl` bit for bit, so that
`--kl-direction reverse` is genuinely the same control and not a reimplementation
that happens to agree.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "research")

from models.event_ar import EventARModel  # noqa: E402
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TH_NULL_CLASS, TICK_CLASS,
)
from w4_klanchor import token_kl as token_kl_ref  # noqa: E402
from w4_kldir import token_kl  # noqa: E402

CKPT = "training/event_ar_v2_s40000.pt"


class Sharpen(torch.nn.Module):
    """Wraps a model so its logits come out multiplied by `k`.

    Sharpening is inverse temperature. At k above one every conditional
    concentrates on its mode and the tails lose mass, which is collapse in the
    token distribution rather than in feature space. Nothing else about the
    model changes, so any difference the two divergences report is attributable
    to the mass movement alone.
    """

    def __init__(self, inner, k):
        super().__init__()
        self.inner = inner
        self.k = k

    def shift_inputs(self, *a):
        return self.inner.shift_inputs(*a)

    def forward(self, *a, **kw):
        return tuple(x * self.k for x in self.inner(*a, **kw))


def build_batch(cfg, dev):
    n, L = 3, 40
    rng = np.random.default_rng(0)
    s = np.full((n, L), S_PAD_CLASS, dtype=np.int64)
    th = np.full((n, L), TH_NULL_CLASS, dtype=np.int64)
    dt = np.zeros((n, L), dtype=np.int64)
    ends = [12, 25, L]
    for i, e in enumerate(ends):
        for t in range(e):
            if rng.random() < 0.3:
                s[i, t] = TICK_CLASS
            else:
                s[i, t] = int(rng.integers(TICK_CLASS + 1, S_PAD_CLASS))
                th[i, t] = int(rng.integers(0, TH_NULL_CLASS))
            dt[i, t] = int(rng.integers(0, 4))
    return (torch.from_numpy(s).to(dev), torch.from_numpy(th).to(dev),
            torch.from_numpy(dt).to(dev),
            torch.randn(n, cfg["cond_dim"], device=dev), ends, L)


def main():
    torch.manual_seed(0)
    dev = torch.device("cpu")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)

    def fresh():
        m = EventARModel(**ck["config"]).to(dev)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        return m

    model, base = fresh(), fresh()
    for q in base.parameters():
        q.requires_grad_(False)

    s_t, th_t, dt_t, cond, ends, L = build_batch(ck["config"], dev)

    # ONE. Identical models, both directions.
    k0f = float(token_kl(model, base, s_t, th_t, dt_t, cond, True))
    k0r = float(token_kl(model, base, s_t, th_t, dt_t, cond, False))
    print(f"  identical models, forward        {k0f:.3e}")
    print(f"  identical models, reverse        {k0r:.3e}")

    # TWO. Masks, by count, against token_logprob_pos's definition.
    pad = s_t >= S_PAD_CLASS
    first_pad = torch.where(pad.any(1), pad.float().argmax(1),
                            torch.full_like(s_t[:, 0], L - 1))
    pos = torch.arange(L).unsqueeze(0)
    live = pos <= first_pad.unsqueeze(1)
    motion = (s_t > TICK_CLASS) & (s_t < S_PAD_CLASS) & live
    want_live = [e + 1 for e in ends[:2]] + [L]
    sn = s_t.numpy()
    want_motion = [int(((sn[i, :e] > TICK_CLASS)
                        & (sn[i, :e] < S_PAD_CLASS)).sum())
                   for i, e in enumerate(ends)]
    got_live, got_motion = live.sum(1).tolist(), motion.sum(1).tolist()
    print(f"  live positions   want {want_live}  got {got_live}")
    print(f"  motion positions want {want_motion}  got {got_motion}")

    # THREE. Perturbed model, forward direction, gradient reaches only live.
    pert = fresh()
    with torch.no_grad():
        for p in pert.parameters():
            p.add_(torch.randn_like(p) * 0.002)
    pert.zero_grad(set_to_none=True)
    k1 = token_kl(pert, base, s_t, th_t, dt_t, cond, True)
    k1.backward()
    gn = float(torch.sqrt(sum((p.grad ** 2).sum() for p in pert.parameters()
                              if p.grad is not None)))
    base_grads = sum(1 for q in base.parameters() if q.grad is not None)
    print(f"  perturbed model, forward         {float(k1):.6f}")
    print(f"  gradient norm into live model    {gn:.6f}")
    print(f"  parameters of base holding grad  {base_grads}")

    # FOUR. Reverse reproduces w4_klanchor exactly, so the control is the same
    # quantity and not a lookalike.
    ref = float(token_kl_ref(pert, base, s_t, th_t, dt_t, cond))
    rev = float(token_kl(pert, base, s_t, th_t, dt_t, cond, False))
    print(f"  reverse here {rev:.8f} vs w4_klanchor {ref:.8f}  "
          f"diff {abs(rev - ref):.3e}")

    # FIVE. THE DESIGN CLAIM. Sharpening is collapse in the token distribution.
    # Forward must grow faster than reverse, because forward weights the
    # abandoned region by the base's probability, which does not vanish.
    print(f"\n  {'sharpen k':>10}{'forward':>12}{'reverse':>12}"
          f"{'fwd / rev':>12}")
    ratios = []
    for k in (1.0, 1.2, 1.5, 2.0, 3.0, 5.0):
        sm = Sharpen(base, k)
        f = float(token_kl(sm, base, s_t, th_t, dt_t, cond, True))
        r = float(token_kl(sm, base, s_t, th_t, dt_t, cond, False))
        ratios.append(f / r if r > 0 else float("nan"))
        print(f"  {k:>10.1f}{f:>12.4f}{r:>12.4f}"
              f"{(f / r if r > 0 else float('nan')):>12.2f}")

    grows = ratios[-1] > ratios[1] and ratios[-1] > 1.5
    print("\n  forward outgrows reverse under sharpening: "
          f"{'YES, the design claim holds' if grows else 'NO, DO NOT RUN THE ARM'}")

    ok = (k0f < 1e-6 and k0r < 1e-6 and got_live == want_live
          and got_motion == want_motion and float(k1) > 1e-4 and gn > 0
          and base_grads == 0 and abs(rev - ref) < 1e-9 and grows)
    print("  PASS" if ok else "  FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
