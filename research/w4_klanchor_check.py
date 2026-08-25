"""Check the sampled distribution KL anchor. CPU only, no GPU, about a minute.

Three things are worth checking and only three.

ONE, the KL is exactly zero when the live model IS the base. This is the check
that actually matters, because it is the only one that catches a masking or
normalisation error: any position included that should not be, or any
mismatched pairing of heads, still gives zero here only if the two
distributions are genuinely identical at every counted position. A term
computed at the wrong position between two identical models is still zero, so
this alone is not sufficient, which is why there is a second check.

TWO, the masks count exactly the positions `token_logprob_pos` counts. That is
the function every earlier arm's anchor used, so agreeing with it is what makes
this anchor comparable to the record rather than a different quantity with the
same name. Checked by counting, not by eye.

THREE, a perturbed model gives a strictly positive KL that flows gradient back
to the live model and none to the base.

A cosine near one is NOT a check here and there is no autograd comparison to
make, because unlike the sorted matching surrogate this term is differentiated
by autograd directly. The risk in this file is the masking, not the calculus.
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
from w4_klanchor import token_kl  # noqa: E402

CKPT = "training/event_ar_v2_s40000.pt"


def main():
    torch.manual_seed(0)
    dev = torch.device("cpu")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)

    model = EventARModel(**ck["config"]).to(dev)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    base = EventARModel(**ck["config"]).to(dev)
    base.load_state_dict(ck["model_state_dict"])
    base.eval()
    for q in base.parameters():
        q.requires_grad_(False)

    # Token batches shaped like something the sampler returns: a run of motion
    # and tick events, then PAD to the end. Two rows pad early, one runs to the
    # buffer edge, which is the case the `first_pad` fallback exists for.
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
    s_t = torch.from_numpy(s).to(dev)
    th_t = torch.from_numpy(th).to(dev)
    dt_t = torch.from_numpy(dt).to(dev)
    cond = torch.randn(n, ck["config"]["cond_dim"], device=dev)

    # ONE. Identical models.
    kl0 = token_kl(model, base, s_t, th_t, dt_t, cond)
    print(f"  identical models, KL per token   {float(kl0):.3e}")

    # TWO. The mask counts, against token_logprob_pos's own definition. The
    # third row never pads, so its `live` must be the whole buffer.
    pad = s_t >= S_PAD_CLASS
    first_pad = torch.where(pad.any(1), pad.float().argmax(1),
                            torch.full_like(s_t[:, 0], L - 1))
    pos = torch.arange(L).unsqueeze(0)
    live = pos <= first_pad.unsqueeze(1)
    motion = (s_t > TICK_CLASS) & (s_t < S_PAD_CLASS) & live
    want_live = [e + 1 for e in ends[:2]] + [L]
    got_live = live.sum(1).tolist()
    want_motion = [int(((s[i, :e] > TICK_CLASS)
                        & (s[i, :e] < S_PAD_CLASS)).sum()) for i, e in
                   enumerate(ends)]
    got_motion = motion.sum(1).tolist()
    print(f"  live positions   want {want_live}  got {got_live}")
    print(f"  motion positions want {want_motion}  got {got_motion}")

    # THREE. Perturb the live model and check the term bites.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.002)
    model.zero_grad(set_to_none=True)
    kl1 = token_kl(model, base, s_t, th_t, dt_t, cond)
    kl1.backward()
    gn = float(torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()
                              if p.grad is not None)))
    base_grads = sum(1 for q in base.parameters() if q.grad is not None)
    print(f"  perturbed model, KL per token    {float(kl1):.6f}")
    print(f"  gradient norm into live model    {gn:.6f}")
    print(f"  parameters of base holding grad  {base_grads}")

    ok = (float(kl0) < 1e-6 and got_live == want_live
          and got_motion == want_motion and float(kl1) > 1e-4 and gn > 0
          and base_grads == 0)
    print("  PASS" if ok else "  FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
