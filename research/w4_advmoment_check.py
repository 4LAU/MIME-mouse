"""Verify that w4_advmoment's chunked surrogate is the exact gradient.

WHY THIS EXISTS. Every earlier arm in this family chunked its relaxed decode on
the grounds that a loss which is a SUM OVER ROWS has an exactly additive
gradient. RESUME job 3b says plainly that a batch mean and standard deviation do
not decompose that way, which is why only the per row arm could be measured like
that. `w4_advmoment`'s objective is a batch statistic and it chunks anyway, by
substituting a surrogate whose coefficients are read off the forward values. That
is a claim, and an unchecked claim about a gradient is exactly the kind of thing
that produced a silently wrong result twice in this workstream.

THE OBSERVATION THAT SETTLES IT. Build a small network, run it on random input,
and compute the gradient of the sorted matching distance two ways: once by
autograd straight through the whole undivided statistic, and once by the chunked
surrogate the arm uses. If the construction is right they agree to floating point
rounding, not merely in direction. A cosine near one is NOT the check; the record
already contains one case where a cosine looked fine and the magnitudes were two
orders of magnitude apart.

No GPU model and no checkpoint. It runs in about a second.
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

from w4_advmoment import w1_terms  # noqa: E402


def reference(lg, lh):
    """The statistic itself, differentiated by autograd end to end."""
    return ((torch.sort(lg, dim=0).values
             - torch.sort(lh, dim=0).values).abs().mean(0)).mean()


def main():
    torch.manual_seed(0)
    n, k, feat, chunk = 96, 8, 12, 24
    net = nn.Sequential(nn.Linear(feat, 32), nn.Tanh(), nn.Linear(32, k))
    inp = torch.randn(n, feat)
    lh = torch.randn(n, k) * 1.7 + 0.4

    net.zero_grad(set_to_none=True)
    reference(net(inp), lh).backward()
    ref = torch.cat([p.grad.reshape(-1).clone() for p in net.parameters()])

    with torch.no_grad():
        w1, coeff = w1_terms(net(inp), lh)
    net.zero_grad(set_to_none=True)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        (coeff[i0:i1] * net(inp[i0:i1])).sum().backward()
    got = torch.cat([p.grad.reshape(-1).clone() for p in net.parameters()])

    with torch.no_grad():
        direct = float(reference(net(inp), lh))
    err = float((got - ref).abs().max())
    rel = err / float(ref.abs().max())
    cos = float(torch.nn.functional.cosine_similarity(got, ref, dim=0))
    print(f"  statistic       {direct:.8f} by autograd, "
          f"{float(w1.mean()):.8f} by w1_terms")
    print(f"  gradient norms  {float(ref.norm()):.8f} reference, "
          f"{float(got.norm()):.8f} chunked")
    print(f"  max abs error   {err:.3e}   relative {rel:.3e}   cosine {cos:.8f}")
    ok = rel < 1e-5 and abs(direct - float(w1.mean())) < 1e-6
    print(f"  {'EXACT' if ok else 'NOT EXACT, do not run the arm'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
