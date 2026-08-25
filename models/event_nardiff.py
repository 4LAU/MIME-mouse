"""Non autoregressive event path model. Absorbing state discrete diffusion.

Registered in /home/aaronadmin/w4_arms/nardiff_prereg.md.

The AR model in `event_ar.py` samples event 1, then event 2 conditioned on what
it already emitted, so a per step bias compounds over about 39 events. This
model produces every event at once from a fully masked sequence and refines it
over a fixed number of passes. Nothing is fed back, so nothing can compound.

Capacity is matched to `event_ar_v2` on purpose. `w4_arcurve` priced capacity,
and leaving it free would confound the architecture change with a capacity one.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.event_stream_polar import (N_S_CLASSES, N_TH_CLASSES,
                                       S_PAD_CLASS, TH_NULL_CLASS)
from models.event_ar import N_DT_CLASSES

# absorbing MASK id sits one past each channel's vocabulary
S_MASK, TH_MASK, DT_MASK = N_S_CLASSES, N_TH_CLASSES, N_DT_CLASSES
VOCABS = (N_S_CLASSES, N_TH_CLASSES, N_DT_CLASSES)
MASKS = (S_MASK, TH_MASK, DT_MASK)


def timestep_embedding(t, dim):
    """Standard sinusoidal embedding of the continuous corruption level."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0)
                      * torch.arange(half, device=t.device).float() / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)


class EventNARDiff(nn.Module):
    def __init__(self, d_model=384, n_heads=6, n_layers=10, d_ff=1888,
                 max_seq_len=256, cond_dim=4, dropout=0.1, cond_dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.cond_dropout = cond_dropout
        self.d_model = d_model

        # one embedding table per channel, each with its own MASK row
        self.emb_s = nn.Embedding(N_S_CLASSES + 1, d_model)
        self.emb_th = nn.Embedding(N_TH_CLASSES + 1, d_model)
        self.emb_dt = nn.Embedding(N_DT_CLASSES + 1, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)

        self.cond_proj = nn.Sequential(nn.Linear(cond_dim, d_model), nn.GELU(),
                                       nn.Linear(d_model, d_model))
        self.t_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                    nn.Linear(d_model, d_model))

        # BIDIRECTIONAL. no causal mask anywhere, that is the whole point
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.head_s = nn.Linear(d_model, N_S_CLASSES)
        self.head_th = nn.Linear(d_model, N_TH_CLASSES)
        self.head_dt = nn.Linear(d_model, N_DT_CLASSES)
        # length is its own problem. supervising 256 mostly PAD positions would
        # spend nearly all the capacity learning to predict PAD.
        self.head_len = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                      nn.Linear(d_model, max_seq_len))

    def _cond_vec(self, cond, training):
        c = self.cond_proj(cond)
        if training and self.cond_dropout > 0:
            keep = (torch.rand(len(c), 1, device=c.device)
                    >= self.cond_dropout).float()
            c = c * keep
        return c

    def length_logits(self, cond):
        return self.head_len(self._cond_vec(cond, self.training))

    def forward(self, s, th, dt, t, cond):
        """s, th, dt are (B, T) class ids possibly holding MASK. t is (B,)."""
        B, T = s.shape
        x = self.emb_s(s) + self.emb_th(th) + self.emb_dt(dt)
        x = x + self.pos(torch.arange(T, device=s.device))[None]
        x = x + self._cond_vec(cond, self.training)[:, None, :]
        x = x + self.t_proj(timestep_embedding(t, self.d_model))[:, None, :]
        h = self.norm(self.trunk(x))
        return self.head_s(h), self.head_th(h), self.head_dt(h)

    # ------------------------------------------------------------ sampling --
    @torch.no_grad()
    def sample(self, cond, n_steps=32, temperature=1.0, th_temp=None,
               dt_temp=None):
        """One trajectory per row. Fixed pass count, no selection, no best of N.

        Unmasking order is RANDOM, which is the reverse process of an absorbing
        chain. Confidence ordered unmasking is deliberately NOT used: it biases
        toward high probability tokens, which is under dispersion, and the
        contract scorer punishes exactly that through its dispersion ratios.
        """
        dev = cond.device
        B, T = len(cond), self.max_seq_len
        tt = (temperature, th_temp if th_temp is not None else temperature,
              dt_temp if dt_temp is not None else temperature)

        # the length head is trained on (L - 1), so the drawn class IS L - 1.
        # using it directly as L would shorten every trajectory by one event,
        # which the contract reads through event count and duration.
        L = torch.multinomial(F.softmax(self.length_logits(cond), -1),
                              1).squeeze(1) + 1
        live = torch.arange(T, device=dev)[None] < L[:, None]     # (B, T)

        toks = [torch.full((B, T), m, dtype=torch.long, device=dev)
                for m in MASKS]
        masked = live.clone()
        # random priority per cell decides the order cells are revealed in
        prio = torch.rand(B, T, device=dev)

        for i in range(n_steps):
            t_now = torch.full((B,), 1.0 - i / n_steps, device=dev)
            logits = self.forward(toks[0], toks[1], toks[2], t_now, cond)
            # reveal the cells with the lowest priority among those still
            # masked, so each pass uncovers about 1/n_steps of the sequence
            n_live = live.sum(1, keepdim=True).float()
            n_reveal = (n_live * (1.0 / n_steps)).ceil().long()
            pr = torch.where(masked, prio, torch.full_like(prio, 2.0))
            order = pr.argsort(dim=1)
            rank = order.argsort(dim=1)
            reveal = masked & (rank < n_reveal)
            if i == n_steps - 1:
                reveal = masked
            for c in range(3):
                p = F.softmax(logits[c] / tt[c], dim=-1)
                draw = torch.multinomial(p.reshape(-1, p.shape[-1]),
                                         1).reshape(B, T)
                toks[c] = torch.where(reveal, draw, toks[c])
            masked = masked & ~reveal

        s, th, dt = toks
        s = torch.where(live, s, torch.full_like(s, S_PAD_CLASS))
        th = torch.where(live, th, torch.full_like(th, TH_NULL_CLASS))
        dt = torch.where(live, dt, torch.zeros_like(dt))
        # a masked cell can only survive a bug; make that loud rather than silent
        assert not (masked & live).any(), "cells left masked after the schedule"
        return s, th, dt
