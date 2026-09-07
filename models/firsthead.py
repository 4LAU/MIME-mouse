"""q(e0 | cond), the first event head that seeds the served autoregressive model.

A small MLP on a Fourier featured condition that draws a trajectory's opening
event as p(s) p(th | s) p(dt | s, th), the AR model's own within step chain.
It exists because position 0 is the one position whose context is the four
number condition alone, and `models.event_ar.EventARModel` is measurably the
wrong conditional there: it is trained by `research/w4_firsthead.py` on the
position 0 tokens of the corpus (checkpoint `training/firsthead_q.pt`) and
served by `generate.py`, which forces its draw into position 0 of
`EventARModel.sample` so the free running stream starts from the better
conditional.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models.event_ar import N_DT_CLASSES
from models.event_stream_polar import (N_S_CLASSES, N_TH_CLASSES,
                                       S_PAD_CLASS, TH_NULL_CLASS, TICK_CLASS)


class FirstHead(nn.Module):
    """q(e0 | cond) = p(s) p(th | s) p(dt | s, th), the AR model's own within
    step chain, on a Fourier featured condition."""

    def __init__(self, d=512, n_freq=6):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq).float() * np.pi / 4)
        inp = 4 + 4 * 2 * n_freq
        self.inp = nn.Sequential(nn.Linear(inp, d), nn.GELU(), nn.Linear(d, d),
                                 nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.s_head = nn.Linear(d, N_S_CLASSES)
        self.s_emb = nn.Embedding(N_S_CLASSES, d)
        self.th_norm = nn.LayerNorm(d)
        self.th_head = nn.Linear(d, N_TH_CLASSES)
        self.th_emb = nn.Embedding(N_TH_CLASSES, d)
        self.dt_norm = nn.LayerNorm(d)
        self.dt_head = nn.Linear(d, N_DT_CLASSES)

    def feat(self, cond):
        x = cond.unsqueeze(-1) * self.freqs
        return torch.cat([cond, torch.sin(x).flatten(1), torch.cos(x).flatten(1)], -1)

    def forward(self, cond, s, th):
        h = self.inp(self.feat(cond))
        zs = self.s_head(h)
        zth = self.th_head(self.th_norm(h + self.s_emb(s)))
        zdt = self.dt_head(self.dt_norm(h + self.s_emb(s) + self.th_emb(th)))
        return zs, zth, zdt

    @torch.no_grad()
    def sample(self, cond, s_temp, th_temp, dt_temp):
        h = self.inp(self.feat(cond))
        s = torch.multinomial(torch.softmax(self.s_head(h) / s_temp, -1), 1).squeeze(-1)
        th = torch.multinomial(torch.softmax(
            self.th_head(self.th_norm(h + self.s_emb(s))) / th_temp, -1), 1).squeeze(-1)
        motion = (s > TICK_CLASS) & (s < S_PAD_CLASS)
        th = torch.where(motion, th, torch.full_like(th, TH_NULL_CLASS))
        dt = torch.multinomial(torch.softmax(
            self.dt_head(self.dt_norm(h + self.s_emb(s) + self.th_emb(th))) / dt_temp, -1),
            1).squeeze(-1)
        return s, th, dt
