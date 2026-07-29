"""P3: pinned-endpoint coordinate inpainting with rectified flow.

Generates mouse paths as (x, y) positions on the fixed 125 Hz / 192-slot
grid used by prepare_training_data.py. The start point, end point, and
padding region are supplied to the model as known values and the model
fills in the interior, the way image models fill a masked region. Exact
arrival is therefore free by construction.

A coupled discrete channel (CANDI-style absorbing mask) decides per-step
stall / no-stall so exact zero-velocity moments exist in the output. Pure
continuous diffusion cannot produce them, which the May 2026 record shows
is what killed the earlier CFM/DDPM line (curvature collapse).

Representation per slot: normalized coordinates (start at origin, path
scaled so the endpoint sits at unit distance). Condition vector is the
standard [log_dist, log_dur, cos_angle, sin_angle].
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class InpaintBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.film = nn.Linear(d_model, d_model * 2)

    def forward(self, x, cond_emb, key_padding_mask=None):
        scale, shift = self.film(cond_emb).unsqueeze(1).chunk(2, dim=-1)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + self.drop(h)
        x = x * (1.0 + scale) + shift
        x = x + self.ff(self.norm2(x))
        return x


class InpaintFlowModel(nn.Module):
    """Rectified flow over (x, y) grids with a coupled stall channel.

    Input channels per slot: noisy coords (2), known coords where the
    inpainting mask is set else zero (2), known flag (1), stall state (1,
    absorbing value STALL_MASK when undecided).
    """

    STALL_MASK = -1.0

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 192,
        cond_dim: int = 4,
        cond_dropout: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.cond_dropout = cond_dropout

        self.input_proj = nn.Linear(6, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.layers = nn.ModuleList([
            InpaintBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.cont_head = nn.Linear(d_model, 2)
        self.disc_head = nn.Linear(d_model, 1)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_t, known_val, known_flag, stall_state, t_cont, cond,
                pad_mask=None):
        """t_cont: (B,) flow time in [0, 1], 1 is pure noise.

        pad_mask: (B, T) bool, True marks padding slots excluded from
        attention keys. All slots still produce outputs.
        """
        B, T = x_t.shape[:2]
        inp = torch.cat([
            x_t,
            known_val,
            known_flag.unsqueeze(-1).float(),
            stall_state.unsqueeze(-1),
        ], dim=-1)
        x = self.input_proj(inp) + self.pos_embed(torch.arange(T, device=inp.device))

        t_emb = self.time_embed(t_cont * 999.0)
        if self.training and self.cond_dropout > 0:
            keep = (torch.rand(B, 1, device=cond.device) > self.cond_dropout).float()
            cond = cond * keep
        combined = t_emb + self.cond_embed(cond)

        for layer in self.layers:
            x = layer(x, combined, key_padding_mask=pad_mask)
        x = self.norm(x)
        return self.cont_head(x), self.disc_head(x).squeeze(-1)

    @staticmethod
    def q_flow(x0, t_cont, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        t = t_cont.view(-1, 1, 1)
        x_t = (1 - t) * x0 + t * noise
        velocity = noise - x0
        return x_t, noise, velocity

    @torch.no_grad()
    def flow_sample(self, cond, known_val, known_flag, pad_mask=None,
                    n_steps: int = 64, cfg_scale: float = 0.0):
        """Sample with the known region held to its noised value each step
        (replacement inpainting), then clamped exactly at the end.
        """
        B, T = known_flag.shape
        dev = cond.device
        noise0 = torch.randn(B, T, 2, device=dev)
        xt = noise0.clone()
        stall_s = torch.full((B, T), self.STALL_MASK, device=dev)
        mflag = torch.ones(B, T, device=dev)
        kf = known_flag.unsqueeze(-1).float()

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_c = 1.0 - i * dt
            xt = xt * (1 - kf) + ((1 - t_c) * known_val + t_c * noise0) * kf
            t_vec = torch.full((B,), t_c, device=dev)
            v_pred, sl = self.forward(xt, known_val, known_flag, stall_s,
                                      t_vec, cond, pad_mask)
            if cfg_scale > 0:
                v_u, sl_u = self.forward(xt, known_val, known_flag, stall_s,
                                         t_vec, torch.zeros_like(cond), pad_mask)
                v_pred = v_u + cfg_scale * (v_pred - v_u)
                sl = sl_u + cfg_scale * (sl - sl_u)
            xt = xt - dt * v_pred

            frac = 1.0 - t_c
            if frac > 0.3:
                conf = torch.abs(sl)
                thresh = max(0.5, 3.0 * (1.0 - frac))
                reveal = (conf > thresh) & (mflag > 0.5)
                stall_s = torch.where(reveal, (torch.sigmoid(sl) > 0.5).float(), stall_s)
                mflag = torch.where(reveal, torch.zeros_like(mflag), mflag)

        final_stall = torch.where(mflag > 0.5, (torch.sigmoid(sl) > 0.5).float(), stall_s)
        xt = xt * (1 - kf) + known_val * kf
        return xt, final_stall
