"""W4: left-to-right autoregressive event model.

Motivation, from measurements in HANDOFF.md rather than from taste:

- The gap to human is entirely joint structure. Every marginal is already
  correct (`w4_redundancy`) and rank-transforming to identical marginals moves
  the score by 0.007 (`w4_coupling`). What is wrong is local arrangement in
  time (`w4_arrangement`, `w4_joint2d`).
- The masked model's own local conditionals are mutually inconsistent.
  `w4_refine` runs them as a Gibbs sampler, which by construction converges to
  their stationary distribution, and the score gets monotonically WORSE with
  every sweep, 0.6559 to 0.7701. A set of conditionals that disagree with each
  other has no joint to be correct about. This is the defect this file targets
  and it is the one measurement with no confound, because it compares the model
  against itself.
- Every decode-time repair is closed: reveal order (`w4_order`,
  `w4_order_resid`), local refinement (`w4_refine`), sampling temperature
  (`w4_sharpness`, where the served 1.0 is already optimal in both directions).

A chain-rule factorization has one joint by construction, so the Gibbs failure
cannot occur. Three further consequences, each tied to a recorded number:

1. Every position is supervised on every example. Masked training supervises
   only the hidden fraction, median 0.296, so the same corpus yields roughly
   three times the learning signal here.
2. The remaining displacement is EXACT at every step. Under scattered masking
   the pointer's position is unknowable past the first gap, so
   `EventStreamPolarModel.prefix_resid` can only measure from the longest
   revealed prefix and returns one vector per sequence. Here it is per
   position and exact, in training and in sampling alike. The whole W3 P1
   aiming programme was an attempt to approximate this.
3. Time is a whole-millisecond choice. 98.4 percent of recorded human event
   times sit within a microsecond of an integer millisecond and none exceed
   150. The continuous flow head reproduces the marginal but not the lattice,
   and `w4_ms_lattice` measured that tell as worth 0.7240 to 0.5649 on
   duration alone. Tokenizing removes it by construction rather than by
   rounding afterwards.

PRIOR FAILURE THIS MUST NOT BE CONFUSED WITH. `resid_v3`, `v4` and `v5` trained
the masked model on contiguous suffix masks, v5 with the loss concentrated on
the next-token conditional, and all three scored 0.88 to 0.99 against a 0.647
base. Those were 4000-step fine-tunes at lr 2e-5 from a scattered-mask
checkpoint with the dt head frozen, and the trainer's own log records the loss
flat from step 1000. They also left attention bidirectional: suffix masking
hides token VALUES, the positions are still attended. This model is trained
from scratch with the future physically unreachable.

Emission order within a step is p(s) p(th | s) p(dt | s, th): the speed and
turn heads keep the conditional structure the masked model already has, and
dwell time is conditioned on the motion it accompanies rather than the reverse.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.event_stream_polar import (
    N_S_CLASSES, N_TH_CLASSES, S_PAD_CLASS, TH_BINS, TH_NULL_CLASS,
    TICK_CLASS, class_to_dtheta, class_to_speed,
)

# Whole-millisecond dt vocabulary. Class k means k ms; class 0 is the
# sub-half-millisecond bucket and decodes to DT_ZERO_MS. The corpus maximum is
# 150 ms, so nothing is clipped in practice.
DT_MAX_MS = 150
N_DT_VALS = DT_MAX_MS + 1          # 0..150
DT_PAD_CLASS = N_DT_VALS           # 151, input context only past the end
N_DT_CLASSES = N_DT_VALS + 1       # 152
DT_ZERO_MS = 0.5

# Start-of-sequence context token, one per stream.
S_BOS_CLASS = N_S_CLASSES          # 131
TH_BOS_CLASS = N_TH_CLASSES        # 257
DT_BOS_CLASS = N_DT_CLASSES        # 152

STATE_DIM = 6


def dt_ms_to_class(dt_ms: torch.Tensor) -> torch.Tensor:
    """Milliseconds -> whole-ms class, clamped into the vocabulary."""
    return torch.round(dt_ms.float()).long().clamp(0, DT_MAX_MS)


def class_to_dt_ms(c: torch.Tensor) -> torch.Tensor:
    """Whole-ms class -> milliseconds. PAD and the zero bucket both decode to
    a positive dwell so no decoded timestamp can collide with its neighbour."""
    ms = c.float().clamp(0, DT_MAX_MS)
    return torch.where(ms < 0.5, torch.full_like(ms, DT_ZERO_MS), ms)


def prefix_state(s_cls: torch.Tensor, th_cls: torch.Tensor,
                 dt_ms: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """Exact geometric and temporal state BEFORE each position.

    Everything is in the conditioning frame, the frame the dtheta tokens
    already live in: headings integrate from zero and the target sits at
    (exp(cond[:, 0]), 0). Position i sees only events strictly before it, so
    this is causal and can be recomputed identically during sampling.

    Returns (B, T, STATE_DIM):
        log1p(distance still to cover)
        unit x, unit y of that remaining vector
        elapsed time as a fraction of the commanded duration
        step index as a fraction of the buffer
        log1p(distance travelled so far)
    """
    B, T = s_cls.shape
    dev = s_cls.device
    s = class_to_speed(s_cls.clamp(max=N_S_CLASSES - 1))
    dth = class_to_dtheta(th_cls.clamp(max=N_TH_CLASSES - 1))
    motion = (s_cls > TICK_CLASS) & (s_cls < S_PAD_CLASS)
    heading = torch.cumsum(torch.where(motion, dth, torch.zeros_like(dth)), dim=1)
    dx = torch.where(motion, s * torch.cos(heading), torch.zeros_like(s))
    dy = torch.where(motion, s * torch.sin(heading), torch.zeros_like(s))

    # exclusive cumulative sums: index i holds the sum over j < i
    zero = torch.zeros(B, 1, device=dev)
    px = torch.cat([zero, torch.cumsum(dx, dim=1)[:, :-1]], dim=1)
    py = torch.cat([zero, torch.cumsum(dy, dim=1)[:, :-1]], dim=1)
    dt_s = class_to_dt_ms(dt_ms) / 1000.0 if dt_ms.dtype == torch.long else dt_ms / 1000.0
    el = torch.cat([zero, torch.cumsum(dt_s, dim=1)[:, :-1]], dim=1)

    tx = torch.exp(cond[:, 0]).unsqueeze(1)
    rx, ry = tx - px, -py
    rn = torch.sqrt(rx * rx + ry * ry)
    dur = torch.exp(cond[:, 1]).unsqueeze(1).clamp(min=1e-3)
    idx = torch.arange(T, device=dev, dtype=torch.float32).unsqueeze(0).expand(B, T)
    return torch.stack([
        torch.log1p(rn),
        rx / (rn + 1e-6),
        ry / (rn + 1e-6),
        (el / dur).clamp(0.0, 4.0),
        idx / float(T),
        torch.log1p(torch.sqrt(px * px + py * py)),
    ], dim=-1)


class CausalBlock(nn.Module):
    """Pre-norm block with a causal attention mask and FiLM from the static
    conditioning, matching CANDIBlock's conditioning placement so the two
    families differ only in what this file is meant to change."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
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
        self.p_drop = dropout

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(cond_emb).unsqueeze(1).chunk(2, dim=-1)
        h = self.norm1(x)
        B, T, D = h.shape
        q, k, v = (self.qkv(h).view(B, T, 3, self.n_heads, self.d_head)
                   .permute(2, 0, 3, 1, 4).unbind(0))
        a = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.p_drop if self.training else 0.0)
        a = a.transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.proj(a))
        x = x * (1.0 + scale) + shift
        x = x + self.ff(self.norm2(x))
        return x


class EventARModel(nn.Module):

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        cond_dim: int = 4,
        dropout: float = 0.1,
        cond_dropout: float = 0.1,
        emit_order: str = "s_th_dt",
        head_coupling: str = "add",
        mix_hidden: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.cond_dropout = cond_dropout
        assert emit_order in ("s_th_dt", "dt_s_th"), emit_order
        self.emit_order = emit_order

        self.s_embed = nn.Embedding(N_S_CLASSES + 1, d_model)     # +1 BOS
        self.th_embed = nn.Embedding(N_TH_CLASSES + 1, d_model)   # +1 BOS
        self.dt_embed = nn.Embedding(N_DT_CLASSES + 1, d_model)   # +1 BOS
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.state_proj = nn.Sequential(
            nn.Linear(STATE_DIM, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.layers = nn.ModuleList([
            CausalBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Within-step chain rule. Both orders are exact factorizations of the
        # same joint, so this is not a modelling approximation, it is which
        # conditional the network has to represent.
        #
        # "s_th_dt" (v1): p(s) p(th | s) p(dt | s, th). Keeps the speed and
        # turn structure the masked model already had.
        #
        # "dt_s_th" (v2): p(dt) p(s | dt) p(th | s, dt). A mouse reports on a
        # fixed poll cadence, so the displacement recorded in one sample is
        # roughly velocity times the interval it covers. Choosing the speed
        # without knowing the interval forces the network to marginalize over
        # it, and marginalizing is exactly what smooths a sequence. Measured
        # symptom in event_ar_v1: speed lag1 autocorrelation 0.6849 against a
        # human 0.5952, with the human lag2-above-lag1 alternation almost
        # gone (0.6914 / 0.6849 = 1.010 against a human 1.045).
        self.s_head = nn.Linear(d_model, N_S_CLASSES)
        self.s_ctx_embed = nn.Embedding(N_S_CLASSES, d_model)
        self.th_norm = nn.LayerNorm(d_model)
        self.th_head = nn.Linear(d_model, N_TH_CLASSES)
        self.th_ctx_embed = nn.Embedding(N_TH_CLASSES, d_model)
        self.dt_norm = nn.LayerNorm(d_model)
        self.dt_head = nn.Linear(d_model, N_DT_CLASSES)
        if emit_order == "dt_s_th":
            self.dt_ctx_embed = nn.Embedding(N_DT_CLASSES, d_model)
            self.s_norm = nn.LayerNorm(d_model)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Within-step COUPLING, separate from the within-step ORDER above.
        #
        # With "add", the default and what every checkpoint on disk was trained
        # with, the emitted speed reaches the direction logits only through the
        # additive term inside th_norm. th_head is linear, so that term is a
        # fixed logit offset per speed class, rescaled by the one scalar the
        # LayerNorm contributes. x is causal and cannot see s(t), so the head
        # cannot let the speed change the SHAPE of the direction distribution.
        #
        # `research/w4_headcap.py` priced that restriction at 0.2311 nats se
        # 0.0018 on 60,000 held out trajectories, with three controls clean: an
        # arm in the head's own function class bought +0.0000, a deeper arm
        # with no speed input bought +0.0037, and a parameter matched arm fed a
        # permuted speed stream bought +0.0011. The gain is the information,
        # not the depth and not the width.
        #
        # "mlp" adds back exactly the arm that bought it, output layer zero
        # initialised so the model is bit identical to "add" until trained.
        # dt is deliberately left alone: its measured interaction was 0.0024.
        self.head_coupling = head_coupling
        assert head_coupling in ("add", "mlp", "mlp_nos"), head_coupling
        if head_coupling != "add":
            n_in = d_model * (2 if head_coupling == "mlp" else 1)
            self.mix_norm = nn.LayerNorm(d_model)
            if head_coupling == "mlp":
                self.mix_emb = nn.Embedding(N_S_CLASSES, d_model)
                nn.init.xavier_uniform_(self.mix_emb.weight)
            self.th_mix = nn.Sequential(
                nn.Linear(n_in, mix_hidden), nn.GELU(),
                nn.Linear(mix_hidden, N_TH_CLASSES))
            nn.init.xavier_uniform_(self.th_mix[0].weight)
            nn.init.zeros_(self.th_mix[0].bias)
            nn.init.zeros_(self.th_mix[-1].weight)
            nn.init.zeros_(self.th_mix[-1].bias)

    def trunk(self, s_prev, th_prev, dt_prev, state, cond):
        """s_prev/th_prev/dt_prev are the tokens SHIFTED RIGHT by one, with BOS
        at position 0. state is prefix_state, already exclusive of position i."""
        B, T = s_prev.shape
        x = (
            self.s_embed(s_prev)
            + self.th_embed(th_prev)
            + self.dt_embed(dt_prev)
            + self.pos_embed(torch.arange(T, device=s_prev.device))
            + self.state_proj(state)
        )
        if self.training and self.cond_dropout > 0:
            keep = (torch.rand(B, 1, device=cond.device) > self.cond_dropout).float()
            cond = cond * keep
        c = self.cond_embed(cond)
        for layer in self.layers:
            x = layer(x, c)
        return self.norm(x)

    def _s_emb(self, s):
        return self.s_ctx_embed(s.clamp(max=N_S_CLASSES - 1))

    def _th_emb(self, th):
        return self.th_ctx_embed(th.clamp(max=N_TH_CLASSES - 1))

    def _dt_emb(self, dt):
        return self.dt_ctx_embed(dt.clamp(max=N_DT_CLASSES - 1))

    # --- s_th_dt heads -----------------------------------------------------
    def th_logits(self, x, s_cur):
        z = self.th_head(self.th_norm(x + self._s_emb(s_cur)))
        if self.head_coupling == "add":
            return z
        h = self.mix_norm(x)
        if self.head_coupling == "mlp":
            h = torch.cat([h, self.mix_emb(s_cur.clamp(max=N_S_CLASSES - 1))],
                          dim=-1)
        return z + self.th_mix(h)

    def dt_logits(self, x, s_cur, th_cur):
        return self.dt_head(self.dt_norm(
            x + self._s_emb(s_cur) + self._th_emb(th_cur)))

    # --- dt_s_th heads -----------------------------------------------------
    def dt_logits_first(self, x):
        return self.dt_head(self.dt_norm(x))

    def s_logits_given_dt(self, x, dt_cur):
        return self.s_head(self.s_norm(x + self._dt_emb(dt_cur)))

    def th_logits_given_s_dt(self, x, s_cur, dt_cur):
        return self.th_head(self.th_norm(
            x + self._s_emb(s_cur) + self._dt_emb(dt_cur)))

    def forward(self, s_prev, th_prev, dt_prev, state, cond,
                s_cur, th_cur, dt_cur=None):
        x = self.trunk(s_prev, th_prev, dt_prev, state, cond)
        if self.emit_order == "dt_s_th":
            return (self.s_logits_given_dt(x, dt_cur),
                    self.th_logits_given_s_dt(x, s_cur, dt_cur),
                    self.dt_logits_first(x))
        return (self.s_head(x),
                self.th_logits(x, s_cur),
                self.dt_logits(x, s_cur, th_cur))

    @staticmethod
    def shift_inputs(s_cls, th_cls, dt_cls):
        """Right-shift the token streams and put BOS at position 0."""
        B = s_cls.shape[0]
        dev = s_cls.device
        s_prev = torch.cat([torch.full((B, 1), S_BOS_CLASS, device=dev,
                                       dtype=torch.long), s_cls[:, :-1]], dim=1)
        th_prev = torch.cat([torch.full((B, 1), TH_BOS_CLASS, device=dev,
                                        dtype=torch.long), th_cls[:, :-1]], dim=1)
        dt_prev = torch.cat([torch.full((B, 1), DT_BOS_CLASS, device=dev,
                                        dtype=torch.long), dt_cls[:, :-1]], dim=1)
        return s_prev, th_prev, dt_prev

    @torch.no_grad()
    def sample(self, cond, seq_len=None, temperature=1.0,
               th_temperature=None, dt_temperature=None, force=None,
               th_beta=None, dt_tilt=None, th_tilt=None, s_tilt=None,
               tick_th_null=True, s_bias=None, th_bias=None, dt_bias=None):
        """One trajectory per row of cond, generated strictly left to right.

        No KV cache: each step re-runs the trunk over the prefix, which is
        exact and has no cache-invalidation surface. Returns integer class
        tensors (s_cls, th_cls, dt_cls), all (B, T), PAD-terminated on the
        speed stream exactly as the serving decoder expects.

        `force` is (s, th, dt, mask), four tensors, the first three (B, T)
        class streams and mask a bool. Wherever mask is set the token is
        taken from those streams instead of from this model's own head, and
        every other position is generated normally on top of it. It exists so
        `w4_prefix` can hand the model a real human opening and measure whether
        the model's own history is what poisons it. A mask is used rather than
        a prefix length because the control arm that makes the result readable
        scatters the same number of forced tokens across the sequence.

        mask is (B, T) to force all three channels together, or (B, T, 3) to
        force them independently in the order (s, th, dt). The per-channel form
        exists so `w4_timing` can hand the model a real human clock while it
        still chooses its own speeds, which is what separates a timing defect
        from a speed defect. A (B, T) mask is broadcast to all three channels,
        so every earlier caller keeps its exact previous behaviour.

        Each of `temperature`, `th_temperature`, `dt_temperature` may be a
        scalar (exact previous behaviour) or a list/tuple indexed at step i,
        a per position schedule. The schedule form exists so `w4_warmtemp`
        can serve the opening at temperature 1 and the tail at the served
        temps. Temps are logit divisors only, so the schedule form cannot
        change the number or order of rng draws.

        `th_beta` is a callable taking the model's own surprise at the speed it
        just emitted, shape (B,), and returning an inverse temperature of the
        same shape to scale the direction logits by. It exists so `w4_tempgen`
        can apply the confidence correction `w4_price` fitted and measure what
        it is worth on the contract scorer, rather than extrapolating it through
        an exchange rate. It applies only to the `s_th_dt` emit order, which is
        the one this model uses and the only one the correction was fitted on.

        `dt_tilt` is a (B,) tensor, one number per row, adding a bias linear in
        the dt class index to the timing head's logits. It exists so `w4_dttilt`
        can correct the duration response `w4_evcount` measured, and it is a
        serving time change rather than a diagnostic because it reads only
        `cond` and still emits one trajectory per row with no selection. Like
        `th_beta` it applies only to the `s_th_dt` emit order.

        `th_tilt` is a (B,) tensor, one number per row, held fixed for the whole
        trajectory, adding a bias linear in normalised turn magnitude to the
        direction head's logits. Drawn once per row from a prior by the caller,
        it is a per trajectory latent rather than a correction: it exists so
        `w4_latent` can supply the shared factor `w4_copula` measured missing
        between the six wobble features. Like `th_beta` and `dt_tilt` it applies
        only to the `s_th_dt` emit order.

        `s_tilt` is a (B,) tensor, one number per row, adding a bias linear in
        the speed class index, hence in log speed, to the speed head's logits.
        It is the third arm of the per trajectory factor: `w4_copula`'s leading
        eigenvector loads on movement_duration and mean_velocity about as
        heavily as on the direction features, so a latent that moves only
        direction reproduces about half of it. Held fixed across a trajectory
        alongside th_tilt and dt_tilt it drives all nine loaded features from
        one z.

        `tick_th_null` makes the timing head see the NULL turn marker at a no
        motion event, which is the only thing it was ever trained on there.
        See the comment at the call site. It defaulted to False while it was
        being measured; `w4_ticknull` then read it PARTIAL on three paired
        seeds of four thousand, 0.6423 to 0.6215, every seed negative, so it
        now defaults to True and this is the served path. Pass False to
        reproduce any measurement recorded before 2026-08-10.

        `s_bias`, `th_bias` and `dt_bias` are (C,) tensors, one entry per
        class of that head, added to the head's logits after the temperature
        division. Unlike the three tilts they are not a one parameter family:
        they can move mass between specific classes rather than only tilting
        the distribution. `w4_margfix` fits them to the gap between the corpus
        token marginal and this model's own free running marginal. The PAD
        entry of `s_bias` and the NULL entry of `th_bias` must be zero, since
        those classes control event count and the no-motion marker rather than
        a rate, and the fitting code centres each vector so the correction does
        not move the length distribution to first order.

        With force=None and every tilt None the only line below that differs
        from the pre 2026-08-10 sampler is the tick_th_null substitution.
        """
        B = cond.shape[0]
        T = seq_len or self.max_seq_len
        dev = cond.device
        th_temp = temperature if th_temperature is None else th_temperature
        dt_temp = temperature if dt_temperature is None else dt_temperature

        s_cls = torch.full((B, T), S_PAD_CLASS, device=dev, dtype=torch.long)
        th_cls = torch.full((B, T), TH_NULL_CLASS, device=dev, dtype=torch.long)
        dt_cls = torch.zeros((B, T), device=dev, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool, device=dev)

        for i in range(T):
            s_T = temperature[i] if isinstance(temperature, (list, tuple)) else temperature
            th_T = th_temp[i] if isinstance(th_temp, (list, tuple)) else th_temp
            dt_T = dt_temp[i] if isinstance(dt_temp, (list, tuple)) else dt_temp
            s_prev, th_prev, dt_prev = self.shift_inputs(s_cls, th_cls, dt_cls)
            state = prefix_state(s_cls, th_cls, dt_cls, cond)
            x = self.trunk(s_prev[:, :i + 1], th_prev[:, :i + 1],
                           dt_prev[:, :i + 1], state[:, :i + 1], cond)[:, -1]

            x1 = x.unsqueeze(1)
            if self.emit_order == "dt_s_th":
                dtp = torch.softmax(
                    self.dt_logits_first(x1).squeeze(1) / dt_T, dim=-1)
                dt_i = torch.multinomial(dtp, 1).squeeze(-1).clamp(max=DT_MAX_MS)

                sp = torch.softmax(
                    self.s_logits_given_dt(x1, dt_i.unsqueeze(1))
                    .squeeze(1) / s_T, dim=-1)
                s_i = torch.multinomial(sp, 1).squeeze(-1)

                thp = torch.softmax(
                    self.th_logits_given_s_dt(x1, s_i.unsqueeze(1),
                                              dt_i.unsqueeze(1))
                    .squeeze(1) / th_T, dim=-1)
                th_i = torch.multinomial(thp, 1).squeeze(-1)
            else:
                s_z = self.s_head(x)
                if s_tilt is not None:
                    # w4_hesit. The speed arm of the same per trajectory factor
                    # that th_tilt and dt_tilt carry. Speed classes are log
                    # spaced, 1 + log bin, so a bias linear in the class index
                    # is a bias linear in log speed. TICK and PAD are not speeds
                    # and are left alone: TICK is the no-motion marker and
                    # biasing it would change the pause rate, which is a
                    # different intervention, and biasing PAD would change the
                    # event count, which is a third one.
                    k = torch.arange(s_z.shape[-1], device=s_z.device,
                                     dtype=s_z.dtype)
                    k = (k - 1) / (S_PAD_CLASS - 2) - 0.5
                    k[TICK_CLASS] = 0.0
                    k[S_PAD_CLASS:] = 0.0
                    s_z = s_z + s_tilt.unsqueeze(-1) * k
                s_lz = s_z / s_T
                if s_bias is not None:
                    # w4_margfix. A full PER CLASS additive bias, unlike the
                    # tilts above which are one scalar times a linear function
                    # of the class index. Added AFTER the temperature division
                    # so its size means the same thing at any temperature.
                    s_lz = s_lz + s_bias
                sp = torch.softmax(s_lz, dim=-1)
                s_i = torch.multinomial(sp, 1).squeeze(-1)

                th_z = self.th_logits(x1, s_i.unsqueeze(1)).squeeze(1)
                if th_beta is not None:
                    # w4_tempgen. The direction head's confidence is corrected
                    # by how surprised this model just was by its OWN emitted
                    # speed. srp = -log p(s) - H(p) is conditional-mean-zero
                    # under a correct model, so a table indexed by it is a
                    # measurement of a defect and not a free parameter.
                    lg = torch.log(sp.clamp(min=1e-30))
                    ent = -(sp * lg).sum(-1)
                    srp = -lg.gather(-1, s_i.unsqueeze(-1)).squeeze(-1) - ent
                    b = th_beta(srp).unsqueeze(-1)
                    th_z = th_z * b
                if th_tilt is not None:
                    # w4_latent. A PER TRAJECTORY shared factor on the direction
                    # head. One scalar is drawn per row before the first event
                    # and held fixed for the whole trajectory, biasing the turn
                    # logits linearly in normalised turn magnitude. w4_copula
                    # measured the six wobble features coupling at 0.4985 in
                    # humans and 0.3333 here, with the sub families internally
                    # consistent and decoupled from each other; a single factor
                    # held across the trajectory drives all six together, which
                    # is exactly the missing structure. This makes generation a
                    # latent variable model, p(traj) = integral p(traj|z) p(z),
                    # and it still emits one trajectory per row with no
                    # selection, so it is a serving time change and not a
                    # diagnostic.
                    m = torch.arange(th_z.shape[-1], device=th_z.device,
                                     dtype=th_z.dtype)
                    m = (m - TH_BINS // 2).abs() / (TH_BINS // 2)
                    # class TH_NULL_CLASS is the no-turn marker for tick and
                    # PAD positions, not a turn of any magnitude. Leaving it in
                    # the basis would let a positive z raise the probability of
                    # emitting NULL, which is a different intervention than the
                    # one being measured. Centre over the real bins only and
                    # give NULL exactly zero bias.
                    m = m - m[:TH_BINS].mean()
                    m[TH_NULL_CLASS:] = 0.0
                    th_z = th_z + th_tilt.unsqueeze(-1) * m
                th_lz = th_z / th_T
                if th_bias is not None:
                    th_lz = th_lz + th_bias          # w4_margfix
                thp = torch.softmax(th_lz, dim=-1)
                th_i = torch.multinomial(thp, 1).squeeze(-1)
                if tick_th_null:
                    # w4_ticknull. Training sets the turn token to NULL wherever
                    # there is no motion and masks those positions out of the
                    # turn loss entirely, so the turn head is never trained
                    # there and the timing head has only ever seen NULL at a
                    # tick. Below, the same substitution is made AFTER the
                    # timing head has run, which means the served path feeds an
                    # untrained arbitrary turn into it at exactly those events.
                    # Humans put a 1 ms wait on 37.2 percent of no motion
                    # events and the served model on 5.8 percent, and teacher
                    # forced on a real history the head predicts 0.294 against
                    # a true 0.297, so the head is right and the call is wrong.
                    # Doing it here instead matches training. Measured by
                    # `w4_ticknull`: the rate moves 0.050 to 0.372 against a
                    # human 0.372 and the contract reads 0.6423 to 0.6215 on
                    # three paired seeds, so this is now the default.
                    th_i = torch.where(
                        (s_i > TICK_CLASS) & (s_i < S_PAD_CLASS), th_i,
                        torch.full_like(th_i, TH_NULL_CLASS))

                dt_z = self.dt_logits(x1, s_i.unsqueeze(1),
                                      th_i.unsqueeze(1)).squeeze(1)
                if dt_tilt is not None:
                    # w4_dttilt. A duration conditional tilt on the timing head,
                    # one parameter, using only cond. w4_evcount measured the
                    # model's mean inter event time rising with commanded
                    # duration at slope 0.2933 against the human 0.6023, so a
                    # bias linear in the dt class index and scaled by the row's
                    # centred log duration moves that slope directly. It is
                    # applied to logits, so it changes the distribution's shape
                    # and not just its mean, and lambda = 0 is exactly the
                    # unchanged served path.
                    k = torch.arange(dt_z.shape[-1], device=dt_z.device,
                                     dtype=dt_z.dtype)
                    k = k / max(dt_z.shape[-1] - 1, 1) - 0.5
                    dt_z = dt_z + dt_tilt.unsqueeze(-1) * k
                dt_lz = dt_z / dt_T
                if dt_bias is not None:
                    dt_lz = dt_lz + dt_bias          # w4_margfix
                dtp = torch.softmax(dt_lz, dim=-1)
                dt_i = torch.multinomial(dtp, 1).squeeze(-1).clamp(max=DT_MAX_MS)

            motion = (s_i > TICK_CLASS) & (s_i < S_PAD_CLASS)
            th_i = torch.where(motion, th_i,
                               torch.full_like(th_i, TH_NULL_CLASS))

            # Teacher forcing, applied AFTER the heads have run so the sampled
            # token is simply discarded, and BEFORE the done mask so a forced
            # PAD terminates its row the same way a generated one would.
            if force is not None:
                f_s, f_th, f_dt, f_mask = force
                take = f_mask[:, i]
                if take.dim() == 1:
                    take_s = take_th = take_dt = take
                else:
                    take_s, take_th, take_dt = take[:, 0], take[:, 1], take[:, 2]
                s_i = torch.where(take_s, f_s[:, i], s_i)
                th_i = torch.where(take_th, f_th[:, i], th_i)
                dt_i = torch.where(take_dt, f_dt[:, i], dt_i)

            # once a row has emitted PAD it stays padded; nothing after the
            # first PAD is read by the decoder but keeping it clean means
            # prefix_state stays meaningful for the rows still running.
            s_i = torch.where(done, torch.full_like(s_i, S_PAD_CLASS), s_i)
            th_i = torch.where(done, torch.full_like(th_i, TH_NULL_CLASS), th_i)
            dt_i = torch.where(done, torch.zeros_like(dt_i), dt_i)
            s_cls[:, i], th_cls[:, i], dt_cls[:, i] = s_i, th_i, dt_i
            done = done | (s_i >= S_PAD_CLASS)
            if bool(done.all()):
                break

        return s_cls, th_cls, dt_cls
