"""w4_screen. The cheap paired screen harness, W4's autoresearch loop.

UNREGISTERED. Nothing this file prints is quotable, goes in a headline, or
supports a serve decision. It exists to KILL ideas fast: every arm in a run
shares the rows, the condition, the q event 0 draw and the same stream of
sampling innovations (common random numbers), so the difference between two
arms is read at the paired standard deviation, a few thousandths, on one or
two seeds and a thousand rows. A survivor gets its own registration in
/home/aaronadmin/w4_arms/step0_prereg.md and the twelve seed protocol.
Levels printed here are at a different n from the record and are NOT
comparable to any number in it; only the differences mean anything.

The first family of arms, `coh*`, is the FAILURE_MAP direction B screen.
Teacher forced maximum likelihood learns each step's conditional with
whatever the prefix does not pin down marginalised out, and the served
sampler redraws that unpinned content independently at every step
(`torch.multinomial`). Here each head is instead sampled by inverse CDF from
a uniform u_t = Phi(z_t) where z_t is a Gaussian AR(1) process per row and
per head, z_t = rho z_{t-1} + sqrt(1 - rho^2) e_t. Every per step conditional
is left exactly as the model states it (a bijective reordering of the classes
followed by an inverse CDF is an exact draw from the categorical), only the
JOINT over steps changes, and rho = 0 is an ordinary iid sample. One
trajectory per row, no selection, no retraining, no judge.

Row machinery is w4_qladder's byte for byte: held out rows, rng(1000+seed)
pick, eligibility length > 4, the row's own condition, served temps, q event
0 forced at torch offset +7. Reads training/events_*.npy and checkpoints,
never the protected eval file, never candi_polar_flow_best.pt.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

os.environ.setdefault("EVENT_CHOICE_TEMP", "10")
os.environ.setdefault("EVENT_SNAP", "2.5")
os.environ.setdefault("EVENT_DUR_STD", "1.0")
os.environ.setdefault("DUR_EMPIRICAL", "1")
os.environ.setdefault("EVENT_BESTOF", "1")
os.environ.setdefault("EVENT_SIR", "1")
os.environ.setdefault("EVENT_ORDER", "gumbel")
os.environ.setdefault("EVENT_STEPS", "100")
os.environ.setdefault("EVENT_CFG_W", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp                      # noqa: E402
import ledger                                                     # noqa: E402
import scoring                                                    # noqa: E402
from features import extract_feature_matrix                       # noqa: E402
from models.event_ar import (DT_MAX_MS, class_to_dt_ms,           # noqa: E402
                             dt_ms_to_class, prefix_state)
from models.event_ar_latent import load_latent                    # noqa: E402
from models.event_stream_polar import (S_PAD_CLASS, TH_BINS, TH_NULL_CLASS,  # noqa: E402
                                       TICK_CLASS, dth_lattice_to_class,
                                       s2_to_class)
from w4_firsthead import FirstHead, Q_PATH                        # noqa: E402

TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
MAX_T = 256
KMAX = 4
OUT_DIR = "/home/aaronadmin/w4_arms/screens"

# Class orderings the inverse CDF walks. A permutation lists the classes in
# the order their probability mass is stacked, so u near 0 lands at the
# first entry and u near 1 at the last; NULL is appended last so the served
# tick_th_null substitution stays the only way it is emitted.
_pos = list(range(0, TH_BINS // 2))                  # 0 .. +pi
_neg = list(range(TH_BINS // 2, TH_BINS))            # -pi .. 0
TH_SIGNED = _neg + _pos + [TH_NULL_CLASS]
_mag = [0]
for k in range(1, TH_BINS // 2):
    _mag += [TH_BINS - k, k]                          # -k then +k, by magnitude
_mag += [TH_BINS // 2]                                # the pi bin
TH_MAG = _mag + [TH_NULL_CLASS]
assert sorted(TH_SIGNED) == list(range(TH_BINS + 1))
assert sorted(TH_MAG) == list(range(TH_BINS + 1))

# Arm registry. rho per head; th_order is the ordering the turn head's
# inverse CDF walks. "served" is the sampler on the record, torch.multinomial,
# and is the equivalence check for the rho 0 base.
VARIANTS = {
    "served":    None,
    "base":      dict(rho_s=0.0, rho_th=0.0, rho_dt=0.0, th_order="signed"),
    "cohS9":     dict(rho_s=0.9, rho_th=0.0, rho_dt=0.0, th_order="signed"),
    "cohT9":     dict(rho_s=0.0, rho_th=0.9, rho_dt=0.0, th_order="signed"),
    "cohT9m":    dict(rho_s=0.0, rho_th=0.9, rho_dt=0.0, th_order="mag"),
    "cohD9":     dict(rho_s=0.0, rho_th=0.0, rho_dt=0.9, th_order="signed"),
    "cohA5":     dict(rho_s=0.5, rho_th=0.5, rho_dt=0.5, th_order="signed"),
    "cohA9":     dict(rho_s=0.9, rho_th=0.9, rho_dt=0.9, th_order="signed"),
    "cohA9m":    dict(rho_s=0.9, rho_th=0.9, rho_dt=0.9, th_order="mag"),
    "cohA99":    dict(rho_s=0.99, rho_th=0.99, rho_dt=0.99, th_order="signed"),
    "cohSD9":    dict(rho_s=0.9, rho_th=0.0, rho_dt=0.9, th_order="signed"),
    # single stage speed draw: the termination and tick decisions share the
    # coherent uniform, so the event count moves with rho as well
    "cohS9pad":  dict(rho_s=0.9, rho_th=0.0, rho_dt=0.0, th_order="signed",
                      pad_coh=True),
    # FAILURE_MAP direction A: a per trajectory latent the decoder was TRAINED
    # to read (training/train_event_ar_latent.py). `ckpt` names the latent
    # checkpoint, `z` says where the latent comes from at serving time: the
    # prior N(0, I) drawn once per row from the CRN generator, or zero, the
    # prior mean, which isolates what the fine tune did to the decoder from
    # what the draw adds. Both use the served sampler on the served rng
    # stream, so against "served" the only change is the model and z.
    "latent":    dict(ckpt="LATENT", z="prior"),
    "latent0":   dict(ckpt="LATENT", z="zero"),
    "latentH":   dict(ckpt="LATENT", z="half"),
    # training/finetune_event_ar.py arms, served sampler on the served rng
    # stream, only the checkpoint differs from "served"
    "pert":      dict(ckpt="PERT"),      # M4, perturbed prefixes
    "mol":       dict(ckpt="MOL"),
    "cat":       dict(ckpt="CAT", z="cat"),     # K codes, one drawn from the learned prior
    "catU":      dict(ckpt="CAT", z="catU"),    # same codes, drawn uniformly
    "plan":      dict(ckpt="PLAN", z="plan"),   # observed plan, one draw from the plan model
    # ORACLE, NOT SERVABLE: the row's OWN real plan handed to the decoder.
    # It reads the trajectory being imitated, so it can never be a serving
    # configuration and its number is never a result. It is the ceiling of
    # the observed plan family: if the decoder cannot beat served even with
    # the true plan, no plan model can save the arm.
    "planR":     dict(ckpt="PLAN", z="real"),
    "plan0":     dict(ckpt="PLAN", z="zero"),   # same decoder, plan zeroed       # M5, ordinal mixture heads
    # MASS PRESERVING DIRECTION TEMPERATURE, served checkpoint, served rng
    # stream, served temperatures. The only change from "served" is that the
    # direction divisor is applied inside turn magnitude lobes with the lobe
    # masses held at their temperature 1 values. `w4_occupancy` measured the
    # served temperatures making 0.4422 reversals per trajectory against a
    # human 0.6660, with temperature 1 at 0.6797, so the divisor is the whole
    # deficit. tau is the lobe edge in direction bins, 64 being the quarter
    # turn the reversal rate is counted at.
    "lobe64":    dict(th_lobe_tau=64),
    "lobe32":    dict(th_lobe_tau=32),
    "lobe96":    dict(th_lobe_tau=96),
    # The same construction on the speed head, where the lobes are fixed by
    # meaning: the no motion marker, the ordinal speeds, and PAD. The served
    # speed divisor moves the pause rate and the event count along with the
    # speed shape; `w4_occupancy` reads 4.7185 still runs and 55.58 events
    # against a human 4.2168 and 52.41, with temperature 1 at 4.2378 and 54.42.
    "slobe":     dict(s_lobe=True),
    "bothlobe":  dict(s_lobe=True, th_lobe_tau=64),
}


def inv_cdf(p, u, perm):
    """Exact categorical draw: stack the classes in `perm` order and read off
    the first class whose cumulative mass exceeds u."""
    if perm is not None:
        p = p[:, perm]
    cdf = torch.cumsum(p.double(), dim=-1)
    idx = (cdf < u.double().unsqueeze(-1)).sum(-1).clamp(max=p.shape[-1] - 1)
    if perm is not None:
        idx = perm[idx]
    return idx


@torch.no_grad()
def sample_coh(model, cond, s_T, th_T, dt_T, force, var, gen):
    """The served s_th_dt path of EventARModel.sample with tick_th_null and
    no tilts, the three multinomials replaced by inverse CDF draws from
    per row AR(1) uniforms. rho 0 on every head is an iid sample."""
    B = cond.shape[0]
    T = model.max_seq_len
    dev = cond.device
    perm_th = torch.tensor(TH_SIGNED if var["th_order"] == "signed" else TH_MAG,
                           device=dev, dtype=torch.long)
    rho = torch.tensor([var["rho_s"], var["rho_th"], var["rho_dt"]],
                       device=dev, dtype=torch.float32)
    innov = torch.sqrt(1.0 - rho * rho)
    # stationary start so rho 0 gives unit variance at every step
    z = torch.randn(B, 3, device=dev, generator=gen)

    s_cls = torch.full((B, T), S_PAD_CLASS, device=dev, dtype=torch.long)
    th_cls = torch.full((B, T), TH_NULL_CLASS, device=dev, dtype=torch.long)
    dt_cls = torch.zeros((B, T), device=dev, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool, device=dev)

    for i in range(T):
        s_prev, th_prev, dt_prev = model.shift_inputs(s_cls, th_cls, dt_cls)
        state = prefix_state(s_cls, th_cls, dt_cls, cond)
        x = model.trunk(s_prev[:, :i + 1], th_prev[:, :i + 1],
                        dt_prev[:, :i + 1], state[:, :i + 1], cond)[:, -1]
        x1 = x.unsqueeze(1)

        e = torch.randn(B, 4, device=dev, generator=gen)
        z = rho * z + innov * e[:, :3]
        u = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

        sp = torch.softmax(model.s_head(x) / s_T, dim=-1)
        if var.get("pad_coh", False):
            s_i = inv_cdf(sp, u[:, 0], None)
        else:
            # Two exact stages so the coherent uniform drives only the speed
            # MAGNITUDE. Which kind of event this is, a tick, a motion or the
            # terminating PAD, is drawn iid from the head's own mass on the
            # three kinds, so the event count and the pause rate keep the
            # served statistics whatever rho is. The smoke run with a single
            # stage moved mean events 56 to 74 at rho 0.9, which is the
            # termination decision being held at one quantile, not the
            # effect under test.
            uk = 0.5 * (1.0 + torch.erf(e[:, 3] / math.sqrt(2.0)))
            p_tick = sp[:, TICK_CLASS]
            p_mot = sp[:, TICK_CLASS + 1:S_PAD_CLASS].sum(-1)
            kind = (uk >= p_tick).long() + (uk >= p_tick + p_mot).long()
            pm = sp[:, TICK_CLASS + 1:S_PAD_CLASS]
            pm = pm / pm.sum(-1, keepdim=True).clamp(min=1e-30)
            s_mot = inv_cdf(pm, u[:, 0], None) + TICK_CLASS + 1
            s_i = torch.where(kind == 0, torch.full_like(s_mot, TICK_CLASS),
                              torch.where(kind == 1, s_mot,
                                          torch.full_like(s_mot, S_PAD_CLASS)))

        thp = torch.softmax(
            model.th_logits(x1, s_i.unsqueeze(1)).squeeze(1) / th_T, dim=-1)
        th_i = inv_cdf(thp, u[:, 1], perm_th)
        th_i = torch.where((s_i > TICK_CLASS) & (s_i < S_PAD_CLASS), th_i,
                           torch.full_like(th_i, TH_NULL_CLASS))

        dtp = torch.softmax(
            model.dt_logits(x1, s_i.unsqueeze(1), th_i.unsqueeze(1))
            .squeeze(1) / dt_T, dim=-1)
        dt_i = inv_cdf(dtp, u[:, 2], None).clamp(max=DT_MAX_MS)

        motion = (s_i > TICK_CLASS) & (s_i < S_PAD_CLASS)
        th_i = torch.where(motion, th_i, torch.full_like(th_i, TH_NULL_CLASS))

        if force is not None:
            f_s, f_th, f_dt, f_mask = force
            take = f_mask[:, i]
            s_i = torch.where(take, f_s[:, i], s_i)
            th_i = torch.where(take, f_th[:, i], th_i)
            dt_i = torch.where(take, f_dt[:, i], dt_i)

        s_i = torch.where(done, torch.full_like(s_i, S_PAD_CLASS), s_i)
        th_i = torch.where(done, torch.full_like(th_i, TH_NULL_CLASS), th_i)
        dt_i = torch.where(done, torch.zeros_like(dt_i), dt_i)
        s_cls[:, i], th_cls[:, i], dt_cls[:, i] = s_i, th_i, dt_i
        done = done | (s_i >= S_PAD_CLASS)
        if bool(done.all()):
            break
    return s_cls, th_cls, dt_cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_hm_mlp.pt")
    ap.add_argument("--latent-ckpt", default="event_ar_latent.pt",
                    help="what the latent* arms load")
    ap.add_argument("--pert-ckpt", default="event_ar_pert.pt")
    ap.add_argument("--mol-ckpt", default="event_ar_mol.pt")
    ap.add_argument("--cat-ckpt", default="event_ar_cat.pt")
    ap.add_argument("--plan-ckpt", default="event_ar_plan.pt")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--s-temp", type=float, default=0.95)
    ap.add_argument("--th-temp", type=float, default=0.90)
    ap.add_argument("--dt-temp", type=float, default=1.00)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--base", default="base")
    ap.add_argument("--e0", default="q", choices=["q", "free"],
                    help="q forces event 0 from the served first event head "
                         "(the A19 configuration); free is the k0 arm")
    ap.add_argument("--tag", required=True, help="screen name, used in the "
                    "output file and the ledger note")
    a = ap.parse_args()
    arms = a.arms.split(",")
    assert all(x in VARIANTS for x in arms), arms
    assert a.base in arms, "the base arm must be in --arms"
    os.makedirs(OUT_DIR, exist_ok=True)

    dev = esp._DEVICE
    lengths = np.load("training/events_len.npy")
    N = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(N, min(N, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(N), trained)
    rng = np.random.default_rng(1000 + a.seed)
    elig = held[lengths[held] > KMAX]
    pick = np.sort(rng.choice(elig, a.n, replace=False))
    print(f"  SCREEN {a.tag}, UNREGISTERED, levels not comparable to the record",
          flush=True)
    print(f"  corpus {N:,}, held out {len(held):,}, eligible {len(elig):,}, "
          f"using {a.n:,}, seed {a.seed}, e0 {a.e0}", flush=True)

    s2 = np.load("training/events_s2.npy", mmap_mode="r")[pick]
    dth = np.load("training/events_dth.npy", mmap_mode="r")[pick]
    dt_ms = np.load("training/events_dt.npy", mmap_mode="r")[pick].astype(np.float64)
    conds = np.load("training/events_cond.npy")[pick]
    L = np.minimum(lengths[pick], MAX_T).astype(np.int64)
    B = len(L)

    real_s = np.full((B, MAX_T), S_PAD_CLASS, dtype=np.int64)
    real_th = np.full((B, MAX_T), TH_NULL_CLASS, dtype=np.int64)
    real_dt = np.zeros((B, MAX_T), dtype=np.int64)
    sc = s2_to_class(torch.from_numpy(np.asarray(s2, dtype=np.int64))).numpy()
    tc = np.where(np.asarray(s2) > 0,
                  dth_lattice_to_class(
                      torch.from_numpy(np.asarray(dth, dtype=np.int64))).numpy(),
                  TH_NULL_CLASS)
    dc = dt_ms_to_class(torch.from_numpy(dt_ms)).numpy()
    for i in range(B):
        n = int(L[i])
        real_s[i, :n] = sc[i, :n]
        real_th[i, :n] = tc[i, :n]
        real_dt[i, :n] = dc[i, :n].clip(0, DT_MAX_MS)

    cond_t = torch.from_numpy(conds[:, :4].astype(np.float32))
    angs = np.arctan2(conds[:, 3].astype(np.float64),
                      conds[:, 2].astype(np.float64))

    ck = torch.load(f"training/{a.ckpt}", map_location=dev, weights_only=False)
    model, _, _ = load_latent(f"training/{a.ckpt}", dev)
    models = {a.ckpt: model}

    def model_for(var):
        name = var["ckpt"] if var and "ckpt" in var else a.ckpt
        name = {"LATENT": a.latent_ckpt, "PERT": a.pert_ckpt,
                "MOL": a.mol_ckpt, "CAT": a.cat_ckpt,
                "PLAN": a.plan_ckpt}.get(name, name)
        if name not in models:
            m, _, cfg = load_latent(f"training/{name}", dev)
            models[name] = m
            print(f"  loaded {name} z_dim {cfg.get('z_dim')}", flush=True)
        return models[name]
    qk = torch.load(Q_PATH, map_location=dev, weights_only=False)
    q = FirstHead(**qk["config"]).to(dev).eval()
    q.load_state_dict(qk["model_state_dict"])
    print(f"  {a.ckpt} step {ck.get('step')}  q best epoch {qk['best']['epoch']}"
          f"  AR temps s {a.s_temp} th {a.th_temp} dt {a.dt_temp}", flush=True)

    def decode_all(s_np, th_np, dtc_np):
        paths = []
        for i in range(len(s_np)):
            d = class_to_dt_ms(torch.from_numpy(dtc_np[i])).numpy()
            dz = (np.log(np.maximum(d, 0.05)) - esp._DT_MEAN) / esp._DT_STD
            p = esp._decode(dz, s_np[i], th_np[i], 0.0, 0.0, float(angs[i]))
            if p is not None and len(p) >= 4:
                paths.append(np.asarray(p, dtype=np.float64))
        return paths

    plan_real = {}

    def generate(name):
        var = VARIANTS[name]
        model = model_for(var)
        g_s = np.full((B, MAX_T), S_PAD_CLASS, dtype=np.int64)
        g_th = np.full((B, MAX_T), TH_NULL_CLASS, dtype=np.int64)
        g_dt = np.zeros((B, MAX_T), dtype=np.int64)
        for c0 in range(0, B, a.batch):
            sl = slice(c0, min(c0 + a.batch, B))
            cb = cond_t[sl].to(dev)
            nb = cb.shape[0]
            f = None
            if a.e0 == "q":
                fs = torch.from_numpy(real_s[sl]).to(dev).clone()
                fth = torch.from_numpy(real_th[sl]).to(dev).clone()
                fdt = torch.from_numpy(real_dt[sl]).to(dev).clone()
                mask = torch.zeros((nb, MAX_T), device=dev, dtype=torch.bool)
                torch.manual_seed(a.seed * 100003 + c0 + 7)
                qs, qth, qdt = q.sample(cb, 1.0, 1.0, 1.0)
                fs[:, 0], fth[:, 0] = qs, qth
                fdt[:, 0] = qdt.clamp(max=DT_MAX_MS)
                mask[:, 0] = True
                f = (fs, fth, fdt, mask)
            if (var is None or "ckpt" in var or "th_lobe_tau" in var
                    or "s_lobe" in var):
                if var is not None and "z" in var:
                    zg = torch.Generator(device=dev)
                    zg.manual_seed(a.seed * 100003 + c0 + 53)
                    if var["z"] == "real":
                        if "feats" not in plan_real:
                            plan_real["feats"] = np.load("training/plan_feats.npy")[pick]
                        pq = model.plan_sampler.quant
                        z = torch.as_tensor(
                            pq.standardise(plan_real["feats"][sl]).astype(np.float32),
                            device=dev).to(cb.dtype)
                    elif var["z"] == "plan":
                        z = model.plan_sampler(cb, generator=zg)
                    elif var["z"] == "cat":
                        z = model.draw_codes(cb, generator=zg)
                    elif var["z"] == "catU":
                        k = torch.randint(model.K, (nb,), device=dev, generator=zg)
                        z = torch.nn.functional.one_hot(k, model.K).to(cb.dtype)
                    else:
                        z = torch.randn((nb, model.z_dim), device=dev, generator=zg)
                        z = z * {"prior": 1.0, "half": 0.5, "zero": 0.0}[var["z"]]
                    cb = torch.cat([cb, z], 1)
                torch.manual_seed(a.seed * 100003 + c0)
                with torch.no_grad():
                    s_o, th_o, dt_o = model.sample(
                        cb, temperature=a.s_temp, th_temperature=a.th_temp,
                        dt_temperature=a.dt_temp, force=f,
                        th_lobe_tau=(var or {}).get("th_lobe_tau"),
                        s_lobe=bool((var or {}).get("s_lobe", False)))
            else:
                gen = torch.Generator(device=dev)
                gen.manual_seed(a.seed * 100003 + c0 + 31)
                s_o, th_o, dt_o = sample_coh(model, cb, a.s_temp, a.th_temp,
                                             a.dt_temp, f, var, gen)
            n_got = s_o.shape[1]
            g_s[sl, :n_got] = s_o.cpu().numpy()
            g_th[sl, :n_got] = th_o.cpu().numpy()
            g_dt[sl, :n_got] = dt_o.cpu().numpy()
        return decode_all(g_s, g_th, g_dt), g_s, g_th

    out = {"tag": a.tag, "ckpt": a.ckpt, "seed": a.seed, "n_rows": int(B),
           "e0": a.e0, "temps": [a.s_temp, a.th_temp, a.dt_temp],
           "base": a.base, "arms": {}}
    print(f"\n  {'arm':>8}{'level':>9}{'n':>6}{'motion':>8}{'ticks':>7}"
          f"{'allev':>8}{'revers':>8}{'collapse':>10}{'minutes':>9}", flush=True)
    for name in arms:
        t0 = time.time()
        paths, g_s, g_th = generate(name)
        F = extract_feature_matrix(paths)
        F = F[np.all(np.isfinite(F), 1)]
        F = F[np.random.default_rng(a.seed).permutation(len(F))]
        r = scoring.score_features(F)
        motion = (g_s > TICK_CLASS) & (g_s < S_PAD_CLASS)
        n_ev = float(motion.sum(1).mean())
        # The event count splits into two kinds and they need opposite fixes,
        # so report both. `chk_human_rates.py` counts a human trajectory's
        # events as its full length, ticks included, so `n_all` is the only
        # column here comparable to the human 51.996; `n_ev` is motion only
        # and every earlier screen printed that under the name "events".
        n_tick = float((g_s == TICK_CLASS).sum(1).mean())
        n_all = n_ev + n_tick
        # Reversals per trajectory, the rate `w4_occupancy` reads at 0.6660 in
        # humans and 0.4422 on the served temperatures. A turn bin b is a
        # signed turn ((b + 128) % 256 - 128) / 128 in half turns, so a
        # reversal is a quarter turn, 64 bins, and only motion events carry a
        # turn at all.
        mag = np.abs(((g_th.astype(np.int64) + TH_BINS // 2) % TH_BINS)
                     - TH_BINS // 2)
        n_rev = float((motion & (g_th < TH_BINS) & (mag > TH_BINS // 4))
                      .sum(1).mean())
        out["arms"][name] = dict(level=float(r["auc_rf_oob"]), n=int(len(F)),
                                 mean_motion_events=n_ev, mean_ticks=n_tick,
                                 mean_all_events=n_all, reversals=n_rev,
                                 collapse=bool(r["collapse_flag"]),
                                 variant=VARIANTS[name])
        print(f"  {name:>8}   {r['auc_rf_oob']:.4f} {len(F):5d} {n_ev:7.1f}"
              f" {n_tick:6.1f} {n_all:7.1f} {n_rev:7.3f} "
              f" {str(bool(r['collapse_flag'])):>8}"
              f" {(time.time() - t0) / 60:8.1f}", flush=True)

    b = out["arms"][a.base]["level"]
    print(f"\n  paired differences against {a.base}, negative is toward human:")
    for name in arms:
        d = out["arms"][name]["level"] - b
        out["arms"][name]["diff_vs_base"] = float(d)
        print(f"  {name:>8}  {d:+.4f}")

    path = f"{OUT_DIR}/{a.tag}_s{a.seed}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n  wrote {path}")
    print("  UNREGISTERED SCREEN. one trajectory per row, no selection,"
          " levels not comparable to the record, differences suggestive only")
    rid = ledger.append_row(
        "w4_screen",
        {"tag": a.tag, "seed": a.seed, "n": a.n, "e0": a.e0, "arms": arms,
         "base": a.base, "ckpt": a.ckpt},
        "ok",
        metrics={"screen_base_level": b,
                 "screen_best_diff": float(min(
                     out["arms"][x]["diff_vs_base"] for x in arms))},
        artifacts=[path],
        notes=f"UNREGISTERED CHEAP SCREEN {a.tag}. Not quotable, no headline,"
              f" no serve decision. Levels at n={a.n} are not comparable to"
              f" the record; only paired differences against {a.base} mean"
              f" anything, and only as a kill or keep signal ahead of a"
              f" registration.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
