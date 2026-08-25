"""Is `w4_estimator`'s cosine between the two estimators readable at all? Registered before it ran.

THE PROBLEM THIS EXISTS TO FIX

`w4_estimator` prints, on every tenth step, the cosine between the pathwise
straight through gradient and the score function gradient of the SAME surrogate
on the SAME batch. The first two readings were +0.08 and +0.03, and the
registration in that file says a cosine near zero means RELAX is not worth
building, because a control variate reduces variance only in so far as it
correlates with what it corrects.

That reading does not follow, and the reason is already in the record. Job 2
measured the score function gradient's batch to batch agreement at batch 96 and
found none. An estimator with no self agreement is close to pure noise at this
batch size, and the cosine between two noisy estimates of the same true
direction is ATTENUATED by the noise in both of them:

    cos(g_st, g_sf)  ~  cos(mu_st, mu_sf) * sqrt(r_st) * sqrt(r_sf)

where mu is the direction each estimator converges to with unlimited samples and
r is that estimator's reliability, the squared correlation between one estimate
and its own mean direction. If r_sf is near zero then the observed cosine is
near zero NO MATTER WHAT cos(mu_st, mu_sf) is. The measurement cannot tell
"the relaxed decode points somewhere useless" apart from "the score function
estimate is too noisy to correlate with anything", and those two license
opposite conclusions about whether RELAX is worth building.

So the number is uninterpretable as it stands. This file makes it interpretable
by measuring the two reliabilities and dividing them out.

WHAT IS MEASURED

The model and the critic are FROZEN. No optimiser step is ever taken on the
generator. That is the point: the attenuation model above assumes each estimator
has one fixed mean direction mu, which is only true if the parameters are not
moving. `w4_estimator` measures its cosine during training, where mu drifts
between steps, so even the pairing there is loose. Here the only thing that
varies between measurements is the batch draw, which isolates estimator noise
exactly.

Each of K batches is sampled once, decoded once, and its surviving rows split at
random into two halves A and B. Each half computes its OWN critic coefficients
against its OWN human sample, so the halves are independent draws of the same
half batch estimator rather than two views of one coupled batch statistic. Then
six gradients come off each batch:

    st_A  st_B      pathwise, through the relaxed decode
    sf_A  sf_B      score function, no relaxation anywhere
    nll_A nll_B     supervised teacher forced NLL on human tokens

and they give, per batch,

    r_st    = cos(st_A, st_B)        reliability of the pathwise estimator
    r_sf    = cos(sf_A, sf_B)        reliability of the score function estimator
    r_nll   = cos(nll_A, nll_B)      the KNOWN GOOD control, same footing
    c_cross = mean of cos(st_A, sf_B) and cos(st_B, sf_A)
    c_same  = mean of cos(st_A, sf_A) and cos(st_B, sf_B)

c_cross is the cross estimator cosine on INDEPENDENT draws, which is what the
attenuation formula applies to. c_same shares a batch draw between the two
estimators, so any excess of c_same over c_cross is shared sampling noise rather
than agreement in direction; `w4_estimator`'s printed cosine is a c_same, so the
gap prices exactly how much of it was an artifact of the shared batch.

The disattenuated quantity, all three terms at the same half batch footing:

    d = c_cross / sqrt(r_st * r_sf)

which estimates cos(mu_st, mu_sf), the agreement between the two estimators'
true directions with the noise divided out.

THE KNOWN GOOD CONTROL, AND WHY THERE IS ONE

Job 2's own carried forward lesson is never to register an absolute threshold on
an unfamiliar statistic without a known good instance measured on the same
footing, because its first attempt did exactly that and is uninterpretable. A
split half reliability is such a statistic. The supervised NLL is the control:
it is the one gradient in this workstream with an uncontested mean direction, it
is measured on the same batches through the same code path, and every gate below
is a DIFFERENCE against it rather than a level.

WHY THE STANDARD ERRORS HERE ARE NOT THE ONES `w4_cosse` CORRECTED

`w4_cosse` found the errors on every earlier gradient cosine understated,
because those averaged over all pairs from a pool of gradients and divided by
the root of the PAIR count when each gradient appears in many pairs. Nothing
here is a U statistic. Each batch yields exactly one independent reading of each
quantity and the batches are independent draws, so the plain standard error over
K batches is correct and no jackknife is needed. The paired differences are
taken batch by batch, which removes the batch draw the arms share.

READING IT, AND THE GATES

PRIMARY, and it is an instrument check, not a result. Paired per batch,
r_nll - r_sf:
    r_sf below r_nll by 3 se or more, AND r_sf within 2 se of zero
        the score function estimate is blind at batch 96. `w4_estimator`'s
        cosine line says nothing about RELAX and must be WITHDRAWN, not read.
        This is what Job 2 predicts.
    r_sf within 2 se of r_nll
        FALSIFIER. The score function estimate is as self consistent as the
        known good control, the attenuation worry is unfounded, and the near
        zero cosine in `w4_estimator` is a real statement about direction.

SECONDARY, the RELAX question, and it is READABLE ONLY IF both r_st and r_sf
clear zero by 2 se. If they do not, d is a ratio by something indistinguishable
from zero and must not be quoted at all.
    d below 0.2        the two estimators genuinely disagree in direction.
                       RELAX is not worth building, and the pathwise gradient is
                       not estimating the surrogate it is written to estimate.
    d above 0.7        they agree in direction once noise is removed, and
                       estimator bias is NOT the collapse mechanism, which would
                       send task 24 back to the objective.
    d above 1.3        UNINTERPRETABLE, not agreement. The attenuation model has
                       broken, most likely because the two estimators share
                       noise the cross construction was supposed to remove, or
                       because a reliability came out near zero and the divisor
                       is meaningless. Report it as broken and do not read the
                       secondary gate.

The 1.3 bound is there because the record's other rule is that a gate needs a
lower bound as well as an upper one, after an arm passed a threshold by
overshooting in the direction nobody checked.

NO COSINE HERE IS QUOTED PAST TWO DECIMALS. The same deterministic per row
quantity read +0.0818 and +0.0667 on identical batches in two runs, half a
standard error apart, from GPU accumulation order alone.

WHAT THIS DOES NOT DO. It does not score anything, it does not train anything,
and it produces no AUC. It decides whether one printed diagnostic in a sibling
file may be read, and if so how. It cannot rank the two estimators for quality,
because reliability is not correctness: an estimator can be perfectly self
consistent about a direction that is biased, which is the whole reason the
paired dispersion measurement in `w4_estimator` remains the headline and this
remains a supplement.

THE SPECTRAL ARM, `--objective spec` or `both`, REGISTERED 2026-08-13 BEFORE IT RAN

Why it is here rather than in a file of its own. The question is the same
question: is this objective's gradient signal or noise at a batch size anyone
can afford. Everything that answers it already exists in this file, so the arm
is a third reward through the same frozen model, the same halves, the same
paired control, and the same standard errors.

Why this reward and not the mid band one that was proposed. The teacher forced
mid band excess is not a defect. `w4_forcing` ran the identical pipeline on data
the model generated itself, where nothing is wrong by construction, and inside
the registered band the forcing increment read 1.173 against a construction
artefact of 1.172, residual 1.002. `w4_dtcal` and `w4_launch` then found the one
step conditionals calibrated on every axis they tested. So there is nothing for
a supervised spectral term to correct, and the only spectral discrepancy still
standing is arm C: the model generating FREELY reads +0.1825 at 8.3 sd against
people. That is a joint property, it is only reachable through sampling, and
that is why this arm is measured on free running samples with a score function
gradient rather than teacher forced.

What is different about it, and the reason it might be better conditioned than
the critic reward. The W1 critic objective is a genuine batch statistic: a
sorted matching couples every row to every other, so one row's coefficient
depends on the whole draw. Band power is a per ROW scalar, b_i, one row's mean
standardised speed power inside `w4_timing`'s registered 11 to 41.5 Hz band. The
measured loss is the LINEAR one,

    L = mean_i b_i / H        advantage   adv_i = (b_i - mbar) / H

with H the human population mean of the same quantity and mbar this half's own
mean serving only as a baseline. The direction it estimates is fixed, so its
reliability is a clean statement about whether the direction that moves band
power is estimable at all.

WHY THE LINEAR FORM AND NOT THE MATCHING ONE, decided before the arm ran and on
a number measured on human data alone. The natural loss is (mbar / H - 1)^2,
whose advantage is the same (b_i - mbar) multiplied by a scalar 2(mbar / H - 1)
/ H. On corpus rows the per row band power has mean 0.1729 and standard
deviation 0.2229, so at 46 rows the half's own mbar carries a standard error of
19 percent of H while the model's excess is about 18 percent of H. That scalar's
SIGN would therefore flip on a large fraction of halves, and the squared form's
reliability would be dominated by prefactor sign noise rather than by anything
about the objective. Any real implementation would hold that prefactor on a
running estimate across steps rather than recomputing it per batch, which is the
linear form with a slowly varying scale. Measuring the strawman instead would
understate the objective, so the linear form is what is registered. Each half's
own mbar / H is recorded per batch, so the squared form's extra sign noise can
be read off afterwards without a second run.

WINDOW RETENTION, and why the batch is raised rather than the window shortened.
`w4_timing`'s one centred window of 64 samples at 125 Hz needs 512 ms of
resampled trace, and only 43 percent of corpus rows are that long. That is not a
defect of this code: it reproduces `w4_forcing`'s reported human retention of
43.5 percent to within a tenth of a point, which is the check that the band here
is the band there. It does mean a batch of 96 leaves halves of about 20, too
small to compare against a critic arm measured on halves of 46. The window is
the registered instrument and is NOT shortened to buy rows back. The batch is
raised instead, and both arms are then split on the SAME surviving rows so the
paired comparison is at one half size for both.

GATES, all paired batch by batch against arms measured on the SAME draws.

    r_spec above zero by 3 se AND above r_sf by 3 se
        the reward's per row structure is worth real reliability. A training arm
        is justified at ordinary batch size, which is the only cheap route left
        in this workstream.
    r_spec within 2 se of zero
        the spectral objective is as blind as the others at this batch. It buys
        NOTHING over the critic reward and needs the same order of magnitude
        more batch. No training arm without that batch.
    r_spec above zero by 3 se but within 2 se of r_sf
        both are equally conditioned. The reward is not the lever; the batch is.
    anything else
        in between, reported as such, no verdict.

The implied batch for a usable estimator is reported the same way the size sweep
reported it, from SNR = r / (1 - r) scaling linearly in the number of rows.

CAVEAT, stated before the number exists. H is estimated from corpus rows drawn
from the same pool the NLL control uses, not from the held out rows `w4_timing`
used for arm C, so it is not the arm C constant and no absolute defect size may
be read off this file. It sets the sign and rough scale of the prefactor only,
and at a discrepancy near 18 percent the sign is not in question.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, ".")
sys.path.insert(0, "research")
sys.path.insert(0, "research/autoloop")

from features import FEATURE_NAMES  # noqa: E402
from models.event_ar import EventARModel  # noqa: E402
from models.event_stream_polar import S_PAD_CLASS, TH_NULL_CLASS  # noqa: E402
from w4_estimator import DROP, SQUASH, squash, w1_terms  # noqa: E402
from w4_rollout import (  # noqa: E402
    CKPT, COOL_C, COOL_MAX_S, D, GATE_C, HZ, KILL_C, RESUME_C, TRAINED,
    anchor_nll, decode_batch, decode_indexed, dt_to_z, gpu_temp, load_human,
    token_logprob_pos,
)
from w4_softdec import soft_forward, straight_through  # noqa: E402

# w4_timing is imported LAST and with the environment restored around it. It
# calls os.environ.setdefault on nine serving variables at module scope. Those
# are already read by the time esp is imported above, so the import cannot
# change this process's sampling, but it would leave a different environment
# behind for anything that reads one later, and the base, quantile and size
# sweep runs in this file were taken without it. Snapshot and restore removes
# the question rather than arguing it. The band, the window and the speed
# signal come from there rather than being restated, so the quantity this arm
# trains on is the quantity that measured the defect.
_ENV_BEFORE = dict(os.environ)
from w4_timing import (  # noqa: E402
    BAND_HI_HZ, BAND_LO_HZ, signals, windows,
)
os.environ.clear()
os.environ.update(_ENV_BEFORE)

OUT_JSON = "research/w4_estimator_rel.json"


def flat_grad(model):
    """Current accumulated gradient as one flat float32 vector on the CPU.

    Kept off the GPU because six of these live at once and the card holds the
    model and the sampler's activations. Dots are taken in float64; float32
    returned 1.0064 for a pair of identical vectors earlier in this workstream.
    """
    return torch.cat([(torch.zeros_like(p).reshape(-1) if p.grad is None
                       else p.grad.detach().reshape(-1)).cpu()
                      for p in model.parameters()])


def cos(a, b):
    na = float(a.double().pow(2).sum()) ** 0.5
    nb = float(b.double().pow(2).sum()) ** 0.5
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float((a.double() * b.double()).sum()) / (na * nb)


def band_sel(w):
    """Which rfft bins of a w sample window fall inside the registered band."""
    f = np.fft.rfftfreq(w, d=1.0 / HZ)
    return (f >= BAND_LO_HZ) & (f <= BAND_HI_HZ)


def band_power(path, w, sel):
    """One row's mean standardised speed power inside the registered band.

    `signals` and `windows` are w4_timing's, so the resampler, the speed
    definition and the one centred window per trajectory are the same ones that
    measured arm C. The four line periodogram is written out rather than calling
    its `psd`, because that stacks rows and silently drops the short ones, and
    here every row has to keep its position in `keep`. A row that cannot produce
    a window returns None and is excluded from the arm by the caller.
    """
    sig = signals(path)
    if sig is None:
        return None
    seg = windows(sig["speed"], w)
    if seg is None:
        return None
    seg = seg - seg.mean()
    sd = seg.std()
    if sd < 1e-9:
        return None
    win = np.hanning(w)
    p = np.abs(np.fft.rfft((seg / sd) * win)) ** 2 / (win * win).sum()
    return float(p[sel].mean())


def paths_of(s_cls, th_cls, dt_cls, angles, keep):
    """The decoded path of every SURVIVING row, by the call decode_batch makes.

    Decoded a second time rather than returned from `decode_batch`, because
    widening that function's return arity would touch every caller in the
    workstream for one diagnostic. Rows in `keep` have already passed its length
    and finiteness checks, so the decode here cannot fail.
    """
    out = []
    for i in keep:
        s = s_cls[i]
        pad = np.flatnonzero(s >= S_PAD_CLASS)
        L = int(pad[0]) if len(pad) else len(s)
        p, _ = decode_indexed(dt_to_z(dt_cls[i][:L]), s[:L], th_cls[i][:L],
                              0.0, 0.0, angles[i])
        out.append(p)
    return out


def mean_se(v):
    """Mean and its standard error over INDEPENDENT batch readings.

    Plain divisor, deliberately. See the docstring: this is not the U statistic
    w4_cosse had to jackknife, because each batch contributes exactly one
    reading of each quantity.
    """
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=np.float64)
    if len(v) < 2:
        return float("nan"), float("nan"), len(v)
    return float(v.mean()), float(v.std(ddof=1) / len(v) ** 0.5), len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=24,
                    help="independent measurement batches. Each yields one "
                         "reading of every quantity")
    ap.add_argument("--batch", type=int, default=96,
                    help="96 to match w4_estimator and Job 2 exactly. The "
                         "halves are therefore 48, and every reliability is a "
                         "HALF batch reliability")
    ap.add_argument("--tau", type=float, default=4.0)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--critic-hid", type=int, default=128)
    ap.add_argument("--critic-out", type=int, default=8)
    ap.add_argument("--critic-lr", type=float, default=1e-3)
    ap.add_argument("--critic-warm", type=int, default=16,
                    help="batches of generated rows used to fit the critic "
                         "before any measurement. The generator never moves "
                         "during these, so the critic is trained against the "
                         "same frozen model it will grade")
    ap.add_argument("--critic-steps", type=int, default=40)
    ap.add_argument("--w1-target", choices=("sample", "quantile"),
                    default="sample",
                    help="what the W1 term is measured against. `sample` is "
                         "what every arm in this family does: a fresh draw of "
                         "n human rows per step. `quantile` uses the exact "
                         "order statistics of the WHOLE human reference at the "
                         "same n plotting positions, which is the population "
                         "target the sample draw is a noisy estimate of. The "
                         "coefficients are signs of a sorted comparison, so at "
                         "n 48 the sample draw flips them for reasons that "
                         "have nothing to do with the model")
    ap.add_argument("--sample-chunk", type=int, default=96,
                    help="rows per sampler forward pass. Decoupled from "
                         "--batch so the batch can be raised past what the "
                         "card fits in one pass")
    ap.add_argument("--objective", choices=("w1", "spec", "both"),
                    default="w1",
                    help="which rewards to measure. `w1` is the original two "
                         "estimator arm and is the default so the base, "
                         "quantile and size sweep runs stay reproducible. "
                         "`spec` measures only the band power reward. `both` "
                         "puts them on the same batches, which is the paired "
                         "comparison the spectral registration asks for.")
    ap.add_argument("--spec-window", type=int, default=64,
                    help="window length in 125 Hz samples, w4_timing's default")
    ap.add_argument("--spec-ref", type=int, default=4000,
                    help="corpus rows used to estimate the human band power "
                         "mean H")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--init", type=str, default=None)
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()

    t = gpu_temp()
    if t > GATE_C:
        print(f"  GPU at {t}C, above the {GATE_C}C launch gate. Not starting.")
        return
    out_json = (OUT_JSON.replace(".json", f"_{a.tag}.json") if a.tag
                else OUT_JSON)

    dev = torch.device("cuda")
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    ck = torch.load(a.init or CKPT, map_location=dev, weights_only=False)
    cap = int(ck["config"]["max_seq_len"])

    s2a = np.load(f"{D}/events_s2.npy", mmap_mode="r")
    dtha = np.load(f"{D}/events_dth.npy", mmap_mode="r")
    dta = np.load(f"{D}/events_dt.npy", mmap_mode="r")
    lens = np.load(f"{D}/events_len.npy")
    cond_all = np.load(f"{D}/events_cond.npy", mmap_mode="r")
    ok = np.flatnonzero(np.load(f"{D}/events_feat18_ok.npy"))
    perm = ok[rng.permutation(len(ok))]
    train_rows = perm[:400000]

    HX = np.load("data/human_ref_features_sir.npy").astype(np.float64)
    HX = HX[np.isfinite(HX).all(1)]
    rng.shuffle(HX)
    ctr = np.median(HX, 0)
    scl = np.percentile(HX, 75, 0) - np.percentile(HX, 25, 0)
    ctr_t = torch.tensor(ctr, dtype=torch.float32, device=dev)
    scl_t = torch.tensor(scl, dtype=torch.float32, device=dev)
    tk = [FEATURE_NAMES.index(f) for f in TRAINED]
    live = torch.tensor([i for i, f in enumerate(TRAINED) if f not in DROP],
                        device=dev)
    ZH = torch.tensor(((HX - ctr) / scl)[:, tk], dtype=torch.float32,
                      device=dev)

    model = EventARModel(**ck["config"]).to(dev)
    model.load_state_dict(ck["model_state_dict"])
    for p in model.parameters():
        p.requires_grad_(True)

    critic = nn.Sequential(nn.Linear(len(tk), a.critic_hid), nn.ReLU(),
                           nn.Linear(a.critic_hid, a.critic_hid), nn.ReLU(),
                           nn.Linear(a.critic_hid, a.critic_out)).to(dev)
    copt = torch.optim.Adam(critic.parameters(), lr=a.critic_lr,
                            weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    nparam = sum(p.numel() for p in model.parameters())
    print(f"\n  frozen model, {nparam / 1e6:.1f}M parameters, no generator "
          f"step is ever taken")
    print(f"  batch {a.batch}, halves of {a.batch // 2}, "
          f"{a.batches} measurement batches")
    print(f"  critic warmed on {a.critic_warm} batches against this same "
          f"frozen model", flush=True)

    therm = {"peak": 0}

    def thermal():
        t = gpu_temp()
        therm["peak"] = max(therm["peak"], t)
        if t >= COOL_C:
            c0 = time.time()
            while gpu_temp() > RESUME_C and time.time() - c0 < COOL_MAX_S:
                time.sleep(10)
            t = gpu_temp()
            therm["peak"] = max(therm["peak"], t)
        return t

    def draw():
        """One sampled, decoded batch: the hard features and the kept rows."""
        rows = np.sort(train_rows[rng.choice(len(train_rows), a.batch,
                                             replace=False)])
        cond = torch.tensor(np.asarray(cond_all[rows],
                                       dtype=np.float32)).to(dev)
        ang = np.arctan2(cond[:, 3].cpu().numpy().astype(np.float64),
                         cond[:, 2].cpu().numpy().astype(np.float64))
        model.eval()
        # Sampling is chunked so the batch can be raised well past what one
        # forward pass fits, which is the whole point of the size sweep.
        with torch.no_grad():
            parts = [model.sample(cond[c0:c0 + a.sample_chunk], seq_len=cap)
                     for c0 in range(0, len(cond), a.sample_chunk)]
        s = torch.cat([p[0] for p in parts])
        th = torch.cat([p[1] for p in parts])
        dt = torch.cat([p[2] for p in parts])
        del parts
        torch.cuda.empty_cache()
        X, keep, _ = decode_batch(list(s.cpu().numpy()),
                                  list(th.cpu().numpy()),
                                  list(dt.cpu().numpy()), ang)
        return s, th, dt, cond, ang, X, keep

    # ------------------------------------------------------------ critic warm
    # The generator is frozen throughout, so every warm batch is a sample from
    # the identical distribution the measurement batches come from.
    buf = []
    for w in range(a.critic_warm):
        if thermal() >= KILL_C:
            print(f"  GPU at or above the {KILL_C}C kill during the critic "
                  f"warm up. Stopping.", flush=True)
            return
        _, _, _, _, _, X, keep = draw()
        if len(keep) < 16:
            continue
        Xht = torch.tensor(X, dtype=torch.float32, device=dev)
        buf.append(squash((Xht - ctr_t) / scl_t)[:, tk].detach())
    XG = torch.cat(buf, 0)
    lab = torch.cat([torch.zeros(256, device=dev), torch.ones(256, device=dev)])
    for _ in range(a.critic_steps * a.critic_warm):
        gi = torch.randint(0, len(XG), (256,), device=dev)
        hi = torch.randint(0, len(ZH), (256,), device=dev)
        out = critic(torch.cat([XG[gi], squash(ZH[hi])], 0))
        copt.zero_grad(set_to_none=True)
        sum(bce(out[:, j], lab) for j in range(a.critic_out)).backward()
        copt.step()
    for p in critic.parameters():
        p.requires_grad_(False)
    print(f"  critic fitted on {len(XG)} generated rows, now frozen",
          flush=True)
    del buf, XG
    torch.cuda.empty_cache()

    # The population target, computed once because the critic is frozen. Every
    # human row's critic value, sorted per column, so a half of size n can read
    # off its n order statistics by interpolation instead of drawing n rows.
    with torch.no_grad():
        LH_sorted, _ = torch.sort(critic(squash(ZH)), dim=0)
    print(f"  W1 target `{a.w1_target}`, population reference {len(LH_sorted)} "
          f"rows", flush=True)

    def human_order_stats(n):
        """The reference's order statistics at the n plotting positions.

        Linear interpolation into the sorted population, at the same
        (i + 0.5) / n levels a sorted sample of size n estimates. This is what
        the fresh draw is a noisy estimate OF, so swapping it in changes the
        estimator's variance without moving what it is aimed at.
        """
        lv = (torch.arange(n, device=dev, dtype=torch.float32) + 0.5) / n
        pos = (lv * len(LH_sorted) - 0.5).clamp(0, len(LH_sorted) - 1)
        lo = pos.floor().long()
        hi = pos.ceil().long()
        w = (pos - lo.float()).unsqueeze(1)
        return LH_sorted[lo] * (1 - w) + LH_sorted[hi] * w

    # ------------------------------------------------- the spectral reference
    # H, the human population mean of the per row band statistic. Human token
    # streams go through the SAME decoder the generated ones do, which is what
    # w4_timing means by an identical resampler on every arm; taking the corpus
    # paths from anywhere else would put a second decoder in the comparison.
    kinds = {"w1": ("st", "sf"), "spec": ("spec",),
             "both": ("st", "sf", "spec")}[a.objective]
    sel_band = band_sel(a.spec_window)
    Hband = float("nan")
    if "spec" in kinds:
        rref = np.sort(train_rows[rng.choice(len(train_rows), a.spec_ref,
                                             replace=False)])
        hstreams, _ = load_human(rref, cap, s2a, dtha, dta, lens, cond_all)
        hb = []
        for sc, tc, dc, angle in hstreams:
            p, _ = decode_indexed(dt_to_z(dc), sc, tc, 0.0, 0.0, angle)
            if p is None or len(p) < 4:
                continue
            v = band_power(p, a.spec_window, sel_band)
            if v is not None:
                hb.append(v)
        Hband = float(np.mean(hb))
        print(f"  band {BAND_LO_HZ} to {BAND_HI_HZ} Hz, window "
              f"{a.spec_window} at {HZ:.0f} Hz, {int(sel_band.sum())} bins",
              flush=True)
        print(f"  human reference H {Hband:.4f} from {len(hb)} corpus rows",
              flush=True)
        del hstreams

    # ------------------------------------------------------------ measurement
    # Each spectral half appends its own mbar / H here, in the order the halves
    # are computed. Recorded rather than used: it is what makes the squared
    # form's prefactor sign noise readable without a second run.
    half_ratio = []
    pr = {"sf": [], "spec": []}

    def half_grad(kind, s, th, dt, cond, ang, Xht, keep, sel, bp=None):
        """One estimator's gradient over the kept rows listed in `sel`.

        The critic coefficients, and the spectral arm's prefactor and baseline,
        are computed WITHIN this half against this half's own reference, so two
        disjoint halves are independent draws of the same half batch estimator
        and not two coupled views of one batch statistic.
        """
        model.zero_grad(set_to_none=True)
        rows_t = torch.tensor(sel, device=dev)

        if kind == "spec":
            # The linear form. mbar is a baseline only and comes from this half,
            # never from the batch, so the two halves stay independent draws.
            # See the registration for why the squared form is not the one
            # measured and how its extra sign noise is recovered afterwards.
            b = bp[rows_t]
            mb = b.mean()
            adv = torch.zeros(len(keep), device=dev)
            adv[rows_t] = (b - mb) / Hband
            half_ratio.append(float(mb) / Hband)
            coeff = None
        else:
            with torch.no_grad():
                zt = squash((Xht - ctr_t) / scl_t)[:, tk]
                lg = critic(zt[rows_t])
                if a.w1_target == "quantile":
                    lh = human_order_stats(len(sel))
                else:
                    lh = critic(squash(ZH[torch.randint(0, len(ZH),
                                                        (len(sel),),
                                                        device=dev)]))
                _, coeff_h = w1_terms(lg, lh)
            # coeff_h is indexed by position within `sel`; lift it back to a
            # full length row indexed tensor so the chunk loop can address it
            # the same way both estimators do.
            coeff = torch.zeros(len(keep), a.critic_out, device=dev)
            coeff[rows_t] = coeff_h

            if kind == "sf":
                # coeff is zero off `sel`, so the row rewards are too, and the
                # baseline is the mean over this half alone.
                with torch.no_grad():
                    adv = (coeff * critic(zt)).sum(1)
                adv[rows_t] = adv[rows_t] - adv[rows_t].mean()

        if kind in ("sf", "spec"):
            # Participation ratio of the per row advantage, (sum|adv|)^2 over
            # n * sum adv^2. It is 1 when every row contributes equally and
            # 1/n when one row carries the whole gradient.
            #
            # Why it is recorded. A score function estimator's variance is set
            # by the reward's spread, and if the advantage is heavy tailed then
            # the EFFECTIVE number of rows is far below n and the reliability
            # grows much more slowly than the SNR model's linear rate. That is
            # the one mechanism that would produce a reliability ceiling across
            # a batch sweep, and this is the cheapest way to see it: pure
            # statistics on a tensor that already exists, no extra backward
            # pass, no effect on any gradient.
            with torch.no_grad():
                av = adv[rows_t].abs()
                s2 = float((av * av).sum())
                pr[kind].append(float(av.sum()) ** 2 / (len(sel) * s2)
                                if s2 > 0 else float("nan"))

        angt = torch.tensor(ang, dtype=torch.float32, device=dev)
        for i0 in range(0, len(ang), a.chunk):
            i1 = min(i0 + a.chunk, len(ang))
            sub = [j for j in sel if i0 <= keep[j] < i1]
            if not sub:
                continue
            loc = torch.tensor([keep[j] - i0 for j in sub], device=dev)
            row = torch.tensor(sub, device=dev)
            if kind in ("sf", "spec"):
                lp, _ = token_logprob_pos(model, s[i0:i1], th[i0:i1],
                                          dt[i0:i1], cond[i0:i1], False)
                (adv[row] * lp.sum(1)[loc]).sum().backward()
                del lp
            else:
                Xs, _ = soft_forward(model, s[i0:i1], th[i0:i1], dt[i0:i1],
                                     cond[i0:i1], angt[i0:i1], tau=a.tau)
                zs = squash((straight_through(Xht[row], Xs[loc])
                             - ctr_t) / scl_t)[:, tk]
                zmix = squash((Xht[row] - ctr_t) / scl_t)[:, tk].clone()
                zmix[:, live] = zs[:, live]
                (coeff[row] * critic(zmix)).sum().backward()
                del Xs, zs, zmix
            torch.cuda.empty_cache()
        return flat_grad(model)

    def nll_grad(rows):
        """Supervised teacher forced NLL on human tokens. The known good
        control, taken through the same accumulate and flatten path."""
        model.zero_grad(set_to_none=True)
        ah, akept = load_human(np.sort(rows), cap, s2a, dtha, dta, lens,
                               cond_all)
        M = len(ah)
        for j0 in range(0, M, a.chunk):
            j1 = min(j0 + a.chunk, M)
            grp = ah[j0:j1]
            L = max(len(r[0]) for r in grp)
            AS = torch.full((len(grp), L), S_PAD_CLASS, dtype=torch.long)
            ATH = torch.full((len(grp), L), TH_NULL_CLASS, dtype=torch.long)
            ADT = torch.zeros((len(grp), L), dtype=torch.long)
            for i, r in enumerate(grp):
                AS[i, :len(r[0])] = torch.from_numpy(r[0])
                ATH[i, :len(r[1])] = torch.from_numpy(r[1])
                ADT[i, :len(r[2])] = torch.from_numpy(r[2])
            acond = torch.tensor(np.asarray(cond_all[akept[j0:j1]],
                                            dtype=np.float32)).to(dev)
            nll = anchor_nll(model, (AS.to(dev), ATH.to(dev), ADT.to(dev)),
                             acond, False)
            (nll * (len(grp) / M)).backward()
            torch.cuda.empty_cache()
        return flat_grad(model)

    rec = {"r_st": [], "r_sf": [], "r_nll": [], "c_cross": [], "c_same": []}
    if "spec" in kinds:
        rec["r_spec"] = []
    hist = {"config": vars(a), "H_band": Hband, "batches": []}
    t0 = time.time()

    for b in range(1, a.batches + 1):
        if thermal() >= KILL_C:
            print(f"  GPU at or above the {KILL_C}C kill after "
                  f"{COOL_MAX_S}s of cooling. Stopping.", flush=True)
            break
        s, th, dt, cond, ang, X, keep = draw()
        if len(keep) < 32:
            continue
        Xht = torch.tensor(X, dtype=torch.float32, device=dev)

        bp = ratio = None
        if "spec" in kinds:
            # Band power is a per row scalar, so it is computed once for the
            # whole batch and each half indexes its own rows out of it. A row
            # whose window cannot be built is dropped from BOTH halves before
            # they are split, so the two remain the same estimator.
            vals = [band_power(p, a.spec_window, sel_band)
                    for p in paths_of(s.cpu().numpy(), th.cpu().numpy(),
                                      dt.cpu().numpy(), ang, keep)]
            ok_rows = [j for j, v in enumerate(vals) if v is not None]
            if len(ok_rows) < 32:
                continue
            bp = torch.full((len(keep),), float("nan"), device=dev)
            bp[torch.tensor(ok_rows, device=dev)] = torch.tensor(
                [vals[j] for j in ok_rows], dtype=torch.float32, device=dev)
            ratio = float(np.mean([vals[j] for j in ok_rows]) / Hband)
        else:
            ok_rows = list(range(len(keep)))

        idx = np.asarray(ok_rows)[rng.permutation(len(ok_rows))]
        h = len(idx) // 2
        A, B = sorted(idx[:h].tolist()), sorted(idx[h:2 * h].tolist())

        gA = {k: half_grad(k, s, th, dt, cond, ang, Xht, keep, A, bp)
              for k in kinds}
        gB = {k: half_grad(k, s, th, dt, cond, ang, Xht, keep, B, bp)
              for k in kinds}

        arows = train_rows[rng.choice(len(train_rows), a.batch, replace=False)]
        nA = nll_grad(arows[:a.batch // 2])
        nB = nll_grad(arows[a.batch // 2:])

        r = {"r_nll": cos(nA, nB)}
        if "st" in kinds:
            r.update({"r_st": cos(gA["st"], gB["st"]),
                      "r_sf": cos(gA["sf"], gB["sf"]),
                      "c_cross": 0.5 * (cos(gA["st"], gB["sf"])
                                        + cos(gB["st"], gA["sf"])),
                      "c_same": 0.5 * (cos(gA["st"], gA["sf"])
                                       + cos(gB["st"], gB["sf"]))})
        if "spec" in kinds:
            r["r_spec"] = cos(gA["spec"], gB["spec"])
        for k in rec:
            rec[k].append(r.get(k, float("nan")))
        hist["batches"].append({"batch": b, "kept": int(len(keep)),
                                "n_spec": len(ok_rows), "ratio": ratio,
                                "half_ratio": half_ratio[-2:], **r})
        line = (f"  batch {b:>3}  kept {len(keep):>3}  "
                + ("" if "st" not in kinds else
                   f"r_st {r['r_st']:+.2f}  r_sf {r['r_sf']:+.2f}  ")
                + ("" if "spec" not in kinds else
                   f"r_spec {r['r_spec']:+.2f}  m/H {ratio:.3f}  ")
                + f"r_nll {r['r_nll']:+.2f}  {thermal()}C  "
                + f"{time.time() - t0:.0f}s")
        print(line, flush=True)
        del gA, gB, nA, nB, Xht
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- verdict
    # The rows an arm actually split on, which is the surviving rows for the w1
    # arms and the window bearing subset when the spectral arm is in play.
    _key = "n_spec" if "spec" in kinds else "kept"
    nrows_half = (float(np.mean([x[_key] for x in hist["batches"]])) / 2.0
                  if hist["batches"] else float("nan"))

    for k, v in pr.items():
        if v:
            pm = mean_se(v)
            print(f"\n  per row advantage participation ratio, {k}: "
                  f"{pm[0]:.3f} se {pm[1]:.3f} over {pm[2]} halves")
            print(f"  effective rows {pm[0] * nrows_half:.0f} of "
                  f"{nrows_half:.0f}. Below about a third is the heavy tail "
                  f"regime\n  in which reliability grows far slower than the "
                  f"SNR model's linear rate.")
            hist[f"participation_{k}"] = {"mean": pm[0], "se": pm[1]}

    m = {k: mean_se(v) for k, v in rec.items()}
    print(f"\n  {'quantity':<10}{'mean':>9}{'se':>9}{'n':>5}")
    for k in ("r_st", "r_sf", "r_spec", "r_nll", "c_cross", "c_same"):
        if k in m:
            print(f"  {k:<10}{m[k][0]:>+9.2f}{m[k][1]:>9.2f}{m[k][2]:>5}")

    # Paired differences, batch by batch, so the shared batch draw cancels.
    def paired(u, v):
        d = [x - y for x, y in zip(rec[u], rec[v])
             if np.isfinite(x) and np.isfinite(y)]
        return mean_se(d)

    def implied(mean_r, n_half):
        """Rows per half a reliability of 0.5 needs, from SNR = r / (1 - r)
        scaling linearly in the number of rows. Reported as the size sweep
        reported it and not as a promise."""
        if not np.isfinite(mean_r) or mean_r <= 0:
            return float("nan")
        return n_half * (1.0 - mean_r) / mean_r

    if "spec" in kinds:
        p_nll_spec = paired("r_nll", "r_spec")
        print(f"\n  paired r_nll - r_spec    {p_nll_spec[0]:+.2f} se "
              f"{p_nll_spec[1]:.2f}   "
              f"{abs(p_nll_spec[0] / p_nll_spec[1]):.1f} se")
        p_spec_sf = paired("r_spec", "r_sf") if "sf" in kinds else None
        if p_spec_sf is not None:
            print(f"  paired r_spec - r_sf     {p_spec_sf[0]:+.2f} se "
                  f"{p_spec_sf[1]:.2f}   "
                  f"{abs(p_spec_sf[0] / p_spec_sf[1]):.1f} se")
        rt = [x["ratio"] for x in hist["batches"] if x["ratio"] is not None]
        nh = float(np.mean([x["n_spec"] for x in hist["batches"]])) / 2.0
        print(f"  band power ratio m/H     {np.mean(rt):.3f}, the excess this "
              f"reward is pointing at")

        spec_live = m["r_spec"][0] > 3 * m["r_spec"][1]
        beats_sf = (p_spec_sf is not None
                    and p_spec_sf[0] > 3 * p_spec_sf[1])
        ties_sf = (p_spec_sf is not None
                   and abs(p_spec_sf[0]) < 2 * p_spec_sf[1])
        print("\n  THE SPECTRAL GATE")
        if not spec_live:
            print("  BLIND. The band power gradient is within 3 se of zero "
                  "self agreement\n  at this batch. The per row reward buys "
                  "nothing the critic reward did\n  not, and this objective "
                  "needs the same order of magnitude more rows.\n  No training "
                  "arm at ordinary batch size.")
        elif beats_sf:
            print(f"  LIVE, and better conditioned than the critic reward by "
                  f"more than 3 se.\n  The per row structure is worth real "
                  f"reliability. Implied rows per half\n  for a reliability of "
                  f"0.5: {implied(m['r_spec'][0], nh):.0f}, against "
                  f"{nh:.0f} here. This is the only\n  cheap route left in the "
                  f"workstream and a training arm is justified.")
        elif ties_sf:
            print(f"  LIVE but no better than the critic reward, within 2 se "
                  f"of it. Both are\n  equally conditioned, so the reward is "
                  f"not the lever and the batch is.\n  Implied rows per half "
                  f"for a reliability of 0.5: "
                  f"{implied(m['r_spec'][0], nh):.0f}, against {nh:.0f} here.")
        else:
            print(f"  In between the registered bands, no verdict. Implied "
                  f"rows per half for\n  a reliability of 0.5: "
                  f"{implied(m['r_spec'][0], nh):.0f}, against {nh:.0f} here.")
        hist["spectral_gate"] = {
            "spec_live": bool(spec_live), "beats_sf": bool(beats_sf),
            "ties_sf": bool(ties_sf), "ratio": float(np.mean(rt)),
            "implied_half_rows": implied(m["r_spec"][0], nh),
            "n_half": nh}

    if "st" not in kinds:
        hist["peak_c"] = therm["peak"]
        with open(out_json, "w") as f:
            json.dump(hist, f, indent=2)
        print(f"\n  peak {therm['peak']}C, {time.time() - t0:.0f}s, "
              f"wrote {out_json}", flush=True)
        return

    p_nll_sf = paired("r_nll", "r_sf")
    p_nll_st = paired("r_nll", "r_st")
    p_same_cross = paired("c_same", "c_cross")
    print(f"\n  paired r_nll - r_sf      {p_nll_sf[0]:+.2f} se "
          f"{p_nll_sf[1]:.2f}   {abs(p_nll_sf[0] / p_nll_sf[1]):.1f} se")
    print(f"  paired r_nll - r_st      {p_nll_st[0]:+.2f} se "
          f"{p_nll_st[1]:.2f}   {abs(p_nll_st[0] / p_nll_st[1]):.1f} se")
    print(f"  paired c_same - c_cross  {p_same_cross[0]:+.2f} se "
          f"{p_same_cross[1]:.2f}   shared batch draw, not agreement")

    sf_zero = abs(m["r_sf"][0]) < 2 * m["r_sf"][1]
    st_zero = abs(m["r_st"][0]) < 2 * m["r_st"][1]
    blind = (p_nll_sf[0] > 3 * p_nll_sf[1]) and sf_zero
    falsified = abs(p_nll_sf[0]) < 2 * p_nll_sf[1]

    print("\n  PRIMARY, the instrument check")
    if blind:
        print("  The score function estimate is BLIND at this batch size: it "
              "is\n  within 2 se of zero self agreement and below the "
              "supervised control\n  by more than 3 se. w4_estimator's "
              "cosine line says nothing about\n  RELAX and must be WITHDRAWN, "
              "not read. Job 2's prediction holds.")
    elif falsified:
        print("  FALSIFIER MET. The score function estimate is as self "
              "consistent as\n  the known good control. The attenuation worry "
              "is unfounded and\n  w4_estimator's near zero cosine is a real "
              "statement about direction.")
    else:
        print("  NEITHER. The score function sits between the control and "
              "zero, so\n  the cosine is attenuated but not destroyed. Read "
              "the disattenuated\n  value below and quote the attenuation "
              "alongside it, never the raw\n  cosine on its own.")

    print("\n  SECONDARY, the RELAX question")
    if sf_zero or st_zero:
        print("  NOT READABLE. A reliability is indistinguishable from zero, "
              "so the\n  disattenuated ratio divides by noise. Not quoted, by "
              "registration.")
        d = float("nan")
    else:
        d = m["c_cross"][0] / (m["r_st"][0] * m["r_sf"][0]) ** 0.5
        print(f"  disattenuated cos(mu_st, mu_sf)  {d:+.2f}")
        if d > 1.3:
            print("  UNINTERPRETABLE, not agreement. The attenuation model has "
                  "broken.\n  Do not read this as the estimators agreeing.")
        elif d < 0.2:
            print("  The two estimators genuinely disagree in direction. RELAX "
                  "is not\n  worth building and the pathwise gradient is not "
                  "estimating the\n  surrogate it is written to estimate.")
        elif d > 0.7:
            print("  They agree in direction once noise is removed. Estimator "
                  "bias is\n  NOT the collapse mechanism, which sends task 24 "
                  "back to the objective.")
        else:
            print("  Between the registered bands. Neither conclusion is "
                  "licensed.")

    hist["summary"] = {k: {"mean": m[k][0], "se": m[k][1], "n": m[k][2]}
                       for k in m}
    hist["paired"] = {"r_nll_minus_r_sf": p_nll_sf,
                      "r_nll_minus_r_st": p_nll_st,
                      "c_same_minus_c_cross": p_same_cross}
    hist["disattenuated"] = d
    hist["verdict"] = {"blind": bool(blind), "falsified": bool(falsified),
                       "sf_zero": bool(sf_zero), "st_zero": bool(st_zero)}
    hist["peak_c"] = therm["peak"]
    with open(out_json, "w") as f:
        json.dump(hist, f, indent=2)
    print(f"\n  peak {therm['peak']}C, {time.time() - t0:.0f}s, "
          f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
