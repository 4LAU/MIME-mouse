# W3 proposal: teach the model to arrive, and to vary

Status: DRAFT for L. Nothing here is running yet. Local GPU work needs no
sign-off; anything with cloud cost stops for approval first.

## The two measured problems

The program's remaining gap is not one problem but two, and both were
measured this week.

First, the model does not aim. It IS told the destination (the conditioning
vector encodes the exact displacement to cover; verified against the
training data 2026-07-21), but it draws the whole path open-loop: it never
checks where it is against where it is going, so step errors accumulate to
a median miss of 58 pixels. Forcing exact arrival afterwards costs about
+0.078 on the realism score, and the cost grows with the size of the miss.
Delivered paths must land on the exact pixel, so today every served path
pays the worst band of that tax.

Second, the model is too average. Human movement has rare extremes: fast
flicks, sharp jerks, odd hesitations. Those extremes are inside what the
model CAN produce, but it draws them far too rarely, and the detector reads
that missing spread as a tell. Turning up sampling randomness makes the
score worse, not better, because the added variance is the wrong kind. The
spread has to be learned, not injected.

W1 and W2 closed the cheap alternatives: imitating winning examples does not
transfer the signal, and steering toward hand-picked statistics hits the
stats without fooling the detector. Scale alone is also the wrong lever: the
failure is in what the model is asked to learn, not how much it saw.

## The design

One retrained model, two changes, both trained on the same 4M human
trajectories the current model already uses.

1. Close the loop. Instead of one fixed "cover this displacement"
   instruction for the whole draw, condition every step on the REMAINING
   offset to the target. The training data supervises this for free: at any
   point along a recorded human path, the remainder of that path is exactly
   how a human closes the remaining gap, including the slow-down and small
   corrections near the end that Fitts' law describes. A model trained this
   way steers itself back on course as it draws, so arrival stops depending
   on perfect open-loop execution. The goal is a native median miss within
   a few pixels, where the correction tax measured this week is 2 to 4
   times smaller, and the residual correction is invisible or nearly so.

2. Let the model pick its own character. The current production model is
   handed a set of target motion statistics drawn from a bank of real human
   paths. W2 proved that pipeline's ceiling: hitting statistics does not
   fool the detector. Replace it with a small learned "character" input the
   model infers from real paths during training and draws fresh at serving
   time. Path-level variety then lives in that input by construction, which
   attacks the too-average problem at its root: the model no longer has to
   explain all human variety with per-step noise, which is exactly what
   collapses into averaged, samey paths today.

## Pre-registered probes, in order

P1, aiming only. Retrain the current architecture unchanged except the
conditioning becomes the per-step remaining offset instead of one fixed
displacement. Success: median native miss 15px or less, AND one-shot score
with exact arrival enforced materially below the current 0.728. This
isolates how much of the arrival tax disappears when the model steers,
before any character work.

P2, aiming plus learned character. Add the learned character input on top
of P1's conditioning. Success: one-shot exact-arrival score materially
below P1's, with the spread of the tell features (max velocity, max
acceleration, jerk) moving toward human levels rather than past them.

Each probe starts with a one-epoch timing burst so the full-run cost is
known before committing to it. All runs follow the standing thermal policy
and get ledger rows. The single decision metric stays exactly as
registered: RF-OOB AUC, N=2000 per class, against the validation human
set, with arrival enforced on every scored path from here on.

## How the probes iterate: an autoresearch loop

Once P1's training script exists and the timing burst has measured cycle
cost, iteration runs as an autoresearch loop in the style Karpathy
published in March 2026: an agent repeatedly proposes one change to the
mutable training code, runs a short fixed-budget training burst, scores
the result on the frozen metric, keeps the change if the number improved
and reverts it if not, and logs every attempt. This repo is already built
for it. The scoring contract is frozen code the loop cannot touch, the
ledger is the append-only experiment log, and the protected checkpoint
and eval sample sit outside the loop's reach by standing rule.

Two adaptations from Karpathy's setup. The keep-or-revert score is a pair,
native median miss plus one-shot AUC with arrival enforced, because P1 can
trade one against the other and a single number would hide that. And the
loop iterates on short training bursts rather than full retrains; the
timing burst tells us the burst length at which rankings are trustworthy.
The full retrain happens once, at the end, with the winning recipe. The
loop is not appropriate earlier than this: it needs a mutable script to
iterate on and a cheap experiment to score, and until P1's first version
runs, neither exists.

## What this does not touch

The 0.504/0.513 CANDI headline stays untouched, as do the protected
checkpoint, the held-out eval sample, and all scoring code. The 2-second
serving budget holds: both changes are training-time only; serving is
still one draw from one model.

## Costs and the decision for L

Everything above runs on the local GPU in watchdogged bursts; the timing
burst tells us whether a full retrain is hours or days before we commit.
Cloud spend, if a run turns out to need it, comes back to L with the
measured local timing as evidence.

The fallback question is now answered: with exact arrival enforced and the
judge shown the corrected candidates, the K-filter fallback reads about
0.58 (row W3_groundwork_...e7f67c96), a small penalty over the 0.54 to
0.56 it reads without arrival. Two details matter. The judge must see the
corrected paths; correcting after judging costs three times as much
(about 0.65). And 0.58 at roughly a second per request is packageable as
an interim product today, while still short of the 0.50 goal, which
remains this proposal's job.

## P3 addendum, written 2026-07-21 after P1 and P2 closed

P1 and P2 are both closed, and the full record is in the ledger and in
HANDOFF_W3.md. What they establish, jointly: this trunk plans a whole path
from its static conditioning and ignores feedback bolted on afterwards, so
it will never natively arrive; and its decoder squashes any character
command about five to one, so it will never natively vary enough, with
guidance recovering only part of the gap before the output goes
off-manifold. On top of that, the product pipeline itself is saturated. A
better generator, two generators mixed, and three different arrival
corrections all leave the corrected K-filter number at 0.58 to 0.59. The
floor belongs to the model family.

P3 is therefore a new model line, not another round of surgery on this
one. The design that follows most directly from the evidence is continuous
inpainting over resampled paths. Instead of generating a stream of speed
and turn tokens and then paying a correction tax to hit the target, the
model generates the path as coordinates on the fixed 125Hz grid the
evaluator already resamples to, with the start and end pixels pinned as
known values and the model filling in the middle, the same way image
models fill in a masked region. Arrival stops being a constraint to
enforce and becomes part of the canvas. Duration is drawn empirically at
serving exactly as today, so the 2 second budget holds: serving is still
one draw from one model.

That design attacks both closed problems at once. Exact arrival is free by
construction, deleting the 0.078 correction tax and the whole correction
question. And generating in continuous coordinate space removes the
token-decoding bottleneck that P2 located: there is no discrete
speed-class vocabulary whose conditional distribution the trunk can
collapse toward the mean, and diffusion-style objectives are trained
explicitly to match the data distribution rather than the most likely
token, which is where the under-dispersion enters.

Pre-registered first probe, local GPU, fixed budget: a small model on this
design, trained on the same frames as everything else, evaluated one-shot
at N=2000 with the standing metric, no correction step because none is
needed. Gates, in order: it must beat 0.70, the best guided one-shot the
current family reaches, and the tell-feature spread must clear half of
human, which no configuration of the current family reaches at any
guidance weight. Below 0.65 with spread passing justifies scaling it;
between 0.65 and 0.70 buys one diagnosed iteration; above 0.70 kills the
line in one cycle, cheaply. Cloud spend, if scaling ever needs it, comes
back to L with measured local timing as evidence, per the standing rule.
