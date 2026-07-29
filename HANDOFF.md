# Handoff

The single session-start document for this repo. Read it before touching
anything. Repo is MIME-mouse on WSL2 Ubuntu, branch main. Nothing is
mid-flight; the GPU is idle and no runs are pending.

Earlier handoffs and plans are in archive/ with a note on why each was
overtaken. They are not instructions.

## Start here: where things stand, 2026-07-26

The mandate is research, not shipping. The goal is a generative model that
returns ONE trajectory from A to B and scores 0.50. Anything that generates
several candidates and picks among them is out of scope, so the selection
product and its 0.58 are not the number being moved.

  standing single-trajectory number   0.6986
  what it means                       one path per request, decided before it
                                      is scored, landing on the requested
                                      whole pixel every time, no candidate pool
  how to reproduce it                 research/w3_jog_on_resid.py with
                                      --ckpt event_polar_4m_resid_v2.pt
  scorer                              research/autoloop/scoring.py, NOT
                                      evaluate.py (see Evaluation below)

Everything bolted onto the current model family has now been measured and is
flat: six aiming fine-tunes (P1), the character latent plus guidance (P2),
pool mixing, correction geometry, the learned adversarial critic
(ADVERSARIAL_CRITIC.md, Phase 1 failed), the RL pilot (RL_PILOT.md), and
supervised imitation of selection winners (W1). Getting materially below 0.70
means a different architecture, which is design work and GPU time, not another
tuning cycle.

The last finding is the one to carry forward, because it changes how to read
everything before it. The 0.078 arrival tax was blamed on the model missing
its target by 58 px. That was wrong. Comparing raw model output to corrected
model output, instead of only comparing corrected output to the human, showed
the sampler was fine and the correction was injecting the defect. Every
measurement taken before 2026-07-26 used the defective correction as its arm.

Next steps are at the bottom of this file, in order.

## What this project is

Academic research into generative models of human mouse movement. The question
is whether a model can produce individual trajectories whose motion statistics
are indistinguishable from recorded human ones.

Realism is measured with a classifier two-sample test, the standard way to
evaluate a generative model when you have real samples to compare against: fit
a RandomForest to separate synthetic paths from held-out human paths on 18
motion features, and report its out-of-bag AUC. 0.50 means the classifier
cannot tell the two apart, which is the target. Higher means the model has a
tell. METHODOLOGY.md explains why a classifier is used instead of per-feature
distance measures: it catches joint-distribution mismatches that marginal
comparisons miss. When these documents say "detector" they mean this
RandomForest, which is a measuring instrument built and trained inside this
repo, not any external or third-party system.

## The product shape, and the constraint that drives W3

L's target: a request for a trajectory from point A to point B returns ONE
generated trajectory, on demand, in about 2 seconds, scoring 0.50.

Pre-generated libraries are out. Covering enough distances and angles takes
hours to days, which cannot happen on a client device at import. And a stored
path was drawn for a different A to B, so it must be stretched or rotated to
reach the requested endpoint. That transform is expensive on the metric, and
this repo has already measured it (EXPERIMENTS.md): retrieval plus rotation
0.94, best corpus rotate 0.670, similarity transform 0.682, nearest-match
library serving 0.815. Against a 0.50 goal, warping a stored path costs more
than the entire remaining gap. That is the real reason libraries are out, and
it holds even with unlimited pre-generation time.

The trajectory must also LAND exactly. L's tolerance is 0px: the delivered
path has to end on the requested pixel, because the intended use is moving to
a specific point on screen. Anything less makes the output unusable regardless
of its realism score.

## The arrival tax, measured 2026-07-20, reading corrected 2026-07-26

The measurements in this section stand. The conclusion drawn from them, that
the lever is native arrival, was wrong and is reversed further down. Read both.

Exact arrival is geometrically free. You can always move points to hit the
target. What costs is what moving them does to the realism score, and that had
never been measured here.

Row W3_groundwork_...0e65cfd1, results in
research/w3_landing_price_results.json. N=6000, seed 42, integer targets,
one-shot event_polar_4m_fc_v2 at the locked recipe. Every correction forces
exact arrival and rounds to whole pixels:

  uncorrected (arrives exactly 0.3% of the time)           0.6500
  magnitude-weighted additive correction, 100% arrival     0.7283
  rotate + scale correction, 100% arrival                  0.7342

The uncorrected 0.6500 reproduces the 0.6544 recorded earlier, so the harness
is consistent. Forcing arrival costs about +0.078.

The tax scales with how far the path missed:

  miss 2 to 5px     tax +0.019
  miss 5 to 15px    tax +0.020
  miss 15 to 40px   tax +0.035
  miss 40 to 100px  tax +0.051

The model currently misses by a median 58px, 30.3% of the travel distance, so
it sits in the most expensive band. Landing natively within about 15px would
cut the tax roughly 2.5x. Caveat: per-bin absolute AUCs are inflated by small
sample size (n=412 to 2277 against a 2000-per-class recipe); only the
within-bin tax delta is a fair comparison.

Corrections must stay on the integer lattice. Real mouse coordinates are whole
pixels, and EXPERIMENTS.md records that sub-pixel positions alone read 0.69, so
a fractional coordinate is its own tell regardless of anything else.

Why the model misses at all: CORRECTED 2026-07-21. The conditioning
(log_dist, log_dur, cos angle, sin angle) IS the exact net displacement of
the training path (verified: exp(log_dist) times (cos, sin) reproduces the
summed event steps to the pixel), so the model is told the endpoint. What it
lacks is feedback: the whole path is drawn open-loop against one fixed
instruction, position error accumulates step by step, and nothing corrects
it. There is no correction step anywhere in experiments/event_stream_polar.py.
See archive/W3_PROPOSAL.md; the fix proposed there was per-step residual
re-conditioning, and it failed. Superseded by the 2026-07-26 section below:
the tax is mostly the correction operator, not the aim. Not more
endpoint information.

The consequence drawn at the time: because the cost scales with correction
size, arrival has to be part of what the model is asked to do rather than a fix
applied afterwards. That is what sent six fine-tunes at native aiming, and all
six failed. The tax turned out to be mostly a defect in the correction operator
itself, which no amount of better aiming would have reached. What survives is
the narrower claim: post-hoc correction is never free, not even at 2 px.

## The arrival tax and the product number survive a degeneracy control, 2026-07-26

Rows W3_groundwork_...86be14c7 and ...0893dd67.

The scorer reads more than motion. Real recordings are whole pixels at irregular
times, and features.resample_trajectory interpolates them onto the 125 Hz grid,
which leaves runs of points that are bitwise exactly collinear. Their angle
difference is exactly 0.0, np.sign(0.0) is 0, and num_direction_changes counts
nothing there. Generated paths carry no such exact structure, so the count
roughly doubles. research/p3_ceiling_probe.py measured it: nudging real paths by
one billionth of a pixel moves the contract AUC from 0.65 to 0.84 on its own.

Every arrival correction shifts points and re-rounds them, which is exactly the
operation that erases that structure. So the +0.078 arrival tax above could have
been the cost of moving the path, the cost of erasing the structure, or both.

research/degeneracy_panel.py separates them. It leaves scoring.score_features
untouched and adds a second reading in which the arm AND a human reference built
from raw recordings are both nudged by 1e-9 px, which removes the exact
structure from both sides. A difference that survives that is about movement.

Arrival tax, research/w3_arrival_tax_control.py, on the same 6000 cached
one-shot paths as the original run, no regeneration:

  arm             contract   rebuilt   control
  none              0.6500    0.6451    0.6337
  additive          0.7283    0.7353    0.7085
  rotate            0.7444    0.7377    0.7080
  real (holdout)    0.4778    0.4851    0.4990

  tax             original  contract    control
  additive         +0.0783   +0.0783    +0.0749

The contract column reproduces the July 20 run exactly. The tax survives at
+0.075. It is motion damage, so W3's premise holds and P1's six cycles were
aimed at a real problem. The rotate arm reads 0.7444 rather than the recorded
0.7342 because the original rotate code is not in the repo and this imports
w3_correction_lab.correct_similarity, which falls back to additive on degenerate
paths.

Product number, research/w3_product_control.py, pool_s42_k32 at K=32, judge
seeds 0/1/2: contract 0.5833 (sd 0.0073, reproducing the recorded 0.5833),
control 0.5889 (sd 0.0085). Inside seed noise. The packageable number is honest.

Building the panel's human reference is the part to get right. The first attempt
used the {split}_positions.npy grid, which prepare_training_data.py made by
resampling and distance-normalizing, so its paths are fractional and its
movement_duration is quantized to 175 distinct values against the contract
reference's 903. That rebuild read 0.6382 against the contract reference and
biased every arm scored against it, which produced a wrong intermediate result
(a tax of +0.055) before the cause was found. The panel now builds from the raw
segmented recordings in data/demo_pool.npz and reads 0.4922, chance.
research/degeneracy_panel.py --self-test asserts both properties; any future
reference must clear it.

Side note on P3. Its representation ceiling of 0.6456 (research/
p3_ceiling_results.json) is the same measurement as that 0.6382, so the ceiling
is dominated by the 192-slot grid's quantized duration, with the direction-change
degeneracy second. The conclusion that P3 could not reach its 0.70 gate is
unchanged; the cause is better understood.

CANDI read through the same panel, research/w3_candi_control.py, row
W3_groundwork_...426caf73. Fresh n=2000 at the published convention (steps 200,
guide 0.15, perp 0.85, rotate, seed 42), checkpoint md5 verified unchanged:

  arm             published  contract   rebuilt   control
  rounded            0.7520    0.9520    0.9493    0.9185
  unrounded               -    0.7525    0.7677    0.8290
  real (holdout)          -    0.4778    0.4851    0.4990

Two things fall out.

The published 0.752 reproduces on the UNROUNDED arm, 0.7525, not the rounded
one. research/phase_a_baseline.py's --no-round help text says the default
matches the published config; it does not, and research/phase1_score.py's
docstring (no_round=True) is the correct account. Rounding costs 0.20. Worse,
experiments/candi.py:296 rounds unconditionally with no way to turn it off, so
the module you would actually call to serve CANDI produces the 0.95 arm, not the
published one.

And CANDI's number is the one the control does move. 0.7525 becomes 0.8290, so
about 0.077 of it was the exact-arithmetic effect rather than motion. That
matters for the family comparison, because the event-stream arms corrected for
arrival read 0.7085 (additive) and 0.7080 (rotate) in the same control column.
Under the contract scorer the two families look close (0.728 against 0.752);
read fairly they are not, and the event-stream family is ahead by about 0.12.
The ranking is the same in the rebuilt column, so it does not depend on the
nudge.

The older CFM and DDPM line is dense-125Hz too and has not been read through the
panel. It is far off the pace either way, so it is low priority.

## Numbers are not currently comparable across model families

CANDI applies endpoint correction as standard, so its numbers are measured on
paths that arrive and include the cost of arriving. CORRECTED 2026-07-27: this
used to say the correction is rotate. experiments/candi.py:45 defaults
CANDI_CORRECT to "additive"; rotate is what the published convention sets by
hand. It matters for reading old probes, because anything that did not set the
variable ran on the additive operator, which is the defective one. Every event-stream number, including W0's 0.539
fallback and W1's 0.654 one-shot baseline, pays none of that cost and is
flattered by roughly the tax above. Do not compare across families without
stating which side pays.

CORRECTED 2026-07-26: this paragraph used to attribute the 0.504 and 0.513
headline to CANDI. It is not a CANDI number. EXPERIMENTS.md:2839 records it as
the 33-dim judge's set-level selection result, and EXPERIMENTS.md:2319 names the
generator behind that candidate pool as event_polar_4m_fc_v2.pt. The headline
belongs to the event-stream family with selection on top. CANDI's own best is
0.752, and the panel above reads it at 0.829.

## W3 P1 verdict, 2026-07-21

P1 (native arrival through closed-loop conditioning) is closed as a FAIL
after six ledgered cycles, rows W3_P1_...e56aced0 through ...9e999b85. The
short version: a feedback channel telling the model its remaining offset is
either ignored (when the static conditioning already carries the endpoint),
too weak to redirect the draw (draft-error variant, v6: realism held at raw
0.6688 but the miss stayed at 67.8px), or only becomes informative when the
reveal order is forced left to right, which wrecks realism beyond recovery
at fine-tune budgets (v3 to v5, raw 0.88 to 0.99). The model plans the whole
path from the static conditioning; per-iteration feedback does not change
that plan. Native exact arrival needs a different architecture, not another
conditioning channel. Do not reopen P1 as a conditioning tweak; the full
post-mortem is in the ledger notes of row ...9e999b85. The 0.58 corrected
fallback remains the packageable product answer. P2 (learned character
latent) proceeds independently; it targets under-dispersion, not arrival.

## W3 P2 verdict, 2026-07-21

P2 (learned whole-path character latent) produced the program's first real
one-shot improvement and then failed to move the product number. Three
training cycles (char_v1 to v3, posterior collapse fixed with free bits) plus
a latent-control probe established that the representation works: latents
encoded from extreme real paths generate measurably more extreme output. The
bottleneck is the generator's conditional response, which squashes any
character command about five to one, the same wall W2 hit from the feature
side. Classifier-free guidance at sampling recovers part of it: corrected
one-shot AUC 0.728 to about 0.70, replicated across spec seeds 42 and 43 and
across serving configs (rows ...779b23b1, ...c4c975ca). Pushing guidance
harder lifts the tell-feature spread further but goes off-manifold and the
AUC worsens, the same failure shape as raw temperature.

The product routing test then came back flat (row ...7957c1d4). A fresh 64k
candidate pool from the guided model, same 2000-spec stream as the base pool,
scored through the corrected-then-judged K filter: base pool 0.5833, guided
pool 0.5903 at K=32, seed noise about 0.01, statistically indistinguishable.
Mind the seed noise when reading old single-draw fallback numbers; the
much-quoted 0.5747 was a lucky draw. Conclusion: one-shot quality gains of
this size do not compound through best-of-32 selection, because selection
already recovers the same tail from the base pool. The honest product number
at exact arrival stays 0.58 to 0.59. Moving it needs either a much larger
one-shot gain or something that changes what selection can reach, not a
better average candidate. A mixed-pool probe (both generators, 64 candidates
per request, row ...177c7795) confirmed generator diversity does not move it
either, and a correction-scheme lab (row ...d25f77e2) confirmed the arrival
tax is not correction geometry: bending only the tail of the path or
rotating the whole path instead of shearing it lands within seed noise of
the additive baseline on both pools. Every cheap lever on the product
pipeline is now measured and flat. The 0.58 to 0.59 floor belongs to the
model family itself.

Ops note for this machine: two worker crashes during scoring were numpy
AVX512 invalid-opcode faults under WSL2 (same instruction offset both
times). Setting NPY_DISABLE_CPU_FEATURES to the AVX512 family fixes it.

## The arrival tax is mostly the operator, not the aim, 2026-07-26

Rows W3_groundwork_...f72bea9d and ...9852d77e. This reverses the reading in
"The central finding" above, which attributed the +0.078 to the model missing
by 58 px and pointed the work at native arrival. The measurement that broke it
open was putting raw model output next to corrected model output per speed
class, instead of only comparing corrected output to the human.

  mean |turn| deg, speed class 12 to 21   human 18.85  raw 14.53  corrected 26.14
  mean |turn| deg, speed class 32         human 10.36  raw 10.47  corrected 18.07
  straight share, speed class 32          human 66.3%  raw 61.4%  corrected 44.6%

The sampler is at or below the human almost everywhere. correct_additive puts
the excess in. It adds a smooth drift to every position and rounds each one
independently; rounding a ramp is a staircase, and the risers land in the
middle of straight runs, where one riser is a 45 or 90 degree turn.

correct_jog (research/w3_aiming_price.py) spends the error as |err_x| + |err_y|
single-pixel changes on the longest steps, longest first, and leaves every
other step byte identical to the model's own. Arrival is exact by construction.

  event_polar_4m_fc_v2, n=6000      additive 0.7283   jog 0.7144   -0.0139
  event_polar_4m_resid_v2, n=1998   additive 0.7210   jog 0.6986   -0.0223

0.6986 is the standing single-trajectory number: one path per request, exact
pixel arrival, no candidate pool, no selection. The gain is larger on resid_v2,
which the operator was not developed against, than on fc_v2, which it was.

Verification, research/w3_jog_verify.py: arrival asserted at 100% on both arms;
collapse flag and collapsed-feature list identical before and after; 5 of 18
dispersion ratios move closer to the human on both arms; subsampling without
replacement, 12 draws at 0.75, favours jog in 11 of 12 on each arm. Two earlier
instruments were wrong and both are recorded in that file's docstring. Seed
sweeping is a no-op here (the contract pins RF_SEED=42 and the jitter is 0).
Bootstrapping with replacement saturates the forest at 0.80 to 0.89 and
compresses the gap toward zero.

Two dead ends worth not repeating. Error diffusion along the path is
arithmetically identical to the additive correction (carrying the fraction
forward and rounding each step equals rounding the running total). Gating the
discharge on a step-length floor creates an unbounded carry that dumps tens of
pixels on one event, and scored 0.744 to 0.764 at 4, 8 and 16 px floors.

The six P1 aiming checkpoints are now tabled in EXPERIMENTS.md with their
per-checkpoint miss and AUC; they had only ever existed as loose JSON.

## The three re-measurements under the repaired operator, 2026-07-27

Rows W3_groundwork_...cb65f60c, ...7c280abe and ...82cb7306. These are the
three next steps the previous session left, done in order. All CPU, all on
cached paths, no GPU and no checkpoint touched. None of them reaches 0.50 and
none was meant to. Read the third one before designing anything.

### 1. The selection product does not move, and selection turns out to be
### operator-specific

research/w3_product_jog.py, pool_s42_k32 at K=32, judge seeds 0/1/2, through
the degeneracy panel.

  arm             contract   rebuilt   control   sd(contract)
  additive          0.5833    0.6042    0.5889         0.0073
  jog               0.5909    0.5855    0.5771         0.0029
  jog-served        0.6790    0.6876    0.6652         0.0073
  real (floor)      0.4778    0.4851    0.4990

The additive arm reproduces the recorded 0.5833 to four places, so the harness
is consistent. Swapping in the repaired operator moves it +0.0076 against a
seed sd of 0.0073, which is nothing. The one-shot gain of -0.0139 does not
compound through selection, exactly the shape the P2 routing test found. The
packageable answer stays 0.58 to 0.59.

The jog-served arm is the part worth keeping. It takes the additive-fitted
judge's winners and serves those same candidates with jog instead, and reads
0.6790, a full 0.096 worse than serving them the way the judge expected. The
two judges agree on which candidate wins on only 52% of specs. So the judge is
not picking realistic candidates. It is picking candidates that its own
operator will damage least, and a winner set does not carry across operators.
Anyone quoting 0.58 has to say which operator both the judge and the server
used, and they have to be the same one.

### 2. The arrival tax still scales with the miss, so aiming is worth about
### a quarter of the remaining gap

research/w3_tax_bands_jog.py. Two corrections first.

The handoff said the tax "is 0.067 with jog on resid_v2". It is not. 0.067 is
the ADDITIVE tax on that model (0.7210 against raw 0.6538). The jog tax is
0.0449. The repaired operator cuts the tax by about a third, not by a hair.

  model       raw     additive        jog     tax add    tax jog
  fc_v2    0.6500       0.7283     0.7144      0.0783     0.0644
  resid_v2 0.6538       0.7210     0.6986      0.0672     0.0449

Then the measurement, and it needed two instruments because the obvious one is
confounded. Splitting the arm by how far it actually missed makes the jog tax
look flat: 0.010 to 0.020 across every band from 2 to 100px, against additive's
monotonic 0.019 to 0.051, which reproduces the 2026-07-20 record exactly. Taken
alone that would say better aiming buys nothing.

It is the wrong read. Each band holds different paths at a different raw AUC
(0.77 to 0.91), so the correction is being priced against a moving floor, and
in the high bands the forest is already near saturation and cannot register
more damage. Holding the path fixed and injecting a target at a chosen distance
removes the confound, and on the model's own raw paths the jog tax does scale:

  injected miss px        1        5       15       40       60       80
  fc_v2 jog tax      0.0123   0.0253   0.0247   0.0495   0.0792   0.1019
  resid_v2 jog tax   0.0042   0.0063   0.0240   0.0443   0.0627   0.0980

Landing within 15px instead of the current 55 to 58px is worth about 0.036 to
0.050 under the repaired operator. The remaining gap from 0.6986 to 0.50 is
0.199, so native arrival is worth roughly a quarter of it. Aiming is not a dead
lever. P1 stays closed because a conditioning channel bolted onto this trunk
cannot deliver it, not because the prize disappeared, and native exact arrival
is a live requirement for P3.

One caveat on the operator itself: jog is WORSE than additive beyond a 100px
miss (fc_v2 tax 0.077 against 0.044). Expected, since spending several hundred
pixels as single-pixel jogs has to touch nearly every step. It is the right
operator in the regime better aiming would put us in, not everywhere.

### 3. The finding P3's design brief rests on does not survive the raw column

research/w3_raw_column_reread.py. This is the one that decides how much of the
record can be trusted, and the answer is: less than hoped, in a specific place.

Audit first. Of the nine W3 probes that scored a correct_additive arm, eight
had no raw column. w3_stall_pattern is the only exception, which is why it is
the probe that caught the operator. Four of the eight concluded something about
TURNING, which is precisely what the operator manufactures.

The envelope_ceiling decomposition is the one that matters, because its
conclusion is the sentence the P3 brief is built on: turning carries the gap,
and the LOCAL wobble half carries it while the whole-path excursion half does
not. Re-run over three arms, what each family is worth if matched perfectly:

  fc_v2                    raw    additive        jog
  envelope              0.0028      0.0169     0.0128
  turning               0.0601      0.1432     0.1167
  derivatives           0.0036     -0.0015     0.0115
  turning: wobble       0.0185      0.1022     0.0708
  turning: excursion    0.0174      0.0192     0.0262

  resid_v2                 raw    additive        jog
  turning               0.0510      0.1160     0.0804
  turning: wobble       0.0189      0.0943     0.0598
  turning: excursion    0.0129      0.0074     0.0151

On the model's own output wobble is worth 0.0185 and excursion 0.0174. They are
the same size. The 5x asymmetry that said "a per-position head is where the gap
is" is the correction, not the model, and it replicates as an artifact on both
checkpoints. Turning as a whole is worth 0.060 on raw against 0.143 on
additive, so more than half of even the headline number was operator.

Two neighbouring findings fall with it. style_variance's premise was that
angular_velocity_mean is the RF's heaviest feature at weight 0.113; on raw it
is 0.060, below curvature_mean and path_efficiency. And turn_floor's "the model
over-turns 2.5x at 100 to 400 px/s" reverses sign. Mean absolute turn per
motion event:

  speed class    px/s    human      raw   additive       jog   human share
  1 to 5          125    31.94    26.57      37.72     28.80         22.7%
  6 to 11         202    31.16    30.89      41.08     25.94          7.4%
  12 to 21        250    18.85    14.53      26.14     18.50          8.3%
  22 to 31        319    22.84    21.93      28.33     20.95          9.3%
  32 to 99       1020     9.80     9.42      11.39     10.64         45.4%

The model UNDER-turns in every class. The correction overshoots it, and jog
lands close to human almost everywhere.

What survives: the gap is still in turning rather than the velocity envelope
(envelope is worth 0.003 on raw), and the model still under-turns at low speed,
26.6 degrees against the human 31.9 in the biggest class. What does not
survive: the local-versus-whole-path split. On correct measurements those two
halves are the same size, so "P3 must be locally dispersion-calibrated" cannot
be justified by that finding any more and has to be re-derived before anything
is designed against it.

## Where the program stands

- W0 (done): per-request K=32 candidate filtering reads 0.539 at ~1s/request.
  Backfilled row, never independently re-run, and does not include arrival.
- W2 (done, FAIL): steering generation toward target feature values hit those
  values without reducing the AUC.
- W1 (done, clean NEGATIVE): fresh-init supervised training on set-level
  winners reads 0.8331 one-shot against a 0.70 gate, worse than the 0.6544
  baseline it started from. Winner imitation does not internalise the
  selection signal. Rows W1_scratch_...9407d9fd, ...b4996735, ...694e74a5.
- W3 groundwork (done): the failure is conditional under-dispersion in the
  training objective, not a capacity ceiling and not a sampling knob. Human
  extremes are inside the model's support but drawn far too rarely, and
  raising sampling temperature makes the output easier to tell apart, not
  harder, because the added variance is off-manifold. Rows
  W3_groundwork_...ce210375, ...b7753a76. archive/PLAN.md's "W3 = SCALE" is
  superseded: scaling would spend money against the wrong diagnosis.

## What the score actually is, 2026-07-27

Two runs on top of the three re-measurements, both CPU, both raw output.
`research/w3_p3_fork.py` (row ...d0b3d7de) and `research/w3_route_variance.py`
(row ...b2fe71f4). Together they replace the design reason the raw-column
re-read destroyed, and the replacement points somewhere else.

RETRACTED IN PART, same day, by the two runs in "The gate, and what it took
down" below. Read that section before using anything here. What follows is
still an accurate account of what the instruments said; the last subsection
explains why two of the three conclusions do not survive their controls.

### The gap is coverage, not texture, and that survives the raw column

w3_missing_paths' covered / uncovered split was scored on a single additive
arm like the seven other audited probes, so it could not be used as it stood.
Re-run across all three arms on both checkpoints:

```
fc_v2                  all real   covered   uncovered    spread
raw                      0.6451    0.4237      0.9066    0.4829
additive                 0.7353    0.4821      0.9400    0.4579
jog                      0.7165    0.4981      0.9386    0.4405
recorded                 0.7353    0.4821      0.9400    0.4579

resid_v2                 0.6534    0.4628      0.9248    0.4621   (raw)
                         0.7178    0.5157      0.9334    0.4177   (additive)
                         0.6992    0.5057      0.9335    0.4278   (jog)
```

The additive row reproduces the record exactly. Unlike the wobble split, this
one survives, and it is STRONGER on raw: fc_v2 spread 0.4829 raw against
0.4579 additive. On its own raw output the model reads 0.4237 against the three
quarters of human movement the forest places it near, which is below chance,
and 0.9066 against the quarter it does not. The operator was slightly masking
this effect, not creating it.

So the model is not slightly wrong everywhere. It is indistinguishable from
people on most of human movement and absent from a quarter of it, and the
entire score comes from the absence.

### It is not global under-dispersion either

std(arm) / std(human) per descriptor, raw output, computed off the resampled
path rather than the 18 features so it does not restate what the forest saw:

```
descriptor          fc_v2 raw   resid_v2 raw
detour_ratio             0.35           0.22
overshoot                0.66           0.37
paused_fraction          0.92           0.95
n_pauses                 1.17           1.02
reversals                0.93           0.97
peak_speed               1.22           0.93
duration_s               0.94           0.89
straight_dist_px         1.28           1.33
median ratio             0.94           0.94
```

Pacing, pausing and reversals are at human spread. Two descriptors collapse,
and both are whole-path route shape. Same family that the raw re-read showed
is worth as much as local wobble, so two independent instruments now agree.

### The mechanism, after the first candidate was falsified

The obvious explanation was variance concentration: detour_ratio is a sum over
every emitted step, so an autoregressive sampler should concentrate it and the
collapse should deepen on longer paths. It was pre-registered that way and it
reverses:

```
straight px       steps h  steps m   fc_v2 detour   resid_v2 detour
20 to 59               23       37          0.153             0.100
59 to 134              35       55          0.303             0.187
134 to 267             45       64          0.438             0.519
267 to 501             54       72          0.459             0.620
501 and up             69       88          0.792             0.723
```

Control descriptors sit near 1.0 in every bin, so the trend is not the binning.
Concentration is dead. The collapse is a SHORT-path failure.

The coverage profile says what the missing paths are. The uncovered quarter is
ordinary at the median and differs entirely in the tail:

```
group           n   straight px   detour p50   detour p90   steps p50
covered      1000           212         1.08         1.61          47
uncovered    1000           150         1.07         3.85          39
all real     4000           187         1.06         2.14          41
model (raw)  5998             -         1.07         1.69           -
```

resid_v2 replicates: uncovered p90 4.04, model p90 1.59. And the human's route
is not set by the request at all. Free share 1.00, R squared of log detour on
log distance 0.003, human detour sd 3.967.

Put together: people choose a route freely, and on short movements they
sometimes wander three to four times the direct distance. The model never does,
its p90 is 1.69. A coherent excursion needs dozens of consecutive steps to
agree on going the wrong way, and independent per-step sampling cannot produce
that at any temperature. Raising EVENT_CHOICE_TEMP adds per-step entropy
without adding the correlation. On long requests ordinary jitter inflates
travelled distance enough to fake detour variance, which is why the collapse
only shows nakedly on short ones.

That single fact sits behind three separate results already on record: the
coverage split, P2's decoder squashing a character latent five to one, and P1's
six failures. All three are a per-step decoder that cannot hold a path-level
commitment. The P3 target is therefore a persistent route commitment plus
endpoints exact by construction, and NOT local dispersion calibration.

## The gate, and what it took down, 2026-07-27

The paragraph immediately above was the P3 brief for about an hour. Two runs
killed it, rows ...429342ee and ...b70bc3af, both CPU, both under twenty five
seconds. P1 spent six fine-tunes on a prize nobody had priced first, so these
priced this one first.

### The route tail is not the gap (`research/w3_oracle_route.py`)

Two instruments pointing opposite ways.

The clean one is an ablation with no synthetic paths anywhere. Thin real human
paths by rejection sampling until their detour distribution matches the
model's, which strips the wandering tail (p90 2.18 down to 1.74, against the
model's 1.69) and changes nothing else, then score them through the ordinary
contract.

```
arm                                        contract   rebuilt   control
fc_v2 raw                                    0.6500    0.6451    0.6337
fc_v2 + injected routes                      0.7584    0.7600    0.7525
human, route tail clipped to model           0.5236    0.5164    0.5096
human, duration clipped (null control)       0.5152    0.5003    0.5017
real (holdout)                               0.4778    0.4851    0.4990
```

0.5236 against a 0.4778 floor looks like 27 percent of the gap until the null
control is read. Thinning on duration_s instead, a descriptor the model already
matches at 0.94 spread, reaches 0.5152 on its own. So rejection thinning buys
0.0373 by itself and the route-specific effect is 0.0084, about five percent of
the gap and inside what the instrument can resolve.

The forward instrument went the wrong way. Bending the model's own paths with a
smooth excursion sized from the human distribution moved 0.6500 to 0.7584.
Mechanics were sound: 46.1 percent of paths bent, endpoint drift at most
5.6e-12 px so arrival stayed exact by construction, speed profile preserved
step by step and duration allowed to grow. On its own this arm is weak evidence
because it cannot separate "the tail is not the gap" from "a sine excursion
does not look human". Read with the ablation it is consistent.

### The coverage split is mostly a narrowing artefact (`research/w3_uncovered_anatomy.py`)

This was meant to ask what the uncovered quarter is, since detour turned out to
be a correlate rather than a cause. Against that quarter the answer looked
sharp: the forest needs turning most (fc_v2 worth 0.2037, alone 0.8653),
wobble beats excursion three to one on raw output (0.1364 against 0.0447),
every family is worth zero or negative against the covered quarter, and the
model sits 0.72 sd below the uncovered half on curvature_std while sitting 0.06
sd above the covered half. Both checkpoints agree.

Then the control. Those halves are chosen BY the forest, so splitting on
separability and re-measuring separability is partly circular. Split the real
paths on a single raw feature instead, and split them at random to price the
narrowing itself:

```
fc_v2, 500 real paths per half        arm vs low   arm vs high
RANDOM quartile (control)                 0.6286
curvature_std                             0.9058        0.9302
curvature_mean                            0.9127        0.9299
path_efficiency                           0.9115        0.9174
max_deviation                             0.8978        0.8943
movement_duration                         0.8893        0.9016
```

A random quartile reads 0.6286, sd 0.0082 over three draws, matching the
whole-arm 0.6451. Every feature-defined quartile reads 0.89 to 0.93, including
movement_duration, which the model already matches. Narrowing the real side on
any feature at all, relevant or not, buys about +0.27 of apparent
separability. resid_v2 replicates (random 0.6303).

So the 0.9066 uncovered figure is what any narrow slice reaches and carries
close to no information, and the turning decomposition computed against it is
describing arm-versus-narrow-slice in general. The other side is still real:
0.4237 sits far BELOW the 0.6286 random control, so a quarter of human paths
genuinely do lie inside the arm's cloud. But no single feature identifies them,
so it is not a localisable deficit and not something an architecture can aim at.

### The artefact, priced (`research/w3_narrowing_audit.py`, row ...4f5f6329)

Rather than re-run every probe that might share the confound, it is measured
once as a curve. fc_v2, real side narrowed to a fraction, sample size held
equal across conditions so only the SELECTION differs:

```
kept       k    random    by feature     min     max   by forest   excess
1       2000    0.6503        0.6546  0.6469  0.6648      0.6650    0.0147
0.5     1000    0.6420        0.8199  0.7808  0.8497      0.8213    0.1793
0.25     500    0.6076        0.8943  0.8777  0.9304      0.9066    0.2990
0.125    250    0.5791        0.9449  0.9155  0.9875      0.9350    0.3558
0.0625   125    0.5306        0.9694  0.9517  1.0000      0.9882    0.4576
```

Random narrowing does not inflate, it slightly deflates. Narrowing on a single
feature buys +0.287 at a quarter, and the LEAST inflating of all 18 features
still reaches 0.8777. The forest's own split reaches 0.9066, which is +0.012
over an arbitrary feature. resid_v2 replicates. The inflation grows as the
slice shrinks.

Both sides are exposed. Narrowing the ARM to a quarter buys +0.272 by feature
and +0.239 by forest. So w3_landing_price's 2 to 100px band gradient, the
number that sent six P1 fine-tunes, was read off inflated absolute AUCs. Its
within-band deltas partly cancel that, but they were still priced against a
moving floor, which is why the fixed-path injection sweep replaced it.

### The correction: the coverage story survives, with its direction reversed

The sweep above only takes each feature's TOP quartile, which controls the
uncovered side and not the covered one. Taking every feature's BOTTOM quartile
too:

```
                     as quoted    random    excess   best single feature
fc_v2    uncovered      0.9066    0.5923   +0.3143
         covered        0.4237    0.5923   -0.1686                0.8716
resid_v2 uncovered      0.9248    0.6259   +0.2990
         covered        0.4628    0.6259   -0.1631                0.8732
```

Narrowing in either direction on any feature inflates: the lowest any single
feature's bottom quartile reaches is 0.8716. The forest reaches 0.4237. So the
two halves are not symmetric artefacts.

- The UNCOVERED side is pure artefact. +0.31 excess, against +0.29 for an
  arbitrary feature. It carries no information beyond being narrow.
- The COVERED side is real, and strongly so. The forest identifies a quarter of
  real human paths that genuinely sit inside the arm's cloud, 0.44 below what
  any single feature reaches, and the property is multivariate: no one feature
  finds those paths.

The readable fact is therefore "a quarter of human movement is fully covered",
NOT "a quarter is missing". The missing-quarter reading is what w3_missing_paths
and w3_p3_fork both quoted, and it is the artefact.

### The rule this leaves behind

Any arm scored against a narrowed subset of EITHER side needs a same-size
random-subset control, and a same-size single-feature control in the direction
being claimed. Report the excess, never the raw AUC against a subset.

Static audit of the record is in `w3_narrowing_audit.py`'s `AUDIT` dict and in
the results JSON. EXPOSED: `w3_missing_paths`, `w3_p3_fork`,
`w3_critic_coverage`, `w3_landing_price`. Exposed in interpretation only:
`w3_corpus_coverage`, which reads corpus survival rates rather than an AUC, so
the inflation does not enter its instrument but its "model cannot produce"
group label is unsupported. Clean: `w3_envelope_ceiling`,
`w3_raw_column_reread`, `w3_oracle_route`, and the injection half of
`w3_tax_bands_jog`.

### Where that leaves P3

No localised deficit survives. The arm is diffusely separable across human
movement, whole arm 0.645 and random slices 0.61, and the one real subpopulation
result points at what the model already does WELL rather than at a gap. The
only surviving measured requirement is endpoints exact by construction, worth
0.036 to 0.050, which rests on `w3_tax_bands_jog`'s injection sweep and is
untouched by any of this.

## What the gap is made of, and the second architecture proposal it killed, 2026-07-27

The narrowing audit left P3 with no localised target, so the next question was
not WHERE the gap is but WHAT KIND of thing it is. Two probes, both structural
rather than subset-based, so the narrowing artefact cannot reach either.

### Calibration is a third of it (`research/w3_joint_structure.py`, row ...4dfd438c)

Rank-transform every one of the arm's 18 features onto the human quantiles for
that feature. Afterwards all 18 marginals are exactly human by construction and
whatever the forest still finds is dependence structure. Fitted on one human
half, scored against the other, because fitting and scoring on the same rows
overstated the marginals by nearly two to one on the first attempt.

```
              as it is    marginals matched    marginals worth    share of gap
fc_v2           0.6801               0.6221            +0.0580             32%
resid_v2        0.6684               0.6059            +0.0624             37%
```

So roughly two thirds of the surviving gap is not per-feature calibration. The
reference arm for the other extreme, human marginals with dependence destroyed
by independent column shuffles, reads 0.9998, so dependence is something the
forest can see enormously well on this feature set when it is wrong.

The rank correlation matrices name the disagreement: Frobenius gap 0.746 on
fc_v2 and 0.854 on resid_v2. Every one of fc_v2's worst eight pairs involves
`path_efficiency` and every one of resid_v2's involves `angular_velocity_mean`.
The theme both times is that in people how hard you move is coupled to how
directly you move and how much you turn, and in the model those couplings sit
near zero.

One correction to the record while here. EXPERIMENTS.md convicts the pair
`mean_acceleration` / `mean_jerk`, human r=-0.025 against synthetic r=+0.999.
That figure is Pearson on a different sample and is not reproducible as stated.
These features carry extreme outliers, `mean_acceleration` spans -2.3e5 to
5.1e7 on the human reference, so Pearson here is decided by a handful of paths:
the human pair reads 0.9999 on Pearson, 0.193 on Spearman, and -0.138 inside
the slowest decile. On rank correlation on the event-stream model the pair is
arm +0.140 against human +0.193, which is not a defect at all.

### Repairing the couplings is worth nothing (`research/w3_coupling_gate.py`, row ...76303ff4)

Naming a coupling gap is not pricing it, and `w3_oracle_route` had already
shown once that a plausible-looking gap can be worth zero. So the couplings were
repaired artificially and re-scored, the same shape of gate.

The instrument is a Gaussian copula rewrite: map the arm to normal scores by
rank, whiten by its own Cholesky factor, recolour by the human's, map back onto
whichever marginals the arm is supposed to have. That fixes every pairwise rank
dependence exactly while leaving marginals untouched, so the two effects stay
separable and no arm receives a value it did not generate.

```
arm                                     AUC    vs raw   gap closed
--- fc_v2, theme feature path_efficiency
raw                                  0.6602    0.0000            -
marginals only                       0.6048   -0.0554         35%
couplings only                       0.6702   +0.0100         -6%
both                                 0.5899   -0.0703         45%
couplings: path_efficiency only      0.6668   +0.0066         -4%
scrambled target (control)           0.9999   +0.3397       -217%
--- resid_v2, theme feature angular_velocity_mean
raw                                  0.6507    0.0000            -
marginals only                       0.5968   -0.0539         37%
couplings only                       0.6581   +0.0073         -5%
both                                 0.5811   -0.0696         47%
couplings: angular_velocity_mean only 0.6732  +0.0225        -15%
scrambled target (control)           0.9998   +0.3491       -238%
```

Row ...0f13f30b supersedes the first run of this gate, ...76303ff4, whose fc_v2
column read 0.6276 and 0.6083 on the two matched arms. Marginal matching reads
values off the sorted human half, so an arm with more rows than the reference
lands several rows on every human value and the forest can see the resulting
grid. fc_v2 had 5999 rows against 2000 human, worth a free 0.018, and resid_v2
had 1998, worth nothing. Arms are now capped at the human fit size, which costs
nothing because the scorer balances to the smaller side regardless. The null
below is unaffected either way, since the couplings-only arm keeps its own
values and never touches the grid.

Real against real floor 0.5039.

Repairing every pairwise coupling to human moves the score by nothing, on both
checkpoints, and repairing only the theme feature each model's rank gaps point
at moves it slightly the wrong way. Two controls make that readable rather than
suspicious. The scrambled arm, the same transform aimed at a human correlation
matrix with its features relabelled, goes to 0.9999, so the transform has
enormous power and the null is not a no-op. And the repair was verified to
land: the normal-score correlation gap to human falls from 0.709 to 0.046 on
fc_v2 and 0.731 to 0.061 on resid_v2, with the worst surviving pair gap 0.05.
The couplings really were fixed and the forest really did not care.

### What that leaves, and it is the useful part

Marginals and pairwise dependence TOGETHER reach only 0.5899 and 0.5811 against
a 0.5039 floor. An arm can have exactly human marginals and exactly human
pairwise correlations and still be caught. So the majority of the surviving gap
is higher-order structure that no pairwise summary of these 18 features
describes, and any architecture argued for on the basis of a correlation table
is being argued for on a third of the problem at most.

Note the phrasing. The survivor is NOT outside the feature table: the forest
only ever sees these 18 numbers, so everything it catches is expressible in
them by definition. It is outside the AVERAGES and the SINGLE-NUMBER-PER-PAIR
summary of them. That distinction is what the next probe acts on.

This killed the second architecture proposal of the day. The first, a whole-path
continuous generative model, was already a closed family: EXPERIMENTS.md
measures DDPM at 0.9291, CFM at 0.9191, v136 spatial at 0.9899 and v137 eta=0 at
0.9546, against the discrete event-stream trunk's 0.645. The second, keep the
discrete trunk but make each step a joint speed-and-direction decision, rested
entirely on the coupling gaps above, and the gate says closing them is worth
zero. Neither should be revived without new evidence.

### The structure is conditional on duration (`research/w3_conditional_gate.py`, row ...f28ce8d0)

FIRST POSITIVE LOCALISATION THIS SESSION, and the only one that survives its
controls. Everything above is a null; this is not.

If the survivor is dependence that one correlation number per pair cannot hold,
the obvious candidate is dependence that CHANGES with something. In people it
does: split the human paths into thirds by duration and the rank correlation
between `max_velocity` and `curvature_mean` runs -0.42, -0.12, +0.07 across
them. It changes sign. A global correlation matrix records one number there and
throws the rest away.

So the repair was redone inside each tercile of a conditioning feature, which
hands the arm the human dependence structure as it varies rather than averaged
over. Strictly richer than the global repair, and it reduces to it when nothing
varies. Scored on the whole arm against the whole held-out human half, so no
subsetting enters the comparison and the narrowing rule does not apply.

The control is the same banded repair over RANDOMLY assigned bands. Three 18x18
correlation matrices estimated from a third of the rows each are noisier than
one from all of them, and noise alone moves an AUC, so the random-band arm is
the only honest baseline. It carries identical estimation noise and no
conditioning information.

```
conditioner                  fc_v2 AUC   vs random     resid_v2 AUC   vs random
global repair (no bands)        0.5899                       0.5811
RANDOM bands (control)          0.5836                       0.5794
path_efficiency                 0.5936     +0.0101           0.5601     -0.0193
angular_velocity_mean           0.5892     +0.0056           0.5750     -0.0043
angular_velocity_std            0.5725     -0.0111           0.5734     -0.0059
velocity_skewness               0.5778     -0.0058           0.5790     -0.0004
movement_duration               0.5592     -0.0244           0.5491     -0.0303
```

`movement_duration` replicates 8 of 8 across seeds 42, 7, 1234 and 99 on both
checkpoints, range -0.0115 to -0.0256 against a random-band spread of about
0.004. Nothing else replicates: `path_efficiency` is 6 of 8 and goes the wrong
way twice on fc_v2.

Where that puts the total. From raw to the duration-banded arm is 65% of the
distance to the floor on fc_v2 (0.6602 to 0.5592, floor 0.5039) and 69% on
resid_v2 (0.6507 to 0.5491). Marginals plus duration-conditional dependence is
about two thirds of the whole problem.

A second reading points the same way. After the global repair, the features the
forest still leans on have alone-AUC at or BELOW chance on resid_v2
(`mean_jerk` 0.4850, `movement_duration` 0.4924, `curvature_mean` 0.4909) while
still carrying leave-one-out worth. Individually they say nothing; the forest is
using them only in combination. That is what conditional structure looks like
from the detector's side.

The brief this first suggested, that the model produces one regime and rescales
it, was tested the same day and is FALSE. See the next section. The AUC result
above stands; the reading of it did not.

### The duration brief is wrong, and what replaces it (`research/w3_duration_response.py`, row ...187cacde)

Reading the trunk first: it is ALREADY handed the duration. Its conditioning
vector is `[log_dist, log_dur, cos(angle), sin(angle)]`, with `log_dur` drawn
from the duration prior before a single event is generated. So the question was
never whether to give it duration, it was whether it uses it.

It does. Distance was held inside a 129 to 254 px band, the duration prior was
replaced with draws from one human duration tercile at a time, and 1500 paths
were generated per band on identical specs.

```
                        commanded    realized     obeyed
resid_v2   band 0          0.183s      0.182s
           band 1          0.425s      0.425s
           band 2          0.983s      0.948s        96%
fc_v2      band 2          0.983s      0.958s        97%
```

And the dependence structure swings with it. Mean rank-correlation swing across
the three bands, over all 153 pairs: human 0.191, resid_v2 0.156, fc_v2 0.142.
The model is not flat. It is not uniformly under-coupled either: the median
per-pair ratio of model to human coupling strength is 1.00 on both checkpoints
and the signs agree 90 to 93 percent of the time.

A second correction, and this one is on the section above. Holding distance
fixed collapses most of the swing that motivated it. `max_velocity` against
`curvature_mean` runs -0.42, -0.12, +0.07 across duration terciles of ALL human
paths, and -0.093, +0.170, +0.107 inside a single distance band. Duration and
distance are strongly related in people, so a large part of what looked like a
duration effect was a distance effect wearing its clothes. The banded-repair
AUC result is unaffected, it is an AUC and it replicated 8 of 8, but its
mechanism story was wrong.

**What replicates, on both checkpoints, is narrow and worth having.** In the
longest duration band every one of the six largest structural misses involves
`path_efficiency`:

```
pair, longest band              human    resid_v2    fc_v2
max_velocity / path_efficiency  -0.650     -0.208    -0.248
std_velocity / path_efficiency  -0.649     -0.161    -0.187
mean_velocity / path_efficiency -0.641     -0.169    -0.157
max_acceleration / path_eff     -0.562     -0.206    -0.263
std_acceleration / path_eff     -0.543     -0.181    -0.243
std_jerk / path_efficiency      -0.472     -0.167    -0.233
```

In people a long fast movement bows away from the straight line, hard. In both
models it stays nearly straight no matter how fast it goes, at about a third of
the human coupling. This converges with `w3_joint_structure`, which named
`path_efficiency` as fc_v2's theme feature from a completely different
instrument.

This is NOT the route-shape idea `w3_oracle_route` killed. That injected detours
independent of speed and made the arm worse. This says detour must be COUPLED to
speed and duration, which the injection test never varied. Different claim, and
the injection result does not bear on it either way.

Priced the same day, and it is not the carrier. See below.

### The speed-to-straightness defect is a correlate, not the carrier (`research/w3_efficiency_gate.py`, row ...b8cfaca3)

Same discipline as every other gate today: named is not priced. All arms share
the same duration terciles and the same human band marginals, so the only thing
that varies is which couplings get repaired, and the base is banded-marginals
rather than the random-band arm. That choice matters. The random-band arm also
loses the duration-conditional marginal match, so it is not a clean base for a
COUPLING decomposition; it is reported below but not used as the reference.

Four seeds, both checkpoints, mean AUC.

```
arm                                    fc_v2    resid_v2
raw                                   0.6555      0.6537
banded marginals (base)               0.5729      0.5684
  + path_efficiency couplings only    0.5791      0.5754
  + path_efficiency and max_deviation 0.5858      0.5742
  + every coupling EXCEPT it          0.5796      0.5621
  + all couplings                     0.5511      0.5479
random bands, target only             0.6080      0.6119
real against real floor               0.4992      0.4992
```

Repairing the speed-to-straightness couplings ALONE makes the arm worse on both
checkpoints, +0.0062 and +0.0070, and adding `max_deviation` does not rescue it.
Repairing every coupling except it is roughly neutral, +0.0067 and -0.0063.
Only the complete per-band repair helps, -0.0218 and -0.0205.

So no single feature carries the win, and partial repair is actively
detectable: splicing one correct row into an otherwise wrong matrix leaves an
internally inconsistent joint structure, which is its own signature. The
speed-to-straightness gap is real as a description, it replicates on both
checkpoints, and it is a correlate of the deficit rather than its cause. Same
shape as the detour result in `w3_oracle_route`. Do not build on it.

Useful byproduct, and it corrects the section above. The duration-banded win
splits roughly evenly between marginals-within-band and couplings-within-band:
from the global repair at 0.5899 and 0.5811, banded marginals alone take 44 and
38 percent of the remaining distance and banded couplings the other 56 and 62.
`w3_conditional_gate` attributing its win to couplings was imprecise.

The standing mandate is research, not shipping: reach 0.50 with a generative
model that returns a single trajectory from A to B. Anything that generates
candidates and picks among them is out of scope, so the K=32 selection product
is not the target and its 0.58 is not the number being moved.

Steps 1 to 3 of the previous list are DONE, 2026-07-27, and written up in "The
three re-measurements under the repaired operator" above. What they changed:

- the packaged 0.58 is unmoved and still honest, with the caveat that the judge
  and the server must use the same arrival operator
- aiming is worth about 0.04 to 0.05, so native arrival is a real P3
  requirement rather than a closed question
- the local-versus-whole-path turning split, which is what the P3 brief below
  was written against, is an artefact of the old correction operator

That leaves P3, and one prerequisite before it.

1. Re-derive the architecture brief. DONE, 2026-07-27, rows ...d0b3d7de and
   ...b2fe71f4, written up in "What the score actually is" below. The P1 half
   stands and is now stronger: the model must be natively endpoint-conditioned,
   because feedback channels bolted onto this trunk are ignored (the trunk
   plans the whole path from its static conditioning) and because landing
   within 15px is worth 0.036 to 0.050 that no post-hoc operator can recover.
   The P2 half stands but has been re-aimed: the model is NOT globally
   under-dispersed, so "dispersion calibration" as a blanket requirement is
   wrong. What does not stand at all is LOCAL dispersion calibration, which
   came from envelope_ceiling's wobble-versus-excursion split and is an
   artefact of the old correction operator.

   The oracle gate that this step called for has since been run, and it and its
   follow-up took the route-diversity half of the brief down with them. See
   "The gate, and what it took down" above.

2. Then P3 itself. What is left to build on after 2026-07-27 is one measured
   requirement and one open question.

   The requirement: endpoints exact by construction, not corrected afterwards.
   Generate the residual from the straight line rather than absolute positions,
   so both ends are right and cannot be missed. Worth 0.036 to 0.050 from
   w3_tax_bands_jog, deletes the arrival tax at the same time, and nothing
   today touched the evidence for it. Cheap and low risk.

   The open question: there is currently NO measured localisation of the
   remaining deficit. Every candidate this session tested was an instrument
   artefact, and the controlled reading is that the arm is diffusely separable
   everywhere rather than absent from any identifiable region. A session that
   wants a target has to find one that survives a random-subset control, and
   should assume the existing record does not contain one. Do not restart P1 or
   P2 on the strength of anything retracted above.

   Superseded in part, same day, by `w3_conditional_gate.py`. There IS now a
   measured localisation, it just is not a subpopulation of paths. Exactly human
   marginals plus exactly human pairwise dependence still reads 0.59 and 0.58
   against a 0.50 floor, but making that dependence conditional on movement
   duration takes it to 0.5592 and 0.5491, beating a matched-noise control 8
   times out of 8. That is about two thirds of the whole gap accounted for by
   marginals plus duration-conditional dependence.

   Two architecture proposals died on 2026-07-27 before that one landed, one
   because its family was already measured at 0.92 to 0.99 in EXPERIMENTS.md and
   one because the coupling gaps it was built to close are worth zero. Do not
   revive either. The duration-conditioning brief is the only one with a price
   measured before the build.

3. The one untried lead, and it is cheap. The covered quarter is real signal
   and nobody has ever looked at it, because every session so far has asked
   what the model MISSES. Profiling those paths is descriptive statistics, not
   an AUC against a subset, so the narrowing artefact does not reach it. If the
   quarter the model already matches turns out to be a recognisable kind of
   movement, that is the first honest handle on the problem in weeks. If it
   turns out to be a random-looking mixture, that is also worth knowing: it
   would say the model is uniformly a bit wrong and there is no subpopulation
   structure to exploit at all.

3. Optional and cheap, if a session wants more of the record cleaned: four of
   the eight raw-column-less probes have now been re-read. The remaining ones
   are stall_surgery (its hold-deletion score was measured on the additive arm
   only), gap_anatomy, critic_coverage and corpus_coverage. corpus_coverage's
   conclusion is about the corpus builder and does not depend on the arm, so it
   is safe. The other three are unaudited.

4. Anything needing cloud spend stops for L sign-off. Local GPU does not.

## The ground-up architecture, and where the score actually lives, 2026-07-27

L's direction on this day: stop tweaking inside the existing trunk's
assumptions, go to the adjacent fields that solved "generated sequences look
human until a classifier reads them", and design from first principles. Done,
and the answer is not the one the reading suggested.

### What the adjacent fields converged on

Handwriting synthesis, signature synthesis, speech synthesis and human motion
generation all hit this wall and all solved it the same way: stop generating
the signal, generate a compact PLAN and render it. For rapid human movement the
established plan is Plamondon's Kinematic Theory and its Sigma-Lognormal model,
where a movement is the vector sum of a few overlapping ballistic strokes, six
numbers each, validated on handwriting, signatures, saccades, wrist movement,
gait and stroke gestures. For aimed pointing specifically the motor control
literature says the same thing with minimum jerk strokes: a primary ballistic
submovement plus a small number of corrections, and the published mouse work
puts the count at 2 to 6 for most trials.

There is also directly comparable published work. The BeCAPTCHA-Mouse group
trained a GAN on human mouse trajectories and reported output that convinced
the eye while a random forest still separated it easily. That is our result, in
a paper, with the same classifier family. This is a documented open problem and
not a defect in this repo's engineering.

### The instrument, and why the plan hypothesis was testable for free

research/w4_submovement_ceiling.py fits N minimum jerk submovements to a real
path. Position is linear in the stroke amplitudes, so for any timing the
amplitudes solve in closed form and only 2N timing numbers need searching.
Variable projection, which is what makes 2000 paths a CPU job.

Rendering holds the observation layer fixed on purpose: the plan is evaluated at
the SAME irregular timestamps the human path was sampled at and put back on the
same integer lattice.

The sweep over stroke count, 2000 paths, floor 0.4889. Median fit error beside
each.

  N=1    4 numbers   10.54 px   0.9983
  N=2    8 numbers    3.96 px   0.9425
  N=3   12 numbers    2.30 px   0.9082
  N=4   16 numbers    1.54 px   0.8828
  N=6   24 numbers    0.98 px   0.8270
  N=8   32 numbers    0.73 px   0.7844
  N=12  48 numbers    0.47 px   0.7123

A smooth curve that passes within half a pixel of every human sample still
reads 0.712, and at N=12 the leftover has sd 0.34 px, which is essentially pure
rounding at 0.29. The AUC reaches the floor only by exact reconstruction, never
by compact approximation. Plan plus its OWN leftover is exactly the floor at
every N, so the render-and-round harness is not leaking.

That looked like a verdict on the parameterisation. It is not, and the control
that shows why is the most important measurement of the day.

### The contract is a texture instrument, not a shape instrument

research/w4_texture_sensitivity.py perturbs REAL human paths and nothing else.
2000 paths, internal AUC against the disjoint human half.

  real (floor)                0.4889
  real + 0.25 px noise        0.5731
  real + 0.5 px noise         0.8661
  real + 1.0 px noise         0.9503
  real + 2.0 px noise         0.9838
  real, 3-pt smoothed         0.5907
  real, 5-pt smoothed         0.5793

Half a pixel of noise on genuinely human data reads 0.866. A 3-point moving
average, which changes the actual shape of the movement by far more than a
pixel, costs 0.59. The contract is dominated by sample-scale texture and is
comparatively indifferent to movement shape.

Two consequences that were not previously on the record.

First, the plan's 0.827 at 0.98 px error is BETTER than real human data
perturbed by a comparable amount (0.950 at 1 px), so it was never evidence
against the parameterisation. The whole stroke-count curve above is a
sensitivity curve for the metric wearing a parameterisation's clothes.

Second, and this is the one worth carrying forward: it very likely explains why
every continuous whole-path family here landed at 0.92 to 0.99 (DDPM 0.9291,
CFM 0.9191, v136 0.9899) while the discrete event stream trunk is the only
survivor. Continuous models produce off-lattice positions that get rounded, and
that is precisely the 0.95 arm above. The trunk emits integer steps natively and
never holds a fractional position, so it never pays that penalty at all. The
"continuous architectures are closed" reading in EXPERIMENTS.md is therefore NOT
safe: those families were never tested with a lattice-native emitter, and the
number they scored may be mostly renderer. Nobody has re-tested them, and there
are no cached outputs to re-test cheaply, so this is a hypothesis with a strong
mechanism, not a result.

Error diffusion, carrying the rounding error forward instead of discarding it,
was tried and is a NO-OP here: 0.8668 rounded against 0.8668 diffused on the
noise arm, 0.8276 against 0.8276 on the plan arm. At this sampling rate the
per-sample steps are already several pixels, so there is nothing sub-pixel for
the accumulator to fix. A lattice-native emitter has to get the angular texture
right, not just the rounding.

### The texture cannot be added on top, which kills the naive two-layer design

Same script, part two, at N=6 on 1940 paths.

  plan only                   0.8276
  plan + own leftover         0.4889   exact reconstruction, pipeline control
  plan + own leftover REVERSED 0.8298
  plan + own leftover rolled  0.8466
  plan + another path's       0.8636
  plan + another, rescaled    0.8402
  plan + speed-scaled white   0.8305

Reversing a path's OWN leftover keeps its magnitude, its spectrum and its lag
structure and destroys only the alignment, and it costs the entire distance. The
leftover is not a stationary process. Any design that renders a smooth
continuous plan and adds a noise model on top cannot work, however good the
plan and however well matched the noise.

### And the plan layer is not where the deficit is

research/w4_plan_space.py fits plans to the trunk's own cached output and to
real paths, throws the paths away, and scores the PLANS. 16 plan descriptors,
contract estimator and seed, human ref plans as the reference.

  floor, human plans against human plans   0.4815
  fc_v2 plans                              0.5595   excess +0.0780
  resid_v2 plans                           0.5505   excess +0.0689

The same checkpoints' PATHS read 0.6539 and 0.6467. So the trunk's motor plans
are close to human while its paths are not. The commitment hypothesis, that the
trunk never decides on a whole movement and so its whole-path properties emerge
as an average, is measurable and comes out small.

Confirmed from the other side: plan space is BLIND to what dominates the
contract. The real-plus-half-a-pixel arm reads 0.867 on motion features and
0.406 against a 0.5016 floor in plan space, which is below the floor.

The channel is named. On real against real-plus-noise the forest puts 0.187 on
angular_velocity_mean and 0.156 on angular_velocity_std, a third of the total in
two turn-rate features. On the trunk the importances are FLAT at 0.05 to 0.07
across all eighteen, which is yesterday's diffuse deficit again. Rank agreement
between the noise channel and the trunk channel is +0.480 (fc_v2) and +0.412
(resid_v2). So the trunk's problem is related to the texture channel but is not
simply too much or too little noise.

### The hand-built emitter fails, and its null arm says why

Before pricing a build on the renderer lead, the cheap half was worth testing:
if rounding is what convicts a smooth curve, then CHOOSING the whole-pixel steps
instead of rounding each sample independently should rescue it.

The channel diagnostic said what to choose for. Measured on 1200 paths:

  arm                 zero-turn  straight  |turn| med   dwell  step med
  real                    0.359      2.42      7.12 d   0.102     2.24
  plan N=12, rounded      0.360      2.41      5.37 d   0.119     2.24
  real + 0.5 px           0.214      0.85     11.96 d   0.049     2.24
  fc_v2                   0.318      2.03      7.86 d   0.152     2.00
  resid_v2                0.334      2.28      7.21 d   0.135     2.00

zero-turn is the share of resampled samples that turn EXACTLY zero, which the
125 Hz resampler creates by putting several collinear points on each original
whole-pixel segment. straight is the mean length of the exactly-straight run a
sample sits in. So the rounded plan already matches real on collinearity, on
straight-run length, on dwell and on step size. Its one visible defect is turn
magnitude: 5.37 degrees where a human turns 7.12. Adding half a pixel of noise
breaks all of them at once, which is why noise is expensive.

research/w4_lattice_emitter.py (row ...842e1356) targets that one defect. At
each sample it carries the accumulated tracking error, forms the candidate
integer steps around the error-corrected ideal step, draws a target turn from
the human distribution conditioned on step length, and takes the candidate that
best trades turning against tracking. Error feedback repays any step taken to
make a turn, so the path still lands where the plan says. 969 plans at N=12,
0.51 px median fit error, human turn table from the fit half, scored against the
disjoint holdout half.

  real (floor)                      0.4698
  rounded (base)                    0.6807
  emitted, human turns w=0.1        0.7480
  emitted, human turns w=0.25       0.7376
  emitted, human turns w=0.5        0.7408
  emitted, human turns w=1.0        0.8088
  emitted, human turns w=3.0        0.8435
  emitted, own turns w=1.0          0.7958
  emitted, w=0                      0.6807   control, equals base exactly

Every weight is worse than the base and monotonically worse with weight. The
w=0 control reproduces the base to the digit, so the candidate machinery is a
true no-op and all the damage is the turn targeting.

The null arm is the finding. `own turns` draws each target from the ROUNDED
PLAN's OWN turn distribution, so the marginal is preserved exactly and only the
sequence is randomised. That alone costs +0.115. Turn structure is SEQUENTIAL,
not marginal. No emitter that matches per-step turn statistics can work, however
well calibrated, because the statistic it matches is not the thing being read.

### What this does and does not license

Dead: the ground-up plan-plus-renderer architecture as proposed. The plan layer
is already close to human, so modelling plans better cannot move a number that
does not live there. Cost was one CPU afternoon and no GPU time, which is what
the gate was for.

Also dead: any two-layer design that adds a noise process to a smooth path, and
any hand-built statistical emitter that renders one. That is three failures in
one day of the same idea, supply the missing local texture to a smooth path,
tried three ways: continuous noise added, a real leftover transplanted, and
integer steps chosen to match turn statistics. All three moved the score the
wrong way. It is the same shape as yesterday's partial coupling repair. Partial
texture repair is actively detectable.

Weakened, not disproved: the continuous whole-path families may have been
convicted on their renderer rather than their architecture. The negative half
still holds, rounding really does cost that much. But the positive half now has
one failed attempt against it, and the only emitter left that could pass is a
LEARNED autoregressive model over integer steps, which is exactly what the
surviving event-stream trunk already is. So the lead no longer points at a
rewrite. It points back at the trunk.

That is the honest close of the architecture question. The trunk is right at the
plan level, right on every step-level marginal that was measured, and still
reads 0.70.

## The metric is fully redundant, and what that means, 2026-07-27

This is the most consequential result in the file and it explains every failed
repair in it, so read it before proposing anything.

### The problem is obedience, not design

The trunk is FEATURE CONDITIONED. Before each path it is handed an 18-dim
character vector drawn from a KDE over a bank of 8192 real human feature
vectors (experiments/event_stream_polar.py), and asked to produce a path with
that character. The contract measures those same 18 numbers. So if the model
obeyed its command exactly, the score would be 0.50 BY CONSTRUCTION: human
feature vectors in, paths carrying those feature vectors out.

research/cond_realization_probe.py measured the obedience on 2026-07-26.
Correlation between commanded and realized, per feature:

  mean_jerk                  -0.005      angular_velocity_mean      0.512
  mean_acceleration           0.167      mean_velocity              0.591
  time_to_peak_velocity       0.168      std_jerk                   0.617
  movement_duration           0.172      std_acceleration           0.651
  velocity_skewness           0.213      max_deviation              0.666
  num_direction_changes       0.224      max_acceleration           0.768
  curvature_std               0.228      std_velocity               0.787
  path_efficiency             0.246      max_velocity               0.816
  curvature_mean              0.306
  angular_velocity_std        0.334      MEAN 0.41

The model listens on speed and ignores the shape family. That single number,
0.41, is a complete account of the 0.70.

Two inference-time repairs of it are already closed, and both failed for the
same reason: you cannot make a model obey at sampling time a command it never
learned to obey at training time.

  affine pre-distortion (cond_realization_probe, lam 0.5 and 1.0) inflates
  realized sd to 1.3 and then 2.3 and scores 0.719 and 0.812 against a 0.647
  base. Aiming off-center multiplies the noise, not the signal, when r < 1.

  guidance (w3_guidance_capacity, row ...57837200) goes 0.6546, 0.6904, 0.8049
  over w=0, 2, 4 on the pure arm. It has ample authority over the velocity
  channels, inflating max_velocity spread to 6.2x human, and cannot reach
  curvature at all.

Note the serving default: EVENT_CFG_W is 0 in the locked recipe, so the shipped
number is the unguided one, and the sweep says that is the right choice.

### Curvature variety is genuinely absent, everything else is present

w3_guidance_capacity concluded "curvature variety is absent from the model, not
merely unexpressed" from a steering probe, which cannot separate absent from
unsteerable. research/w4_variety_vs_steering.py (row ...1252df17) measures the
marginals directly against real humans. Spread as a fraction of the human sd:

                        fc_v2   resid_v2
  curvature_mean         0.10       0.05
  curvature_std          0.14       0.09
  path_efficiency        0.80       0.76
  angular_velocity_std   0.98       0.99
  angular_velocity_mean  1.03       1.01
  max_deviation          1.02       1.12
  num_direction_changes  0.97       0.97

Human ref against human holdout puts measurement noise on this family at 0.085,
so the curvature rows are a real order of magnitude and the rest are not. The
model has ordinary human variety in speed, timing, direction changes and
deviation, and essentially ONE curvature. Guidance could not steer curvature
because there is nothing there to steer.

This is a DIFFERENT defect from low adherence and needs a different fix.
angular_velocity_mean has full spread and only 0.51 adherence, which is a
steering failure. curvature has 0.31 adherence and no spread, which is a
capability failure. Do not treat them as one problem.

### No per-feature repair can work here, and this is now measured

Single-feature contract-estimator AUC, balanced at n=2000, floor in brackets:

  movement_duration      0.732 (0.514)     <- larger than the full 18 at 0.660
  time_to_peak_velocity  0.567 (0.486)
  max_velocity           0.564 (0.513)
  curvature_std          0.544 (0.495)
  max_acceleration       0.534 (0.474)
  curvature_mean         0.530 (0.502)
  everything else        within 0.03 of its own floor

The duration carrier has a clean cause, and it is the temporal analogue of the
whole-pixel spatial lattice this repo already understands. Human timestamps sit
on a 1 ms grid, so human durations do too: 903 distinct values in 2000 samples.
The model's are continuous, 2000 distinct in 2000. A tree splits between grid
points and every sample in the gaps is synthetic. The coarse shapes agree; the
density ratio by decile of the human duration axis runs 0.87 to 1.16 across the
whole range, so mean, sd and every percentile match and ONLY the grid differs.

Snapping the model's timestamps to the same 1 ms grid repairs the column:

  movement_duration      0.732 -> 0.567     removes 0.165 of excess
  time_to_peak_velocity  0.567 -> 0.521     removes 0.046

Those are the two largest carriers in the panel. The joint score does not move.

  all 18, internal       0.6602 -> 0.6490
  contract scorer        0.6624 -> 0.6678   WORSE

The detector reroutes through the other seventeen columns. This is the measured
form of what the flat importance tables have implied since w3_joint_structure:
the 18 features are 18 correlated views of ONE global mismatch, not 18
independent tests. Removing the single largest signal buys nothing because the
same fact is legible everywhere else.

### What this closes and what it opens

Closed, structurally and for good: every per-feature and per-mechanism repair.
Not because each was a bad idea, but because the metric cannot be moved that
way. This retroactively explains the whole failure record. Coupling repair worth
nothing, guidance backfiring, arrival correction small, correction geometry
nothing, texture transplant negative, lattice emitter negative, affine
conditioning correction negative. They are all the same result.

Anyone proposing a targeted fix to a named feature should be shown this section
first. The prior is now strongly against it and the burden is to explain why
their repair is not the eighth instance of the same mistake.

Open, and this is the whole remaining program: make the generative distribution
itself closer. Concretely, and in the order the evidence supports:

1. Wire the conditioning in properly. The command currently enters weakly.
   Modulating every layer with it (FiLM or adaptive layernorm, the change that
   made image diffusion controllable) is the standard fix for a model that
   ignores its condition, and it has never been tried here.
2. Drop cond_dropout. It is 0.1, which is the classifier-free guidance training
   recipe, and it teaches the model to work without the command. It only pays
   off if guidance is used at serving, and the sweep says guidance must stay at
   0. So the repo is paying the cost of the technique and taking none of the
   benefit.
3. Train on whether it obeyed. Roll out, extract the 18 features, penalise the
   gap to the command. This is ordinary supervision against a known target, not
   the adversarial critic and not the GRPO reward, both of which are closed.
4. Give curvature something to vary. Nothing in the current emission decides how
   much a path bends, which is consistent with the 0.10 spread.
5. Train longer. The checkpoint stopped at step 12000, epoch 11. Adherence is
   exactly the sort of capability that arrives late.

Data is the ceiling behind all of it: 50000 paths, 3.2M samples, in
data/demo_pool.npz. Worth establishing whether more recordings can be obtained
before assuming 5 is enough.

Caveat on the instrument, stated plainly. Plan space is 16 hand-built
descriptors. It is not blind, since it separates the trunk at +0.07 over a floor
it sits exactly on, but its resolution is unknown and it cannot prove there is
no plan-level difference it fails to see. amp_over_net also runs a 9.4 SD gap
between the two human halves, which is a heavy tail from dividing by a near-zero
net displacement, so that column is unreliable and should be dropped or clipped
if this is rerun.

## Hard rules (verbatim from the standing mandate)

- NEVER modify training/candi_polar_flow_best.pt
  (MD5 91326a29750789f3167055324ef377c5). Verify with md5sum after every run.
- NEVER touch data/human_eval_features.npy.
- NEVER modify scoring code.
- git add files individually, never `git add .`.
- Repo is public: docs must read human-written, no em or en dashes.
- L is non-technical: report what and why in plain terms, brief summary
  first, no praise, max 25 words between tool calls.
- Thermal: launch gate 75C, watchdog kill 83C, cooldown only if peak >= 79C.
- Log every run as a row in research/autoloop/ledger.jsonl via
  research/autoloop/ledger.py append_row (status must be ok/failed/killed).

## Environment

- Repo: /mnt/c/Users/aaron/Code/mouse-trajectory-synthesis
- Python: ~/venvs/mime/bin/python, always with env PYTHONPATH=.
- Check for stray GPU processes BEFORE launching, both sides. Linux:
  nvidia-smi. Windows: powershell.exe Get-Process (WSL cannot see Windows
  processes any other way).
- Long runs: nohup + research/gpu_watchdog.py --pid <PID> --log <path>
  --threshold 83 --interval 60 --max-minutes 90. The --log flag is REQUIRED.
- Background with `;` separators, not `&&`. Backgrounding a `cd X && cmd`
  chain backgrounds the whole chain and leaves the parent shell in the wrong
  directory, which silently breaks redirects.
- Generation is slow: 2000 one-shot paths take ~147s, 6000 take ~442s. Python
  buffers stdout to a log file, so an empty log does not mean a dead process.
  research/w3_landing_cache.pkl (13M, untracked) holds the 6000 generated
  paths from the arrival-tax run and skips regeneration.

## Evaluation, read before running anything

Do NOT use evaluate.py for any decision. It loads
data/human_eval_features.npy, the final untouched eval sample.
research/autoloop/scoring.py is the metric contract, defaults to
data/human_val_features_grpo.npy, and raises on any path containing
"human_eval". Several older handoffs tell you to run evaluate.py anyway; that
instruction is wrong and would spend the held-out sample permanently.

For event-stream checkpoints use research/w1_oneshot_score.py, which wires
specs, generation and the contract scorer together:

    env PYTHONPATH=. ~/venvs/mime/bin/python research/w1_oneshot_score.py \
      --ckpt event_polar_4m_fc_v2.pt --n 2000 --seed 42

That reads 0.6544, matching README.md's documented ~0.65, and is the sanity
anchor. Confirm it before trusting any new model's number.

The locked decode recipe includes EVENT_CHOICE_TEMP=10. Omitting it inflates
one-shot AUC badly (0.9387 instead of 0.6544). w1_oneshot_score.py defaults it
to 10. The 0.596 to 0.60 figures in EXPERIMENTS.md are WITH selection
(EVENT_SIR=8), not one-shot.

## Uncommitted state

Branch w3-p3-representation-ceiling carries the work. The probe scripts, their
result JSON, EXPERIMENTS.md, METHODOLOGY.md, README.md and this file are
committed and pushed as of 2026-07-26.

Still untracked and deliberately so: the path caches (research/*.pkl, large
binaries) and the pid and log scratch files.

RESOLVED 2026-07-27: research/autoloop/ is tracked. scoring.py, ledger.py,
ledger.jsonl, LEADERBOARD.md, loop.py, runner.py and backfill.py are all in
git ls-files. Earlier handoffs flagged it as untracked and unbacked-up and
raised it with L four times; that is no longer the case and the worry can be
dropped.

Note: ~120 tracked files show as modified in git status. That is a line-ending
artifact of the Windows mount, not real edits. Do not sweep them into a commit.

## Open questions for L

- Cloud budget ceiling for W3, if a training run eventually needs one.
- Whether to package the fallback as a usable product now, in parallel with
  research. ANSWERED 2026-07-21 (row W3_groundwork_...e7f67c96): with exact
  arrival enforced and selection run on the corrected candidates it reads
  about 0.58 at roughly 1s/request, so it is packageable. RE-READ 2026-07-27
  under correct_jog (row ...cb65f60c): unchanged at 0.5909 against 0.5833,
  inside seed noise. The number is honest under either operator. The one
  condition is that the judge and the server must use the SAME operator, since
  winners picked under one read 0.096 worse when served under the other.

## The model is graded in a different space from the one it is commanded in, 2026-07-27

Run `w4_pipeline_agreement_2026-07-27T211030+0000_e68df74a`, script
`research/w4_pipeline_agreement.py`. CPU only, no generation.

There are two implementations of the 18 features. `features.py` resamples to
125Hz and extracts, and it is what the contract scores. `detector_features` in
`training/train_events_polar_dm.py` is a differentiable reimplementation on
padded frames, and it is what built the checkpoint's feature bank and what
computes the character command during training. Its docstring argues the
approximation bias cancels because generated and real batches both go through
it. That argument is correct for the MMD term it was written for. It does not
hold for conditioning, where the command comes from one implementation and the
grade comes from the other, so any systematic disagreement is a standing error
between what the model is told to do and what it is measured on.

Method: one set of 1500 real human validation paths, resampled once, then laid
out both ways. Same numbers, two containers, so nothing but the extractor
differs. Then column by column agreement.

Pearson agreement averages 0.782 but that is the wrong statistic for three of
these columns. Rank agreement averages 0.923. Reading both together:

```
  feature                  pearson  spearman
  path_efficiency            1.000     1.000
  max_deviation              1.000     1.000
  movement_duration          1.000     1.000
  mean_velocity              0.979     0.996
  std_velocity               0.965     0.994
  max_velocity               0.960     0.994
  std_jerk                   0.949     0.988
  std_acceleration           0.947     0.989
  max_acceleration           0.947     0.989
  velocity_skewness          0.968     0.987
  num_direction_changes      0.900     0.934
  mean_acceleration          0.036     0.921
  curvature_std              0.624     0.882
  curvature_mean             0.575     0.873
  mean_jerk                  0.031     0.856
  angular_velocity_mean      0.875     0.833
  angular_velocity_std       0.827     0.744
  time_to_peak_velocity      0.494     0.633
```

Three findings, in descending confidence.

**One real bug: time_to_peak_velocity.** `detector_features` computes it as a
softmax weighted average of `torch.arange(n) / speed.shape[1]`, where
`speed.shape[1]` is the PADDED width of the batch, not the number of valid
frames in that particular path. A path occupying 200 of 539 padded frames
therefore has its peak time divided by 539. The mean command is 0.041 where the
contract reads 0.355, and rank agreement is 0.633. Recomputing with a hard
argmax normalized by each row's own valid length gives mean 0.341 against the
contract's 0.355 and rank agreement 0.978. The trunk has been commanded a
quantity that is about eight times too small and only loosely ordered like the
one it is scored on. Measured adherence on this feature is 0.168, and it cannot
be much higher than the bridge allows.

**One deliberate tradeoff that costs curvature variety.** The curvature
denominator floors speed at 30 px/s (`speed_mid = speed[:, :-1].clamp(min=30.0)`)
with the comment that 1/speed^3 gradients on sub-pixel frames were the dominant
source of gradient noise and that v2 diverged without it. `features.py` floors
at 1e-6. Curvature in this contract is dominated by slow frames, which is
exactly the regime the floor removes. The effect is not a clip of rare
outliers: only 0.16 percent of samples exceed the detector's 1e4 clamp, so the
clamp is innocent and the floor is the cause. Command spread is 1.483 against
the contract's 3.329. Removing the floor lifts rank agreement from 0.873 to
0.935 and spread from 1.483 to 2.368, still short of 3.329. This lines up with
the independently measured curvature collapse in
`w4_variety_vs_steering_2026-07-27T185816+0000_1252df17`, where model curvature
spread was 0.10 to 0.14 of human against a measurement floor of 0.085. The
model may be producing the curvature variety its training signal asked for,
in a signal that cannot represent the variety the contract measures.

**Two false alarms, recorded so nobody rediscovers them.** `mean_acceleration`
and `mean_jerk` show Pearson 0.036 and 0.031, which looks catastrophic and is
not. Both are signed telescoping statistics with extremely heavy tails, and
rank agreement is 0.921 and 0.856. Pearson is measuring the tails, not the
bridge. An earlier reading of this session called these broken; they are not.

Bridge quality caps adherence but does not determine it. Every column with rank
agreement below 0.9 has adherence at or below 0.31. But `movement_duration`,
`path_efficiency` and `velocity_skewness` all bridge at 0.99 or better and
still sit at 0.172, 0.246 and 0.213, so a clean bridge is necessary and not
sufficient. Fixing the bridge is not a substitute for the obedience work, it is
a precondition for two of the eighteen columns.

`w4_command_ceiling` is RETIRED by this, ledgered as failed under
`...617afd57`. Its headline of 0.9708 for the noise-free bank was impossible on
its face, since the bank is real human data. It was measuring this mismatch.
The question it asked, what perfect obedience would actually score, is still
open and still worth answering, but it has to be answered inside one pipeline.

What this does not change: the fc_v3 run training now addresses obedience
capacity, not the space the commands live in. Its time_to_peak_velocity and
curvature columns stay capped by the bridge whatever it scores. Repairing the
bridge means recomputing the bank and retraining, so it is a separate run.

## The conditioning programme was built on a false premise, 2026-07-27

Runs `w4_command_ceiling_v2_2026-07-27T212050+0000_88d91a81` and
`w4_variety_mechanisms_2026-07-27T212445+0000_f14e31ab`. CPU only.

The argument that justified every hour spent on obedience was this: the trunk is
handed an 18-dim character command drawn from real human feature vectors, so if
it realized that command exactly its output distribution would BE the human
distribution and the score would be 0.50 by construction. Obedience was
therefore the only thing left to fix.

The argument has an unexamined step, and the step is false. Serving does not
command a real human feature vector. `experiments/event_stream_polar.py` line 91
draws a kernel density sample:

```
  _FEAT_BW = float(os.environ.get("EVENT_FEAT_BW", "0.25"))
  feat = bank[row] + _FEAT_BW * randn(18)
```

Measured, in the checkpoint's own log-scaled z-space, with the round trip
verified exact (raw bank 0.5143, round-tripped bank 0.5144):

```
  arm                                 contract AUC
  bank rows, no noise                       0.5144
  bank + BW 0.05                            0.9970
  bank + BW 0.1                             0.9991
  bank + BW 0.25                            0.9997   <- serving default
  bank + BW 0.5                             0.9998
  bank + BW 0.25, columns shuffled          0.9999
```

Perfect obedience to the command serving actually issues scores 0.9997, not
0.50. A fifth of the bandwidth still scores 0.9970. The premise is not
approximately right, it is inverted.

**Why.** Real human 18-feature vectors are not free-floating. They are 18
measurements of one path and they are locked together. The PCA spectrum in log
space, standardized:

```
  PC1  6.363   PC5  1.105   PC9  0.199   PC13 0.026
  PC2  4.074   PC6  0.954   PC10 0.136   PC14 0.019
  PC3  2.031   PC7  0.740   PC11 0.104   PC15 0.006
  PC4  1.635   PC8  0.548   PC12 0.058   PC16 0.004
                                         PC17 0.004
                                         PC18 0.00005
```

Seven of eighteen components sit below one percent of the largest, 95 percent of
the variance is in the first eight, and the smallest over largest ratio is
8.09e-06. PC18 is an exact constraint. Independent noise of 0.25 sd is four
orders of magnitude larger than the real spread along that direction, so it
leaves the manifold instantly and visibly.

It is not only the thin directions. Noise shaped with the real covariance scores
0.9996 and noise confined to the top six components scores 0.9958, so the sheet
is curved as well as thin. This is the same geometry `w4_texture_sensitivity`
ran into when real paths plus half a pixel of noise read 0.866, and the same
geometry behind the redundancy result earlier today: the 18 features are 18
correlated views of one object, and that object has about eight degrees of
freedom.

**What this means for the standing number.** The model's disobedience has been
protective. Adherence sits at 0.41, and the 59 percent it ignores is what keeps
its output on the path manifold instead of tracking an off-manifold command.
Raising obedience while the bandwidth stays at 0.25 moves the model toward a
target that scores 1.0. PREDICTION RECORDED BEFORE THE fc_v3 RUN FINISHED: the
FiLM run now training may score WORSE than its parent, and if it does that is
the theory working, not the run failing.

**The fix is a serving-side environment variable.** `EVENT_FEAT_BW=0` commands
verbatim bank rows on any existing checkpoint, no retraining. Ceiling 0.5144
instead of 1.0.

**Variety cannot be bought back synthetically.** If the bandwidth goes to zero
the obvious worry is that every path carries one of a fixed set of fingerprints.
Convex interpolation between real bank rows is the standard manifold-safe
substitute for Gaussian jitter, and it was tested:

```
  bank rows verbatim                                  0.5144
  interpolate toward 1 of 8 nearest, half range       0.6717
  interpolate toward 1 of 3 nearest, half range       0.6680
  interpolate toward 1 of 1 nearest, half range       0.6782
  interpolate toward 1 of 8 nearest, full range       0.7288
  interpolate toward a RANDOM other row               0.7693
  independent noise 0.05 sd                           0.9969
  independent noise 0.25 sd                           0.9997
```

Interpolation beats jitter by a mile and still loses to verbatim by 0.15. The
telling number is that interpolating toward a nearest neighbour barely beats
interpolating toward a random row, 0.7288 against 0.7693. If the manifold were
locally flat those would be far apart. They are not, so it is curved on the
scale of the nearest-neighbour distance and local linear mixing leaves it too.

So variety has to come from bank size, not from synthesis. The checkpoint bank
holds 8192 rows and scoring uses 2000 paths, so at research scale verbatim
sampling costs no variety at all. At production scale it would, but the mandate
is not shipping.

**What is now open.** With the bandwidth at zero the ceiling is 0.5144 and
obedience becomes worth exactly what the original argument claimed. The two
programmes compose: fix the bandwidth so obedience is pointed at a reachable
target, fix the time_to_peak_velocity bridge so two of the columns are even
commandable, then raise adherence. None of those three had been true at once
before today.

## The model is narrow, not noisy, and the gap has a name, 2026-07-27

Run `w4_manifold_projection_2026-07-27T213226+0000_9fad4e09`. CPU only.

`w4_command_ceiling_v2` established that the human feature manifold is thin, and
the obvious next hypothesis was that the model fails by sitting off it, the way
the KDE bandwidth does. That hypothesis is REFUTED. Projecting model output onto
the human manifold's own principal basis and comparing variance per component:

```
  component  loads on                                human   model   ratio
  PC1        max_velocity, max_acceleration,         6.153   5.404    0.88
             std_velocity, std_acceleration
  PC2        num_direction_changes, curvature_std,   4.246   2.414    0.57
             movement_duration, curvature_mean
  PC3        mean_acceleration +0.655,               1.964   0.237    0.12
             mean_jerk +0.654
  PC4        angular_velocity_mean, duration         1.659   1.252    0.76
  PC5        curvature vs angular_velocity contrast  1.088   0.594    0.55
  PC6        ...                                     1.017   0.915    0.90
  PC12..PC17 the thin directions                                 0.69 to 1.32
  PC18       the exact constraint                    0.0003  0.00001  0.03
```

The thin directions are fine. Ratios there run 0.69 to 1.32, at or below human,
and PC18 is a third of a percent of an already negligible variance. The model
does not manufacture impossible combinations. It manufactures a subset of
possible ones.

The deficit is on the FAT directions, and PC3 is the headline at 0.12. PC3 is
almost purely `mean_acceleration` and `mean_jerk`, the two signed telescoping
statistics, which reduce to `(v_last - v_first) / T` and its jerk analogue.
Physically that is the net speed envelope of the movement: whether it is
accelerating or decelerating taken as a whole. Humans vary enormously in this,
some launching fast and coasting in and some ramping the whole way. The model
has one envelope and reuses it, covering 12 percent of the human range.

Two independent checks say this is structural and not an artifact. First, PC3's
two features are also the two worst adherence features in the whole table,
`mean_jerk` at -0.005 and `mean_acceleration` at 0.167, so the variance collapse
and the obedience failure are one fact seen twice. Second, PC1 at 0.88 loads on
exactly the features that DO obey, `max_velocity` 0.816, `std_velocity` 0.787,
`max_acceleration` 0.768. The ordering matches end to end.

This unifies three separate results from today. The redundancy finding said the
18 features are 18 views of one object. The PCA says that object has about eight
degrees of freedom. The adherence table says which of the eight the model holds.
One nearly right, three at roughly half, one almost absent.

It also revises the prediction recorded before fc_v3 finished. Two mechanisms
are now live and they point opposite ways. If the model's problem is chasing an
off-manifold command, more obedience hurts. If the problem is narrowness, more
obedience widens the output and helps. The fc_v3 number discriminates.

CAVEAT, do not skip: the synthetic arrays used here are
`research/phase1_score_phase1_features.npy` and
`research/w3_tail_ceiling_features.npy`, dated Jul 19 and Jul 20 and scoring
0.7662 and 0.7560. They are not current fc_v2 output. The two agree with each
other to two decimal places on every component, which is reassuring but is not
the same as being current. Re-run on live fc_v2 output before building on this.

CORRECTION, same day: an earlier draft of this paragraph called fc_v2 "the
current fc_v2 at 0.6986". That conflates two different things and the number
does not belong to fc_v2. 0.6986 is `event_polar_4m_resid_v2`, with exact pixel
arrival enforced on every served path, under the Section 7.12 scorer that
METHODOLOGY.md line 1667 explicitly labels "different scorer". fc_v2 under
`research/w1_oneshot_score.py` reads 0.6541, 0.6544, 0.6496 and 0.6327 across
replicates. Those two numbers are not comparable without establishing the
mapping between the two measurement paths first, and that has not been done.

The caveat is now CLOSED by `fc_v3_feat_film_train_2026-07-27T215550+0000_d567babb`,
which measured coverage on live output from both checkpoints. PC3 reads 0.12 on
live fc_v2 and 0.12 on live fc_v3, matching the stale arrays exactly. The other
components moved by up to 0.12 between stale and live, so PC3 is the component
that held, and it is the one the argument rests on.

RETRACTED, same day, by `w4_robust_coverage_2026-07-27T233940+0000_2c119db8`.
The 0.12 is real arithmetic and it is not a real gap. Every ratio in the table
above is a ratio of standard deviations, and PC3 loads almost entirely on
`mean_acceleration` and `mean_jerk`, whose human standard deviations are 3.0e6
and 4.0e8. Those numbers are set by roughly 0.13 percent of paths. Four
trajectories in three thousand decide what PC3's human variance is, so the
statistic describes those four and nothing else.

Redone with robust spread, half the 16 to 84 interpercentile range, which the
tails cannot move:

```
  component   sd ratio (the old table)   robust ratio
  PC1                          0.88          0.95
  PC2                          0.57          0.94
  PC3                          0.12          0.92
  PC4                          0.76          1.03
  PC5                          0.55          0.87
  PC6 to PC10                  varies        0.82 to 1.00
```

Nine of the eighteen features have standard deviation ratios at or below 0.01
with robust ratios near 1.0. The whole column was measuring tail behaviour.

Two things in the section above survive and one other does not. The claim that
the model does not manufacture impossible combinations still holds. The claim
that "curvature variety is genuinely ABSENT" at 0.10 to 0.14 is downgraded, not
retracted: robust ratios are 0.54 for `curvature_mean` and 0.43 for
`curvature_std`, so the spread is about half, not absent. Curvature is the only
marginal with a real gap left. The section's central argument, that the model is
narrow on the fat directions, does not survive. It is not measurably narrow
anywhere except curvature.

A mechanism for PC3 was also tested and refuted. The hypothesis was that human
timestamps sit on a coarse grid and the model's do not. Adding sub-millisecond
jitter to the human paths left PC3 variance at 1.97 against 1.98.

## The gap is coupling, not marginals, and it is higher order than pairwise, 2026-07-27

Run `w4_marginal_vs_coupling_2026-07-27T234322+0000_667de494`. CPU only.

With coverage no longer explaining anything, the question becomes how much of
the score is reachable by fixing the eighteen distributions one at a time. This
answers it exactly, by rank-mapping each model column onto the human marginal.
That makes every marginal identical by construction while leaving the copula,
the way the columns move together, untouched.

```
  floor, real vs real                                0.4973
  model as generated                                 0.6573
  model, marginals forced EXACTLY human              0.6322
  real human, all coupling destroyed                 0.9998
```

Perfect marginals buy 0.025 of a 0.160 gap. That is 16 percent. The other 84
percent is coupling. The last row is the control and it is the important one:
human data with correct marginals and shuffled coupling scores 0.9998, so the
detector reads coupling almost exclusively.

No single direction carries it either.

```
  all 18 components   0.6594     (floor 0.4894)
  PC1 to PC4          0.5494     (floor 0.5186)
  PC5 to PC8          0.5677     (floor 0.5014)
  PC9 to PC13         0.5603     (floor 0.5008)
  PC14 to PC18        0.5339     (floor 0.4909)
  best single PC (5)  0.5556
```

Every subspace is weakly detectable and none is close to the full 0.6594. The
signal is distributed.

And it is not pairwise. Mean absolute rank correlation is 0.301 for human and
0.301 for model, with mean absolute difference 0.039. The model already has the
pairwise structure. So a training signal that matches means, variances and
correlations cannot reach this, and that rules out a large family of fixes
before any of them is attempted.

## The one term that could reach it has been switched off the whole time, 2026-07-27

Runs `w4_mmd_blindness_2026-07-27T235022+0000_8944a419`,
`w4_mmd_queue_2026-07-27T235022+0000_52b8b0a7` and
`w4_mmd_alignment_2026-07-27T235022+0000_3fab6935`. CPU only.

`training/train_events_polar_dm.py` already has a term for exactly the quantity
the previous section localized. `match_loss` carries a kernel MMD alongside a
quantile term and a covariance term. The quantile term matches marginals, the 16
percent. The covariance term matches pairwise structure, which is already right.
The MMD is the only one that can see higher order joint structure, and its own
docstring concedes it is "nearly blind at feasible batch sizes".

It is blinder than that. `--batch-size` defaults to 64 and the loop halves it,
so the MMD is computed on 32 rows per side. Measured over 200 draws in the
detector space it works in, it ranks a model batch above a human batch 63
percent of the time. A coin flip is 50 percent. Cohen's d is 0.41.

The cleanest way to see it is to walk a mixture path from the model
distribution to the human one, replacing model rows with human rows a fraction
at a time:

```
  fraction human    current term    queued term
        0%              0.3036         0.0623
       20%              0.3012         0.0417
       40%              0.3019         0.0319
       60%              0.3038         0.0240
       80%              0.2996         0.0154
      100%              0.2984         0.0091
```

The current term is flat. It reads 0.3036 on the model and 0.2984 on real human
data, a 1.7 percent change across the entire distance the training is supposed
to close. It has no descent direction to give. Two independent gradient draws on
the same rows agree at cosine 0.543, so roughly half of what it does contribute
is noise.

A hypothesis worth recording as dead: the bandwidths are NOT the problem. The
tuple is (0.25, 0.5, 1.0, 2.0, 4.0) used as sigma squared, and the median
pairwise squared distance in this space is 24.4, so the largest kernel value at
a typical distance is 4.7e-02. That looks fatal and is not. Median heuristic
bandwidths, 6.1 through 97.7, measure 64 percent separability against 63.
Sample count is the entire effect. At 512 rows per side both bandwidth sets
reach 100 percent.

512 generated trajectories per step is not affordable. It is also not necessary.
Only the fresh rows need gradients, so the partners can come from a ring buffer
of recently generated rows at no generation cost:

```
  1. current, 32 generated vs 32 human            56%     d 0.23
  2. 32 generated vs 2048 human                   72%     d 0.79
  3. 32 generated plus 992 queued vs 1024 human  100%     d 7.77
  4. 512 generated vs 512 human, brute force     100%     d 4.56
```

Arm 3 beats an actual sixteen times larger batch while generating the same 32
trajectories. Enlarging only the human side, arm 2, is not enough. Gradient
agreement between independent draws goes from 0.543 to 0.843.

Two details make the estimator correct rather than merely larger, both now in
`mmd_rbf_queued`. Averaging the ordinary pool by pool statistic would spread
each fresh row's gradient across a matrix that is about 97 percent constant, so
it averages over fresh rows instead. And the generated block carries gradient
through one argument where the symmetric form carries it through two, so that
block takes a factor 2. Gradient norm on the fresh rows lands at 0.0104 against
0.0285 for the original, hence `--w-mmd 2.7` to hold the balance against the
anchor term fixed.

Finally, the term is worth sharpening because it points at the right target.
Destroying the copula in graded steps with marginals held exact moves the MMD
and the contract AUC together, monotonically at every level, Pearson 0.877:

```
  coupling destroyed    MMD      RF-OOB AUC
        0%            9.43e-03     0.4944
       10%            1.48e-02     0.6726
       25%            3.07e-02     0.9053
       50%            5.91e-02     0.9908
      100%            8.45e-02     0.9994
```

So this is a plumbing defect in the loss, not a missing architecture, and it sits
directly on the 84 percent the previous section localized. Whether fixing it
moves the score is a separate question and is being measured, two arms from the
same fc_v3 start, 1500 steps each, differing only in `--mmd-queue`.

One caveat to carry forward. The queue holds rows generated under slightly older
weights. That staleness is the standard memory bank tradeoff and is small here,
1024 rows at 64 per step is sixteen steps at a 2e-5 learning rate, but it was
tested against a fixed model rather than a moving one, so it is assumed and not
measured.

Also fixed in passing: this loop's checkpoint save omitted `feat_mu`, `feat_sd`,
`feat_bank` and `feat_bank_log_dist`, which serving reads. Any checkpoint it
produced could not be sampled from at all. They are now carried over from the
parent verbatim, which also keeps the serving distribution identical.

## The MMD repair works, does not help, and says why, 2026-07-27

Run `jq_queued_mmd_train_2026-07-28T011156+0000_71b2f1bd`, with
`w4_objective_vs_metric_2026-07-28T004341+0000_7ac305e0`. GPU, about 110
minutes, peak 71C.

Two arms from the same fc_v3 start, 1500 steps each, differing only in whether
the MMD used the queued estimator. Scored one-shot at n=2000, no selection.

```
                       parent   500     1000    1500
  control, blind MMD   0.6459  0.6676  0.7013  0.7252
  repaired MMD         0.6459  0.6836  0.7144  0.7046
```

The stage is monotonically DESTRUCTIVE. Lower is better and every arm moves the
wrong way, further the longer it trains. The repair shows no consistent
advantage, worse at 500 and 1000 and better at 1500, all inside the roughly
0.01 run to run noise this scorer has. Do not run this stage as it stands.

The repair is not what failed, and this is the part worth carrying forward. The
queued arm reached the LOWEST detector-space MMD of any checkpoint measured,
0.0437 against its own parent's 0.0447, while the control went the other way to
0.0729. So the term became genuinely optimizable and the model genuinely
optimized it. It bought nothing in contract space.

Across the four checkpoints there are feature matrices for:

```
                    contract AUC   detector MMD   detector quant
  fc_v2                 0.6327        0.0665         0.2041
  fc_v3                 0.6523        0.0447         0.2019
  jq_ctrl_s1500         0.7252        0.0729         0.2724
  jq_queue_s1000        0.7144        0.0437         0.2411

  correlation with contract AUC:  MMD +0.075    quant +0.938
```

The training's joint-matching term does not predict the score. The marginal
term does, and both arms made it worse.

Best explanation on the evidence: the objective is computed in
`detector_features` space, which is a distorted view of the contract space.
Two distortions are confirmed, `time_to_peak_velocity` ranking at Spearman
0.633 against the contract (see `w4_ttp_repair`) and the deliberate
`clamp(min=30.0)` speed floor under curvature. Matching a joint distribution in
a distorted space does not transfer to the space that grades it.

This reorders the work. Repairing the feature space is not a follow-up to joint
matching, it is a PREREQUISITE. Fix the space, rebuild the bank, retrain, and
only then is joint matching worth attempting again.

TENSION, recorded rather than smoothed over. `w4_mmd_alignment` measured Pearson
0.877 between this same MMD and the contract AUC. That test varied ONLY coupling,
on human data, with marginals held exact. This one compares real checkpoints
where marginals move too and dominate. The two results are consistent, and the
second is the situation training is actually in.

PROCESS NOTE. The first queue arm died at step 1000 and the log filter carried
no error pattern, so the cause was discarded silently and the first control
against queue comparison was 1500 steps against 1000. The table above is from a
rerun that completed at exit code 0, so the crash was transient and is
unexplained. Filters now include Error and Traceback.

## Where this leaves the programme, 2026-07-27

Three things are now known that were not known this morning, and they compose.

The gap is coupling, 84 percent of it, and it is not visible in any low order
projection: pairwise copulas sit at 1.08x the human split-half floor and triples
at 1.09x, both essentially at noise, while the full eighteen-way classifier
reads 0.657. There is no summary statistic to target, which is the argument for
a generic joint matcher rather than a hand-built one.

The generic joint matcher already in the code was statistically dead, and is now
repaired and verified to work as designed.

And repairing it changed nothing, because the space it works in is not the space
that grades the model.

So the single next thing that has to be true is that the training and the
contract measure the same quantities. `fix_ttp=True` is landed and default off
awaiting a bank rebuild. The curvature clamp is the other known distortion and
has not been addressed. Nothing else on the list is worth running before that.

RETRACTION, same day, of the two sections above wherever they explain the
negative result. Run `w4_mmd_symmetric_2026-07-28T011545+0000_c818e5f2`.

The explanation given above, that the objective does not predict the score and
that the detector space is to blame, was built on a measurement error of mine.
Both tables ranked checkpoints by the VALUE of `mmd_rbf_queued`. That function
is built to be gradient-correct, which puts coefficient 2 on the generated block
where the true MMD squared has 1. Its value is therefore not the true statistic
plus a constant, and it must not be used to rank distributions. Only its
gradient was ever meant to be used.

Redone with the symmetric `mmd_rbf` at 1024 per side, every conclusion flips:

```
  checkpoint        AUC    MMD contract   MMD detector
  fc_v2          0.6327       0.01417       0.01859
  fc_v3          0.6523       0.01370       0.01632
  jq_queue       0.7144       0.02006       0.02573
  jq_ctrl        0.7252       0.02214       0.03408

  correlation with the contract AUC:  contract +0.970   detector +0.900
```

and under graded corruption across the range real checkpoints actually occupy,
AUC 0.49 to 0.80, the correlation is +0.986.

So the MMD is a good proxy for the score, in BOTH spaces. The space is not the
problem, and the earlier claim that the queued arm reached the lowest MMD of any
checkpoint is also wrong. By the true statistic it went from 0.01632 at fc_v3 to
0.02573.

That changes what the negative result means. The fine-tune did not optimize the
wrong target. It made the model worse on the very quantity it was optimizing,
which is why the score moved with it. Meanwhile the in-training batch estimates
FELL over the same steps, control mmd 0.337 at step 1000 to 0.268 at 1500. A
training estimate that improves while held-out distribution match degrades is
the signature of the model exploiting the estimator rather than the
distribution, and at batch 32 there is a great deal of estimator to exploit.

That reading is consistent with the blindness result rather than in tension with
it, and it points somewhere specific: the queued estimator was applied at the
same 1500 steps and the same learning rate as the control, and the thing to test
is whether divergence is slower or absent when the estimator is hard to exploit.
The 500-step numbers hint at nothing useful yet, 0.6836 queued against 0.6676
control, both already worse than the 0.6459 parent.

UNAFFECTED by this error, since both were measured with the symmetric `mmd_rbf`:
the 63 percent separability at batch 32, and the queued gradient agreement of
0.843 against 0.543.

STILL TRUE and unchanged: the stage as it stands is destructive and should not
be run. What is no longer supported is the claim that repairing the feature
space is the prerequisite. That repair is still worth doing on its own evidence,
`w4_ttp_repair`, but it is not the explanation for this result.

## The stage optimizes a distribution the model never serves, 2026-07-28

Run `w4_train_serve_gap_2026-07-28T012815+0000_4f4b5d3d`.

This is the explanation for the destructive fine-tune, and it is not about the
MMD estimator at all.

The match loss never sees a served trajectory. It sees the output of
`partial_reveal` plus `st_complete`, run through `stream_to_frames`. I scored
that exact object with the contract scorer, on the same checkpoint that serves
at 0.6459:

```
  real tokens through the frame pipeline (floor)   0.5727 to 0.6071
  training path, reveal 0.0                        0.9532
  training path, reveal 0.9, feat=None             0.9020
  training path, reveal 0.9, feat supplied         0.8810
  training path, 12 steps, choice_temp 4.0         0.8762
  training path, 100 steps, choice_temp 10.0       0.8614
  fc_v3 as actually served                         0.6459
```

The training distribution reads roughly 0.88. The served distribution reads
0.6459. Those are not the same object, and they are about 0.25 AUC apart. Every
gradient this stage has ever taken was computed to move the 0.88 distribution
toward human. A better gradient on the wrong distribution is still the wrong
gradient, which is exactly why repairing the estimator changed nothing.

Two obvious candidate causes were tested and both are largely refuted as the
main driver. The character command, which training omits with `feat=None` while
serving always passes it, is worth 0.9020 against 0.8810. Sampler fidelity, 12
reveal steps at choice_temp 4 against serving's 100 at 10, is worth 0.8762
against 0.8614. Both move the number by about 0.02 against a gap of 0.25.

A separate and independent problem showed up in the same measurement. Real
tokens pushed through `stream_to_frames` score 0.5727 to 0.6071 across batches,
not the 0.497 real against real floor. The reconstruction is itself lossy, so
even a perfect distributional match in training space would leave a detectable
gap by construction. The 0.03 spread across three runs of the same measurement
also says this harness carries real run to run noise, and single points from it
should be read with that in mind.

What follows constructively. For this stage to be able to work at all, the loss
has to be computed on serving faithful samples run through the serving
reconstruction. That is expensive, which is where the queued estimator earns its
place: its value was never the statistics, it is that 32 fresh rows plus a queue
beat 512 brute forced rows, so the fresh sample budget per step drops by a
factor of 16 and a slow faithful sampler becomes affordable. The two results fit
together, but only in that order.

A correction for the record. Earlier in this session I read `partial_reveal` as
revealing ground truth tokens and told L that revealing 90 percent of the real
sequence still scored 0.899, calling it a candidate bug. It does not reveal
ground truth. It reveals the model's own sampled tokens, so `reveal_frac` is
simply how much of the sequence the iterative sampler produces before the
one shot completion finishes the rest. There is no bug there.

## No ceiling in the representation, but the trainer measures through a warped lens, 2026-07-28

Run `w4_token_ceiling_2026-07-28T031047+0000_ff75e9fd`.

Two results. The first one is good news and settles a question that has been
open by assumption rather than by measurement.

Real human trajectories, encoded into the model's own speed and dtheta
vocabulary and decoded straight back through the serving decoder, score 0.5118
against a split half floor of 0.467 to 0.497. Nothing is generated in that
measurement. The tokenisation is very nearly lossless, so there is no
representational ceiling standing between the model and the floor, and the
whole of the 0.6459 belongs to the model. The lattice work turns out to carry a
lot of that: dropping both EVENT_SNAP and the integer rounding moves the same
round trip from 0.5118 to 0.6175.

```
  real human tokens, nothing generated, contract pipeline
    serving decode, snap 2.5 + round    0.5118
    round only, no snap                 0.5275
    snap only, no round                 0.5249
    neither                             0.6175
    training renderer, stream_to_frames 0.5751
  reference split-half floor            0.467 to 0.497
  fc_v3 as actually served              0.6459
```

The second result is the actionable one. That last line of the block is the
differentiable renderer the match loss is computed on, fed the same real human
tokens, and it reads 0.5751 where the shipped decoder reads 0.5118. About 0.06
of the gap the loss is trying to close is manufactured by the renderer itself.
It is present in every gradient the stage has ever taken and no setting of the
model parameters can remove it.

Put beside `w4_train_serve_gap`, that is a complete account of why the stage
diverges. The objective is computed on a rendering that is offset from the
shipped one, applied to samples the model never serves. Both halves of the
measurement are wrong, in the same direction, and the optimiser has been
faithfully chasing the sum of the two errors.

The implied order of work is now specific rather than speculative. Make the
differentiable renderer agree with `experiments/event_stream_polar._decode`,
then make the sampled half serving faithful, and only then is there any point
asking whether the estimator is sharp enough. The queued MMD keeps its place
at the end of that queue rather than the front, and its value was never the
statistics, it is that 32 fresh rows plus a ring buffer beat 512 brute forced
ones, so a slow faithful sampler becomes affordable.

CORRECTION, self caught in the same session. Two cuts of this investigation
called `features.extract_features` directly on raw event points, which skips
`resample_trajectory`. The contract entry point is `extract_feature_matrix`,
which resamples to a uniform 125Hz grid first. That produced two numbers that
are pure harness artifact, real tokens through the serving decoder at 0.9255
and fc_v3 as served at 0.9592, and an explanation built on top of them about
3ms events aliasing into an 8ms scorer. None of it is real. Redone through
`extract_feature_matrix` the served value is 0.6606 against the known 0.6459,
the agreement one would expect. The `stream_to_frames` numbers elsewhere in
this investigation are unaffected, because that path already emits a uniform
125Hz series and the contract resample is an identity on it. That was checked
rather than assumed: 0.5751 through both entry points.

## Training through the served sampler, 2026-07-28

New trainer, `training/train_events_polar_sfmmd.py`.

Both defects found today have one cause. A pathwise gradient has to flow back
through generation, so generation had to be made differentiable, and every step
of making it differentiable moved it away from what is served. The
straight-through Gumbel completion, the frame grid renderer and the held real
timings are all consequences of that single constraint, and between them they
put the objective 0.25 AUC away from the served distribution and 0.06 away from
the shipped decoder.

The score-function estimator drops the constraint:

```
  grad E[c] = E[ c(x) * grad log p_theta(x) ]
```

Nothing between the logits and the cost has to be differentiable. So generation
becomes the exact serving sampler, the renderer becomes the exact serving
decoder, and the features come from `features.extract_feature_matrix`, which is
the contract itself rather than a GPU analogue of it. Both defects close at
once, because they were the same defect.

The premise was checked before spending GPU on it, which is the whole point of
having built the train/serve measurement first:

```
  the new trainer's own generated batch, contract scorer
    rollout,  12 reveal steps    0.7060
    rollout,  24 reveal steps    0.6777
    rollout, 100 reveal steps    0.6382
  fc_v3 as actually served       0.6459 to 0.6523
  the OLD trainer's own batch    0.8762 to 0.9532
```

At 100 steps the trainer grades an object indistinguishable from what is
served. The pilot runs at 24 to keep cost linear in reveal steps, which leaves
a 0.03 residual, roughly the run to run noise of the metric and two orders
below the 0.25 it replaces.

The price of a score-function gradient is variance, and this is where the queue
result earns its place rather than being a stranded method finding. The
per-sample cost is the MMD witness against a ring buffer of 1024 past generated
rows and 1024 real rows, which `w4_mmd_queue` measured at 100 percent
separability and Cohen's d 7.77 from 32 fresh samples, against 63 percent for
the 32 by 32 estimator the old trainer used. A low variance per sample cost is
precisely what this estimator needs, and it was not available before.

Two implementation notes worth keeping. The replay runs one reveal step at a
time and calls backward on each, so peak memory is flat in the number of reveal
steps rather than growing with it, which is what makes 24 or 100 steps a cost
question rather than a memory one. And the surrogate is divided by `seq_len`,
because log p is a sum over every revealed position and runs to hundreds, so
without that it would swamp the anchor as soon as `clip_grad_norm_`
renormalises the two together.

This is policy-gradient machinery and the GRPO pilot is closed, so the
difference should be stated rather than glossed. GRPO optimised a scalar reward
per sample produced by a learned critic, and it failed the way reward model
methods fail, by finding the critic's blind spots. There is no critic and no
reward here. The per-sample cost is the MMD witness against real data, a fixed
statistic of two empirical distributions, and it cannot be reduced without
actually moving the generated distribution toward the real one. If it diverges
the way the match loss stage did, that distinction is wrong and the approach
should be dropped rather than tuned.

## The difference is arrangement in time, not any property of the movement

Every fine-tune in this programme has aimed at the wrong target, and this
section is the measurement that says so.

Start with a defect that is real, correctly diagnosed, correctly fixed, and
worth nothing. Human event timestamps are quantised to whole milliseconds,
because that is what mouse hardware reports. Every duration in the reference
prints to exactly three decimals and the median distance from the millisecond
lattice is 1.4e-5 seconds, which is float64 representation error. The serving
decoder computes `dt_s` from a continuous exponential and emits arbitrary
floats, median distance from the lattice 0.25 ms. A forest splits at arbitrary
precision, so it reads that off almost perfectly: `movement_duration` alone
scored 0.7240. Rounding the emitted event times to the whole millisecond
removes the tell, taking that feature to 0.5649 against its own 0.4903 floor.
The full eighteen feature score does not move, 0.6601 before and 0.6644 after,
inside the plus or minus 0.03 this harness shows run to run. See
`w4_ms_lattice`.

That result is the useful one, because it is the first hard evidence that no
single feature repair can move this number. `w4_redundancy` then made it
general. Taken alone, every one of the eighteen features sits at its own split
half floor; the best is `movement_duration` at 0.5615 against 0.4903. Deleting
any single feature from the set costs at most 0.0053 out of 0.6798. So the
model has learned every individual property of human movement correctly, and
the entire detectable difference lives in how those properties combine.

`w4_coupling` confirmed that from the other direction. Rank transform both
sides onto identical uniform marginals, which destroys every marginal and
leaves only the dependence structure, and the score falls only from 0.6835 to
0.6761. Essentially all of it is dependence.

Two candidate mechanisms for that dependence were tested and both are dead.

The first was that the reveal samples positions too independently. A decoder
that reveals many events per step decides those events blind to one another,
and drawing the pieces of a sample too independently attenuates every
dependence, which matches the direction of the disagreements: human
`path_efficiency` couples hard to speed, duration and direction changes, and
generated couples about half as hard, same sign throughout. The prediction is
that raising the step count reduces the independence, pushing the correlation
slope toward one and the score down. It does not. `w4_attenuation` measured the
slope of generated rank correlation regressed on human rank correlation across
all 153 pairs, and it sits flat at 0.92 to 0.94 from 12 steps to 100 while the
score fails to fall monotonically. What survives that run is the fit quality:
r = 0.98. The dependence structure is the right shape and uniformly about seven
percent weak, so pairwise correlation is not the target either.

The second was that the model cannot hold a straight line. `w4_joint2d` found
the same deficit in every curvature pair, a four to four and a half percent
shortfall in the cell holding the lowest sixth of curvature spread and the
lowest sixth of angular velocity spread, with the generated cloud twenty to
twenty six percent wider there than human. That reads as a model that always
adds a little wobble where a person adds none. It is wrong, and reversed.
`w4_straight` compared raw turn tokens, before any decoding, and the model is
straighter than human: 39.4 percent of its motion events carry no turn against
32.4 percent for people, and its straight runs are longer, median 6 against 4
and p90 20 against 14.

Those two facts reconcile only one way. Both of the deficient features are
within path standard deviations. Landing in that corner requires turning that
is gentle and spread evenly along the whole path. Longer straight runs drawn
from the same turn histogram means the same total turning delivered in fewer
and bigger bursts, which raises the within path spread while every marginal
stays exactly correct. The turning is clumped where human turning is spread.

`w4_arrangement` measured what that is worth, using real human tokens only so
that no model error is anywhere in the picture. Decoded untouched through the
serving decoder they score 0.5576. Permute the speed values inside each window
of sixteen events, keeping every value and changing nothing but their order,
and the same real human data scores 0.8595. Permuting turns inside sixteen
gives 0.7368, across the whole path 0.7922, and timings inside sixteen 0.6295.

Local sequencing carries far more signal than the entire gap being chased. The
model at 0.66 sits between untouched real data and locally scrambled real data,
which is where a model with correct marginals and damaged local order would sit.
This is also why every marginal matching objective in the programme was unable
to help: none of them can see the difference between a sequence and a
permutation of itself.

The prime suspect is the decoder. It reveals events in gumbel perturbed
confidence order, so an event is generated conditioned on a scattered subset of
the sequence rather than its own neighbours, it is never revisited, and two
events revealed in the same step do not see each other at all. `order="l2r"`
already exists in `sample()` and reveals strictly left to right, which at 256
steps for a 256 position budget is fully sequential and removes the conditional
independence entirely. That sweep is `w4_order`. The docstring for `sample()`
warns that pure confidence order over straightens paths, `path_efficiency`
0.994 against human 0.949, so the sweep reports `path_efficiency` next to the
score to catch the same failure if left to right shares it.

## Both decode time repairs failed, and the second one says where the fault is

One caveat belongs on the previous section before anything is built on it.
`w4_arrangement` corrupts real human data, so it prices how sensitive the
detector is to local arrangement. It does not by itself establish that the
model's error is of that kind. What establishes that is three independent
measurements agreeing: every marginal correct, dependence correct in shape to
r = 0.98, and the model's own turning measurably clumped where human turning is
spread. The shuffle prices the channel; the other three say the model is on it.

The obvious repair was to change the reveal order. `w4_order` closes it. Left to
right at 100 steps scores 0.8348 against 0.6446 for the current gumbel order,
and drops `path_efficiency` to 0.8098 against a human median of 0.9430, which is
the wandering the `sample()` docstring attributes to random order. The
confidence order is doing necessary work: it lays down high confidence anchors
early and that is what fixes the macro shape of the path. It cannot simply be
swapped out. The remaining three arms were killed once this was unambiguous.

That run had a real flaw, caught after it was first written up as closed. It ran
on `event_polar_4m_fc_v3.pt`, whose `resid_embed` is `None`, so the aiming
channel was switched off, and left to right is the one order where
`prefix_resid` tracks a genuine partial state. The test therefore removed the
one advantage the sequential order has. `w4_order_resid` reruns it on
`event_polar_4m_resid_v6.pt`, which has the channel, comparing within the one
checkpoint: gumbel at 100 steps 0.6458 with `path_efficiency` 0.9209, left to
right at 100 steps 0.8755 at 0.7864, left to right at 256 steps 0.8448 at
0.7647. The conclusion survives the correction unchanged.

What it does not establish is that sequential generation is wrong in general.
This model is trained by hiding a random subset of positions, so a contiguous
revealed prefix is a context it has essentially never been shown, and decoding it
left to right asks for a prediction it was never taught. `w4_prefixcond` tried to
measure that penalty directly and is recorded as inconclusive by construction:
the prefix cross entropy is much higher, plus 1.31 nats on speed at fifteen
percent hidden, but filling a gap between two revealed neighbours is
intrinsically easier than predicting forward from the past alone, so a model
trained purely on prefixes would show a penalty on that comparison too. Nothing
short of an actual prefix trained baseline separates the two explanations. Do not
rerun it.

The second repair keeps the order and adds a local pass afterwards. Hide a
random subset of the finished events, redraw them with everything else visible,
repeat. Each redrawn event then sees its immediate neighbours, which it never
did the first time, and the macro shape survives because the events that stay
visible hold it. This is the standard corrector for a masked model and each pass
is one sweep of a Gibbs sampler on the model's own conditionals.

It fails, monotonically in the total volume of redrawing. `w4_refine`: 0.6559
unchanged, 0.6698 at six passes hiding eight percent, 0.7071 at six hiding
fifteen, 0.7213 at sixteen hiding eight, 0.7540 at sixteen hiding fifteen,
0.7701 at six hiding thirty, with `path_efficiency` sliding from 0.9353 down to
0.8481.

That negative is worth more than the positive would have been, because it is in
distribution. The training schedule runs half of its examples at under thirty
percent of positions hidden and a third under fifteen, so a fifteen percent
redraw is a state this model saw constantly. Repeated redraws move a sample
toward the stationary distribution of the model's own conditionals. That
distribution is worse than the model's one shot sample. So the conditionals
themselves are wrong, and the partly greedy confidence ordered reveal has been
compensating for them all along. Both decode time routes are now closed and the
defect is in the trained model.

One explanation covers every measurement in this session. Suppose the per event
predictions are centred correctly but too uncertain. Then the marginals come out
right, because a diffuse distribution centred correctly still integrates to the
right marginal. The dependence comes out the right shape and uniformly weakened,
which is exactly the slope 0.92 at r = 0.98 of `w4_attenuation`, because
independent noise added to each draw attenuates every dependence by a common
factor. The local sequencing comes out noisy, which is the channel
`w4_arrangement` prices. And a Gibbs pass drifts, because it reinjects that noise
at every sweep rather than averaging it away.

It also makes a prediction, and the refinement sweep already contains a first
read on it. Dropping the redraw temperature from 1.0 to 0.8 at six passes
recovers about a third of the damage, 0.7071 back to 0.6792, with
`path_efficiency` back from 0.8759 to 0.8999. Sharper redraws hurt less, which is
what an over-diffuse conditional predicts and what a mis-centred one does not.

`w4_sharpness` tested the same thing on the sampler itself rather than on a
corrector pass, and refutes it. Sweeping the sampling temperature gives 0.7184 at
0.7, 0.6929 at 0.8, 0.6580 at 0.9, 0.6433 at 1.0 and 0.6737 at 1.1. The locked
serving temperature of 1.0 is already the optimum, and sharpening the sampler
makes things worse in both directions from there. Lowering the turn head
temperature alone barely moves anything, 0.6396 at 0.9 and 0.6308 at 0.8. So the
model's one shot conditionals are not globally over diffuse. The
over diffuseness story survives only as a statement about the corrector, where
sharpening helps because it damps a divergence, not about the sampler.

The temperature sweep leaves a residue that names the real shape of the problem.
At the optimum, `curvature_std` is 0.2812 against a human 0.3459, so the paths
are too smooth. Turning the temperature up to 1.1 fixes exactly that,
`curvature_std` 0.3259, and breaks `path_efficiency`, 0.9125 against a human
0.9430. One scalar cannot satisfy both, and that is the signature of
misspecification rather than miscalibration. The model does not have the right
distribution scaled wrong. It has the wrong distribution.

Taken together with the two failed decode time repairs, every sampling side
route is now closed: the reveal order cannot be changed, local repair diverges,
and the temperature is already where it should be. The defect is in what the
model learned, and the one clean piece of evidence about its nature is
`w4_refine`, which compares the model against itself and has no confound: the
stationary distribution of its own local conditionals is worse than the sample
the ordered reveal produces, so those conditionals are mutually inconsistent.

Worth recording alongside this: the six best scores in the whole ledger, from
0.5391 up, are all `per_item_sir_judge` arms drawing eight to thirty two
candidates and choosing among them, which the mandate disqualifies. They run at
sampling temperature 0.7. That is a selection result and not a number this
programme may claim, but the temperature it settled on points the same way as
everything above.

## The conditionals do not compose, so the factorization is the thing to change, 2026-07-27

Everything at decode time is now closed. The reveal order cannot be changed
(`w4_order`, `w4_order_resid`), a local repair pass diverges (`w4_refine`), and
the sampling temperature is already at its optimum in both directions
(`w4_sharpness`). Whatever is wrong is in the trained model, and one of those
three negatives says what kind of wrong.

`w4_refine` is the one measurement in the whole session with no confound,
because it compares the model against itself rather than against a human
reference or a corrupted control. Hiding a fraction of a finished sample and
redrawing it under the model's own conditionals is one sweep of a Gibbs
sampler. Iterating it converges, by construction, to the stationary
distribution of those conditionals. The score gets monotonically worse with
every sweep, 0.6559 unchanged to 0.7701 at six passes hiding thirty percent,
and this happens at hidden fractions the model saw constantly in training. A
family of conditionals whose stationary distribution is worse than the sample
they were used to produce do not agree with each other. There is no single
joint distribution they are all conditionals of.

That is not a bug in this checkpoint. It is a structural property of any-order
masked modelling. One network is asked to supply p(x_i | x_S) for every subset
S, nothing in the objective ties those answers together, and the ordered reveal
at sampling time has been quietly compensating for the disagreement. Every
result in this session is consistent with it: marginals correct
(`w4_redundancy`), pairwise dependence correct in shape and uniformly weak
(`w4_attenuation`), local arrangement damaged (`w4_arrangement`,
`w4_joint2d`), and a temperature knob that cannot satisfy curvature and
straightness at the same time (`w4_sharpness`).

A chain-rule factorization cannot have this defect. p(x) = prod p(x_i | x_<i)
is a joint distribution by arithmetic, whatever the network predicts. That is
the reason to build `models/event_ar.py`, and it is a reason of a different
kind from the ones this programme has been running on: not a hypothesis about
what the model is getting wrong, but a property the current architecture cannot
have and the replacement has for free.

Three further consequences, each attached to a number already in this file:

- Supervision. Masked training teaches only the hidden positions, median mask
  fraction 0.2963. Teacher forcing teaches every position on every example, so
  the same corpus carries roughly three times the signal.
- Aiming. Under scattered masking the pointer's position is unknowable past the
  first gap, which is why `prefix_resid` measures from the longest revealed
  prefix and returns one vector per sequence. Left to right it is exact and per
  position, in training and sampling alike. The entire W3 P1 aiming programme,
  six checkpoints, was an attempt to approximate a quantity that is free here.
- Time. 98.4 percent of recorded human event times are within a microsecond of
  a whole millisecond and none exceed 150, so time becomes a 151 way choice
  rather than a continuous head. `w4_ms_lattice` priced that tell at 0.7240 to
  0.5649 on duration alone, and rounding after the fact recovered it without
  touching the contract score. Tokenizing removes it by construction instead.

### The prior failure this is not

`resid_v3`, `v4` and `v5` trained the masked model on contiguous suffix masks,
v5 with an exponential weighting that concentrated the loss on exactly the next
event the left to right decoder consumes. All three failed badly, raw 0.88 to
0.99 against a 0.647 base, and the recorded conclusion was that a 4k step fine
tune cannot turn a scattered mask infilling model into a competent left to
right generator.

That conclusion is correct and it is not an argument against this build. Those
runs were 4000 step fine tunes at lr 2e-5 from a bidirectional checkpoint with
the dt head frozen, and the trainer's own log records the loss flat from step
1000, which is what running out of adaptable weights looks like. They also left
attention bidirectional: suffix masking hides token VALUES, every position is
still attended, so the model was never prevented from learning representations
that assume a visible future. `models/event_ar.py` is trained from scratch with
a causal attention mask, verified by construction rather than by assumption:
editing an event at position 15 changes the model's output at positions 0
through 14 by exactly 0.0.

`w4_prefixcond` was an attempt to measure the training coverage gap directly
and is recorded as inconclusive by construction. The prefix penalty is large,
plus 1.31 nats on speed at fifteen percent hidden, but filling a gap between
two revealed neighbours is intrinsically easier than predicting forward from
the past alone, so a prefix trained model would show a penalty on that
comparison too. It cannot separate the two explanations and no reweighting of
it can. Do not rerun it.

### What is being built

`models/event_ar.py`, 7.95M parameters, 8 causal layers, d_model 256. Emission
order within a step is p(s) p(th | s) p(dt | s, th), which keeps the speed and
turn conditional structure the masked model already has and conditions dwell
time on the motion it accompanies. Conditioning is the same four dimensional
vector, plus a per position exact state channel carrying distance still to
cover, its direction, elapsed fraction of the commanded duration, step
fraction, and distance travelled.

`training/train_event_ar.py` trains it from scratch, plain teacher forced cross
entropy on all three streams, supervised at every real position plus the single
terminating PAD that teaches the model to stop. `research/w4_ar_eval.py` scores
it one trajectory per spec with no selection, decoding through the unchanged
serving decoder so the number is comparable to every other one shot row in the
ledger.

The honest prior: this is the first architecture change in the programme
motivated by a property the current family provably lacks rather than by a
diagnosis of its output, but a coherent joint is a necessary condition and not
a sufficient one. A left to right model with a coherent joint can still place
that joint in the wrong place. The measurement that would settle it is the
contract score at temperature 1.0 against the 0.6433 the masked model reaches,
and the diagnostic that would say whether the mechanism did what it claims is
whether the local arrangement statistics move, not whether the marginals do,
since the marginals were never the problem.

## The autoregressive model trained, and the mechanism came back half right, 2026-07-28

`event_ar_v1.pt`, 7.95M parameters, 8 causal layers, 40k steps from scratch on
1.5M trajectories, 6960 seconds, GPU peak 72C. Scored by `research/w4_ar_eval.py`,
one trajectory per spec, no candidates, no selection, decoded through the
unchanged serving decoder.

```
  s_temp   contract   dur_only   path_eff   curv_std   miss_px   n_ev_p50
    0.90     0.6076     0.5413     0.9251     0.3897       4.9         46
    1.00     0.6271     0.5295     0.9117     0.4234       4.9         42
    0.80     0.6590     0.5414     0.9434     0.3354       5.2         49
    1.10     0.7165     0.5248     0.8650     0.5891       5.8         37
  (human)                          0.9430     0.3459
```

The headline is marginal and should be reported that way. 0.6076 against the
masked model's best one shot 0.6433 is a 0.036 move against 0.03 run to run
noise. Three other results are not marginal.

### Arrival, which closed a programme, is solved incidentally

Median miss from the commanded endpoint is 4.9px. The masked base misses 58px.
Six checkpoints, `resid_v1` through `v6`, were built specifically to fix that
and the best reached 55.3px against a 15px gate that none of them cleared,
which is what closed W3 P1. This model clears that gate by a factor of three on
its first training run with no aiming machinery of any kind.

The reason is structural rather than lucky. Under scattered masking the
pointer's position is unknowable past the first gap, so `prefix_resid` can only
measure from the longest revealed prefix and returns one vector per sequence.
Left to right the remaining displacement is exact at every step, and it is
handed to the trunk as a per position channel. W3 P1 spent six checkpoints
approximating a quantity that this factorization makes free.

It also changes how the two scores compare. Forcing arrival on the masked model
costs between 0.045 and 0.078. Judged on paths that actually land on the target,
which is the only kind the product can use, this is 0.6076 against roughly 0.70.

### The millisecond tell is gone by construction, and the opposite tell was checked

Duration alone separates human from generated at 0.5295. The masked model's
continuous flow head scores 0.7240 on the same instrument and post hoc rounding
takes it to 0.5649, so choosing whole milliseconds directly beats rounding
afterwards.

The opposite risk was measured rather than assumed. Generated timestamps land on
the whole millisecond grid to 4.8e-7 s where human ones sit at 1.4e-5 s, thirty
times cleaner, which would have been a new tell in the other direction. It is
not: 0.5295 is near the floor, so the forest cannot use it.

### The misspecification signature is gone

On the masked model no single temperature satisfied curvature and straightness
together. At its optimum `curvature_std` was 0.2812 against a human 0.3459, and
the temperature that fixed curvature broke `path_efficiency` to 0.9125 against a
human 0.9430. Here temp 0.8 gives `curvature_std` 0.3354 against 0.3459 and
`path_efficiency` 0.9434 against 0.9430, both essentially exact at the same
time. The tension that named the old defect does not exist in this family.

Worth recording alongside it: the temperature where those two marginals match
human best is NOT the temperature that scores best, 0.6590 at 0.8 against
0.6076 at 0.9. Consistent with `w4_redundancy` and `w4_coupling`. The marginals
were never the gap and matching them better does not buy the score.

### The mechanism claim, and the half that failed

`research/w4_seqstats.py` measures arrangement directly on the token streams,
before any decoding, so no part of the serving pipeline can hide a difference.
Pooled autocorrelation is computed after standardizing within each trajectory,
which makes it blind to level and spread, the properties already known correct.

```
                         human    masked served    ar v1
  speed acf lag1        0.5952           0.5205   0.6849
  speed acf lag2        0.6220           0.6727   0.6914
  speed acf lag4        0.4607           0.5496   0.5219
  turn acf lag1         0.2744           0.2691   0.2509
  turn acf lag2         0.1303           0.1925   0.1462
  turn gap dispersion   2.7781           5.3281   1.8692
  straight run p90         5.0              7.0      5.0
```

Turning is decisively repaired. The masked model's turn gap dispersion is 5.33
against a human 2.78, which is the clumping `w4_joint2d` and `w4_straight`
inferred indirectly, now measured directly. The AR model reaches 1.87, cutting
the error from 2.55 to 0.91, and its straight run p90 lands on the human 5.0
exactly against the masked model's 7.0. Turn autocorrelation at lag 2 goes
0.1925 to 0.1462 against a human 0.1303.

Speed is NOT repaired and reverses sign. Human lag1 is 0.5952, the masked model
undershoots at 0.5205, and the AR model overshoots at 0.6849. The error
magnitude is comparable, 0.075 against 0.090, in the opposite direction.

This is the half that matters most, because `w4_arrangement` already priced the
two channels. Shuffling speeds inside a window of sixteen takes real human data
from 0.5576 to 0.8595; shuffling turns takes it to 0.7368. The channel that got
fixed is the cheaper one and the expensive one traded one error for another.
That is the specific reason the score moved 0.036 while arrival improved
twelvefold and the timing tell was eliminated outright.

### What the speed number says about the cause

Human speed carries a lag2 autocorrelation ABOVE its lag1, 0.6220 against
0.5952, a ratio of 1.045. That is the signature of an alternating component
sitting on a smooth trend: a large step tends to be followed by a small one,
which is what a fixed poll cadence does to a continuous movement. The masked
model exaggerates it, ratio 1.292. `ar_v1` has all but lost it, ratio 1.010: it
learned the trend and not the alternation.

Two candidate causes, one cheap and one expensive, and they are distinguishable.
If the per step speed distribution is merely too narrow, heating the speed head
alone walks lag1 down toward human and the defect is calibration. If it cannot,
the defect is in the factorization, and there is a specific suspect.

`ar_v1` emits p(s) p(th | s) p(dt | s, th), choosing the speed before it knows
the interval that speed covers. A mouse reports on a fixed poll cadence, so the
displacement in one sample is roughly velocity times its interval. Choosing
speed first forces the network to marginalize over the interval, and
marginalizing is exactly what smooths a sequence. `models/event_ar.py` now
carries `emit_order="dt_s_th"`, p(dt) p(s | dt) p(th | s, dt), which is the same
joint by the chain rule and asks the network for a different conditional. The
v1 checkpoint still loads strict against the updated file, so nothing already
measured is invalidated.

## The pauses are corrective reversals, the model knows it, and does not do it, 2026-07-28

This section exists because L asked a question no measurement in this programme
had asked: WHY does a person's mouse stop moving one event in ten? L's account
was that a person sets off toward an element they have not properly looked at,
aims wrong, stops, and makes a second more targeted movement, sometimes
overshooting and coming back. That is also the standard account in motor
control, Woodworth's two component model and Meyer's optimized submovement
model. Every result below follows from taking it seriously.

### First it killed the retrain that was about to be launched

`event_ar_v1` emits p(s) p(th | s) p(dt | s, th), choosing a displacement before
knowing the interval it covers. The plan of record was `event_ar_v2` with
`--emit-order dt_s_th`, two GPU hours, on the argument that a mouse reports on a
fixed poll cadence so displacement is velocity times interval, and choosing
displacement first forces the network to average over the interval, which would
destroy the alternation `w4_seqstats` found missing.

`research/w4_dtstruct.py` tested that on the corpus alone, no GPU, no sampling.

```
  stream                         lag1     lag2     lag3     lag4 lag2/lag1
  displacement s               0.6379   0.6690   0.5390   0.4919     1.049
  interval dt                  0.0973   0.1073   0.0475   0.0521     1.104
  velocity s/dt                0.6120   0.6411   0.5196   0.4751     1.048
  displacement s (motion)      0.7325   0.6824   0.5847   0.4914     0.932
```

Dividing the interval out leaves the alternation exactly as it was, 1.048
against 1.049, and intervals are nearly uncorrelated between events at 0.097. So
the alternation is not carried by the interval and the emit order cannot be its
cause. The retrain was cancelled before it started.

The fourth row says where the alternation does live. Restricted to events that
actually move, the ratio falls below one and the decay becomes monotone. The
alternation is the interleaving of zero displacement still events among moving
ones, which is a statement about still placement, not about speed.

### Both models are wrong about still placement, from opposite sides

`research/w4_tickstruct.py`, on token streams before any decoding.

```
  arm              share   gapVMR   run50   run90   allAc1  ratio   motAc1  ratio
  human           0.0948  21.5687     2.0    21.0   0.5572  1.141   0.6894  0.951
  masked served   0.1567  17.5409     1.0    12.0   0.5166  1.298   0.7406  0.946
  ar event_ar_v1  0.0621  25.2242     3.0    37.0   0.7354  0.953   0.7586  0.924
```

With the stills removed the three speed autocorrelations nearly agree, 0.6894,
0.7406 and 0.7586. Speed itself was never far wrong. What `w4_seqstats` scored
as a speed arrangement defect is still placement, and the two models bracket
human from opposite sides: the masked model emits two thirds too many and
chops the moving runs in half, `event_ar_v1` emits a third too few and lets
them run to nearly twice the human length.

### A pause is a reversal, not a rest

`research/w4_submove.py`. The first version of this probe counted every still
run and found nothing: heading change across a pause 0.164 rad against a
baseline 0.157. That version was misspecified. Most still runs are a single 8ms
sample where the pointer did not travel far enough to register a lattice step,
which is quantization rather than a decision, and there are thousands of them
against a few hundred real pauses. Requiring a pause to last at least 40ms,
below any plausible visual correction latency and well above one poll interval,
changes the answer completely.

```
  arm                nPause  durMs  remFrac  remLast  turnPause  turnBase  overshoot
  human                0.23     78    0.364    0.353      1.511     0.157      0.936
  masked served        0.24     70    0.319    0.143      1.178     0.157      0.981
  ar event_ar_v1       0.05     65    0.451    0.423      1.657     0.196      0.914
```

A real human pause lasts a median 78ms and the direction of travel changes
1.511 rad across it, about 87 degrees, against 0.157 rad across uninterrupted
motion. The obvious objection is that the pointer is crawling either side of a
pause, a median 1.4px per event, where a one pixel step right followed by a one
pixel step up is a right angle by construction. A speed matched control
refutes it and the effect grows with speed.

```
   speed bin    nP  turnPause      nB  turnBase   ratio
         1-2   118      1.178   14344     0.157    7.50
         2-3    28      1.098    7546     0.133    8.28
         3-5    26      2.475    8376     0.133   18.68
         5-8    14      2.612    6833     0.123   21.29
          8+    10      2.481   14912     0.123   20.22
```

At five to eight pixels per event the heading change across a pause is 2.6 rad,
about 150 degrees, which is close to a reversal. Together with the 93.6 percent
of human paths that come within 10 percent of the target and then leave again by
more than 5 percent of the distance, this is L's mechanism measured: the launch
misses, the person stops for about 78ms, turns most of the way around, and comes
back.

Two parts of the account do not survive contact with the corpus and are recorded
so nobody re-derives them. Pauses do not sit at the target: 0.364 of the
distance still remains at the median pause and the last pause is no closer than
the first. The movement after a pause is not slower: the median speed ratio is
1.000 with a huge spread, p10 0.47 and p90 2.98, and the mean log ratio is
slightly positive. Nor does pause count rise with distance, r 0.018.

### The AR model knows all of this and still will not do it

`event_ar_v1` produces 0.05 real pauses per trajectory against a human 0.23, one
movement in twenty instead of one in four, and when it does pause it re-aims
harder than a person, 1.657 rad. A shortfall in a sampled rate has two very
different causes, so `research/w4_stillcal.py` teacher forces the model on 5426
real human trajectories, 302218 events and 1539 real pause onsets, and reads the
conditional off directly with no sampling in the loop.

```
     remaining        n   empirical  predicted   onsetRate
     0.00-0.10    64185      0.1402     0.1390     0.00907
     0.10-0.20    33956      0.0897     0.0885     0.00389
     0.20-0.35    36651      0.0842     0.0828     0.00385
     0.35-0.50    28051      0.0729     0.0727     0.00203
     0.50-0.70    33825      0.0746     0.0720     0.00287
     0.70-1.00    69622      0.0864     0.0866     0.00290
          1.00+   35928      0.1082     0.1060     0.00913
```

Overall it predicts a still at 0.0968 against an empirical 0.0979, it tracks the
empirical rate to three decimals in every bucket, and at a real pause onset it
puts 0.3238 against 0.0512 at an ordinary moving event, a lift of 6.32x. The
model learned pauses correctly and completely. The shortfall is not knowledge.

The same table says which states matter. The human still rate is U shaped in how
far the pointer still is from where it ends up: 0.1402 within a tenth of the
distance, a trough of 0.0729 in the middle, and 0.1082 once the pointer is
FARTHER from the target than the entire movement was long. Real pause onsets
concentrate at exactly those two extremes, 0.00907 and 0.00913 against 0.00203
in the trough. The second extreme is a person who has gone badly wrong, and it
is 11.9 percent of all human events.

A conditional that is right on human prefixes and a sampled rate that is wrong
looks like it can only be reconciled by state visitation, so that was measured
too, in `research/w4_statevisit.py`, as the share of events at each distance
from the trajectory's own endpoint.

```
  arm              0-.10  .10-.20  .20-.35  .35-.50  .50-.70  .70-1.00   >1.00  maxP90
  human           0.1976   0.1098   0.1107   0.0925   0.1132    0.2480  0.1282   1.109
  masked served   0.2336   0.1266   0.1288   0.0971   0.1125    0.2315  0.0699   1.005
  ar event_ar_v1  0.1829   0.1091   0.1245   0.0995   0.1111    0.2332  0.1398   1.122
```

That refutes it. `event_ar_v1` reaches the badly wrong state MORE often than a
person, 0.1398 against 0.1282, and its worst excursion at p90 is 1.122 against
1.109. It gets into trouble at the human rate, its conditional says a person
would stop there, and it does not stop. The served masked model is the one that
under-visits, at 0.0699. This is left unexplained rather than papered over.

### And then the channel was priced, and it is nearly worthless

`research/w4_stillprice.py` prices still structure the way `w4_arrangement`
priced speed order: corrupt only that property in real human data and read the
detector. Placement is corrupted by a riffle that keeps the still subsequence
and the moving subsequence each in their original internal order and redraws
only the interleaving, so every marginal survives bit for bit. Count is
corrupted by dropping or duplicating stills with their interval carried to the
neighbour, so total elapsed time and the moving subsequence survive exactly.

```
  passthrough    0.5344
  riffle whole   0.5860     placement fully destroyed
  riffle w=16    0.5416     placement destroyed locally
  share 0.0621   0.5409     the count event_ar_v1 emits
  share 0.1567   0.5260     the count the masked model emits
  share 0.3000   0.5435     three times the human count
```

Destroying still placement outright is worth 0.052. Destroying it locally is
worth 0.007. The count is worth nothing at all, at any of the three arms, and
0.5260 sits below the floor. Against this, `w4_arrangement`'s speed shuffle in
windows of sixteen took human data from 0.5576 to 0.8595.

So the mechanism is real, it is now measured far more precisely than anything
before it in this programme, and the detector barely cares. Fully destroyed the
whole channel is worth under 0.05 against a gap of 0.11 to 0.14, and at the
error sizes the two models actually carry it is worth about 0.01. Do not spend
training on still placement or still count.

The general lesson is the more useful part. Every hand picked statistic priced
in this programme has come back small while the joint keeps separating at 0.61
to 0.64. That is now three independent times: marginals in `w4_redundancy`,
speed and turn arrangement in `w4_arrangement`, and stills here. The next
measurement should stop proposing channels and ask the detector which of its
eighteen features carries the signal and how many of them it needs.

### What this is not

It is not a leak of future information. `prefix_state` builds its remaining
displacement against `exp(cond[:, 0])`, the COMMANDED straight line distance,
which is supplied at serving time exactly as it is in training. A person aiming
at an element on a page can see where it is too. The asymmetry is not in what
the model knows, it is that the model steers more accurately than a person does
from the same knowledge.

Nor is it the masked model's defect. `w4_refine` showed the masked model's local
conditionals are mutually inconsistent, so no joint exists for them to be wrong
about. A chain rule factorization has exactly one joint by arithmetic. If its
conditionals were exactly right the sampled distribution would be right too, so
the conditionals are wrong somewhere, and this measurement narrows where: not in
the still head, which is calibrated bucket by bucket, but in whatever governs
whether the path goes wide enough to need rescuing.

## The gap is coupling, the local conditionals are all right, and an independent auditor now exists, 2026-07-28

This section replaces guessing channels with asking the detector, and it ends
with the control L asked for: a second instrument that shares nothing with
`scoring.py` and that must never be trained against.

### Asking the detector what it uses

`research/w4_whatsees.py`, on `event_ar_v1` at temp 0.9, n 2500. The local AUC
recipe reproduces the contract scorer exactly, 0.6668 both ways, so the
instrument is trustworthy.

```
  single feature                 auc      greedy forward                 auc    gain
  movement_duration           0.5699      +movement_duration          0.5699  0.0699
  angular_velocity_mean       0.5310      +angular_velocity_mean      0.5574 -0.0126
  mean_acceleration           0.5304      +angular_velocity_std       0.5966  0.0392
  path_efficiency             0.5290      +curvature_mean             0.6205  0.0239
  curvature_std               0.5265      +mean_acceleration          0.6370  0.0165
  ...                                     +max_deviation              0.6517  0.0147
  max_deviation               0.4901      +std_acceleration           0.6547  0.0030
  std_acceleration            0.4726      +path_efficiency (10th)     0.6652  0.0049
```

No single feature reaches 0.57. Eight of the eighteen are at or below chance
alone. Removing any one of the eighteen costs at most 0.0244. And the second
greedy step goes DOWN, so nothing pairs usefully with duration.

The tell is `angular_velocity_std`. Alone it is 0.4958, literally uninformative.
It is the second most expensive feature to remove and it buys the largest greedy
gain of any step after the first. Information that appears only in combination
is coupling, not marginals.

READ THE SINGLES AND GREEDY COLUMNS WITH CARE. `w4_dvjoint` established that
this instrument's AUC is NOT additive across dimensionality. `movement_duration`
alone is 0.5699, `mean_velocity` alone is 0.4933, and the two together are
0.4939, below the weaker of the pair. That survives setting `max_features=None`
so every feature is available at every split, which rules out random split
dilution: it is genuine out of bag variance from carrying an extra continuous
column. So a single feature AUC and an eighteen feature AUC are measured under
different amounts of regularisation and cannot be subtracted from one another.
The singles and greedy tables above are within dimensionality rankings and
nothing more. The `drop_one` table is 17 against 18 and is fine, and every
eighteen against eighteen arm in this section is fine. The coupling conclusion
rests on those, not on the singles.

### Pricing the marginals against the coupling

`research/w4_copula.py`. A rank transform writes a human column's sorted values
into a generated column's sorted positions, so the generated marginal becomes
exactly human while every dependence stays exactly as the model produced it.

The first version of that file used the scoring reference as its own donor,
which put identical float values on both sides. Out of bag forest predictions
then invert, because an out of bag human row lands in a leaf holding only its
generated twins. Every single feature AUC came back at 0.04 to 0.07, far BELOW
chance, which is the artifact announcing itself, and it biases the headline arm
downward. The donor is now `data/human_ref_features_sir.npy`, 4000 human rows
with zero rows in common with the scoring reference, verified in the script
before it will run. Do not remove that check.

```
  baseline                     0.6668
  marginals fixed              0.6208
  human through transform      0.5148     the null for this instrument
  gen shuffled                 0.9990
  both shuffled                0.6640
```

CORRECTED BELOW BY `w4_dvjoint`, and the correction matters. Reading the
untransformed baseline against the 0.5148 TRANSFORMED null gives 0.152 and a
seventy percent coupling share. That is wrong: the baseline has not been through
the transform, so its null is two disjoint human samples with no transform at
all, which is 0.4889. With each arm against its own null the baseline excess is
0.178 and the marginals fixed excess is 0.099, so perfect one dimensional
statistics on all eighteen features buy back 0.079, which is 44 percent. The
split is roughly 56 coupling against 44 marginals. Coupling is still the
majority and no training could beat perfect marginals, so a distribution fix
still does not reach 0.50, but it is not the four to one story the first reading
gave.

The 0.9990 row is the same statement from the other side. Scramble the model's
own cross feature structure and the detector spots it instantly. It is
overwhelmingly a relationship reader.

`both shuffled` at 0.6640 is a counterfactual and bounds nothing. With both
joints reduced to products of their marginals, eighteen weak independent
differences compound in a way they cannot when the features are correlated, so
that arm can sit above the baseline. It is evidence the marginals are jointly
non trivial and nothing more.

### Which couplings, and in which direction

`research/w4_couplemap.py`, pairs on the marginal matched matrix so every
remaining signal is coupling. Null all 18 0.5148, null median pair 0.5032.

```
  largest rank correlation error                          rhoH    rhoG    drho
  max_velocity + angular_velocity_std                   -0.024  -0.210  -0.186
  max_velocity + angular_velocity_mean                  -0.081  -0.265  -0.184
  std_velocity + angular_velocity_mean                  -0.096  -0.279  -0.183
  std_velocity + angular_velocity_std                   -0.049  -0.223  -0.174
  max_acceleration + angular_velocity_std               -0.013  -0.183  -0.170
  max_deviation + angular_velocity_std                   0.172   0.004  -0.168
  max_acceleration + angular_velocity_mean              -0.075  -0.233  -0.158
  max_velocity + curvature_mean                         -0.024  -0.182  -0.158
  max_acceleration + num_direction_changes               0.248   0.093  -0.155
  std_acceleration + num_direction_changes               0.016  -0.139  -0.155
```

All ten pair a speed or acceleration feature with a turning feature, all with
the same sign, the model always more negative. Two rows say it without any
statistics. Human wiggle accumulates into real displacement from the straight
line at 0.172 and the model's cancels out at 0.004. Human direction changes
arrive with acceleration bursts at 0.248 and the model's barely do at 0.093.

`research/w4_partial.py` then shows the whole thing is mediated by two features.

```
  mean absolute coupling error, control 'none':        0.1689
  mean absolute coupling error, control 'duration':    0.1197
  mean absolute coupling error, control 'dur + speed': 0.0249
```

The top pair makes it concrete. Raw, human `max_velocity` against
`angular_velocity_std` is -0.024 and the model is -0.210. Control for duration
and mean velocity and BOTH flip to positive and agree: human +0.249, model
+0.246. The raw negative correlation was duration and speed acting on both
columns at once, and once that is removed the model's remaining structure is the
human structure.

Conditional on duration and mean velocity the model is human. That does NOT
make duration wrong, and reading it that way cost an hour. It means the error
lives in how duration relates to turning.

### Three hypotheses died here, and the negative results are the section

FIRST, that the model cannot turn while moving fast. `research/w4_speedturn.py`
refutes it outright. Sharp turn share in the three fastest speed bands is 0.032,
0.035, 0.039 for `event_ar_v1` against a human 0.032, 0.027, 0.037, and the
within path speed against turn correlation is -0.225 against a human -0.203.
Event by event the speed and turn joint is right. The coupling defect is
trajectory level, not event level.

SECOND, that duration itself is broken. `research/w4_durfit.py` refutes it three
ways. A human token stream's intervals sum to its claimed duration at a ratio of
1000.000 at p10, p50 and p90, so the convention is exact and the unit is
milliseconds against seconds. The `esp._duration` sampler reproduces human
p(duration | distance) with a conditional mean error under 0.07 in log space in
every distance band and a spread ratio of 1.030. The evaluation specs cover the
same distances people actually moved, 26.4 against 26.4, 183.9 against 185.3,
948.8 against 947.3 px. Realized durations track human at every percentile. The
only real defect is range: the model never emits a movement under 48ms or over
3.38s where humans span 30ms to 7.77s, which is about 1.5 percent of cases and
cannot carry 0.5699.

THIRD, that the model fills extra time by moving slowly instead of correcting.
The opposite. At matched distance tercile AND matched duration tercile the model
turns MORE than a person in nearly every cell.

```
  cell     n H / n F   ndirchg H/F   angvelstd H/F   curvstd H/F   maxdev H/F
  d0 t0     393 / 437     7 / 7      34.43 / 44.88   0.13 / 0.24   2.32 / 2.48
  d0 t1     201 / 294    16 / 20     54.02 / 61.18   0.60 / 0.83   3.94 / 5.75
  d0 t2      66 /  73    34 / 40     55.60 / 72.01   1.04 / 3.36   9.78 / 8.74
  d1 t1     270 / 391    20 / 24     41.96 / 46.35   0.31 / 0.40  15.37 / 19.60
  d1 t2     214 / 297    43 / 51     50.07 / 57.25   0.96 / 1.50  32.24 / 36.25
  d2 t2     400 / 466    50 / 58     50.19 / 44.85   1.48 / 1.34 109.67 /125.03
```

More turning, less achieved by it.

### The turning tail, and the bracketing

A fourth story, that human corrections are sustained arcs, dies on the corpus
alone. Human signed heading change has lag 1 autocorrelation -0.363, strongly
ALTERNATING, with a median same sign run of one event. Most human turning is
cancelling micro jitter exactly like a model's. The structure is in the tail.

`research/w4_turnruns.py` measures that tail. `organis` is the fraction of all
absolute heading change carried by same sign runs longer than two events.

```
  arm                     acf1   runMu    p90    p99    max  shrLong   organis   netP90
  human                 -0.363   1.424      2      6     19   0.0777    0.1775    3.387
  masked served         -0.414   1.425      2      5     22   0.0806    0.1382    2.823
  ar event_ar_v1        -0.324   1.566      3      7     36   0.1089    0.2193    3.166
```

The masked model makes the right NUMBER of organised runs and they achieve less,
0.1382 against 0.1775. `event_ar_v1` makes too many and they carry too much,
0.2193. The two models bracket human from opposite sides.

That is the third independent statistic this session where they do. Still share:
masked 0.1567 high, human 0.0948, AR 0.0621 low. Pause rate: masked 0.24
matching, human 0.23, AR 0.05 low. Organised turn share: masked low, AR high.
Neither model has a shared architectural blind spot. Each is roughly twenty to
twenty five percent off, in different directions, on statistic after statistic,
and no single one of them prices above 0.05 against a gap of 0.15. That is the
shape of the problem and it is why every hand picked channel has priced small.

### The auditor

L asked how we know we are not fitting the detector instead of building
something human like. The structural answer is that nothing here has ever been
trained against it, all training is likelihood on real recordings, and the two
approaches that would have trained against a detector are closed. The real risk
is the research loop, which is a slow hand applied gradient on the scorer.

`research/w4_audit.py` is the control. It shares nothing with `scoring.py`: no
resample to 125Hz, no eighteen features, no forest. It fits order 2 Markov
models over coarsened token streams, one on human and one on generated, and
reads the held out likelihood ratio. Two guards are built in. A length only arm,
because if length separates then any result may just be length. And a shuffled
order arm, which must fall toward chance or the instrument is reading marginals
rather than sequence structure.

```
  arm                   alphabet      auc   lenAuc  shufAuc
  masked served                s   0.6940   0.6601   0.6761
  masked served               th   0.6992   0.6601   0.6745
  masked served             s,th   0.6771   0.6601   0.6420
  ar event_ar_v1               s   0.5399   0.5047   0.5110
  ar event_ar_v1              th   0.5802   0.5047   0.5387
  ar event_ar_v1            s,th   0.5591   0.5047   0.5280
```

On `event_ar_v1` both guards pass. Length alone is 0.5047, chance, so the
instrument is not reading length. Shuffling order drops `th` from 0.5802 to
0.5387, so it is reading order, with a genuine sequence structure excess of
0.0415. And it separates at 0.58. The non humanness is real and not an artifact
of the lens, which is the answer to the question.

The masked rows carry a second finding. Its event COUNT alone separates at
0.6601, worse than the whole contract score, and the contract scorer is
structurally blind to it because it resamples to a fixed rate before measuring
anything. `event_ar_v1` does not have this defect at all. A real defender
watching a raw event stream would see it immediately. This is a concrete reason
to build on the AR line rather than the masked one, and it would never have
surfaced through `scoring.py`.

`research/w4_audit.py` must NEVER be used as a training signal. Its only value
is being the thing nobody aimed at.

### The decision this sets up

`training/train_events_polar_sfmmd.py` computes its cost from
`features.extract_feature_matrix`, the contract's own function. It is a fine
tune that pushes the model's output distribution toward human ON THE DETECTOR'S
EIGHTEEN FEATURES. It would very likely move the number. It is not disqualified
by the mandate, since it still returns one trajectory with no selection, but it
is training on the lens.

The baseline above makes that decidable on evidence. Run it, then re run the
auditor. Contract score improves and the auditor holds or improves, the gain is
real. Contract score improves and the auditor gets worse, the lens was fitted
and the honest report is the auditor number, not the headline.

Baseline to beat, `event_ar_v1`: contract 0.6668, auditor th 0.5802, auditor
order excess 0.0415, auditor length 0.5047.

### The pair the partial correlation pointed at carries nothing, `w4_dvjoint`

`w4_partial` gave a claim sharp enough to be wrong, so it got tested. If the
model is human conditional on duration and mean velocity, the whole residual
gap should live in the joint distribution of those two. Fix that one bivariate
distribution and leave every other column as the model made it, and the score
should collapse.

It does not. Every arm below is scored against its own null, because the rank
transform itself moves the number and comparing a transformed arm to an
untransformed null is exactly the error corrected above.

```
  arm                 cols      auc     null   excess
  baseline              18   0.6668   0.4889   0.1780
  pair only              2   0.4939   0.4987  -0.0048
  pair marginals        18   0.6683   0.5036   0.1648
  pair joint            18   0.6706   0.5125   0.1581
  pair joint, cols       2   0.4527   0.4898  -0.0370
  all marg + joint      18   0.6298   0.5304   0.0994
  all but pair          18   0.6179   0.5185   0.0995
```

Making p(mean velocity | duration) exactly human buys 0.020 of a 0.178 gap. On
top of all eighteen marginals it buys exactly zero, 0.0995 against 0.0994.
Matching the other sixteen marginals and leaving the pair completely alone gets
the same 0.0995 that matching all eighteen gets. The pair contributes nothing,
jointly or marginally.

So the eighty five percent collapse in `w4_partial` is real as a statement about
rank correlations and does NOT carry the detector's signal. Two features can
mediate every pairwise rank correlation in the matrix and still hold none of
what separates the samples. This is the fourth hypothesis to die this session
and the fourth time the defect has refused to sit in any low dimensional
summary anyone can name.

That is now the finding, stated positively. Marginals are 44 percent and every
one of them is individually near human. Coupling is 56 percent and it is not in
any pair, not in the pair that mediates all the rank structure, and not in any
single feature. It is distributed. Every hand picked channel this programme has
priced has come back small for the same reason, and the accumulating evidence is
that there is no channel to find, only a model that is about twenty percent off
on many things at once.

Two consequences for anyone continuing. First, stop pricing channels. Nine have
now been priced and the largest was 0.05 against a 0.178 gap. Second, the thing
that actually moves a distributed twenty percent error is capacity and data on
the likelihood objective, which is the unglamorous answer the evidence has been
pointing at for some time.

## The model is starving, not memorising, and the 8GB card is a batch limit rather than a size limit

Nine channels priced small, four hypotheses dead in a day, and `w4_dvjoint`
showing the defect sits in no low dimensional summary anyone can name. At that
point the remaining move is the unglamorous one, and the only question worth
asking first is which unglamorous one, because more capacity and more data need
opposite responses and neither was ever measured.

### Held out likelihood, `w4_arfit`

`train_event_ar.py` kept no validation split and saved only a training loss EMA,
so `event_ar_v1.pt` could not answer this on its own. It is recoverable anyway:
the training subset is drawn with a hardcoded `default_rng(123)`, so the exact
split can be reconstructed after the fact. 1,500,000 of 4,028,855 trajectories
were trained on for 3.41 epochs and 2,528,855 were never seen once.

Both sides scored with the trainer's own loss, dropout off, 40,000 rows each.

```
  arm                s        th       dt     total
  trained on    2.0663    1.1683   1.0736    4.3081
  held out      2.0726    1.1781   1.0819    4.3326
  gap          +0.0063   +0.0099  +0.0083   +0.0245
```

0.0245 on a total of 4.31 is 0.57 percent, and the gap is the same near zero on
all three streams separately. The model has memorised nothing. It is capacity
and optimisation limited, and the 2.5M unused trajectories were never the
constraint. More parameters and more steps are the lever, and more data is not.

READ THE LIMIT OF THIS RESULT. It establishes headroom on the likelihood
objective, which is what the model optimises. It does NOT establish that a lower
loss produces a lower contract AUC. That link is assumed, it has never been
measured in this programme, and it is the thing the scaled run is instrumented
to answer.

### What the card will actually take, `w4_arbench`

The first attempt to price a scaled run failed, and how it failed was the useful
part. At d_model 512, 12 layers, 45.55M parameters, batch 128, the card sat at
7,923 of 8,188 MiB reporting 100 percent utilisation at 34W and took over six
seconds a step. A 5.7x parameter increase producing a 44x slowdown is not
compute scaling. Under WSL2 an oversubscribed allocation spills to host memory
instead of raising an error, so the symptom of running out of VRAM here is
silent slowness, not a crash.

Timed on synthetic batches of the right shape so data loading is excluded:

```
  d_model:layers:d_ff:heads:batch   params   s/step  alloc  reserv  hrs/40k
  256:8:1024:4:128                    7.9M    0.158   3422    3548      1.8
  384:10:1536:6:128                  21.7M    0.317   6348    6424      3.5
  512:12:2048:8:128                  45.6M    7.012  10210   10346     77.9
  512:12:2048:8:64                   45.6M    0.294   5520    5544      3.3
```

The third and fourth rows are the same model. It wants 10,210 MiB on an 8,188
MiB card at batch 128 and 5,520 MiB at batch 64, where it runs 24 times faster.
So 8GB constrains the BATCH, not the model size, and 45.6M is reachable in 3.3
hours. That is the opposite of the conclusion the first probe suggested.

Two operational notes that cost time here. Piping a long running job through
`tail` buffers everything until the pipeline ends, so a timeout kill produces a
completely silent log; write to a file and read it instead. And `alloc` under
the card limit is not sufficient, because `reserved` near the limit means the
run is one fragmentation event from thrashing.

### The first scaling run, `event_ar_v2`

Chosen config is 384:10:1536:6 at batch 128, 21.7M parameters, 40,000 steps,
n_train 1,500,000. Every one of those numbers except the model size is v1's,
including the `default_rng(123)` subset, so this varies capacity and nothing
else. 45.6M at batch 64 is affordable too and is the obvious follow up, but it
changes batch size and sample budget at the same time as capacity and would not
isolate anything.

Three changes to `train_event_ar.py` that this run needed, all of them fixing
gaps that cost time today:

  `batch_losses`     the CE computation factored out of the training loop so
                     training and validation cannot drift apart
  `--val-every`      held out loss during training, drawn from the rows the
                     `n_train` selection did not take. v1 recorded nothing and
                     `w4_arfit` had to reconstruct the split to answer a
                     question that should have been a logged column
  `--snapshot-every` numbered checkpoints. Without them a scaling run yields
                     exactly one point, and the open question is the SHAPE of
                     loss against AUC, which needs a curve

Set to snapshot every 5,000 steps and validate every 2,000, giving eight models
and twenty validation points. Baseline to beat is v1's held out total of 4.3326
and its contract AUC of 0.6668.

Measured rate is 0.33 s/step against the benchmark's 0.317, so the synthetic
timing held up once real data loading was included, and the run is about 3.8
hours with validation. First validation point, step 2000: held out 5.4911
against a training EMA of 5.4112. Do not read the gap early, the EMA lags a
model that is still moving fast at peak learning rate.

This machine has bluescreened under sustained load, so the run is never left
unguarded: a watchdog samples the card every 20 seconds and kills training at
83C. Steady state on this config is 69C at 70W, against 48C idle.

The result that matters is not whether the loss falls. It will. It is whether
the contract AUC follows it down, measured across the eight snapshots. If loss
improves and AUC does not, that is a more important finding than any score and
it should be reported as the finding rather than answered with another config.

## The 21.7M model beats the 7.95M model on held out likelihood, and the in-training validation number is not the number to compare

`event_ar_v2` ran 384:10:1536:6 at batch 128 for 40,000 steps on n_train
1,500,000, holding v1's sample budget, batch size, step count and
`default_rng(123)` subset fixed so capacity is the only variable. 21.67M
parameters against v1's 7.95M. 3.7 hours, 73C peak against an 83C kill limit,
8 numbered snapshots, 20 validation points.

### Read the two rulers separately or get the answer backwards

Held out loss logged during training ended at 4.4024 against v1's `w4_arfit`
number of 4.3326, which reads as the bigger model losing. It is not the same
measurement. `validate()` averages per batch means with equal weight per batch
over 20,000 rows drawn with `default_rng(7)`. `w4_arfit` weights by supervised
token count over 40,000 rows drawn with seed 0. Two rulers, one offset of
roughly 0.16, and no license to subtract one from the other.

`w4_arfit` run identically on both checkpoints, both in eval mode, same 40,000
rows, same seed:

```
  arm             v1 7.95M    v2 21.7M     delta
  trained on        4.3081      4.2142   -0.0939
  held out          4.3326      4.2458   -0.0868
  gap              +0.0245     +0.0316   +0.0071
```

The bigger model is 0.0868 better on trajectories it has never seen, and its
generalisation gap is 0.75 percent of its total. It memorised nothing either.
So capacity is the lever ON THE LIKELIHOOD OBJECTIVE, and 2.7x capacity buys
about 0.087 nats at fixed data and fixed steps.

The in-training curve is still valid against itself, and it says the run was
not finished:

```
   2000  5.4911     12000  4.6324     22000  4.5073     32000  4.4195
   4000  4.9977     14000  4.6000     24000  4.4895     34000  4.4124
   6000  4.8321     16000  4.5753     26000  4.4677     36000  4.4056
   8000  4.7693     18000  4.5459     28000  4.4452     38000  4.4037
  10000  4.6995     20000  4.5159     30000  4.4335     40000  4.4024
```

Monotone down at all 20 points, still falling at the last one, and the gap
against the training EMA sits around 0.09 to 0.11 throughout without widening.
A model being hurt by its own size turns upward. This one never does. It
decelerates because the cosine schedule anneals the learning rate to 1.2e-06 by
step 40,000, a horizon chosen for a model a third its size. So 4.2458 is where
the schedule stopped it, not where it converged, and a longer run at the same
capacity is unpriced rather than exhausted.

### What this does not establish

Nothing here says a lower likelihood produces a lower contract AUC. Those are
different objectives and the link has never been measured in this programme.
The 8 snapshots exist for exactly that and `w4_arcurve` is the instrument.
Until that curve is in, "capacity is the lever" is a claim about the training
loss and must not be quoted as a claim about the score.

An operational note on cost. AR sampling at `--batch 500` sits at 7,872 of
8,188 MiB drawing 26W where training drew 70W, and each 2,000 trajectory
evaluation takes over 12 minutes, so a 9 checkpoint curve is 90 minutes. Two
candidates, an allocation near the card limit and a launch bound sequential
decode, and they are distinguishable only by measurement that does not contend
with the running job. Price it before the next curve. This measurement gets
repeated after every scaling run, so the cost compounds.

## AR sampling was 19x slower than it needed to be, and the card never said so

`w4_arcurve` was killed after 30 minutes on its first of 9 checkpoints, which
put a single curve at 4.5 hours. `w4_sampcost` priced why. `event_ar_v1.pt`,
one `model.sample` call per batch size, timing only:

```
  batch   s/call  traj/s   alloc  reserv  hrs/9ckpt@2000
    500    427.2     1.2    2312   22936            4.27
    250    176.3     1.4    1244   22762            3.53
    125     21.0     6.0     714    8162            0.84
     64      5.3    12.2     431    1956            0.41
     32      1.4    23.2     256     294            0.22
```

Throughput rises monotonically as batch FALLS, which no compute bound workload
does. The `reserv` column is the tell: 22,936 MiB reserved on an 8,188 MiB card
against 2,312 MiB actually live. The job never needed the memory. `sample`
keeps no KV cache and re-runs the trunk over the prefix, so it requests a
slightly larger workspace at each of 256 steps and the caching allocator ends
up holding hundreds of block sizes it cannot reuse. Past the card limit that
spills to host memory silently, exactly as `w4_arbench` found for training.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` addresses the fragmentation
directly rather than dodging it:

```
  batch   s/call  traj/s   alloc  reserv  hrs/9ckpt@2000
    500     43.3    11.5    2312    2468            0.43
    125      9.3    13.4     652     720            0.37
     64      4.2    15.1     406     460            0.33
     16      1.1    14.7     215     238            0.34
```

Reserved at batch 500 falls from 22,936 to 2,468 MiB and throughput goes flat
in batch size, which is the signature of a workload that is no longer memory
bound. `w4_arcurve` now sets the env var and defaults `--batch 64`, passed
through to `w4_ar_eval`. Batch 32 unflagged nominally read 23.2 traj/s, higher
than anything flagged, but the readings scatter about 30 percent, and the flag
is what protects a 21.7M model at a batch size that was only ever measured safe
for a 7.95M one.

Two things to carry forward. Any AR generation on this card gets the env var,
not just this script. And sampling is still doing roughly 128x the necessary
arithmetic because there is no KV cache; a cache is the real fix and is worth
maybe another order of magnitude, but it changes generation itself and must not
be written while generation is the thing under measurement.

## Likelihood and the contract are tightly linked, and that is what closes the scaling road

This is the measurement `event_ar_v2` was built to produce and the one the
programme had never made. `w4_arcurve`, 8 v2 snapshots plus v1, identical n
2000, seed 0, temp 1.0, batch 64 for every row.

```
  checkpoint          step   trainEma   heldOut   contract   missP50   nEvP50
  event_ar_v1        40000     4.4199    4.3326     0.6573       5.0       40
  event_ar_v2_s5000   5000     4.7939    4.9977     0.7589       9.9       44
  event_ar_v2_s10000 10000     4.6130    4.6995     0.7338       7.0       44
  event_ar_v2_s15000 15000     4.4806    4.6000     0.6861       5.5       40
  event_ar_v2_s20000 20000     4.4287    4.5159     0.6875       5.5       40
  event_ar_v2_s25000 25000     4.3926    4.4895     0.6571       4.8       38
  event_ar_v2_s30000 30000     4.3234    4.4335     0.6655       4.6       40
  event_ar_v2_s35000 35000     4.3052    4.4124     0.6505       4.5       39
  event_ar_v2_s40000 40000     4.3274    4.4024     0.6526       4.4       39
```

IGNORE THE SCRIPT'S OWN SUMMARY LINE. It reports `4.3326 -> 4.4024` and a
slope of -0.0675 by taking v1's `heldOut` from `w4_arfit` and v2's from the
in-training `validate()`. Those are the two rulers with the ~0.16 offset. The
only valid regression is across the 8 v2 rows, which share one ruler.

```
  arm                          slope AUC/nat     r    resid sd
  all 8 v2 points                    +0.1904  +0.953    0.0131
  step >= 20000 only                 +0.2341  +0.772    0.0110
```

r 0.953 over a 0.6 nat span. The link between the likelihood objective and the
contract scorer is real, tight, and now measured rather than assumed. Anything
that lowers held out loss lowers contract AUC at roughly 0.19 per nat.

### The exchange rate is what kills scaling

0.6526 to 0.50 is 0.65 to 0.80 further nats at the measured slopes. `w4_arfit`
priced capacity on one ruler across both models: 2.7x bought 0.0868 nats. Held
flat, that is 7.5 to 9.2 further 2.7x steps, so 1.7e3x to 9.6e3x capacity, so
1.4e10 to 7.6e10 parameters. For a cursor.

The endpoints say the same thing without any extrapolation. v1 contract 0.6573,
v2 contract 0.6526, a delta of 0.0047 against a noise band of 0.03. Tripling
capacity bought 0.16 of one noise unit. Taking those two points alone gives
0.054 AUC/nat, three times WORSE than the within-run slope, which is expected:
inside a run "lower loss" and "more trained" are the same variable, so 0.19 is
an upper bound on what capacity specifically buys.

Meanwhile arrival miss went 9.9px to 4.4px across the same span and n_events
44 to 39. The model got materially better at the task. The detector did not
move. That gap is the whole result.

### The independent auditor agrees, once it is asked the same question

`w4_audit` on v1 was on file at n 1500. Compared against v2 at n 2000 it looked
like v2 had shrunk its order signal by two thirds. It had not. Re-run at
matched n 2000, seed 0:

```
  alphabet   v1 real   v2 real   v1 shuf   v2 shuf   v1 len   v2 len
  s           0.5334    0.5367    0.5379    0.5314   0.5035   0.5011
  th          0.5917    0.5947    0.5656    0.5776   0.5035   0.5011
  s,th        0.5784    0.5625    0.5521    0.5506   0.5035   0.5011
```

Identical within noise on every row. Third time in one session that an
unmatched instrument manufactured a difference that a matched one erased.

Two things the auditor adds that the contract scorer cannot see. `len` sits at
chance for both, so the AR family does not carry the masked model's event count
defect (0.6601). And `real` minus `shuf` is 0.005 to 0.026, so under this lens
almost all residual detectability is in the per token MARGINALS, not in
sequence order. The AR models are 0.53 to 0.59 here against 0.65 on the
contract, which is not a contradiction, it is two lenses.

### What is now closed and what that leaves

More data was already excluded: `w4_arfit` showed no memorisation with 2.5M
trajectories unused. More capacity is now excluded at any price this programme
could pay. Those were the two brute force roads. Neither is a matter of
patience or budget, and neither should be reopened without a NEW argument, not
a bigger run.

What survives is the only remaining lever that could move held out loss by 0.7
nats: the representation and the factorisation itself, meaning what the tokens
are and what the model is asked to predict at each step. The tight r 0.953 is
the reason this is worth doing rather than a reason for despair. It says the
likelihood objective is a valid, cheap proxy, so a structural change can be
screened on loss in hours without generating a single trajectory.
