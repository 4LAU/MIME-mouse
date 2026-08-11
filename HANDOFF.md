# Handoff

The single session-start document for this repo. Read it before touching
anything. Repo is MIME-mouse on WSL2 Ubuntu, branch main. Nothing is
mid-flight; the GPU is idle and no runs are pending.

Earlier handoffs and plans are in archive/ with a note on why each was
overtaken. They are not instructions.

## Start here: where things stand, 2026-08-10

The mandate is research, not shipping. The goal is a generative model that
returns ONE trajectory from A to B and scores 0.50. Anything that generates
several candidates and picks among them is out of scope, so the selection
product and its 0.58 are not the number being moved.

  standing single-trajectory number   0.6215
  model                               models/event_ar.py, the autoregressive
                                      event model, checkpoint
                                      training/event_ar_v2_s40000.pt
  what it means                       one path per request, decided before it
                                      is scored, no candidate pool
  the floor to read it against        0.5330. That is the recorded human
                                      TRAINING corpus scored through the same
                                      tokens and the same decoder against the
                                      scoring reference set, so it is what a
                                      perfect model of this corpus would read,
                                      not 0.50
  scorer                              research/autoloop/scoring.py, NOT
                                      evaluate.py (see Evaluation below)

The 0.6986 that stood here until 2026-08-07 was the earlier masked-token
family. It is superseded by the autoregressive event model above and the two
numbers are not comparable: they use different checkpoints and the older one
carried an endpoint correction this line does not.

Read the last section of this file before anything else. The short version is
that the model's per step behaviour is not the problem and has now been shown
exact on all three heads. Teacher forced on real human histories the speed,
turn and timing heads reproduce the true conditionals to three or four decimal
places. Free running, the same model is 1.68 times wrong on the sub
millisecond wait rate, and the result is a coherent displacement of the whole
eighteen feature vector worth 0.107 paired.

That retires per step diagnostics as a source of yield. Four consecutive
investigations found a genuine token level defect, fixed or priced it, and
moved the contract by nothing or close to nothing. The remaining error is made
by free running composition, so the next arm has to be one whose objective sees
the model's own rollouts.

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

## Duration obedience is not the defect, and the earlier claim that it was is withdrawn

`w4_whatsees` measured `movement_duration` at 0.5699 on `event_ar_v1`, the
largest single feature tell in that panel. Earlier in this session that got read
as the model being handed a correct duration and producing something else. That
reading was wrong and it is the same trap this document already records under
`w4_dvjoint`: an RF OOB AUC on one feature is a ranking WITHIN that
dimensionality and says nothing about whether the underlying quantity is
correct. A feature can be the most detectable one available while still being
nearly right, if everything around it is more right.

Two things closed this without spending a training run.

### The command is human by construction

`experiments/_common.py`, `DurationModel`, under `DUR_EMPIRICAL=1`, which the
locked serving recipe sets. The commanded duration is not sampled from a fitted
distribution. It is drawn from `self._bin_durs[bin_idx]`, an actual reservoir of
up to 20,000 real human log durations from the matching log distance bin, with
0.02 of jitter on top. So the instruction the model receives is a real human
duration for a real human distance. The command cannot be the fault, and any
proposal to model the duration command better is proposing to replace human data
with a fit of human data.

### The model obeys the command to about one percent, `w4_durobey`

n 2000, `event_ar_v2_s40000.pt`, one trajectory per spec, no selection.

```
  commanded vs realized duration
    correlation in log space        0.9986
    median realized / commanded     1.0003
    p10 / p90 of that ratio         0.9910 / 1.0136
    median commanded / realized s   0.4018 / 0.4030

  arm               contract  dur_only      n
  as_served           0.6413    0.5359   2000
  time_rescaled       0.6391    0.5481   2000
```

`time_rescaled` is an ORACLE and is not a proposal. It uses the model's own
realized duration to compute the exact per sequence factor that would make the
realized duration equal the commanded one, applies it to every dt, and rounds
back onto the whole millisecond lattice. It is also a decode time repair, and
decode time repairs are closed, so it exists only to price the ceiling.

That ceiling is nothing. Perfect duration obedience moves the contract by
-0.0022, a fourteenth of the plus or minus 0.03 band, and pushes duration alone
the WRONG way from 0.5359 to 0.5481 because the rescale perturbs a lattice the
features are sensitive to. Note also that v2's own duration alone figure is
0.5359, not v1's 0.5699.

Duration is closed. No training run should aim at it, and no future reading of a
single feature AUC should be treated as evidence that the underlying quantity is
wrong.

## Where the defect lives: local conditional or compounding drift

This is the fork that decides what the next training run is for, and the two
answers need opposite responses.

LOCAL. `p(next token | history)` is wrong everywhere by a little, and a real
human history buys the model nothing. Then the lever is the objective or the
factorisation, and exposure to its own output is irrelevant.

COMPOUNDING. The conditional is close to right, but the model conditions on its
own past and small errors accumulate over a 256 step sequence. Then the lever is
a training time exposure fix, and no amount of objective work touches it.

`models/event_ar.py` gained a `force` argument on `sample()` to make this
measurable: four tensors, three class streams and a bool mask, and wherever the
mask is set the token comes from those streams instead of the model's own head.
It is applied AFTER the heads run, so the sampled token is simply discarded, and
BEFORE the done mask, so a forced PAD terminates its row exactly as a generated
one would. With `force=None` every line runs as it did before. That was
deliberate: generation is the thing under measurement and must not move while it
is being measured.

### The first attempt cannot answer it, `w4_prefix`

Forces the first k tokens of a real held out human sequence and lets the model
generate the rest. n 2000 drawn, 1795 used at length at least 12.

```
       k   placement  contract      n  frac forced
       0      prefix    0.6337   1794       0.0000
       2      prefix    0.6405   1795       0.0352
       4      prefix    0.6234   1795       0.0703
       8      prefix    0.6037   1795       0.1407
      16      prefix    0.6028   1795       0.2779
      16   scattered    0.8120   1795       0.2779
      32      prefix    0.5845   1795       0.4956
      64      prefix    0.5654   1795       0.7528
      64   scattered    0.6177   1795       0.7528
     128      prefix    0.5445   1795       0.9372
    full         all    0.5589   1795       1.0000
```

The prefix column falls monotonically and any single step is inside the noise
while a slide that consistent across a 64 fold range is not. It is still
unreadable, for two reasons.

The passthrough control missed. k full never consults the model and is the exact
operation `w4_token_ceiling` measured at 0.5118. Here it reads 0.5589, a 0.047
miss on a 0.03 band. The likely cause is this run's length at least 12 filter,
which removes short flicks and hands the detector a population tell that has
nothing to do with the model. That inflation is common to every row, so the
shape survives it, but the instrument is not the one it was supposed to be.

The worse problem is that the sweep partly measures itself. `frac forced` rises
with k, so at k 64 against a median length of 44 about three quarters of the
scored output IS real human data. The falling curve conflates a better
conditioned model with less model. The scattered arm was built to break exactly
that tie and cannot, because splicing real tokens into mid flight positions is
destructive on its own: 0.8120 against the prefix arm's 0.6028 at an identical
forced fraction. Position matters a great deal. The direction is not readable
off this run.

Do not rerun `w4_prefix` and do not cite its curve as evidence of compounding
error. It is inconclusive by construction, in the same category as
`w4_prefixcond`.

## The fork resolves to both, and compounding error is the smaller half

`research/w4_suffix.py`, run `w4_2026-07-29T074350+0000_017f236c`. This is the
clean replacement for `w4_prefix` and it answers the local versus compounding
question that run could not.

The design change is that nothing human is inside what gets scored. Every arm
takes a real held out sequence, cuts it at event k, and scores ONLY what comes
after the cut. Three arms share the same cut and the same rows:

  self    the model generated the whole thing, so its history at the cut is its
          own output
  forced  the model was handed the real first k tokens and generated from there,
          so its history at the cut is genuinely human
  human   the real continuation

self minus forced is the entire question. Nothing else differs between those
two arms.

The readout is NOT the contract. The contract's class 0 is a fixed file of whole
point A to point B movements, which is the right reference for a whole synthetic
movement and the wrong one for a fragment: a mid movement fragment does not
start from rest and does not end where it was aimed. A first pilot scored real
human fragments against the contract at 0.8309 whole and 0.9216 truncated. The
detector was already at near certainty on genuine human data, which leaves no
room to see a difference between arms, so a null would have meant nothing. The
readout here is instead a suffix against suffix two sample test: the real
suffixes split in half, one half is class 0 for all three comparisons, the other
half is the human probe. Identical class sizes, identical reference, same kind
of object on both sides. `scoring.py` is untouched and is not in this path.
NEVER ledger these numbers as contract scores.

10,000 drawn from the 2,528,855 the `default_rng(123)` training subset never
selected, 4,091 rows with at least 48 events, median length 76.

```
      k    probe  whole suffix      first 16  n/class
      8human (floor)        0.4832        0.4865     1825
      8     self            0.6057        0.5578     1825
      8   forced            0.5846        0.5251     1825
             gap           -0.0211       -0.0327           dropped 441

     16human (floor)        0.5104        0.5077     1715
     16     self            0.6293        0.5637     1715
     16   forced            0.6092        0.5267     1715
             gap           -0.0201       -0.0370           dropped 661

     32human (floor)        0.4900        0.4832     1435
     32     self            0.6275        0.5577     1435
     32   forced            0.5898        0.5160     1435
             gap           -0.0377       -0.0418           dropped 1220
```

The floor rows land at 0.4832 to 0.5104, so the reference and probe halves do
not differ for reasons of their own and the model rows are readable.

Four findings.

Compounding error is real. self minus forced is negative in all six cells, and
in the truncated column it clears the floor's own distance from 0.5 at k 16 and
k 32. Handing the model a genuinely human history makes its next events more
human. This is the first clean evidence of that in the whole workstream.

It is the smaller half. Measured on the first 16 generated events at k 32, self
sits 0.0745 above its floor and forced sits 0.0328 above the same floor. Real
history removes 56 percent of the near horizon defect and leaves the rest. Over
a whole suffix it removes 27 percent. Whatever fraction an exposure fix could
recover, it is bounded by those numbers and it is not the whole gap.

It saturates by k 8. Thirty two real events do not buy meaningfully more than
eight. `forced` above floor on the whole suffix is flat at 0.1014, 0.0988 and
0.0998 across the three k, and `self` above floor is flat too. The model reaches
a wrong steady state fast and stays there rather than diverging further. Any fix
premised on runaway divergence is aimed at something that does not happen.

A large local component survives any amount of clean history. Even generating
from a real 32 event human opening, the model's next 16 events read 0.5160
against a 0.4832 floor and its whole continuation reads 0.5898 against 0.4900.
The per step conditional is wrong on its own.

One reading trap. Do NOT compare magnitudes in the whole column against the
truncated column. A longer sample is easier for a classifier to separate at a
constant per step error rate, so the whole column reading higher than the
truncated column is expected and is not evidence that damage grows with horizon.
Comparisons are only ever within a column.

What this rules in and out. An exposure fix at training time is now justified by
evidence rather than by analogy, but it is priced: it attacks at most half of one
component. It should not be sold as the thing that closes the gap. The residual
local defect is the larger target and it is not resolution, not capacity, not
data, not duration obedience and not the pause channel, all of which are closed.

The measurement that would price the exposure family properly is a drift rate
curve: hold the history at a real 32 events and sweep how many generated events
get scored, 4, 8, 16, 32. The floor arm controls for length at each point and no
human material sits in the scored window. A flat curve means the defect is
locked in at the first few events and no exposure fix helps. A rising curve
means error accumulates and gives the accumulation rate directly. Roughly one
generation pass on the existing script.

## The exposure fix is priced and it is not the answer

`research/w4_drift.py`, run `w4_2026-07-29T082724+0000_dd47fc09`. `w4_suffix`
appeared to justify a training time exposure fix. This run prices it before any
GPU goes into one, and the price is bad.

The history is held at a real 32 events and the only thing that moves is how far
ahead we score: 4, 8, 16 and 32 events after the cut. Identical rows and
identical generations at every m, 20,000 drawn, 5,849 with at least 64 events,
3,635 surviving every arm at the longest window, median length 95.

Raw AUC rises with m for a reason that has nothing to do with the model: a
longer sample is easier for a classifier to separate at a constant per step
error rate. So the headline is a ratio,

  r(m) = (forced(m) - floor(m)) / (self(m) - floor(m))

which is the fraction of the free running defect that SURVIVES being handed a
genuinely human history. Both terms carry the same information per sample factor
and it divides out to first order. r near 0 means a real history fixes the
model, r near 1 means it buys nothing.

```
     m    floor     self   forced   self-fl   forc-fl       r   spread      n
     4   0.4977   0.5120   0.4983    0.0142    0.0006    0.04   0.0052   1590
     8   0.4899   0.5379   0.5235    0.0480    0.0336    0.70   0.0045   1744
    16   0.5154   0.5630   0.5424    0.0476    0.0271    0.57   0.0092   1807
    32   0.4964   0.5801   0.5681    0.0837    0.0718    0.86   0.0152   1817
```

The floors land at 0.4899 to 0.5154, so the instrument is valid.

The m 4 row is at the detection floor and MUST NOT carry the conclusion. Its
whole signal is a self above floor of 0.0142 against a spread of 0.0052. Over
four events there is barely anything for the classifier to see in any arm, so
r = 0.04 there is not evidence that a clean start briefly makes the model
perfect. It is evidence that four events is too short a window to measure.

Over the readable range, m 8 and up, r sits at 0.57 to 0.86. Propagating the
spread through the ratio gives an uncertainty near 0.27, so those three points
are not distinguishable from each other and no trend with horizon is
established. The defensible statement is a level, not a curve: a genuinely human
history removes roughly 15 to 45 percent of the defect and the rest is there
regardless.

Put that beside `w4_suffix`'s saturation result, where thirty two real events of
history bought no more than eight, and the picture closes. Compounding error is
real and it is a minority contributor. Scheduled sampling, DAgger style
exposure, and every other training time fix in that family attack at most a
third of the gap, and they cost a day of GPU each. DO NOT fund one as the thing
that closes the gap. If one is ever run it must be sold as a partial and
measured against this ceiling.

What that leaves is the local conditional, and the evidence now says the same
thing from two directions. `w4_couplemap` and `w4_copula` put coupling at
roughly 56 percent of the feature level gap against 44 percent for marginals.
This run says the defect is present within the first several generated events
even when everything before them is genuinely human. Both point at what one step
of the model does, not at what a hundred steps of it accumulate.

One suspect is not yet closed and is worth a cheap look before anything is
trained. Within an event the model already factorises correctly, s then theta
given s then dt given s and theta, so the chain rule is intact and this is NOT
the composition failure that `models/event_ar.py` was built to fix. But the
mechanism carrying that conditioning is the weakest one available: an additive
context embedding into a LayerNorm, `th_head(th_norm(x + s_ctx_embed(s)))`. The
cond vector by contrast gets FiLM, a learned scale and shift. An additive
embedding before a linear head is close to a fixed logit offset per conditioning
class, so the within event dependence can be represented but only weakly, which
is exactly the attenuation signature `w4_attenuation` found at the feature level
on the masked model. Measuring it directly on the AR model at the TOKEN level,
rank correlations among s, theta and dt within an event and across adjacent
events, generated against real, is CPU work on a few minutes of generation and
would either open a concrete architecture change or close the last local
suspect. That has not been done.

## The couplings are the right strength, so the conditioning path is closed

`research/w4_coupletok.py`, run `w4_2026-08-04T192527+0000_a8e75877`. The
suspect raised at the end of the `w4_drift` section is now dead, and it is worth
recording that it was killed rather than quietly dropped.

The suspicion was that `models/event_ar.py` carries its within event
conditioning through the weakest mechanism in the network, an additive context
embedding into a LayerNorm before a linear head, while the cond vector next to
it gets FiLM. If that were starving the dependence, the model would have the
right shape and not the right strength, which is the attenuation signature
`w4_attenuation` found at the feature level on the older masked model.

The readout is Spearman rank correlation, which ignores each variable's own
marginal. `w4_redundancy` established the marginals are already right, so a
measure blind to marginals isolates what is left. Computed INSIDE each
trajectory over its motion events and Fisher z averaged across rows, so between
trajectory variation driven by the cond vector cannot contribute: that is
conditioning the model already gets and is not what was under test. Both arms
go through identical class to physical maps, so quantisation is not a difference
between them. 8,000 drawn, 5,020 human and 4,339 generated rows carrying at
least 24 motion events.

```
  pair                      human      se    model      se   ratio     diff
  speed x |turn|          -0.1932  0.0040  -0.2141  0.0040    1.11  -0.0209
  speed x dt              -0.1579  0.0034  -0.1503  0.0036    0.95   0.0076
  |turn| x dt              0.0660  0.0027   0.0707  0.0029    1.07   0.0046
  speed t x speed t+1      0.7866  0.0027   0.8064  0.0025    1.03   0.0199
  |turn| t x |turn|t+1     0.3364  0.0028   0.3115  0.0029    0.93  -0.0249
  turn t x turn t+1       -0.3929  0.0032  -0.3394  0.0037    0.86   0.0534
  dt t x dt t+1            0.0278  0.0058   0.0204  0.0067    0.74  -0.0073
  speed t x |turn|t+1     -0.1826  0.0041  -0.1954  0.0041    1.07  -0.0128
  |turn| t x speed t+1    -0.1723  0.0038  -0.1931  0.0038    1.12  -0.0208
  speed t x speed t+2      0.7244  0.0028   0.7349  0.0028    1.01   0.0104
  dt t x dt t+2            0.1500  0.0040   0.1592  0.0046    1.06   0.0093
```

Every sign matches human and the ratios scatter on both sides of 1. There is no
systematic attenuation. On several channels the model is slightly MORE coupled
than a person. DO NOT rewire `th_head` and `dt_head` to FiLM on this evidence.
The additive conditioning is adequate and that day of GPU is not owed.

The `dt t x dt t+1` ratio of 0.74 is not a finding. Its human value is 0.0278,
so the ratio is a small number divided by a small number and the module's own
closing note says not to read one there.

The instrument is not merely insensitive, and the proof is inside the table.
`turn t x turn t+1` comes back 14 percent short at about 15 standard errors, so
a shortfall of that size on any other channel would have shown. The caveat is
that this is an internal control on one channel and there is no external
positive control, so the sensitivity claim does not extend to shortfalls much
smaller than 14 percent.

### The one deficit is already on the record

Signed turn autocorrelation, human -0.3929 against model -0.3394, is NOT a
discovery. `w4_seqstats` and `w4_turnruns` already recorded human -0.363 against
`event_ar_v1` -0.324, and the section above them already says `ar_v1` "learned
the trend and not the alternation". This run confirms the same deficit on
`event_ar_v2` with a different instrument, a different row filter and a
different checkpoint, which is worth having as independent agreement and is
worth nothing as news.

Speed lag 1 autocorrelation points the same way from the other side, 0.8064
against 0.7866, model HIGHER. Higher autocorrelation is smoother. So both
readable deviations say the model is marginally too smooth at the highest
frequency, and both are in the twenty percent band that `w4_turnruns` already
identified as the size of every deviation this programme has found.

### Where that leaves the search

This closes the last local architectural suspect anyone has named. Combined with
`w4_drift` closing the exposure family and `w4_token_ceiling`, `w4_arfit`,
`w4_arcurve`, `w4_durobey`, `w4_stillprice` and `w4_dvjoint` closing resolution,
capacity, data volume, duration obedience, pauses and the duration velocity
copula, the position is the one `w4_turnruns` already stated and this run
independently reaches: each model is roughly twenty to twenty five percent off,
in different directions, on statistic after statistic, and no single hand picked
channel prices above 0.05 against a gap of 0.15.

The gap is diffuse. That is a finding about the shape of the problem, not a
failure to find the cause, and it is the reason every targeted fix has priced
small. Any next move that picks one more channel and fixes it should be expected
to buy about 0.01, and should be proposed with that number attached.

## The objective cannot see coherent structure, and nobody has ever looked at the spectrum, 2026-08-04

PRE REGISTERED. This section was written BEFORE the measurement it proposes was
run, so that the prediction and its falsifier are on the record ahead of the
number. The result subsection at the end is appended afterwards and says plainly
whether the prediction held.

### The correction this makes to the section above

The close above says "the gap is diffuse" and prices any further channel at
0.01. The first half of that stands: no single hand picked channel has priced
above 0.05, and that is now tested to exhaustion. The second half does not, and
the reason is a distinction that was collapsed.

Every instrument this programme has built measures a LOW ORDER quantity: a mean,
a spread, a pairwise rank correlation, an autocorrelation at lag one or lag two.
Dozens of them agree. But the contract is a RandomForest, and a forest is not a
moment matcher. It partitions the 18 dimensional feature space and finds a
region where one class piles up. A distribution can match every marginal and
every pairwise correlation and still put eight percent of its mass in a corner
where the other puts two. In high dimensions that is the ordinary case, not a
contradiction.

So "no single channel is broken" and "the gap is spread thinly everywhere" are
two different claims and the close above ran them together. Nothing measured so
far distinguishes them, because every instrument was one or two dimensional.

### The fact this rests on

`w4_seqstats` recorded, and the section at line 2790 states, that human speed
carries a lag 2 autocorrelation ABOVE its lag 1, 0.6220 against 0.5952, a ratio
of 1.045. `event_ar_v1` reads 1.010. The note at the time was that the model
"learned the trend and not the alternation", and it was filed and not pursued.

A monotonically decaying autocorrelation is what a diffusion or a smooth drift
produces. Human speed does not decay monotonically. A non monotone
autocorrelation is the signature of an OSCILLATORY component, something with a
preferred period riding on top of the smooth motion. The model reproduces almost
none of it.

`w4_coupletok` reaches the same place from the other side on `event_ar_v2`:
signed turn alternation -0.3394 against human -0.3929, and speed lag 1 0.8064
against human 0.7866, model smoother. Three instruments, two checkpoints, one
direction. The model is too smooth at the finest scale and only there.

### The mechanism, and why it is structural rather than a tuning miss

Cross entropy is a sum of per step surprises. A component's value to that
objective is its contribution to predicting the NEXT token. A small amplitude
oscillation is nearly worthless by that measure: knowing it barely changes the
next step's distribution, so it carries almost no gradient.

Its contribution to a statistic computed over the WHOLE trajectory is a
different size. A coherent component is phase locked, so its contributions add
linearly across steps and the total grows in proportion to the number of steps.
An incoherent component of the same per step amplitude adds in quadrature and
grows as the square root. Over a sixty step movement that is a factor of about
eight between how much the training objective values coherent structure and how
much any whole trajectory statistic does.

That is a mismatch between what the model is trained on and what it is scored
on, and no amount of data, capacity or architecture repairs it, because all
three make the model better at the objective and the objective is what is blind.
It is a mechanism, not a theorem, and the measurement below is what decides
whether it is operating here.

### What it explains that nothing else has

The reason to prefer it is that it accounts for the failures rather than merely
surviving them.

- Why the contract is a texture detector. Half a pixel of noise on genuinely
  human data reads 0.866, while a 3 point moving average, which changes the
  actual shape of the movement far more, costs 0.59. Fine texture is high
  frequency content. The 18 features are dominated by the top of the band, which
  is where a missing oscillation would sit.
- Why every attempt to add texture to a smooth path failed, three ways in one
  day, all moving the score the wrong direction: continuous noise added, a real
  leftover transplanted, integer steps chosen to match turn statistics. Noise is
  incoherent and spread across the whole band. A coherent narrow band component
  cannot be rebuilt out of it. Those were not three bad ideas, they were one
  wrong assumption tried three times.
- Why every marginal and all eleven pairwise couplings check out. A small
  oscillation moves means and spreads hardly at all, and rank correlation at lag
  one is a poor detector of a component whose period is longer than one step.
- Why the exposure family priced at a third. Compounding error is incoherent
  drift, a different object from a missing coherent component.
- Why more data and more capacity bought nothing.
- Why the served recipe leans on EVENT_CHOICE_TEMP=10. Extra randomness is
  standing in for missing structure, which is trading one wrong texture for
  another.

### Why this has never been looked at

There is no spectral analysis anywhere in this repository's research history.
`grep` for spectrum, spectral, fourier, fft, psd, periodogram and oscillation
across HANDOFF.md returns nothing. Every "band" in the record is a DISTANCE band
or a DURATION band. The only fft in the codebase is inside
`external_detectors.py`, which computes a spectral centroid as one of its five
channel features, so an independently written detector already treats frequency
content as discriminative.

The programme came at this as a statistics problem and built statistics. The
frequency domain is orthogonal to every measurement made so far, which is
exactly the property that would let a deficit survive all of them.

One methodological note that matters for the design. The lag 1 versus lag 2
finding above is on the EVENT INDEX, and the event stream has a non uniform time
base because every event carries its own dt. So its lag axis is events, not
milliseconds, and it is a distorted view of anything with a period in real time.
The measurement has to run on the contract's own 125 Hz resampled path, which is
the only axis with physical units and is also what the detector actually sees.

### The prediction, and what would kill it

Run: `research/w4_spectrum.py`. Power spectral density of speed and of signed
turn, human against model, on the 125 Hz resampled path, fixed length windows so
trajectory length cannot confound, human split in half as the instrument's own
floor.

  CONFIRMED   a band where human carries power and the model does not, narrow
              rather than spread, and outside the human against human floor.
              The mechanism is operating and the thing to build is named below.
  KILLED      the two spectra lie on top of each other within the floor, or the
              deficit is broadband. Broadband is ordinary under dispersion or
              over dispersion and is NOT this mechanism. Say so and drop it.

A deficit that appears only in the raw spectrum and vanishes once each row is
standardised is an amplitude difference, which `w4_redundancy` already covers,
and does not count. Both are computed for that reason.

### What gets built if it confirms, in the order it would be tried

First, the cheap one. Hand the model the phase. If human motion carries a
rhythm, the model currently has to recover where it is in that rhythm by
inference through a softmax that discards it. An explicit oscillator phase fed
in per step, at the frequency the measurement names, costs about a day and hands
the model the one thing the objective cannot teach it to represent. If the
mechanism is right and this does not move the number, that is itself
informative.

Second, the real one. Add a term to the loss that compares the frequency content
of what the model generates against the frequency content of human motion, and
train on it alongside cross entropy. This puts gradient exactly where cross
entropy is blind. It needs no second network, no adversary, and no generating
several candidates and selecting among them, so it stays inside the mandate. It
does not touch the scorer's 18 features. The obstacle is that the model emits
discrete tokens by sampling and sampling is not differentiable, so the spectral
gradient needs a relaxation to get back through.

Neither is authorised by this section. Both are gated on the measurement.

### THE RESULT. The falsifier fired, and the run found something else

Runs `w4_2026-08-04T220506+0000_f825e2fd` (serving decoder) and its
EVENT_SNAP=0 EVENT_ROUND=0 control. n=20,000 drawn from the never seen 2.5M,
18,049 human and 17,903 model rows decoded and resampled, about 7,800 centred
windows per arm at 64 samples, 1.95 Hz per bin.

KILLED, on the criterion written above before the number existed. The prediction
was a NARROW band where human carries power and the model does not. What the
standardised speed table shows is broadband, the same direction in about thirty
consecutive bins from 6 Hz to Nyquist. The pre registration names broadband as
ordinary dispersion and explicitly not this mechanism. It is not this mechanism.

The premise failed its own check as well, and in the direction opposite to the
one guessed. The non monotone lag 2 above lag 1 signature does NOT come from
splicing the ticks out. It comes from putting them in:

```
  series                        lag1      lag2      lag3     r2/r1
  human motion events only    0.6958    0.6379    0.5231    0.9168
  human every event           0.6260    0.6343    0.4947    1.0132
  model every event           0.6928    0.6436    0.5273    0.9291
```

So the signature tracks the alternation between moving and pausing, which is a
channel `w4_stillprice` already closed, and not an oscillation in the motor
signal. On the 125 Hz axis the contract actually reads it is absent altogether,
human 0.8254 / 0.7313 / 0.6332, cleanly monotone. There is no oscillation. The
mechanism argued above may still be true of sequence models in general, but it
is not what is happening here and nothing in this repository should cite it as
though the measurement supported it.

### The direction correction, which matters beyond this run

The record has said repeatedly, including in the `w4_coupletok` section directly
above, that the model is too SMOOTH at the highest frequency. Every one of those
statements is read off an EVENT INDEX statistic. On the 125 Hz time axis the
sign reverses:

```
  speed lag 1     human 0.8254     model 0.8034
```

Lower autocorrelation is rougher, not smoother. The model is too JITTERY on the
only axis the contract sees. The event index and the time axis disagree in sign
on this question because the event stream's spacing is itself a variable, so any
over smoothing claim resting on an event index number has to be re read on the
time axis before it is used. `w4_seqstats`, `w4_turnruns` and `w4_coupletok` all
need that caveat attached.

### What was actually found

Standardised, so a difference in overall amplitude cannot produce it, speed only:

```
  freq Hz     human     model    ratio    floor   se
    1.95      8.966     8.511    0.949    0.990   0.010
    5.86      2.898     3.063    1.057    1.025   0.019
   11.72     0.4469    0.5342    1.195    1.012   0.031
   17.58     0.2104    0.2532    1.204    1.031   0.035
   29.30     0.1523    0.1758    1.154    0.994   0.038
   41.02     0.1191    0.1488    1.250    1.138   0.044
   62.50    0.08999    0.1054    1.171    1.091   0.058
```

The model carries 10 to 25 percent MORE spectral power than a human at every
frequency above about 6 Hz, and about 5 percent LESS below 2 Hz. The floor is
two disjoint halves of real human data through the identical pipeline and sits
at a median of 1.00, range 0.94 to 1.14. Turning is clean over the whole band,
0.96 to 1.02 against a floor of 0.93 to 1.06, so this is a speed only deviation.

It is thirty consecutive bins agreeing, which is why it is worth more than its
per bin margin suggests.

### It is the model, not the renderer

Required control, because the serving decoder rounds to integer pixels and snaps
slow steps to lattice directions, and this file already prices half a pixel of
rounding on real human data at 0.866. If the model emitted more slow steps than
a human it would eat more snapping and the renderer would manufacture the excess.

EVENT_SNAP=0 EVENT_ROUND=0, both arms, everything else identical:

```
  freq Hz    serving    control
    1.95       0.949      0.953
   11.72       1.195      1.186
   17.58       1.204      1.181
   41.02       1.250      1.207
   62.50       1.171      1.149
```

Every bin moves by less than its own standard error. The excess is in the token
stream. The renderer is not making it.

### The reading, which is POST HOC and owes its own test

Stated as post hoc on purpose. It was not predicted, it is a reading of a
result, and it must not be cited as though this run tested it.

The two halves of the tilt are the two halves of one failure. Power below 2 Hz
is the movement's velocity envelope, the smooth submovement structure. Power
above 6 Hz is step to step jitter. The model has too little envelope and too
much jitter, which is what a model does when it is UNCERTAIN AT EVERY STEP: it
hedges its conditional mean toward the average, which flattens the envelope, and
it expresses the leftover uncertainty as softmax entropy, which comes out as
broadband high frequency noise after sampling. Humans commit and the model
hedges.

This is consistent with marginals being correct, which `w4_redundancy`
established and which looks like a contradiction until the arithmetic is
written down. The marginal is the average of the conditionals. Conditionals that
are individually too wide, with conditional MEANS that are correspondingly too
flat, average back to the right marginal. So "every marginal is right" was never
evidence that the conditionals were right, and this programme has been reading
it that way.

The cheap test, and it is a diagnostic and not a serving change: sweep the
sampling temperature and watch the high frequency excess. If the excess tracks
temperature it is sampling entropy and the reading holds. If it does not move,
the excess is structural in the conditional means and the reading is wrong. One
GPU run. Nothing above is authorised until that has answered.

### The temperature sweep. The reading holds, and temperature cannot act on it

Arms at sampling temperature 0.8 and 1.2 against the 1.0 already measured,
n=20,000 each, everything else identical. Model over human RAW speed power, so
no standardisation coupling:

```
  freq Hz     t=0.8     t=1.0     t=1.2
    1.95      0.790     1.367     1.447
    5.86      0.526     1.498     2.072
    9.77      0.508     2.243     4.160
   13.67      0.465     2.699     5.664
   17.58      0.371     2.533     5.418
   41.02      0.298     2.376     4.580
   62.50      0.199     1.128     3.232
```

Monotone in temperature in every bin, and the swing between 0.8 and 1.2 is five
to ten times in power. The pre registered criterion is MET: the excess IS
sampling entropy, and the post hoc hedging reading survives its own test.

The reading it survives, restated with the number that matters. At the SERVED
setting the model's within movement speed energy is 1.4 to 2.7 times a human's.
That is conditional over dispersion, variation inside one movement, and it is
compatible with `w4_redundancy`'s correct marginals because a marginal pools
across trajectories while this is measured inside them.

### The part that is not news, and must not be reported as though it were

Temperature cannot fix it, and the record already knew that. Interpolating the
table above, total power crosses human at roughly t=0.88 for the bulk of the
band and about t=0.99 at the very top. But the standardised SHAPE is flattest at
t=1.0, range 0.95 to 1.25, and both neighbours are much worse: t=0.8 runs 0.66
to 1.97 and t=1.2 runs 0.80 to 2.01. So the setting that matches total energy
badly distorts the shape and the setting that matches the shape leaves the
energy about twice too high.

`w4_sharpness` reached exactly this conclusion from different quantities, at the
optimum `curvature_std` 0.2812 against a human 0.3459 while t=1.1 fixes
curvature and breaks `path_efficiency`, and wrote the verdict that still stands:
"One scalar cannot satisfy both, and that is the signature of misspecification
rather than miscalibration. The model does not have the right distribution
scaled wrong. It has the wrong distribution."

This run reproduces that from the frequency domain, an instrument with no
overlap with the one that produced it. That is worth having as independent
agreement and it is worth nothing as news.

### What IS new, and it names the axis

`w4_sharpness` also has the contract AUC by temperature: 0.7184 at 0.7, 0.6929
at 0.8, 0.6580 at 0.9, 0.6433 at 1.0, 0.6737 at 1.1. Put that next to the
spectral optima:

```
  contract AUC optimum          t = 1.0
  spectral SHAPE optimum        t = 1.0
  spectral ENERGY optimum       t = 0.88 approximately
```

The AUC optimum sits on the SHAPE optimum, not the energy one. Moving toward
correct total energy, 1.0 down to 0.9, makes the model MORE detectable, 0.6433
up to 0.6580. So the detector is reading the shape of the speed spectrum and
tolerating an energy excess of roughly two times. Any future proposal that
targets the amount of jitter rather than its distribution across frequency is
pushing on the axis the contract does not score, and this is the measurement
that says so.

One apparent contradiction that is not one. `w4_sharpness` says the paths are
too SMOOTH by `curvature_std`, and the time axis autocorrelation here says the
speed is too ROUGH. Both can hold: curvature is turning per unit distance, a
spatial quantity, and speed autocorrelation is temporal. Too smooth in space and
too jittery in time is a coherent description, not a conflict. Whether the
excess high frequency speed energy is what depresses `curvature_std`, by
inflating the distance denominator, is a mechanism worth checking and is NOT
established here.

### Where this leaves it

The serving side is closed and was already closed. Temperature is at its joint
optimum, it cannot reach the human spectrum from any setting, and the reason is
that the conditionals have the wrong shape rather than the wrong width. The
remaining lever is what the model is trained to optimise, which is the direction
the pre registration named before any of this ran and which this sweep now
supports from a second instrument. Nothing in that direction is authorised here.

## Is the wrong spectral shape made by WHEN the events fire or by WHAT SPEED they carry, 2026-08-05

PRE REGISTERED. Written before the code exists and before any number does.
Read the criteria below as binding, including the invalidity gate.

### The question, and why it is the one worth asking now

The temperature sweep established that the contract reads the SHAPE of the
speed spectrum and tolerates roughly a two times excess in its total energy.
The record's standing verdict on the model is `w4_sharpness`'s "it has the
wrong distribution". That is true and close to useless, because it indicts the
whole model at once. Nothing in the record narrows it to a component.

A trajectory reaches the contract as a list of events, and each event carries
two separable things: WHEN it fires, its dt, and WHAT SPEED it carries. The
contract never sees events. It sees a uniform 125 Hz resampling of them, which
means the event timing is not merely a property of the output, it is the clock
that decides how much of each speed value survives onto the measured grid. An
event with a long dt is interpolated across many grid samples and contributes
almost nothing above a few Hz. An event with a short dt lands on adjacent
samples and contributes its full high frequency content.

So a model whose speed values were perfect but whose event timing was wrong
would still produce the wrong spectral shape, and so would a model with perfect
timing and wrong speeds. Every measurement in this repo mixes the two. No run
has ever separated them.

### The design

Four arms on the same held out rows, n=20,000, seed 0, same instrument as
`w4_spectrum`, one centred 64 sample window per row.

```
  arm   event timing      speed values     how produced
  A     human             human            the real held out trajectory
  B     human             model            dt teacher forced, s and th sampled
  C     model             model            free running, exactly as served
  D     model             human            offline recombination, no GPU
```

A is the reference every ratio is taken against. C is the arm already measured
and is the thing to be decomposed. B is the decisive arm. D is the mirror and
is explicitly the weakest of the four, because pairing one row's speed sequence
with another's timing is a chimera no model produced. D may agree with B or it
may flag the decomposition as unclean. D may NOT overturn B.

B needs one change to `EventARModel.sample`: the existing `force` argument
carries a single mask shared by all three channels, and this needs dt forced
while s and th are sampled. The mask becomes optionally per channel. The
`force=None` path must remain bit for bit identical, and that is to be VERIFIED
against the current code on a fixed seed, not asserted. Generation is the thing
under measurement and must not move while it is being measured.

### The statistic

Mean STANDARDISED speed power ratio over the 16 bins from 11.72 to 41.02 Hz,
minus the mean split half floor over the same bins. Call it E.

The standardised channel, not the raw one. The raw channel's split half floor
is 0.7498 with a mean per bin standard error of 0.42, because mean raw power is
dominated by a handful of large amplitude trajectories and the two halves of
real human data do not agree with themselves. It is unusable for a threshold.
The standardised channel's floor is 1.0367 with a per bin standard error of
0.0371 and is well behaved. It is also the channel the temperature sweep showed
the contract actually scores.

Measured today for arm C: mean ratio 1.1817, mean floor 1.0367, so

```
  E_C = +0.1451
```

Bins within a band are correlated because they come from the same windows, so
the band mean standard error is NOT the per bin value over sqrt(16). Taking the
conservative choice and treating the band mean standard error as the full per
bin 0.037, E_C is about four standard errors from zero.

### Criteria, fixed in advance

Validity gate, checked BEFORE any ratio is read. Arm B's row retention rate,
the fraction of drawn rows yielding a usable 64 sample window, must be within
20 percent relative of arm C's. Forcing human dt while the model still decides
its own termination can shorten rows, and a length difference between arms
would show up as a spectral one. If the gate fails the run does not answer the
question and no verdict is reported.

```
  TIMING dominant   E_B < +0.05    at least two thirds of the excess removed
  VALUES dominant   E_B > +0.11    at most one quarter of the excess removed
  SPLIT             otherwise      report the fraction, neither is dominant
```

The two thresholds sit about 1.6 conservative standard errors apart, so SPLIT
is a likely outcome and is a real answer rather than a failure. B and C share
the reference arm and most of the pipeline, so their difference is better
determined than either alone, but that is an argument for reading the
difference and not for narrowing the thresholds after the fact.

Sanity check that fires regardless of verdict: E_B must not be materially
BELOW zero. If forcing human timing produces less high frequency energy than a
human, the forcing is broken and no verdict holds.

Mirror, subordinate: E_D above +0.05 supports TIMING, E_D below +0.02 supports
VALUES. If D contradicts B, the report says the decomposition is not clean. It
does not pick the arm that reads better.

### What each verdict buys, and what it does not

TIMING would put the defect in the dt head and would mean every past reading of
the speed values as too jittery was reading a clock error through them. VALUES
would put it in the speed head and would retire the timing hypothesis. SPLIT
would say both heads carry it, which is the least convenient answer and is the
one no build should be authorised against without a further run.

None of the three moves the score. This is a diagnostic and it produces no
serving change. Its whole value is that it would be the first time this project
names a component rather than the model.


### AMENDMENT, made before the GPU arms ran

Two corrections. Both were found by running the arms that need no model, which
is why they were run first.

**The dt quantisation worry was wrong.** The premise was that the model's dt
alphabet is whole milliseconds while human dt is not, so every spectral
comparison in this repo had been charging the model for a rounding a human never
suffers. Measured on 298,633 held out events, the round trip error is a mean
absolute 0.0062 ms and an rms of 0.0526 ms, which is 0.7 percent of one 125 Hz
sample. Human dt is already whole milliseconds to within storage precision, and
the sub millisecond values that prompted this were float16 artefacts in an
unfiltered subset. Arm Aq stays in the design because it costs nothing and
closes the question, but it prices at nothing and no reading depends on it. The
confound does not exist.

**The registered statistic was mis specified and is replaced.** E was defined as
the band mean ratio MINUS the reference arm's split half floor. A floor cannot
serve as a subtracted offset. It is one draw from a distribution whose standard
deviation at these sample sizes is 0.03 to 0.05, so subtracting it injects that
entire error into the statistic. The failure is demonstrable rather than
theoretical: the reference arm compared against ITSELF has a ratio of exactly 1
by construction, so the statistic must read exactly 0, and with the floor
subtracted it read +0.1173, which is four fifths of the entire effect the run
was built to decompose.

The estimator itself is sound. Over 200 random half splits of real human data
the band mean ratio sits at 1.0033 with a standard deviation of 0.0401, centred
where it should be. The single odd or even split used as a floor was simply an
unlucky draw, 0.8827 in the 6,000 row run and 1.0367 in the 20,000 row run.

Amended definition:

```
  E = mean standardised speed power ratio over 11.0 to 41.5 Hz, MINUS ONE
  null sd = 400 draw bootstrap, reference resampled against itself at the
            arm's sample size and the reference's sample size
```

The null is 1.0 by construction, a self comparison reads exactly 0, and the
floor is now used as what it is, an error bar.

The thresholds move with it. The intent registered in advance was two thirds of
the excess removed for TIMING and at most one quarter removed for VALUES, and
that intent is preserved and restated as a fraction of arm C's own excess, which
is what the question always meant:

```
  TIMING dominant   B keeps  < 1/3 of C's excess
  VALUES dominant   B keeps  > 3/4 of C's excess
  SPLIT             otherwise
```

The validity gate on retention, the sanity check that E_B must not be materially
negative, and arm D's subordinate status are all unchanged.

**This makes the existing `w4_spectrum` result stronger, not weaker.** Arm C's
band mean standardised ratio was 1.1817. Under the amended statistic that is
E_C = +0.1817 against a null standard deviation of roughly 0.026 at that sample
size, so about seven standard deviations. The floor subtraction had been quietly
removing a +0.037 fluctuation and understating the tilt. The direction and the
conclusion of that run are unaffected, and the tilt it found is larger than it
reported.

One consequence for the record. Every `floor` column printed by `w4_spectrum`
and its control carries this same 0.03 to 0.05 single draw noise and should be
read as an error bar rather than a baseline. The readings that rest on it are
unaffected, because they were qualitative, the turn channel sitting at 0.96 to
1.02 against a floor near 1.0. The control run's "max shift 1.08 SE" used the
per bin ratio standard error and not the floor, so it is untouched.


### SECOND AMENDMENT, also made before the deciding run

The registered arm B is not well posed for this checkpoint and is withdrawn.
Found by the n=800 smoke test, which is why it was run.

`event_ar_v2`'s config is `emit_order: 's_th_dt'`. Within one step the model
samples speed FIRST, then direction conditioned on the speed, then dt
conditioned on BOTH. dt is the last channel in the chain. Arm B forced dt after
all three heads had already run, which substitutes a clock that the speed at
that same event was never conditioned on. The within step coupling
p(dt | s, th) is the strongest coupling in the factorisation and arm B broke it
at every single event.

The smoke test says so plainly. Against the human reference, arm C free running
reads E = +0.1997 and arm B reads E = +0.5213, two and a half times WORSE than
the model left alone. Arm D, the offline chimera, reads +0.8088. Both mixed arms
being far worse than either pure arm is the signature of a broken pairing, not
of a clean separation. Arm B as registered measured a contradiction, not a
timing defect.

The correct forcing is a PREFIX of the emit order, because a prefix leaves every
downstream head properly conditioned. That gives:

```
  arm E   force (s, th) from the human, model chooses dt conditioned on them
```

Every quantity the dt head sees is then real human data, so the head is measured
on distribution and in isolation. Arm E also forces the speed stream's PAD, so
it terminates exactly where the human row does, which makes its retention match
arm A by construction and turns the validity gate into a check that the forcing
took at all.

The reading becomes an ELIMINATION rather than a two sided decomposition. Arm E
differs from arm C only in that its speeds and directions are real:

```
  E clean    the dt head is innocent, the speed head carries the tilt
  E tilted   the dt head carries it even when everything it sees is real
```

Thresholds carry over unchanged in intent, applied to how much of arm C's excess
arm E RETAINS: under a third means the speed head, over three quarters means the
dt head, between is a split.

What this costs, stated rather than hidden. The complementary arm, model speeds
on a forced human clock, is NOT AVAILABLE for an `s_th_dt` checkpoint. There is
no way to force the last channel of the chain without breaking the two above it.
So the elimination rests on arm E alone, and if arm E comes back clean the
inference that the speed head is responsible carries the assumption that the
tilt belongs to one head rather than to an interaction between them. That
assumption is not tested here and must not be quietly dropped later. A
`dt_s_th` checkpoint would test it directly, and the record already lists such a
retrain among the closed items, so the material may exist.

Arm D is retained with a new meaning: it is arm E without the conditioning, the
same human speeds on a model clock that never saw them. E minus D prices what
conditioning buys. D stays subordinate and may not overturn E.


### The result. The clock is wrong in the 11 to 22 Hz band, and most of what looked worse than that was drift, 2026-08-05

Three arms decided this and they are reported in the order they have to be read.

**Arm F, the seam control, PASS.** Forcing all three channels routes real human
tokens through the sampler and the serving decoder and must come back as arm Aq.
It does, exactly. E is +0.0000 at 0.0 sd, all 32 per bin ratios read 1.000 to
three decimals, and the window count matches Aq to the row (7,866 against 7,866).
The forcing machinery, the mask indexing and the decode path introduce nothing.

That is all arm F can prove, and stating the limit matters more than stating the
pass. With all three channels forced there is no free channel left to disturb, so
arm F verifies plumbing and is silent on the question that actually threatens the
reading. Arm G was built to answer that one.

**Arm E, the registered arm.** Human speeds and directions forced, model choosing
its own clock conditioned on them. E is +0.3864 at 17.5 sd against Aq, where the
free running model, arm C, reads +0.1825. Handing the model real speeds made the
spectrum WORSE, not better. That is outside the range the registered thresholds
assumed, which was 0 to E_C, so the share language in the pre registration does
not apply and the fraction must not be quoted as a share of a fixed budget. The
validity gate passed: arm E retention 37.3% against arm A 43.5%, relative
difference 14.1% against a 20% gate.

**Arm G, the drift control.** Arm E has a hole in it. At step i the model sees
real human speed and direction but its OWN previously generated dt, a hybrid
history it never saw in training, so arm E's excess could be exposure drift
rather than a defective clock. Arm G removes drift completely: one fully teacher
forced forward pass over the real sequences, dt drawn once per position from
p(dt_i | real history, real s_i, real th_i). Every quantity conditioned on is
real. No autoregression, so nothing can accumulate. It is also a single forward
pass instead of 256 sequential ones, which is why it costs roughly a
two hundred and fiftieth of a generation arm and ran with a 74C peak.

Arm G reads **+0.1221 at 5.7 sd**, which is 31.6% of arm E. Retention 43.2%
against arm A's 43.5%, so the pass is clean.

Under the rule registered before the arm ran, G_HEAD_MIN 0.50 and G_DRIFT_MAX
0.25, that lands in the MIXED band and the registered verdict is *no clean
reading on magnitude*. That is the honest headline and it is recorded as such.

**The per bin table says considerably more than the registered scalar, and it is
reported here as an observation rather than as the registered statistic.** Arm G
and arm E are not the same defect at two sizes. They are different shapes.

    freq Hz      arm C     arm E     arm G
       3.91      0.997     0.840     0.959
       9.77      1.129     1.062     1.126
      13.67      1.174     1.262     1.252
      17.58      1.203     1.426     1.280
      21.48      1.202     1.395     1.215
      31.25      1.177     1.473     1.063
      41.02      1.251     1.442     0.971
      50.78      1.029     1.275     0.819
      62.50      1.182     2.428     0.819

Arm G is a BUMP. It rises from 1.00 at 5.9 Hz to a peak of 1.280 at 17.6 Hz, then
decays back through 1.0 around 39 Hz and ends in a DEFICIT, 0.819 at the top of
the band. Arm E is a monotone ROTATION that keeps climbing to 2.428 at 62.5 Hz.

So the two questions separate cleanly even though the scalar did not:

1. **The dt conditional has a real defect and it is narrow and mid band.** With
   every drift path cut, the model's one step clock still puts about 25 to 28
   percent too much speed power into 12 to 22 Hz. That is significant at 5.7 sd
   and it is not plumbing, because arm F reads zero through the same code.
2. **The high frequency blowup in arm E is drift, not the clock.** Above roughly
   40 Hz arm G is BELOW human while arm E is far above it and climbing. That part
   of arm E appears only when the model feeds its own dt history back to itself,
   and it is the largest part of arm E's excess.

The 12 to 22 Hz band is where the corrective submovement structure lives, and the
record already contains an unexplained AR pause shortfall and an unexplained
training renderer excess. Whether those are the same defect seen three ways is
not established and must not be asserted.

**What this does NOT license.** It does not license a phase conditioning build or
a spectral loss term. Both remain NOT AUTHORISED, and the reason is unchanged:
`w4_spectrum` killed the missing narrow band coherent oscillation hypothesis, and
a mid band power excess is not the same claim as a missing oscillation. It also
does not license reading arm G's magnitude as the size of the fixable defect,
because 31.6% sits inside the band the pre registration reserved for no reading.

**The honest cost.** Arm G measures the one step conditional, not the model's
joint. Every dt is drawn independently given a real history, so arm G's sequence
is a draw from the product of one step conditionals, not from the model. That is
the whole point of the arm and it is also its limit: it can prove the conditional
is tilted, and it cannot say what the joint would look like if the conditional
were fixed. The complementary arm that would settle it, model speeds on a forced
human clock, is still NOT AVAILABLE for an `s_th_dt` checkpoint.

## How is the dt conditional wrong, 2026-08-05

WRITTEN BEFORE ANY CODE EXISTED, and the honest label first: **this is
CHARACTERISATION, not a hypothesis test.** It has no single registered statistic
and no pass or fail. Arm G established that the model's one step clock is tilted
in the 12 to 22 Hz band with every contamination removed. It did not say in what
way the conditional is wrong, and there is no reason to guess when the arm G
machinery already answers it directly and for free.

The instrument is the arm G forward pass reused. One fully teacher forced pass
over held out human rows gives, at every position, the model's complete
distribution over the next interval given a fully real history and that position's
own real speed and direction. Next to it sits the interval the human actually
produced. Comparing a predicted distribution against the realised value at a
million positions is a calibration question and it is a solved one.

The reason to register anything at all is that a calibration plot admits a story
for every shape it can take, so the readings are fixed here in advance.

**The primary read, the randomised PIT histogram.** For each position take the
model's CDF at the realised interval, randomised within the class so a discrete
alphabet does not manufacture structure. Perfect calibration is exactly uniform.

    U shaped, mass piled at both ends -> the model is OVER CONFIDENT, its
        intervals are too narrow, reality lands in its tails far too often
    HUMP shaped, mass piled in the middle -> the model is UNDER CONFIDENT, its
        intervals are too wide and it hedges
    TILTED, mass sloped to one side -> the model's clock is BIASED, it is
        systematically fast or slow rather than mis scaled
    FLAT -> the conditional is calibrated and the arm G tilt is NOT a defect of
        this distribution at all, in which case the tilt has to live in the
        dependence between successive intervals and not in any one of them

That last branch is the one that would hurt, and it is stated first precisely
because it is the outcome that would cost the most. If the PIT comes back flat
then the per position conditional is right, the arm G reading has to be re read as
a statement about correlation rather than about the head, and the current
description in this file is wrong and gets corrected rather than qualified.

**The secondary reads, all conditional on the same pass.**

1. Predicted mean and predicted sd against the realised value, so a bias and a
   dispersion error can be told apart rather than blurred into one PIT shape.
2. The same, sliced by the realised interval, so a failure confined to short
   intervals is not averaged away against long ones. The 12 to 22 Hz band
   corresponds to periods near 45 to 83 ms and the slicing must be fine enough to
   see it.
3. The same, sliced by the position's own real speed, because the whole reason
   this checkpoint emits speed first is that interval and speed are coupled, and
   a conditional that is calibrated on average can still be wrong at the speeds
   that matter.
4. The lag 1 and lag 2 autocorrelation of the arm G interval sequence against the
   human one. Arm G draws each interval independently given a real history, so if
   the conditional carries the dependence correctly these should match. A gap here
   is the direct measurement of the branch above.

**What may and may not follow.** Nothing about serving. No build is authorised by
this run. Phase conditioning and the spectral loss term remain NOT AUTHORISED and
a calibration result cannot revive them, because neither was gated on calibration.
This run is allowed to do exactly one thing, which is to say what kind of wrong
the dt conditional is, so that the next proposal is aimed rather than guessed.

### AMENDMENT to the calibration characterisation, made before the deciding run

Two of the secondary reads registered above are unsound and are REMOVED, not
qualified. An n=800 smoke test surfaced both. Neither removal touches the primary
read, and the primary read is the one the conclusion will rest on.

**Removed 1, slicing by the realised interval.** Registered as secondary read 2.
It conditions on the OUTCOME. Regression to the mean then produces a bias that
slopes positive at short realised intervals and negative at long ones in ANY
model, correct or not, and the smoke test duly produced exactly that pattern:
+0.97 ms in the shortest bin falling monotonically to -75 ms in the longest. That
is the estimator, not the model. The legitimate version slices by the model's OWN
predicted mean, which is known before the outcome, and that is what replaces it.

**Removed 2, the interval autocorrelation comparison.** Registered as secondary
read 4, and it was to be the direct measurement of the flat branch. It is biased
by construction. A one step draw is conditionally independent of its neighbours
GIVEN a real history, whereas a human interval feeds forward into its own
successor, so the draw's autocorrelation is attenuated relative to the human even
if the conditional is exactly right. The smoke test showed lag 1 matching almost
exactly, +0.0970 against +0.0992, with lag 2 short by -0.0410, and there is no way
to tell that apart from the structural attenuation without a known correct
conditional to calibrate against, which does not exist. Reporting it with a caveat
would put a number in the record that cannot be read.

**What replaces read 4, and it is strictly better.** Slice the PIT by what the
model ALREADY KNEW when it predicted: the previous interval, and the one two steps
back. A conditional that carries the dependence is uniform inside EVERY such
slice, not merely uniform on average, so a tilt inside a lag 2 slice with none
inside a lag 1 slice says the head tracks the immediately preceding interval and
misses the structure one step further back. This has no confound, it needs no
reference model, and it answers the flat branch directly rather than by proxy.

**Also measured, and nil.** The dt head leaves mass on the PAD class, which
`class_to_dt_ms` CLAMPS to 150 ms rather than rejecting, so any such mass would be
counted as a 150 ms interval. Measured at live positions: mean 0.00e+00, max
8.6e-07. It does not exist, and therefore `w4_timing` arm G, which sampled the full
alphabet without renormalising, is unaffected. This file renormalises anyway.

**Standing.** The primary read, the randomised PIT histogram, is UNCHANGED from
its registration, as are its four branch readings. The smoke test at n=800 already
shows it flat, ratio 0.993 and mean 0.5031, which is the branch registered above
as the one that would cost the most. If n=20000 confirms it then the arm G section
in this file gets CORRECTED and not qualified, exactly as registered.

### CORRECTION to the arm G section above, 2026-08-05

The pre registration said that a flat PIT would make the arm G reading WRONG
rather than incomplete, and that it would be corrected and not qualified. The PIT
came back flat. This is that correction.

**What the arm G section above claims, and which no longer stands:** "the dt
conditional has a real defect and it is narrow and mid band", and "with every
drift path cut, the model's one step clock still puts about 25 to 28 percent too
much speed power into 12 to 22 Hz". The measurement is real. The attribution to a
defective dt conditional is not supported and is withdrawn.

**Why.** At 1,034,517 held out positions the one step dt conditional passes every
calibration test available:

    randomised PIT              end/mid 1.007, mean 0.5008, KS 0.0018
    overall bias                +0.000 ms on a 9.666 ms mean
    PIT by previous interval    worst tilt 0.0037 across 12 slices
    PIT by interval two back    worst tilt 0.0062 across 12 slices
    PIT by real speed           worst tilt 0.0053 across 7 slices
    reliability by own predicted mean   bias within 0.23 ms in every bin
    PIT autocorrelation         lags 1,2,3,4,5,8, worst 2.8 sd, |acf| <= 0.0027

A conditional that is uniform overall, uniform inside every slice of everything it
already knew, unbiased at every predicted level, and serially independent in its
residuals, is not distinguishable from correct by any instrument in this repo.

**And the reason arm G's excess is not evidence against that.** Arm G is under
correlated relative to the human EVEN WHEN THE CONDITIONAL IS EXACTLY RIGHT. A
human interval feeds forward into the conditional of its own successor; a
resampled one does not. At lag one:

    human   Cov(dt_i, dt_i+1) = E[Cov(dt_i, mu_i+1(dt_i) | H_i)] + Cov(mu_i, mu_i+1)
    arm G   Cov(dt_i, dt_i+1) =                                    Cov(mu_i, mu_i+1)

The first term is present in the human and absent in arm G, and it vanishes for a
perfect model exactly as readily as for a bad one. Under correlation is extra high
frequency power. So arm G's +0.1221 is the sum of a possible model defect and a
construction artefact of unknown size, and nothing measured so far separates them.

This is the same defect that removed the raw interval autocorrelation from
`w4_dtcal` before it ran. It was correctly identified there and NOT carried across
to arm G, which was already built. That is the error: an estimator bias was named
in one place and left standing in another that has it just as badly.

**What survives.** Arm F still passes and the plumbing is still exact. The arm E
versus arm C comparison, +0.3864 against +0.1825, is untouched by this, because
both arms are generated the same way and the artefact is common to neither: arm C
and arm E both run autoregressively. What falls is only the attribution of arm G's
number to the dt head.

**The control that settles it, and the only one that can.** Generate sequences
FROM the model. On its own samples the model IS the true conditional by
construction, exactly and with no estimation error. Run the identical arm G
pipeline on them, one step resample against model generated reference. Any excess
that appears there is PURE construction artefact, because there is no defect left
for it to measure. Registered before running:

    artefact >= 0.10   arm G's +0.1221 is explained by construction, the dt head
                       is exonerated, and the mid band bump is an artefact
    artefact <= 0.04   the artefact is small, arm G's excess is mostly real, and
                       the withdrawn attribution above is reinstated
    in between         subtract and report both numbers with the residual

No build is authorised either way. Phase conditioning and the spectral loss term
remain NOT AUTHORISED.

### The artefact control, and it lands ON the threshold, 2026-08-05

The control registered above ran. Generate from the model, where the model IS the
true conditional of its own samples by construction, then apply the identical arm
G one step resample. No defect exists there, so anything it reads is pure
estimator.

    SELF CHECK   arm SA against itself      E +0.0000  PASS
    ARTEFACT     arm SG against arm SA      E +0.0387  null sd 0.0212  +1.8 sd
    retention    SA 43.3%, SG 44.5%

**The registered verdict must not be read off the threshold, and the script now
says so itself.** The rule was: artefact >= 0.10 exonerates the dt head, <= 0.04
reinstates the attribution, in between subtract and report both. The reading is
+0.0387, which is 0.0013 below the 0.04 line. That margin is **0.063 of one null
standard deviation**. A threshold call resting on six percent of one sd is
spurious precision, and the fact that it fell on the flattering side of the line
is luck rather than evidence. It is therefore read as the registered IN BETWEEN
case, which is what the numbers actually support:

    arm G excess            +0.1221
    artefact                +0.0387   (31.7% of arm G)
    residual                +0.0834   (2.8 sd against a combined sd of 0.0301)

At +1.8 sd the artefact is not separable from zero, and it is just as
inseparable from twice its own size. Both numbers are reported and neither is
quoted alone.

**The per bin shape is the more informative result, and it is unflattering.** The
artefact is not spectrally neutral. It is a mid band BUMP with a high frequency
DEFICIT, which is the same signature arm G has:

    freq Hz    arm G vs human    artefact vs model
      11.72             1.236                1.132
      15.62             1.267                1.147
      17.58             1.280                1.155
      21.48             1.215                1.090
      31.25             1.063                0.986
      41.02             0.971                0.896
      50.78             0.819                0.837
      62.50             0.819                0.777

Same peak location, 17.58 Hz. Same sign everywhere. Same crossing below one near
39 Hz. Same deficit at the top. The estimator, applied to data with NO defect in
it, manufactures a smaller copy of the exact pattern that was written up as a
finding. That is much stronger evidence against the original reading than the
band mean scalar is, because a shape match cannot happen by coincidence in the
way a single number can.

**Standing.** The attribution of arm G's mid band bump to the dt conditional stays
WITHDRAWN. It is not reinstated by a threshold call worth 0.06 sd, and the shape
match argues actively against reinstating it. What can be said is bounded and
small: about a third of arm G's excess is provably the estimator, the rest is a
2.8 sd residual of a statistic whose own artefact reproduces its shape, and the
one step resample is not a sound instrument for spectral claims about this model.
`w4_dtcal`'s finding is untouched and remains the solid result: the dt conditional
passes every calibration test at 1,034,517 positions.

**Consequence for method, and it generalises past this run.** Two separate
diagnostics in this workstream have now failed the same way: a statistic was built,
looked clean, and turned out to carry a bias that a control on known good data
reproduces. The rule that follows is that no generated artefact may be read as a
model property until the identical pipeline has been run on data whose truth is
known by construction. Arm F was that check for the plumbing and passed. It was
never that check for the estimator, and nothing was.

**Still NOT AUTHORISED:** phase conditioning, the spectral loss term. Unchanged.

## Does arm E's forcing carry an estimator artefact too, 2026-08-05

PRE REGISTERED, written before any code exists, and written because the rule
established one section above requires it: no generated artefact may be read as a
model property until the identical pipeline has been run on data whose truth is
known by construction. Arm E has never had that check. Arm F checked the plumbing.
Nothing checked the estimator.

**The claim under test.** Arm E reads +0.3864 against human where free running arm
C reads +0.1825, and that was reported as "handing the model real human speeds
makes it worse, not better". It is the last standing result of this workstream and
it is the reason the reading was interesting at all.

**Why it is suspect.** Arm E forces human s and th at every position while the
model supplies its own dt. In the model's own generative order dt_i influences
s_i+1 and th_i+1. Under forcing, s_i+1 and th_i+1 come from the human and were
produced alongside the HUMAN's dt_i, not the model's. So every position after the
first sits in a context the joint never produces: real speeds paired with a clock
that did not generate them. That is the same class of mismatch that turned out to
manufacture arm G's entire signature, and there is no reason to assume arm E is
immune just because it was measured first.

**The control.** Identical to the one that priced arm G, applied to the forcing
construction instead of the resampling one.

    arm SA   model generated free running, the synthetic reference. On these rows
             the model IS the true generator, exactly, by construction.
    arm SE   the SAME rows' s and th forced back into the model, model supplying
             its own dt, exactly as arm E does to human rows
    arm SC   a second, independent free running generation. Two draws from one
             distribution, so this is the NULL and it must read zero.

There is no model defect anywhere in SA, SE or SC. Every one of them comes from
the model. So whatever SE reads against SA is pure construction artefact, and
whatever SC reads against SA is the noise floor of the whole pipeline.

**Registered thresholds.**

    VALIDITY   |E(SC vs SA)| must be within 2 null sd of zero. If two independent
               model generations do not agree, the pipeline is not measuring what
               it claims and NO verdict is reported.

    artefact >= 0.25   arm E's +0.3864 is dominated by construction. The "real
                       speeds make it worse" reading FALLS and is withdrawn.
    artefact <= 0.08   construction is minor and arm E's reading STANDS.
    in between         subtract, report both numbers and the residual.

**And the boundary rule, registered here because it was learned the hard way one
section above.** If the reading lands within one null sd of either threshold, the
threshold call is REFUSED and the in between case is reported instead. A verdict
that flips on less than the measurement's own resolution is not a verdict.

**The per bin shape is read too, and it is not subordinate.** The artefact control
for arm G was decided by shape rather than by its scalar: a bump at the same peak
bin with the same high frequency deficit cannot coincide by chance. If SE against
SA reproduces arm E's monotone rotation, climbing to roughly 2.4 at 62.5 Hz, then
arm E's shape is the estimator's shape and the scalar hardly matters.

**Thermal.** Two generation passes, and three if SC runs. The run that was killed
at 84C did three. The corrected watchdog is armed on the right process name this
time, and the script checks the GPU between passes and waits for it to fall below
the launch gate rather than running them back to back.

No build is authorised by this either way. Phase conditioning and the spectral loss
term remain NOT AUTHORISED.

### The result. Inside the registered band arm E's effect is ENTIRELY construction, 2026-08-05

`research/w4_forcing.py`, n=20000 drawing the same 18,081 held out rows w4_timing
used, `event_ar_v2_s40000.pt`, temperature 1.0, three full generation passes.
Result in `research/w4_forcing.json`.

    SELF CHECK   arm SA against itself            E +0.0000   exact, as required
    VALIDITY     arm SC, a second free draw       E +0.0273   +1.3 sd   PASS
    ARTEFACT     arm SE against arm SA            E +0.1720   +8.1 sd   null sd 0.0213

    retention    SA 43.5%   SE 36.7%   SC 43.5%

The validity gate passes, so the reading is admissible. Two independent free
running generations agree with each other to well inside the noise, which is what
makes the third arm readable at all.

**The registered scalar verdict is the in between case.** The artefact is +0.1720
against arm E's +0.3864, which is 44.5%, sitting between the registered 0.08 and
0.25. It is 3.7 null sd from the nearer threshold, so this is a genuine in between
and not a boundary refusal. Residual +0.2144 at 7.0 sd on the combined error bar.
Both numbers are reported and neither stands alone, exactly as registered.

**The retention match is the first thing that should have been noticed.** On
synthetic rows with no defect in them, forcing drops retention from 43.5% to
36.7%. On human rows, arm E drops it from 43.5% to 37.3%. Those are the same
number. The retention loss in arm E is not human data being hard, it is what
forcing does, and it reproduces on data where nothing is wrong.

**The shape splits arm E in two, and this is the finding.** The registered shape
clause asked whether SE reproduces arm E's monotone rotation climbing to about 2.4
at 62.5 Hz. It does not. SE peaks at 1.24 near 21 Hz and comes back to 1.10 at the
top. So by the registered clause arm E's high frequency shape is NOT the
estimator's. But dividing arm C out of arm E first, which prices what FORCING adds
rather than what the model does, tells a sharper story. This decomposition was not
registered and is reported as an observation.

    freq Hz   arm C   arm E   E over C   SE artefact   residual
       5.86   1.056   0.870      0.824         0.865      0.952
       9.77   1.129   1.062      0.941         1.053      0.893
      13.67   1.174   1.262      1.075         1.197      0.898
      17.58   1.203   1.426      1.185         1.207      0.982
      21.48   1.202   1.395      1.161         1.244      0.933
      31.25   1.177   1.473      1.251         1.157      1.082
      41.02   1.251   1.442      1.153         1.113      1.036
      50.78   1.029   1.275      1.239         0.980      1.264
      62.50   1.182   2.428      2.054         1.102      1.864

    band            forcing increment   construction   residual
    below 11 Hz                 0.883          0.937      0.944
    11 to 41.5 Hz               1.173          1.172      1.002
    above 41.5 Hz               1.412          1.036      1.359

Inside the pre registered band, the band the registered statistic actually
measures, the forcing increment is 1.173 and the construction artefact is 1.172.
They agree to three decimals and the residual is 1.002. **Within the registered
band, arm E's entire effect is construction.** There is nothing left over.

Above 41.5 Hz, where the registered statistic does not look, construction reads
1.036 and the real forcing increment reads 1.412. That part is real and this
control does not explain it. It is also the part nearest the 62.5 Hz Nyquist of
the 125 Hz contract axis, where resampling artefacts live, so it deserves its own
control before anything is read into it rather than being promoted now.

**What is withdrawn.** The reading that handing the model real human speeds makes
its clock measurably worse is WITHDRAWN inside the registered band. It was
construction. Arm E forces human s and th while the model supplies its own dt, so
from the second event onward every position sits in a context the joint never
produces: real speeds beside a clock that did not generate them. That mismatch
alone, on data with no defect, produces the same retention loss and the same
mid band excess.

**What survives.** Arm C, free running against human, still reads +0.1825 at 8.3
sd. That arm generates everything itself and forces nothing, so none of this
touches it. The gap between the model and people is real. What has now failed
twice is the attempt to LOCALISE that gap by forcing part of the human sequence
back in and reading the remainder.

**The method rule, now stated in its general form.** Teacher forcing a strict
subset of a jointly generated sequence creates an off distribution context that
is a property of the construction, not of the model. Any statistic read off such
an arm carries that signature. Arms F and SC prove the plumbing and the noise
floor and neither can see this. The only instrument that can is the same arm run
on data the model itself generated, where the truth is known by construction, and
it must be run BEFORE the arm is interpreted rather than after a claim has already
been made. Two claims have now been retracted for want of it.

**Thermal and safety.** Three generation passes, roughly 55 minutes each, with the
registered inter pass cooldown firing both times: the script held at 77C and 78C
until the GPU fell to 60C before starting the next pass. Spot readings across the
run sat at 77 to 79C and never approached the 83C kill line. `best.pt` MD5
verified `91326a29750789f3167055324ef377c5` unchanged, eval data and scoring code
untouched.

**Second watchdog defect, recorded because it is the same class of error as the
first.** `watchdog2.sh` matched the run by script NAME. Every `$(pids)` command
substitution forks a subshell that inherits the watchdog's own command line, which
contains the pattern, and that fork has a different PID from `$$`, so the self
exclusion missed it. The kill path was unaffected, since the real process matched
too and would have been killed at 83C, but completion was never detected and the
peak was never reported. The peak for this run is therefore bounded by about
twenty spot readings at 77 to 79C rather than measured. Replaced by
`watchdog3.sh`, which takes the run's PID. Matching on a name was the mistake both
times.

DIAGNOSTIC ONLY. No serving change follows. No build is authorised by this. Phase
conditioning and the spectral loss term remain NOT AUTHORISED.

## Is the excess there from the first moment, or does it build up along the movement, 2026-08-05

PRE REGISTERED, written before any code exists.

**Why this and not something else.** Two facts now stand and they do not fit
together on their own. `w4_dtcal` says the one step conditional is calibrated
against real history, six ways, with a bias of +0.000 ms. Arm C says the model
free running is +0.1825 over human at 8.3 sd in the registered band. Right at
every step, wrong over a trajectory. Two attempts to localise that by forcing part
of a human sequence back in have both been withdrawn, and the reason is now
understood and general, so a third attempt of that kind is not on the table.

The remaining question that separates two whole families of fix is WHERE ALONG THE
MOVEMENT the excess lives. If it is present in the first half second it is a
property of the model's texture from the outset. If it grows as the movement runs
it is accumulation, the model drifting into states it never occupies in training.
These call for opposite work and there is currently no evidence either way.

**The instrument, and why it cannot carry the fault that killed the last two.**
Free running generation on one side, real human recordings on the other, nothing
forced, nothing mixed. The comparison is model against human exactly as arm C
makes it. The construction artefact class requires a hybrid context and there is
no hybrid context anywhere in this design.

**The statistic.** Resample to the contract's 125 Hz grid, take the speed series,
then slide a 64 sample window with a hop of 8. For each window remove the mean,
divide by that window's own standard deviation, apply Hann, take the periodogram
and average the 11 to 41.5 Hz bins. Standardising per window is deliberate: it
removes amplitude and leaves only the SHAPE of the texture, so the fact that a
person is obviously faster in the middle of a movement than at its ends does not
enter. Regress log band power on the window's normalised centre position, 0 at the
start of the trace and 1 at its end, WITHIN each trace. That per trace slope is
the unit of analysis.

    SLOPE_DIFF = mean over model traces of the slope
               - mean over human traces of the slope

Zero means the texture excess is the same shape at the end of a movement as at its
start. Positive means it grows.

**Thresholds, derived rather than chosen.** The measured excess is a ratio of
1.1825, which is 0.1677 in logs. If the log excess runs linearly from a at the
start of a movement to a+d at its end, its average is a + d/2 = 0.1677. Pure
accumulation means a = 0 and therefore d = 0.335. Pure flatness means d = 0.

    SLOPE_DIFF >= 0.17     ACCUMULATION. At least half the excess is growth
                           along the movement. The work is long horizon
                           consistency.
    SLOPE_DIFF <= 0.05     PRESENT FROM THE START. At most about a seventh is
                           growth. The work is the texture itself.
    in between             MIXED. Report the number and both shares, neither
                           alone.
    BOUNDARY               within one null sd of either threshold the threshold
                           call is REFUSED and the in between case is reported.

The conversion from d to a share of the excess is an approximation, because the
excess was measured with one centred window per trace and this uses many. The two
numbers are anchors that make the thresholds principled rather than invented. They
are not an exact accounting and must not be quoted as one.

**A third outcome is registered now so it cannot be reinterpreted later.** A
NEGATIVE SLOPE_DIFF is possible and would mean the excess is concentrated at the
START of movements and fades. That is neither branch above, it would point at the
launch of a movement rather than at its texture or its drift, and it is written
down here so that if it happens it is a result rather than a surprise.

**The confound that has to be handled, and how.** `w4_timing`'s `windows()` takes
one centred window per trace precisely so that a duration difference between arms
cannot become a spectral one. This design deliberately takes several, so that
protection is gone and must be replaced. Traces are binned by their resampled
length into deciles of the HUMAN length distribution, SLOPE_DIFF is computed inside
each bin, and the bins are pooled weighted by human count. Any bin holding fewer
than 100 traces on either side is dropped and reported as dropped. Only traces long
enough for at least two windows can produce a slope at all, so the population here
is longer than the population arm C measured, and that is stated rather than
hidden.

**Validity gate.** Split the human traces at random in half and run the identical
length matched pipeline on one half against the other. It must land within 2 null
sd of zero. If the pipeline manufactures a slope difference between two samples of
the same population then nothing downstream is readable. A model against model
version of the same gate is NOT run, and the reason is that `w4_forcing` already
established that two independent free running draws agree to +0.0273 at 1.3 sd; a
second generation pass would buy an hour of GPU and no new information.

**Null sd** by bootstrapping traces within each length bin and repooling, 400
draws, so the error bar carries the pooling and the per trace variance together.

**Thermal.** One generation pass, roughly 55 minutes. The human side is decode only
and costs nothing. Watchdog armed by PID, `watchdog3.sh`, because matching on the
script name has now failed twice in two different ways.

DIAGNOSTIC ONLY, never a contract score. One trajectory per row, no selection, no
reranking. No serving change follows and no build is authorised by either outcome.
Phase conditioning and the spectral loss term remain NOT AUTHORISED.

### AMENDMENT to the position registration, made before the model side was generated

Three changes, all forced by a power calibration run on HUMAN DATA ALONE. No model
output existed when any of them was chosen, so none of them can have been tuned
toward an answer. That is the whole reason the calibration was run this way round.

**1. The estimator changes, the estimand does not.** The registration said fit a
line inside each trace and average those slopes. On human data that estimator has
a null sd of 0.58 against thresholds of 0.17 and 0.05, so every possible outcome
was a boundary refusal and the run would have been worthless. The cause is
structural rather than a shortage of data: the median trace covers only about a
third of the 0 to 1 position range, and a within trace regression divides by the
variance of that narrow span, which multiplies the noise. Replaced by pool then
fit. Every window from every trace contributes to a mean log band power profile
across position bins, the two populations are differenced inside each length bin
and pooled, and one weighted line is fitted to the pooled excess profile. Same
quantity, log band power per unit of normalised position. Null sd 0.089.

**2. The window drops from 64 samples to 32, and the hop from 8 to 4.** Null sd
0.089 to 0.047. It also admits shorter traces, so 73% of rows contribute instead
of 43%, which makes this population much closer to the one arm C measured rather
than a long tail of it. At 125 Hz a 32 sample window still resolves the 11 to 41.5
Hz band in 8 bins. Halving the hop again to 2 doubled the window count and moved
the null sd by 0.0006, which is what establishes that the remaining variance is
limited by the number of traces and not by the number of windows. There is no more
power to be had here by rearranging the estimator.

**3. Position bins 8, length bins 5.** Chosen among a set of configurations whose
null sds were 0.047 to 0.054, which is to say tied. Selection was on the null sd
and on nothing else; the half against half POINT estimate was not used to choose,
because picking the configuration whose gate happens to sit nearest zero would be
selecting on noise.

    config                          null sd, half against half
    pos 5, len 10, w 64, hop 8          0.0887
    pos 5, len  5, w 64, hop 8          0.0890
    pos 4, len  5, w 64, hop 8          0.1056
    pos 6, len  5, w 32, hop 4          0.0486
    pos 5, len  5, w 32, hop 4          0.0606
    pos 8, len  5, w 32, hop 4          0.0471   FROZEN
    pos 6, len  3, w 32, hop 4          0.0516
    pos 10, len 5, w 32, hop 4          0.0536
    pos 6, len  5, w 32, hop 2          0.0480

**What this test can and cannot resolve, stated now rather than discovered after.**
The frozen design gives a null sd of about 0.047 for two halves, so about 0.033 for
the real comparison, which uses the full population on both sides. Against that:

    ACCUM_DOMINATES 0.17 sits about 5 sd from zero. If accumulation dominates,
                    this test will say so cleanly, and it can equally rule that
                    out cleanly.
    FLAT 0.05       sits about 1.5 sd from zero. The boundary rule will refuse
                    any result between roughly 0.017 and 0.083, which is a wide
                    band around this threshold. This test CANNOT certify perfect
                    flatness. It can only bound the growth from above.

Both thresholds stand as registered. What is added is that a 2 sd interval on
SLOPE_DIFF is reported whatever branch fires, because a branch alone hides how far
the number could move, and near the FLAT threshold that is most of the information.

**One more correction that follows from the window change.** The thresholds were
anchored to a total excess of 0.1677 in logs, measured at w=64 with one centred
window per trace. This pipeline runs at w=32 over a longer reaching population and
its own total excess need not be that number. The pipeline now measures its own
total and quotes the growth share against both, saying which is which. The
registered thresholds are not moved, because moving a threshold after seeing the
data it will be applied to is the thing pre registration exists to prevent.

### The result. Neither branch. The model UNDER MODULATES its texture across a movement, 2026-08-05

`research/w4_position.py`, n=20000 drawing the same 18,081 held out rows, frozen
design w=32, hop=4, 8 position bins, 5 length bins. Result in
`research/w4_position.json`.

    H_human        13,215 traces, 73.1% of rows, 190,786 windows
    M_model_free   13,182 traces, 72.9% of rows, 190,913 windows

    VALIDITY   human half against human half  +0.0339 at +0.7 sd   PASS
    SLOPE_DIFF                                -0.1805 at -5.2 sd
    2 sd interval                             [-0.2504, -0.1106]

The two arms produce the same number of traces and the same number of windows to
within a fifth of a percent, so the duration confound the length matching exists to
control did not even arise.

**This is the registered THIRD outcome, and it is the reason it was registered.**
The result is strongly NEGATIVE. Neither hypothesis the run was built to separate
is right. It is not accumulation, which would have been positive. It is not a
uniform excess present from the start, which would have been zero. The excess
FADES along the movement.

    mean log band power in the 11 to 41.5 Hz band, by position along the path

    position       0.06     0.19     0.31     0.44     0.56     0.69     0.81     0.94
    human       -1.8145  -1.8644  -1.8495  -1.8094  -1.7360  -1.6359  -1.4824  -1.3564
    model       -1.5727  -1.7097  -1.7040  -1.6488  -1.5864  -1.4888  -1.4189  -1.3508
    excess      +0.2418  +0.1524  +0.1848  +0.1796  +0.1566  +0.1392  +0.0542  +0.0057

**Read the two raw rows, not just the difference, because that is where the
mechanism is.** A person's high frequency speed texture RISES steadily across a
movement, from -1.8145 at the start to -1.3564 at the end, a climb of +0.4581. That
is the corrective phase near the target and it is exactly what the human motor
literature would predict. The model climbs too, from -1.5727 to -1.3508, but only
by +0.2219, which is 48 percent of the human climb.

**So the model does not have too much texture in general. It has the wrong texture
PROFILE.** It launches a movement already about as jerky as a person is when
homing in on a target, and it then fails to build the way a person does. The two
arms end up in the same place, +0.0057 apart in the final bin, which is nothing.
Almost the entire measured gap between the model and people, in this band, is
spent in the first two thirds of a movement, and the largest single excess is in
the very first window.

Put in one line: **a person starts smooth and finishes jerky, the model starts
jerky and finishes the same way, and it modulates about half as much as it should.**

**What this rules out.** Accumulation is dead as an explanation of the band excess.
Whatever is wrong is not the model wandering into states it never saw, because a
drift story predicts the excess growing with distance from the start and the
measurement says it shrinks with a slope of -0.18 at 5.2 sd. This also sits
comfortably with `w4_drift` having priced the whole training time exposure fix
family at a third or less, and with `w4_dtcal` finding the one step conditional
calibrated. The model self corrects rather than drifting.

**What it points at, stated as candidates and not as a conclusion.** The onset of a
movement is where the model has the least context: no history at all, only the four
number conditioning vector. It is also where the AR pause shortfall would show, the
known 0.05 against 0.23, since a human's opening is where stillness lives. Neither
of those is established by this run and neither should be quoted as if it were.
What IS established is where to look, and it is the launch, not the drift.

**Method note, and it is the reason this run is worth anything.** The estimator and
the window were both changed after the registration, and both changes were forced
by a power calibration that ran on HUMAN DATA ALONE with no model output in
existence. The registered estimator had a null sd of 0.58 against thresholds of
0.17 and 0.05 and would have returned a boundary refusal no matter what the model
did. Calibrating power against a human against human null cannot leak an answer,
which is why it was done in that order. The full amendment, including the
configuration table and the explicit statement that this design CANNOT certify
perfect flatness, is in the amendment section above.

**Thermal and safety.** One generation pass, roughly 55 minutes, peak 79C MEASURED
this time, the first run in this sequence where the watchdog reported a real peak
rather than a spot reading. `watchdog3.sh` tracks the run by PID. The cooldown flag
raised at 79C. `best.pt` MD5 unchanged, eval data and scoring code untouched.

DIAGNOSTIC ONLY. No serving change follows. No build is authorised by this. Phase
conditioning and the spectral loss term remain NOT AUTHORISED.

## Is the LAUNCH conditional itself wrong, or only what the model does with it, 2026-08-05

PRE REGISTERED, written before any code exists.

**Where this comes from.** `w4_position` put almost the whole band excess in the
first two thirds of a movement, largest in the very first window, and showed the
model modulating its texture only 48 percent as much as a person across a
movement. That says the launch. It does not say whether the model's one step
PREDICTION at the launch is wrong, or whether the prediction is fine and only the
free running behaviour diverges. Those are different defects with different fixes
and the difference is cheap to settle.

**The instrument, and why it carries no construction artefact.** One teacher forced
forward pass over real held out sequences. At every position the model's own
predicted distribution is read against the REAL next token given the REAL history.
Nothing is generated, nothing hybrid exists, and the model is evaluated in exactly
the regime it was trained in. This is the `w4_dtcal` setup, which is the one
measurement in this workstream that has never had to be withdrawn. It costs a
single forward pass rather than an hour of sampling.

**What `w4_dtcal` did not cover.** It sliced the dt conditional by previous
interval, by interval two back, by real speed and by the model's own predicted
mean, and found it calibrated in every one. It never sliced by POSITION IN THE
SEQUENCE, and it never looked at the SPEED head or the DIRECTION head at all.
Those are precisely the gaps `w4_position` now points into.

**The primary statistic, chosen because it works for every head including the
circular one.** For a calibrated model the average negative log likelihood of the
real token equals the average entropy of the model's own predicted distribution.
Their difference estimates the average KL divergence from the truth to the model
and is non negative, so larger is worse and zero is perfect:

    KLhat(head, slice) = mean NLL of the real token - mean predicted entropy

Reported per head for s, th and dt, sliced by position index in the sequence:
0, 1, 2, 3, 4, 5 to 7, 8 to 11, 12 to 19, 20 to 31, 32 and beyond. This is
entropy free by construction, which matters because raw NLL must vary with
position for an honest reason: a person's speed at the very first event is near
zero and easy to predict, and mid movement it is not. A raw NLL curve would show a
rise with position in a perfect model. KLhat does not.

**Registered thresholds, expressed relative to the model's own mid sequence
performance so no absolute scale has to be assumed.** Let R = KLhat at position 0
divided by KLhat pooled over positions 12 and beyond, per head.

    R >= 2.0     the LAUNCH CONDITIONAL is itself materially worse. The defect is
                 in what the model predicts at movement onset and is directly
                 addressable in training.
    R <= 1.2     the launch conditional is FINE. The prediction is right and only
                 the free running behaviour diverges, which points at sampling
                 and at the joint rather than at any single conditional.
    in between   MIXED, report the curve and both numbers.
    BOUNDARY     within one null sd of either threshold the threshold call is
                 REFUSED and the in between case is reported.

**The secondary read, and it is the direct test of the w4_position mechanism.**
Randomised PIT for the two ordered alphabets, s and dt, per position slice. A PIT
histogram with a HUMP in the middle, end over mid below 1, means the model is OVER
DISPERSED, putting more spread on the next token than the truth has. `w4_position`
says the model launches too jerky, so an over dispersed speed conditional at
positions 0 to 3 would confirm that finding at the level of the one step
prediction and make it immediately actionable.

    end over mid <= 0.95 at positions 0 to 3 on the SPEED head
                 -> over dispersed at launch, consistent with w4_position
    end over mid >= 1.05
                 -> UNDER dispersed at launch, which would CONTRADICT w4_position
                    and would mean one of the two runs is wrong. Registered now so
                    that outcome cannot be quietly reinterpreted.

The direction head is circular, so a PIT over its alphabet is not meaningful and
none is computed. It gets KLhat only, and that is stated rather than the PIT being
computed anyway and read as if it meant something.

**Null sd** by bootstrapping whole SEQUENCES, 400 draws. Positions inside one
sequence are correlated and the resampling unit is never the position.

**A sanity floor that has to hold.** KLhat must be non negative at every slice
within its error bar. A materially negative KLhat means the estimator is broken,
not that the model beats the truth, and the run is reported as failed rather than
interpreted.

**Cost.** One forward pass, no generation, minutes. No thermal concern, but the
watchdog is armed by PID regardless.

DIAGNOSTIC ONLY, never a contract score. No serving change follows and no build is
authorised by either outcome. Phase conditioning and the spectral loss term remain
NOT AUTHORISED.

### CORRECTION to the launch registration, made on the smoke test before the deciding run

**The registered primary statistic was wrong and is withdrawn.** The registration
said that for a calibrated model the mean NLL of the real token equals the mean
entropy of the model's own predicted distribution, so their difference estimates a
KL divergence and is non negative. The first half is true and the second half does
not follow. Writing p for the truth and q for the model:

    mean NLL          = CE(p, q) = -sum p log q
    mean entropy      = H(q)     = -sum q log q
    their difference  = CE(p, q) - H(q) = KL(p || q) + H(p) - H(q)

That is a KL PLUS an entropy difference. It is not a divergence, it is not non
negative, and it goes negative whenever the model's own entropy exceeds the
truth's, which is exactly what an over dispersed model does. The statistic
therefore cannot distinguish "the conditional is wrong" from "the conditional is
wider than the truth", and those are the two things this run exists to tell apart.

**The registered sanity floor caught it.** The floor said a materially negative
value means the estimator is broken rather than the model beating the truth. On a
1,359 row smoke test it fired immediately, with the direction head negative at
EVERY position from -0.03 to -0.10 rather than at one, which is a systematic
signature and not noise. No GPU was wasted, no deciding run was made and nothing
was reported on this basis. That floor existed because the same class of error has
now bitten this workstream three times, and this is the first time it was caught
before rather than after.

**What replaces it.** The randomised PIT, which is entropy free and is the
instrument `w4_dtcal` already used correctly. Two scale free readings per head per
position slice, both directly comparable across slices of different size:

    TILT   |PIT mean - 0.5|, a bias in the conditional
    SHAPE  end density over middle density. Above 1 is a U and the model is UNDER
           dispersed and too confident. Below 1 is a hump and the model is OVER
           dispersed, hedging wider than the truth.

Standard caveat, stated rather than buried: PIT uniformity is a NECESSARY
condition for a correct conditional and not a sufficient one. Non uniformity
proves miscalibration; uniformity does not prove correctness. `w4_dtcal` lives with
the same caveat and it is the reason its result was phrased as "calibrated on every
axis tested" rather than "correct".

**The direction head keeps a PIT but loses the dispersion reading.** Uniformity
under any fixed ordering of the alphabet remains a valid test of miscalibration.
The U versus hump interpretation requires the alphabet to be ORDERED, and direction
is circular, so SHAPE is computed for s and dt only. This is a change from the
registration, which computed no PIT at all for th.

**New registered thresholds on the SPEED head, pooled over positions 0 to 3, which
is the launch.** `w4_dtcal`'s worst tilt over every slice it tested was 0.0062, so:

    TILT >= 0.02    a real bias at launch, three times anything previously seen
    TILT <= 0.008   no detectable bias at launch
    in between      MIXED

**The dispersion thresholds are UNCHANGED from the original registration**, because
they were sound and were written before any of this:

    SHAPE <= 0.95   OVER dispersed at launch, consistent with w4_position. The
                    model hedges wider than the truth at movement onset, which is
                    what too much launch texture looks like one step at a time.
    SHAPE >= 1.05   UNDER dispersed, which CONTRADICTS w4_position and means one
                    of the two runs is wrong.

SHAPE is read twice, against 1.0 and against the model's own mid sequence value
from position 12 on, so a global bias in the statistic cannot be mistaken for a
launch specific one. The boundary rule applies to every threshold above.

**Mean NLL and mean entropy are still reported by position, as DESCRIPTIVE
quantities only, and the printout says on its face that their difference is not a
divergence.** They are worth seeing. They decide nothing.

The R ratio and its 2.0 and 1.2 thresholds are WITHDRAWN entirely, since they were
defined on the broken statistic.

### The result. The one step conditional is CALIBRATED and is not the seat of the under modulation, 2026-08-05

Corrected estimator, n = 20000, 18,081 rows, 1,034,517 live positions. The
validity arm passed first and cleanly, before anything touching a real token was
read:

    head              n      tilt     shape
    s         1,034,517    0.0001     1.004
    th          934,604    0.0002     1.006
    dt        1,034,517    0.0003     0.998

Uniform by construction and uniform in fact, so the PIT path is sound. That is the
check the withdrawn floor should have been.

**The registered primary call REFUSES at the boundary.** Speed head tilt pooled
over positions 0 to 3 is 0.0084 against a clean threshold of 0.008, which is
0.0004 away with a bootstrap sd of 0.0011. Registered as in between, and it sits
hard against the clean end.

**The registered dispersion call is CLEAN and says NOT mis dispersed.** Absolute
shape at launch is 0.994, which is 0.044 from the over dispersed threshold and
0.056 from the under dispersed one against a bootstrap sd of 0.011. That is a
genuine in between and not a refusal. The relative read, 0.957 against mid
sequence, is a boundary and is refused.

So the registered answer is: **the speed conditional at launch is not detectably
biased and is not detectably mis dispersed.** w4_position's under modulation is
NOT a one step dispersion fault. Taken with `w4_dtcal`, which found the one step
timing conditional calibrated on every axis it tested, two independent
instruments now say the same thing about the conditional.

The by position table, registered in advance as reportable, carries the reason the
pooled number came out where it did:

    positions       tilt s    shape s   tilt th   tilt dt   shape dt
            0       0.0223      0.875    0.0065    0.0183      0.993
            1       0.0044      1.018    0.0041    0.0077      1.029
            2       0.0061      1.026    0.0007    0.0018      1.024
            3       0.0008      1.065    0.0041    0.0011      1.000
            4       0.0020      1.058    0.0033    0.0001      1.040
       5 to 7       0.0000      1.018    0.0047    0.0034      1.010
      8 to 11       0.0013      1.043    0.0055    0.0002      1.023
     12 to 19       0.0006      1.031    0.0040    0.0010      1.012
     20 to 31       0.0002      1.039    0.0040    0.0023      0.997
    32 to 255       0.0005      1.041    0.0039    0.0015      1.014

**From position 1 onward the conditional is calibrated to about a thousandth in
tilt with no position trend at all.** Shape sits mildly above 1 everywhere, 1.02 to
1.06, which is slightly UNDER dispersed and slightly over confident, and it is flat
across the whole sequence. Flat is the operative word. `w4_position` measured a
TREND across a movement, and there is no trend here to be its cause.

**Position 0 is a large exception and is POST HOC, so it is recorded and NOT
promoted.** The very first event alone carries tilt 0.0223 on speed and 0.0183 on
timing, against a thousandth or two everywhere after, and shape 0.875 where every
other position is above 1. Position 0 carries exactly one sample per sequence, so
there is no within sequence correlation and the null sd is analytic at 0.00215,
which puts the speed tilt at 10.4 sd. The pooled PIT mean of 0.4916 with tilts
under 0.006 at positions 1 to 3 places position 0's signed mean near 0.478, below
one half, and below one half means the real first event is SLOWER than the model
expects. Too fast and too wide at the very first event is the same direction as
`w4_position`'s "the model starts too jerky", read one event at a time.

It is not the same magnitude and it cannot be. `w4_position`'s first bin spans a
32 sample window at 125 Hz, which is 256 ms and many events, and the excess ran
across bins 0 through 6. One event cannot account for a profile that wide. So
position 0 is a real defect at 10.4 sd, it points the same way, and it is at most
a small part of the thing. Pooling 0 to 3 diluted it fourfold, which is why the
registered number landed on its threshold. The pooling was registered and the
verdict stands as refused; this paragraph does not overturn it.

**What this closes.** The one step conditional as the seat of the under modulation
is DEAD. Not the timing conditional, by `w4_dtcal`. Not the speed conditional, by
this run. Not accumulation or drift, by `w4_position` at 5.2 sd. Not exposure, by
`w4_drift` pricing that family at a third or less. The defect is in the JOINT, in
what free running sampling composes out of conditionals that are each individually
calibrated, and that is a much harder object than any of the things now ruled out.

Phase conditioning and the spectral loss term remain NOT AUTHORISED. Nothing here
authorises a build.

## Does the model CONDITION its texture on the history, or regress to the mean, 2026-08-05

Pre registered. Thresholds fixed before the file exists.

**Why this question and not another.** The chain rule is unforgiving. If every
conditional q(x_t | history) equalled p(x_t | history) for every history, the
joints would be identical and there would be no defect. `w4_launch` found the
conditionals calibrated, so exactly one of two things must be true:

    (a) the conditionals are calibrated MARGINALLY but wrong CONDITIONALLY, and
        the residual compounds over a hundred steps
    (b) the conditionals are right on REAL histories and wrong on the histories
        the model itself generates

Branch (b) is exposure bias in its precise form and it is not directly
measurable, because it needs the human continuation of a machine written prefix
and no such thing exists. `w4_drift` already priced that whole family at a third
or less of the gap. Branch (a) IS measurable and has never been tested here.

**The gap a PIT cannot see.** A randomised PIT pools over every history at a given
position. A model can be perfectly uniform there while being wrong after every
individual history, as long as the errors cancel in the pool. Too narrow after
volatile histories and too wide after calm ones averages to uniform. That specific
failure has a name, regression to the mean, and it is what under modulation IS.
A model that ignores how textured the movement has been so far and emits average
texture will look calibrated at every position and will produce the flat profile
`w4_position` measured.

**The instrument.** The same teacher forced forward pass, re sliced. Instead of
slicing the PIT by position, slice it by a property of the REAL HISTORY: the local
texture of the previous 8 events, measured as the mean absolute successive
difference of the speed class. Quintiles of that covariate. A correct conditional
is uniform inside EVERY slice of ANY function of the history, so any structure
across quintiles is a conditional error.

**The statistic is SHAPE, not tilt, and the reason matters.** Regression to the
mean is not a directional bias, it is a width error whose direction depends on
which way the truth happened to move, so it cancels in the PIT MEAN. It does not
cancel in the width. After a volatile history a mean regressing model is too
NARROW and its PIT is a U with shape above 1. After a calm history it is too WIDE
and its PIT is a hump with shape below 1.

    D = shape(top quintile) - shape(bottom quintile), speed head

    D >= 0.06    REGRESSION TO THE MEAN. the model does not condition its texture
                 on the history's texture, which is under modulation seen one step
                 at a time and is addressable in training
    D <= 0.02    the model DOES condition its texture properly, branch (a) is dead
                 as well, and the defect is in a place none of these instruments
                 has yet reached
    in between   MIXED
    BOUNDARY     within one bootstrap sd of a threshold the call is REFUSED

**Position is confounded and is controlled by construction.** Volatile histories
happen later in a movement, and shape drifts mildly with position, 1.02 to 1.06 in
`w4_launch`. So D is computed WITHIN each position band, 8 to 11, 12 to 19, 20 to
31, and 32 on, and pooled across bands. A slope that is really position in
disguise cannot survive that.

**Validity arm, and here it is a real one.** The same quintile slicing applied to
tokens drawn from the model's own predictive law at each position. The covariate
comes from the real history and is IDENTICAL between the two arms, so if the
estimator manufactures a slope it will manufacture it there too. D on the validity
arm must be flat within noise. If it is not, the run is reported as failed rather
than interpreted. This is what the withdrawn floor should have been and it is the
same arm that passed cleanly in `w4_launch`.

The direction and timing heads are reported and are not what the thresholds were
written for. `w4_position` measured speed texture and nothing else.

Same caveat as always, stated rather than buried. Uniformity inside every slice is
still only a NECESSARY condition. A flat D proves the model conditions on THIS
function of the history, not on every function of it.

DIAGNOSTIC ONLY, never a contract score. No serving change follows and no build is
authorised by either outcome. Phase conditioning and the spectral loss term remain
NOT AUTHORISED.

### AMENDMENT to the conditional texture registration, made on the smoke test before the model side was read

Two changes, both made while the run was still aborting at the validity arm and
before any real token statistic had been printed.

**One, the validity criterion is bootstrapped rather than a fixed constant.** The
registration says the validity arm must be "flat within noise". The script
implemented that as a fixed 0.02, which is not the same thing and is below the
noise floor at small n. On 2,717 rows the arm read -0.0187, +0.0072 and +0.0222
across the three heads, which is what pure noise looks like at that size: the
shape statistic uses four histogram bins at each end against four in the middle,
so with a few thousand samples per cell its sd alone is near 0.03. The criterion
is now |D| greater than two bootstrap sd, which is the registered wording
implemented correctly. This cannot leak the answer, because the validity arm never
touches a real token.

**Two, n rises from 20,000 to 100,000.** This is a teacher forced forward pass with
no generation, so it costs minutes rather than the near hour a generation arm
costs, and 2.5 million held out rows are available. The projected pooled sd falls
from roughly 0.028 at smoke size to roughly 0.005, which puts the registered
thresholds of 0.02 and 0.06 at about 4 and 12 sd rather than on the noise floor.

**Nothing else moves.** The statistic is still end density over middle density, the
thresholds are still 0.06 and 0.02, the position banding is unchanged, and the
boundary rule stands. Both changes were chosen from the validity arm and the
arithmetic of the estimator, with the real token side never printed.

### The result. No texture conditioning fault. The model is mildly and UNIFORMLY too confident, 2026-08-05

n = 100000, 90,497 rows, 5,135,552 live positions, 4,411,576 with a complete 8
event history. Validity arm first, and it passed on every head at 0.2, 0.6 and 1.0
sd, which says the amendment was the right call and the slicing manufactures
nothing.

**The registered call REFUSES at the boundary.** Speed head D is +0.0231 with a
bootstrap sd of 0.0047, which is 0.0031 from the "conditions properly" threshold of
0.02. Reported as the in between case.

**The table says more than the scalar, and it does not say regression to the
mean.** Regression to the mean predicts a MONOTONE rise in shape from the calmest
quintile to the most volatile. That is not what is there:

    s head            Q1       Q2       Q3       Q4       Q5         D
    positions
     8 to 11       1.034    1.039    1.028    1.035    1.021   -0.0128
    12 to 19       1.001    1.042    1.037    1.036    1.032   +0.0317
    20 to 31       1.028    1.041    1.037    1.030    1.027   -0.0014
    32 to 255      1.014    1.040    1.025    1.037    1.047   +0.0334

Two bands slope up, one is flat, one slopes down. Within the bands that do slope,
the movement is Q1 sitting below the others rather than a rise across the row.
There is no monotone texture dependence, and the pooled D is carried by the
largest band. The dt head is flatter still, pooled D -0.0081, with every band
between -0.012 and -0.000.

**What IS there is flat.** Every cell of that table sits between 1.00 and 1.05.
The speed conditional is mildly UNDER dispersed, about three percent too narrow,
and it is three percent too narrow everywhere: at every position, after calm
histories and after volatile ones alike. `w4_launch` saw the same thing sliced by
position, 1.02 to 1.06 from position 1 on with no trend. Two different slicings
now agree that the miscalibration is a uniform mild over confidence and not a
conditional failure.

**So neither branch survives on the evidence available.** Branch (b), exposure, was
priced at a third or less by `w4_drift`. Branch (a), a conditional that is
marginally right and conditionally wrong, is not detected against the history's
own texture. The registered caveat is now the operative one: a flat D proves the
model conditions on THIS function of the history and not on every function of it.

**Post hoc, recorded and NOT promoted.** The direction head's signed tilt is
negative everywhere and shrinks monotonically across texture quintiles in all four
bands, from about -0.007 after the calmest histories to about -0.003 after the
most volatile. It is consistent in sign and ordering across every band, which
noise would not be, and it is tiny. It was found by looking, the thresholds were
not written for it, and the direction head is not what `w4_position` measured.

**The next slice is PROGRESS, and it is the one that maps onto the finding.** The
covariate tested here was the history's recent texture. `w4_position` did not
measure texture against texture, it measured texture against POSITION WITHIN THE
MOVEMENT, and found the human curve climbing twice as fast as the model's. The
model can only produce that curve if it knows how far through the movement it is,
which it must compute from the history plus the four conditioning numbers. Nothing
has yet asked whether it does. That is the same cheap forward pass sliced a third
way and it is the direct one step image of the defect.

Phase conditioning and the spectral loss term remain NOT AUTHORISED.

## Does the model condition on PROGRESS through the movement, 2026-08-05

Pre registered. Thresholds fixed before the file exists, and two of the three are
carried over unchanged from `w4_condtex` because the statistic and the sample size
are identical.

**Why progress and why it is a FAIR covariate.** `w4_position` measured texture
against elapsed fraction of the movement and found the human curve climbing twice
as fast as the model's. `w4_condtex` sliced by the history's own recent texture and
found nothing. Progress is the covariate that actually matches the defect and it
has never been sliced.

It is fair because the model is TOLD the answer. `training/prepare_events.py`
builds `events_cond.npy` as [log_dist, log_dur, cos, sin] with log_dur the log of
the movement's total duration in seconds, and `events_dt.npy` is the same
durations in milliseconds. So

    progress(t) = (sum of dt over events before t) / (exp(cond[1]) * 1000)

is a function of the HISTORY and the CONDITIONING, both of which the model has at
every step. This is the difference between a valid conditional calibration test
and an invalid one. Slicing a PIT by a quantity that depends on the future proves
nothing, because the conditional was never required to be uniform across slices of
the future. Slicing by a quantity the model was handed is a real test.

**Two co primaries, because betting on one moment risks missing the defect.** On
the speed head, across progress quintiles:

    D    = shape(top quintile) - shape(bottom quintile)
    Trng = max tilt across quintiles - min tilt across quintiles

    D >= 0.06      or  Trng >= 0.02    the conditional depends on progress WRONGLY
    D <= 0.02      and Trng <= 0.008   the model conditions on progress correctly
    otherwise      MIXED
    BOUNDARY       within one bootstrap sd of a threshold that call is REFUSED

The D thresholds are `w4_condtex`'s, unchanged, and its measured bootstrap sd of
0.0047 at this sample size is what makes them 4 and 12 sd from zero. The tilt
thresholds are `w4_launch`'s, unchanged.

**Two independent control schemes, and the verdict requires BOTH.** Progress is
confounded twice over and each control kills one confound:

    within POSITION bands   progress varies because total duration varies, so a
                            slope here could be the model treating short movements
                            differently from long ones
    within DURATION bands   progress varies because position varies, so a slope
                            here could be position, which w4_launch already showed
                            is flat

A real dependence on progress must appear in both. **If the two schemes disagree in
sign or only one clears its threshold, the registered answer is CONFOUNDED**, and
that is reported as the result rather than the more interesting of the two.
Position 0 is excluded, since progress is identically zero there and `w4_launch`
already showed that position is anomalous on its own.

**Validity arm, in both schemes.** Tokens drawn from the model's own predictive law,
sliced by the same progress quintiles from the same real histories. Flat within two
bootstrap sd or the run is reported as failed rather than interpreted. This is the
arm that passed at 0.2, 0.6 and 1.0 sd in `w4_condtex`.

**A descriptive panel that should agree with the shape reading, and a warning if it
does not.** Two curves against progress quintile:

    what humans DID       mean |s(t) - s(t-1)| over real tokens
    what the model EXPECTED   mean over the model's predicted law of the same thing

That is `w4_position`'s curve rendered one step at a time and in the token
alphabet rather than in the spectrum. If the human curve climbs and the model's
expectation does not, the model under predicts step to step change late in a
movement, and the PIT shape MUST then read above 1 at high progress. The two
readings are linked by arithmetic, so **if the panel and the shape disagree, the
estimator is wrong and neither is reported.** It decides nothing on its own.

DIAGNOSTIC ONLY, never a contract score. No serving change follows and no build is
authorised by any outcome. Phase conditioning and the spectral loss term remain
NOT AUTHORISED.

### CORRECTION to the progress rule, and the result, 2026-08-05

**The printed verdict is WITHDRAWN. The registered rule was one sided and could
not express "flat".** It read

    D <= 0.02   the model conditions on progress correctly

which is satisfied by any negative number, however large. Regression to the mean
predicts a POSITIVE D and that is the shape the threshold was written for, so the
possibility that the slope ran the other way was never encoded. The measured D was
-0.0301, which is 6.3 sd from zero, and the script duly printed "THE MODEL
CONDITIONS ON PROGRESS CORRECTLY". It does not. Flat means near zero in either
direction, so the rule now reads |D|, and the correction moves the answer AWAY
from a clean conclusion rather than toward one.

**Corrected registered verdict: BOUNDARY, the call is REFUSED.** Under |D|, the
position controlled scheme is MIXED at 0.0301 and the duration controlled scheme is
a boundary at 0.0164 against a threshold of 0.02 with a bootstrap sd of 0.0044. The
rule requires both schemes to agree and they do not clear it. Both tilt ranges are
clean at 0.0028 and 0.0037 against a threshold of 0.008, so there is no progress
dependent BIAS. Whatever is there is a width effect.

**What the numbers say outside the verdict.** The effect is small, consistent, and
in the opposite direction to the one registered:

    control by position   D -0.0301  sd 0.0048   6.3 sd from zero
      cells   -0.0222  -0.0384  -0.0343  -0.0327  -0.0454  -0.0238
    control by duration   D -0.0164  sd 0.0044   3.7 sd from zero
      cells   -0.0330  -0.0429  -0.0041  -0.0116  -0.0123

All eleven cells across both control schemes are negative. The validity arm's
largest excursion anywhere was 0.0058, so the noise floor is about a fifth of the
position controlled effect. Negative D means shape is higher at LOW progress: the
speed conditional is about three to four percent too NARROW early in a movement and
close to calibrated late. Read plainly, real human behaviour early in a movement is
more variable than the model expects.

**The descriptive panel agrees, which is the check that says the estimator is not
lying.** Mean absolute step to step change in speed classes:

    quintile          n      human      model    ratio
          Q1  1,009,001    11.5194    11.4707   0.9958
          Q2  1,009,020    11.5539    11.5126   0.9964
          Q3  1,009,012    11.2520    11.2092   0.9962
          Q4  1,009,011    10.8933    10.8653   0.9974
          Q5  1,009,011    10.4160    10.4020   0.9987

The model tracks the human curve to within half a percent at every progress level,
and the shortfall is largest at low progress and smallest at high progress, which
is the same ordering the shape reading gives. The registered linkage holds: the two
readings agree, so neither is discarded.

Note also that this human curve FALLS with progress, 11.52 down to 10.42, while
`w4_position`'s human band power RISES with progress. There is no contradiction,
because these measure different things. `w4_position` reads 11 to 41.5 Hz power in
a trace resampled to a uniform 125 Hz clock, and this reads change between
consecutive EVENTS, which are not uniformly spaced in time. Events crowding
together late in a movement raises band power while step to step change falls. The
two are only reconcilable through dt, and that is a fact about the instruments, not
a finding.

**Where this leaves the search.** Progress does not join recent texture and exposure
as fully dead, but it does not explain the gap either. A three percent width error
concentrated early is real at 6.3 sd and is far too small to be the whole of a
movement level defect that `w4_position` measured at 5.2 sd on a completely
different scale. Every one step instrument built so far, on every slicing tried,
returns the same picture: a conditional that is correct to within a few percent
almost everywhere. The defect continues to sit in the joint.

Phase conditioning and the spectral loss term remain NOT AUTHORISED.

## Are the model's errors INDEPENDENT across steps, as a correct model requires, 2026-08-05

Pre registered. Thresholds fixed before the file exists.

**Why this is the right instrument and why it is sharper than the likelihood gap.**
Everything measured so far is a MARGINAL property: the conditional is uniform
within this slice, within that slice. Marginal calibration at every step does not
imply a correct joint, and the thing it fails to constrain is exactly the
DEPENDENCE between steps. That is where a three percent per step error becomes a
movement level defect, and it is the only place left for it to be.

There is an exact property to test. Under a correct model the randomised PIT values
u(t) are not merely uniform, they are INDEPENDENT across t. This is the Rosenblatt
transform and it is an identity, not an approximation: if q equals p then the
sequence of u values is i.i.d. Uniform. So any autocorrelation in u PROVES the
joint is wrong, and it proves it without needing to know p, without generating
anything, and without the construction artefact that withdrew arms G and E.

Marginal uniformity has been verified to about a thousandth in tilt. Independence
has never been looked at.

**The primary is the VOLATILITY version, and the reason is the whole point.** Two
autocorrelations at lag 1 on the speed head:

    rho_level = autocorrelation of u(t)
    rho_vol   = autocorrelation of |u(t) - 0.5|

rho_vol is the one that matters. It measures whether being surprised at step t
predicts being surprised at step t+1, which is volatility clustering, which is
texture. A person's jitter comes in bursts. If the model reproduces the marginal
amount of jitter but not its clustering, every marginal instrument passes and the
assembled movement is smooth in the wrong way. That is the exact shape of
`w4_position`'s finding and no test so far could have seen it.

    either rho >= 0.05    MATERIAL residual dependence, the joint is wrong in its
                          step to step structure and that is addressable in
                          training
    both rho <= 0.01      no residual dependence at lag 1
    otherwise             MIXED
    BOUNDARY              within one bootstrap sd of a threshold the call is
                          REFUSED

These are materiality thresholds and not significance thresholds, deliberately. At
five million positions the analytic null sd is about 0.0005, so almost any
autocorrelation is statistically significant and the question is whether it is big
enough to matter. Bootstrap is over whole SEQUENCES, since positions inside one are
the very thing being measured.

**The full lag profile from 1 to 8 is reported**, because the decay shape says
whether this is a one step coupling or a long memory the model is missing, and
those are different problems.

**Validity arm, and here it is exact.** The self sampled tokens' PIT values are
i.i.d. BY CONSTRUCTION, so their autocorrelation is the exact null for this
statistic including the dilution the randomisation introduces. It must sit within
two bootstrap sd of zero at every lag or the run is reported as failed rather than
interpreted.

**The test is conservative and cannot manufacture an effect.** The randomised PIT
adds independent uniform noise, which dilutes any true autocorrelation toward zero.
It can understate the dependence. It cannot invent it. The self sampled arm carries
the identical dilution, so the comparison is fair.

Autocorrelation is computed WITHIN sequences only, never across a boundary, and
only where both endpoints are live. The timing head is reported alongside and the
direction head is reported for the level statistic only.

DIAGNOSTIC ONLY, never a contract score. No serving change follows and no build is
authorised by any outcome. Phase conditioning and the spectral loss term remain
NOT AUTHORISED.

### AMENDMENT to the independence validity gate, made on the smoke test before the model side was read

Two defects in the gate, both found while it was still running on 3,000 rows and
both fixed before any full sample result existed.

**One, the gate could silently no op.** `boot_sd` asked for more bootstrap draws
than it was given and returned nan, and every downstream test was written as
`sd > 0`, which nan fails. So a missing sd disabled the check instead of tripping
it, and the arm printed "worst 0.0 sd" and passed vacuously on every series. An
unmeasurable sd is now a FAILURE. A gate that can no op is worse than no gate, and
this workstream has already lost two claims to instruments that were not checked.

**Two, the registered rule was a multiplicity error.** The registration said the
arm must sit within two bootstrap sd of zero AT EVERY LAG, across eight lags. Under
the null that fails about thirty one percent of the time on a perfect estimator,
because it compares a maximum over eight tests against a threshold meant for one.
A gate that cries wolf on a third of correct runs is not a gate.

The corrected rule is two gates and both must hold:

    lag 1        |z| <= 2.0   a single test at the lag the verdict is read on
    every lag    |z| <= 3.0   a family of eight, about two percent family wise

**This makes the gate less likely to fire, which is the direction that favours
reading the result, so it is stated plainly rather than buried.** The
justification is the arithmetic of the null and not the answer: the correction was
derived from the self sampled arm alone, which never touches a real token, and the
real token side had not been printed. The tightening at lag 1 is the part that
matters, since lag 1 is the only lag the registered verdict uses.

Nothing else moves. The materiality thresholds of 0.05 and 0.01 stand, the primary
is still rho_vol at lag 1 on the speed head, and the boundary rule stands.

### SECOND AMENDMENT to the independence validity gate, and a note on the direction of travel

**This is the second time this gate has been loosened, which is exactly the pattern
that should be distrusted, so the reasoning is written out rather than asserted.**

The gate failed on dt:vol at lag 3, at 3.04 sd against a family threshold of 3.0.
Five series are tested at eight lags each, which is forty tests, and at a three sd
per test threshold about ten percent of PERFECT estimators produce at least one
excursion. So the multiplicity error the first amendment fixed within a series
still exists across series.

**The fix is NOT another significance threshold.** Chasing the correct sigma for
forty tests would be a third loosening of the same kind and would eventually
rationalise the gate out of existence. The right rule is the one this whole test
already runs on and states in its own docstring: MAGNITUDE, not significance.

    a validity excursion fails the gate only if it is both statistically
    detectable AND at least one fifth of the NEGLIGIBLE threshold, which is 0.002

The justification is arithmetic. The verdict's smallest meaningful level is 0.01
and its materiality level is 0.05. An estimator bias bounded below 0.002 cannot
create, mask, or move a reading at either. One fifth of the smallest meaningful
level is the line, chosen as a fraction of an already registered number rather than
fitted to what was observed.

**The observed value is stated so the reader can judge that for themselves.** The
failing excursion was +0.0014 on dt:vol at lag 3. That is 0.0014 against a
materiality threshold of 0.05, so it is thirty five times below the level that
would matter, and it is on the TIMING head at lag 3 while the registered verdict is
read on the SPEED head at lag 1. The speed head's own validity readings at lag 1
are 0.7 sd on vol and 0.1 sd on level, both clean under every version of this gate.

**What this amendment does not do.** It does not touch the materiality thresholds
of 0.05 and 0.01, the primary statistic, the boundary rule, or the lag profile.
Both the failure and this reasoning stay in the record next to whatever the run
returns, rather than the run being presented as having passed cleanly.

### THIRD AMENDMENT to the independence test. The primary statistic is replaced, and this one TIGHTENS the test, 2026-08-05

Two things happened here in order, and the order matters, because the first one
was wrong.

**What prompted it.** The registered primary is the autocorrelation of the
randomised PIT. The randomisation is what makes the PIT exactly uniform on a
discrete alphabet, and it does that by adding noise that is independent by
construction, which dilutes the very dependence the test exists to find. The
registration called the test CONSERVATIVE and left it there. It never measured how
conservative. A null on an instrument of unknown power is close to worthless, so I
measured it against the mid PIT, which is the same quantity with no added noise.
The mid PIT read +0.0549 on the speed volatility series against the randomised
+0.0085, a factor of seven, and +0.3159 on timing against +0.0017, a factor of a
hundred. The undiluted speed reading sat ABOVE the materiality threshold of 0.05.
On that basis the registered null looked like an artefact of a blunt instrument.

**Why the first replacement was wrong.** I then built the obvious null for the mid
PIT: the same statistic on a token drawn from the model instead of from the truth,
and the difference between the two. That difference is BIASED, and it is biased in
the direction that would have manufactured a finding.

In the self sampled arm the drawn token is thrown away. It never enters the
history, so it cannot influence the model's distribution at the next step. In the
real arm the token IS the history. A surprising token at step t changes what the
model predicts at step t+1, so surprise at t and surprise at t+1 are correlated
even under a PERFECT model. That correlation is real, it is not a defect, and the
self sampled arm does not contain it, so subtracting one from the other leaves it
behind as a false positive.

The arithmetic says the bias falls entirely on the volatility series. The level
series is exactly unbiased, because the mid PIT has conditional mean one half
identically, which makes it a martingale difference and forces every
autocorrelation to zero. The volatility series has no such guarantee, because the
expected surprise magnitude depends on the distribution at that step. The smoke
test showed exactly that signature and I should have read it as a warning rather
than as a result: all three level series flat at +0.0017, +0.0024, -0.0053, and
both volatility series positive at +0.0211 and +0.0849.

**The replacement that is correct.** Centre each series on its own conditional
mean, which is available in closed form from the model's distribution at that step
and costs one weighted sum over the alphabet.

    e(t) = m(t) - 1/2                        the level residual
    r(t) = |m(t) - 1/2| - E[|m - 1/2| ; p]   the volatility residual

Both have conditional mean zero given the history by construction, and both are
functions of the history one step later, so under a correct model every
autocorrelation at every lag is EXACTLY zero. That is the same identity the
Rosenblatt argument gave, with none of the added noise and none of the feedback
bias. No null arm is needed for the verdict. The self sampled arm stays as the
estimator's validity check, where it is exact.

**What the centring did to the alarm that started this.** It removed it. On the
same smoke sample the centered speed volatility residual reads +0.0093 against the
randomised +0.0085, so the registered instrument was keeping ninety one percent of
the signal, not fifteen. On timing it kept twenty three percent, not one. The
seven fold and hundred fold dilutions I measured were almost entirely the context
contamination and the feedback term, not lost signal. The registered primary was
therefore never as blind as the mid PIT comparison implied, and the +0.0549 that
looked like a material finding was an artefact of an uncentered statistic.

**A note on the direction of travel, since this is the third amendment to the same
file.** The first two loosened the validity gate. This one does not. It replaces
the primary with a strictly sharper statistic, and it also removes a finding that
the intermediate version would have reported. Both amendments before it were made
on the smoke test before the model side was read. This one was made on the smoke
test as well, but only after a first replacement had already been coded and run,
and that replacement is written down above rather than deleted, because a wrong
statistic that was nearly promoted is part of the record.

**What this amendment does not change.** The materiality thresholds of 0.05 and
0.01, the boundary rule, the bootstrap over whole sequences, the lag profile, the
two validity gates, and the magnitude floor of 0.002 all stand as registered. The
randomised series is still computed and reported next to the primary, so the
dilution is a number in the output rather than an assumption in the prose.

### The result. The errors ARE independent across steps, and the dilution alarm was false, 2026-08-05

Run at n=100000, 90,497 rows, 5,135,552 live positions, 4,645,723 of them with a
direction to predict. The same rows w4_progress used. Peak GPU 74C, no cooldown
required. Checkpoint MD5 unchanged, evaluation data and scoring code untouched.

**Validity first.** All ten series, at all eight lags, read 0.0 sd on both gates.
The largest validity excursion anywhere in the table is 0.0014. The estimator does
not manufacture dependence.

**The primary.** Speed head, lag 1, centered residual:

    rho_vol     +0.0033   bootstrap sd 0.0005    negligible
    rho_level   -0.0005   bootstrap sd 0.0005    negligible

Both are fifteen times below the NEGLIGIBLE threshold of 0.01 and fifty times below
MATERIAL. The registered call is NO RESIDUAL DEPENDENCE AT LAG 1.

**The dilution alarm is retracted, with numbers.** The whole third amendment was
prompted by a mid PIT reading seven times larger than the randomised one on speed
and a hundred times larger on timing. Against the properly centered statistic the
factor is gone:

          series   centered   randomised     kept
         s level    -0.0005      -0.0006     1.03
           s vol    +0.0033      +0.0034     1.06
        th level    +0.0048      +0.0028     0.59
        dt level    -0.0049      -0.0026     0.54
          dt vol    +0.0093      +0.0029     0.31

On the head the verdict is read on, the registered instrument was keeping all of
the signal, not fifteen percent. The seven fold and hundred fold gaps were the
uncentered statistic measuring the model's own predictive distribution moving with
context, which is not an error, plus the feedback term described in the amendment.
The registered primary was never blind. Only on the timing head does centring
change the reading materially, and there it triples a number that is still below
the negligible level.

**What is nonetheless there, stated rather than buried.** At five million positions
the bootstrap sd is 0.0005, so +0.0033 is six and a half sd from zero. Under a
correct model this quantity is EXACTLY zero, so the joint is provably not exactly
right. The speed volatility profile is positive at every one of the eight lags and
decays slowly, 0.0033, 0.0039, 0.0031, 0.0028, 0.0024, 0.0021, 0.0019, 0.0018,
which is the signature of genuine volatility clustering that the model
under produces. The timing head is the largest at +0.0093. This is a real defect
and it is a third of a percent, against a registered materiality level of five
percent that was fixed before the file existed. It is not the defect being hunted.

**Where this leaves the search.** Four instruments have now been pointed at the one
step conditional from four directions: launch, history texture, progress through
the movement, and step to step independence. All four come back clean or near
clean, and the independence test is the one that could see structure the other
three cannot, because marginal calibration at every step does not constrain the
joint. It does not find it either. The remaining common finding across all four is
a uniform three to four percent under dispersion, too small and too flat to be the
5.2 sd defect w4_position measures.

Phase conditioning and the spectral loss term remain NOT AUTHORISED. No serving
change follows from this run and no build is authorised by it.

## Are the three channels wrong TOGETHER, 2026-08-05

Registered before the file exists. Thresholds fixed here.

**The gap this closes.** Every instrument so far has looked at speed, direction and
timing ONE AT A TIME. w4_launch, w4_condtex, w4_progress and w4_indep are all per
head. None of them can see whether the three channels are wrong together, and the
contract scorer's features are mostly about how the three relate to each other:
curvature is direction against speed, the velocity profile is speed against time.
A model can be perfectly calibrated on each channel separately and still couple
them wrongly. That is the one direction nothing has looked at.

**The identity, which is the same one that made w4_indep exact.** The model emits
in the order s, then th given s, then dt given s and th. So given the history at
step t:

    r_s(t)   is a function of the history and s(t)
    r_th(t)  has conditional mean zero given the history and s(t)
    r_dt(t)  has conditional mean zero given the history, s(t) and th(t)

Each residual is centred on a conditional that the earlier ones are already
measurable with respect to. So every cross channel correlation is EXACTLY zero
under a correct model, at lag zero and at every lag in both directions. No
generation, no null arm needed for the verdict, no construction artefact.

**Three residual kinds per head.**

    lvl(t) = m(t) - 1/2                        signed surprise, m is the mid PIT
    vol(t) = |m(t) - 1/2| - E|m - 1/2|         surprise magnitude
    srp(t) = -log p(k(t)) - H(p(t))            surprise, ORDERING FREE

The primary is srp. lvl and vol both depend on where the alphabet is cut, which is
arbitrary for the circular direction head, so they are reported as profile and the
verdict is not read on them. srp has conditional mean zero because the expected
negative log probability under p is exactly the entropy of p.

**A distinction that has to be stated, given the history.** srp is built from the
same two quantities as the WITHDRAWN KLhat estimator, negative log probability and
entropy. KLhat was withdrawn because their MEANS do not form a divergence: the
difference is a KL plus an entropy gap, so it is not non negative and goes negative
exactly when the model is too wide. None of that is being used here. srp is used
only as a conditionally mean zero residual inside a CORRELATION, and the property
being relied on is that E[srp | history] is exactly zero, which is true and was
never the disputed part. Its mean is not interpreted as a distance from anything.

**The primary reading.** The three lag zero cross channel correlations of srp:

    s with th,   s with dt,   th with dt

    any |rho| >= 0.05    MATERIAL cross channel dependence
    all |rho| <= 0.01    the channels are not coupled wrongly at lag zero
    otherwise            MIXED
    BOUNDARY             within one bootstrap sd of a threshold the call is REFUSED

**Thresholds are the same numbers w4_indep used, deliberately.** 0.05 and 0.01 on
the same correlation scale, so the two readings are directly comparable and neither
is fitted to what was observed. Bootstrap over whole SEQUENCES.

**Profile, not verdict.** Lags 1 through 4 in both directions, all three pairs, all
three kinds.

**VALIDITY ARM.** Tokens drawn from the model at each step, each head from its own
conditional given the REAL preceding tokens. Given the history and the real earlier
tokens the drawn heads are independent, so every cross correlation in this arm is
an exact zero and any deviation is estimator error. It is a weaker null than the
real arm in one respect worth stating: it does not reproduce the real arm's chain,
because the drawn token is not what the next head conditions on. It is an exact
null for the ESTIMATOR, which is what a validity arm is for.

Gates carried over from w4_indep without change, including the second amendment:

    lag 0      |z| <= 2.0   a single test, at the lag the verdict is read on
    every lag  |z| <= 3.0   the family
    magnitude  an excursion below 0.002 does not fail the gate
    an unmeasurable bootstrap sd is a FAILURE, not a pass

**DIAGNOSTIC ONLY.** Never a contract score. No serving change follows and no build
is authorised by any outcome. Phase conditioning and the spectral loss term remain
NOT AUTHORISED.

### The result. MATERIAL. The three channels ARE wrong together, and it is the first material finding in this whole sequence, 2026-08-05

Run at n=100000, 90,497 rows, 5,135,552 live positions, 4,645,723 with a direction.
The same rows w4_indep used. Peak GPU 74C, no cooldown required. Checkpoint MD5
unchanged, evaluation data and scoring code untouched.

**Validity first, and it is the cleanest arm any of these tests has produced.**
All nine pair and kind combinations, at all nine lags, read 0.0 sd. The largest
value anywhere in the eighty one cell validity table is 0.0010. The smoke test
excursion at 2.0 sd was noise and it vanished at scale exactly as more data
predicted, which is why the gate was not touched.

**The primary.** Surprise, lag zero, the three channel pairs:

    s with th    +0.1309   bootstrap sd 0.0008   MATERIAL
    s with dt    +0.0619   bootstrap sd 0.0008   MATERIAL
    th with dt   +0.0245   bootstrap sd 0.0007   mixed

Under a correct model each of these is EXACTLY zero. The speed and direction pair
is one hundred and sixty four bootstrap sd from zero and two and a half times the
registered materiality threshold. The registered call is MATERIAL CROSS CHANNEL
DEPENDENCE.

**The shape of it, which is as informative as the size.**

The coupling is entirely CONTEMPORANEOUS. Every cross lag entry, at lags one
through four in both directions, sits between -0.0017 and +0.0069 against a lag
zero reading of +0.1309. Whatever this is, it happens inside a single step and
does not propagate. That rules out a slow drift and it rules out the model
mistiming a coupling it otherwise has.

It shows up in SURPRISE and almost not at all in the mid PIT residuals:

    s with th    srp +0.1309   vol +0.0086   lvl -0.0032
    s with dt    srp +0.0619   vol +0.0204   lvl +0.0015
    th with dt   srp +0.0245   vol +0.0099   lvl -0.0033

That contrast is a mechanism hint. The mid PIT residuals are bounded in a half
unit interval, so they weight all events alike. Surprise is a log probability, so
it is dominated by RARE events. A coupling that is fifteen times larger in log
space than in bounded space is a coupling concentrated on rare moments, where all
three channels are simultaneously improbable under the model. That is what an
abrupt corrective submovement looks like: the speed jumps, the direction turns and
the timing tightens at the same instant, and the model does not anticipate the
combination even though it has the marginal rate of each right.

**What is proved and what is not.** Proved: the model is wrong, and it is wrong in
a way no per head instrument could see, at two and a half times the threshold that
was registered before the file existed. Not proved: WHICH of three mechanisms is
producing it.

    a  between sequence. some movements are harder for every head at once, and
       the model does not know that it should be less certain on them
    b  between position. some moments inside a movement are hard for every head,
       same failure of self knowledge
    c  within moment. given the same moment, speed being surprising and direction
       being surprising are genuinely linked and the model factorises them apart

All three are real defects and all three break the identity, but they call for
different fixes, and c is the one that says the chain rule factorisation itself is
being under used rather than the confidence being miscalibrated. Separating them
is the next test and it is a slicing test, not a new instrument: the identity is
preserved under conditioning on anything the model can compute from the history,
which includes its own predicted entropy and the position index. Sequence level and
position level and entropy level slices all qualify.

**No build is authorised by this.** Phase conditioning and the spectral loss term
remain NOT AUTHORISED. This is a diagnostic and no serving change follows from it.
The finding is a direction to look, not a licence to train.

## WHICH mechanism produces the cross channel coupling, 2026-08-05

Registered before the file exists. Thresholds fixed here.

w4_cross measured +0.1309 between speed surprise and direction surprise at lag
zero, where a correct model gives exactly zero. Three mechanisms could produce it
and they call for different fixes. This separates them.

    a  between sequence   some movements are harder for every head at once and the
                          model does not know it should be less certain on them
    b  between position   same failure of self knowledge, at the level of moments
                          inside a movement rather than whole movements
    c  within moment      given the same moment, speed being surprising and
                          direction being surprising are genuinely linked, and the
                          chain rule factorisation is being under used

**Why slicing is legal here and what the limit of that is.** The identity that
makes the pooled number exactly zero survives conditioning on ANY quantity the
model could compute from the history, because the argument is a tower property
over the history. The model's own predicted entropy at a step qualifies, since it
is a function of the distribution the model produced from the history alone. The
position index qualifies. The conditioning vector qualifies. What does NOT qualify
is anything built from the outcome, including the realised surprise itself and any
per sequence average of it. Panels built on those are DESCRIPTIVE ATTRIBUTION and
are labelled as such in the output. They say where the observed covariance sits.
They are not tests and no verdict is read on them.

**Panel 1, descriptive. Between and within sequence.** Decompose the covariance
into the part explained by sequences differing in their mean surprise and the part
that remains inside sequences. Reports the between fraction. Descriptive because
the per sequence means use the outcome.

**Panel 2, THE TEST. Within joint entropy cells.** Bin the model's predicted speed
entropy and predicted direction entropy into five quantiles each, giving twenty
five cells, edges fixed at the full sample so the bootstrap does not resample the
binning. Correlation within each cell, then the size weighted average. This asks
whether the coupling is explained by the model already knowing which moments are
uncertain.

    retains >= 0.50 of the pooled reading   WITHIN MOMENT coupling dominates
    retains <= 0.20 of the pooled reading   EXPLAINED by the model's own uncertainty
    otherwise                               MIXED
    BOUNDARY, within one bootstrap sd of a threshold, the call is REFUSED

Retention is the natural statistic because the question is not whether the
within cell number is large in absolute terms but whether conditioning on what the
model already knew makes the coupling go away.

**Panel 3, test. Within position bands.** The same bands w4_progress used, 1 to 4,
4 to 8, 8 to 12, 12 to 20, 20 to 32, 32 up. Profile and flatness. A coupling that
lives in one band is a different object from one that is everywhere.

**Panel 4, descriptive attribution. Concentration.** What share of the total
covariance comes from the top decile of speed surprise. This is the number that
says how rare the responsible moments are. It conditions on the outcome, so it is
attribution and never a verdict.

**VALIDITY ARM.** The self sampled arm through every panel. Exact zero by
construction in each cell. Gates carried over unchanged, including the magnitude
floor of 0.002 and the rule that an unmeasurable bootstrap sd is a FAILURE.

**DIAGNOSTIC ONLY.** Never a contract score. No serving change follows and no
build is authorised by any outcome. Phase conditioning and the spectral loss term
remain NOT AUTHORISED.

### AMENDMENT to the mechanism test's validity gate, made on the failure and before the model side was read

The gate failed at full scale on the speed and direction pair, worst cell -0.0136
against a magnitude floor of 0.01. The self arm numbers alone say why, and they are
the only numbers that were looked at:

    cell sizes for that pair run from 2,486 to 657,180, a range of 264
    the failing cell holds 2,486 positions, so its own noise is 0.020
    the reading is therefore 0.7 sd from zero, which is noise
    the size weighted within cell mean, which is what the VERDICT reads, is
      -0.00029 on speed and direction, +0.00054 on speed and timing, and
      -0.00045 on direction and timing

**The registered rule was half a rule.** It carried a magnitude floor, so a tiny
but significant excursion could not fail the gate, and that half was right. It
carried no noise floor, so a large but insignificant excursion could fail it, and
with cells varying by a factor of 264 in size that half was unusable. Speed entropy
and direction entropy are strongly correlated, so the corner cells of the five by
five grid are nearly empty by construction, and a fixed magnitude compared against
a cell noisier than the threshold itself can only produce false failures.

**The completion.** A cell fails only if it is BOTH large enough to matter AND
distinguishable from zero:

    |rho| >= 0.01  AND  |rho| * sqrt(n) >= 3.0

This is the symmetric counterpart of the floor already registered, not a relaxation
of it. The same rule applies to the position bands.

**And a gate is added, on the statistic the verdict actually reads.** The size
weighted within cell average in the self arm must be within the same 2 sd and 0.002
magnitude treatment as the pooled reading. That gate did not exist before, which
was the real defect: the test was gating on the worst of twenty five cells while
reading its verdict off a weighted average that no gate protected. The new gate
passes at 0.0003, 0.0005 and 0.0005.

**The run is repeated rather than rescored.** The saved file already holds both
arms, and the run is deterministic, so recomputing the verdict from disk would give
identical numbers. It is re run anyway so that the run producing the verdict is one
where the gate passed on its own, and so that no question remains about whether the
gate was shaped by the model side. The model side has not been read.

**Nothing else moves.** The retention thresholds of 0.50 and 0.20, the boundary
rule, the entropy binning, the position bands, the minimum cell size and the
descriptive panels all stand as registered.

### The result. WITHIN MOMENT. The coupling is genuine, it is not the model's self knowledge, and it lives in a tenth of the moments, 2026-08-05

Re run at n=100000 with the amended gate, 90,497 rows, 5,135,552 live positions.
Peak GPU 73C. Checkpoint MD5 unchanged, evaluation data and scoring code untouched.

**Validity.** Clean everywhere. Pooled, weighted within cell average and every cell
and band excursion at zero. The corrected gate passes on its own.

**Panel 2, the test.**

      pair    pooled    within   retains
       sth   +0.1309   +0.0877      0.67
       sdt   +0.0619   +0.0660      1.07
      thdt   +0.0245   +0.0263      1.07

All three are at or above the registered 0.50 line, so the call is WITHIN MOMENT
COUPLING. Conditioning on the model's own predicted uncertainty removes a third of
the speed and direction coupling and none at all of the other two. On speed and
timing and on direction and timing the coupling is very slightly STRONGER inside
entropy cells than pooled, which is the opposite of what mechanism b predicts.

**Panel 3, position bands.** Flat. Speed and direction runs +0.1461, +0.1374,
+0.1404, +0.1330, +0.1229, +0.1023 from the first band to the last. Present
everywhere, mildly stronger early, never absent. This is not a launch effect and
not a tail effect.

**Panel 1, descriptive.** Between sequence variation carries 0.04, 0.03 and 0.03 of
the covariance. Mechanism a is essentially absent. Whole movements being uniformly
harder than the model expects is not what this is.

**Panel 4, descriptive, and the most actionable number here.** The top decile of
speed surprise carries 0.71 of the speed and direction covariance and 0.92 of both
covariances involving timing. The coupling lives in roughly a tenth of the moments.

**What the four panels say together.** At about one moment in ten the real movement
does something the model finds improbable in speed, direction and timing AT ONCE,
the model has no idea those moments are coming, and it treats the three channels as
separable given the history when they are not. That is the signature of an abrupt
corrective submovement: the speed spikes, the direction turns and the timing
tightens in the same instant. The model reproduces the marginal rate of each of
those things, which is why every per head instrument passed, and does not
reproduce their arrival as a package.

**What this does NOT yet establish.** The model's architecture already has a
coupling path: the direction head is conditioned on the emitted speed and the
timing head on both. So the finding is not that the path is missing, it is that the
path is carrying too little. Which is a different claim and needs its own check
before anything is built on it. Note also that the ADDITIVE WITHIN EVENT
CONDITIONING suspect is already CLOSED in this file, and whatever was closed there
has to be read before that ground is walked again.

**No build is authorised by this.** Phase conditioning and the spectral loss term
remain NOT AUTHORISED. This is a diagnostic. Nothing about the serving recipe
changes and no training run follows from it without a separate decision.

## WHERE in the input does the conditioning fail, 2026-08-06

Registered before the file exists. Thresholds fixed here.

**First, the relationship to the CLOSED additive conditioning suspect, because
this must not be a quiet reopening.** `w4_coupletok` measured Spearman rank
correlations among speed, turn magnitude and dt inside trajectories, generated
against real, and found no attenuation: every sign matched and the ratios
scattered on both sides of one. That closure stands and nothing here contradicts
it. Spearman is a monotone rank statistic computed over all events in a
trajectory, so it is dominated by the typical ninety percent and it deliberately
compresses extremes. w4_couple has just located the defect in the top decile of
surprise, which carries between 0.71 and 0.92 of the covariance. Bulk coupling
being right and tail coupling being wrong are not in conflict, and the instrument
that closed the first question is the wrong shape for the second. **The FiLM
rewrite remains NOT AUTHORISED and this test does not bear on it.**

**The test.** The identity again, sliced somewhere new. The direction head's
residual satisfies

    E[ srp_th | history, s(t) ] = 0 exactly, under a correct model

so it also satisfies E[srp_th | s(t)] = 0. The emitted speed is part of what the
direction head conditions on, so slicing by it is legal, and the resulting CURVE

    g(b) = mean srp_th over positions whose emitted speed falls in bin b

is zero at every b under a correct model. Its shape says exactly where in the input
space the conditioning fails, which is what a correlation cannot say. Same
construction for the timing head against emitted speed and against turn magnitude.

Ten quantile bins on the emitted value, edges fixed at the full sample so the
bootstrap does not resample the binning, minimum 500 per bin.

**The registered reading.** How much of the coupling w4_cross measured is
reproduced by the curve alone:

    explained = Cov( srp_s , g(bin of s) ) / Cov( srp_s , srp_th )

    >= 0.50   the defect IS a function of the emitted value, and the curve
              characterises it completely
    <= 0.20   the coupling is finer grained than any function of the emitted
              value and the curve is not the description
    otherwise MIXED
    BOUNDARY, within one bootstrap sd of a threshold, the call is REFUSED

Same 0.50 and 0.20 as w4_couple, on the same explained fraction scale,
deliberately, so the two readings are comparable and neither is fitted.

**Reported alongside, not a verdict.** The curve in nats and as a fraction of the
head's mean entropy at those positions, so a bias can be read as a proportion of
what the head knows rather than as a bare number.

**VALIDITY ARM.** Self sampled tokens. The curve is zero at every bin by
construction. Gates as amended: magnitude floor 0.002 in correlation units and,
for the curve itself, a bin fails only if it is both above the floor and beyond
three sd of its own size implied noise. An unmeasurable bootstrap sd is a FAILURE.

**DIAGNOSTIC ONLY.** Never a contract score. No serving change follows and no
build is authorised by any outcome. Phase conditioning, the spectral loss term and
the FiLM rewrite all remain NOT AUTHORISED.

### The result. FINER GRAINED. The curve is real and it is not the seat of the coupling, 2026-08-06

Run `w4_2026-08-06T013035+0000_6028a160`. n=100000, 90,497 rows, 5,135,552 live
positions, 4,645,723 of them with a direction to predict. The same rows w4_couple
used. GPU peak 73C, no cooldown. `training/candi_polar_flow_best.pt` MD5
`91326a29750789f3167055324ef377c5` verified unchanged after the run, evaluation
data and scoring code untouched.

One pre run correction to the registration, made before any model side number was
seen. The registration said the per bin validity gate would use a magnitude floor
in correlation units and three sd of the bin's size implied noise. The curve is a
mean in nats, not a correlation, so a size implied noise assuming unit spread is
an assumption rather than a measurement. Replaced with the bin's OWN spread,
`se = sd(residuals in bin) / sqrt(n in bin)`. Positions inside a bin are very
nearly independent, w4_indep put the residual step to step dependence at 0.0033,
so this slightly understates the noise, and understating the noise makes a null
arm gate STRICTER. That is the safe direction, so the correction is conservative.

```
VALIDITY   self sampled tokens, the curve is zero at every bin by construction
     s->th  worst bin +0.0017  worst 1.0 sd  excursions 0  smallest bin 359,210
     s->dt  worst bin +0.0019  worst 1.2 sd  excursions 0  smallest bin 304,775
    th->dt  worst bin +0.0041  worst 2.4 sd  excursions 0  smallest bin 252,804
   PASS. thirty bins, worst 2.4 sd, no bin both above the floor and past 3 sd.
```

THE CURVE. Mean surprise of the responder head, in nats, by decile of the driver
value that head is already conditioned on. Zero at every bin under a correct
model. Positive means the head is worse there, negative means better.

```
   slice       d0       d1       d2       d3       d4       d5       d6       d7       d8       d9
   s->th        -        -  -0.0301  -0.0585  -0.0627  -0.1316  -0.1016  +0.0081  +0.0469  +0.0021
   s->dt  +0.0234        -  +0.0232  +0.0130  +0.0108  +0.0175  -0.0133  -0.0158  -0.0006  +0.0334
  th->dt  +0.0388  +0.0276  +0.0192  +0.0040  +0.0110  +0.0131  +0.0126        -        -  -0.0040

 as a fraction of what that head knows, its own entropy at those positions
   s->th        -        -   -0.062   -0.190   -0.128   -0.182   -0.070   +0.004   +0.019   +0.001
   s->dt   +0.017        -   +0.016   +0.011   +0.011   +0.022   -0.015   -0.019   -0.001   +0.034
  th->dt   +0.024   +0.024   +0.021   +0.004   +0.012   +0.014   +0.014        -        -   -0.004
```

Two deciles are blank on `s->th` and one on `s->dt` because the speed alphabet is
discrete with a heavy mode, so quantile edges collide and the mass lands in eight
bins rather than ten. Not a defect, but it means the resolution at the low end is
coarser than ten.

THE REGISTERED READING.

```
   slice   cov full  cov curve    share      sd
   s->th    +0.1218    -0.0023    -0.02    0.00
   s->dt    +0.0764    -0.0016    -0.02    0.00
  th->dt    +0.0265    +0.0021     0.08    0.00
-> FINER GRAINED THAN THE EMITTED VALUE
```

All three are far below the registered 0.20, none is near a boundary, and the
bootstrap sd is below 0.005 on all three. The call is not close.

WHAT THIS SAYS, and it is two things that are both true.

First, the curve is REAL and it is large. The direction head's accuracy swings by
up to nineteen percent of its own entropy depending on which speed it was just
handed. In the middle deciles the residual is strongly NEGATIVE, which means the
head is BETTER there than its own entropy claims, so it is hedging, spreading mass
it does not need to spread. At the high end it turns slightly positive. That is a
genuine conditioning defect and it is not noise, the validity arm's worst bin is
0.0041 against a middle decile reading of 0.1316, a factor of thirty two.

Second, and this is the registered question, that curve explains essentially NONE
of the cross channel coupling. Share is negative, meaning the small part the curve
does track runs the wrong way. Fix every point on that curve and the +0.1309 from
w4_cross would still be there.

So the failure is not a fixed miscalibration table indexed by the emitted value.
It depends on the emitted value AND the history together.

WHAT THIS ELIMINATES. Any remedy whose whole content is a better function of the
emitted token. A richer speed embedding, a wider token alphabet for the driver, a
learned per token bias on the responder head, a lookup correction: all of them can
at best flatten the curve, and flattening the curve moves the coupling by about
two percent in the wrong direction. None of these were authorised and none is now.

WHAT SURVIVES. Combine with w4_couple panel 4, where the top decile of speed
surprise carried 0.71, 0.92 and 0.92 of the coupling. The defect tracks HOW
SURPRISED the model was by the token it emitted, not WHICH token it was. Those are
different objects. Which token it was is in the input by construction. How
surprised the model was is not in the input at all, it is a comparison between the
token and the distribution the trunk implied a moment earlier, and nothing in the
forward pass computes that comparison and feeds it onward.

That is a hypothesis, not a finding, and the next test has to be the one that
separates them.

RELATIONSHIP TO THE CLOSED SUSPECT, restated because this is exactly where a quiet
reopening would happen. `w4_coupletok` closed the additive within event
conditioning on Spearman rank correlations that scattered around one. That closure
STANDS. This result is consistent with an interaction defect but does not
establish one, because the marginal curve being uninformative about the coupling
is not evidence about any particular mechanism inside the head. The FiLM rewrite
of `th_head` and `dt_head` remains NOT AUTHORISED, as do phase conditioning and
the spectral loss term. Nothing here authorises a build.

## Is the failure the model not knowing how surprised it was, 2026-08-06

REGISTERED BEFORE THE FILE EXISTS.

The question. w4_condshape showed the coupling is not a function of WHICH token was
emitted. w4_couple showed it concentrates where the emitted token was IMPROBABLE.
Put together they point at one object: the model's own surprise at what it just
emitted, `srp_s(t) = -log p(s(t)) - H(p_s(t))`. This test asks whether that single
scalar is the whole story.

Why the slice is legal, which matters because a wrong answer here is the same class
of error as the withdrawn KLhat. `srp_s(t)` is a function of the history and of
`s(t)`, and nothing else. The direction head conditions on both. So under a correct
model `E[ srp_th | history, s(t) ] = 0` exactly, and therefore
`E[ srp_th | srp_s(t) ] = 0` exactly. The curve `g(b) = mean srp_th over positions
whose speed surprise falls in bin b` is zero at every b. No null arm is needed for
the verdict, though one is run anyway.

The trap to avoid. This is legal for judging the RESPONDER's residual and illegal
for judging the DRIVER's own. Conditioning `srp_s` on a function of `srp_s` is
conditioning on the outcome and manufactures structure under a perfect model. Only
`s->th`, `s->dt` and `th->dt` are computed, never `s->s`.

The construction is w4_condshape's, unchanged, with the driver replaced. Ten
quantile bins on the driver's surprise, edges fixed once at the full sample so the
bootstrap does not resample the binning, minimum 500 per bin, bootstrap over
sequences with the curve refitted inside every draw.

THE REGISTERED READING, deliberately the same numbers as w4_couple and
w4_condshape so the three are comparable and none is fitted.

```
explained = Cov( srp_s , g(bin of srp_s) ) / Cov( srp_s , srp_th )

  >= 0.50   the coupling IS the model's own surprise. one scalar the forward
            pass does not compute is the whole defect, and the curve says how
            the responder should have moved as a function of it
  <= 0.20   even the model's own surprise does not capture it, and the defect
            needs the full joint of history and token
  otherwise MIXED
  BOUNDARY, within one bootstrap sd of a threshold, the call is REFUSED
```

Reported alongside but never a verdict. The curve in nats and as a fraction of the
responder head's entropy. The share for `s->dt` and `th->dt`. The share in the
self sampled arm, which should be near zero and is a check on the statistic rather
than on the model.

VALIDITY ARM. Self sampled tokens through the identical estimator path. The curve
is zero at every bin by construction. Gates as amended in w4_condshape, a bin
fails only if it is both above the 0.002 magnitude floor and beyond three sd of
its own measured spread, and an unmeasurable bootstrap sd is a FAILURE.

A NEGATIVE RESULT HERE IS THE INFORMATIVE ONE and it is worth saying in advance.
If a single scalar the model never computes turns out to carry the whole cross
channel defect, that is a sharp, buildable statement. If it does not, then four
tests in a row have narrowed the defect without locating it, and the honest
conclusion is that the coupling is not reducible to any low dimensional summary,
which is itself a result and should be written as one rather than chased further.

DIAGNOSTIC ONLY. No serving change follows and no build is authorised by any
outcome. Phase conditioning, the spectral loss term and the FiLM rewrite all
remain NOT AUTHORISED, and `w4_coupletok`'s closure stands regardless of what this
returns.

### The result. EXPLAINS. The defect is the model's own surprise, and it is the covariate w4_condtex went looking for and missed, 2026-08-06

n=100000, 90,497 rows, 5,135,552 live positions, 4,645,723 of them with a direction
to predict. Identical forward pass and identical estimator path to w4_condshape,
one command line flag apart. GPU peak 74C, no cooldown.
`training/candi_polar_flow_best.pt` MD5 `91326a29750789f3167055324ef377c5`
verified unchanged, evaluation data and scoring code untouched.

On the smoke, and recorded because it would otherwise look like a suppressed
failure. At n=3000 the validity arm threw one excursion of thirty bins, `th->dt`
at 3.3 sd. NOTHING WAS CHANGED. A per bin three sd gate over a family of thirty
near independent bins fails on a perfect estimator about eight percent of the
time, so a single 3.3 at smoke size is what chance looks like rather than what
bias looks like, and the full scale run on an independent 90,497 rows is the test
that matters. It came back at 1.7 sd on that same slice with zero excursions,
which is the confirmation. Had it repeated, the family arithmetic would have gone
into an amendment and the run would have been repeated from scratch.

```
VALIDITY   self sampled tokens, the curve is zero at every bin by construction
     s->th  worst bin +0.0025  worst 1.9 sd  excursions 0  smallest bin 464,572
     s->dt  worst bin +0.0029  worst 1.7 sd  excursions 0  smallest bin 513,554
    th->dt  worst bin +0.0029  worst 1.7 sd  excursions 0  smallest bin 464,572
   PASS, and cleanly. worst bin 0.0025 against a model side reading of 0.2260,
   a factor of ninety.
```

THE CURVE. Mean surprise of the responder head, in nats, by decile of the model's
OWN SURPRISE at the driver token. Zero at every bin under a correct model.

```
   slice       d0       d1       d2       d3       d4       d5       d6       d7       d8       d9
   s->th  -0.1982  -0.0953  -0.0752  -0.0657  -0.0628  -0.0648  -0.0606  -0.0350  +0.0285  +0.2260
   s->dt  -0.0047  -0.0145  -0.0080  -0.0109  -0.0093  -0.0105  -0.0045  -0.0006  +0.0103  +0.1647
  th->dt  +0.0018  +0.0055  +0.0007  +0.0024  +0.0005  -0.0091  +0.0025  +0.0121  +0.0134  +0.0692

 as a fraction of what that head knows, its own entropy at those positions
   s->th   -0.145   -0.100   -0.080   -0.068   -0.063   -0.060   -0.050   -0.027   +0.022   +0.152
   s->dt   -0.004   -0.014   -0.008   -0.011   -0.009   -0.010   -0.005   -0.001   +0.010   +0.141
  th->dt   +0.002   +0.006   +0.001   +0.002   +0.000   -0.009   +0.003   +0.017   +0.012   +0.057
```

THE REGISTERED READING.

```
   slice   cov full  cov curve    share      sd
   s->th    +0.1218    +0.0965     0.79    0.00
   s->dt    +0.0764    +0.0429     0.56    0.00
  th->dt    +0.0265    +0.0143     0.54    0.01
-> THE DEFECT IS A FUNCTION OF THE MODEL'S OWN SURPRISE
```

All three clear the registered 0.50, none is within a bootstrap sd of it, and the
primary clears it by more than half again. Against the same threshold on the same
scale, the emitted VALUE scored -0.02. One scalar the forward pass never computes
carries four fifths of the coupling that the entire token it was computed from
carries none of.

WHAT THE SHAPE SAYS, and the sign is the whole point.

`s->th` is monotone across all ten deciles. When the model is UNSURPRISED by the
speed it just emitted the direction head's residual is NEGATIVE, down to -0.1982
nats, which means the real direction was MORE likely than the head's own entropy
claimed. It is hedging, holding open possibilities it did not need to hold open.
When the model is SURPRISED the residual is POSITIVE, up to +0.2260 nats, which
means the head was too sure. The direction head's confidence is TOO FLAT. It does
not sharpen when the moment is easy and it does not widen when the moment is hard.

The top decile is not a smooth continuation of the ramp. d8 reads +0.0285 and d9
reads +0.2260, an eight fold jump, and `s->dt` does the same thing, +0.0103 to
+0.1647, a sixteen fold jump. The defect is concentrated in the last tenth,
exactly where w4_couple's panel 4 put it at 0.71, 0.92 and 0.92.

The two errors are opposite in sign and they very nearly cancel when pooled. That
is not an aside, it is the reconciliation with everything that came before.

WHAT THIS RECONCILES, and none of it is overturned.

`w4_launch` found the one step conditional calibrated to about a thousandth in
tilt from position 1 onward, with shape flat at 1.02 to 1.06. `w4_dtcal` found the
timing conditional calibrated on every axis it tested. Both are marginal
statistics, pooled over histories. A head that is too confident in a tenth of
moments and not confident enough in the other nine tenths is very nearly calibrated
on average, and the residue is exactly a mild uniform over confidence of a few
percent. That is the 1.02 to 1.06. Those closures stand and this explains the
number they left behind.

`w4_condtex` went looking for precisely this failure. Its registration named it:
"too narrow after volatile histories and too wide after calm ones averages to
uniform. That specific failure has a name, regression to the mean, and it is what
under modulation IS." It found nothing, pooled D +0.0231 at the boundary, no
monotone trend across texture quintiles, and it registered the reason in advance:
"a flat D proves the model conditions on THIS function of the history and not on
every function of it," with the branch (a) reading being "the defect is in a place
none of these instruments has yet reached."

It sliced by a hand chosen texture statistic, the mean absolute successive
difference of the speed class over the previous eight events. That was the wrong
covariate. The right one is the model's own surprise, which is not a property of
the history at all but a comparison between the emitted token and the distribution
the model itself produced. Sliced that way the effect that was invisible at
+0.0231 pooled is a monotone ramp spanning 0.42 nats. `w4_condtex`'s caveat has
come true, on its own terms, in the way it said it might.

WHAT IS NOW LOCATED. The failure is that the model does not know how surprised it
was. `srp_s(t) = -log p_s(s(t)) - H(p_s(t))` is a deterministic function of the
trunk output and the emitted token, so the information is present in principle.
Computing it requires evaluating the speed head's own softmax at the realised
token, which is a comparison between a function of the trunk state and a function
of the token. The direction head receives the token as an ADDITIVE embedding into
a LayerNorm, `th_head(th_norm(x + s_ctx_embed(s)))`. An additive combination
followed by an MLP can approximate such a comparison. Whether this one does, and
whether training ever rewarded it, is not measured here.

THAT LAST PARAGRAPH IS AN ARCHITECTURAL ARGUMENT AND NOT A MEASUREMENT, and it is
the exact point at which a closed suspect would get quietly reopened. It is not
reopened. `w4_coupletok` closed the additive within event conditioning on Spearman
rank correlations that scattered around one, and that closure STANDS. The FiLM
rewrite of `th_head` and `dt_head` remains NOT AUTHORISED, as do phase
conditioning and the spectral loss term. Nothing here authorises any of them,
because none of them is the intervention this result points at.

DIAGNOSTIC ONLY, never a contract score. No serving change follows.

## What is the defect actually WORTH, 2026-08-06

REGISTERED BEFORE THE FILE EXISTS. This is a pricing instrument, not another
detector, and it is registered because the temptation after w4_selfsurprise is to
go straight to a fine tune.

THE IDENTITY THAT MAKES THIS FREE. Write the responder head's logits as z and
apply an inverse temperature b, so the predicted law is softmax(b z). Then

    NLL(b) = -b z_k + logsumexp(b z)
    d NLL / d b  at b = 1  =  E_p[z] - z_k

and the surprise residual is

    srp = -log p_k - H = (-z_k + lse) - (-E_p[z] + lse) = E_p[z] - z_k

They are the SAME OBJECT. The curve w4_selfsurprise measured is exactly the
gradient of the head's loss with respect to a confidence correction. So the
achievable loss reduction from correcting the defect inside the temperature family
is not a guess, it is a one dimensional convex minimisation on logits that are
already computed.

To second order the gain in bin b is `g(b)^2 / (2 Var_p[z])`, but the script does
the exact minimisation rather than the quadratic approximation, because g reaches
0.226 nats in the top decile and the approximation is only good for small g.

WHY THIS DECIDES SOMETHING. `w4_arcurve` measured the exchange rate between held
out loss and the contract scorer across eight snapshots of one model on one ruler:
0.1904 AUC per nat, r 0.953, residual sd 0.0131. `event_ar_v2_s40000` sits at
0.6526 contract and 4.4024 nats. The distance to 0.50 is 0.65 to 0.80 nats at that
rate. So any intervention worth less than about 0.05 nats is worth less than 0.01
AUC and is not the thing being chased, however significant it is statistically.
w4_selfsurprise's effect is 164 sd from zero and that says nothing whatever about
its size in nats.

THE REGISTERED READING, and the threshold comes from the exchange rate rather than
from taste.

```
predicted AUC gain = 0.1904 * (nats recovered), reported with the fit's own
residual sd of 0.0131 attached so it is never quoted as exact

  >= 0.05 nats   MATERIAL. worth building. the fine tune is authorised on this
                 evidence and the AUC arm is worth its GPU hours
  <= 0.01 nats   NEGLIGIBLE. the defect is real, correctly measured and worth
                 nothing at the scorer. the finding stands as science and this
                 line CLOSES for AUC purposes without a single training step
  otherwise      MIXED, and the decision goes to L with the number attached
```

WHAT IS PRICED. Three families, cheapest first, each a strict superset of the one
before, so the sequence says where the money stops being available.

```
  1  one global inverse temperature per head              1 parameter
  2  a per decile inverse temperature indexed by srp of   10 parameters
     the driver token, which is exactly the w4_selfsurprise curve
  3  a per decile temperature AND a per decile mix toward 20 parameters
     the head's own marginal, which can widen or sharpen
     in ways a temperature cannot
```

Family 3 is included because a temperature can only rescale, and if the defect is
partly a SHAPE error rather than a WIDTH error then family 2 underprices it. If
family 3 beats family 2 by a wide margin the pricing itself is wrong and must be
redone in a richer family before any conclusion is drawn.

HONEST FITTING. Every parameter is fitted on one half of the SEQUENCES and scored
on the other half, so a gain is a held out gain. Ten or twenty parameters against
4.6 million positions will not overfit measurably, and the split costs nothing, so
there is no reason to leave the hole open.

THE CEILING IS ALSO REPORTED. Family 3's gain is a lower bound on what correcting
the defect is worth, not the whole of it, because the true correction need not lie
in any of these families. Reported alongside as context and never as the verdict:
the total held out NLL of each head, so the gain can be read as a fraction of what
there is to win.

DIAGNOSTIC ONLY. No serving change follows. The locked serving recipe does not
move. Phase conditioning, the spectral loss term and the FiLM rewrite all remain
NOT AUTHORISED, and this instrument cannot authorise them because it prices a
correction rather than a mechanism.

### The result. NEGLIGIBLE. The defect is real, correctly measured, and worth 0.001 AUC, 2026-08-06

n=100000, 90,497 rows, 5,135,552 live positions, logits kept at 250,000 of them,
fit half 124,162 positions and score half 125,838 split by SEQUENCE. GPU peak 73C.
`training/candi_polar_flow_best.pt` MD5 `91326a29750789f3167055324ef377c5`
verified unchanged.

ONE AMENDMENT, made on the smoke and in the CONSERVATIVE direction. A fourth
correction family was added after the smoke returned NEGLIGIBLE and before the
full run: per decile temperature plus a per decile mix toward the class
distribution OF THAT DECILE, rather than toward the head's global marginal. It can
move probability toward whatever directions actually occur at surprising moments,
so it can RE RANK and not merely widen. A richer family can only RAISE the price,
never lower it, so adding one while the verdict stands at NEGLIGIBLE can only
weaken that verdict and never manufacture it. That is why it was added rather than
deferred.

```
  head baselines on the score half, held out
      s    2.0782 nats
     th    1.1307 nats
     dt    1.0719 nats
                                     sum 4.2808, against 4.4024 in training's
                                     own validate(), the known ruler offset

  WHAT THE CORRECTION IS WORTH, in nats, held out
     slice     base     fam1     fam2     fam3     fam4  best-f1
     s->th   1.1307  +0.0008  +0.0061  +0.0063  +0.0063  +0.0055
     s->dt   1.0719  +0.0000  +0.0009  +0.0009  +0.0009  +0.0008
    th->dt   1.0413  +0.0000  +0.0001  +0.0001  +0.0001  +0.0001

  fitted per decile inverse temperature, s->th
     1.236  1.146  1.100  1.142  1.112  1.090  1.068  1.048  0.982  0.850
```

THE MECHANISM IS CONFIRMED EXACTLY. The fitted temperature runs 1.236 at the
calmest decile down to 0.850 at the most surprising, monotone. Above one means
sharpen, below one means widen. That is w4_selfsurprise's curve read back through
an independent estimator: the head should be MORE confident when it was
unsurprised and LESS confident when it was surprised, which is precisely the
"confidence is too flat" reading. Two instruments, one a conditional mean and the
other a maximum likelihood fit, agree on both the sign and the shape.

THE SIZE KILLS IT.

```
  PRIMARY, s->th beyond a global temperature   +0.0055 nats
  all three slices summed, optimistic ceiling  +0.0064 nats
  at 0.1904 AUC per nat (w4_arcurve, resid sd 0.0131)
     predicted AUC gain, ceiling  0.0012      0.6526 -> 0.6514
-> NEGLIGIBLE, against a registered 0.01 nat floor and a 0.15 AUC gap
```

The whole defect is worth half a percent of the direction head's own loss and one
tenth of one percent of the distance to 0.50. It is nine tenths of an order of
magnitude below the 0.05 nat threshold that would have authorised a fine tune. NO
TRAINING STEP WAS SPENT.

FAMILY 4 IS THE CHECK THAT THIS IS NOT AN UNDERPRICE, and it passes. Family 4 has
twenty parameters and can re rank the direction distribution toward whatever
actually occurs at surprising moments. It beats the ten parameter temperature only
family by 0.0002 nats, from 0.0061 to 0.0063. So the defect really is a WIDTH
error and not a SHAPE error, the temperature family was the right one, and the
price is not an artefact of too narrow a correction.

WHAT THIS DOES AND DOES NOT CLOSE.

It CLOSES, for AUC purposes: the model's ignorance of its own surprise as a route
to the contract score. The confidence flatness defect, the cross channel coupling
it explains four fifths of, and every correction that is a function of the driver
token's surprise. The registered fine tune is NOT authorised, because the number
that would have authorised it came in eight times too small.

It does NOT close the science. w4_cross, w4_couple, w4_condshape and
w4_selfsurprise are all correct and all stand. A 164 sd effect that is worth 0.001
AUC is exactly what this programme keeps producing, and the pricing instrument is
the thing that was missing, not the diagnostics.

THE GENERAL LESSON, and it is worth more than this result. Statistical
significance and nats are unrelated quantities here, and every diagnostic in this
file has been reported in significance. `w4_price` converts a measured conditional
mean violation into nats in closed form, because the surprise residual IS the
gradient of the loss with respect to the correction. FROM NOW ON NO DIAGNOSTIC
SHOULD MOTIVATE A BUILD UNTIL IT HAS BEEN PRICED THIS WAY. The instrument is
cheap, one forward pass, and it is general: any conditional mean violation on any
history measurable slice can be run through it.

That reframing also prices the whole family of remaining suspects in advance. The
gap to 0.50 is 0.65 to 0.80 nats. The direction head's ENTIRE loss is 1.13 nats
and the timing head's is 1.07. Any defect confined to one head's conditioning can
at most be worth that head's whole loss, and in practice a small fraction of it.
The arithmetic says the gap is not in a conditioning defect of the kind these four
tests were built to find, and four tests finding real effects worth a thousandth
each is the evidence for that rather than an argument against it.

DIAGNOSTIC ONLY. Phase conditioning, the spectral loss term and the FiLM rewrite
all remain NOT AUTHORISED. `w4_coupletok`'s closure stands. The locked serving
recipe does not move.

### WITHDRAWAL. The NEGLIGIBLE verdict rested on an exchange rate that does not apply to it, 2026-08-06

Withdrawn within the hour, before anything was built on it, and recorded rather
than edited away.

WHAT IS NOT WITHDRAWN. The measurement. The confidence correction recovers 0.0055
nats per direction prediction, held out, and family 4 shows that is not an
underprice within any correction indexed by the driver's surprise. That number
stands, as does the fitted temperature curve, as does everything in w4_cross,
w4_couple, w4_condshape and w4_selfsurprise.

WHAT IS WITHDRAWN. The conversion of 0.0064 nats into 0.0012 AUC, and the
NEGLIGIBLE verdict that rested on it.

THE ERROR. `w4_arcurve`'s 0.1904 AUC per nat was measured across eight snapshots
of ONE model along ONE training run. That is a trajectory on which the loss falls
EVERYWHERE AT ONCE. It measures how inefficiently generic likelihood improvement
converts into detector relevant divergence. It is not a conversion factor for a
TARGETED correction, and using it as one assumes the very thing that is in
question, namely that all nats are worth the same AUC. They are not. A nat removed
from a direction the detector reads is worth a great deal and a nat removed from
one it ignores is worth nothing.

I said the verdict was robust to a tenfold error in the rate. That was the part to
check, and it does not hold. Here is the accounting.

The detector is a two sample test on 18 features. Its score puts a FLOOR under the
divergence between the two distributions. With TV about 2(AUC - 0.5) for a balanced
optimal test, and Pinsker's KL >= 2 TV^2, and the random forest being suboptimal so
that the floor is conservative:

```
  observed contract AUC                              0.6526
  implied total variation                            0.3052
  implied trajectory level KL, at least              0.1863 nats
  token predictions per trajectory, 39 events        113
  implied per token excess KL, at least              0.00165 nats
  the model's held out loss per token                4.2808 nats
  so the detector needs only this fraction to be
  excess in order to explain the WHOLE current gap   0.038 percent
```

And the correction, converted to the same units by the chain rule for KL, which
makes trajectory KL the SUM of the per position conditional KLs:

```
  0.0055 nats per direction prediction
  x 39 events x 0.90 of them carrying a direction
  = 0.1930 nats per trajectory
  = 1.04x the entire divergence the detector's own score requires as a minimum
```

THE CORRECTION IS THE RIGHT ORDER OF MAGNITUDE. It is not a thousandth of what is
needed. On this accounting it is approximately all of it. Whether it is the SAME
divergence the detector reads is an entirely separate question and nothing here
answers it, but the scale argument that produced NEGLIGIBLE is void.

THE LARGER CASUALTY. The same exchange rate carries the conclusion earlier in this
file that closing the gap costs 0.65 to 0.80 further nats and therefore 1.7e3x to
9.6e3x capacity, which is where MORE CAPACITY was priced out of reach. That
conclusion inherits the identical flaw. 0.80 nats is what GENERIC training would
cost at its measured inefficiency. The MINIMUM consistent with the observed AUC is
0.00165 nats per token, which is 0.19 nats per trajectory, four hundred times
smaller. The capacity closure is not reinstated by this and is not overturned by
it either. It is UNPRICED, and it was priced with the wrong instrument.

## Measure it instead of extrapolating it, 2026-08-06

REGISTERED BEFORE THE FILE EXISTS. The exchange rate is the disputed object, so
the experiment must not use it.

THE EXPERIMENT. `w4_price` has already fitted the optimal confidence correction:
ten inverse temperatures indexed by the model's own surprise at the speed it just
emitted. Apply that correction AT GENERATION TIME and score the result with the
contract scorer. This replaces an extrapolation with a measurement, and it costs
one generation run.

TRANSFER, and this is the part that is easy to get wrong. The correction was
fitted on deciles of srp_s under REAL human tokens. At generation the model samples
its own speeds, so its own surprise is systematically smaller and the deciles do
not line up. A rank indexed table would silently apply the wrong correction. So
the table is applied by srp_s VALUE, by linear interpolation on the bin centres in
nats with flat extrapolation at both ends, which transfers correctly between any
two distributions of srp_s. `w4_price` is re run once solely to record the bin
edges alongside the temperatures it already fitted.

THE ARMS, paired and in one session so the generation path cannot drift between
them.

```
  baseline    the locked recipe, unchanged, n 2000, seed 0
  corrected   identical, plus beta(srp_s) on the direction head at every step
  placebo     identical, plus the SAME ten temperatures applied in REVERSE
              order, so the correction has the same magnitude and the same
              marginal distribution of temperatures and the wrong sign
```

The placebo is the arm that makes this readable. A per step temperature that
varies at all changes the generated distribution, and some of that change could
move AUC for reasons unrelated to the defect. The reversed table is the same
intervention with the sign of the mechanism flipped. If corrected and placebo move
together, the movement is not the mechanism.

THE REGISTERED READING, on the contract scorer, against the paired baseline from
the same session.

```
  corrected improves by >= 0.02 AND the placebo does not      MECHANISM CONFIRMED.
      the fine tune is authorised and the exchange rate is dead as an instrument
  corrected improves by <= 0.005                              the correction does
      not touch what the detector reads. NEGLIGIBLE is reinstated, now on a
      measurement rather than an extrapolation
  corrected and placebo move together                         the movement is the
      perturbation and not the mechanism. no build follows
  otherwise                                                   MIXED
```

0.02 is chosen because it is below the 0.03 run to run noise band quoted for this
scorer but the comparison is PAIRED on the same conditioning set and the same
seed, which removes most of that noise. Three seeds per arm, and the call is made
on the mean with the paired spread reported.

MANDATE CHECK, because a serving side intervention is exactly where the rule gets
broken by accident. One trajectory per request. No candidate generation, no
selection, no rejection, no scoring of anything before it is emitted. The
correction is a deterministic function of the model's own state at that step. This
is legal under the standing mandate and it does not move the locked serving
recipe, which stays exactly as it is; these are ARMS.

DIAGNOSTIC ONLY as to mechanism. A positive result authorises the fine tune and
nothing else. Phase conditioning, the spectral loss term and the FiLM rewrite all
remain NOT AUTHORISED, and `w4_coupletok`'s closure stands.

### Interruption. The machine crashed mid run and the arms were relaunched, 2026-08-06

The first attempt at the three arm run did not finish. The machine went down
about one arm and a half in. It had been sitting at 75C, which is the launch
gate and well under the 83C watchdog kill, so the crash is not attributable to
temperature on the evidence available. This machine has bluescreened under
sustained load before and that is the only pattern to point at.

WHAT WAS LOST. The nine arm result files were being written under
`/tmp/claude-*/scratchpad`, and a reboot wipes `/tmp`. Everything in the repo
survived: `research/w4_price.json` with the fitted table, the `models/event_ar.py`
`th_beta` hook, the `research/w4_ar_eval.py` arm flags, this file, and the ledger.
`training/candi_polar_flow_best.pt` verified unchanged at MD5
`91326a29750789f3167055324ef377c5`.

WHAT WAS SEEN BEFORE THE CRASH, and it is NOT a result. Baseline seed 0 scored
0.6522 and the corrected arm seed 0 scored 0.6466, an improvement of 0.0056 with
the sign convention that lower is closer to the 0.50 goal. One seed, and the
placebo for that seed never ran. Under the registered reading 0.0056 sits just
above the 0.005 NEGLIGIBLE line and far below the 0.02 confirmation line, which
is MIXED territory, and a single unpaired seed cannot be read at all. This is
recorded so the number is not rediscovered later and mistaken for a finding. The
relaunched run regenerates it from scratch.

WHAT CHANGED FOR THE RELAUNCH. Nothing about the experiment, the arms, the
thresholds or the registered reading. Three operational changes only.

```
  outputs now go to /home/aaronadmin/w4_arms, which survives a reboot
  each arm is skipped if its output file already exists, so the runner can be
      re run after any interruption and continues where it stopped
  90s idle between arms, and the watchdog kill is tightened from 83C to 79C
      for this run only, because a crash already happened on this workload
```

The tightened watchdog is conservative in the safe direction. It can only stop
the run early, never let it run hotter.

NOISE CALIBRATION, unplanned and useful. The relaunched baseline at seed 0 scored
0.6581. The pre crash baseline at the same seed, same checkpoint, same config,
scored 0.6522. That is a 0.0059 spread between two runs that differ in nothing
except the wall clock, and it is a direct measurement of something the
registration assumed away. The registration argued the arms are PAIRED on the
same conditioning set and the same seed and that this removes most of the run to
run noise. Pairing does remove the conditioning set variation. It does not remove
the sampling variation, because CUDA generation is not bit deterministic and
`--seed` does not pin the model's sampling stream to a repeatable sequence. This
is already documented in `research/w1_oneshot_score.py`, which recorded 0.649649
and 0.661526 for one identical config on 2026-07-27.

The consequence, stated before the arms are read so it cannot be fitted to them.
A single paired difference carries about 0.006 of irreducible noise. The 0.02
confirmation threshold is a little over three times that, so it survives. The
0.005 NEGLIGIBLE line does NOT survive on one seed, since it sits inside the
noise. It is only readable on the mean of three seeds, where the standard error
on the mean falls to roughly 0.0034, and even then it is a weak call. Read the
mean and report the spread. Do not read any single seed.

### The result. THE PERTURBATION, NOT THE MECHANISM. And a real effect nobody registered, 2026-08-06

Nine arms, base / fitted / reverse crossed with seeds 0, 1, 2, one session, one
checkpoint `event_ar_v2_s40000.pt`, n 2000 per arm, contract scorer.

```
              s0      s1      s2      mean
  base      0.6581  0.6568  0.6687  0.6612
  fitted    0.6467  0.6304  0.6325  0.6366
  reverse   0.6433  0.6495  0.6331  0.6420

  paired improvement, base minus arm, positive is closer to 0.50
  fitted   +0.0114 +0.0263 +0.0362  +0.0247   sd 0.0125  sem 0.0072
  reverse  +0.0149 +0.0073 +0.0356  +0.0193   sd 0.0146  sem 0.0085

  fitted minus reverse, the mechanism with the shared component removed
           -0.0035 +0.0190 +0.0006  +0.0054   sem 0.0069
```

THE REGISTERED READING, APPLIED VERBATIM. The confirmation branch required the
corrected arm to improve by at least 0.02 AND the placebo not to. The corrected
arm cleared 0.02 at +0.0247. The placebo did not fail to move; it improved by
+0.0193, which is the same improvement inside the noise. That is the third
branch. THE MOVEMENT IS THE PERTURBATION AND NOT THE MECHANISM. NO BUILD FOLLOWS.
The confidence fine tune is NOT authorised.

WHAT THE MECHANISM IS WORTH, stated as a bound rather than a point. The fitted
arm beat the placebo by +0.0054 with a standard error of 0.0069 on three seeds,
so the mechanism specific effect is not distinguishable from zero. Powering this
to resolve an effect of that size needs a per seed sd of 0.0119 driven down to a
sem near 0.0017, which is about forty nine seeds per arm. That is roughly
seventeen hours of supervised generation to settle a quantity whose best point
estimate is one fifth of what the corrected arm as a whole delivered. It is not
worth buying. The srp indexing is closed.

THE FINDING NOBODY REGISTERED, and it is the reason this run was not a waste.
BOTH arms improved, on all three seeds each, six positive readings out of six.
The contract AUC moved from 0.6612 to about 0.64. That is larger than the 0.006
run to run noise measured above and it reproduced in both directions of the
table, so it is real and it is NOT the surprise indexing.

What the two arms share is the only thing left to attribute it to. The fitted
table and the reversed table contain the SAME ten inverse temperatures, so their
mean is identical at 1.0774. Both arms therefore sharpen the direction head by
about eight percent on average, and they differ only in which surprise decile
gets which value. The shared component is a single global scalar. The obvious
reading is that the direction head is uniformly a little too soft at serving and
the surprise structure is decoration on top of a constant.

## Is the whole effect one scalar, 2026-08-06

REGISTERED BEFORE THE RUN. This is the cheapest possible test of the reading
above and it costs one control arm.

THE EXPERIMENT. Drop the table entirely and apply a CONSTANT inverse temperature
of 1.0774 to the direction head, which is exactly the mean of the ten values both
arms shared. `research/w4_ar_eval.py` already carries `--th-temp`, and an inverse
temperature of 1.0774 is a temperature of 0.9282, so no new code is needed. Three
seeds, 0, 1 and 2, paired against the same three baselines already measured.

THE REGISTERED READING.

```
  constant improves by >= 0.015          the entire effect is one scalar. the
      table is decoration. this is a serving side temperature finding, not an
      architecture finding, and it must be checked against the closed
      "sampling temperature tuning" line before anything is claimed
  constant improves by <= 0.005          the effect needs the per step variation
      even though it does not need the SIGN of it, which would be a genuinely
      strange result and would reopen the question
  otherwise                              partial, report the split
```

MANDATE CHECK. One trajectory per request, no candidates, no selection. A
constant temperature on one head is a decode parameter and nothing more.

CAUTION REGISTERED IN ADVANCE. Sampling temperature tuning is on the closed list.
If this control lands at or above 0.015 the honest description is that a
previously closed knob was closed on the FLOW model and on the speed head, and
that the AR model's direction head was never swept separately. That is a
bookkeeping gap, not a breakthrough, and it must be written up that way. It also
does not touch the mandate's target: 0.64 is not 0.50.

### The result. IT IS ONE SCALAR. The table was decoration, 2026-08-06

Three seeds, constant direction head temperature 0.928049, which is the
reciprocal of the 1.077529 mean inverse temperature both tables shared. Paired
against the same three baselines.

```
              s0      s1      s2      mean
  base      0.6581  0.6568  0.6687  0.6612
  fitted    0.6467  0.6304  0.6325  0.6366
  reverse   0.6433  0.6495  0.6331  0.6420
  const     0.6274  0.6531  0.6455  0.6420

  improvement over the paired base
  fitted   +0.0114 +0.0263 +0.0362  +0.0247   sem 0.0072
  reverse  +0.0149 +0.0073 +0.0356  +0.0193   sem 0.0085
  const    +0.0307 +0.0037 +0.0233  +0.0192   sem 0.0081
```

THE REGISTERED READING, APPLIED VERBATIM. The threshold for "the entire effect is
one scalar" was 0.015. The constant arm delivered +0.0192. THE ENTIRE EFFECT IS
ONE SCALAR. The ten value table, the surprise indexing, the whole apparatus of
`w4_price`, contributed nothing that a single number does not. The three
interventions land at +0.0247, +0.0193 and +0.0192 with standard errors near
0.008, which makes them indistinguishable from each other, and the simplest one
is the one to keep.

WHAT THIS ACTUALLY SAYS ABOUT THE MODEL, stated plainly because it is the only
part with any content. The AR model's direction head is uniformly too soft at
serving. Sharpening it by eight percent moves the contract AUC from 0.6612 to
0.6420. That is a calibration statement about one head and it is interpretable:
the model spreads its probability over directions more widely than the data does.

WHAT THIS IS NOT. It is not an architecture finding. It is not a mechanism. It is
a decode parameter, and `--th-temp` has been sitting in `research/w4_ar_eval.py`
unswept the entire time. The bookkeeping is that "sampling temperature tuning" was
closed on the FLOW model and on the SPEED head, and the AR model's direction head
was never swept separately. That is a gap in our own coverage, not a discovery.

AND IT DOES NOT MOVE THE TARGET. 0.6420 is not 0.50. The split half floor is 0.467
to 0.512. This buys about two points of the roughly sixteen that remain, from a
model that its own cosine schedule cut off before convergence.

THE OVERFITTING OBJECTION, which is real and is registered here before anything
further is tuned. A temperature chosen to lower the contract AUC is, by
construction, fitted to the detector. Every previous closure in this file
concerned a structural claim about the generative process, and a structural claim
that survives is evidence about the process. A scalar tuned against the scorer is
not. If a sweep is run it must be understood as measuring how much of the current
gap is attributable to one head's calibration, and NOT as progress toward a model
that is human like. The two are different claims and only the first is supported.

## The direction head temperature sweep, 2026-08-06

REGISTERED, NOT RUN. Deliberately not launched. It is fifteen arms, roughly two
hours of sustained generation, and the standing rule is supervised sessions only.
This machine crashed on this workload earlier today. It waits for a session with
someone watching it.

THE EXPERIMENT. `--th-temp` at 0.85, 0.90, 0.928, 0.96 and 1.00, three seeds each,
paired against the three baselines already measured and stored in
`/home/aaronadmin/w4_arms`. 1.00 is the served value and doubles as a second
baseline, which is a free internal consistency check: it must reproduce the
existing base arms to within the 0.006 noise floor or the run is not trusted.

THE REGISTERED READING.

```
  the minimum sits at an interior point and the curve is smooth   the direction
      head has a calibration optimum and it is worth about what the curve says.
      report it as a calibration measurement, never as progress
  the minimum sits at the 0.85 edge                               the sweep did
      not bracket it. do NOT extend the sweep chasing it. an unbounded
      sharpening that keeps helping is the detector being fitted, not a
      calibration optimum, and the honest report is that the knob is unbounded
      in the direction that flatters the scorer
  no value beats 1.00 by more than 0.006                          the +0.0192
      above does not replicate and this whole line closes
```

WHAT MUST NOT FOLLOW FROM IT. No training change. No architecture change. The
served recipe stays where it is until there is a reason beyond a scorer number.
Phase conditioning, the spectral loss term and the FiLM rewrite all remain NOT
AUTHORISED.

## The exchange rate for a TARGETED correction, 2026-08-06

REGISTERED BEFORE THE FILE EXISTS. This is the instrument the withdrawal above
said was missing, and it is registered before anything is computed because the
number it produces decides whether the capacity closure gets reopened.

WHY IT IS NEEDED. `w4_arcurve`'s 0.1904 AUC per nat is dead as an instrument and
was withdrawn. It measured eight snapshots along ONE training trajectory, where
the loss falls everywhere at once, so it prices GENERIC likelihood improvement at
its measured inefficiency. It is not a conversion factor for a targeted
correction. Two conclusions rested on it. The first, the NEGLIGIBLE verdict on
`w4_price`, has since been settled by measurement rather than extrapolation: the
three arm run showed the movement was the perturbation and not the mechanism. The
second has not been settled at all. The claim that closing the gap costs 0.65 to
0.80 further nats, and therefore 1.7e3x to 9.6e3x capacity, is where MORE
CAPACITY was priced out of reach, and it is UNPRICED. That is the largest loose
end in this file and this instrument is aimed at it.

WHAT WE ALREADY HAVE AND HAVE NOT READ BACK. The constant temperature arm is a
TARGETED intervention with a measured AUC effect of +0.0192 and a likelihood cost
that is computable in closed form from logits that have already been computed
once. That is one point on the targeted exchange rate curve, sitting on disk,
never read. Nothing needs to be generated to read it.

THE INSTRUMENT. `research/w4_beta_curve.py`. One teacher forced forward pass over
the same held out corpus, the same checkpoint `event_ar_v2_s40000.pt` and the
same sequence level split as `w4_price`, so every number composes with the ones
already recorded. It evaluates the direction head's mean NLL at a dense grid of
FIXED inverse temperatures rather than fitting one, and reports the whole curve.
No fitting is required for the primary quantity. Paired per position differences
are used for the error bars, because the same positions are scored at every grid
point and the unpaired spread would be an order of magnitude too pessimistic.

WHAT IT PRODUCES ON ITS OWN, without a single generated trajectory.

```
  beta*         the likelihood optimal inverse temperature for the direction
                head, held out
  the crossover the beta above which the sharpening makes the model WORSE by its
                own held out likelihood than leaving it alone
  the curve     NLL(beta) across the five settings the sweep would use, so the
                sweep becomes a set of paired (nats, AUC) points instead of five
                AUC numbers with nothing to hang them on
  the rate      AUC per trajectory level nat for the ONE targeted point already
                measured, which is the quantity `w4_arcurve` could never supply
```

THE REGISTERED READING, and the grid points are named now so none of this can be
chosen afterwards. The AUC winning constant was a direction head temperature of
0.928049, which is an inverse temperature of 1.077529.

```
  beta* >= 1.0775   the sharpening that wins on the contract scorer sits at or
      before the likelihood optimum. it removes divergence the model itself
      agrees is there, the overfitting objection is materially weakened, and the
      sweep is worth its hours because the resulting curve is then a statement
      about the model rather than about the detector

  beta* < 1.0775    the AUC winning sharpening OVERSHOOTS the likelihood
      optimum. this is AMBIGUOUS and this instrument cannot resolve it. two
      causes produce it and they have opposite implications: the scorer being
      fitted, or exposure bias, since beta is fitted under real human history and
      applied under the model's own. record the size of the overshoot, report
      both readings, and do NOT pick one

  crossover <= 1.0775   the constant arm made the model objectively worse by its
      own held out likelihood while improving the contract AUC. that is the
      strongest evidence this programme could produce that the scorer and the
      likelihood disagree about what human means, and it would reduce the entire
      temperature line to a scorer artefact
```

THE CONFOUND, named in advance so it is not discovered as a convenience later.
Every NLL here is TEACHER FORCED, measured with real human tokens in the history.
Every AUC is FREE RUNNING, measured with the model's own tokens in the history.
The two need not share an optimum and the difference between them is exposure
bias, which is a real and separately known defect of this architecture. This
instrument cannot separate exposure bias from detector fitting and does not claim
to. It bounds the size of whatever the disagreement is.

THE EXTRAPOLATION, flagged before it is made. Converting one measured (nats, AUC)
point into a prediction about the whole 0.16 gap is exactly the move that was
just withdrawn, and it is not made as a verdict. It is reported as the hypothesis
the sweep exists to test at five magnitudes, with the linearity assumption stated
every time it is quoted.

DIAGNOSTIC ONLY. Reads `training/events_*.npy` and one checkpoint. It touches no
evaluation data, no scoring code and never `training/candi_polar_flow_best.pt`.
No serving change follows. Phase conditioning, the spectral loss term and the
FiLM rewrite all remain NOT AUTHORISED and nothing here can authorise them.

### The amendment to the direction head temperature sweep, made before it runs

The sweep registered above is amended, and the amendment is made now, before any
sweep data exists, from numbers already published in this file. Adjusting a
reading after seeing results is forbidden and this is not that.

WHAT CHANGES. Two things and nothing else. The arms, the seeds, the thresholds on
AUC and the pairing all stay exactly as registered.

```
  1  the held out direction head NLL is recorded alongside the AUC at each of
     the five settings, which costs one forward pass and no generation
  2  the registered reading gains the branch below, which is sharper than
     "the minimum sits at the 0.85 edge" and is available only because of it
```

WHY. The five settings, converted to inverse temperature, are 1.0000, 1.0417,
1.0776, 1.1111 and 1.1765. That grid was chosen for the AUC question and it
happens to straddle the point where sharpening stops improving the model's own
likelihood and starts degrading it. So the sweep already contains a likelihood
control it was never read as having. Without the NLL column it produces five AUC
numbers separated by less than their own noise. With it, it produces five paired
points on the targeted exchange rate curve, which is the object the capacity
closure needs and which no experiment in this file has ever measured.

THE ADDED BRANCH.

```
  AUC keeps falling as beta rises past the crossover, into settings where the
  model is objectively worse by its own held out likelihood      the scorer is
      being fitted. this is not a calibration optimum and the temperature line
      closes as an artefact, whatever the AUC numbers say

  AUC turns around at or before the crossover                    the optimum is
      a genuine calibration optimum and the (nats, AUC) pairs are a real
      exchange rate curve for a targeted correction, with curvature
```

WHAT STILL MUST NOT FOLLOW. No training change. No architecture change. The
served recipe does not move. The overfitting objection registered above stands
whatever this returns, and the honest description of a favourable result remains
that it measures how much of the gap one head's calibration accounts for, never
that it is progress toward a human like model.

### The result. The generic rate understated the targeted one by at least 55x, and the sweep is now a sharp test, 2026-08-06

n 100000, 90,497 rows, 5,135,552 live positions of which 4,645,723 carry a
direction, fraction 0.9046. Logits kept at 250,000 positions, split by SEQUENCE
into a fit half and a score half, identical corpus and identical split to
`w4_price`. GPU peak 75C. `training/candi_polar_flow_best.pt` MD5
`91326a29750789f3167055324ef377c5` verified unchanged.

THE INSTRUMENT AGREES WITH THE ONE THAT CAME BEFORE IT, which is the first thing
to check and it passes. `w4_price` fitted a single global inverse temperature by
golden section and got 1.0503 worth 0.000796 nats. This file evaluated a fixed
grid and found the held out optimum at 1.0500 worth 0.000797 nats. Two different
estimators, same corpus, agreeing to three decimal places. The curve is real.

```
  beta*        1.0500   th_temp 0.9524   gain +0.000797 nats  sem 0.000174
  crossover    1.0904   th_temp 0.9171   above this the model is WORSE by its
                                         own held out likelihood than untouched
  the AUC winning arm, beta 1.077529, th_temp 0.928049
               gain +0.000382 nats per direction prediction, sem 0.000270
```

THE REGISTERED READING, APPLIED VERBATIM. beta* is 1.0500 and the AUC winning
constant is 1.0775, so beta* < 1.0775 and THE SECOND BRANCH FIRES. The sharpening
that wins on the contract scorer OVERSHOOTS the likelihood optimum, by 0.0275 in
beta. That branch was registered as AMBIGUOUS between two causes with opposite
implications, the scorer being fitted and exposure bias, and the registration
says record the overshoot and do NOT pick one. It is recorded and neither is
picked. The third branch does NOT fire: the crossover is 1.0904, which is above
1.0775, so the constant arm did not make the model objectively worse. It sits
past the peak but still on the profitable side of it.

WHAT THE SWEEP GRID TURNS OUT TO BE, and this is why the amendment was worth
making. The five settings were chosen for the AUC question months before any of
this existed. They straddle the crossover almost perfectly.

```
   th_temp     beta       gain      sem   the model's own held out verdict
    1.0000   1.0000  +0.000000        0   the served value, by definition zero
    0.9600   1.0417  +0.000808  0.000145   BETTER, and within noise of the optimum
    0.9280   1.0775  +0.000382  0.000270   BETTER, past the peak, still positive
    0.9000   1.1111  -0.000857  0.000387   WORSE
    0.8500   1.1765  -0.005324  0.000615   WORSE, at 8.7 sigma
```

Two settings on the profitable side, one at the peak, two on the far side where
the model is objectively degraded. That is a discriminating design and it was an
accident. The added branch is now a clean binary: if contract AUC keeps falling
at 0.90 and 0.85, the scorer prefers a model that is worse by its own likelihood
and the temperature line closes as an artefact. If AUC turns around at or before
th_temp 0.9171, it is a genuine calibration optimum. Before the amendment the
sweep was five AUC numbers separated by less than their own noise. It now has a
reading that does not depend on resolving them against each other.

THE TARGETED EXCHANGE RATE, the quantity `w4_arcurve` could never supply.

```
  the const arm removed  0.000382 nats per direction prediction
                       x 39 events x 0.9046 carrying a direction
                       = 0.0135 nats per trajectory
  and it bought          0.0192 AUC, three paired seeds, sem 0.0081

  TARGETED   1.42 AUC per trajectory nat, point estimate
  GENERIC    0.001685 AUC per trajectory nat, the withdrawn 0.1904 per token
             divided by 113 token predictions per trajectory
  ratio      845x
```

THE ERROR BAR IS LARGE AND IS CARRIED THROUGH RATHER THAN HIDDEN. The likelihood
gain at the AUC arm is +0.000382 with sem 0.000270, which is 1.4 sigma from zero.
The denominator of that rate is barely distinguishable from nothing. Pushing both
errors two sigma in the direction that flatters the generic rate gives 0.09 AUC
per trajectory nat, still 55x. Pushing them the other way sends the rate to
infinity, because the denominator crosses zero. So the honest statement of the
result is a one sided bound and not a point estimate:

    the targeted route is AT LEAST 55x more AUC efficient per nat than generic
    likelihood improvement, with a point estimate near 845x and no upper bound

The unbounded end is not a nuisance, it is a live possibility with content. If
the true likelihood gain is zero, the constant arm moved the contract AUC by
0.0192 while changing the model's held out loss not at all, which would mean the
scorer reads a direction the likelihood does not price. That is the same
disagreement the second branch flagged, seen from the other side.

WHAT THIS DOES TO THE CAPACITY CLOSURE, which is what the instrument was built
for. The closure said the gap costs 0.65 to 0.80 further nats and therefore 1.7e3
to 9.6e3 times the capacity. Those are nats per TOKEN, so per trajectory the
closure was assuming 73 to 90 nats of divergence had to be removed. Pinsker's
inequality applied to the observed AUC says the divergence that actually has to
be removed is at least 0.2079 nats per trajectory, and the measured targeted rate
says the amount is of that order rather than four hundred times it.

The closure is not reinstated and it is not merely unpriced. IT IS OVERTURNED AS
PRICED. It converted a targeted requirement into a generic cost at an exchange
rate that is wrong by at least one and probably two to three orders of magnitude,
and the number it produced, three to four orders of magnitude more capacity, was
manufactured by that conversion.

WHAT DOES NOT FOLLOW, and this is the part that would be easy to get wrong. It
does NOT follow that more capacity would work. The 55x to 845x figure is
precisely a statement that GENERIC improvement is a bad instrument for this gap,
and scaling is generic improvement. The correct conclusion is that the gap is
small and specific, roughly a fifth of a nat per trajectory sitting somewhere the
detector can see, and that brute scaling is the wrong tool for it. The old
closure reached a defensible destination, that scaling is not the answer, by
arithmetic that was wrong by a factor of four hundred, and the reason that
matters is that the same arithmetic made the gap look unreachable by ANY route.
It is not unreachable. It is small.

IT ALSO PRICES THE TRUNCATED RUN, an open thread since `event_ar_v2` was cut off
by its cosine schedule. Finishing that run is generic improvement and is
therefore charged at the generic rate. Even a generous 0.05 nats per token from
finishing it, 5.65 nats per trajectory, buys about 0.0095 AUC at 0.001685 per
trajectory nat. Half a point of the sixteen, cheaply, and it is not the answer.
That thread can stop being described as unpriced.

DIAGNOSTIC ONLY. Every NLL here is teacher forced and every AUC it is compared
against is free running, so the overshoot at the second branch remains ambiguous
between detector fitting and exposure bias and nothing here attributes it. No
serving change follows. The locked serving recipe does not move. Phase
conditioning, the spectral loss term and the FiLM rewrite all remain NOT
AUTHORISED.

### Interruption. The watchdog fired at 79C and a hole in the watchdog itself, 2026-08-06

The sweep launched at 19:55 and the watchdog killed it at 20:18, twenty two
minutes and two and a half arms in, on reaching the tightened 79C threshold. It
worked. Nothing was lost that a re run does not regenerate, the third arm was
killed before it wrote its output file, and the skip if exists property means the
restart continues from where it stopped.
`training/candi_polar_flow_best.pt` MD5 `91326a29750789f3167055324ef377c5`
verified unchanged.

THE HOLE, recorded because it was survived by luck and not by design.
`wd2.sh` was written to kill `w4_ar_eval.py` and `w4_beta_curve.py` and then exit.
It did not kill the RUNNER. So the runner would have woken from its idle gap and
launched the next arm with no watchdog alive anywhere on the machine, on hardware
that bluescreened on this exact workload the day before. It did not happen only
because the kill landed inside the 90 second gap and the runner was stopped by
hand during it. The original `wd.sh` killed `arms.sh` and that line was dropped
when the script was extended. Fixed: the runner is now killed FIRST, before the
python, and `sweep.sh`, `arms.sh` and `const.sh` are all covered.

THE THERMAL FACT, which is about this workload and not about this run. The arms
run at 77 to 79C sustained and a 90 second idle gap does not shed it. The three
earlier nine arm runs peaked at 78C, so they were sitting one degree under the
threshold the whole time rather than comfortably below it. The gap is widened to
240s for the restart. Supervised sessions only, unchanged.

WHAT THE TWO COMPLETED ARMS SAY, and it is preliminary at two seeds but it is
large and it is the arm that matters most.

```
  th_temp 0.85   s0 0.6258   s1 0.6197
  paired base    s0 0.6581   s1 0.6568
  improvement       +0.0324     +0.0371     mean +0.0347 on two seeds

  for comparison, th_temp 0.928 on three seeds        mean +0.0192
```

th_temp 0.85 is beta 1.1765, which is past the crossover at 1.0904, where the
model is WORSE by its own held out likelihood by 0.0053 nats at 8.7 sigma. The
contract scorer prefers it, and prefers it by nearly twice what it preferred the
near optimal 0.928. The improvement is climbing monotonically as the model gets
objectively worse.

THAT IS THE ARTEFACT BRANCH, on both registrations at once. The amendment's added
branch reads "AUC keeps falling as beta rises past the crossover, into settings
where the model is objectively worse by its own held out likelihood: the scorer
is being fitted. this is not a calibration optimum and the temperature line
closes as an artefact." The original registration reads "an unbounded sharpening
that keeps helping is the detector being fitted, not a calibration optimum."
They agree. Called preliminary at two seeds, not settled.

Without the likelihood column this arm would have read as the best number of the
session. That is what the amendment bought.

### The reduced restart, and the scope reduction is post hoc and labelled

Seven arms, not ten. `t0.96` is DROPPED. It sits on the healthy side of the
crossover at beta 1.0417 with a gain of +0.000808 nats, so it speaks only to the
interior minimum branch, and the t0.85 result has made that branch moot. The
registered question is binary and the seven remaining arms answer it.

THIS DECISION WAS MADE AFTER SEEING THE FIRST TWO ARMS. It is a reduction in
scope and not a change to any threshold or to any reading, both of which stand
exactly as registered, but it was taken with data in hand and it is labelled that
way rather than presented as the original plan. L was told it was post hoc before
approving it.

```
  t0.85 s2    completes the critical mean to three seeds
  t1.00 x3    the internal consistency check. it must reproduce arm_base_s* to
              within the 0.006 noise floor or the whole comparison is untrusted
  t0.90 x3    confirms the trend is monotone inside the degraded region
```

Ordered most decisive first so that another thermal kill costs the least.

### The result. THE TEMPERATURE LINE IS A SCORER ARTEFACT, and this morning's +0.0192 was it, 2026-08-06

The critical arm completed at three seeds before the second thermal kill. Paired
against the same three baselines, with the held out likelihood column that the
amendment added.

```
                s0      s1      s2     mean   improvement    sem   held out nats
  base        0.6581  0.6568  0.6687  0.6612                       0 by definition
  th 0.928    0.6274  0.6531  0.6455  0.6420    +0.0192    0.0081   +0.000382 BETTER
  th 0.85     0.6258  0.6197  0.6208  0.6221    +0.0391    0.0046   -0.005324 WORSE
```

THE REGISTERED READING, APPLIED VERBATIM. The amendment's added branch reads
"AUC keeps falling as beta rises past the crossover, into settings where the
model is objectively worse by its own held out likelihood: the scorer is being
fitted. this is not a calibration optimum and the temperature line closes as an
artefact." th 0.85 is beta 1.1765, past the crossover at 1.0904, and it is worse
by its own held out likelihood at 8.7 sigma. The contract scorer prefers it, and
prefers it by more than twice what it preferred the near optimal 0.928. THE
BRANCH FIRES. The original registration's edge branch fires with it: the minimum
sits at the 0.85 edge, unbracketed, and that registration says an unbounded
sharpening that keeps helping is the detector being fitted and must NOT be
chased by extending the sweep. It is not extended.

WHAT THIS RETIRES. The +0.0192 that opened this session is the same artefact at a
milder setting, where it happened to land on the profitable side of the
likelihood curve and so looked like a calibration finding. It is not two points
of the sixteen. It is not progress toward a human like model. The "WHAT REMAINS
TRUE AND IS WORTH SOMETHING" paragraph written this morning, that the direction
head is uniformly too soft at serving, is WITHDRAWN as an interpretation. What is
true is narrower: sharpening the direction head lowers this detector's score, and
it keeps lowering it after the model has stopped improving.

THE ONE THING THIS DOES NOT EXCLUDE, and it is the same confound the beta curve
registered and refused to resolve. Every likelihood number is TEACHER FORCED and
every AUC is FREE RUNNING. Exposure bias could legitimately move the free running
optimum sharper than the teacher forced one, which would make th 0.85 genuinely
better under free running while worse under teacher forcing. The reason the
artefact reading is preferred is that exposure bias predicts a SHIFTED optimum,
not an absent one, and nothing here turns around. But the turnaround was never
bracketed, so this is an argument and not a measurement. Recorded as such.

MONOTONICITY IS UNTESTED. The branch fires on the endpoint comparison, 0.928
against 0.85. The intermediate point th 0.90, beta 1.1111, the only other setting
past the crossover, was not run. If it were to land ABOVE 0.85's improvement
there would be an interior minimum near 0.90 and the artefact reading would
weaken. That arm remains available and is the only remaining arm with any
information in it.

`t1.00` IS VACUOUS AND MUST NOT BE RUN. It was registered as a free internal
consistency check. It is free of information too. `models/event_ar.py` line 365
reads `th_temp = temperature if th_temperature is None else th_temperature`, so
with the swept speed temperature at 1.0, passing `--th-temp 1.00` and passing
nothing are the SAME COMPUTATION. `research/w4_ar_eval.py` and
`models/event_ar.py` were both last modified 2026-08-05 21:11, before every arm
file in `/home/aaronadmin/w4_arms` was written, so there is no code drift for it
to catch either. Those three arms would re measure the known 0.006 noise floor
for a third time. Recorded here so nobody spends the GPU hour rediscovering it.

### The second thermal kill, and the same watchdog hole a second time, 2026-08-06

The restart was killed at 79C after seventeen minutes, having completed one arm.
Widening the idle gap from 90s to 240s did not help and the kill came SOONER than
with the narrow gap, which points at the machine being heat soaked after a full
day of load rather than at the duty cycle. GPU peaks: the beta curve 75C, the
first sweep 79C, the restart 79C. The earlier nine arm and three arm runs peaked
at 78C, one degree under the threshold, so this workload has been running at its
thermal limit all along rather than comfortably inside it.

THE WATCHDOG HOLE RECURRED, in the fix for itself. The repair after the first
kill added `pkill -f 'w4_arms/sweep.sh'`. The runner for the restart was named
`sweep2.sh`, created after that line was written, and the pattern does not match
it. So the runner survived the kill a second time, in its idle gap, with no
watchdog left alive. It was stopped by hand 26 seconds in, again inside the gap,
and again that was timing rather than design.

PATTERN MATCHING ON RUNNER NAMES IS ABANDONED. The contract is now a SENTINEL
FILE. `wd2.sh` writes `/home/aaronadmin/w4_arms/STOP` as the FIRST thing it does
on firing, before any kill, so a runner waking during the kill still stops even
if everything after it fails. Every runner checks for `STOP` and for the
watchdog's own liveness before launching each arm and exits if either check
fails. A runner that does not carry both checks is not safe to launch. `sweep2.sh`
carries them.

## Is the speed lag 2 bump a new fact, or w4_position wearing a different hat, 2026-08-06

The autocorrelation thread was left at the top of the open list with a claim
attached to it that this section starts by withdrawing.

**The correction, made before anything is built on it.** I described the speed
autocorrelation gap twice as "a SIGN error, not a magnitude error" and said no
temperature could produce it. The first half is overstated. The partial
autocorrelations at lag 2 are POSITIVE on both sides, human 0.4464 and model
0.3418, so nothing has an inverted sign in the sense that phrase implies. What
is actually true is narrower and still real: the human autocorrelation is NON
MONOTONE, it rises from lag 1 to lag 2, and the model's decays monotonically.

    independent replication, 18,024 human trajectories, zero GPU
    lag        1       2       3       4       5       6
    human   0.6466  0.6778  0.5508  0.5049  0.3988  0.3523
    ac2 minus ac1   +0.0313        w4_seqstats had +0.0268 on a different draw

    model, base arm, from w4_ar_eval
    ac1 0.7266   ac2 0.6893   ac2 minus ac1   -0.0373

The ordering difference replicates and is far outside the draw to draw noise.
The part of the original claim that survives is that temperature cannot make it:
sharpening or softening the speed head scales the conditional dispersion, which
moves every lag in the same direction, and a monotone decay stays monotone. A
mixture of AR(1) processes with different persistence cannot make it either.
That family is convex decreasing, so it can give ac2 well above ac1 squared,
which is observed, but never ac2 above ac1. Producing ac2 above ac1 needs an
alternating component at period 2 sitting on top of a smooth trend.

**Why that is not yet a reason to build anything.** Two recorded results stand
in the way and both were read before this was written.

The first is the per feature closure. Speed autocorrelation is NOT one of the
eighteen contract features. It reaches the detector only through two of them,
because for a stationary speed series the acceleration and jerk dispersions are
fixed by the autocorrelation shape:

    var(first difference)   = 2 sigma^2 (1 - ac1)
    var(second difference)  =   sigma^2 (6 - 8 ac1 + 2 ac2)
    ratio                   = (3 - 4 ac1 + ac2) / (1 - ac1)      sigma cancels

    human   3.088      model   2.863      model short by 7.3 percent

That ratio is scale free and is exactly a lag 2 probe, so the route from this
diagnostic to the metric runs through `std_jerk` and `std_acceleration`. Which
makes any repair built on it a targeted fix to a named feature, and the section
"No per feature repair can work here, and this is now measured" says such a fix
should be shown that section first. The eighteen features are eighteen
correlated views of one global mismatch; repairing the largest carrier there
moved the joint score the WRONG way. The prior against this route is strong and
this section does not claim to have met the burden.

The second is `w4_position`. It found that human high frequency speed texture
RISES along a movement, by +0.4581 in log band power, and that the model climbs
only 48 percent as much, with almost the entire gap spent in the first two
thirds. An alternating period 2 component in speed IS high frequency texture, at
the Nyquist frequency of the event stream. So the bump measured here and the
band power profile measured there may be one fact seen twice. If they are, this
thread carries no new information and building on it would be spending GPU on a
restatement.

**The registered test, and it is free.** Position resolve the bump on the human
side. Compute ac1 and ac2 within thirds of each trajectory, standardizing with
the whole sequence so the thirds stay comparable, pooled across sequences, with
the standard error bootstrapped over SEQUENCES because positions within a
trajectory are not independent. Sequences shorter than 24 events are dropped so
that each third holds at least eight, and the whole corpus bump is reported on
the same filtered set so the length filter cannot be blamed for a difference.

    LATE CONCENTRATED   last third bump exceeds first third bump by more than
                        2 se AND first third bump is within 2 se of zero
                        -> the SAME FACT as w4_position. Redundant view.
                        Thread closes, nothing gets built on it.

    UNIFORM             every third's bump is above zero by more than 2 se and
                        no third differs from another by more than 2 se
                        -> a structural fact SEPARATE from the texture profile.
                        Thread stays alive and earns a model side measurement.

    EARLY CONCENTRATED  first third bump exceeds last third by more than 2 se
                        -> CONTRADICTS w4_position's direction. One of the two
                        measurements is then wrong. Registered now so that
                        outcome cannot be quietly reinterpreted as support.

    MIXED               anything else. Report the curve and both numbers, no
                        verdict, no build.

**The tokenization control, in the same run.** The bump is measured on tokens,
so it gets compared against the same statistic on raw speeds before the class
round trip. If the two differ by more than 2 se the bump is partly a
quantization artefact and every reading above is reported with that attached.

Nothing here authorises a build. Phase conditioning, the spectral loss term and
the FiLM rewrite remain NOT AUTHORISED. The only outcome that even asks for the
next measurement is UNIFORM.

### An amendment to the above, made before it runs

Reading the tokenizer turned up a third control that has to be in the same run,
and it is a candidate to explain the whole effect rather than a refinement.

The speed stream is not all motion. A speed class at or below `TICK_CLASS`
decodes to a speed of exactly 0, and those tick events are interleaved with
moving ones. A stream that alternates tick, move, tick, move IS an alternating
component at period 2 by construction, and it would raise ac2 above ac1 with no
motor texture involved anywhere. If human tick placement alternates more than
the model's does, the bump is a fact about event scheduling, not about how
people move, and the whole "smooth trend plus period 2 texture" reading in the
previous section is wrong.

So the run also reports the bump on the motion only subsequence, ticks dropped
and the remaining speeds concatenated, alongside the tick share.

    if the bump SURVIVES tick removal at more than 2 se above zero
        the alternation is in the speeds themselves and the position resolved
        reading above stands as written

    if the bump DISAPPEARS with ticks removed, within 2 se of zero
        the bump is tick scheduling. The previous section's reading is
        withdrawn, the thread stops being about speed texture, and what is left
        is a question about tick placement that the position branches do not
        address

This changes no threshold and no branch above. It adds one control and one way
for the section that precedes it to be wrong. Still zero GPU, still nothing
authorised to build.

### THE RESULT. The tick control fired, and this whole thread was already closed in this file

`research/w4_acf_position.py`, 20,000 human trajectories of at least 24 events,
zero GPU, standard errors bootstrapped over sequences at 400 resamples.

```
  tick share of all human events   0.1119

  WHOLE SEQUENCE, tokenized speeds
    lag              1        2        3        4        5        6
    acf         0.5925   0.6565   0.5435   0.5048   0.4213   0.3686
    bump ac2 minus ac1  +0.0641  se 0.0021

  TOKENIZATION CONTROL, raw sqrt(s2) speeds
    acf         0.5926   0.6567   0.5436   0.5050   0.4215   0.3688
    tokenized minus raw +0.0000  se 0.0032   clean

  TICK CONTROL, motion events only
    acf         0.7009   0.6763   0.5967   0.5189   0.4437   0.3675
    bump                -0.0246  se 0.0010   DISAPPEARS
```

The amendment's second branch fired. The bump is tick scheduling. With the
zero speed events dropped the human autocorrelation is cleanly monotone and the
non monotonicity is gone, so the "smooth trend plus a period 2 motor texture"
reading in the section above is WITHDRAWN. The tokenization control is clean to
four decimal places, which rules out the quantizer as the source but does not
rescue the reading.

The position resolved table ran and is recorded for completeness, but its
branches were conditional on the bump surviving tick removal and it did not, so
the EARLY CONCENTRATED label the script printed is VOID. It is describing where
the ticks sit, not where any texture is. That gate belonged in the script and
was only in the prose, which is a defect in the instrument, not a result.

```
    third           ac1      ac2      bump       se
    first        0.8328   0.8952   +0.0624   0.0037
    middle       0.4691   0.5411   +0.0720   0.0027
    last         0.4648   0.5157   +0.0510   0.0023
    last minus first bump  -0.0115  se 0.0035
```

**None of this is new, and that is the finding that matters.** Two sections of
this file, both written before this session, already contain it.

The 2026-08-04 spectral run recorded the identical mechanism and said so in
almost the same words, that the signature "does NOT come from splicing the ticks
out, it comes from putting them in", with human motion only 0.6958 / 0.6379
against human every event 0.6260 / 0.6343. Those are the numbers reproduced
above at a larger n and with error bars, 0.7009 / 0.6763 against 0.5925 /
0.6565. Same signs, same conclusion, no disagreement. That section also recorded
that on the 125 Hz time axis, which is the ONLY axis the contract scorer reads,
the bump is absent altogether, human 0.8254 / 0.7313 / 0.6332, cleanly monotone.

And `w4_stillprice` had already priced the channel this leaves behind. Still
placement fully destroyed in real human data is worth 0.052, destroyed locally
0.007, and at the still counts the two models actually emit it is worth about
0.01, against a gap of 0.11 to 0.14. Its closing line is "do not spend training
on still placement or still count."

**The process failure, stated plainly because it cost a segment of a session.**
The 2026-08-04 section carries an explicit standing instruction: the event index
and the time axis disagree in SIGN on smoothness, so "any over smoothing claim
resting on an event index number has to be re read on the time axis before it is
used", naming `w4_seqstats` as one of the three files needing that caveat. The
autocorrelation thread was promoted to the top of the open list in RESUME.md on
the strength of `w4_seqstats`'s `s_ac1` and `s_ac2`, which are event index
numbers, without that re read. The instruction to check was already in the file
and was not followed. Everything downstream of it, including the claim that this
was a sign error no temperature could produce and the jerk to acceleration
algebra motivating a build, was reasoning on a statistic the record had already
disqualified for this purpose.

What the run legitimately adds is small and is stated as such: an independent
replication at 20,000 trajectories with bootstrapped error bars, a clean
tokenization control that was not previously run, and the position resolution.
None of it changes a decision.

**Closed.** The speed autocorrelation thread is closed, on the event axis
because the effect is tick interleaving rather than motor texture, and on the
time axis because the effect is not there at all. The jerk and acceleration
route to the detector is not pursued: it was a targeted fix to two named
features, and the per feature closure governs it.

## The typicality test. Can the model's own likelihood see what the detector sees, 2026-08-06

Carried in RESUME.md as "the joint likelihood gap and typicality test, designed
but not run". No design was ever written down, so this is the registration.

**Why this and not another channel.** The record now says four things that only
fit together one way. Every hand picked channel prices small, no single one
above 0.05 against a gap of 0.15. The per feature closure says the eighteen
features are eighteen views of one mismatch and repairing the largest carrier
moved the joint the WRONG way. The one step conditionals are calibrated, the
event level speed and turn joint is right, and the fitted inverse temperature is
1.05, so the model is very nearly calibrated by its own likelihood. And the
targeted exchange rate says the whole remaining gap is about a fifth of a nat
per trajectory sitting somewhere specific. A model that is calibrated everywhere
anyone has looked, and still separable at 0.66, raises one question that has
never been asked directly: does its OWN likelihood see the defect at all.

**The instrument.** Take held out human sequences the model never trained on,
with their true conditioning vectors. Generate one model sequence per held out
human sequence using EXACTLY that human's conditioning vector, one shot, no
candidates, no selection. The conditioning distribution is then identical on
both sides by construction and cannot be a confound. Run one teacher forced pass
over each set and read the model's own per token negative log likelihood on
both. Then ask how separable the two sets are USING THAT NUMBER ALONE.

**What the difference of means is and is not.** The mean NLL on generated
sequences is a Monte Carlo estimate of the model's own entropy H(p). The mean on
real sequences is the cross entropy H(q,p). Their difference is
KL(q||p) + H(q) - H(p), which is NOT KL and must never be reported here as KL.
The separability is a legitimate measure on its own terms and does not depend on
that decomposition.

    BRANCH A, BLIND        AUC from the model's own NLL is at or below 0.53 AND
                           within 2 se of the real against real floor
                           -> the training objective cannot see the defect. The
                           contract detector separates at 0.6612 on the same
                           samples the model considers unremarkable. Maximum
                           likelihood on this factorization is then the wrong
                           objective rather than an under optimised one, and no
                           amount of capacity, data or decoding tuning closes
                           the gap. This would be the strongest negative result
                           in the programme and it redirects everything to
                           objective design.

    BRANCH B, VISIBLE      AUC at or above 0.58
                           -> the model assigns systematically different
                           likelihood to its own samples than to real data. It
                           already knows. The defect then lives in the sampling
                           path rather than in the learned distribution, and
                           what is wrong is reachable without retraining.

    BRANCH C               between 0.53 and 0.58. Report the number and the
                           direction. No verdict, no build.

**The direction, registered separately because it flips the reading.**

    generated NLL BELOW real   the model emits sequences MORE probable under
                               itself than real data is. Mode seeking, too
                               regular. This is what the record predicts:
                               longer organised turn runs, too few pauses, too
                               persistent on the event axis.
    generated NLL ABOVE real   over dispersed, the sampler wanders into its own
                               low probability regions.

The size of that gap in nats per trajectory converts through the TARGETED
exchange rate, 1.4231 AUC per trajectory nat with a 2 sigma floor of 0.0927,
into a predicted AUC. If the prediction EXCEEDS the observed excess of 0.1512
then the typicality gap over explains the entire gap and the exchange rate's
linearity assumption is refuted at that point rather than supported. That
outcome is registered as informative, not as a failure of this run.

**Two guards. If either fails the primary reading is void, not adjusted.**

    LENGTH GUARD   AUC using sequence length alone must sit near the floor. The
                   model chooses its own stopping point, so if length separates
                   then any NLL result may be length wearing a disguise. This is
                   `w4_audit`'s lenAuc guard and it is used the same way.
    FLOOR GUARD    two disjoint halves of the real held out set, scored by the
                   same instrument, must land within 2 se of 0.50.

**Scope, fixed now.** n=1500 paired sequences, `event_ar_v2_s40000.pt`, the same
checkpoint, split seeds and never seen partition as `w4_price` and
`w4_beta_curve`, so every number composes with what is recorded. Sequences under
12 events are dropped on both sides. Live positions only, the terminator is
excluded, and the direction head is read only where a direction exists, all
matching `w4_beta_curve` exactly.

This is a DIAGNOSTIC. No serving change and no training change follows from any
branch without a separate registration. Phase conditioning, the spectral loss
term and the FiLM rewrite remain NOT AUTHORISED. Generation is the hot half, so
it runs under the watchdog and the STOP sentinel contract.

### THE RESULT. BRANCH B. The model's own likelihood sees almost everything the detector sees

Run `w4_2026-08-06T222201+0000_fced2552`. 1,500 held out human sequences paired
one to one with 1,500 generated sequences on identical conditioning vectors,
`event_ar_v2_s40000.pt`, one shot, no selection. Peak 76C, no thermal kill.

```
  head       real nll    gen nll  gen - real      auc      se
  all          1.3985     1.6644     +0.2659   0.6345  0.0105
  s            2.1187     2.3313     +0.2126   0.5809  0.0108
  th           1.1537     1.4283     +0.2746   0.6046  0.0107
  dt           0.8899     1.2077     +0.3178   0.6430  0.0098

  FLOOR GUARD   real vs real          0.5167  se 0.0124   PASS
  LENGTH GUARD  length alone          0.5121  se 0.0088
  contract detector, same checkpoint  0.6612
```

**A single scalar, the model's own average surprise, separates its samples from
real data at 0.6345 against a contract detector that manages 0.6612 with
eighteen features and a random forest.** Both guards pass. The real side
reproduces `w4_beta_curve` on an independent draw, direction head 1.1537 against
1.1307, so the scoring pass is the same instrument.

**The direction is the opposite of the registered prediction and it is the
finding.** The registration predicted generated NLL BELOW real, mode seeking,
too regular. It is above, by a lot, on every head separately. Since
H(q,p) = H(q) + KL(q||p) and KL is non negative:

    H(p) - H(q)  =  0.2659 + KL(q||p)  >=  0.2659 nats per token

The model's entropy exceeds the data's by at least 0.2659 nats per token. It is
OVER DISPERSED, and the excess is enormous next to the 0.0018 nats per token
that the Pinsker floor says the whole contract gap is worth.

**This reconciles with beta star = 1.05 rather than contradicting it, and the
reconciliation is the mechanism.** On REAL histories the model is very nearly
perfectly calibrated: the best inverse temperature is 1.05 and buys 0.0008 nats.
On its OWN histories it is 0.2659 nats per token less certain. Teacher forced
and free running are measurably different regimes for this model, and every
likelihood number in this file until now was measured in the first one.

**Why maximum likelihood cannot see this, stated as the mechanism it is.** MLE
minimises H(q,p). That penalises p for failing to COVER q and does not penalise
p at all for putting mass where q has none. A model can therefore be far broader
than the data while paying almost nothing in held out likelihood. That is
exactly the configuration measured here: excellent cross entropy, near perfect
temperature calibration, and at least 0.27 nats per token of excess breadth.

That single fact accounts for the whole failure record without anything else
being added. Per feature repairs fail because no feature is wrong, the
distribution is too wide. Every hand picked channel prices small because breadth
is not a channel. Generic likelihood improvement is worth 845x less than a
targeted one because likelihood is already near optimal and is blind to the
defect. And the detector separates at 0.66 because a random forest reading
eighteen correlated views of one distribution is very good at spotting one that
is too broad.

**A correction to this morning's temperature verdict, and it is a withdrawal.**
Today's sweep found that sharpening the direction head buys AUC while making
teacher forced held out likelihood worse, and this file concluded SCORER
ARTEFACT on exactly that ground. Under the measurement above that inference does
not hold. Sharpening reduces breadth. A mode covering objective evaluated on
real histories is precisely the yardstick that must punish it. So the pattern
that was read as an artefact is the predicted signature of partially correcting
an over dispersed model, and the exposure bias reading the registration named
and argued against now has direct measurement behind it.

    WITHDRAWN: "THE TEMPERATURE LINE IS A SCORER ARTEFACT", 2026-08-06.
    The AUC gains at th_temp 0.928 and 0.85 are not established as artefacts.
    The direction head temperature closure in the DO NOT RE DERIVE list is
    reopened along with it. What remains true is that temperature alone does
    not reach the target: th_temp 0.85 sits at 0.6221 and touches one of three
    heads.

**The registered conversion was ill posed and is withdrawn, not reported.** The
registration said to convert the typicality gap through the targeted exchange
rate. That was a category error, mine, made before the run. The exchange rate
prices a CHANGE to the model in nats of held out likelihood. The typicality gap
is a property of the CURRENT model and is not a change to anything. The script
printed a predicted AUC of 66.07, which is the arithmetic announcing that its
input was the wrong quantity. Nothing about linearity is established or refuted
by it.

**Caveat on the dt column, which carries the largest gap.** `EventARModel.sample`
clamps sampled dt at `DT_MAX_MS`, so generated dt can pile onto a value the head
gives little mass, inflating that column specifically. The headline does not rest
on it: `s` involves no clamp at +0.2126 and `th` involves none at +0.2746.

**Scope, stated rather than buried.** 141 of 1,500 generated sequences fell below
12 events and were dropped to match the real side's filter. That removes short
sequences from the generated set only, which makes the test conservative.

**What is now open and what adjudicates it.** The over dispersion is measured on
whole sequences and is not yet position resolved. If the gap GROWS along the
trajectory it is accumulation, which `w4_position` ruled out on the ground that
the band power excess shrinks with distance from the start, and one of the two
would then be wrong. If it is flat or present from the first event it is not
accumulation and the model is in a broader regime immediately. That measurement
is cheap, uses the streams this run already knows how to make, and should be run
before anything is built.

## Where the over dispersion lives along a movement, 2026-08-06

`w4_typicality` measured at least 0.2659 nats per token of excess breadth on
whole sequences. Whole sequence numbers cannot distinguish a model that starts
in the wrong regime from one that wanders into it, and those two have different
causes and different fixes. This resolves the same gap by position.

**It also adjudicates against a recorded result, which is why the branches are
written this way.** `w4_position` found the model's high frequency excess is
LARGEST in the first window and SHRINKS with distance from the start, at a slope
of -0.18 and 5.2 sd, and used that to rule accumulation out. `w4_drift` priced
the whole training time exposure fix family at a third or less on the same
reasoning. If the typicality gap GROWS along a movement, those results and this
one cannot both be right.

**Instrument.** The `w4_typicality` pass, with per token NLL kept by position
instead of summed per sequence. Primary axis is FRACTIONAL position, t over L, in
eight bins, because every sequence contributes to every bin and there is
therefore no survival conditioning: a late absolute index is only reachable by
long sequences, a late fractional bin by all of them. Absolute index 0 to 5 is
reported as a secondary read, where survival conditioning is negligible because
nearly every sequence is still live, and it is the direct test of "wrong from
the first event".

    GROWS       last bin gap exceeds first bin gap by more than 2 se
                -> accumulation. CONTRADICTS w4_position and w4_drift. One of
                them or this is wrong and the disagreement gets resolved before
                anything is built on either.

    IMMEDIATE   first bin gap is above zero by more than 2 se AND no bin differs
                from another by more than 2 se
                -> not accumulation. The model is in a broader regime from the
                start, which is consistent with w4_position and points at the
                objective rather than at exposure along the rollout.

    SHRINKS     first bin gap exceeds last bin gap by more than 2 se
                -> the excess is concentrated at launch, the same shape and the
                same direction w4_position found in the band power. The two
                measurements would then be one fact seen twice, which is the
                outcome that most constrains what to build.

    MIXED       anything else. Report the curve, no verdict.

Standard errors bootstrap over SEQUENCES at 400 resamples, as everywhere else in
this programme, because positions inside one trajectory are not independent.

**One thing changes in the instrument and it is recorded here rather than left
in the code.** The generated token streams are saved this time. `w4_typicality`
regenerated 1,500 sequences and threw them away, so every follow up question
costs another twenty five minutes of GPU and another thermal cycle. Saving them
makes every later position, head or subset question free. The generation seed
and batching are unchanged, so the streams reproduce the run already recorded.

Diagnostic only. Nothing is authorised to be built by any branch. Phase
conditioning, the spectral loss term and the FiLM rewrite remain NOT AUTHORISED.

### THE RESULT. IMMEDIATE. Correct at the first event, over dispersed at every one after, and flat

```
  whole sequence check   real 1.2677  gen 1.5914  gap +0.3236

  FRACTIONAL POSITION, eight bins, no survival conditioning
    bin        real      gen       gap      se
    0.06     1.7874   2.0717   +0.2843  0.0343
    0.19     1.3602   1.6934   +0.3332  0.0298
    0.31     1.2844   1.6199   +0.3355  0.0301
    0.44     1.2195   1.5583   +0.3388  0.0285
    0.56     1.1834   1.5141   +0.3307  0.0286
    0.69     1.1326   1.4663   +0.3337  0.0277
    0.81     1.1124   1.4235   +0.3112  0.0266
    0.94     1.0171   1.3383   +0.3212  0.0261
    last minus first gap  +0.0369  se 0.0431

  ABSOLUTE INDEX
    idx        real      gen       gap
    0        3.4655   3.4985   +0.0329
    1        2.0927   2.3266   +0.2338
    2        1.7485   1.9470   +0.1985
    3        1.6287   1.8764   +0.2478
    4        1.5305   1.8382   +0.3077
    5        1.5267   1.7823   +0.2556
```

The IMMEDIATE branch fired. The gap does not grow: last minus first is +0.0369
against a standard error of 0.0431. There is no accumulation, so nothing here
contradicts `w4_position` or `w4_drift`, and the training time exposure fix
family stays closed on the ground it was closed on.

**The absolute index table carries the sharp part and it was not predicted.** At
event 0 the gap is +0.0329, which is nothing. At event 1 it is +0.2338 and it
stays there for the rest of the movement. Event 0 is the only position with no
history: the model sees the conditioning vector and nothing else. So the model's
opening distribution is CORRECT, and the excess breadth appears the moment its
own history enters and then neither grows nor shrinks. It is a step, not a
drift.

Note on the two whole sequence numbers. `w4_typicality` reported +0.2659 and
this reports +0.3236 on the same run. `w4_typicality` weights each SEQUENCE
equally, this weights each TOKEN equally, and NLL falls along a movement, so
token weighting favours long sequences. Both are correct and they answer
different questions. Neither is a discrepancy.

### The decomposition that separates the two remaining explanations, and it is free

"Over dispersed from event 1 onward" still has two causes and they need
different fixes.

    OBJECTIVE   the learned conditional is intrinsically too broad wherever
                there is history, and the states the model visits are beside
                the point.
    STATES      the conditional is fine, but the model's own emitted history
                puts it in genuinely higher entropy states than real histories
                do.

Three quantities separate them exactly, and two are already measured.

    A = H(q,p | real states)   cross entropy on real data       1.2677
    B = H(p   | real states)   the model's OWN entropy at real states
    C = H(p   | gen  states)   the model's own entropy at its states  1.5914

    C - A = 0.3236 = (B - A) + (C - B),  exact, no approximation

    B - A  is intrinsic over dispersion at states the model is known to handle
           well. It is nonzero only if the objective left breadth unpenalised.
    C - B  is the state distribution effect, the model's own history taking it
           somewhere hotter.

B needs no generation and no new sampling. It is the mean entropy of the same
conditional distributions the teacher forced pass over real data already
computes, minus nothing. The streams are saved, so this costs one forward pass.

    B - A DOMINANT, at least twice C - B
        -> the objective is the defect. Maximum likelihood did not penalise
        breadth and the model is broad everywhere. Sharpening at serving is
        then a real correction rather than detector fitting, which is the
        reading this morning's withdrawn artefact verdict now leaves open.

    C - B DOMINANT, at least twice B - A
        -> the states are the defect, not the conditional. The model is fine
        where people go and is taking itself elsewhere.

    NEITHER dominant, within a factor of two
        -> report both, claim neither, and say so.

Registered before the run. Diagnostic only, nothing authorised to be built.

### THE RESULT. STATES DOMINANT, 95 percent. The mode covering account is WRONG

```
  A  H(q,p | real states)   cross entropy on real   1.2677  se 0.0160
  B  H(p   | real states)   model entropy, real     1.2834  se 0.0148
  C  H(p   | gen  states)   model entropy, own      1.6020  se 0.0158
     cross check, gen NLL at gen states             1.5914   w4_typpos had 1.5914

  B - A  intrinsic over dispersion at real states   +0.0157      4.7 percent
  C - B  the state distribution effect              +0.3185     95.3 percent
  C - A  total                                      +0.3342
```

**The instrument validates itself.** Tokens drawn from the model's own
conditionals must have mean NLL equal to those conditionals' entropy. Measured
1.5914 against 1.6020, agreeing to 0.0106 on a quantity of 1.6. That identity
was not enforced anywhere in the code and it holds, so the entropy and the cross
entropy paths are both correct.

**The mechanism proposed in the section above is refuted by its own follow up.**
That section argued the defect was maximum likelihood failing to penalise
breadth, leaving the conditionals too broad everywhere, and it called that a
unifying account of the whole failure record. The decomposition prices that
explanation at 4.7 percent. At states real people visit, the model's entropy and
its cross entropy agree to 0.0157 with standard errors of 0.015, which is
consistent with zero. There is no meaningful intrinsic over dispersion. The
account was wrong and is withdrawn.

**What survives, stated precisely.** The bound H(p) - H(q) >= 0.334 nats per
token still holds, because H(q) <= H(q,p) = A and H(p) = C. The model's overall
distribution IS broader than the data's. What is now known is that this does not
come from broad conditionals. It comes from the model visiting states real
movements do not.

A caveat that must travel with B - A. B = A is what a perfect model gives, so a
measured B - A of zero RULES OUT gross over dispersion at real states. It does
not prove the conditional correct, because B - A equals
H(p|real) - H(q|real) - KL(q||p|real) and a nonzero entropy excess exactly
cancelled by a nonzero KL would also read zero. The claim here is the exclusion,
not the proof.

**The reading this leaves, and it is a tail problem rather than a breadth
problem.** Conditionals that are right on average at real states, combined with
a model that reaches hotter states than real data does, can only be reconciled
one way: the mass is misplaced in the TAIL. Average cross entropy is almost
insensitive to a small amount of misplaced tail mass, so the likelihood looks
excellent, while sampling hits that tail some of the time and a single unusual
token is enough. `w4_typpos` measured exactly that shape. The gap is +0.0329 at
event 0, where there is no history and the two sides are the same state, and
+0.2338 by event 1, after ONE self generated token, and then flat for the rest
of the movement. One token is enough and it does not compound.

That also explains the whole record in a way the withdrawn account did not.
Average likelihood is near optimal and beta star is 1.05 because tails barely
enter an average. Generic likelihood improvement is worth 845x less than a
targeted one for the same reason. No per feature repair works because no feature
is wrong. And the detector separates at 0.66 because trajectories that took one
tail draw look wrong afterwards.

**What this does NOT settle.** Whether serving time sharpening or truncation
fixes it. The mechanism predicts it should help, because both remove exactly the
tail draws, and the temperature sweep did buy AUC. That is a prediction with
support, not a result. The withdrawal of the artefact verdict stands on its own
grounds and is not strengthened by this section.

**The measurement this sets up.** The tail hypothesis is directly checkable and
needs no generation, because the streams are saved. Read the per token
probability the model assigns to what was actually emitted, on real sequences
and on its own, and compare the LOW quantiles rather than the means. If the tail
account is right the means will be close, already known, and the generated
sequences will carry many more very low probability tokens, with the excess
concentrated in a small fraction of positions rather than spread over all of
them.

## Is it a few ruined moments or a uniform shift, 2026-08-06

**First, a correction to the registration sentence that closed the section
above.** It said "if the tail account is right the means will be close, already
known". That is wrong and the numbers contradicting it are in the same section:
the means are 1.2677 and 1.5914, differing by 0.3342. What the tail account
actually predicts is that the MEDIAN is close while the mean gap is carried by a
small fraction of positions. The test below is written on that, not on the
sentence it replaces.

**The exact decomposition this uses.** A mean is the integral of the quantile
function, so

    mean_gen - mean_real  =  integral over u in [0,1] of [Qgen(u) - Qreal(u)]

which splits the 0.3342 into contributions by quantile band with no
approximation and no binning choice. Per token surprise, all live positions, the
three heads pooled and also reported separately.

    TAIL        the top 10 percent of positions carry at least 50 percent of the
                mean gap AND the median shift is under 25 percent of it
                -> a small number of very bad moments. Serving time truncation
                or sharpening removes exactly those draws and is then the
                obvious thing to price next.

    UNIFORM     the median shift is at least 75 percent of the mean gap
                -> every position is a little worse. The tail account is wrong,
                the defect is diffuse, and truncation cannot help because there
                is no tail to cut.

    MIXED       anything else. Report the quantile curve and claim neither.

**A second axis, because it changes what a fix would even look like.** Are the
bad moments spread thinly across all generated movements, or concentrated in
some of them. Reported as the share of sequences carrying at least one token
below p = 0.001, real against generated, and the share of the total excess those
sequences carry. If a minority of trajectories hold most of the damage then the
reachable prize is larger than the mean gap suggests, and if every trajectory
carries a little then it is smaller.

This is stated as a reading, not a branch, because no threshold for it was
defensible before seeing the scale. It is reported and not used to decide
anything.

No generation. The `w4_typpos` streams are reused, so this is one forward pass
per side. Diagnostic only, nothing authorised to be built.

### THE RESULT. UNIFORM. The tail account is refuted, and so is the third mechanism in a row

```
  253,876 real tokens, 217,786 generated

  POOLED   mean gap +0.3236   median shift +0.5091   (157.3% of the gap)
    quantile band       contribution    share      positions
    0.00 to 0.50           +0.0947    29.3%          50%
    0.50 to 0.75           +0.1335    41.2%          25%
    0.75 to 0.90           +0.0605    18.7%          15%
    0.90 to 0.95           +0.0187     5.8%           5%
    0.95 to 1.00           +0.0163     5.0%           5%

  top 10 percent of positions carry 10.8 percent of the gap
  median shift is 157.3 percent of the gap

  share of sequences with at least one token below p=0.001
    real 48.8 percent    generated 55.3 percent
```

The UNIFORM branch fired and it fired hard. The top decile of positions carries
10.8 percent of the gap against the 10 percent it would carry if the excess were
spread perfectly evenly. There is no tail. The median position is shifted by
0.5091, MORE than the mean gap of 0.3236, so the typical position is hurt worse
than the average one. Extreme events barely differ, 48.8 against 55.3 percent of
sequences. The same shape holds on all three heads separately.

If anything the gap is smallest exactly where a tail account needs it largest.
Averaged inside each band the difference runs 0.19, 0.53, 0.40, 0.37, 0.33 from
the confident end to the surprised end. It peaks in the middle and falls at both
ends.

**Truncation and sharpening cannot work by the mechanism proposed for them.**
There are no rare bad draws to cut. Whatever serving time sharpening does for
the contract score, it is not removing a tail, because there is no tail.

**Three mechanisms proposed in this session, three refuted by the next
measurement.** Mode covering breadth, refuted by `w4_entdecomp` at 4.7 percent.
Accumulating drift, refuted by `w4_typpos`, which found no growth. Rare ruinous
draws, refuted here. Each was proposed as the reading of a real result and each
was wrong. The results themselves have all held and have all narrowed the
target, so the failure is specifically in proposing mechanisms, not in the
measurements. The response is to stop proposing them.

**What is now established, with no story attached.**

1. At states real movements visit, the model's entropy and cross entropy agree
   to 0.0157. Gross over dispersion there is excluded.
2. At states the model's own output reaches, its entropy is 0.3185 higher.
3. That excess appears after ONE self generated event, is at full size by event
   1, and neither grows nor shrinks for the rest of the movement.
4. It is spread evenly over positions, not concentrated in bad moments, and is
   mildly largest at middling confidence.
5. The model's own average surprise separates its output from real data at
   0.6345, against a contract detector at 0.6612.

Read together with no mechanism added, those say the model's own trajectories
occupy a systematically different and uniformly higher entropy region of state
space than real ones, from the first event, and that this is a property of WHICH
STATES it reaches rather than of how it behaves at any given state.

**The next measurement follows from that sentence and needs no hypothesis.** The
state is not a black box. `prefix_state` is four numbers per position, built from
the conditioning vector and the prefix, and it is what the trunk actually reads.
The states themselves can be compared directly, real against generated, with no
model evaluation and no GPU at all. That answers "which states differ and along
which coordinate" by measurement rather than by another guess, and the streams
are already saved.

## Which coordinate of the state carries the excess, 2026-08-06

Registered before the script exists.

Everything established so far points at one sentence. The model's conditionals
are right where real movements go, and its own trajectories reach states that
are uniformly hotter. The obvious next question is WHICH states, and unusually
for this file that question needs no hypothesis, because the state is not
hidden. `prefix_state` is six numbers per position and it is exactly what the
trunk reads:

```
  0  log1p(distance still to cover)
  1  unit x of the remaining vector
  2  unit y of the remaining vector
  3  elapsed time as a fraction of the commanded duration
  4  step index as a fraction of the buffer
  5  log1p(distance travelled so far)
```

So instead of proposing a fourth mechanism I am going to measure which of those
six coordinates the 0.3185 nat state effect lives on. Three refuted guesses is
enough evidence that guessing is the weak step.

**The decomposition, which is exact.** Take the per token entropy of the model
under a forward pass, which is what `w4_entdecomp` already summed to B and C.
Bin every token, real and generated, by one state coordinate, using bins cut at
the REAL distribution's quantiles so the bins are not chosen by the thing being
measured. Write `w[b]` for the share of tokens in bin b and `Hbar[b]` for the
mean entropy in bin b. Then

```
  C - B = sum_b (wgen[b] - wreal[b]) * Hbar_real[b]        BETWEEN
        + sum_b  wgen[b] * (Hbar_gen[b] - Hbar_real[b])    WITHIN
```

exactly, with no residual. BETWEEN is the part explained by the model visiting
that coordinate's values in the wrong proportions. WITHIN is the part that
survives after holding that coordinate fixed. A coordinate that carries the
effect shows a large BETWEEN. A coordinate that is irrelevant shows a BETWEEN
near zero and passes the whole gap to WITHIN.

**Branches, fixed now.**

- **NAMED.** Some coordinate has BETWEEN of at least 50 percent of `C - B`.
  That coordinate is the mechanism and the report names it, along with the
  direction of the shift.
- **DIFFUSE.** No coordinate reaches 25 percent. The state shift is not along
  any single readable axis, the marginals are close and the difference is in the
  joint geometry, and the nearest neighbour read below becomes the primary
  result rather than the secondary one.
- **PARTIAL.** In between. Report the full six way table and claim only what the
  numbers support.

**A prediction I am recording so it can be wrong.** I expect coordinates 3 and 4
to be inflated by a length confound rather than by anything mechanistic, since
generated sequences set their own lengths. If either of those wins, the honest
reading is a confound and not a mechanism, and I will say that rather than
claim a result. I am registering that now precisely so I cannot claim it later
as a discovery.

**The joint read, secondary unless DIFFUSE fires.** Marginal binning is blind to
a difference that lives only in the joint. So also whiten the six dimensional
state using the real covariance and measure, for a sample of positions at
matched fractional position, the distance from each generated state to its
nearest real state, against the leave one out nearest real to real distance. If
the generated states sit off the real manifold rather than merely being
distributed differently along it, that shows up here and nowhere in the
marginals.

**What either answer buys.** A named coordinate is directly actionable, because
`prefix_state` is computed the same way in training and in sampling, so a
mismatch on a readable coordinate is a fixable defect rather than a fact about
maximum likelihood. DIFFUSE plus an off manifold nearest neighbour result is
also informative, and points at compounding geometry rather than at any single
control variable, which would be the first result in this thread to argue for a
change in architecture rather than in objective.

Zero GPU beyond two forward passes. No generation, the streams are saved.

### THE RESULT. DIFFUSE, and the state is not where the defect lives at all

```
  87,313 real positions, 74,368 generated
  B 3.7317   C 4.6913   C - B +0.9596

  coordinate                     BETWEEN      se    share    WITHIN   gen mean  real mean
  0 log1p remaining distance     +0.0681  0.0257     7.1%   +0.8914     4.5015     4.3682
  1 unit x remaining             +0.0051  0.0118     0.5%   +0.9545     0.5968     0.7038
  2 unit y remaining             +0.0006  0.0092     0.1%   +0.9590     0.0235     0.0051
  3 elapsed / commanded          -0.0240  0.0061    -2.5%   +0.9836     0.4916     0.4787  (confounded)
  4 index / buffer               +0.0549  0.0232     5.7%   +0.9047     0.1735     0.1907  (confounded)
  5 log1p distance travelled     +0.0674  0.0174     7.0%   +0.8922     4.8067     4.5600

  VERDICT  DIFFUSE
```

The units differ from `w4_entdecomp` because this run averages per POSITION
over three heads while that one averaged per TOKEN. The gap is the thing to
compare and it reproduces: 0.9596 / 3 = 0.3199 against 0.3185. Same quantity.

**No coordinate reaches 25 percent.** The largest is 7.1 percent and the two I
registered in advance as length confounded are not even the largest, so the
confound I warned about did not arise. WITHIN is 0.89 to 0.98 on every single
coordinate. Holding any one of the six fixed leaves essentially the whole effect
standing.

**And the joint read says the states are not off manifold either, after a bug in
my own first version of it was fixed.** The first run returned a ratio of 2.949
and would have been reported as a clean off manifold result. It was an artefact.
The reference cloud was drawn from the same pool as the real queries, so a real
query's nearest neighbour was almost always the ADJACENT POSITION OF ITS OWN
MOVEMENT, a state one event away and therefore trivially close, while a
generated query had no such partner. Excluding same sequence reference points
puts both sides on the same question, how far is this state from the nearest
state of a DIFFERENT real movement:

```
  median nearest real distance, real query        0.3836
  median nearest real distance, generated query   0.3959
  ratio                                           1.032
```

Generated states sit on the real manifold, at the same typical distance from
other real movements that real states are. Caveat stated plainly: 20,000
reference points in six whitened dimensions cannot rule out a subtle joint
difference. It rules out a gross one, and the exact marginal decomposition
agrees with it.

**So the conclusion is a relocation, and it is the most useful thing this thread
has produced.** The trunk reads two things, the six number geometric summary and
the TOKEN HISTORY ITSELF. Every measurement above says the geometric summary is
right. The model's own trajectories are in the right place, at the right time,
with the right distance remaining, having travelled the right distance, facing
the right way, and they reach those states in the right proportions. The 0.32
nats is therefore carried by the only other thing in the input.

**The defect is in the texture of the emitted token sequence, not in where the
movement is.** The model's samples integrate to a geometrically correct
trajectory and are made of a sequence of speed, turn and timing tokens whose
local pattern the model itself has not seen in training. That is why the excess
appears after a single event, why it never grows, and why it is spread evenly
over every position. It is not drift and it is not a bad moment. Every event is
emitted from a history that is slightly the wrong shape, and the wrongness is
replenished at each step rather than accumulated.

This also explains why the contract detector and the model's own likelihood
agree so closely at 0.6612 and 0.6345. Both are reading local sequence texture.
The eighteen contract features are overwhelmingly texture statistics.

**Next, and it is the same machinery pointed one layer up.** Run the identical
exact BETWEEN and WITHIN decomposition on features of the recent token history,
the things the trunk can see and `prefix_state` cannot. Last speed, last turn
magnitude, last inter event time, the local speed change, the tick fraction over
a short window, the events since the last tick. If the state coordinates all
returned WITHIN and a history feature returns BETWEEN, that names the defect in
a quantity that a loss term or an input can address. The streams are saved and
this is again free.

## Which feature of the token history carries the excess, 2026-08-06

Registered before the script exists. Direct continuation of the DIFFUSE result
above, using the identical exact decomposition, aimed at the only remaining
trunk input.

Eight causal history features, all computed strictly from events before the
position, matching `prefix_state`'s convention exactly so nothing leaks:

```
  0  last speed
  1  last turn magnitude
  2  last inter event time
  3  last speed change, |s(t-1) - s(t-2)|
  4  mean speed over the last five events
  5  tick fraction over the last eight events
  6  events since the last tick
  7  turn persistence, sign agreement of the last two turns
```

Same BETWEEN and WITHIN split, same twenty real quantile bins, same sequence
clustered bootstrap, same branch thresholds as the state coordinate run: NAMED
at 50 percent, DIFFUSE below 25 percent for every feature, PARTIAL in between.

**One property of this run that the state coordinate run did not have, stated
before the numbers arrive.** The six `prefix_state` coordinates are close to a
complete description of the geometric state, so their shares were roughly
comparable to each other. These eight features are heavily correlated with one
another, so their BETWEEN shares OVERLAP and will not sum to anything
meaningful. A large share for two correlated features is one finding, not two.
For that reason a joint decomposition on the top two features together is also
computed, and the honest headline number is the joint one, not the largest
single one.

**What each branch would mean.**

- Features 0, 3 or 4 winning points at the speed profile, and the actionable
  target is the speed head's local dynamics.
- Features 1 or 7 winning points at turn structure, meaning the model's turns
  are individually plausible but their sequencing is not.
- Features 2, 5 or 6 winning points at the timing and tick channel. `w4_stillprice`
  already priced the still channel at about 0.01 AUC against a gap of 0.11 to
  0.14, so if the tick features win here that is a genuine tension between two
  results in this file and it gets reported as a tension, not resolved by
  picking the one I prefer.
- DIFFUSE again would say the defect is not in any short window summary either,
  and the remaining candidate is long range sequence structure that only the
  attention sees. That would be a real narrowing and not a failure.

No new mechanism is being proposed here. The measurement is chosen because it
exhausts the trunk's inputs, not because I expect a particular answer.

### THE RESULT. NAMED. Timing and speed roughness, jointly 55 percent

```
  87,313 real positions, 74,368 generated,  C - B +0.9596

  shares OVERLAP, these features are correlated, they do not sum

  history feature              BETWEEN      se    share    WITHIN   gen mean  real mean
  0 last speed                 +0.2135  0.0346    22.3%   +0.7461     9.2949     7.2476
  1 last turn magnitude        +0.1273  0.0197    13.3%   +0.8323     0.2777     0.3010
  2 last inter event time      +0.5034  0.0438    52.5%   +0.4562     0.0093     0.0081
  3 last speed change          +0.2574  0.0229    26.8%   +0.7022     2.8578     1.9730
  4 mean speed last 5          +0.2404  0.0372    25.0%   +0.7192     9.0624     7.0489
  5 tick frac last 8           +0.1002  0.0176    10.4%   +0.8594     0.0620     0.0818
  6 events since tick          +0.0895  0.0205     9.3%   +0.8701    12.8782    12.0127
  7 turn persistence           +0.0742  0.0091     7.7%   +0.8854    -0.1504    -0.1781

  JOINT, last inter event time by last speed change, 8 by 8   +0.5298   55.2%

  VERDICT  NAMED, jointly
```

The contrast with the state coordinate run is the whole point. There the largest
single share was 7.1 percent and WITHIN was 0.89 to 0.98 everywhere. Here one
feature alone reaches 52.5 percent and drops WITHIN to 0.4562. Same
decomposition, same bins, same bootstrap, same data. The geometric state is
clean and the token history is not.

**The tick tension clause registered above does not fire.** Features 5 and 6,
the still and tick channel, are 10.4 and 9.3 percent, near the bottom of the
table. `w4_stillprice` stands unchallenged. The timing feature that wins is the
inter event interval of MOTION events, which is a different channel from still
placement and still count, and the distinction matters enough to say twice.

**Direction of every mean, which is where the actionable content is.** The model
emits events further apart in time, 0.0093 against 0.0081 seconds, at higher
speed, 9.29 against 7.25, with speed changing more violently between them, 2.86
against 1.97, a 45 percent excess in roughness. Turns are slightly smaller and
slightly less persistent. Nothing here is subtle and none of it was visible in
the geometry, because larger steps taken less often integrate to the same
trajectory.

#### EXPLORATORY, not registered, prompted by the means above

Four of the eight features could be one primitive fact. Fewer events over the
same commanded distance and duration forces every inter event time up and every
step size up as arithmetic. So the event count itself was checked, paired on
identical conditioning:

```
  1,359 movements, paired
  events per movement   real 60.30   gen 54.72   ratio 0.9075  se 0.0218
  paired difference     -5.58  se 1.433   median 0.0
  gen shorter in 49.9 percent of movements

  by commanded duration quartile
    0.026s to 0.224s   real 31.62  gen  25.32  ratio 0.801
    0.224s to 0.401s   real 53.54  gen  37.30  ratio 0.697
    0.401s to 0.695s   real 67.98  gen  54.17  ratio 0.797
    0.695s to 2.715s   real 87.67  gen 101.83  ratio 1.161
```

The marginal is nearly innocent. The median paired difference is exactly zero
and the model produces the shorter sequence in 49.9 percent of movements, which
is a coin flip. The CONDITIONAL is not innocent at all. Real event count spans
2.77x across the duration quartiles and the model's spans 4.02x. **The model
over responds to commanded duration.** It emits far too few events for short and
middling movements and too many for long ones, and the two errors cancel in the
average, which is exactly how a defect this large stays invisible in any
aggregate statistic.

This is EXPLORATORY. It was not registered, it was prompted by looking at the
means in the table above, and it has one obvious confound in that generated
lengths are self determined while real ones are not. It gets its own registered
test before anything is claimed from it or built on it. What it is enough to
support right now is a single sentence: the four correlated speed and timing
features in the table are plausibly one fact wearing four hats, and that fact is
about how many events the model chooses to spend.

## How many events the model spends, registered properly, 2026-08-06

Promoting the exploratory event count look to a real test. Registered before the
script exists.

**The three confounds in the exploratory version, and what each gets.**

1. The `gL >= 12` filter dropped 141 of 1500 generated movements. If those are
   concentrated at short commanded durations then dropping them HID part of the
   undershoot, so the exploratory number is conservative rather than inflated.
   This run uses all 1500 and reports the filtered numbers alongside.
2. Both sides are capped at MAX_T. The share of sequences at the cap is reported
   for both, and if either exceeds 2 percent the cap is a live confound and the
   result is reported as bounded rather than clean.
3. Quartiles of duration ignore distance, which is conditioned jointly. This run
   fits both together.

**The test.** Fit the human conditional event count by ordinary least squares on
real sequences only,

```
  log L  =  a  +  b * log distance  +  c * log duration
```

then compare the model's fitted b and c to the human's, and take the residual of
each generated sequence against the HUMAN fit. Sequence bootstrap throughout.

- **CONFIRMED** if the duration slope differs from the human slope by at least
  three standard errors.
- **NOT CONFIRMED** otherwise, and the exploratory finding is withdrawn.

**The part that decides whether this matters at all, and it is the real
question.** A defect existing is not the same as a defect that costs score. So
the residual is also used directly as a one number detector. Take the residual
of every sequence against the human fit, real and generated, and compute the
two sample AUC of that single scalar.

That number is a LOWER BOUND on what any detector can achieve from this defect
alone, because a scalar that separates at some level means a detector with
access to it separates at least that well. It is not a claim about what the
contract detector currently uses, and this file has been burned before by
confusing those two, so it is written down here that it is a lower bound and
nothing else.

- **DOMINANT** if the residual alone reaches AUC 0.58. Against the contract's
  0.6612 that would be half the available signal in one scalar, and event count
  becomes the main target.
- **MINOR** below 0.54. A real defect that is not worth the training budget, and
  the timing and roughness result above has to be attacked some other way.
- **PARTIAL** in between, report and claim neither.

**What I expect, recorded so it can be wrong.** The slope test confirms and the
AUC comes out MINOR or low PARTIAL. The reasoning is that the marginal was a
coin flip, and a conditional error whose two halves cancel gives a detector less
to work with than its size suggests. If the AUC comes out DOMINANT I will have
been wrong in the useful direction.

Zero GPU, zero generation, one closed form fit.

### THE RESULT. CONFIRMED at 10.9 sigma, and PARTIAL at the very top of the band

```
  1500 paired movements, no length filter
  the gL >= 12 filter would have dropped 141
  at the MAX_T cap   real 0.00%   generated 0.40%   clean

                          intercept  log distance  log duration
  human fit                  3.9607        0.0447        0.3974
  model fit                  3.7682        0.1080        0.7056
  model minus human         -0.1925       +0.0633       +0.3082

  duration slope difference +0.3082  se 0.0282   +10.91 sigma
  distance slope difference +0.0633  se 0.0194   +3.27 sigma

  SLOPE VERDICT  CONFIRMED

  signed residual AUC   0.5781  se 0.0104
  absolute residual AUC 0.5329  se 0.0110      contract detector reads 0.6612

  COST VERDICT  PARTIAL, 0.5781

  by commanded duration quartile, all movements
     0.026s to  0.190s   real  31.67  gen  18.83  ratio 0.594
     0.190s to  0.367s   real  48.30  gen  35.18  ratio 0.728
     0.367s to  0.662s   real  66.86  gen  50.30  ratio 0.752
     0.662s to  2.715s   real  85.85  gen  97.01  ratio 1.130
    span across quartiles   human 2.71x   model 5.15x
```

Confounds resolved as registered. The cap is clean at 0.00 and 0.40 percent. The
`gL >= 12` filter WAS hiding part of the effect exactly as predicted: the
shortest quartile ratio goes from 0.801 with the filter to 0.594 without it. The
exploratory number was conservative. The distance slope is also off, at 3.27
sigma, but it is a fifth the size of the duration slope error.

**My registered prediction was wrong.** I wrote that I expected MINOR or low
PARTIAL, reasoning that a conditional error whose halves cancel gives a detector
little to work with. It came out at 0.5781, which is 0.0019 below the DOMINANT
threshold and within a fifth of a standard error of it. **One scalar, the event
count residual against the human conditional, carries 48 percent of the contract
detector's entire margin over chance.** The cancelling halves argument was
simply wrong, because a detector conditions on the commanded duration and does
not have to see the marginal at all.

**The mechanism is forced by algebra, not proposed.** Duration is obeyed almost
perfectly on both sides, real median 1.0000 and generated median 1.0000 of
commanded. Total time is therefore fixed, so event count and mean inter event
interval are two names for one quantity, and the slopes must sum to one:

```
  human  count slope 0.3974  +  mean dt slope 0.6023  =  0.9997
  model  count slope 0.7056  +  mean dt slope 0.2933  =  0.9989
```

**A human asked for a longer movement mostly slows the clock between events. The
model mostly adds more events.** A human doing a movement four times longer
stretches the gaps about 2.3x and adds only about 1.5x the events. The model
stretches the gaps 1.4x and adds 2.4x the events. The model is behaving much
closer to a fixed rate sampler than a person does, and it has learned about half
of the human modulation, landing at 0.29 on a path from 0.00 for a pure fixed
rate to 0.60 for a human.

This is a restatement of the count finding, not an independent one, and it is
labelled as such. Its value is that it names the defect in the channel that
would have to be changed, the dt head's response to commanded duration, rather
than in a downstream count.

**Where this leaves the thread.** For the first time there is a defect that is
named, measured, mechanistically located in one head, and priced against the
contract at 48 percent of the margin as a LOWER BOUND from a single scalar. The
timing feature that won the history decomposition at 52.5 percent and this are
the same defect seen from two directions, which is the first time in this file
two independent measurements have converged on one cause rather than one
measurement refuting the previous story.

It is a lower bound and it is one scalar, so it is not the whole 0.6612, and
nothing here says the remaining margin is small. The registered next step is to
price what fixing it buys, not to assume.

## What removing the event count defect would buy, 2026-08-06

Registered before the script exists.

A defect that is detectable is not automatically a defect worth fixing. The
question is what the contract score would be if the model's event count obeyed
the human conditional, and that can be answered without retraining.

**The probe.** Generate the standard 1500 with the served path, unchanged, and
score it. Then resample the generated set so that its event count residual
against the human conditional matches the human residual distribution, and score
the resampled set. The drop is the price.

**This is a DIAGNOSTIC and it is DISQUALIFIED as a generation method.** Choosing
which generated movements to keep is selection, and the mandate rules selection
out for anything that produces output. It is being used here only to estimate a
counterfactual score, which is a legitimate and standard use, and it must never
be quoted as a result the model achieved. Written down so a later reader cannot
mistake the number.

**The placebo, which is what makes it readable.** Resampling reduces the sample
and the random forest out of bag AUC is not invariant to sample size, so a drop
could be an artefact of n alone. Every matched draw is therefore paired with a
RANDOM draw of the identical size from the same generated pool. The reportable
quantity is matched minus random, not matched minus baseline. Twenty draws of
each, and the standard error across draws is reported.

**Branches.**

- **WORTH IT** if matched minus random is at most minus 0.04. That is a quarter
  of the margin over chance recovered from one defect, and the training change
  gets built.
- **NOT WORTH IT** if matched minus random is greater than minus 0.015, or is
  positive. Event count is detectable but removing it does not move the
  contract, the lower bound was loose, and the thread moves to the roughness
  half of the history result.
- **PARTIAL** in between.

**What I expect, recorded so it can be wrong.** Somewhere in PARTIAL, around
minus 0.02 to minus 0.03. The reasoning is that the eighteen contract features
do not include event count directly, so the defect reaches the detector only
through its effect on rendered smoothness and duration statistics, and that
transmission should cost some of the 0.078 the raw scalar carries. I was wrong
in the optimistic direction on the last registered prediction, so this one is
deliberately not adjusted to compensate for that.

GPU, one generation of 1500 at temperature 1.0. Watchdog required.

### THE RESULT. The probe returns NOT WORTH IT, and the probe itself is the wrong instrument

**First, a methodological failure in my own first version, caught before it was
reported.** The registered design said resample the generated set to match the
human event count residual, and pair every matched draw with a same size RANDOM
draw so that the sample size dependence of the random forest out of bag AUC
cancels. I implemented both arms by drawing WITH REPLACEMENT. That is invalid.
A duplicated row breaks the out of bag estimate outright, because a copy of an
in bag movement lands in the out of bag set and is classified for free. It
produced this:

```
  baseline contract AUC 0.7155 on 1500
  matched 0.8537    random 0.8365    diff +0.0172  se 0.0027   6.29 sigma
```

Both arms score far ABOVE the baseline they are subsets of, which is impossible
and is the tell. The matched arm concentrates on fewer unique movements, so it
duplicates more, so it scores higher, and the entire 6.29 sigma signal was the
difference in duplicate rate. **The registered placebo controlled for sample
size and not for duplicate rate.** Writing a placebo is not the same as writing
the right placebo, and the number it produced was large, significant and
completely fake.

Redone with thinning, no replacement in either arm, acceptance proportional to
target share over generated share:

```
  baseline contract AUC 0.7119 on 1500
  retained 380 of 1500 per draw, 20 draws

  matched  0.6916  se 0.0034
  random   0.6721  se 0.0063
  MATCHED MINUS RANDOM  +0.0195  se 0.0069   +2.82 sigma

  VERDICT  NOT WORTH IT, per the registered branch
```

Both arms now sit below the 1500 baseline as a subset must. The registered
branch fires: anything above minus 0.015 is NOT WORTH IT, and this is plus
0.0195.

**But the SIGN is positive, and that is not what "the defect does not matter"
looks like.** Removing a defect should leave detection unchanged or harder.
Making the model's event count look human made it EASIER to detect, at 2.8
sigma. I checked the one confound I could name, that matching on a residual
which correlates with duration at +0.3776 in the model and 0.0000 in humans must
reshape the duration mix, and that duration is itself a contract feature.

**The check refutes my own explanation.** Matching moved the generated
`movement_duration` median from 0.3910 to 0.4210 against the contract human
median of 0.4130, so the log gap IMPROVED from 0.0547 to 0.0192. Duration got
better while the score got worse. I am not proposing a fourth mechanism.

**What this run actually establishes, stated at the strength it supports.** A
selection based counterfactual cannot isolate this defect. Selecting movements
whose event count looks human also selects on everything correlated with event
count, and the model's event count is correlated with a great deal by
construction, because the defect IS a duration dependent slope error. The
positive sign is the instrument reporting that it did not do what was asked of
it. The honest conclusion is NOT "fixing event count would not help" but
**"this probe cannot answer the question, and provides no evidence either way."**
The registered NOT WORTH IT verdict is recorded because it was registered, and
it is explicitly NOT relied on.

**The right instrument, and it is strictly better than the one just used.**
Intervene at GENERATION rather than by selection. Add a duration conditional
tilt to the dt head's logits at sampling time, one parameter, using only `cond`
which is available at serving:

```
  dt logit bias  =  lambda * (log D - mean log D) * normalised class index
```

Positive lambda pushes toward longer inter event times for long movements and
shorter ones for short movements, which is exactly the direction that moves the
mean dt slope from the model's 0.2933 toward the human 0.6023. Calibrate lambda
by matching the slope, then generate fresh and score.

That produces ONE trajectory per spec with no selection, so unlike the probe
above it is mandate compliant as a method and not merely as a diagnostic. It
changes the generating process rather than filtering its output, so it isolates
the defect instead of dragging every correlate along with it. And it costs one
sampling time argument, no retraining.

It is also not on the NOT AUTHORISED list, which covers phase conditioning, the
spectral loss term, and the FiLM rewrite of `th_head` and `dt_head`. A logit
bias at sampling is none of those.

## The generation time timing tilt, 2026-08-07

Registered before the sweep runs. The `sample` change itself is already in
`models/event_ar.py` and is a no op at lambda zero.

One parameter, added to the timing head's logits at sampling:

```
  dt logit bias  =  lambda * (log D - mean log D) * (class index / max - 0.5)
```

`lambda` positive lengthens inter event times for long movements and shortens
them for short ones, moving the mean dt slope from the model's 0.2933 toward the
human's 0.6023. It reads only `cond`, emits one trajectory per spec, and selects
nothing, so unlike `w4_evprice` it is a candidate serving change and not merely
a diagnostic.

**Calibration then test, in that order, and the calibration does not look at the
contract score.** Sweep lambda, measure the mean dt slope on log duration for
each, pick the lambda whose slope is closest to the human 0.6023, and only then
read the contract AUC. Choosing lambda by the contract score would be fitting
the detector, which this file has been careful never to do.

**The placebo, and it is what makes the result readable.** The same sweep is run
at NEGATIVE lambda of equal magnitude, which moves the slope further from human.
If the contract improves for both signs then the tilt is helping through
something other than the duration response, and the result is an artefact.
`w4_ar_eval`'s `th_beta_arm=reverse` is the same idea and it is why that arm was
readable.

**Branches.**

- **WORKS** if the calibrated arm improves the contract by at least 0.02 AND
  the reversed arm does not improve.
- **ARTEFACT** if both signs improve by at least 0.01.
- **NO EFFECT** if the calibrated arm moves the contract by less than 0.01
  either way. The event count defect is then real, measured, detectable in
  isolation at 0.5781, and simply not what the contract detector is reading, and
  the thread moves to the roughness half of `w4_histfeat`.

**What I expect, recorded so it can be wrong.** NO EFFECT or a small
improvement, under 0.02. I have now been wrong on two of two registered
predictions, optimistic once and pessimistic once, so this one is written from
the reasoning and not from a running correction: the eighteen contract features
contain no event count term, the defect reaches the detector only through
rendered timing statistics, and `w4_evprice` gave no evidence that removing it
helps.

Baseline to beat is the 0.7119 measured on this checkpoint at temperature 1.0 in
`w4_evprice`, NOT the 0.6612 quoted elsewhere in this file, which is a different
configuration. That discrepancy is noted and is not resolved here.

## Where the detector actually looks, 2026-08-07

Registered before running `research/w4_featmap.py`.

**Why this is being run now, and why it should have been run much earlier.** Every
measurement in this session has located the defect in the model's own
coordinates, event counts, inter event times, speed changes. None of them asked
the reciprocal question: of the eighteen numbers the contract detector actually
computes, which ones carry the separation. That gap is the reason `w4_evprice`
could not price the event count defect. Reading `features.py` makes the problem
concrete. Every feature is computed on a trajectory resampled to 125 Hz, so the
detector never sees an event, only the interpolated path. Event count reaches it
through nothing more direct than what a different number of events does to
resampled kinematics, and at a generated mean inter event time near 9 ms the
resampling grid at 8 ms is finer than the events themselves. The model's event
budget is largely invisible to the thing being fooled.

**The measurement.** Three read only views of the same random forest recipe,
against the same `data/human_val_features_grpo.npy` the contract uses, on the
1500 movement generated matrix already cached in `research/w4_evprice_cache.npz`.
No generation, no GPU, and `research/autoloop/scoring.py` is imported and never
edited.

1. Single feature AUC, eighteen fits of one column each. What that feature knows
   on its own, including everything it shares with the others.
2. Leave one out AUC, eighteen fits of seventeen columns. The drop is what only
   that feature knows.
3. Leave one group out, on five semantic groups fixed here in advance: SPEED
   (columns 0 to 3), ROUGHNESS (4 to 8, the acceleration and jerk moments),
   GEOMETRY (9 to 13), TIMING (14 and 15), ANGULAR (16 and 17).

**The two views bound the answer and neither is trustworthy alone.** Correlated
features make leave one out understate a feature, because the forest simply
routes around it through its partner, and make single feature AUC overstate it,
because shared credit is counted once per feature. A defect is only established
where both views agree. Reporting one of them alone would be the same mistake as
reporting the matched arm of `w4_evprice` as an achieved score.

**Branches, fixed now.**

- **ROUGHNESS DOMINANT** if the accleration and jerk group is the largest leave
  one group out drop and that drop is at least 0.05.
- **DIFFUSE** if no group drop reaches 0.05, meaning the separation is spread and
  no single kinematic family is the handle.
- **ELSEWHERE** if some group other than roughness is the largest.

**What I expect, recorded so it can be wrong.** ROUGHNESS DOMINANT. The reasoning
is that `w4_histfeat` measured the generated last speed change at 2.86 against a
human 1.97, a 45 percent excess in exactly the quantity the acceleration and jerk
columns integrate, and that this defect, unlike event count, survives resampling
intact. The competing possibility I can name is that SPEED wins instead on
levels rather than roughness, since the generated last speed is 9.29 against a
human 7.25.

**What this cannot settle.** A group that carries the separation is not
automatically a group that can be fixed, and a drop measured by deleting a column
from the detector is not the same quantity as the gain from correcting the
generator. That confusion is what made the `w4_evprice` probe uninterpretable.
This measurement chooses the target. It does not price it.

### THE RESULT. DIFFUSE, and the prediction was wrong in the way that matters

ROUGHNESS DOMINANT was wrong. The verdict is DIFFUSE and it is the most
informative thing measured in this session.

Full eighteen column AUC 0.7149, se 0.0016, five forest seeds. Against that:

| view | best | which |
|---|---|---|
| single feature alone | 0.5801 | angular_velocity_mean |
| single feature, drop on removal | +0.0101 | angular_velocity_mean |
| group alone | 0.6266 | GEOMETRY |
| group, drop on removal | +0.0186 | ANGULAR |

Nothing carries it. The largest loss from deleting an entire kinematic family
from the detector is 0.0186 out of the 0.2149 the detector holds above chance,
under nine percent. ROUGHNESS, the family I predicted, gives up 0.0116.
GEOMETRY gives up nothing at all, minus 0.0003, inside its own standard error.

**Redundancy is not the explanation, and ruling that out is what makes this
result mean something.** A small leave one out drop is the ordinary signature of
correlated features: the forest loses nothing because a partner column carries
the same information. If that were the whole story here, some single feature or
some group would reproduce most of the full number on its own. None does. The
best single feature reaches 0.5801 and the best group 0.6266, against a full
0.7149. So the columns are neither individually sufficient nor individually
necessary. Each carries a little, they overlap partially, and the detector's
power is in the accumulation.

**Why this retroactively explains four failures.** Every measurement in this
session, and the mechanism proposals before them, looked for a single dominant
cause and then tried to price it. `w4_evprice` could not price the event count
defect. Mode covering breadth, accumulating drift and rare ruinous draws were
each refuted. The common assumption underneath all of them was that a defect
large enough to name is a defect large enough to move the contract. This result
says that assumption is false here, and false structurally rather than by
accident: the detector reads no marginal strongly enough for correcting one
marginal to show up.

**What it forecasts about the work already queued.** The duration conditional
timing tilt corrects one marginal. The moment matched tilt now designed against
the token level table corrects three. On this evidence both should be expected to
buy very little, and the registered NO EFFECT prediction for `w4_dttilt` looks
better founded now than when it was written. That is a forecast made before
either was run, and it is recorded here so it can be checked rather than
reconstructed afterwards.

**What it does not establish, and the next measurement follows from exactly
this.** A diffuse result is consistent with two very different worlds. In the
first, the marginals really are all slightly wrong and their small errors add up,
in which case fixing marginals is slow but does work. In the second, the
marginals are nearly right and what separates is the dependence between them, the
joint shape rather than any axis, in which case no amount of marginal correction
helps at all and the entire moment matching program is capped well above chance.
This measurement cannot tell those apart. The next one is built to.

## Marginals or dependence, 2026-08-07

Registered before running `research/w4_joint.py`.

**The question, and why it decides what the rest of this work should be.** The
diffuse result leaves two worlds. Either the eighteen marginals are each slightly
wrong and their errors accumulate, in which case correcting marginals one at a
time is slow but works and the moment matching program is the right program. Or
the marginals are close to right and what separates is the dependence between
them, in which case correcting marginals cannot reach chance no matter how many
are corrected, and the effort has to go somewhere else entirely. Everything
queued behind this depends on the answer, so it is worth doing before any of it.

**The measurement, and it is exact rather than estimated.** For each of the
eighteen columns independently, replace every generated value by the human value
at the same rank. That imposes the human marginal on the generated column
exactly, and because it is a monotone map applied within a column it leaves the
generated rank dependence between columns untouched. Score the result. What
survives is, by construction, the separation that no marginal correction can ever
remove. This is the price probe `w4_evprice` tried and failed to be, and it
avoids that failure because it resamples nothing: same rows, same count, no
duplicates, no selection, so the out of bag estimate is not corrupted.

**The split, which is the part that could silently invent a result.** Matching
generated marginals to the same human sample the score is computed against would
fit that sample's particular order statistics and manufacture an improvement.
The human set is therefore cut in two. Marginals are matched to the first half
and every score is computed against the second half, which the matching never
sees. Both arms are balanced to the same size.

**Per group prices, on the generator side this time.** The same map is applied to
one group at a time, using the group definitions already fixed in
`w4_featmap`. Unlike a leave one out drop from the detector side, this asks the
question actually of interest: what would perfectly correcting this family of
kinematics buy. That is the quantity `w4_evprice` was reaching for.

**Branches, fixed now.**

- **DEPENDENCE DOMINANT** if full marginal matching leaves AUC at or above 0.62,
  that is if it recovers less than half the 0.2149 the detector holds above
  chance. Marginal correction is then capped and the program changes.
- **MARGINALS DOMINANT** if full marginal matching brings AUC to 0.55 or below.
  The moment matching program is then the right one and should be pushed hard.
- **MIXED** in between, with the split reported.

**What I expect, recorded so it can be wrong.** DEPENDENCE DOMINANT. The
arithmetic behind that: the strongest single feature separates at 0.5801 and most
sit near 0.51, which is a discriminability around 0.1 standard deviations each.
Eighteen such signals combined independently would land near 0.62, and the
detector reaches 0.7149. The excess over what independent accumulation can
explain has to come from structure between the columns. I was wrong predicting
ROUGHNESS DOMINANT one measurement ago, and the correction I took from being
wrong is exactly this reasoning, so it deserves less confidence than the number
it produces.

### THE RESULT. DEPENDENCE DOMINANT, and it caps the whole marginal program

Perfect marginal correction on all eighteen features buys 0.0394 of the 0.2148
the detector holds above chance. Eighteen percent. The remaining 0.6754 survives
having every marginal replaced by the human one exactly.

| arm | AUC | se |
|---|---|---|
| generated, untouched | 0.7148 | 0.0025 |
| all 18 marginals matched to human | 0.6754 | 0.0015 |

**Two artefact hypotheses were raised against this and both were refuted, which
is the only reason the number is being reported.** The first: linear
interpolation between human order statistics produces values no human row can
hold, and `num_direction_changes` is an integer count with 93.5 percent ties.
Fixed by landing on the nearest actual order statistic. The number moved from
0.6743 to 0.6743. The second: `argsort` gives tied rows distinct arbitrary ranks,
which would scatter identical generated values to different human values and
decorrelate that column from the rest, inflating exactly this answer. Fixed with
average ranks so ties stay tied. The number moved to 0.6754. Three
implementations, one conclusion, and the two mechanisms I proposed to explain it
away were both wrong. The fixes were kept because they are correct on principle,
not because they changed anything.

**Correcting one family at a time makes the sample MORE detectable, and this
reproduced across all three implementations.**

| group matched | AUC | bought | mean R squared on the other 17 |
|---|---|---|---|
| SPEED | 0.7224 | -0.0076 | 0.947 |
| ROUGHNESS | 0.7274 | -0.0126 | 0.683 |
| GEOMETRY | 0.7675 | -0.0527 | 0.898 |
| TIMING | 0.7148 | +0.0000 | 0.624 |
| ANGULAR | 0.7556 | -0.0408 | 0.930 |

The eighteen features are close to mutually determined. On normal scores, each is
predictable from the other seventeen at R squared up to 0.990, with
`std_acceleration` 0.990, `curvature_mean` 0.988 and `max_velocity` 0.985.
Replacing one family's values while leaving its near partners generated produces
rows that violate relations holding in humans and in the generator alike, and the
forest finds those hybrids easily. The account is consistent in direction for
GEOMETRY and ANGULAR, the two high R squared groups with the largest penalties,
and for TIMING, the lowest at 0.624 with no penalty at all. It does NOT explain
SPEED, which has the highest R squared of any group at 0.947 and one of the
smallest penalties. So the account is supported and not established, and the
group column should not be read as a price for any generator side fix.

**The finding underneath the finding.** Human and generated mutual predictability
agree almost exactly, column by column, mostly inside 0.01. The two exceptions
are `mean_jerk`, 0.109 human against 0.055 generated, and `path_efficiency`,
0.764 against 0.720. So the model has already learned the gross dependence
structure between kinematic features. Whatever separates at 0.6754 is not the
marginals, which have now been priced at eighteen percent, and does not look like
the linear dependence either. That points at higher order joint structure, and
the next measurement tests it directly instead of inferring it from here.

**What this does to the queued work, and it is severe.** The forecast recorded in
the previous section is now quantified. Marginal correction of every kind, the
duration conditional timing tilt, the three parameter moment matched tilt, and any
successor built the same way, is capped at 0.6754 even if it works perfectly on
every one of the eighteen. It cannot reach 0.50. The moment matching program is
not merely slow, it is bounded away from the target, and that is a structural
statement rather than an empirical one about this checkpoint.

## Linear dependence or higher order, 2026-08-07

Registered before running `research/w4_copula.py`.

**The question.** Marginal correction is now priced at eighteen percent and
0.6754 survives it. That surviving separation is dependence between the eighteen
features. Dependence splits into a part carried by the correlation matrix, which
a generator could plausibly be pushed to fix, and a part carried by everything
else, tail dependence, conditional heteroskedasticity, higher moments, which it
could not be pushed to fix by any comparable means. Which one holds the 0.1754
decides whether there is a tractable target left in the feature space at all.

**The measurement.** Take the generated matrix to normal scores column by column,
whiten it with its own correlation matrix, recolour it with the human
correlation matrix estimated on the held out fitting half, then map back to the
human marginals by rank. The result carries the human marginals exactly and the
human normal score correlation matrix exactly, and keeps the generated higher
order structure. Whatever still separates is higher order by construction. Same
held out split as `w4_joint`, marginals and correlations both fitted on the half
that is never scored against.

**The control that has to be there.** The recolouring is a linear map on normal
scores and a linear map applied to non Gaussian data changes higher order shape
as a side effect, so a drop could be the map smoothing the data rather than the
correlation being corrected. The placebo recolours with a correlation matrix
estimated on a SECOND independent generated sample instead of the human one. That
applies a map of the same kind and the same magnitude while correcting nothing.
Any drop the placebo also produces is the map, not the correction.

**Branches, fixed now.**

- **CORRELATION CARRIES IT** if the human recolouring lands at or below 0.58 and
  the placebo does not. There is then a concrete target: the model's feature
  correlation matrix.
- **HIGHER ORDER** if the human recolouring stays at or above 0.64. The feature
  space holds no tractable handle and the work has to move to the generative
  process itself.
- **MIXED** in between.

**What I expect, recorded so it can be wrong.** HIGHER ORDER. The reason is
already in hand and is why this is worth running rather than assuming: human and
generated mutual predictability agree column by column to inside 0.01 on sixteen
of eighteen features. If the correlation structure were the defect it should have
shown up there. I predicted ROUGHNESS DOMINANT and was wrong, then predicted
DEPENDENCE DOMINANT and was right, and this one rests on an actual measurement
rather than on an argument, so it should be the most reliable of the three.

### THE RESULT. CORRELATION CARRIES IT, the prediction was wrong, and this is the largest actionable quantity in this file

HIGHER ORDER was wrong.

| arm | AUC | se |
|---|---|---|
| generated, untouched | 0.7148 | 0.0025 |
| human marginals only | 0.6754 | 0.0015 |
| human marginals AND human correlation matrix | 0.5715 | 0.0028 |
| placebo, recoloured to a second generated half | 0.6375 | 0.0067 |

Correcting the eighteen by eighteen correlation matrix takes 0.6754 to 0.5715.
Two thirds of everything the detector holds above chance is removed by fixing
marginals and correlations together, and 0.5715 is within striking distance of
the 0.5118 reconstruction floor that real human tokens reach through the same
serving decoder.

**The placebo did move, so the honest number is smaller than the headline.** The
registered worry was that a linear recolouring on normal scores changes higher
order shape as a side effect. It does. Recolouring to a second generated half
corrects nothing and still buys 0.0379. So the part attributable to having the
correct correlation matrix is the difference between the arms, 0.0660, and the
absolute 0.5715 is an upper bound on what a generator side fix would reach. The
defensible statement is that a generator with correct marginals and correct
correlations lands somewhere between 0.5715 and 0.6375.

**Why the reasoning behind the prediction failed, which matters more than the
prediction.** I argued HIGHER ORDER from human and generated mutual
predictability agreeing to inside 0.01 on sixteen of eighteen features. That
inference was invalid. R squared measures how much of a feature's variance the
others explain in total, and two very different correlation matrices can explain
the same total while distributing it completely differently among the
predictors. The summary statistic I reasoned from was blind to the thing being
asked about. The correlation matrices are in fact far apart, mean absolute off
diagonal difference 0.0854 against a placebo noise level of 0.0306, and a maximum
of 0.2785.

**Where the difference sits, and it is not diffuse this time.** Ranked by
absolute discrepancy, thirteen of the top fourteen pairs involve the same
cluster: `num_direction_changes`, `angular_velocity_mean` and `_std`,
`max_deviation`, `curvature_mean` and `_std`, `path_efficiency`, and their
coupling to `movement_duration`.

| pair | human | generated |
|---|---|---|
| num_direction_changes x angular_velocity_std | 0.392 | 0.075 |
| movement_duration x angular_velocity_std | 0.293 | -0.003 |
| max_deviation x curvature_mean | 0.289 | 0.024 |
| curvature_std x num_direction_changes | 0.550 | 0.313 |
| curvature_std x movement_duration | 0.497 | 0.274 |
| mean_velocity x movement_duration | -0.145 | +0.130 |

In humans these move together. A movement that took longer also wandered more,
changed direction more often, curved more and left the straight line further.
That is one coherent property of a movement showing up in six different
measurements of it. In the generated sample those couplings are largely absent,
several of them sitting at zero, and one has the wrong sign: humans take longer
on slower movements, the model takes longer on faster ones, which is the event
count defect of the previous section seen from the feature side.

This is not a global attenuation. Overall mean absolute correlation is 0.293
human against 0.272 generated, and only 53.6 percent of the 153 pairs are under
correlated. The defect is concentrated in one cluster, and the cluster has an
obvious common sense name.

## Missing shared factor or independent noise, 2026-08-07

Registered before running the residual spread check.

**Two mechanisms produce a decoupled cluster and they need opposite fixes.** In
the first, the model has too little between trajectory variation in these
features because per step sampling noise averages out over a couple of hundred
events, so conditional on the commanded distance and duration each global
feature is nearly pinned and there is little left to correlate. In the second the
model has the right amount of variation but generates it independently per
feature, so nothing shared drives them together. Both show up as low correlation
and they are not the same defect. The first wants a shared factor added. The
second wants independent noise removed and shared noise put in its place.

**The measurement.** Regress each feature on log distance and log duration in
each sample separately and compare residual standard deviations. Distance is
reconstructed for both samples from the features themselves, as mean_velocity
times movement_duration times path_efficiency, so the same quantity is available
on the human side where no conditioning vector is stored.

- **MISSING SHARED FACTOR** if the wobble cluster's generated residual spread is
  clearly below human.
- **INDEPENDENT NOISE** if it matches or exceeds human.

**What I expect, recorded so it can be wrong.** MISSING SHARED FACTOR, from the
averaging argument. My last two mechanism predictions were wrong, and in the
second case the reasoning was invalid rather than merely unlucky, so this is
written knowing it is the weakest kind of claim in this file.

### THE RESULT. INDEPENDENT NOISE, the third mechanism prediction wrong in a row

MISSING SHARED FACTOR was wrong. The wobble cluster's generated residual spread
is ABOVE human, not below, so nothing is missing that a shared factor would
supply on its own. Robust IQR ratios of the generated residual spread to the
human, on log scaled values for the heavy tailed columns:

| cluster | mean IQR ratio, generated over human |
|---|---|
| wobble | 1.25 |
| roughness | 1.21 |
| speed | 1.03 |

The model already varies these features from movement to movement slightly more
than a human does. What it does not do is vary them TOGETHER. That is a different
defect from the one predicted and it wants a different fix: not more variation,
but the same amount of variation made shared instead of independent.

**A first pass of this used standard deviations and was badly misleading.** On
raw standard deviations the roughness family looked catastrophically under
dispersed, ratios of 0.08 to 0.32, which would have supported the opposite
conclusion. Those features are extremely heavy tailed, `mean_jerk` has a human
standard deviation of 1.4e7, so the statistic was tracking a handful of extreme
values rather than the spread. The IQR on log scaled values is the honest
instrument and it reverses the sign of the conclusion.

**Three of the eighteen rows in that table are circular and must be discarded.**
Distance was reconstructed as mean_velocity times movement_duration times
path_efficiency, so the regressors are functions of features 0, 9 and 14 and
their residuals are not meaningful. Recomputing the wobble mean without
`path_efficiency` gives 1.25 rather than 1.23, so the verdict does not depend on
the contaminated rows, but the `movement_duration` ratio of 0.54 quoted from the
standard deviation pass is pure circularity and means nothing.

**Three mechanism predictions in a row have now been wrong**, mode covering
breadth, missing shared factor, and before them the roughness and higher order
calls. The decomposition measurements in this session have held up under
adversarial checking every time; the mechanism guesses attached to them have not
survived once. Predictions about which mechanism is operating should be treated
as the least reliable statements in this file, including the ones written above
in this section.

### THE COLLAPSE FLAG IS ALREADY TRUE, and this file has never once mentioned it

`scoring.score_features` returns an anti Goodhart battery alongside the AUC, and
on this checkpoint it fires.

| feature | dispersion ratio, generated over human | band |
|---|---|---|
| mean_jerk | 0.0682 | 0.2 to 5.0 |
| mean_acceleration | 0.0963 | |
| max_acceleration | 0.1330 | |
| std_acceleration | 0.1879 | |
| max_velocity | 0.1885 | |

Five of eighteen features are outside the band, all in the same direction, all
under dispersed. `grep` finds zero mentions of `collapse_flag` anywhere in the
9200 lines of this file. Every AUC quoted in this record was carried in a
dictionary that also carried this flag, and it was never read.

**Why this matters more than it looks.** The battery exists because of a
documented earlier incident, recorded in `scoring.py` itself: a burst with a
good looking loss and two features collapsed to near constants. The rule L
attached to it is that an AUC near 0.5 with a collapsed feature is not a win, it
is a tell waiting to be found. On this checkpoint the target of 0.50 would
therefore not be an acceptable result even if it were reached tomorrow. Whatever
comes next has to move these ratios toward one at the same time, and any future
claim of success has to quote the flag next to the AUC.

Note that this is a raw standard deviation ratio, so it is sensitive to the same
heavy tails discussed above, and the honest reading is that the generated
extremes are much less extreme than the human ones rather than that the bulk of
the distribution has collapsed. That is still a tell, and it is still failing the
band L set.

## A per trajectory latent at generation time, 2026-08-07

Registered before running `research/w4_latent.py`.

**What it is.** Draw one scalar z per trajectory from a zero mean Gaussian,
before the first event, and hold it fixed for the whole trajectory. At every step
it adds a bias to the direction head's logits, linear in normalised turn
magnitude. Positive z makes that movement wander for its whole length, negative z
makes it run straight. Generation becomes a latent variable model, the marginal
over trajectories being the integral of p(traj given z) against the prior, and it
still emits exactly one trajectory per request with nothing generated twice and
nothing selected. It requires no retraining and reads nothing unavailable at
serving time.

**Why this shape and not another.** `w4_copula` measured the six wobble features
coupling at 0.4985 in humans and 0.3333 here, and the pattern inside that cluster
is specific: pairs within a sub family are already right, curvature_mean against
curvature_std is 0.984 human and 0.982 generated, angular_velocity_mean against
_std is 0.917 and 0.924. What is missing is coupling BETWEEN the sub families,
`num_direction_changes` against `angular_velocity_std` at 0.392 human and 0.075
generated, `max_deviation` against `curvature_mean` at 0.289 and 0.024. Three
internally coherent sub clusters that do not talk to each other is what a model
with no persistent per trajectory state produces, and one factor held across the
trajectory is the minimal thing that makes them talk. The residual spread
measurement rules out the competing fix: the model is not short of variation in
these features, it has about 1.25 times the human amount, so the goal is to move
variation from independent to shared and not to add more.

**Calibration before scoring, and the calibration never sees the contract.**
Sweep sigma, measure the mean pairwise correlation among the six wobble features
for each, pick the sigma closest to the human 0.4985, and only then read the AUC.

**The placebo, and it is the sharp one.** Redraw z at every step instead of once
per trajectory, at the same sigma. That injects exactly the same amount of extra
turn variability with exactly zero shared factor. Any gain that the per step arm
also produces is not the latent, it is just noise on the direction head.

**Branches, fixed now.**

- **WORKS** if the calibrated per trajectory arm improves the contract by at
  least 0.02 and the per step placebo does not improve by 0.01.
- **ARTEFACT** if both arms improve by at least 0.01.
- **NO EFFECT** if the calibrated arm moves the contract by less than 0.01.

**What I expect, recorded so it can be wrong.** A real but partial improvement,
0.01 to 0.04. The reasoning: `w4_copula` priced the entire correlation matrix at
0.0660 net of its placebo, the wobble cluster is a large part but not all of that
matrix, and this intervention addresses one factor within the cluster rather than
every entry of it. Against that, my mechanism predictions are nought for three in
this session, and there is a specific way this can fail that the measurement
cannot rule out in advance: raising the coupling will also move the marginals of
all six features, and marginals were priced at only eighteen percent, so the arm
could buy correlation and give back marginal at the same time. The dispersion
ratios are reported next to the AUC for that reason.

**The collapse flag is reported for every arm.** It is already true at baseline on
five features and an arm that improves the AUC while worsening it is not progress.

### AN ASIDE THAT CHANGED THE DESIGN. The missing factor is hesitancy, and it is not confined to direction

A scalar latent can only add a rank one, positive semidefinite term to the
feature covariance, because its contribution is the outer product of the
sensitivity vector of the features to z. So the achievable share of the gap is
readable in advance from the eigenvalue spectrum of the difference between the
human and generated correlation matrices, with no generation at all.

Wobble block, six by six. Eigenvalues 0.863, 0.012, -0.001, -0.076, -0.171,
-0.627. The leading direction is positive and holds 49.3 percent of the absolute
spectrum, so a shared factor really is the missing thing. But the large negative
eigenvalue is a direction where the model has TOO MUCH correlation, and adding a
positive semidefinite term cannot remove it. A single latent therefore tops out
near half of even this one block.

Full matrix, eighteen by eighteen. Leading share 30.3 percent, top two 44.4
percent. One latent cannot close the correlation gap, and this is a hard ceiling
on the intervention now being built, known before spending the GPU time.

**The leading factor of the full gap, and it has an obvious name.** Loadings
above 0.20:

| feature | loading |
|---|---|
| num_direction_changes | +0.363 |
| movement_duration | +0.356 |
| path_efficiency | -0.318 |
| angular_velocity_std | +0.310 |
| curvature_std | +0.309 |
| max_deviation | +0.307 |
| velocity_skewness | +0.296 |
| curvature_mean | +0.288 |
| mean_velocity | -0.221 |

A movement that takes longer, changes direction more often, curves more, strays
further from the straight line, ends up less efficient, and is slower. That is
one behavioural property of a movement, hesitancy, read off nine different ways.
Humans have it. This model does not, and it does not have it because nothing in
the architecture persists across a trajectory to carry it: the conditioning
vector holds distance, duration and angle, all externally commanded, and there is
no free per movement variable at all. Every movement is improvised from the same
conditional with no plan, so the model produces the average of all plans, whose
cross feature structure is the average structure and not any real one.

**This invalidates the design registered above while confirming its motivation.**
The `w4_latent` tilt drives the direction head only, and the factor loads about
as heavily on `movement_duration` and `mean_velocity` as on the direction
features. A direction only latent addresses perhaps half the loading vector. The
registered run continues as written, because it is registered and because its
placebo is what establishes whether a per trajectory factor does anything at all,
but the number it produces should be read as a lower bound on the idea rather
than a test of it. The corrected design drives all three heads from one z with
the loadings taken from this eigenvector and a single free scale, and the hooks
for two of the three, `dt_tilt` and `th_tilt`, already exist.

## The duration prior the w4 diagnostics ran under was not the served one, 2026-08-07

Registration. Written before `research/w4_served.py` existed.

`research/w4_evprice.py` sets `EVENT_CHOICE_TEMP` and nothing else. Everything
downstream of it, which is every w4 diagnostic in this file, reads the cache it
wrote. So all of them inherited the LIBRARY DEFAULT duration prior from
`experiments/_common.py`, `std_mult=0.7` with a Gaussian per distance bin.

That is not what this repo serves. `generate.py`, `README.md`,
`research/w1_oneshot_score.py`, `research/w3_p1_eval.py`, `w3_p2_eval.py`,
`w3_jog_on_resid.py`, `w3_duration_response.py` and `w4_audit.py` all set
`EVENT_DUR_STD=1.0 DUR_EMPIRICAL=1`. The 0.7 is a bare default that applies only
when nobody sets it, and w4 was the code that did not set it.

`research/w4_durprior.py`, CPU only, no generation, no contract read, on the
same 1500 specs so the between bin component of the spread is held fixed:

```
  setting                        mean       sd      iqr  sd/human  iqr/human
  human val (realised)        -0.9882   0.8683   1.2657     1.000      1.000
  w4 cache (realised)         -0.9719   0.6675   0.8976     0.769      0.709
  std 0.7 gaussian            -1.0010   0.6783   0.9333     0.781      0.737  <- w4 default
  std 1.0 gaussian            -1.0084   0.8677   1.1785     0.999      0.931
  std 1.0 empirical           -0.9602   0.8654   1.2123     0.997      0.958  <- served
```

The commanded spread at the w4 default, 0.6783, reproduces the cache's realised
spread, 0.6675, which is the arithmetic confirming the cache ran at 0.7 Gaussian.
The served setting lands within 0.3 percent of the human standard deviation.
Note also that `std_mult` above 1.0 does nothing once `DUR_EMPIRICAL=1`, because
it then scales only the 0.02 jitter; the rows at 1.25 and 1.5 are identical.

This matters beyond one marginal. The trunk's conditioning vector is
`[log_dist, log_dur, cos, sin]`, and the earlier duration response test in this
file measured that the model OBEYS the commanded duration at 96 to 97 percent.
A duration prior that is 23 percent too narrow therefore suppresses a shared per
trajectory factor at the only point in this architecture where one can enter.
`movement_duration` carries the second largest loading, +0.356, on the hesitancy
eigenvector that four registered measurements in this file converged on as the
core defect. So the defect and the misconfiguration point at the same place, and
the file cannot currently tell them apart.

What is being tested. Regenerate the cache on identical specs and seed under
`EVENT_DUR_STD=1.0 DUR_EMPIRICAL=1`, change nothing else, read the contract once.

Branches, fixed now.

- MATERIAL, the contract improves by 0.02 or more. The misconfiguration was a
  real contributor and every mechanism conclusion in the four w4 sections above
  has to be re derived on the corrected cache before it is trusted.
- MARGINAL ONLY, between 0.005 and 0.02. That is the order a duration marginal
  fix alone buys given `w4_joint` priced ALL eighteen marginals at 0.0394, so the
  hesitancy conclusion survives roughly intact.
- NO EFFECT, under 0.005.

Prediction. MATERIAL. Stated with the caveat that the three mechanism
predictions before this one were all wrong, and one of them for an invalid
reason, so the branch thresholds and not the prediction are what should be
trusted here.

Second thing to watch, recorded now so it cannot be claimed afterwards. The
collapse flag is ALREADY true on this checkpoint on the acceleration and jerk
features. Widening the duration prior adds slow trajectories, and slow
trajectories carry low jerk, so this change could deepen the collapse while
improving the AUC. An AUC gain bought alongside a worse collapse flag is not a
win under this repo's own anti Goodhart rule and will be reported as a loss.

### THE RESULT. MARGINAL ONLY on the AUC, but the dependence moved four times harder

`research/w4_served.py`, 1500 specs, identical seed, one trajectory each, the
only change being `EVENT_DUR_STD=1.0 DUR_EMPIRICAL=1`.

```
                                  w4 cache      served     delta
  CONTRACT AUC                      0.7119      0.6945   -0.0174
  collapsed features                     5           3        -2
  mean |corr gap|                   0.0937      0.0713   -0.0224
  leading gap eigenvalue            1.3992      1.1112   -0.2880
  log duration sd                   0.6675      0.8663   +0.1988
  human log duration sd                         0.8683
```

The registered branch is MARGINAL ONLY. Minus 0.0174 is below the 0.02 line I
fixed in advance, so the prediction of MATERIAL was wrong, and that is now four
mechanism predictions wrong in a row in this file. The branch is what counts and
the branch says the hesitancy conclusion survives.

Three things the AUC column does not say, and they matter more than the verdict.

First, the collapse flag IMPROVED, from five features to three. `max_velocity`
and `std_acceleration` came back inside the dispersion band. The caveat I
registered in advance, that slow trajectories carry low jerk and this could buy
AUC at the cost of a worse collapse, did not happen. This is a clean gain on
both axes at once, which is the first time anything in W4 has managed that.

Second, the duration marginal is now essentially exact: 0.8663 against the human
0.8683, having been 0.6675. That was a one line environment fix, not a model
change, and nothing was retrained.

Third, and this is the part worth carrying forward, the dependence structure
moved much harder than the AUC did. Mean absolute correlation gap fell 24
percent and the leading eigenvalue of the gap fell 21 percent, for an AUC move
of 8 percent of the excess over chance. The detector is not reading the
correlation gap efficiently, which is consistent with `w4_copula` needing the
WHOLE matrix corrected to reach 0.5715 while eighteen perfect marginals bought
only 0.0394.

The residual factor is now cleaner, not gone. Its six largest loadings:

```
    angular_velocity_mean       +0.356
    angular_velocity_std        +0.332
    num_direction_changes       +0.324
    max_deviation               +0.289
    curvature_mean              +0.279
    std_velocity                +0.270
```

`movement_duration` has dropped out of the top six entirely, having been the
second largest loading at +0.356 before. Fixing the duration prior removed
duration's arm of the hesitancy factor and left the wobble cluster behind,
undiluted. So the target is now narrower and better defined than it was this
morning: one shared factor over angular velocity, direction changes, curvature
and deviation, with no duration component left to explain it away.

What this costs the record. Every w4 number quoted above this section was
measured on the 0.7 Gaussian cache. The four mechanism verdicts (DIFFUSE,
DEPENDENCE DOMINANT, CORRELATION CARRIES IT, INDEPENDENT NOISE) were each
qualitative and each held by a wide margin, so a 0.017 shift does not overturn
any of them, but the specific NUMBERS in those sections are stale and should not
be quoted without the caveat. `w4_copula` and `w4_block` are being re run on the
served cache; those two carry the quantities anything downstream depends on.

The general lesson, and it is the one this repo has learned before. The
diagnostic ran against a default rather than against the served configuration,
and it took four registered measurements and most of a day before anyone
compared the two. Check the seam against what serving actually does, first, not
after building a theory on top of it.

### The block decomposition, and a correction to the 0.5715 recorded above

`research/w4_block.py` on both caches, full n, three forest seeds, so the two
columns are directly comparable to each other.

```
                          old cache    served cache
  baseline                   0.7141          0.6938
  marginals only             0.6778          0.6156
  marginals + FULL corr      0.6261          0.5896

  block corrected alone     old       served
  SPEED                  -0.0471      +0.0046
  ROUGHNESS              -0.0207      +0.0049
  GEOMETRY               -0.0488      -0.0478
  TIMING                 +0.0013      -0.0013
  ANGULAR                +0.0071      +0.0078
  WOBBLE                 -0.0371      -0.0421
```

CORRECTION, and it goes against the direction I would prefer. The 0.5715 quoted
in "CORRELATION CARRIES IT" above was measured on a HALVED generated sample, 750
rows against 1000 human, because `w4_copula` splits the generated set to build
an independent placebo matrix. The random forest out of bag AUC in this contract
is sample size dependent, which this file has already noted elsewhere (five
collapse flags at n=1500, zero at n=600). The same construction at the full 1500
reads 0.6261 on that cache, not 0.5715. So the claim that the correlation
ceiling sits "close to the 0.5118 codec floor" was an artifact of sample size
and is withdrawn. On the served cache the honest full n ceiling is 0.5896.

That number is the important one in this section. With PERFECT eighteen
dimensional marginals and a PERFECT eighteen dimensional linear dependence
structure, imposed by construction rather than achieved by any model, the
detector still reads 0.5896. Higher order copula structure holds 0.0896 of the
0.1938 excess, which is more than marginals and linear dependence together buy
from the served baseline. The feature space handle is close to exhausted; what
is left is not expressible as moments of the eighteen features.

The block rows kill a plan. Correcting ONE block of the correlation matrix while
leaving every cross term generated is worse than doing nothing on GEOMETRY and
on WOBBLE, on both caches, by a wide and stable margin. Only the full matrix
helps. `research/w4_latent.py` and `research/w4_hesit.py` were both built to
drive the wobble cluster and were priced by this table at NEGATIVE. Neither
should be run at full n. The dependence defect is distributed across the whole
matrix, not concentrated in a cluster, so a latent that couples six features to
each other and to nothing else makes the sample MORE detectable, not less.

ANGULAR is the only block that is positive on both caches, and it buys 0.0078.

## The decoder ran with the served lattice snap disabled too, 2026-08-07

Registration. Written before the run.

The duration prior was not the only default `w4_evprice.py` failed to set.
`experiments/event_stream_polar.py` reads `EVENT_SNAP` at import, default 0
which is OFF, and `_decode` applies it at line 216. `README.md` and every w3
evaluation path serve with `EVENT_SNAP=2.5`. `research/w4_served.py` fixed the
duration prior and still ran the decoder with snap off, so the seam is only half
checked.

The comment on the flag in `event_stream_polar.py` is the reason this is worth a
run rather than a footnote:

```
  Snap slow steps (s < threshold px) to the integer lattice as whole steps.
  Human slow motion is natively lattice-aligned (repeated identical 1px
  steps, occasional direction changes); rounding a smooth off-lattice path
  instead alternates lattice directions nearly every step, which manufactures
  a 3x angular-velocity excess at slow frames.
```

The residual factor on the served cache loads most heavily on
`angular_velocity_mean` at +0.356 and `angular_velocity_std` at +0.332. Those are
the two features that comment says the missing flag manufactures an excess in.
So the shared factor this file has spent a day treating as a deep architectural
absence may substantially be a decode artifact that a served flag already
removes.

What is being tested. `EVENT_SNAP=2.5` added, identical specs and seed, served
duration prior kept, nothing else changed, contract read once.

Branches, fixed now.

- MATERIAL, 0.02 or more. The angular arm of the residual factor is a decode
  artifact and the remaining target is smaller than this file currently claims.
- MARGINAL ONLY, 0.005 to 0.02.
- NO EFFECT, under 0.005.
- Reported as a LOSS regardless of AUC if the collapse count rises above three.

Prediction. MATERIAL, and larger than the duration fix was. Recorded with the
standing caveat that four mechanism predictions in this file are now wrong and
the thresholds are the trustworthy part, not the guess.

### THE RESULT. MATERIAL, minus 0.0425, and the angular arm was indeed an artifact

`research/w4_served.py` with `EVENT_SNAP=2.5` added, identical specs and seed,
served duration prior kept, one trajectory each.

```
                                  served      + snap     delta
  CONTRACT AUC                    0.6945      0.6519   -0.0425
  collapsed features                   3           3        +0
  mean |corr gap|                 0.0713      0.0583   -0.0130
  leading gap eigenvalue          1.1112      0.8023   -0.3090
```

MATERIAL, and the prediction was right, which breaks a run of four wrong ones.
The collapse count held at three rather than rising, so this is not a loss under
the anti Goodhart rule. Its composition moved, `mean_acceleration` came back
inside the band and `std_acceleration` fell out of it, which is a lateral move
and is recorded rather than claimed as progress.

The residual factor's six largest loadings, on the snap cache:

```
    mean_acceleration           +0.401
    max_deviation               +0.336
    curvature_mean              -0.306
    time_to_peak_velocity       +0.299
    path_efficiency             -0.296
    mean_velocity               +0.277
```

Both angular velocity features are GONE from the top six, having been the two
largest loadings before this run at +0.356 and +0.332. That is the registered
prediction confirmed on the mechanism and not only on the AUC: the angular arm
of the residual factor was the lattice rounding artifact the flag exists to
remove, not an architectural absence.

Where the two environment fixes leave the baseline:

```
  w4 cache, as every section above this one measured it     0.7119
  + EVENT_DUR_STD=1.0 DUR_EMPIRICAL=1                       0.6945
  + EVENT_SNAP=2.5                                          0.6519
```

Minus 0.0600 in total, from two environment flags. No retraining, no model
change, no new code in the sampling path, and no selection. Both flags were
already set by every serving path in this repo and by every w3 evaluation; the
w4 line of work simply never set them, and the four mechanism sections above
were all measured through that gap.

The factor that is left is now a velocity and acceleration factor with a
geometry arm, signed so that higher acceleration goes with larger deviation,
lower curvature and worse path efficiency. It is no longer the "hesitancy"
factor this file named earlier today, and that name should be retired. What
survives of the earlier reading is only the general claim, which four separate
measurements support, that the model's defect is a missing per trajectory
shared factor rather than wrong per event marginals.

Standing instruction for anything that follows. Do not generate against
`research/w4_evprice.py` defaults again. The served configuration for this
checkpoint family is `EVENT_DUR_STD=1.0 DUR_EMPIRICAL=1 EVENT_SNAP=2.5
EVENT_CHOICE_TEMP=10`, and `research/w4_snap_cache.npz` is the cache every
downstream diagnostic should now read.

### The ceilings on the snap cache, and a floor that will not move

`research/w4_block.py` on all three caches, full n, three forest seeds.

```
                          old      served     + snap
  baseline               0.7141    0.6938     0.6519
  marginals only         0.6778    0.6156     0.6029
  marginals + FULL corr  0.6261    0.5896     0.5931
```

The bottom row is the point. The baseline fell 0.062 across the two environment
fixes and the feature space ceiling did not follow it down; it went 0.6261,
0.5896, 0.5931 and has been flat for the last two. On the snap cache the whole
linear dependence correction is now worth 0.0098 over marginals alone, down from
0.0517 on the cache this file was reading this morning.

Read that as a decomposition of the 0.1519 the snap cache holds above chance:

```
  expressible as eighteen marginals            0.0490
  plus the full linear dependence structure     0.0098
  higher order copula structure                 0.0931
```

Both environment fixes removed almost exclusively the part that was expressible
in the first two rows. What is left is 61 percent higher order and it has not
moved. Two independent constructions now agree on a floor near 0.59 for anything
that can be written as moments of the eighteen features, and that floor is
imposed by construction rather than achieved by a model, so no marginal matching
decoder, no moment matching loss and no correlation targeting objective can beat
it. Task 3 in the ledger, the moment matched exponential tilt decoder, was
already marked superseded and this is the second independent reason to leave it.

The block rows repeat what the other two caches said. Single block correction is
negative on GEOMETRY and WOBBLE on all three caches without exception. ANGULAR is
positive on all three and worth 0.0062 here. There is no cluster local
intervention worth building.

### Two knobs checked and cleared, recorded so nobody spends a run on them

`EVENT_CHOICE_TEMP` and `EVENT_ORDER` are read by `event_stream_polar`'s own
sampler. The W4 line calls `EventARModel.sample` in `models/event_ar.py`
directly and uses `event_stream_polar` only for the duration prior, `_decode`
and the two dt constants. Neither model file reads the environment at all. So
both knobs are INERT on this path, and the `EVENT_CHOICE_TEMP=10` that every w4
script sets has never done anything here. Worth stating plainly because the
memory note attached to this project records that omitting it inflates one shot
AUC from 0.65 to 0.94; that is true of the event stream path and does not
transfer to this one.

`_DT_MEAN` and `_DT_STD` come from `event_polar_best.pt`, which is NOT the
checkpoint W4 samples from, and `event_ar_v2_s40000.pt` carries no dt constants
of its own. That looks like the same class of mismatch as the duration prior and
it is not one. W4 computes `dt_z = (log(dt_ms) - _DT_MEAN) / _DT_STD` and hands
it to `_decode`, which inverts the same transform with the same two constants.
It is a round trip and it cancels for any values. Checked and dismissed.

## The last decode artifact flag, EVENT_TICKMERGE, 2026-08-07

Registration. Written before the run.

`EVENT_TICKMERGE` defaults to 0 and `_decode` applies it at line 226. Unlike
`EVENT_SNAP` and the duration prior, this one is NOT part of the served recipe;
`generate.py` and `README.md` both leave it off. So this is a proposed change
rather than a configuration fix, and it has to earn its place on the result.

It merges a generated mid flight tick into the following event when both
neighbours are fast motion, adding the dropped event's time to the next one so
total duration is conserved. Deterministic, applied to every trajectory, no
selection, one trajectory per request. The flag's comment says humans emit about
4.5 mid flight ticks per 1000 events while the sampler's worst paths alternate
tick and motion at 1 ms, and that the resulting timing jitter dominates the
angular velocity features.

What is being tested. `EVENT_TICKMERGE=1` on top of the snap cache
configuration, identical specs and seed, nothing else changed.

Branches, fixed now.

- MATERIAL, 0.02 or more.
- MARGINAL ONLY, 0.005 to 0.02.
- NO EFFECT, under 0.005.
- Reported as a LOSS regardless of AUC if the collapse count rises above three.

Prediction. MARGINAL ONLY. The angular velocity features already left the top
six loadings when snap went on, and the ANGULAR block has only 0.0062 of
correlation room left on this cache, so the artifact this flag targets has
largely been removed by the previous fix. If it lands MATERIAL the two flags are
attacking different parts of the same artifact and that is worth knowing.

### THE RESULT. TICKMERGE is REJECTED, it moves the AUC the wrong way

```
                                  + snap   + tickmerge     delta
  CONTRACT AUC                    0.6519      0.6656   +0.0136
  collapsed features                   3           1        -2
  mean |corr gap|                 0.0583      0.0662   +0.0079
  leading gap eigenvalue          0.8023      0.9987   +0.1964
```

The registered prediction was MARGINAL ONLY improvement. It is not an
improvement at all: the flag makes the sample MORE detectable by 0.0136 and
widens the correlation gap on both measures. Rejected. Note that the verdict
string `w4_served.py` printed reads NO EFFECT, which is wrong for this run; that
script's branch logic was written for a change expected to help and puts any
positive delta in its else arm. The number is right and the label is not.

It also pushes the angular velocity features BACK into the top six loadings,
which snap had removed. Two flags aimed at the same artifact and they are not
complementary; snap does the job and tickmerge partially undoes it.

One thing worth keeping. The collapse count fell from three to one, its best
value anywhere in this file, while the AUC got worse. This repo's anti Goodhart
rule says an AUC gain bought with a worse collapse is a loss. The converse is
not automatically a win and this is not being claimed as one. It does say the
two objectives are not aligned on this checkpoint, which is worth knowing before
anyone treats the collapse flag as a free secondary target.

### A methodological hole, found late and reported rather than papered over

`DurationModel.__init__` ends with `self._rng = np.random.default_rng()`, with no
seed. So every run draws different durations even on identical specs and an
identical spec seed, and `EventARModel.sample` uses the global torch RNG in
`w4_served.py`'s generation loop, which is also unseeded. Two independent
sources of run to run variation, and every delta in the three sections above was
read as though the only difference between arms was the flag.

The first replicate of the snap configuration, same command, same spec seed,
nothing changed, reads 0.6443 against the 0.6519 it is being compared to. That
is a swing of 0.0077 from noise alone, which is larger than the NO EFFECT
threshold I have been registering at 0.005 and comparable to the 0.0136 that
just got TICKMERGE rejected.

What this does and does not touch.

- The duration prior result, minus 0.0174, is about two replicate widths. It
  survives but it is weaker evidence than the section above presents it as.
- The snap result, minus 0.0425, is roughly five replicate widths and survives
  comfortably.
- The TICKMERGE rejection at plus 0.0136 is under two replicate widths and is
  NOT safe on one run. It is being held as rejected only because the correlation
  gap moved the same direction on two separate measures, which noise in the AUC
  alone would not produce.
- Every NO EFFECT threshold registered at 0.005 anywhere in the W4 sections is
  below the noise floor and was never a meaningful branch.

More replicates are running. Nothing above this line should be requoted without
them.

## Is the higher order residual a missing mode, 2026-08-07

Registration. Written before `research/w4_modes.py` ran.

The decomposition on the snap cache leaves 0.0931 of a 0.1519 excess as higher
order copula structure, unreachable by any objective written in terms of the
eighteen features' moments. The question is what KIND of higher order.

The standard generative failure with this signature is mode averaging. A model
fitted by likelihood to a multimodal target places mass between the modes, which
preserves means, variances and correlations while getting the joint density
wrong. That is the pattern exactly.

Two readouts, on pooled normal scores so the marginals of the two sides are put
on one scale first and only dependence can survive, then a PCA to six so the
covariance estimates are honest at n=1500.

  1. Held out mixture log likelihood at k = 1 to 5, fitted on half a sample and
     scored on the other half, separately for each side. Held out, so a larger k
     cannot win by memorising.
  2. The mixture fitted on HUMAN, both samples assigned to its components, and
     the mixing proportions compared. A component holding a real share of humans
     and almost no generated rows is a countable missing regime, and its centre
     in feature space names it.

Branches, fixed now.

- MISSING MODE. Human's held out likelihood gains at least 0.10 nats more from
  extra components than the model's does, AND the worst human component is under
  populated by the model at a ratio of 0.6 or below. A per trajectory mixture
  over regimes becomes the obvious generative change and it is one the trunk can
  carry through `cond`.
- UNEVEN, NOT MULTIMODAL. The proportion test fires and the likelihood test does
  not. The density is misshapen rather than missing a mode, and a discrete
  mixture is the wrong instrument.
- NO MISSING MODE. Neither fires. Mode averaging is excluded and the higher
  order residual is something else, which would make this the strongest negative
  result in W4 and would push the work to the event level rather than the
  trajectory level.

Prediction. MISSING MODE, on the reasoning that the residual factor's signed
loadings on the snap cache describe a coherent behavioural regime (high
acceleration and velocity with large deviation, late peak velocity, low
curvature and poor path efficiency, which reads as a fast overshooting sweep
followed by correction) rather than a diffuse density error. Registered with the
standing caveat that mechanism predictions in this file are running at one
correct out of five.

### THE RESULT. NO MISSING MODE, and it closes the per trajectory latent program

```
  sample             k=1       k=2       k=3       k=4       k=5    best
  human          -10.576   -10.370   -10.385   -10.298   -10.274       5
  generated      -10.564   -10.323   -10.331   -10.238   -10.235       5

  gain over a single Gaussian   human +0.302   generated +0.329
```

Both sides want five components and the model gains slightly MORE from extra
components than the human sample does, +0.329 against +0.302. The registered
condition was that human gain exceed the model's by at least 0.10 nats. It does
not exceed it at all, it is lower. Mode averaging is excluded.

The proportion readout did not fire either, though it came close:

```
   component    human  generated    ratio
           0    0.409      0.451     1.10
           1    0.164      0.121     0.74
           2    0.217      0.141     0.65
           3    0.119      0.140     1.18
           4    0.092      0.147     1.60
```

Component 2 sits at 0.65 against a registered line of 0.60, so NO MISSING MODE
is the branch. Recorded anyway because the direction is interpretable: that
component's centre is low curvature, low angular velocity, high path efficiency,
which is the smooth direct movement, and the model produces 14.1 percent of them
against the human 21.7 percent. The model under-produces clean paths. It did not
clear the bar and is not being promoted to a finding.

The prediction was MISSING MODE and it was wrong. That is one correct out of six
mechanism predictions in this file. The predictions are worthless; the
registered thresholds are the only reason any of these sections mean anything,
and the practice of fixing them in advance should be kept exactly as it is.

What this closes. Three independent measurements now say the same thing about
per trajectory latent variable approaches:

  1. `w4_block`, on all three caches: correcting one block of the correlation
     matrix while leaving cross terms generated is worse than doing nothing.
  2. `w4_hesit`'s Phase B, before this: the best achievable alignment between a
     three head logit tilt and the target direction was 0.4476, implying a
     contract move around 0.004.
  3. `w4_modes`, here: the human sample is not a mixture the model is missing.

`research/w4_latent.py` and `research/w4_hesit.py` should not be run at full n.
The thing that separates these samples is not a per trajectory shared factor,
whatever the correlation gap's leading eigenvector looks like. An eigenvector
exists for any nonzero gap matrix; that one has a coherent behavioural reading
was pattern matching on my part, and the file said so at the time and then spent
a day on it anyway.

Where that leaves the work. The residual is higher order, it is not a mixture
over trajectories, and it is not reachable by any moment based objective. What is
left is within trajectory temporal structure: the model's per event conditionals
can be individually right while the sequence they generate has the wrong
autocorrelation, burstiness or run length structure, and the eighteen features
are aggregates that would register exactly that. That is a property of the AR
model itself rather than of anything at generation time, so the next diagnostic
belongs at the event level, not the trajectory level.

### THE RESULT. The higher order residual is not concentrated either

`research/w4_tail.py`. The instrument: for any Gaussian copula, the correlation
of two normal scores' ABSOLUTE values is fixed by their linear correlation
alone, `corr(|X|,|Y|) = (2/pi)(sqrt(1-r^2) + r*arcsin(r) - 1)/(1 - 2/pi)`. The
observed minus implied difference is therefore zero for any Gaussian copula
whatever its correlation matrix, and non zero only for volatility coupling and
tail dependence. It is orthogonal by construction to everything `w4_joint` and
`w4_copula` already priced. 200 bootstrap resamples per side give each pair its
own error bar.

```
  mean |excess| over the 153 pairs
    human      0.0636
    generated  0.0581
  mean |human minus generated excess|   0.0356
  pairs beyond 3 sigma                  3 of 153
```

The model's pairwise higher order dependence is close to the human sample's in
aggregate, 0.058 against 0.064, and only three pairs of 153 separate beyond
three sigma. The three are `velocity_skewness` with `curvature_std` and with
`curvature_mean`, and `mean_acceleration` with `mean_jerk`.

So the higher order residual is not concentrated in a handful of pairs any more
than the marginals were concentrated in a handful of features.

### What six measurements now say together, and it is one thing

This is the summary the rest of W4 should be read through.

```
  w4_featmap   DIFFUSE. Best single feature alone 0.5801 against 0.7149 for all
               eighteen. Largest group removal worth 0.0186.
  w4_joint     Eighteen PERFECT marginals buy 0.0490 on the snap cache.
  w4_block     The full correlation matrix buys 0.0098 more. No single block is
               worth correcting; two are worth less than nothing.
  w4_copula    Floor near 0.593 for anything expressible in the eighteen
               features' moments, imposed by construction, not achieved.
  w4_modes     Not a missing mixture component. The model is if anything
               slightly MORE multimodal than the human sample.
  w4_tail      Pairwise higher order dependence differs on 3 pairs of 153.
```

There is no concentrated defect at any order. Not in one feature, not in one
group, not in one correlation block, not in a missing regime, not in a handful
of tail dependent pairs. The model is slightly wrong nearly everywhere and a
random forest with eighteen inputs aggregates many small wrongnesses into 0.65.

That is a coherent picture and it explains the whole of W4's history. It explains
why the two environment flags worked: each removed one specific artifact,
duration spread and lattice rounding, and each was worth a real amount. It
explains why every structured intervention has failed: latents, tilts, block
corrections and moment matching all assume a concentrated defect to aim at, and
there is not one. And it predicts, correctly, that the diagnostics kept coming
back DIFFUSE and NO EFFECT.

It also sets what could still work. A uniformly slight error has one uniform
generation time control: the sampling temperature. It has never been swept on
this path with the duration prior and the lattice snap set correctly, and the
collapse flag firing on the acceleration and jerk features says the model is
under dispersed, which points above 1.0. That is `research/w4_temp.py` and it is
the last generation time lever this file has not pulled.

### The replicate noise floor, measured

Three runs of the snap configuration, same command, same spec seed, nothing
changed, before the seeding fix went into `w4_served.py`:

```
  0.6519   0.6443   0.6563      spread 0.0120, sd about 0.006
  collapse 3, 1, 0
```

So the AUC noise floor on a single unpaired run is about 0.006 and the collapse
count is far noisier than that, taking three different values across three
identical runs. Every collapse count quoted anywhere in W4 should be read as
plus or minus two. The anti Goodhart flag is a weak instrument at n=1500 and
should not be used to adjudicate anything on one run.

`research/w4_paired.py` and `research/w4_temp.py` now seed the duration prior
and the torch stream per arm, so arms inside a seed differ only by the change
under test.

## Sampling temperature on the corrected configuration, 2026-08-07

Registration. Written before `research/w4_temp.py` ran.

The six measurement summary above says there is no concentrated defect at any
order. A uniformly slight error has exactly one uniform generation time control
and it is the sampling temperature. It has never been swept on this path with
the duration prior and the lattice snap set correctly, and every temperature
number anywhere in W4 was measured through the same misconfiguration the
2026-08-07 entries just corrected.

Direction. The contract's collapse flag fires on the acceleration and jerk
features, which is UNDER dispersion, so the model is too conservative and the
interesting side is above 1.0. Below 1.0 is included as a control: if 0.9 also
improves the AUC then the sweep is measuring something other than dispersion.

One scalar, applied to every trajectory, one trajectory per request. Nothing
generated twice, nothing selected. Paired within a seed, so the specs, the
duration draws and the torch stream are identical across temperatures.

Branches, fixed now, and they are set against the MEASURED noise floor rather
than the 0.005 lines used earlier in W4, which this file has now established
were below the noise the whole time.

- REAL, a paired mean improvement of 0.015 or more with the same sign on every
  seed. Temperature becomes the first generation time control in W4 that works,
  and the sweep is refined around the best value.
- NOISE, anything smaller, or any improvement whose sign flips across seeds.
- Reported as a LOSS regardless of AUC if the collapse count rises, since a
  higher temperature that widens dispersion without helping is exactly the
  Goodhart failure the flag exists to catch.

Prediction. NOISE, and a small one. The reasoning is that temperature scales
every logit uniformly and the six measurements say the errors are diffuse but
not uniform in sign; broadening everything should overshoot on the features that
are already correctly dispersed while helping the collapsed ones. Recorded with
the standing note that mechanism predictions in this file run at one correct out
of six, so the thresholds carry the weight.

Thermal note. The card holds 76 to 77C under this workload against a tightened
79C kill line, so `cooldown()` to 74 spends its full 420 second budget between
arms. The sweep is sized to three temperatures and two seeds for that reason,
not because six runs is the statistically desirable number.

### An addendum to the temperature registration, written after it launched

I registered the temperature sweep without re-reading this file's own earlier
temperature line, and it says something the registration should have accounted
for. Recorded here rather than folded silently into the prediction, because the
whole value of pre-registration is that the thresholds do not move after the
fact.

The earlier work swept `th_temperature`, the direction head alone, and found
SHARPENING helps the contract:

```
                  mean AUC   improvement   held out nats
  base              0.6612                 0 by definition
  th 0.928          0.6420      +0.0192    +0.000382 BETTER
  th 0.85           0.6221      +0.0391    -0.005324 WORSE
```

Two things follow. First, my registered reasoning, that the collapse flag shows
under dispersion so the interesting side is above 1.0, is contradicted by direct
measurement on this model. The sweep includes 0.9 as a control and will pick that
up, so the design survives the bad reasoning, but the prediction of NOISE was
made on a premise this file had already refuted.

Second, and this matters more than which direction wins. The earlier line
established that sharpening buys detector score by making the model WORSE on its
own held out likelihood, at 8.7 sigma at th 0.85, and buys more the further it
degrades. That is the Goodhart signature in its purest form: the contract scorer
paying for damage to the generative model. So if 0.9 wins this sweep, the win is
not established until the held out likelihood is measured at the same setting,
and a temperature that improves the AUC while worsening held out nats is a loss
under the mandate, which asks for a model that produces human trajectories and
not for a number.

The registered branches stand as written. What this addendum changes is the
meaning of REAL, which now requires a held out likelihood check before the result
counts, not just a paired mean above 0.015.

### Two things this session rediscovered that were already written down

Recorded because the pattern matters more than either item.

The replicate noise floor. `RESUME.md` has carried this since 2026-08-06: "NOISE
FLOOR, measured three times, use it. Two runs of the identical baseline config at
the same seed scored 0.6522 and 0.6581. `--seed` does NOT pin the sampling stream
because CUDA generation is not bit deterministic. Per seed noise is about 0.006.
Never read a single seed. Read the mean of three." Today's three replicates read
0.6519, 0.6443 and 0.6563, sd 0.006, which reproduces that number exactly. So the
noise floor was known, written down, and stated in the file every session is
supposed to start from, and W4 registered NO EFFECT thresholds at 0.005 anyway
and read single runs for a day. The measurement above is a confirmation, not a
discovery, and the section reporting it should be read that way.

The `pkill` footgun. `RESUME.md`: "NEVER `pkill -f <pattern>` from a Bash tool
call. The pattern matches the calling shell's own command line and kills the
session's shell. It happened three times today." It happened a fourth time today,
to me, stopping the replicate run. Kill by explicit PID.

Both are the same failure as the duration prior and the lattice snap: the
information needed was already in the repository and the work proceeded without
checking it. That is now four instances in two days, and it has cost more than
every modelling idea in W4 combined has bought.

### A second addendum. This sweep is near a closed thread and here is why it ran

`RESUME.md`'s DO NOT RE DERIVE list contains "sampling temperature tuning on the
flow model, the speed head AND NOW THE AR DIRECTION HEAD". `models/event_ar.py`
sets `th_temp = temperature if th_temperature is None else th_temperature`, so a
global temperature moves the direction head too. This sweep therefore overlaps a
thread the record calls closed, and that has to be said plainly rather than
discovered by whoever reads this next.

The reason it ran anyway is the only reason that would justify it: every
temperature number in this file was measured through the misconfiguration the
2026-08-07 entries corrected, and correcting it moved the baseline 0.060, which
is three times the largest temperature effect ever recorded here. A closure
established on a baseline that has since moved that far is worth one confirmation
at six runs, and not more than that.

The commitment, made before the result is known. If the sweep returns NOISE, the
closure stands, this is the last temperature run in W4, and the thread goes back
on the list with a note that it was re checked on the corrected configuration.
If it returns REAL it still has to pass the held out likelihood check in the
first addendum before it counts as anything.

### THE RESULT. NOISE below 1.0, a hard LOSS above it, and the collapse flag points the wrong way

Six runs, 1500 specs each, two seeds, three temperatures, specs and duration
draws and the torch stream paired inside a seed. Lower AUC is better.

```
  seed   temp  contract  collapse      n
     0   0.90    0.6496         3   1500
     0   1.00    0.6544         3   1500
     0   1.10    0.7131         0   1499
     1   0.90    0.6525         3   1500
     1   1.00    0.6529         4   1500
     1   1.10    0.7020         0   1500

  temp   mean AUC       sd   collapse   paired vs 1.0
  0.90     0.6511   0.0020       3.00   -0.0026 sd 0.0030
  1.00     0.6536   0.0010       3.50    0.0000
  1.10     0.7075   0.0079       0.00   +0.0539 sd 0.0068
```

Against the registered branches. 0.90 improves the contract by 0.0026 with a
paired sd of 0.0030, which is below the 0.015 threshold and smaller than its own
spread. That is NOISE, so the closure stands, the held out likelihood check is
not triggered because there is nothing to check, and this is the last temperature
run in W4. The thread goes back on the DO NOT RE DERIVE list with a note that it
was re checked on the corrected configuration and did not move.

The registration predicted NOISE. That is prediction 2 correct of 7.

The part that was not registered and matters more. Raising the temperature to
1.10 sets the collapse count to zero on both seeds and costs 0.0539 of AUC, which
is nine times the noise floor and the largest single effect W4 has measured other
than the two configuration corrections themselves. The two readouts move in
opposite directions, hard, and both are computed from the same feature matrix by
the same scorer, so this is not an artifact of one of them: `dispersion_ratios`
is a per feature std ratio that is reported and never fed to the forest, and the
forest always sees all eighteen columns whether or not any of them is flagged.

So the premise this sweep was built on is refuted by the sweep. Its docstring
says "the contract's own collapse flag has been firing on the acceleration and
jerk features, which is UNDER dispersion, so the direction to test is mainly
above 1.0." The dispersion deficit is real and the temperature does close it, and
closing it makes the model easier to detect, not harder. Whatever the forest is
using on those columns, it is not their standard deviation.

The shape of the response says something too. From 0.90 to 1.00 the contract
moves 0.0026 across a tenth of a unit of temperature. From 1.00 to 1.10 it moves
0.0539 across the same tenth, twenty times as much, in the direction that also
takes three or four collapse flags to zero at a stroke. A std ratio crossing 0.2
upward on three separate features from one ten percent change in temperature is
not a bulk of the distribution moving. It is consistent with a small number of
trajectories going wild and dragging the second moment with them, which would
clear the flag and hand the forest an easy tell in the same step.

That is a testable statement rather than an interpretation, and it is registered
below rather than asserted here.

## Does temperature widen the bulk or only add tails, 2026-08-07

Registered before running. The claim under test is the one the sweep result ends
on: that the temperature clears the collapse flag by producing a few extreme
trajectories rather than by widening the distribution the humans actually
occupy, and that those same extreme trajectories are what hands the forest the
0.0539.

Three readouts, all on the same two feature matrices, one seed, temperatures
1.00 and 1.10, everything else paired.

  1. For each feature, the std ratio the collapse flag reads, next to an
     interquartile ratio and a ten percent trimmed std ratio. The std is a
     second moment and four wild rows in fifteen hundred can move it. The IQR
     cannot be moved by anything outside the middle half of the sample. If the
     temperature is widening the bulk, both rise together. If it is adding
     tails, the std ratio rises and the IQR ratio does not.

  2. The share of each feature's generated variance contributed by the one
     percent of rows furthest from the median. Read directly, per feature, at
     both temperatures.

  3. The contract AUC at 1.10 recomputed with the one percent most extreme
     generated rows removed. This is a DIAGNOSTIC and not a generation method.
     Nothing is regenerated, nothing is selected for a deliverable, and no
     configuration in this repository will ever drop rows. It exists to answer
     one question: is the 0.0539 carried by fifteen rows or by fifteen hundred.

Branches, with the thresholds fixed now.

  TAIL INFLATION. On the features flagged at 1.00, the std ratio rises by a
  factor of three or more from 1.00 to 1.10 while the IQR ratio rises by less
  than a factor of 1.5, AND removing the extreme one percent at 1.10 returns
  the AUC to within 0.010 of the 1.00 value. The collapse flag is then a second
  moment artifact on this path, the under dispersion of the bulk is untouched by
  temperature, and no scalar on the logits will reach it.

  GENUINE WIDENING. The IQR ratio rises by at least half as much as the std
  ratio does. The temperature is then really widening the distribution, the
  under dispersion is being corrected, and the AUC loss comes from somewhere
  other than tails, which would be a different and more interesting problem than
  the one this registration expects.

  Anything else is reported as measured with no verdict attached.

Prediction, on the record: TAIL INFLATION. The record on predictions is 2 of 7.

## Does the defect grow with the length of the rollout, 2026-08-07

Registered before running, and registered while the dispersion anatomy above is
still on the GPU, so the two are independent and neither result could have
shaped the other's thresholds.

The reasoning. Six diagnostics agree the gap is diffuse at every order. That is
not what a named defect looks like. It is what an ACCUMULATING error looks like.
`EventARModel` is autoregressive and trained by teacher forcing, so at sample
time it conditions on its own output and any per step bias compounds along the
rollout. Exposure bias is the standard name for this, it produces exactly the
diffuse higher order signature W4 has measured, and it makes one prediction
nothing here has tested: the discrepancy should be small for short trajectories
and large for long ones.

The test is free. Bin both samples by `movement_duration`, which the model is
conditioned on and therefore matches by construction, and run the contract
recipe inside each bin. Duration cannot separate the samples inside a bin where
it is matched, so a bin to bin trend is a statement about what the model does
with a longer rollout rather than about how long the rollouts are. Bootstrap the
generated side inside each bin so the trend is read against its own sampling
error and not against zero.

  DRIFT     The longest bin exceeds the shortest by more than twice the
            bootstrap spread of that difference, AND the AUC is ordered across
            bins. The error compounds. The remedies are named and standard:
            scheduled sampling, a rollout consistent objective, or a non
            autoregressive decoder.

  UNIFORM   Longest minus shortest within twice the bootstrap spread. The per
            step error does not compound, exposure bias is excluded, and one of
            the few remaining structural hypotheses is closed.

  Ordered the wrong way or non monotone is reported as measured with no verdict.

Prediction, on the record: DRIFT. The record on predictions is 2 of 7, and this
one is the first prediction in W4 that names a mechanism with a standard remedy
rather than a knob, so it is also the one worth being wrong about.

### THE RESULT. DRIFT, and immediately a reason not to trust it yet

```
   bin      duration range   n human   n gen      AUC   boot sd   mean ndc
     0        [-inf, 0.25)       664     480   0.6219    0.0081        8.4
     1       [0.25, 0.585)       669     460   0.6521    0.0076       22.7
     2        [0.585, inf)       667     560   0.6624    0.0111       58.8

  longest bin minus shortest  +0.0405  against a bootstrap spread of 0.0138
  monotone in duration  True
```

That clears the registered DRIFT threshold: monotone, and the gap is three
times its own bootstrap spread. Prediction 3 correct of 8.

It does not establish the mechanism, and the reason is in the last column. The
shortest bin holds trajectories with about eight direction changes and the
longest about fifty nine. Every one of the eighteen contract features is a
summary statistic, and a summary computed over eight events is a much noisier
estimate than the same summary over fifty nine. Noise masks a difference. So a
constant per step error, aggregated over a short trajectory, would produce this
exact table without anything having accumulated.

The two explanations cannot be told apart by any statistic computed on aggregate
features against observation length, because both predict the same trend for the
same reason. Registering DRIFT and moving on would have been a mistake, and the
verdict above should be read as "the aggregate trend is real" and not as
"exposure bias is the mechanism".

## Where along the trajectory the error lives, 2026-08-07

Registered before running. This is the control the section above needs.

Do not aggregate. For each step, compare the distribution of what the model
emitted at that step against the human distribution at that step. Every step
index has the same number of samples on both sides, so there is no noise masking
left to confound anything. Step speed and step turn angle, on the same 125 Hz
grid the feature extractor uses, two sample Kolmogorov Smirnov distance per
position.

One confound survives that and has to be designed out. Every trajectory
decelerates into its target, so a model that gets ENDINGS wrong shows a
discrepancy that grows with position without anything having accumulated.
Separating the two needs both axes at once. Split by total length into three
bands and read the discrepancy against NORMALISED position inside each band.

  Accumulation predicts that at the same fraction of the way through, the longer
  band is worse, because more steps have elapsed.

  A terminal effect predicts the same curve in every band with a spike in the
  last bin, because the ending is the ending regardless of length.

Branches, thresholds fixed now. The mid rollout statistic is the mean KS over
position bins 2 through 8, the last bin is excluded from it by construction, and
the bootstrap resamples TRAJECTORIES so the quoted spread is the spread of the
sample rather than of the steps.

  ACCUMULATION    mid rollout KS rises across bands by more than twice the
                  bootstrap spread of that difference, and is ordered.
  TERMINAL        the last bin sits more than two bootstrap spreads above the
                  middle in every band, without the length trend.
  BOTH            both clear.
  UNIFORM         neither clears. The per step error is flat along the rollout,
                  the w4_length trend was the aggregation and not the model, and
                  exposure bias is excluded as the mechanism.

Two things this run also checks that nothing in W4 has checked before. The human
paths are reconstructed from the pool rather than read as features, and the run
asserts that the reconstruction reproduces `data/human_val_features_grpo.npy`
before measuring anything, so the two sides are provably the same sample. And it
is the first W4 diagnostic to look at the trajectories themselves rather than at
the eighteen numbers extracted from them.

Prediction, on the record: ACCUMULATION. The record is 3 of 8.

### THE REGISTRATION ABOVE IS WITHDRAWN, and it destroyed a file on its way in

Two things went wrong and both are the same thing.

**The question was already answered, better, on 2026-07-29.** `RESUME.md`'s DO
NOT RE DERIVE list contains "the whole training time exposure fix family",
"accumulation and drift", "a progress dependent bias" and "step to step residual
dependence". The section at line 5629 of this file states it plainly: "Not
accumulation or drift, by `w4_position` at 5.2 sd. Not exposure, by `w4_drift`
pricing that family at a third or less." The surviving output of that run says
how it was priced, and the instrument was better than the one registered above:

```
  prefix   self conditioned   teacher forced   ratio forced / self
       4        0.0142             0.0006           0.04
       8        0.0480             0.0336           0.70
      16        0.0476             0.0271           0.57
      32        0.0837             0.0718           0.86
```

At a 32 event prefix, 86 percent of the discrepancy is present when the model is
fed a REAL history and only the remainder can be attributed to conditioning on
its own output. That is a direct measurement of exposure bias with the
confound removed by construction, and it is strictly stronger than comparing
Kolmogorov Smirnov curves across length bands. The registration above would have
spent a GPU run to answer worse a question with a better answer already on disk.

So it is withdrawn before it ran. `research/w4_stepdrift.py` exists and is not
going to be run. The `w4_length` trend stands as an observation, its mechanism is
already settled as not accumulation, and the noise masking alternative raised in
the section before this one is the more likely reading of it.

**And writing that script overwrote `research/w4_drift.py`.** The 2026-07-29
script that produced the table above is gone. It was untracked, so there is no
git copy, and a filesystem search found no other. What survives is
`research/w4_drift.json`, quoted in full above, and this file's write up of it.
The new script has been renamed to `research/w4_stepdrift.py` so the name is free
again, but the original cannot be restored and any rerun of it would have to be
written from scratch against the JSON and the prose.

The cause is not the file operation. It is that a diagnostic was named, designed,
registered and written before the DO NOT RE DERIVE list was read, and the list is
in the file the session is supposed to start from. That is the same failure as
the duration prior, the lattice snap, the noise floor and the `pkill` footgun.
This is the fifth instance in two days and the first one that cost an artifact
rather than a run.

### THE RESULT. NEITHER BRANCH, and the table says something neither branch anticipated

```
  temp 1.00  contract 0.6544  collapse 3
  temp 1.10  contract 0.7131  collapse 0

  ratios to human, per feature. * flagged by the contract at temp 1.00
  feature                     std lo  std hi  IQR lo  IQR hi  trim lo  trim hi
   mean_velocity               1.224   1.447   1.114   1.195    1.146    1.316
   std_velocity                0.316   0.525   1.033   1.229    1.028    1.212
  *max_velocity                0.213   0.400   1.042   1.255    1.010    1.346
   velocity_skewness           0.887   1.023   1.011   1.038    0.976    1.022
   mean_acceleration           0.262   0.375   0.957   0.999    1.055    1.178
  *std_acceleration            0.215   0.414   1.099   1.595    1.110    1.599
  *max_acceleration            0.156   0.306   1.075   1.577    1.057    1.688
   mean_jerk                   0.254   0.296   1.129   1.172    1.476    1.614
   std_jerk                    0.301   0.627   1.079   1.663    1.059    1.609
   path_efficiency             1.117   1.233   1.515   2.055    1.213    1.418
   max_deviation               1.319   1.728   1.419   1.658    1.389    1.703
   curvature_mean              2.197   1.802   1.733   2.134   13.624    8.965
   curvature_std               1.665   1.498   2.002   2.774    9.586    7.115
   num_direction_changes       1.139   1.152   1.103   1.103    1.132    1.124
   movement_duration           0.933   0.974   1.064   1.056    1.042    1.043
   time_to_peak_velocity       1.093   1.188   1.173   1.320    1.118    1.250
   angular_velocity_mean       1.124   1.219   1.032   1.129    1.052    1.195
   angular_velocity_std        1.008   1.085   0.893   1.061    0.968    1.083

  on the 3 flagged features, mean rise 1.00 to 1.10:  std x1.92   IQR x1.37

  removing the 1% most extreme generated rows (15 of 1499)
    temp 1.00   0.6544 -> 0.6496   (-0.0048)
    temp 1.10   0.7131 -> 0.7050   (-0.0081)
```

Against the registered branches. TAIL INFLATION wanted the std to rise by three
or more while the IQR rose by less than 1.5, and the trimmed AUC to come back to
within 0.010 of the low temperature value. The std rose 1.92, the IQR rose 1.37,
and removing the extreme one percent left the high temperature arm at 0.7050
against 0.6544. GENUINE WIDENING wanted the IQR to rise by at least half as much
as the std, which at 1.37 against 1.92 it did. So the registered reading is
NEITHER BRANCH by the letter and closer to GENUINE WIDENING than to TAIL
INFLATION, and the prediction was wrong. That is 3 correct of 9.

**What the table actually says, and it is worth more than the branch it failed.**

Read the flagged rows across, not down. `max_velocity` has a standard deviation
ratio of 0.213 and an interquartile ratio of 1.042. `std_acceleration`, 0.215 and
1.099. `max_acceleration`, 0.156 and 1.075. The middle half of each of those
distributions is already the right width, within a few percent, and the flag is
firing entirely on what is outside it. These are heavy tailed features on the
human side, `max_acceleration` has a human standard deviation twelve times its
own mean, so a second moment ratio is dominated by a handful of extreme human
rows the model never produces.

The contract's collapse flag is therefore not saying what W4 has been reading it
as saying for two days. It is not "the model is under dispersed". It is "the
model does not reproduce the human sample's rarest extremes", while the bulk of
the same feature is correct to within a few percent. Every plan built on closing
the dispersion gap was aimed at a defect that is not in the bulk.

That also explains the temperature result without needing tails at all. At 1.10
the interquartile ratios on those features go from about 1.08 to about 1.58,
which is not a correction, it is an overshoot of fifty percent on a width that
was already right. The temperature buys some of the missing tail and pays for it
by breaking a bulk that did not need fixing, and the contract charges 0.0539.

**And the 0.0539 is not carried by a few rows.** Removing the one percent most
extreme generated rows moves the high temperature arm by 0.0081 and the low
temperature arm by 0.0048. The difference between the two arms survives the
surgery almost entirely, so the cost of the temperature is spread across the
whole sample, consistent with everything else W4 has measured about this gap.

**One thing in the table is not about temperature and is the largest number on
it.** `curvature_mean` has a trimmed standard deviation ratio of 13.6 and
`curvature_std` of 9.6, at the SERVED temperature, with the interquartile ratios
at 1.73 and 2.00. Inside the central ninety percent of the distribution, where
there are no outliers left to blame, the model's curvature varies roughly an
order of magnitude more than a human's. No other feature is within a factor of
five of that. It is a bulk defect, it is at the served configuration, and it is
the first thing in W4 that is both large and localised.

It is bounded, and the bound is already measured. `w4_joint` priced eighteen
PERFECT marginals at 0.0490 of the 0.1519, so correcting curvature's marginal
alone cannot buy more than that and will buy a fraction of it. The value of the
observation is not as a lever, it is as a pointer to what the decoder is doing to
angles, and it belongs with the lattice snap finding rather than with the
dispersion story.

### A correction to the curvature paragraph above, made ten minutes after writing it

The trimmed dispersion ratio of 13.6 on `curvature_mean` is real as a number and
the sentence built on it was wrong. "Inside the central ninety percent, where
there are no outliers left to blame" assumed a five to ninety five window trims
the tail. On this feature it does not, because the feature spans five orders of
magnitude inside that window. The percentiles:

```
  curvature_mean       p1      p5     p25     p50     p75       p95       p99
    human           0.000   0.003   0.030   0.118   0.402  3917.217 27035.897
    generated       0.002   0.011   0.071   0.186   0.668 16695.447 52661.120

  curvature_std        p1      p5     p25     p50     p75       p95       p99
    human           0.000   0.006   0.070   0.346   1.898 50250.188162183.886
    generated       0.003   0.020   0.184   0.569   3.591128126.611223354.452

  max_deviation        p1      p5     p25     p50     p75       p95       p99
    human           0.000   0.991   4.475  16.212  53.867   235.628   532.480
    generated       0.740   1.632   6.971  22.525  76.659   355.009   814.987

  path_efficiency      p1      p5     p25     p50     p75       p95       p99
    human           0.100   0.302   0.793   0.943   0.985     0.997     1.000
    generated       0.087   0.213   0.669   0.901   0.971     0.994     0.998
```

So it is not a width statement. It is a LOCATION statement, and a consistent one.
The model's curvature is larger than the human's at every percentile, by about
1.6 times at the median and four times at the ninety fifth. Its maximum deviation
from the straight line is larger at every percentile, by about 1.4 times. Its
path efficiency is lower at every percentile. Those are three different features
and one fact: **the model's paths are systematically less straight than human
paths.**

That is the same direction as `w4_modes` component 2, which holds 21.7 percent of
humans against 14.1 percent of generated and whose centre is low curvature, low
angular velocity and high path efficiency. Two independent instruments now say
the model under produces clean direct movement, and the second one was measured
before the first existed.

It is still bounded by `w4_joint`. Eighteen perfect marginals buy 0.0490 of the
0.1519, so this cannot be worth more than that as a marginal correction. Its
value is that it names a direction in path space rather than a feature, and one
knob in the served decoder acts on exactly that direction.

## Sweeping the lattice snap threshold, 2026-08-07

Registered before running, and checked against `RESUME.md`'s DO NOT RE DERIVE
list first this time. The list contains the hand built lattice emitter, affine
conditioning pre distortion, reveal order changes and local Gibbs refinement. It
does not contain the snap threshold, and the only comparison ever run on it is
present against absent.

The knob. `_decode` in `experiments/event_stream_polar.py` rounds `dx` and `dy`
to whole lattice units for steps slower than `_SNAP`, leaving the integrated
heading continuous. Turning it on at 2.5 was worth 0.0425 this morning, the
largest single improvement in W4's record, and the mechanism is that sub pixel
jitter on slow steps inflates every angular quantity. The section above says the
model's paths are too curved, too deviating and too inefficient at every
percentile. The knob acts on precisely that, and its threshold has never been
moved because 2.5 was inherited from the flow model's serving recipe, not chosen
for this checkpoint.

Arms 2.5, 3.5, 5.0 and 8.0, two seeds, specs and duration draws and the torch
stream paired inside a seed, one trajectory per spec, nothing selected.

Branches, thresholds fixed now. Lower AUC is better.

  REAL       a paired mean improvement of 0.015 or more against 2.5, same sign
             on both seeds, AND the mechanism check below passing.
  NOISE      anything smaller, or any improvement whose sign flips across seeds.

The mechanism check. An AUC improvement counts as confirming the mechanism only
if the median ratios of `curvature_mean`, `max_deviation` and `path_efficiency`
all move TOWARD one. If the AUC improves while those get worse, the win is by
some other route and is reported as an unexplained win, not as a straightness
correction.

Reported as a LOSS regardless of AUC if the collapse count rises. Today's
dispersion result says that flag reports missing extreme tails rather than an
under dispersed bulk, so it is being kept as a guard and not as a target.

Prediction, on the record: REAL. The record is 3 of 9.

### THE RESULT. NOISE by the registered threshold, and the threshold was set wrong for a paired design

```
   seed  snap  contract  collapse  curvature  max_dev  path_eff
      0   2.5    0.6544         3      1.615    1.410     0.958
      0   3.5    0.6492         3      1.587    1.427     0.959
      0   5.0    0.6481         3      1.601    1.422     0.960
      0   8.0    0.6550         3      1.595    1.420     0.960
      1   2.5    0.6529         4      1.671    1.361     0.956
      1   3.5    0.6499         4      1.668    1.375     0.957
      1   5.0    0.6458         4      1.678    1.364     0.956
      1   8.0    0.6516         4      1.665    1.365     0.956

    snap   mean AUC       sd   paired vs 2.5
     2.5     0.6536   0.0010   +0.0000
     3.5     0.6495   0.0005   -0.0041 sd 0.0015
     5.0     0.6469   0.0017   -0.0067 sd 0.0006
     8.0     0.6533   0.0024   -0.0004 sd 0.0013
```

The registered branch is NOISE and it stands as written. The best arm improves
the contract by 0.0067 against a registered REAL threshold of 0.015.

**The threshold was the wrong one to have registered, and saying so is not the
same as moving it.** The 0.015 came from the replicate noise floor, which is the
spread of INDEPENDENT runs of the same configuration, about 0.006 per run. This
design is paired: the specs, the duration draws and the torch stream are
identical between arms inside a seed, so almost all of that noise is common to
the arms and cancels in the difference. The paired standard deviation of the
5.0 arm's improvement is 0.0006, so 0.0067 is eleven times its own error, and
both seeds improved. Importing an unpaired threshold into a paired design set a
bar roughly twenty times too high.

So the honest statement is two sentences that do not contradict each other. By
the criterion registered in advance this is NOISE and no claim is being made
from it. By the statistics the design actually produced, snap 5.0 is a real
improvement of 0.0067, which is 4.4 percent of the 0.1519 excess, and it is
small.

The shape supports it being real rather than a fluke. The response is not
monotone: 2.5 is worst, 3.5 and 5.0 improve, 8.0 returns to baseline. A noise
sequence has no reason to be single peaked at both seeds, and a threshold knob
with a genuine optimum has every reason to be.

**The mechanism check FAILS, and that is the more interesting half.**
`curvature_mean` moves from 1.643 to 1.640 against the human median, which is
nothing. `max_deviation` moves from 1.386 to 1.393, away from one.
`path_efficiency` moves 0.957 to 0.958. Whatever snap 5.0 is doing for the
contract, it is not correcting the straightness defect the previous section
named, because the straightness ratios barely move at all across a threshold
change that doubles the number of steps being quantised.

That is worth stating plainly: the straightness defect is real, it is large in
ratio terms, and the one serving knob that acts directly on sub pixel angular
jitter does not move it. The defect is not sub pixel jitter.

Prediction was REAL. Wrong. The record is 3 correct of 10.

**Nothing here changes the served recipe.** A 0.0067 improvement at a threshold
that was never tuned for this checkpoint is a candidate, not a decision, and it
would need a third and fourth seed and a rerun of the collapse and duration
checks before anyone changed `EVENT_SNAP`. It is recorded so the next session
starts from it rather than rediscovering it.

## Confirming snap 5.0 at fresh seeds, 2026-08-07

Registered before running. Checked against `RESUME.md`'s DO NOT RE DERIVE list,
which does not contain the snap threshold.

The sweep above says snap 5.0 is worth 0.0067 with a paired standard deviation
of 0.0006 at two seeds, and the registered threshold called that NOISE because
the threshold was imported from an unpaired design. A serving recipe is not
changed on two seeds and a post hoc argument about the threshold, so this runs
the two arms alone at three seeds that have never been used, 2, 3 and 4, and
fixes the criterion in advance for the pooled five seed result.

  CONFIRMED       the improvement has the same sign on all three fresh seeds,
                  AND pooled over all five seeds its mean is at least four times
                  its own paired standard deviation, AND the collapse count does
                  not rise. Snap 5.0 then goes to `RESUME.md` as a proposed
                  recipe change with the evidence attached, for L to decide.
  NOT CONFIRMED   anything else. Snap stays at 2.5, the 0.0067 is recorded as
                  unreproduced, and the thread closes.

The mechanism check is not repeated. The sweep already showed the straightness
ratios do not move, so whatever this knob does it is not the correction it was
proposed as, and a confirmation run cannot change that.

No prediction is registered for this one. The quantity was already measured at
two seeds and predicting it again would be predicting a repeat, not a mechanism.

### The threshold has to come from the design that produced the number

This is a standing note, not a result, and it is the most transferable thing
today produced.

W4 has made both possible errors with the same noise floor.

The replicate floor is about 0.006 of contract AUC. That is the spread of
INDEPENDENT runs of one configuration, and it is the right yardstick for a
number read off a single run against a stored reference. Sections before
2026-08-07 registered NO EFFECT thresholds at 0.005 and then read single
unpaired runs, which puts the bar BELOW the design's own error. That direction
manufactures findings.

Today's paired designs hold the specs, the duration draws and the torch stream
fixed between arms, so the run to run component cancels in the difference and
the paired standard deviation is 0.0006 to 0.003 depending on the arm, five to
ten times smaller. The temperature and snap registrations both imported the
0.015 unpaired bar into a paired design, which puts the bar far ABOVE the
design's own error. That direction discards real effects, and it discarded one:
snap 5.0 at 0.0067 with a paired sd of 0.0006.

Neither registration is being retroactively rewritten. Both verdicts stand as
registered, which is the whole point of registering them. What changes is what
gets registered next time:

    unpaired single run against a stored reference    bar at 0.015, three times
                                                      the 0.006 replicate floor
    paired arms inside a seed, several seeds          bar at four times the
                                                      PAIRED sd measured in the
                                                      same run, computed after
                                                      the run and stated as a
                                                      multiplier before it

The second form is registrable in advance without knowing the number, because
the multiplier is fixed in advance and only the scale is measured. That is what
the snap confirmation registered above does.

A related note on the tickmerge rejection, which was a single unpaired run at
+0.0136, about twice the unpaired floor and therefore not decisive on the AUC
alone. It stands because two independent quantities moved with it, the mean
absolute correlation gap and the leading gap eigenvalue, and because the same
run pushed the angular velocity features back into the top six loadings. The
rejection rests on those, not on the 0.0136.

### THE RESULT. NOT CONFIRMED by the letter, and the letter was mis specified a third time

```
   seed  snap  contract  collapse  curvature  max_dev  path_eff
      2   2.5    0.6460         4      1.496    1.304     0.968
      2   5.0    0.6355         4      1.464    1.318     0.968
      3   2.5    0.6499         2      1.607    1.366     0.958
      3   5.0    0.6391         2      1.592    1.365     0.959
      4   2.5    0.6397         0      1.560    1.241     0.971
      4   5.0    0.6261         0      1.523    1.247     0.973

  all five seeds, paired delta of snap 5.0 against snap 2.5
    seed 0   -0.0063
    seed 1   -0.0071
    seed 2   -0.0104
    seed 3   -0.0107
    seed 4   -0.0137
    pooled mean -0.00965   sd across seeds 0.00299   ratio 3.23

  snap 2.5 across five seeds   0.6544 0.6529 0.6460 0.6499 0.6397
  snap 5.0 across five seeds   0.6481 0.6458 0.6355 0.6391 0.6261
  collapse count, per seed, identical in both arms   3 4 4 2 0
```

The registered criterion was "the same sign on all three fresh seeds, AND pooled
over all five seeds its mean is at least four times its own paired standard
deviation, AND the collapse count does not rise". The first and third pass. The
second reads 3.23 against 4. So the verdict is NOT CONFIRMED and it stands as
registered.

**The criterion was wrong, and this is the third mis specified threshold today.**
"Four times its own paired standard deviation" is not a test of whether a mean
differs from zero. The standard deviation across seeds does not shrink as seeds
are added, so that bar demands the effect exceed four times the seed to seed
VARIATION of the effect, which no amount of further sampling can help with. The
quantity that answers the question is the standard error of the mean, 0.00299
over the square root of five, which is 0.00134, and the ratio to it is 7.2. A
sign test on five of five in the same direction gives 1 in 32.

So the honest pair of sentences, again. By the criterion registered in advance,
NOT CONFIRMED. By every conventional statistic, snap 5.0 improves the contract
by 0.0097 with a standard error of 0.0013, on five of five seeds, with the
collapse count identical in both arms at every seed.

Today's three registrations set the bar from the wrong error term three
different ways: an unpaired floor imported into a paired design, twice, and now
a standard deviation used where a standard error was needed. The methodology
note above was written before this run and did not prevent this one, because it
fixed the first error and not the second. Its rule is amended: state the bar as
a multiple of the standard error OF THE STATISTIC BEING TESTED, name which
statistic that is, and check that adding samples can move it.

**What this does and does not license.** It does not license changing
`EVENT_SNAP`. The served value is 2.5 and it stays 2.5 until L decides
otherwise, because a recipe change is L's call and because the registered
verdict is NOT CONFIRMED. What it licenses is putting a specific proposal in
front of L with its evidence: snap 5.0 is worth about 0.010 of contract AUC, it
is the largest serving side gain found since the two configuration corrections
this morning, it costs nothing at generation time, and it does not touch the
model.

**And the mechanism check fails again, harder.** Across five seeds the curvature
ratio moves 1.554 to 1.526, maximum deviation moves away from one, and path
efficiency moves 0.001. The knob doubles the number of steps quantised to the
lattice and the straightness ratios are unmoved. Whatever snap 5.0 buys, it is
not straightness, and the straightness defect named earlier today has now
survived a direct attack by the only serving knob that acts on it.

## Locating the snap optimum, 2026-08-07

Registered before running, with the threshold rule as amended two sections above.

Snap 2.5 is the served value, 3.5 is worth 0.0041, 5.0 is worth 0.0097 over five
seeds, and 8.0 is worth nothing. The response is single peaked and the peak has
been sampled at exactly one point. If a recipe change is going to be put to L it
should be put at the right value, so this samples the interior.

Arms 2.5, 4.0, 5.0 and 6.5, three seeds never used before, 5, 6 and 7, paired
inside a seed as before, one trajectory per spec, nothing selected.

The criterion, stated as a multiple of the standard error of the statistic being
tested, which is the amendment the last section forced.

  BETTER THAN 5.0   an arm whose paired mean against 5.0 is an improvement of
                    more than three times the standard error of that mean, which
                    with three seeds is the seed to seed sd divided by the square
                    root of three. Adding seeds shrinks that bar, which is the
                    property the previous criterion lacked.
  FLAT NEAR 5.0     no arm clears it. The optimum is a plateau, 5.0 is as good a
                    point on it as any, and the proposal to L is unchanged.

The collapse guard stands: any arm whose collapse count rises above the 2.5 arm's
at the same seed is disqualified regardless of AUC.

Prediction, on the record: FLAT NEAR 5.0. The record is 3 correct of 10, and
this prediction is deliberately the boring one because the last two times a
mechanism was predicted here the measurement refused it.

### THE RESULT. FLAT NEAR 5.0, and the pooled eight seed number

```
   seed  snap  contract   seed  snap  contract
      5   2.5    0.6338      6   2.5    0.6433
      5   4.0    0.6148      6   4.0    0.6352
      5   5.0    0.6240      6   5.0    0.6268
      5   6.5    0.6248      6   6.5    0.6351
      7   2.5    0.6484
      7   4.0    0.6433
      7   5.0    0.6467
      7   6.5    0.6418

    snap   mean AUC   paired vs 2.5
     2.5     0.6418   +0.0000
     4.0     0.6311   -0.0107 sd 0.0073
     5.0     0.6325   -0.0093 sd 0.0074
     6.5     0.6339   -0.0079 sd 0.0012

  against 5.0, three seeds, bar three times the standard error of the mean
    4.0   -0.0092  +0.0084  -0.0034   mean -0.0014  bar 0.0156   not better
    6.5   +0.0008  +0.0083  -0.0049   mean +0.0014  bar 0.0115   not better
```

FLAT NEAR 5.0. Neither interior arm beats 5.0 and both have deltas that change
sign across seeds. The prediction was FLAT NEAR 5.0 and it is correct, which
makes the record 4 of 11.

The proposal to L is unchanged and now rests on eight seeds rather than five.

```
  snap 2.5 to snap 5.0, every seed run today
    -0.0063  -0.0071  -0.0104  -0.0107  -0.0137  -0.0098  -0.0165  -0.0017
    mean -0.00953   sd 0.00457   standard error 0.00161   ratio 5.90
    eight of eight seeds improved
```

Two things in that table are worth reading beyond the mean.

The 2.5 baseline itself has a standard deviation of 0.0069 across the eight
seeds, which reproduces the replicate noise floor a fourth time and is why a
single run of anything is worthless here. It also means the served number is
better quoted as 0.6449 plus or minus 0.0069 than as any of the individual
figures this file has quoted for it.

And the plateau has an edge. 4.0, 5.0 and 6.5 are worth 0.008 to 0.011 and are
indistinguishable from each other, while 8.0 at seeds 0 and 1 was worth 0.0004.
Something changes between 6.5 and 8.0. That is not being chased, but it is a
fact about the knob and it is recorded because the next person to touch it will
otherwise assume the response is smooth.

## Why the snap threshold helps, since it is not straightness, 2026-08-07

Registered before running. Checked against DO NOT RE DERIVE: the list holds the
hand built lattice emitter, which is a BUILD, and the dt quantisation confound,
which is about timing. Asking what an already measured serving knob is doing is
neither.

Eight seeds say snap 5.0 is worth 0.0095 and that the curvature and deviation
ratios do not move. So the route is unknown, and a 0.0095 improvement by an
unknown route is worth exactly one run to identify, because if the route
generalises the knob is not the end of it.

Two readouts on the same pair of generated samples, snap 2.5 and snap 5.0,
identical specs, duration draws and torch stream.

  1. In the contract's own coordinates. For each of the eighteen features, the
     change in mean from 2.5 to 5.0, divided by the human standard deviation of
     that feature, so the eighteen numbers are comparable. This says WHICH
     features the knob moves, in the units the detector reads.

  2. In path space, on the 125 Hz grid the feature extractor uses. The human
     pool is stored as integers, so human micro movements land exactly on the
     pixel lattice and the model's do not. Reported for human, snap 2.5 and snap
     5.0: the distribution of per step displacement magnitude at the low end,
     the share of steps under one pixel, and the share of steps whose x and y
     displacements are both within 0.05 of an integer.

Branches for readout 1, thresholds fixed now.

  NAMED ROUTE   the three largest standardised shifts account for more than 60
                percent of the total absolute standardised movement across all
                eighteen. The knob acts on a nameable part of the feature space
                and the next question is whether anything else reaches it.
  DIFFUSE       they do not. The knob moves everything slightly, like every
                other real effect in W4, and there is nothing further to name.

Readout 2 is reported and not branched on, because no threshold for a lattice
occupancy share was defensible before seeing one.

Prediction, on the record: NAMED ROUTE, concentrated in the acceleration and
jerk features, because snap only touches steps below the threshold speed and
those are the features that second and third differences of slow motion feed.
The record is 4 of 11.

### THE RESULT. NAMED ROUTE, the corner is the angular channel and not the one predicted

Registered branch met. The three largest standardised shifts carry 84.1 percent
of the total absolute movement across all eighteen, against a 60 percent bar.
The branch called NAMED ROUTE is the verdict.

The named corner is not the predicted one. The prediction on the record was
acceleration and jerk. Those four features moved by 0.0001 human standard
deviations or less, which is nothing. The entire effect is angular:

```
  angular_velocity_mean        -0.0292 sd     gap 0.3244 -> 0.2953   toward
  angular_velocity_std         -0.0139 sd     gap 0.2105 -> 0.1966   toward
  num_direction_changes        -0.0079 sd     gap 0.1039 -> 0.0960   toward
  time_to_peak_velocity        +0.0039 sd                              away
  everything else              below 0.0025 sd, most of it below 0.0005
```

The three movers are the three largest, they are all in the same channel, and
all three move toward the human mean. Ten of the eighteen technically moved
away, but the largest of those ten is 0.0039 human standard deviations, so the
count is noise around zero and the direction statement belongs to the three.

Why this is the right mechanism after all, stated plainly. `_SNAP` rounds `dx`
and `dy` to whole lattice units for slow steps. Rounding a vector changes its
DIRECTION, not just its length, and the integrated heading is left continuous,
so the realised turn at every snapped step is displaced from the turn the model
asked for. Angular velocity is heading change over time and direction changes
are sign flips of it. Speed, acceleration and jerk are magnitudes, and rounding
a sub 5 pixel step barely moves a magnitude. The prediction had the arithmetic
of the operation right and its geometry backwards. Record: 5 of 12 on branches,
and the mechanism half of this one is wrong.

So the knob's whole action is: it perturbs the headings of slow steps toward
lattice realisable directions, which is what human slow steps actually do, and
buys about 0.008 of contract AUC for it. Curvature does not move, which is the
2026-08-07 finding this run was registered to explain, and now it is explained:
the knob was never touching curvature, it was touching angular velocity, and the
sweep's mechanism check asked the wrong three features.

### The ninth seed disagrees with the other eight, and the proposal to L changes

This run is a fresh paired seed of the same comparison, so it is a ninth
observation whether it was meant as one or not. It went the other way.

```
  snap 2.5   0.6328
  snap 5.0   0.6384      +0.0056, the first seed of nine where 5.0 is worse
```

The nine paired deltas are now -0.0063, -0.0071, -0.0104, -0.0107, -0.0137,
-0.0098, -0.0165, -0.0017 and +0.0056.

```
                   eight seeds        nine seeds
  mean              -0.00953           -0.00784
  sd                 0.00457            0.00661
  standard error     0.00161            0.00220
  mean / se          5.90               3.56
  improved           8 of 8             8 of 9
```

The effect survives at 3.6 standard errors, which still clears the four times
standard error bar this file adopted this afternoon only if it is read loosely,
and does not clear it if it is read strictly. It does not survive as the clean
unanimous result the earlier write up described. **The proposal to L is amended
to about 0.008, at 3.6 standard errors, improving on 8 of 9 paired seeds, with
one seed reversing.** Nobody should read the 5.90 again.

There is a lesson here that is not about snap. Eight paired seeds all agreeing
looked like a settled result this morning. One more seed cut the effect size by
18 percent and the significance ratio by 40 percent. The paired design removes
the run to run component but it does not remove the seed to seed component, and
this file has now seen the seed to seed component be larger than it looked twice
in one day.

### Path space says the knob is invisible where it acts

Readout 2, reported and not branched.

```
  sample      grid          <1 px   lattice     p10     p25     p50     p75     p90
  human       125 Hz        0.353     0.272   0.078   0.500   2.041   6.773  16.199
  snap 2.5    event grid    0.072     1.000   1.000   1.414   3.162  10.000  28.178
  snap 2.5    125 Hz        0.311     0.203   0.149   0.729   2.263   7.280  17.433
  snap 5.0    event grid    0.072     1.000   1.000   1.414   3.162  10.000  28.160
  snap 5.0    125 Hz        0.311     0.204   0.149   0.730   2.286   7.280  17.431
```

Read the 125 Hz rows only. The human paths reconstructed by `phase0_critic` are
already on the 125 Hz grid, so their raw and resampled rows are identical and
the event grid rows have no human counterpart to compare against.

Two things fall out. First, on the grid the extractor reads, the model is close
to the human on this statistic and always has been: 31.1 percent of its steps
are sub pixel against a human 35.3, and 20.3 percent land on the lattice against
a human 27.2. This is not a defect anyone needs to chase. Second, and this is
the point of the readout, snap 2.5 and snap 5.0 differ in the FOURTH decimal on
every one of these numbers. Doubling the threshold doubles the number of steps
that get rounded and leaves the path space step distribution unmoved.

That is consistent with the feature space result rather than in tension with it.
Rounding a slow step changes its direction by a fraction of a radian and its
length by a fraction of a pixel. A displacement magnitude histogram cannot see
that. Angular velocity, which is the derivative of the thing that moved, can.

### What this closes and what it opens

Closed: the question of how snap helps. It helps through the angular channel,
by a mechanism that is now named, and the size of the help is about 0.008 with
one seed in nine reversing it.

Opened, and this is the more useful half: the largest standardised mean gaps
left in the served configuration are angular_velocity_mean at 0.32 human
standard deviations, curvature_mean at 0.32, curvature_std at 0.30,
angular_velocity_std at 0.21 and path_efficiency at 0.19. Every other feature is
inside 0.15 and eleven of the eighteen are inside 0.05. The gap that is left in
the FIRST MOMENT is a shape gap, concentrated in the two channels that describe
how a path turns. This does not contradict `w4_featmap`'s DIFFUSE verdict, which
was about which features a classifier can exploit; it is a statement about where
the means differ, and the two questions have different answers.

## Feature conditioning the event AR trunk, registered 2026-08-07 before any training

This is an architecture change, not a knob, so it is registered at more length
than the diagnostics above.

### The argument

The trunk is told four numbers: how far, how long, and in which direction.
Everything else about the movement has to be invented by accumulating up to 256
correct per event conditionals. Cross entropy cannot help it do that. A per step
error too small for the loss to notice becomes a large error in a whole path
aggregate, and the eighteen contract features are whole path aggregates. The
trunk is being asked to get a global property right through a purely local
objective. Feature conditioning removes the ask: the character becomes an input.

If the model obeyed the command exactly and the commands were drawn from the
right distribution, the generated feature sample would BE that distribution and
the contract would read whatever that distribution reads. So the whole idea has
a ceiling, and the ceiling was measured today rather than assumed.

### The ceiling, measured

Take a random 4000 rows of the human training event corpus, push each row's own
tokens through the SERVING decoder at `EVENT_SNAP=2.5`, extract the eighteen
features, and score against `data/human_val_features_grpo.npy`:

```
  decoded human tokens vs human validation      0.5163   collapse 6
  the model as served today                     0.6449   collapse 0 to 4
```

0.5163 reproduces the 0.5118 already in this file for the serving decoder on
real human tokens, from an independent path, which is worth having. Two things
follow. First, the ceiling of feature conditioning is about 0.516 and the gap
from the served number to it is 0.129, which is the entire remaining problem.
Second, the collapse flag fires SIX times on genuinely human tokens, which is
the fourth independent demonstration that the flag does not measure what its
name says.

A control that matters: the first 8000 rows of the corpus give wildly different
feature medians from the validation human, curvature by four orders of
magnitude. The corpus is ordered by session. Any measurement on a prefix of it
is a measurement of one person's mouse. Sample it randomly, always.

### Why this is not the closed feature conditioning work

The 2026-07-27 section of this file prices the FLOW family's feature
conditioning at a mean commanded to realized correlation of 0.41 and calls that
a complete account of its 0.70. It also separates two different failures:
angular_velocity_mean had full human spread and 0.51 adherence, a STEERING
failure, while curvature had 0.31 adherence and one tenth of the human spread, a
CAPABILITY failure, and it says explicitly that guidance could not steer
curvature because there was nothing there to steer.

The AR trunk does not have the capability failure. `research/w4_disp.py` and the
percentile table above put its curvature at about 1.6 times the human median and
4 times at the 95th percentile. It has too much curvature variety, not too
little. The steering question is open on this trunk and has never been asked.

That same 2026-07-27 section ends by saying the only emitter left that could
pass is a learned autoregressive model over integer steps, which is what this
trunk is. Feature conditioning and that emitter have never been in the same
model.

### The build

`training/prepare_event_features.py` labels every corpus row with the eighteen
features of the path the SERVING DECODER renders from that row's own tokens, not
of the original polled path. Commanding a feature vector the decoder cannot
render is commanding disobedience, and it is also what makes the ceiling above
the right ceiling.

`training/train_event_ar_featcond.py` widens `cond_dim` from 4 to 23: four
geometry numbers, eighteen features, and one flag saying whether the eighteen
are present. Each feature goes through its own empirical quantile function to a
standard normal, with the knots stored in the checkpoint, because the eighteen
span five orders of magnitude and two of them would otherwise own the whole
conditioning vector. The new columns of the condition projection are initialised
to EXACTLY ZERO and every other weight is the parent's, so step zero of the fine
tune reproduces `event_ar_v2_s40000.pt` bit for bit on a flag zero batch.
Dropout on the feature block is 0.1, which keeps an unconditioned mode alive and
leaves classifier free guidance reachable. Guidance is not part of the plan;
this file already prices it as harmful on the flow model.

At serving, a command is drawn from the empirical joint CONDITIONAL on the spec:
nearest neighbours in (log distance, log duration) among labelled training rows,
one chosen uniformly from the k nearest. Drawing from the unconditional joint
would command durations that contradict the commanded duration. One trajectory
per spec, nothing generated twice, nothing selected.

### Readouts and branches, fixed before the run

Three fresh rng seeds, 1500 specs each, the base model and the feature
conditioned model on identical specs and identical duration draws.

Readout 1, the contract. Mean AUC of the feature conditioned model minus mean
AUC of the base model across the three seeds. The statistic being tested is that
difference of means and its standard error is the seed to seed standard
deviation over root three, so adding seeds can move it.

Readout 2, obedience. Spearman correlation between commanded and realised, per
feature, averaged over the eighteen. The comparison number is the flow family's
0.41.

Readout 3, the control. The same model with the flag set to zero and the
eighteen zeroed. This must land within the replicate noise floor of the base
model. If it does not, the fine tune moved the model for reasons that have
nothing to do with conditioning and readout 1 is confounded, in which case no
branch below may be claimed.

```
  WORKS                  readout 1 improves by more than 0.030 AND readout 2
                         exceeds 0.70.
  OBEYS BUT DOES NOT PAY readout 2 exceeds 0.70 and readout 1 improves by 0.030
                         or less. This is the interesting failure: it would say
                         per feature obedience is not the same thing as joint
                         obedience, and the forest lives in the joint.
  DOES NOT OBEY          readout 2 is 0.70 or below. Then the steering failure
                         is a property of feature conditioning in this repo and
                         not of the flow model's renderer, and the whole family
                         is closed rather than half closed.
```

The 0.030 bar is 23 percent of the 0.129 available, and it is far above four
standard errors at any paired spread this file has measured, which is between
0.0006 and 0.007. It is set high on purpose: a fine tune this cheap producing
less than a quarter of the available gap is not worth building a program on.

Prediction, on the record: WORKS. The capability failure that killed the flow
version is measurably absent here, the conditioning already reaches every layer
through `cond_embed`, and the label is exactly the quantity being scored. The
record is 5 of 12 and this file's own standing advice is not to trust the guess.

### Result, 2026-08-09. CONFOUNDED both times, and the prediction was wrong.

The record is now 5 of 13.

Two fine tunes were run from `event_ar_v2_s40000.pt`, both 10,000 steps, both
initialised so that step 0 reproduces the parent bit for bit. The first used the
registered feature dropout of 0.1. The second was not in the plan and is
explained below.

```
  feat dropout 0.1   base 0.6443   fc 0.6179   fc_off 0.7145
                     readout 1 -0.0264, se 0.0087, 3/3 seeds
                     readout 3 +0.0702        readout 2 mean 0.788

  feat dropout 0.5   base 0.6443   fc 0.6287   fc_off 0.6525
                     readout 1 -0.0156, se 0.0106, 2/3 seeds
                     readout 3 +0.0082        readout 2 mean 0.746
```

The first control failed by a factor of ten. That is not a fact about
conditioning, it is a fact about the experiment: at dropout 0.1 the flag zero
path is trained on a tenth of the data the flag one path sees, so `fc_off` is
not a control at all, it is an undertrained mode of the same network. The
registered design was wrong here and the control caught it, which is the only
reason the 0.0264 was not written up as a win.

The rerun at dropout 0.5 trains both paths equally and the control tightens by a
factor of 8.5, to 0.0082 against a 0.0069 floor. Still outside, so the verdict
stands at CONFOUNDED and no branch may be claimed, but the residual is now
1.2 floors rather than 10 and is plausibly the cost of the fine tune itself
rather than a defect in the arm.

Reading past the verdict, because the numbers are informative even where they
are not claimable. The confound free contrast is within one set of weights:
flag on 0.6287 against flag off 0.6525, so switching the eighteen on is worth
0.0238. Against the base model the same switch is worth 0.0156 at se 0.0106,
because roughly 0.008 is spent on the fine tune. That is 1.5 standard errors,
and it is half the registered 0.030 bar. Conditioning does something real and
does not do enough, and no seed count fixes a 0.0156 when the distance to 0.50
is 0.144.

Readout 2 is where the result earns its keep. Obedience is not uniform, it is
ordered, and the order is the same one this file has now found three times:

```
  movement_duration 0.998   mean_velocity 0.971   std_velocity 0.941
  ...
  angular_velocity_mean 0.596   curvature_std 0.534   curvature_mean 0.527
  mean_jerk 0.289
```

The model obeys the coarse geometry of the path almost perfectly and cannot obey
the fine texture at all. Compare that ordering against the feature attribution
of 2026-08-05 and against the snap route of 2026-08-08: the features the trunk
cannot be steered on are the features carrying the separation. Three independent
methods, one conclusion. The residual is not in the shape of the path, it is in
the moment to moment texture, and every arm in this file so far has been
steering the shape.

That closes feature conditioning as a lever and reframes the next one. Do not
run more seeds here.

Artifacts: `research/w4_featcond.json`, `research/w4_featcond50.json`,
`research/w4_fc_train.log`, `research/w4_fc50_train.log`,
`training/event_ar_fc.pt`, `training/event_ar_fc50.pt`.

## Attributing the gap to a token channel, 2026-08-09

Registered before `research/w4_channels.py` was run.

Three independent methods now agree the residual lives in fine motion texture:
the feature attribution of 2026-08-05, the snap route of 2026-08-08, and the
conditioning obedience table above. None of them says WHERE in the emitted
sequence the texture goes wrong, and every arm so far has been designed on a
guess about that. The record is 5 of 13. Measure first.

The decoder consumes three channels: a speed class, a turn class, and a
millisecond timing class. Take a random corpus row, generate one trajectory on
that row's own four dimensional command, truncate both to the common length,
and decode all eight ways of drawing each channel from either source. Two of
the eight are the ceiling and the served number. The other six partition the
gap between the channels.

Nothing is trained, nothing is selected, one trajectory per command. n is 2500
rows before losses, and the eight arms are paired on the identical surviving
rows so the contrasts carry no between sample variance at all.

Readout 1, the eight AUCs. Readout 2, the main effect of each channel, meaning
the mean change from switching that channel from human to generated averaged
over the four settings of the other two. Readout 3, the joint term, the part of
the gap no sum of main effects accounts for.

The control. The all human arm must reproduce the measured ceiling of 0.5163 to
within 0.045. It is the same quantity by a third route, and if truncation to
the common length distorts anything it distorts that first. Outside the band the
run is VOID and nothing may be read from it.

```
  SINGLE CHANNEL   one channel takes more than 60 percent of the gap and
                   neither other reaches 20. The defect is localised and the
                   next arm targets that channel alone.
  TWO CHANNEL      the top two take more than 75 percent between them and each
                   takes more than 25.
  DISTRIBUTED      no channel reaches 60 and no pair reaches 75. Then the gap
                   lives in the structure ACROSS channels, the speed to turn
                   coupling above all, and no arm that repairs a single channel
                   can close it. This is the expensive outcome and it is the
                   one worth knowing early.
  VOID             the all human arm falls outside the ceiling band.
```

One repair is applied to all eight arms including the pure two. A zero length
step carries the null turn class and a moving step carries a lattice class, so
a mixed arm can otherwise pair a moving speed with a null turn, which is not a
sequence either source could emit. A moving step whose turn is null takes the
zero turn class and a still step takes null. On the pure arms this is the
identity, and the script asserts it rather than assuming it.

Prediction, on the record: DISTRIBUTED, with turn the largest single main
effect. The reasoning is that the contract reads sub pixel texture, which
`w4_texture_sensitivity` prices at 0.507 to 0.864 for half a pixel of added
noise, and that speed and turn are physically coupled in human movement by the
two thirds power law, so a model can have both marginals close and still put
the joint in the wrong place. The standing advice in this file is not to trust
the guess.

### Result, 2026-08-09. VOID, and the design is retired rather than repaired.

The record is 5 of 14. The prediction named DISTRIBUTED and cannot be scored,
because the control refused the run.

```
    speed   turn   time   contract
    human  human  human     0.5935     <- control, reference 0.5163
    human  human    gen     0.7125
    human    gen  human     0.6918
    human    gen    gen     0.7691
      gen  human  human     0.7398
      gen  human    gen     0.7025
      gen    gen  human     0.7408
      gen    gen    gen     0.6782
```

The all human arm reads 0.5935 against the 0.5163 measured twice before. The
0.077 of contamination is nine tenths of the 0.0847 gap the arm was built to
partition, so the attribution underneath it is unreadable and is not reported
as a number here. Truncating a human trajectory to the common length cuts its
tail off, and duration, time to peak velocity, path efficiency and max
deviation are all functions of that tail. 2311 of 2500 rows survived, so this
is not a sample size effect, it is the intervention.

The repair is obvious and the design still should not be rerun, for a second
reason visible in the same table. EVERY one of the six mixed arms scores worse
than BOTH pure arms, the worst of them at 0.7691 against a 0.6782 served and a
0.5935 ceiling. A factorial design assumes the intervention is neutral and that
switching a channel reveals what that channel contributes. Here switching any
channel produces an object neither source could emit, and the cost of breaking
the coupling exceeds the whole quantity being measured. No length fix touches
that. The design cannot answer the question it was built for and is retired.

What it does establish, weakly but for free: the three channels are strongly
coupled, since every hybrid is less human than either parent. That is evidence
in the direction the prediction named, and it is not evidence of any particular
channel's share.

The lesson for the next design is to stop constructing sequences and start
restricting views. A discriminator shown only part of a real sequence from each
source never sees an object that could not exist, and the same question, which
aspect of the emitted stream is detectably not human, is answered without any
frankenstein arm. That is the next arm.

Artifacts: `research/w4_channels.py`, `research/w4_channels.json`,
`research/w4_channels.log`.

## Restricted views of the token stream, 2026-08-09

Registered before `research/w4_views.py` was run.

`w4_channels` failed because it built sequences. This builds none. Each side
keeps its own real trajectories and the discriminator is shown less of them:
only the speed marginal, only the turn autocorrelation, only the speed to turn
coupling, and so on. Nothing that could not exist is ever scored.

Take 3000 random corpus rows, generate one trajectory each on that row's own
command, and convert both sides to pixels, radians and milliseconds through
`class_to_speed`, `class_to_dtheta` and `class_to_dt_ms`, so that no statistic
can read a difference of units. Then run the contract's own random forest recipe
on nine restricted views, in three families:

```
  marginal   speed_marg  turn_marg  dt_marg     per event distributions
  temporal   speed_acf   turn_acf   dt_acf      autocorrelation, lags 1 to 8
  coupling   couple_st   couple_sd  couple_td   cross correlation, lags -3 to 3,
                                                plus the two thirds power law
                                                slope, intercept and residual
```

The coarse control. Human and generated trajectories differ in event count and
total duration for reasons that are not texture, and nearly every statistic
above inherits some of that. One view holds only event count, total duration and
total path length. The PRIMARY readout for every other view is the AUC of that
view together with the coarse one, minus the AUC of the coarse one alone. A view
that adds nothing over knowing how big the movement was scores zero however well
it separates by itself.

```
  MARGINAL   the leading view is a marginal one and beats the best view of both
             other families by more than 0.030.
  TEMPORAL   the leader is an autocorrelation view, same margin.
  COUPLING   the leader is a cross channel view, same margin. Then the joint
             structure is the defect, which is what the retired w4_channels
             suggested for free, and the next arm has to model the coupling
             rather than any one channel.
  MIXED      a leader that does not clear both rival families by 0.030. Then no
             family owns the separation and an arm aimed at one is a guess.
  NULL       no view beats the coarse control by 0.030 at all. Then what the
             model emits is not detectably not human in any view here, and the
             separation is entering at decode or resample instead. That would
             be the most surprising and the most useful outcome, because every
             arm in this file has assumed the emitter is at fault.
```

The 0.030 margin is about four times the unpaired replicate spread of 0.0069
this file measures.

Two readouts come free. All views together is an estimate of how detectable the
token stream is in total. The contract AUC on the SAME generated rows, and on
the same human rows, is computed through `scoring`. If all views together lands
well below the contract number, the decoder is amplifying a difference that is
small in the tokens, and that is a different problem from the one being hunted.

This is a diagnostic on token statistics, not the contract. The recipe is copied
exactly so the numbers are the same kind of object, but no decision about the
deliverable is taken from any of them.

Prediction, on the record: COUPLING, with `couple_st` leading. The reasoning is
that human speed and turning are locked by the two thirds power law, that every
hybrid in the retired arm was worse than both its parents, which is what strong
coupling looks like, and that a next token model trained on cross entropy is
scored mostly on marginals and can put a joint in the wrong place at very little
cost in nats. The standing advice in this file is not to trust the guess, and
the record is 5 of 14.

### Amendment before the real run, 2026-08-09

The registration above is amended in one place, and the amendment was made
AFTER seeing numbers, so it is recorded here rather than edited into the text
above.

A 1200 row smoke run showed the coarse view alone reading 0.403. That is not a
weak signal, it is six standard errors below chance, which is what a random
forest read out of bag does when its features carry nothing. Generation is
conditioned on each row's own distance and duration and duration obedience is
0.998, so event count and total duration genuinely match and there was never a
size confound to subtract. Scoring every view by what it adds on top of 0.403
credited each of them with about a tenth of an AUC for nothing.

The readout becomes the split half already used elsewhere in this file. Half the
human rows against the other half is the same test with nothing to find. Half
the human rows against their own paired generated rows is the test. Sample sizes
are identical so the null level, below chance artifact included, is common to
both, and the excess of the second over the first is the readout. The coarse
view stays in the table as a control and is no longer subtracted from anything.

The branch names, the families and the 0.030 margin are unchanged, and the
prediction of COUPLING with `couple_st` leading stands. For the record, the
smoke run's raw per view numbers put `turn_acf` on top at 0.5696 with the three
coupling views between 0.523 and 0.532, which is evidence against that
prediction before the real run has started. It is left standing rather than
quietly revised.

n rises to 4000 rows and the minimum length falls to 16 events, because the
split half halves the sample available to each test.

### Result, 2026-08-09. MIXED, the prediction was wrong again, and the two
### lines at the bottom of the table are the most useful thing measured today.

The record is 5 of 15. COUPLING was predicted with `couple_st` leading.
`couple_st` came LAST of the nine at +0.0183.

```
          view    family    floor   signal    excess
        coarse   control   0.5260   0.4407   -0.0853
    speed_marg  marginal   0.4843   0.5448    0.0605
     turn_marg  marginal   0.4732   0.5460    0.0727
       dt_marg  marginal   0.4807   0.5289    0.0482
     speed_acf  temporal   0.4857   0.5418    0.0561
      turn_acf  temporal   0.4940   0.5401    0.0462
        dt_acf  temporal   0.4905   0.5234    0.0329
     couple_st  coupling   0.5152   0.5335    0.0183
     couple_sd  coupling   0.4881   0.5559    0.0678
     couple_td  coupling   0.5070   0.5498    0.0427

    every view             0.5095   0.6201    0.1107
   contract 18             0.5050   0.6139    0.1089
```

3052 rows survived, 1526 per class after the split half. The floors sit between
0.473 and 0.526 against the 0.467 to 0.497 this file records elsewhere, so the
null is behaving.

Two things, and the second is worth more than the verdict.

First, nothing leads. The best view takes 0.0727 of an available 0.1107 and
five others are within 0.03 of it, so no family owns the separation and an arm
aimed at any one of them is a guess. The nine excesses sum to 0.4454 against a
joint 0.1107, so they are largely reading the same thing from different angles.
That is what a globally slightly wrong distribution looks like, not a defect
with a location. It is also the fourth arm in a row to say the same thing in a
different vocabulary.

Second, every view together reads 0.6201 with an excess of 0.1107, and the
eighteen contract features on the SAME two samples read 0.6139 with an excess of
0.1089. Those are the same number. Two consequences, both of which close
questions this file has been carrying open.

  1. The decoder is not amplifying anything. Whatever separates the model from
     a human is already fully present in the token stream it emits, and the
     snap, the rounding and the resample neither create it nor add to it. The
     hunt belongs in the emitter, which every arm has assumed without checking.
  2. Nine blocks of hand written summary statistics capture essentially all of
     the detectable difference that the contract's own eighteen features do.
     There is no hidden structure that the contract sees and this file's
     descriptive vocabulary misses.

What that does NOT establish is the ceiling on detectability. 0.62 is what a
random forest over hand chosen statistics can find. A learned critic reading the
raw token sequence could find much more, and the gap between those two numbers
decides the next arm. If a critic trained on the sequences reads about 0.62 the
model's distribution is close and the remaining work is calibration. If it reads
0.9 the model is badly wrong in a way no summary statistic here can see, and
descriptive work should stop. That measurement is cheap and is the next arm.

Artifacts: `research/w4_views.py`, `research/w4_views.json`,
`research/w4_views.log`.

## The learned critic bound, 2026-08-09

Registered before `research/w4_critic.py` was run.

`w4_views` established that hand chosen statistics read 0.6201 on the token
stream and that the eighteen contract features read 0.6139 on the same two
samples. What neither says is how much a discriminator that chooses its own
statistics would find. That number bounds from above everything any descriptive
arm in this file can ever reach, and it decides whether the next arm is
calibration or a new objective.

A small bidirectional transformer, 192 wide, four layers, four heads, reads the
three raw class streams of a trajectory plus the four dimensional command it was
made for, and says human or model. 24,000 random corpus rows, one generated
trajectory per row on that row's own command, held out split by trajectory made
before any training, eight epochs, the best held out AUC reported.

The floor, and it is not optional. The identical critic with identical
hyperparameters and identical sample sizes, trained to separate one half of the
human rows from the other half. There is nothing there to find. Whatever it
reads above 0.5 is capacity turning into memorisation, and the headline number
means nothing without it. `w4_views` needed the same control and the coarse view
it replaced was six standard errors below chance, which is a reminder that the
null in this repo is not where intuition puts it.

Both critics are conditional, the command is an input. A critic denied the
command could separate on the marginal mix of commands and would be answering a
different question.

```
  CLOSE    the critic reads 0.68 or less. It finds little more than the
           statistics did, the model's conditional sequence distribution is
           genuinely near the human one, and the remaining work is calibration
           rather than a new training objective.
  WIDE     the critic reads 0.85 or more. The model is wrong in ways nothing in
           this file's descriptive vocabulary can see. Descriptive work stops
           and the next arm trains against a learned signal.
  MIDDLE   anything between. Real hidden structure, not a great deal of it,
           neither branch claimed.
```

Prediction, on the record: WIDE. The reasoning is not that discriminators
usually win, it is the nats. This file measured an optimistic 0.0064 nat
improvement predicting a 0.0012 AUC move, so the model's errors sit where cross
entropy barely charges for them, which is exactly the low probability region a
discriminator finds cheapest to exploit. A critic also sees all 256 events
jointly where the nine views compress to about ninety numbers. The record is 5
of 15 and the last prediction put its named leader last of nine.

### Amendment before the real run, 2026-08-09

Amended after seeing smoke numbers, so recorded here rather than edited above.

A 7000 row smoke run at two epochs read a FLOOR of 0.8280 against a reading of
0.7191. A floor above the thing it is the floor for is not a result, it is a
broken control, and it was broken in the way this file already warns about. The
floor split the human rows into the first half and the second half of a sorted
sample. The corpus is ordered by session, so those two halves are different
people, and the critic was separating whose mouse it was.

The fix. The pool is twice the per class count, and the A and B halves are
assigned by a random permutation, never by position. Generation runs on the A
half only. The reading is A human against A generated, the floor is A human
against B human, and both have the same number of rows per class, which the
smoke version also did not.

One threshold is added that should have been there from the start: if the floor
reads above 0.60 the run is VOID and no reading may be attributed to the model.
CLOSE, WIDE and MIDDLE are unchanged and still judged on the reading itself. The
prediction of WIDE stands.

The smoke reading of 0.7191 is not evidence for anything, because it came from
two epochs on a seventh of the data with a broken companion control, and it is
recorded only so that the real number cannot later be presented as if nothing
had been seen before it.

### Result, 2026-08-10. MIDDLE. The prediction was wrong, and the number is
### more useful than either branch would have been.

The record is 5 of 16. WIDE was predicted, 0.85 or more. It read 0.751.

```
  learned critic, human against its own paired generated rows   0.7509
  floor, the A half of the human pool against the B half        0.5235
  excess                                                       +0.2273

  for comparison, w4_views on the same kind of samples
  nine blocks of hand written statistics                        0.6201
  floor                                                         0.5095
  excess                                                       +0.1107
```

Run twice, once live and once from the cached generation after a crash on the
last line of the file, reading 0.7518 and 0.7509 with floors of 0.5194 and
0.5235. The floor is where a floor should be, so the control that was broken in
the smoke run is fixed and the reading stands.

The comparison in the second block is the result. A critic that chooses what to
look at finds an excess of 0.2273 where every summary statistic in this file put
together finds 0.1107. It finds slightly more than twice as much. Three things
follow.

  1. Descriptive arms are capped. Anything designed by looking at summary
     statistics is working with half the available signal at best, and the half
     it cannot see is not a residue, it is the larger part. Ten arms in this
     file were designed that way. Stop.
  2. The model is not wildly wrong. 0.751 is a long way from the 0.9 that would
     have said the sequence distribution is broken in some gross way. It is
     wrong in a diffuse, structured manner that a learned function can find and
     a moment cannot.
  3. The critic is a stronger discriminator than the contract. It reads 0.751
     where the contract reads about 0.644 on the served model. Anything trained
     to defeat it is being held to a harder standard than the deliverable, which
     is the right direction for the error to point.

Being wrong in the WIDE direction matters for what comes next. Had it read 0.9
the honest conclusion would have been that this trunk cannot be repaired and the
architecture has to change. At 0.751 with a 0.52 floor the distribution is close
enough that a training signal aimed at the critic has somewhere to go.

Artifacts: `research/w4_critic.py`, `research/w4_critic.json`,
`research/w4_critic.log`.

## What the critic is reading, 2026-08-10

Registered before `research/w4_critic_ablate.py` was run.

The critic knows something this file does not: 0.2273 of excess against 0.1107
for every hand written statistic together. The cheapest way to find out what is
to take things away from it and retrain.

This is the question `w4_channels` was built for and could not answer. That arm
spliced streams together and every hybrid lost to both parents, because the
splice made objects neither source could emit. Nothing is spliced here. Each
critic sees real sequences from both sides and is denied a stream, which is a
restriction on the observer rather than a change to the data. A blinded stream
is replaced by a constant rather than deleted, so every variant sees the same
lengths and the same padding, and the architecture, optimiser, epochs, rows and
held out split are identical to `w4_critic` and to each other.

```
  only_speed  only_turn  only_time     what one stream carries by itself
  blind_speed blind_turn blind_time    what is lost when it goes, which is what
                                       that stream carries uniquely
  first32                              every stream, first thirty two events,
                                       every row exactly that long. Removes
                                       length as a cue and asks whether the tell
                                       is in how a movement starts.
  floor                                A half against B half, full streams
```

```
  LOCALISED   some only_x comes within 0.030 of the full critic. That stream
              carries the tell and an arm may target it.
  NECESSARY   no stream is sufficient but blinding some stream costs more than
              0.080. Those streams carry something nothing else duplicates.
  JOINT       no stream is sufficient and none is necessary. The information is
              redundant across streams and lives in their joint structure.
  VOID        the floor reads above 0.60.
```

0.030 and 0.080 are roughly four and ten times the 0.0009 spread between the two
independent readings of the full critic, 0.7518 and 0.7509.

Prediction, on the record: JOINT. Not on instinct, which has been wrong eleven
times of sixteen, but on the two measurements that bear on it. `w4_views` found
nine statistic blocks whose excesses sum to 0.4454 against a joint 0.1107, which
is heavy redundancy between different views of the same sequences. `w4_channels`
found every splice worse than both parents, which is what tight coupling looks
like. Redundancy between views and coupling between streams both predict that no
single stream is either sufficient or necessary.

### Result, 2026-08-10. NECESSARY, time. The first localised finding in this
### whole line of work, and it says the geometry was never the problem.

The record is 5 of 17. JOINT was predicted. It is NECESSARY, time.

```
         variant      AUC   vs full   excess over floor   share of full
            full   0.7500    0.0000              0.2311           100
     blind_speed   0.7439   -0.0060              0.2250            97
      blind_turn   0.7411   -0.0088              0.2222            96
         first32   0.7121   -0.0378              0.1932            84
       blind_time   0.6162   -0.1337              0.0973            42
      only_speed   0.5967   -0.1533              0.0778            34
       only_turn   0.5949   -0.1550              0.0760            33
       only_time   0.5885   -0.1615              0.0696            30
           floor   0.5189                             0
```

Blind the critic to the entire speed stream and it loses 0.006. Blind it to the
entire turn stream and it loses 0.009. Blind it to the millisecond waits and it
loses 0.134, which is 58 percent of everything it had. The geometry of the path
is very nearly innocent and the timing is where the model gives itself away.

The structure underneath that is sharper than the verdict name. Read the pairs.

  turn and time together   0.2250, where turn alone is 0.0760 and time alone
                           0.0696. The pair carries 97 percent where the parts
                           sum to 63. Strongly super additive.
  speed and time together  0.2222 against parts summing to 64. The same.
  speed and turn together  0.0973 against parts summing to 67. SUB additive,
                           so the two geometry streams are largely telling the
                           critic the same thing.

Time is not sufficient on its own, at 30 percent. Either geometry stream plus
time is worth almost the whole thing. So the tell is not in the timing
distribution and not in the shape, it is in the timing GIVEN the shape: the
model does not wait the way a hand waits for the movement it is currently
making. That is one specific, mechanical claim, and it is the first one this
file has been able to make.

`first32` at 84 percent says most of it is already present in the first thirty
two events, with length removed as a cue, so it is concentrated in how a
movement starts rather than spread evenly along it.

Two older results now read differently. `w4_views` ranked `turn_marg` first at
0.0727 and put the dt views at 0.0482 and 0.0329, so the summary statistics
pointed at turning while the learned critic points at timing. That is the
clearest possible demonstration of the previous section's first conclusion, that
arms designed from summary statistics are working with the wrong half. And the
feature conditioning obedience table put `mean_jerk` last at 0.289, the one
contract feature that is a third derivative in time. Both were pointing here.

The next arm is a critic guided fine tune aimed at the temporal head, not the
whole trunk. The evidence says the geometry is already close to right, and the
first rule of a learned objective is not to let it damage what is already
working. Nothing about serving changes: one trajectory per command, no
selection, the critic is a training signal only.

Artifacts: `research/w4_critic_ablate.py`, `research/w4_critic_ablate.json`,
`research/w4_critic_ablate.log`.

## Fine tuning the temporal head against the critic

PRE REGISTERED 2026-08-10, before any code was written and before any number
was seen. `w4_advtime.py`.

The ablation localised the tell. Blinding the millisecond waits costs the
critic 0.134 of 0.231 excess, blinding speed costs 0.006 and blinding turn
0.009, and time is strongly super additive with shape while speed and turn
without time are sub additive. The model does not wait the way a hand waits
for the movement it is currently making. Two older arms agree. `w4_timing` on
2026-08-05 gave the verdict "DT HEAD carries it" from a spectral readout, and
`mean_jerk`, the one contract feature that is a third derivative in time, is
the worst obeyed feature in the whole conditioning study at 0.289.

So the target is the timing head and nothing else.

### What moves

`dt_head` and `dt_norm`. Nothing else. The trunk, the speed head, the
direction head and every embedding are frozen. The speed and direction
conditionals are therefore untouched by construction, which is the strongest
form the geometry guard can take. This is a weight fine tune of a head that
already exists and is not the FiLM rewrite of `th_head` and `dt_head` that
sits on the NOT AUTHORISED list.

### The method

A critic cannot be differentiated through a sampled integer. The usual fix is
policy gradient, which is high variance and slow, and the alternative is to
feed the critic a soft mixture of embeddings, which is off the manifold the
critic was trained on and invites the generator to win by being blurry. This
run uses the straight through estimator instead, so the critic only ever sees
hard tokens in the forward pass and the gradient flows to the timing head as
if the token had been soft. The blur loophole is closed by construction.

Each round.

1. Sample trajectories from the current generator on fresh real commands, at
   temperature 1.0, through the same call serving makes. Nothing is selected
   and nothing is filtered on score.
2. Train a persistent critic further on real human rows against this round's
   generated rows. Persistent rather than fresh so it tracks the generator
   instead of restarting behind it every round.
3. Teacher force the model over its own sampled tokens, form the straight
   through timing embedding, and push the critic towards calling the row
   human. Add an anchor term, the ordinary cross entropy of the real timing
   token on real human sequences, at weight 1.0, so the head cannot buy the
   critic by abandoning the data.

### Controls and thresholds

The comparison is not the remembered 0.6446. The same script scores the
unmodified checkpoint on the identical commands with the identical generation
seeds inside the same run, so baseline and tuned are paired and most of the
run to run spread cancels. Three paired seeds, six thousand trajectories each.

  WIN      tuned beats its own paired baseline by 0.025 or more on the mean
  PARTIAL  tuned beats it by 0.008 or more, the size of the snap 5.0 effect
  NULL     anything less

  VOID     the baseline arm falls outside 0.6446 plus or minus 0.020, because
           then the run is not reproducing the model it claims to be fixing
  VOID     real data timing NLL rises by more than fifteen percent, because
           then the head has been destroyed and any movement in the score is
           a collapse rather than a repair

Reported either way, not gated on: mean event count and mean movement duration
for baseline and tuned against the human mean. Timing pressure is allowed to
change duration, that is what it is for, but if the event count moves more
than ten percent the intervention has reached the geometry through the state
and the result is confounded.

The critic control runs only if the primary reaches PARTIAL or better, because
a null primary has nothing to attribute. It is the identical procedure with
the critic frozen at random initialisation, so its signal is worth nothing. If
that also gains 0.008 or more then the gain is the fine tuning and not the
critic, and the primary is VOID.

Serving does not change. One trajectory per command, no candidates, no
selection. The critic is a training signal and is thrown away at the end.

### Prediction

PARTIAL. The critic reads 0.75 where the contract reads 0.64, so it is
attacking a superset of what the contract can see and the transfer will not be
one for one. Adversarial pressure through a single linear head on a frozen
trunk is a narrow instrument. I expect it to move the score and not to close
the gap. Record before this arm is 5 of 17.

### Amendment, 2026-08-10, after a smoke test and before the real run

Two changes, both made on numbers from a four hundred row smoke test and both
recorded here before the run they affect.

First, the baseline guard was wrong. It was registered against the served
0.6446, but serving runs the whole pipeline while this file calls the sampler
and the decoder directly, and those are not the same measurement. `w4_prefix`
already measured this exact path on its unforced arm and got 0.6337. The guard
is now 0.6337 plus or minus 0.035, the run to run contract noise `w4_prefix`
records. Nothing else about the registration moves. The smoke test read 0.6183
at four hundred rows, which is inside that band and was the thing that exposed
the mistake.

Second, the critic needs warming. After one round on a few hundred generated
rows it read 0.5434, so the first generator updates would have been pushing
against noise rather than against an opponent. Three thousand rows are now
generated from the untouched model and the critic is trained on them before
any generator step. This changes no arm, no control and no threshold; it is
how the opponent is fitted.

Also reduced, for the run to fit inside a supervised session: adjudication is
three paired seeds at four thousand trajectories rather than six thousand. The
seed count is what buys the spread estimate and it is unchanged. For scale, the
smoke test's paired difference on an effectively unchanged model was 0.0106 at
four hundred rows.

### TERMINATED at round 9 of 12, and why

Stopped deliberately, not by a crash. Three measurements taken while it ran
showed it was aimed at something that is not broken, and a fourth found the
thing that is. Recording the numbers it did produce, since a stopped arm still
has to report.

The critic ran away from the generator throughout: 0.696 warmed, then 0.774,
0.775, 0.811, 0.865, 0.861, 0.882, 0.884, 0.938 across rounds one to eight,
while the adversarial loss climbed from 1.41 to about 3.1. Real timing NLL
drifted only 2.7 percent, far inside the fifteen percent VOID line, so the head
was not collapsing. It simply could not win. There is no adjudication, so the
registered thresholds were never reached and the PARTIAL prediction is neither
right nor wrong. The record stays 5 of 17.

### What the probes found instead

Four cheap measurements, all on cached tokens or one forward pass, none of them
needing the GPU the fine tune was using.

One. Per trajectory clock consistency is fine. The within trajectory spread of
the wait, as a fraction of the pooled spread, is 0.665 for a hand and 0.639 for
the model, and the modal wait share is 0.577 against 0.590. If anything the
model is slightly more self consistent than a person. A tempting story, dead.

Two. The wait conditional is wrong in exactly one place. Binning events by step
size, every moving band has a total variation distance between hand and model
of about 0.045, which is the marginal difference and therefore nothing extra.
The no motion band reads 0.396. Humans put a one millisecond wait on 37.2
percent of no motion events and the model does it on 5.8 percent; the model
puts 38.1 percent of them at eight milliseconds where a hand puts 11.5. Half of
a hand's no motion events arrive within three milliseconds of the previous one
and under a fifth of the model's do. On moving events the two agree, 13.8
percent against 12.7.

Three. It is not where the model puts those events. Run structure is identical,
mean run 1.16 both, ninety percent singletons both, and the one millisecond
share does not depend on position within a run for either.

Four, and this is the one that mattered. Teacher forced on a REAL human
history the timing head is almost exactly right at no motion events: it
predicts a one millisecond wait with probability 0.294 where the truth is
0.297, eight milliseconds at 0.117 against 0.113, and three milliseconds or
less at 0.452 against 0.463. On moving events it matches to three decimals. The
head learned the conditional perfectly. So neither its capacity nor the way it
is conditioned is the constraint, and the fine tune was polishing something
already correct.

### The defect

`EventARModel.sample` draws the turn token, computes the timing distribution
conditioned on that raw sampled turn, and only afterwards replaces the turn
with the null marker at no motion events. Training never does this. The dataset
sets the turn token to null wherever there is no motion, and the turn head is
not even trained at those positions because `sup_th` masks them out of the
loss. So at every no motion event the sampler feeds an untrained arbitrary turn
token into the timing head, while every gradient the timing head ever saw had
the null marker there.

Teacher forcing the model on its own generated tokens shows the two paths
disagreeing at precisely those positions and nowhere else. On moving events the
head predicts what the sampler emitted, 0.017 against 0.018 at one millisecond
and 0.364 against 0.369 at eight. On no motion events the head predicts 0.085
and 0.241 where the sampler emitted 0.042 and 0.413.

This is a serving path bug, not a modelling failure.

## Giving the timing head the turn token it was trained on

PRE REGISTERED 2026-08-10, before the measuring script was written and before
any score was seen. `w4_ticknull.py`.

One line. `EventARModel.sample` gains `tick_th_null`, default False so the
served path does not move until this is measured. With it on, the null turn
marker is substituted at no motion events BEFORE the timing head runs instead
of after, which is what training always did.

### Arms

  base     the served path exactly as it is, `tick_th_null` False
  fixed    the same weights and the same commands with it True

Paired. Three seeds, four thousand trajectories each, identical commands and
identical generation seed per pair, so the two differ only by the flag.

### Thresholds

  WIN      fixed beats base by 0.025 or more on the mean paired difference
  PARTIAL  fixed beats it by 0.008 or more
  NULL     anything less

  VOID     base falls outside 0.6337 plus or minus 0.035, the value this same
           path gave in `w4_prefix` and the band amended earlier today
  VOID     the mechanism check fails. The probability of a one millisecond wait
           at a no motion event must move from about 0.06 towards the human
           0.37. If it does not move at least to 0.20 then the flag did not do
           what it claims and no score change may be attributed to it

Reported and not gated: mean event count and mean movement duration for both
arms. The commanded duration is a conditioning input and the state carries
elapsed against commanded, so the model has a route to compensate for faster
ticks elsewhere. If it does, the score may not move even though the defect is
repaired, and that is a result worth stating plainly rather than hiding.

### Prediction

PARTIAL. The defect is large where it lives, a total variation distance of 0.40
on the nine percent of events that carry no motion, and the learned critic put
fifty eight percent of its edge in the timing stream. Against that, the contract
features are computed after resampling to 125 Hz, which is an eight millisecond
grid, and the difference between a one millisecond and an eight millisecond wait
is exactly the scale that grid removes. I expect the tokens to improve more than
the contract score does. Record is 5 of 17.

## The training corpus is not the same sample of humans as the scoring reference, 2026-08-10

Not an arm and not pre registered. A diagnostic, run on CPU while the
`w4_ticknull` job held the GPU, and it changes what the target number means.

Real human trajectories from the training corpus, with no model anywhere in the
measurement, do not read 0.50 against the reference every arm is scored on.
They read about 0.53. The scorer itself is calibrated: the same corpus split
into two disjoint halves and put through the identical forest recipe reads
0.500.

```
  all real recorded human motion, no model involved
    corpus A against corpus B, same pool          0.5002  0.5022  0.4969
    corpus A against human_val_features_grpo      0.5353
    corpus A against human_ref_features_sir       0.5354
    grpo against sir, two references              0.5132
  measured again on other draws of the same thing 0.5298 to 0.5473
```

The training corpus is the outlier. It separates from both references by the
same amount while the two references separate from each other by a third of
that. Four things it is not:

Not the tokenisation. Encoding real events into the model's own speed, turn and
whole millisecond vocabulary and decoding them straight back reads 0.5346
against 0.5473 for the exact integer deltas with no codec at all. Pricing each
lattice separately against an exact reconstruction, speed, turn and clock all
came out inside 0.005 of each other and of the full round trip. The lattices
cost nothing.

Not the decoder. `esp._decode` with snap and rounding reads no higher than
cumulatively summing the raw integer dx and dy, and on one draw slightly lower.

Not the commanded moves. Reweighting the corpus by importance sampling so its
joint distribution over movement duration and mean velocity matches the
reference moves the medians onto the reference, 0.414 to 0.3995 against 0.405
and 584 to 627 against 616, and moves the score from 0.5316 to 0.5298. The
references were not recorded over a different set of commands, or if they were
it is not what the forest is reading.

Not a marginal. The largest robust median shift across all eighteen features is
0.088 pooled IQR units, and only one feature has a shift both references agree
on in sign and size. That feature alone reads 0.5011. The separation lives in
the joint structure across features, which is the same kind of defect the model
has, an order of magnitude smaller.

Two consequences, and the second one is the one that matters.

First, a correction to this file. `w4_token_ceiling` recorded the codec floor at
0.5118 and it has been quoted since as the number a perfect generator would
reach. That measurement is not wrong but it is not the floor for any arm scored
the way arms are scored. `scoring.score_features` keeps `synth[:n_use]`, so
whichever 2000 rows the caller happens to build first are the ones scored, and
this corpus is ordered by session. I made exactly that mistake twice in the
first hour of this diagnostic and read 0.5715 before shuffling the feature rows
and reading 0.535. Any measurement that hands the scorer a sorted index draw is
reading a narrow band of people. `research/w4_ticknull.py` builds its rows in
sorted order and has this property; it does not invalidate the paired
comparison, because both arms share the same commands in the same order, but the
absolute level of both arms is set by which band of commands the draw landed on.

Second, the target. The mandate is one trajectory per command scoring 0.50. A
perfect generative model of this corpus scores about 0.53, because the corpus
itself does, and two independently recorded human references disagree with each
other at 0.513. So 0.50 is not the floor for a model trained here. Somewhere
near 0.51 to 0.53 is, and the difference is not a modelling failure that better
architecture removes. It is a property of which humans were recorded, and it
sets a hard limit that no amount of generative quality can pass while the
training corpus and the scoring reference stay what they are.

This does not stop the current work. The served model reads about 0.63 against a
0.53 floor, so there is still roughly 0.10 of real modelling gap, which is most
of what was thought to be there. It does mean the last 0.03 of the way to 0.50
is not available and should stop being treated as a target.

### Result, 2026-08-10

PARTIAL, and the prediction was PARTIAL. Record 6 of 18.

```
     seed     base    fixed    delta  P1 base  P1 fixed  ev base  ev fixed
     4017   0.6431   0.6140  -0.0291    0.050     0.382     55.0      55.6
     8017   0.6355   0.6267  -0.0088    0.047     0.368     54.3      54.9
    12017   0.6484   0.6237  -0.0247    0.053     0.367     54.7      55.9

  base 0.6423  fixed 0.6215  delta -0.0209, seed spread 0.0107, se 0.0062
  P(1 ms at a no motion event)  0.050 to 0.372, human 0.372
  events 54.7 to 55.5, token duration 528 to 528 ms
```

Every gate passed. The base arm at 0.6423 is inside the registered band. The
mechanism check did not merely clear its 0.20 floor, it landed on the human
number to three decimals, 0.372 against 0.372, which is a stronger result than
the registration asked for and says the timing head had the conditional right
all along and was simply being asked the wrong question. All three seeds are
negative. Event count and total token duration are flat, so the warning written
into the registration, that the model might buy the faster ticks back somewhere
else through the elapsed against commanded state, did not happen.

The prediction reasoning also held. I expected the tokens to improve more than
the contract score, because the contract features are computed after resampling
to 125 Hz and the difference between a one and an eight millisecond wait is
close to what that grid erases. The token statistic moved from a total variation
distance of 0.40 to essentially zero while the score moved 0.021, which is that
effect exactly.

`models/event_ar.py` now defaults `tick_th_null=True`. This is a served path
change and it is the first one in this workstream that came from finding a bug
rather than from adding a mechanism. Every event_ar number recorded before today
was measured with the bug present, so earlier arms were competing against a
baseline that was 0.021 worse than it should have been, and any arm that scored
inside 0.02 of its base deserves rereading before it is trusted.

One caveat on the absolute levels, recorded in the same session as the corpus
diagnostic above. Both arms build their feature rows in sorted corpus order and
`scoring.score_features` keeps the first two thousand, so both are reading a
session ordered band and their absolute levels are inflated by roughly 0.02 to
0.03. The paired delta is not affected, because the two arms share the same
commands in the same order and differ only by the flag. Future arms should
shuffle feature rows before scoring, or generate close enough to two thousand
usable rows that no truncation happens.

### Choosing which humans to train on does not lower that floor, 2026-08-10

Follow up diagnostic, CPU only, same session. If the corpus sits 0.53 from the
reference then some recordings in it are closer than others, and training on the
closer ones would lower the floor for any model. This selects training inputs
and never generated outputs, so it does not touch the one trajectory per command
rule. It does not work, and the way it fails is worth writing down.

Selection was by density ratio. A forest was fitted to tell corpus rows from
`sir` rows on one half of the corpus, then used to score the other half, so no
row was judged by a model that had seen it. Selection used `sir` and evaluation
used `grpo`, so a subset that had merely learned one sample would show it.

```
  subset of the held out half              vs grpo   vs sir    rows
  all of it, no selection                   0.5494   0.5515   11612
  the two thousand most sir like            0.7869   0.6197    2000
  control, a random half                    0.5373   0.5390    5806
```

Hard selection on a classifier score makes the sample much MORE detectable, not
less. Keeping the rows a discriminator likes truncates the distribution in
exactly the directions that discriminator reads, so variance collapses there and
a fresh forest finds the truncation immediately. The soft version of the same
idea, importance reweighting the corpus onto the reference's joint distribution
of duration and mean velocity, was run earlier the same day and moved the score
from 0.5316 to 0.5298, which is nothing.

So the floor is not a selection artifact that a better choice of training rows
removes. It is a property of the recordings. Both the cheap version and the
careful version of this idea are closed.

## With the tick bug fixed the defect moves to the endgame, 2026-08-10

Diagnostics run straight after `w4_ticknull`, on the served sampler with the new
default. Not an arm.

First, the fix confirms itself from the token side, independently of the
contract. The wait conditional binned by step size was the probe that found the
bug. The no motion band read a total variation distance of 0.3964 before and
reads 0.0605 now, while every moving band is where it was.

```
  band             wait TV   turn TV     n gen     n hum
  no motion         0.0605       nan     12321     12822
  slow 1 to 8       0.0667    0.0388     24129     24846
  9 to 24           0.0398    0.0296     25077     25456
  25 to 48          0.0417    0.0623     23522     23194
  49 to 90          0.0470    0.0578     35719     34037
  fast 91 up        0.1245    0.0512     15794     12780
```

The largest remaining conditional defect is the fast band, above about 24 px in
one event. The sampler emits 24 percent more of them than a hand does and
mistimes them at twice the total variation of any other band.

It is not a head defect. Teacher forced on real human sequences the model
predicts a fast step with probability 0.0960 where one really happened 0.0957 of
the time, its speed marginal sits at a total variation of 0.0062 from the data,
and the wait distribution at real fast steps matches bin for bin at 0.0106. Free
running the same model realises 0.1236. Every conditional is right and the
sampler is still wrong, which is drift and not capacity.

Splitting by position says where the drift is, and it is not spread out.

```
  events          human  generated   excess  tf model
  0 to 4         0.1107     0.1325   0.0219    0.1208
  4 to 8         0.1682     0.1774   0.0091    0.1685
  8 to 16        0.1768     0.1719  -0.0049    0.1764
  16 to 24       0.1411     0.1346  -0.0065    0.1410
  24 to 32       0.0997     0.0998   0.0001    0.1000
  32 to 48       0.0656     0.0872   0.0216    0.0658
  48 to 256      0.0446     0.0900   0.0454    0.0451
```

A hand accelerates to a peak around event ten and then decelerates hard into the
target, dropping to 0.0446 by the tail. The model reproduces the acceleration
and the peak and then does not decelerate, holding 0.0900 where a hand is at
0.0446, twice the rate. The teacher forced column tracks the human curve at
every position including the deceleration, so the model knows the endgame
perfectly well when it is standing on a real history and cannot find it on its
own.

That is a specific and physically meaningful defect rather than a diffuse one.
The homing phase of a pointing movement is the part with the tightest feedback
in a real hand and the part where an open loop sampler has the least to hold on
to. `prefix_state` hands the model the exact remaining vector at every step, so
the information is there; what the tail says is that the model's own accumulated
position is drifting far enough that the remaining distance it reads is not the
one a hand would be reading.

The next measurement is terminal accuracy: how far from the commanded target the
generated trajectories actually finish, against a hand, and whether the late fast
steps are the model still travelling because it has not arrived. That is the
thread to pull next and it was not run today.

## Every head is exact and every error is made by free running, 2026-08-10

The terminal accuracy thread named at the end of the previous section was pulled
and it went somewhere better than expected, through five hypotheses that died on
the way. What follows is the whole chain, because the dead ones are as
informative as the surviving one and each was cheap to kill.

### Terminal accuracy

Generated trajectories do arrive. Median terminal error is 0.0162 of the
commanded distance against a human 0.0104 through the same tokens, and the two
are decomposed the same way, a median 0.9998 of the distance covered along the
axis against a human 1.0000, and a lateral miss of 0.0129 against 0.0093. The
arriving is not the problem. The tail is: ten percent of generated trajectories
never come within five percent of the target against three and a half percent of
human ones, and one percent run into the 256 event buffer against 0.08 percent.

Split into terciles by terminal error, the late fast step rate reads 0.0250 for
the best arriving third, which is BELOW the human 0.0509, and 0.1488 for the
worst third against a human worst third at 0.0849. So the endgame excess named
in the previous section is not a uniform failure to decelerate. The best two
thirds decelerate correctly and the worst third carries all of it. Measured
after each trajectory's own closest approach, the leaving phase, the model takes
large steps at 0.2579 against a human 0.0379, and spends twice as many events
there.

### But the flailing tail is worth nothing on the contract

Each trajectory cut at its own closest approach, a rule that uses only the
command and the path so far and is applied to every trajectory with no selection
among candidates, with the identical cut applied to the human corpus as the
control since truncation moves duration and length features on its own:

    human corpus, raw recording              0.5681
    human corpus, through the tokens         0.5330
    human corpus, tokens then cut            0.5597
    generated, whole trajectory              0.6275
    generated, cut at closest approach       0.6624

Cutting moves generated +0.0349 and the human control +0.0268, so the net
attributable to the flailing tail is +0.0082, the wrong sign and inside noise.
The tail is genuinely non human at the token level and the contract cannot see
it. That is the second time this has happened: the tick bug moved a token
statistic from a total variation of 0.40 to zero and bought 0.021.

Two things in that table are worth keeping. The token round trip costs
`-0.0351` against the raw recording, so the representation is not a ceiling and
going through the tokens makes the corpus look MORE like the reference, not
less. And the model sits `+0.0945` above its own representation, which is the
entire quantity this workstream is trying to remove with the corpus against
reference floor already divided out.

A caution paid for twice today. A hand rolled walk of the tokens into continuous
positions is NOT the served path and reads about 0.13 high, because
`esp._decode` snaps every step below 2.5 px to a whole pixel and merges ticks
between two fast steps. Any probe that decodes tokens itself must go through
`esp._decode`. Separately, scoring a set against a lightly perturbed copy of
ITSELF is not a valid control: each trajectory's near twin sits in the opposite
class, the out of bag forest reverses, and the arm reads 0.109 or 0.000 instead
of 0.5. Two controls were lost to that before it was recognised.

### So ask the contract instead of the tokens

Generated and human corpus from the same commands, the same tokens and the same
decoder, so the only thing separating the arms is the model. That reads 0.6068
against a paired floor of 0.5000, and the shape of it is the finding:

    no single feature separates       best alone 0.5567, most sit at 0.50
    no single feature is necessary    best without it 0.6095, base 0.6046
    repairing one marginal HURTS      sixteen of eighteen times, up to 0.6522
    repairing all eighteen at once    0.5296, buys 0.0750 of the 0.1046
    one logistic direction            0.6012 of the 0.6046

The mean shifts are small, a fifth of a human standard deviation at worst. The
SPREAD ratios are not: mean acceleration 2.52, std velocity 2.10, std jerk 2.06,
max velocity 1.88. Movement duration, the one thing the command pins down, sits
at 0.98 and is exactly right.

So the model's feature vector is displaced coherently. Every marginal is a
little wrong in a mutually consistent way, which is why forcing one coordinate
to the truth while the others stay displaced puts the vector somewhere neither
population lives and the forest finds it instantly. Rank mapping preserves the
model's own rank correlations, so the 0.5296 left after repairing everything is
the model's dependence structure, which is nearly right, and the 0.0750 removed
is the displacement, which is nearly all of it.

### Which channel, and is the head wrong

Step size is clean. Its autocorrelation matches the human one to three decimals
at every lag out to thirty, 0.8212 against 0.8185 at lag one, the variogram is
within five percent everywhere, the marginal spread within four percent. The
model does not random walk in how big a step it takes.

The wait does not match. A hand's waits de-correlate to nothing by lag twelve,
reading 0.0048, and the model's are still 0.0748 there and positive at lag
twenty. At motion events the model emits a wait of one millisecond or less at
0.0507 against a human 0.0301, a factor of 1.68, and that survives the obvious
objection: the human rate through the identical token round trip is 0.0301
against 0.0299 raw, so it is not a rounding artefact. Instantaneous velocity is
step size over wait, so those events carry the largest velocities in the
trajectory, and the per trajectory spread of mean velocity comes out at 2.56
times human while step sizes sit at 1.02.

Then the test that decides everything. Teacher forced on 3000 real human
sequences, the dt head predicts a wait of one millisecond or less at 0.0310
against a real 0.0306, in every speed band, with a total variation of 0.0041
over the whole wait distribution:

    band            model P   really was   ratio        n
    all motion       0.0310       0.0306   1.0136   145051
    slow 1 to 8      0.0549       0.0544   1.0083    29884
    9 to 24          0.0462       0.0462   0.9981    30595
    25 to 48         0.0295       0.0289   1.0184    28127
    49 to 90         0.0112       0.0105   1.0707    41058
    fast 91 up       0.0101       0.0098   1.0280    15387

The head is exact and free running is 1.68 times wrong. This now holds for all
three heads. The speed head was shown exact by `w4_ticknull` and again here, the
turn head earlier, and the dt head now.

### What died

  clock chasing        the model hits its commanded duration, log ratio sd
                       0.046, so it is not compressing waits to catch up
  random walk in speed  the step size sequence matches on every second order
                       statistic that exists
  per trajectory offset  shrinking the trajectory level timing offset back to
                       the human dispersion made the contract WORSE by 0.027
  catastrophic tail    humans have the same extreme tail, p99 short wait rate
                       0.7681 and 1.66 percent above 0.30 against the model's
                       2.23 percent, which is a population of high polling rate
                       devices in the corpus and not a defect. Trimming the
                       worst ten percent from both arms moves 0.6025 to 0.5890
  the flailing endgame  real at the token level, worth nothing on the contract

### What this means

Everything the model gets wrong, it makes during free running composition. Every
per step conditional is right, measured directly against real histories, on all
three heads, to three or four decimal places. The result is a coherent and
largely linear displacement of the whole eighteen feature vector worth 0.107
paired, of which 0.075 is the displacement itself and 0.030 is dependence
structure.

That retires a family of work. Four consecutive investigations have found a real
token level defect, fixed or priced it, and moved the contract by nothing or
close to nothing. Per step diagnostics have no remaining yield here, because per
step is the thing that is already correct. The objective has to see the model's
own rollouts. That is the next arm and it is a training arm, not a sampling one.

## The spike rate is the mechanism and the contract still cannot see it, 2026-08-10

Registered before the run as `research/w4_spikerate.py`, hypothesis, controls,
prediction and falsifier fixed in the docstring. The prediction was wrong in a
way that is worth more than being right would have been.

### The bridge was real

The arithmetic bridge from the previous section holds. Instantaneous velocity is
step size over wait, so a near zero wait carries a velocity several times
anything else in its trajectory and the contract differentiates that once more
for acceleration and twice more for jerk. Removing every sub millisecond wait
from the generated token streams and redrawing each from the model's own upper
band:

    spread ratio against the human corpus, 1.00 is exact

    feature                  base   spikes removed   wait marginal mapped
    mean_acceleration       1.606            0.548                  0.850
    std_velocity            2.216            0.948                  1.640
    std_jerk                2.344            0.787                  1.973
    max_velocity            2.085            0.918                  1.368
    std_acceleration        2.216            0.762                  1.697
    mean_jerk               1.275            0.393                  0.672
    movement_duration       0.979            1.024                  0.998

So the sub millisecond wait rate IS what manufactures the over dispersion, and
it manufactures essentially all of it. The mean absolute log spread ratio over
all eighteen features moves 0.3445 to 0.1900.

### And it buys nothing

Scored on the contract, generated base 0.6412 against a human corpus floor of
0.5576, so a gap of 0.0836:

    generated, sub ms rate multiplier 1.0      0.6412   rate 0.0508
    multiplier 0.8                             0.6360   rate 0.0406
    multiplier 0.594, lands on the human rate  0.6384   rate 0.0302
    multiplier 0.4                             0.6380   rate 0.0203
    multiplier 0.0, every spike gone           0.6295   rate 0.0000
    whole wait marginal rank mapped to human   0.6313
    CONTROL rate held, spikes reshuffled       0.6413
    CONTROL speed channel perturbed instead    0.6745

Landing the rate exactly on the human value is worth 0.0028 of 0.0836. Removing
every spike is worth 0.0118. Repairing the entire wait marginal, which is the
upper bound on everything this channel can carry, is worth 0.0099. The dose
curve has no minimum at the human rate and barely has a shape.

The reshuffle control is flat at 0.6413 against a base of 0.6412, so the edit
machinery does nothing on its own and the rows above are readable. The speed
channel control moved 0.033 and moved it the WRONG way, which is the predicted
sign for perturbing a channel already measured clean. The verdict rule written
into the script was two sided and fired NULL on that control; the substantive
verdict is NULL anyway, on the size of the treatment alone, so nothing turns on
it. Recorded because the rule as written is wrong and should be one sided if
this arm is ever rerun.

### The control that decides it

The same edit run backwards on real recorded human trajectories, injecting the
model's own excess into a hand:

    human corpus, unedited                     0.5576   rate 0.0291
    human corpus, multiplier 1.68              0.5495   rate 0.0488
    human corpus, multiplier 2.5               0.6178   rate 0.0726

Giving real humans exactly the model's spike rate does not make them detectable.
It moves them 0.008 and in the wrong direction. The channel only becomes visible
somewhere past 0.05, and the model sits at 0.0508, just underneath. The detector
is insensitive to this quantity over the whole range the model is wrong by.

### What this means

The count is now five. Five times a genuine token level defect has been found,
confirmed against a real human history, fixed or priced, and worth nothing on
the contract: the tick bug, the duration prior, the lattice snap, the flailing
endgame, and now the spike rate. This one is the most informative of the five
because it also carried the entire feature over dispersion, which was the best
remaining candidate for the mechanism, and repairing that over dispersion almost
completely still bought 0.012.

Put beside the earlier paired result, the asymmetry is the finding. Rank mapping
all eighteen FEATURE marginals onto the human corpus buys 0.075 of 0.107. No
token level edit tried so far reaches more than 0.012 of it. The displacement
lives in feature space and nothing available in token space reaches it, which
means it is not any one channel and there is no cheap serving time controller
waiting to be found.

That settles the question the previous section left open. The next arm is a
training arm, the objective has to be stated over the model's own rollouts, and
it now has a specific target rather than a general one: match the eighteen
feature joint distribution of generated trajectories to the human one. Two
attempts in this family have already failed and both failure modes are known and
must be designed against. The GRPO pilot collapsed the model's variety, which a
distribution matching objective penalises directly rather than rewarding. The
learned critic outran the generator, reaching 0.94 by round eight, which a fixed
non learned statistic like a feature moment or an MMD kernel cannot do.

## Rollout level training, registered 2026-08-10 before any training

Written before the script exists. Arms, controls, held out features, thresholds
and the prediction are fixed here and are not edited after reading the output.

### Why this and not another sampling arm

Every per step conditional is exact, measured against real human histories on
all three heads. Five separate token level defects have been found, confirmed
and priced at nothing. Repairing all eighteen feature marginals directly buys
0.075 of 0.107 while no token level edit reaches more than 0.012 of it. The
error is made by free running composition and the objective has to see free
running composition. There is nothing cheaper left.

### The estimator

The obvious route, relaxing the token sampling so the whole rollout is
differentiable, is refused on memory. A rollout is 256 sequential trunk passes
over a growing prefix with no KV cache, and retaining activations for all of
them does not fit in 8 GB at any useful batch size.

The route taken instead uses a property of this trunk. It is causally masked, so
a single full sequence teacher forced pass over a sequence reproduces exactly
the logits that were used when that sequence was sampled. So each step is one no
grad rollout, which is the expensive part, followed by one ordinary forward and
backward over the model's OWN emitted tokens. That is a score function estimator
and it costs one rollout plus one training pass, not 256 of them.

Features are standardised by the human mean and standard deviation. On a batch,
with m and s the batch mean and standard deviation of standardised feature k:

    L = sum_k m_k^2 + sum_k (log s_k)^2

The first term moves location, the second moves spread, and spread is where the
defect is: mean shifts are at worst a fifth of a human standard deviation while
spread ratios run to 2.5. The per trajectory weight is the exact score function
coefficient for that loss, with the batch mean as its control variate:

    w_i = sum_k [ 2 m_k (z_ik - m_k)
                  + (log s_k / s_k^2) ((z_ik - m_k)^2 - s_k^2) ]

and the surrogate minimised is mean_i w_i.detach() * log p(tau_i).

A teacher forced negative log likelihood term on real human batches is added at
weight lambda. It is the anchor, and it is the specific answer to how the GRPO
pilot failed: that run collapsed the model's variety, and here both the variance
term and the likelihood anchor penalise collapse directly rather than rewarding
it. The second known failure, the learned critic reaching 0.94 by round eight
and outrunning the generator, cannot happen because nothing in this objective is
learned. It is a fixed statistic of a fixed feature map.

### The Goodhart guard, which decides the arm

This objective is stated over the same eighteen features the scorer reads, so
the arm is worthless without a held out set. Twelve features enter the loss. Six
never do, chosen to span the families rather than to be easy:

    max_acceleration        velocity_skewness      curvature_std
    num_direction_changes   time_to_peak_velocity  angular_velocity_std

If the trained twelve come into line and the held out six do not, the model
learned the loss and not the movement, and the arm is a FAIL whatever the
contract says. That is the reading, not a caveat on it. A gradient boosted tree
detector, which took no part in the objective, is reported alongside.

### Configuration

    rollout cap        160 events, which covers 96.4 percent of the corpus. The
                       identical cap is applied to the human reference features
                       so truncation cannot separate the arms on its own
    batch              96 rollouts per step
    anchor             real human batches, teacher forced, weight lambda
    optimiser          AdamW at a low learning rate off the existing checkpoint,
                       training/event_ar_v2_s40000.pt, which is not overwritten
    thermal            launch gate 75C, kill 79C for this workload since the
                       machine crashed on it on 2026-08-06, supervised only

### Prediction, fixed before the run

Spread ratios fall toward 1.0 on the trained twelve AND on the held out six. The
contract falls from its 0.6412 base toward 0.60. Reaching the 0.5576 corpus
floor in a short pilot is NOT predicted and would be suspicious rather than
welcome.

### Reading

    CONFIRMED  contract falls at least 0.03 below base, the held out six improve
               by at least half as much proportionally as the trained twelve, and
               scoring.py raises no collapse flag
    PARTIAL    0.01 to 0.03, held out six moving the right way
    GOODHART   trained twelve improve, held out six flat or worse. Reported as a
               fail. This outcome is informative and must not be dressed as a
               partial win
    NULL       under 0.01

## Rollout level training moves the number, 2026-08-10

Registered in cb85e7d before it ran. First arm in this workstream to move the
contract by more than 0.02, and it moved it by 0.038 in ninety training steps.

The run stopped itself at step 90 of a planned 250 when the GPU touched the 79C
kill, which is the tightened limit this workload carries after crashing the
machine on 2026-08-06. Everything below is therefore a third of the intended
pilot. The checkpoint and the result file were both written cleanly on the way
out.

    measurement                       base    step 50    stop (step 90)
    contract scorer                 0.6682     0.6141            0.6301
    second detector, GBM            0.6744     0.6388            0.6248
    spread error, twelve in loss    0.3783     0.1881            0.1856
    spread error, six held out      0.1958     0.1231            0.0887
    location error, twelve          0.1398          .            0.0689
    location error, six held out    0.1209          .            0.0887

The second detector took no part in the objective and fell by 0.0496, further
than the scored one. Both detectors and both spread errors move the same way.

WHAT THE OBJECTIVE ACTUALLY DID

Per feature, spread as a multiple of the training corpus spread. One is right.

    feature                    base    stop    in the loss
    std_jerk                  2.253   0.827    yes
    std_velocity              2.211   1.047    yes
    max_velocity              2.186   1.094    yes
    max_acceleration          2.068   0.934    HELD OUT
    std_acceleration          2.014   0.842    yes
    mean_acceleration         1.422   0.745    yes
    mean_velocity             1.403   0.970    yes
    max_deviation             1.407   1.229    yes
    time_to_peak_velocity     1.185   1.079    HELD OUT
    curvature_mean            1.181   1.304    yes, went the wrong way
    curvature_std             1.097   1.175    HELD OUT, went the wrong way
    mean_jerk                 0.868   0.439    yes, overshot

The row that decides the arm is max_acceleration. It was the worst offender at
2.068, it was never in the loss, and it lands at 0.934. A model fitting the loss
rather than the movement cannot do that. The other five held out measurements
improve too, by 0.56 of what the twelve trained ones improved by, against a bar
of 0.50 fixed before the run.

Two features went the wrong way, both curvature, one trained and one held out.
Mean jerk overshot from 0.868 to 0.439, so it is now too narrow rather than too
wide. Ninety steps of a high variance gradient estimator will do that and the
symmetric log penalty should pull it back given more steps.

THE VERDICT RULE HAS A DEFECT AND IT IS RECORDED, NOT EDITED

The script prints PARTIAL. It prints PARTIAL because the registered rule
requires that scoring.py raise no collapse flag, and that flag is set at every
evaluation including the baseline taken before any training.

That flag fires when any of the eighteen measurements is more than five times
wider or narrower than the same measurement on the reference set. Feeding real
recorded human movement from the training corpus through it fires the flag, on
max_acceleration at 0.146, std_acceleration at 0.168 and max_velocity at 0.181.
The flag is not measuring model collapse. It is measuring the fact that the
reference set was recorded from different people on different hardware, some of
it very high polling rate mice, which gives that set a long tail on exactly
those three quantities. Anything without that tail looks collapsed against it,
real humans included.

So the condition would fail a perfect model, and it cannot discriminate here.
On the two conditions that can, a contract drop of 0.0381 against a 0.03 bar
and a held out ratio of 0.56 against a 0.50 bar, the arm reads CONFIRMED. The
rule stays as registered. The next arm should drop the collapse condition and
replace it with a check against the corpus rather than the reference.

TWO NUMBERS THAT ARE NOT COMPARABLE ACROSS ARMS

This arm's baseline reads 0.6682 where the standing number for the same
checkpoint is 0.6412. The difference is the conditioning commands sampled and
the evaluation size, not a regression, and it is not currently explained beyond
that. Base and stop share the same commands and the same pipeline, so the delta
inside this arm is sound. Any comparison of 0.6301 to a number from another arm
is not.

There are two human sets in play and they must not be confused. The training
corpus, which supplies the mean and spread this objective standardises by, and
the reference set the detector scores against. Against the corpus the model was
about 1.5 times too wide. Against the reference both the model and the corpus
are far too narrow, the model at about a fifth of the reference spread and real
corpus humans at about a seventh. That gap is a large part of why real humans
score 0.55 and not 0.50, and it caps what this training data can buy.

WHAT THIS RETIRES AND WHAT IT OPENS

It retires the reading of the previous five nulls as evidence that the gap is
unreachable. It is reachable, and the thing that reaches it is exactly the
thing the five nulls pointed at: an objective that sees whole generated
trajectories instead of individual decisions.

Open, in order. Get the run past ninety steps, which is a thermal problem and
not a research one. Find out whether the contract keeps falling or flattens,
since it read 0.6141 at step 50 and 0.6301 at the stop while every other
measurement kept improving, and at 1450 rows per evaluation that bounce is
inside the noise and cannot currently be resolved. Then decide whether the
remaining distance to the corpus floor is spread, location, or something the
eighteen measurements do not name.

## Rollout training continuation, registered 2026-08-10 before it runs

The pilot stopped at step 90 of 250 on the thermal abort and every measurement
except the scored detector was still improving when it did. This continues the
same objective from the pilot checkpoint. Nothing about the loss, the twelve
trained features or the six held out ones changes.

Two run controls change and neither touches the objective. The loop now pauses
at 74C until the card is back under 70C, so the 79C abort becomes unreachable
rather than something that catches a run already lost, and --init continues from
a saved checkpoint instead of the base model.

One registered threshold is replaced and this is the record of it. The original
CONFIRMED condition required scoring.py to raise no collapse flag. That flag is
measured against the reference set, it fires on real corpus humans, and it was
set at the pilot's baseline before any training happened, so it cannot separate
a good run from a bad one. It is replaced by a runaway check against the corpus
the model is actually being matched to: no feature's spread ratio outside 0.5 to
2.0. This is not a weakening. The pilot's mean jerk overshot to 0.439 and would
trip the new gate, and the old gate would have passed it.

    CONFIRMED  contract falls at least 0.03 below the continuation's own base,
               the held out six improve by at least half as much as the trained
               twelve, and no feature's spread runs outside 0.5 to 2.0
    PARTIAL    0.01 to 0.03, or a runaway feature
    GOODHART   trained twelve improve, held out six flat or worse. A fail
    NULL       under 0.01

PREDICTION, fixed before the run

The trained spread error keeps falling from 0.1856 and flattens rather than
reaching zero, because the estimator is high variance and the anchor pulls the
other way. The held out six keep tracking. Mean jerk, which overshot to 0.439,
comes back toward 1.0 rather than continuing down, because the log penalty is
symmetric. The scored detector resolves the 0.6141 to 0.6301 bounce and lands
below 0.62.

FALSIFIER

The scored detector flat or rising over the next two hundred steps while the
spread errors keep falling. That would mean the eighteen measurements are not
where the remaining separation lives and the objective has run out of reach,
which is a different conclusion from the pilot's and would send the next arm
after whatever the eighteen do not name.

## Where the remaining gap actually lives, 2026-08-10

NOT pre registered. Run exploratory while w4_rollout was training, to find out
whether that objective had a ceiling. It returns a decomposition, not a verdict.
The marginal rung replicates w4_joint, which was registered on 2026-08-07, so
only the second order rung and the residual are new. research/w4_gapsplit.py,
three seeds, cap 160, the same length w4_rollout evaluates at.

The contract reads eighteen numbers and nothing else, so every objective this
workstream can state is a statement about the distribution of those eighteen.
Each rung below imposes more of the corpus distribution on the generated
features and scores the result through the ordinary contract. The rungs nest, so
the differences partition the gap.

    rung                                                 auc      se
    base, generated features untouched                0.6360  0.0027
    A, every marginal rank mapped onto the corpus     0.6061  0.0050
    B, A plus the corpus correlation matrix           0.5826  0.0054
    C, the corpus itself, the floor                   0.5455  0.0032

    perfect marginal correction is worth             +0.0298
    second order dependence adds                     +0.0235
    survives both, unreachable by moments            +0.0372
    whole gap from base to floor                     +0.0905

All three seeds give the same ordering and the same rough split. The three
differences are each five to ten standard errors clear of zero.

THE CONTROL IS INSIDE THE LADDER

A and B carry the corpus marginals exactly, by construction, and C is the
corpus. Those three rows have identical eighteen marginals and they score 0.606,
0.583 and 0.546. Same numbers, same spreads, and the contract still separates
them. Whatever is left is dependence, and 0.0372 of it is not second order
either.

WHAT THIS PRICES

w4_rollout matches the first two moments of twelve of the eighteen. That is a
subset of rung A, so at most 0.0298 was ever available to it, and it took
roughly that in its first ninety steps. The flattening of its contract number
while its spread errors kept falling is the objective reaching its own ceiling,
not the model failing to learn. A longer run of the same objective cannot buy
what the ladder says is not there.

It also prices the next one. An objective that matches the eighteen dimensional
joint rather than a list of moments has a ceiling of the corpus floor, 0.5455,
which is the whole 0.0905. A fixed distribution distance over the feature batch,
an energy distance or a kernel one, fits the same score function estimator that
w4_rollout already uses, stays a fixed statistic so the learned critic failure
cannot recur, and reaches all orders rather than the first two.

WHAT IS STILL WRONG IN THE DEPENDENCE

The largest correlation errors, generated against corpus, averaged over seeds:

    num_direction_changes   angular_velocity_mean   0.170
    num_direction_changes   angular_velocity_std    0.150
    mean_acceleration       std_acceleration        0.132
    mean_acceleration       max_acceleration        0.130
    mean_acceleration       std_jerk                0.129
    mean_velocity           mean_acceleration       0.127

Two families. The model does not tie how often a path changes direction to how
fast it turns, and it does not tie mean acceleration to the spread of
acceleration. In a real movement those are the same physical fact seen twice.
In the model's output they are close to independent.

THE HARD CEILING, STATED PLAINLY

C is 0.5455, not 0.50. That is real corpus humans through the same tokens
against the same reference, and it is where a perfect model of this training
data lands. Reaching 0.50 would require matching the reference set rather than
the corpus, and the reference set is what the contract scores against, so
training on it would be training on the test. The honest ceiling of this whole
line is about 0.55, and the distance from 0.636 to 0.546 is the entire remaining
project.

## Energy distance objective, registered 2026-08-10 before any training

w4_gapsplit priced the moment objective's ceiling at about 0.030 of a 0.0905 gap
and the first pilot reached it in ninety steps. This replaces the objective and
changes nothing else. It is a flag on research/w4_rollout.py rather than a new
file, because the rollout, the teacher forced pass, the advantage normalisation,
the anchor, the held out six, the thermal control and the verdict rule are all
identical. Only the loss and its per trajectory weights differ.

THE OBJECTIVE

Between the generated batch z and a human batch h the energy distance is

    E = 2 mean_ij ||z_i - h_j|| - mean_ik ||z_i - z_k|| - mean_jl ||h_j - h_l||

zero if and only if the two distributions are equal, at every order rather than
the first two. The last term is constant in the model and is dropped. The score
function coefficient is the derivative of the statistic with respect to
including trajectory i,

    phi_i = (2/m) sum_j ||z_i - h_j|| - (2/n) sum_k ||z_i - z_k||

the factor two on the second term because z_i appears twice in the generated
double sum. Checked against autograd of the statistic under per trajectory
weights, agreeing to 3e-6 in float32.

Still a fixed statistic of a fixed feature map, so the learned critic failure
cannot recur. Still twelve features in, six held out, so the Goodhart guard is
unchanged. Ceiling is the corpus floor, 0.5455, rather than 0.030.

THE ARM

Starts from the same base checkpoint the first pilot started from, not from the
moment trained one, so the two objectives are compared from the same place over
the same number of steps. Same batch, cap, learning rate, anchor weight and
evaluation size as the moment continuation.

    CONFIRMED  contract falls at least 0.05 below its own base, the held out six
               improve by at least half as much as the trained twelve, and no
               feature's spread runs outside 0.5 to 2.0
    PARTIAL    0.02 to 0.05, or a runaway feature
    GOODHART   trained twelve improve, held out six flat or worse. A fail
    NULL       under 0.02

The bar is 0.05 rather than the 0.03 used for the moment arm because 0.03 is now
known to be reachable by matching means and spreads alone. An energy objective
that only matches that has bought nothing new and should not be called a win.

PREDICTION, fixed before the run

Energy tracks the moment objective for the first fifty steps or so, because the
cheap shared correction is the same either way, then keeps going where the
moment objective flattened. It passes 0.05 below its base. The held out six
track better than they did under moments, where the ratio had fallen to 0.35 by
step 150, because a distribution distance has no reason to privilege the twelve
directions it is computed in the way a list of per feature moments does.

FALSIFIER

Energy stalls at the same place moments did, around 0.03 below base, with the
held out ratio falling the same way. That would mean the reachable part of the
gap is the moment part regardless of objective, and that the 0.037 the ladder
says survives all moments is not reachable by training this model on these
features at all. The next question would then be whether the feature map itself,
rather than the objective, is the binding constraint.

## The moment objective converges and the contract does not follow, 2026-08-10

The continuation registered above ran its full 300 steps with the cooling
control working, peak 76C against a 79C kill, 9.2 minutes of cooling inside
65.5 minutes of wall time. VERDICT NULL by the rule as registered.

    eval        contract   gbm     spread trained  spread held  loc trained
    base          0.6364  0.6578          0.2147       0.1246       0.0895
    step 75       0.6303  0.6338          0.1419       0.0854       0.0862
    step 150      0.6225  0.6482          0.0570       0.0702       0.0738
    step 225      0.6271  0.6438          0.1122       0.0939       0.0762
    step 300      0.6281  0.6439          0.1370       0.0576       0.0574

    contract           0.6364 -> 0.6281,  d 0.0083, under the 0.01 NULL bar
    held over trained  0.86, well clear of the 0.50 Goodhart bar
    runaway spread     none

Two readings that were wrong in flight and are corrected here. At step 75 the
contract had moved 0.006 and I called the falsifier on one point; step 150 came
in at 0.6225 and it was still falling. At step 150 the held over trained ratio
had fallen to 0.35 and I reported the arm as drifting into Goodhart; by step 300
it had recovered to 0.86. Neither intermediate reading survived. Read this arm
off its endpoints, not its trajectory.

WHAT IT LEAVES BEHIND

Spread as a multiple of the corpus, at the pilot checkpoint and after 300 more
steps. Nearly everything is now inside fifteen percent of correct.

    std_jerk           1.417 -> 0.921        max_acceleration  1.389 -> 0.953 HELD
    max_velocity       1.437 -> 1.080        curvature_std     1.166 -> 1.031 HELD
    std_velocity       1.341 -> 0.994        angular_vel_std   0.948 -> 0.961 HELD
    std_acceleration   1.244 -> 0.853        movement_duration 0.976 -> 1.000
    curvature_mean     1.337 -> 1.184        mean_jerk         0.513 -> 0.500

That is the whole finding. Sixteen of eighteen spreads sit within fifteen
percent of human, the six held out of the objective moved as far as the twelve
inside it, location error halved, and the contract moved 0.008, which is inside
its own noise floor of about 0.006.

Mean jerk is the one failure. It overshot in the pilot to 0.513 and 300 further
steps left it at 0.500, sitting exactly on the runaway boundary without crossing
it. The symmetric log penalty did not pull it back, which the registered
prediction said it would.

THIS CONFIRMS THE LADDER RATHER THAN CONTRADICTING IT

w4_gapsplit priced perfect marginal correction at 0.0298 of a 0.0905 gap. The
pilot took 0.038 of its own baseline and this continuation took a further 0.008.
The two runs use different evaluation sizes so they do not add cleanly, but the
shape is unambiguous: the moment objective spent its budget early, converged,
and then oscillated. Step 150 was the bottom and steps 225 and 300 came back up
while the objective's own loss got worse, which is a converged run with a step
size too large for the noise near its optimum, not a run still learning.

So the ladder's arithmetic holds from both directions. Matching means and
spreads buys about 0.03 and nothing more, and it does not matter how long the
run is. The next arm has to change the objective, which is the energy distance
registered above.

## The residual is diffuse, not localised, 2026-08-10

w4_gapsplit rerun with the forest's own feature importances recorded at each
rung. Top five at each, averaged over the three seeds.

    base           angular_velocity_mean 0.075  mean_acceleration 0.067
                   angular_velocity_std 0.064   path_efficiency 0.061
    A marginal     mean_acceleration 0.063      angular_velocity_std 0.060
                   path_efficiency 0.059        time_to_peak_velocity 0.059
    B plus corr    mean_acceleration 0.063      velocity_skewness 0.060
                   angular_velocity_std 0.059   path_efficiency 0.059
    C corpus       mean_acceleration 0.061      movement_duration 0.059
                   angular_velocity_std 0.058   velocity_skewness 0.058

Eighteen features, so uniform is 0.056. At the base there is mild concentration,
angular velocity mean at 0.075. By rung B the profile is flat and it is the same
flat profile the forest shows at the floor, where the two sides are real people
against real people.

So the 0.0372 that survives marginals and correlations is not sitting in one or
two features waiting to be attacked. The forest is finding it by using all
eighteen a little, which is the signature of a diffuse difference in the joint
rather than a localised defect. That is the fifth or sixth time this workstream
has gone looking for a single attackable cause and found a distributed one.

It also removes an option that would otherwise have been worth trying, which is
to weight the objective toward whichever features carry the separation. There is
no such feature.

## The contract's own noise, measured 2026-08-10

Every delta in this session was being read against a noise floor estimated from
a single pair of measurements. Measured properly instead. score_features
truncates to min(n_human, n_synth), so which rows survive is decided by the
shuffle. One fixed generated feature matrix of 2421 rows, rescored under twelve
shuffles, same trajectories, same forest seed, same reference:

    0.6408 0.6396 0.6317 0.6296 0.6328 0.6414
    0.6342 0.6402 0.6334 0.6316 0.6336 0.6338

    mean 0.6352   sd 0.0041   range 0.0118

So at the sizes this workstream reports, a difference under about 0.008 is not a
result. Use two standard deviations, 0.008, as the minimum real difference at
n around 2400, and scale as one over the square root of n elsewhere.

Consequences for numbers already recorded. The moment continuation's 0.0083 sits
exactly on that line, which is why NULL was the right call and not a harsh one.
The pilot's 0.038 and the energy arm's first 0.062 are nine and fifteen standard
deviations clear. The gapsplit rungs were each measured over three seeds and
their reported standard errors, 0.0027 to 0.0054, are consistent with this once
the corpus resampling is added on top.

Anyone reading a single evaluation in this workstream should treat movements
under 0.008 as unmeasured rather than small.

## The energy objective works, and is worth a third of what the run printed, 2026-08-10

The first arm in this workstream to move the detector by more than its own
noise. It also produced a headline number that does not survive being measured
properly, and the correction is the more useful half of the entry.

WHAT WAS RUN

`w4_rollout.py --objective energy`, 300 steps from `event_ar_v2_s40000`, score
function gradient through the sampler against the energy distance between the
model's own generated feature distribution and the corpus one. Twelve of the
eighteen features are in the objective, six were held out before anything ran.

```
                          base    step75   step150  step225  step300
  detector               0.6812   0.6193   0.6219   0.6033   0.5949
  second detector        0.6898   0.6391   0.6428   0.6470   0.6108
  spread error, trained  0.3358   0.1085   0.0968   0.0902   0.1172
  spread error, held out 0.1942   0.0496   0.0558   0.0543   0.0547

  VERDICT CONFIRMED   peak 75C, 9.7 min cooling in 65.8 min wall
```

The registered falsifier said energy would stall where the moment objective did,
near 0.6281, with the held out ratio falling the same way. Neither happened.

WHY THE PRINTED 0.0863 IS NOT THE NUMBER

Two things inflate it, both found afterwards by repeating measurements that had
only ever been taken once.

The arm samples at `seq_len` equal to its cap of 160, which is a defective
setting. See the clock entry below. At that setting the base checkpoint scores
0.6746 averaged over five sampler draws, but the arm's single base draw read
0.6812, the highest of the five. At the correct setting the same checkpoint
scores 0.6301.

The arm's final evaluation was likewise one draw. Three independent draws of the
saved checkpoint at the same setting read 0.6208, 0.6170 and the run's own
0.5949, mean 0.6109. The run's was the lowest of the three and the trained model
turns out to be more than twice as variable between draws as the base was.

So the printed fall is a high base minus a low endpoint, both single draws.

WHAT THE ARM IS ACTUALLY WORTH

Measured on repeat draws at the setting the model was trained for, which is the
setting the ladder and the corpus floor are also on:

```
  base checkpoint            0.6301   3 draws, sd 0.0059
  energy checkpoint          0.5990   2 draws, agreeing to 0.0002
  the objective bought       0.0311   about 5 sd
```

For comparison on the same footing, the moment objective's continuation returned
0.0083, which was inside its noise. Energy is roughly four times better and is
the first thing here that clears its own error bar comfortably.

WHERE THAT PUTS IT ON THE LADDER

```
  energy checkpoint                  0.5990
  rung A, every marginal matched     0.6061
  rung B, plus the correlations      0.5826
  rung C, the corpus floor           0.5455
```

Training against the full distribution achieved slightly more than perfectly
correcting all eighteen marginals one at a time would have, and did not reach
the rung where the dependence between features is corrected too. 0.0535 remains
to the floor. The ladder already said the surviving separation is diffuse rather
than carried by any one feature, so the remaining room is in the dependence, and
that is where a distribution matching objective should be able to go.

THE GOODHART CHECK PASSED

The six held out features improved by 72 percent of their starting error against
65 percent for the twelve in the objective, a ratio of 0.64 against a registered
bar of 0.50. An objective that had found a shortcut through the twelve it was
scored on would show the reverse.

THINGS THAT LOOKED WRONG DURING THE RUN AND WERE NOT

The second detector drifted the wrong way for three evaluations, 0.6391 to
0.6470, while the first fell, then resolved to 0.6108 at the end. That
signature can be transient over a 150 step window.

At step 150 the number looked plateaued at 0.6219 after 0.6193. It then fell
0.019 by step 225. Read this arm off its endpoints, not its trajectory. Same
lesson as the moment arm and it bit again.

featloss never trended. It sat between 3.28 and 3.94 for all 300 steps while the
detector fell and the spread error fell by a factor of three. The statistic drops
the constant human to human term and resamples 512 human rows a step, so it is
noisy, but no trend at all across 300 steps is unexplained. Do not read featloss
as a convergence signal on this arm.

WHAT TO DO NEXT

Rerun from the base checkpoint with the clock fixed, so the arm trains on the
trajectories it will be judged on rather than on distorted ones. The detector was
still falling at step 300 with no plateau, so give it more than 300 steps. Every
evaluation should be at least two sampler draws.


## The sampling budget is part of the model's input, and every rollout arm got it wrong, 2026-08-10

A defect in this workstream's arms, not in the model. Found while chasing why two
scripts scored the same untrained checkpoint 0.045 apart.

WHAT IT IS

`prefix_state` returns six numbers describing where the model is in the
movement. The fifth is `idx / float(T)`, documented as "step index as a fraction
of the buffer", where T is the width of the buffer being generated into.
`EventARModel.sample` sets `T = seq_len or self.max_seq_len`. So the `seq_len`
passed to `sample` changes an input the model conditions on at every step.

Training buffers are 256 wide. `training/events_s2.npy` is (4028855, 256), the
dataset sets `T = self.max_len`, and `prefix_state` runs over the full width. The
model only ever saw `idx / 256`.

Every `w4_rollout` arm calls `model.sample(cond, seq_len=a.cap)` with cap 160, at
both the evaluation and the training rollout. At event 50 that tells the model it
is 31 percent through its buffer where training said 20 percent. The clock runs
about 1.6 times fast and the model behaves as though it is running out of room.

WHAT IT IS WORTH

Repeat sampler draws, same cond rows, same scoring shuffle:

```
  base checkpoint     seq_len 160   0.6746   sd 0.0060 over 5 draws
                      seq_len 256   0.6301   sd 0.0059 over 3 draws
                      the budget is worth 0.0445, 7.4 sd

  energy checkpoint   seq_len 160   0.6189   2 draws
                      seq_len 256   0.5990   2 draws
                      the budget is worth 0.0199
```

Not a truncation effect. Mean trajectory length was 55.1 against 56.2 on the base
and both hit the cap about 5 percent of the time. Cutting a 256 budget sample
down to 160 events costs only 0.008 of the 0.045; the rest is the clock.

Training under the fast clock halves the penalty but does not remove it, so the
model partly adapts to the distortion rather than being immune to it.

WHICH MEASUREMENTS ARE ON WHICH SETTING

`probe_runaway`, which produced `gen_tokens.npz`, calls `sample` with no
`seq_len` and gets 256. Everything built on it is correct: the w4_gapsplit
ladder, its rungs, and the corpus floor of 0.5455.

Every `w4_rollout` number is on the broken setting. The pilot, the moment
continuation and the energy arm all sampled at 160 for both purposes.

Within arm comparisons survive, because base and endpoint were measured the same
way. Absolute numbers do not transfer and no rollout arm number may be placed
against the ladder without being re measured.

HOW TO FIX IT

Sample at the model's own `max_seq_len` and cut afterwards if a shorter
trajectory is wanted, rather than sampling into a short buffer. Do not pass
`seq_len` below 256 to this checkpoint except to reproduce an old measurement.


## Three noise figures, and the one to actually use, 2026-08-10

They are not interchangeable and the difference between them changed a verdict.

```
  same rows reshuffled, forest randomness only          sd 0.0041
  disjoint halves, adds finite sample                   sd 0.0072
  repeat sampler draws, base checkpoint                 sd 0.0060
  repeat sampler draws, energy checkpoint at 160        sd 0.0141
```

The first was recorded here earlier and is the narrowest question you can ask:
one fixed set of generated rows, rescored. It says nothing about drawing
different trajectories, which is what every evaluation in a training run does.

Bootstrapping rows with replacement does NOT give the third figure. It reads
0.8360, because duplicate rows break the out of bag forest, the same failure
mode already recorded for self copy controls. Use disjoint halves.

The last line is the important one. A trained checkpoint is not as stable as the
base it came from, so the base's figure must not be borrowed for it. Any
comparison between two evaluations needs at least two sampler draws on each side,
and this workstream has quoted single draw numbers throughout.
