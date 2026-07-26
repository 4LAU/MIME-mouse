# Handoff: W1 closed, W3 opens on the arrival tax

Read this whole file before touching anything. Repo is MIME-mouse on WSL2
Ubuntu. Nothing is mid-flight; the GPU is idle and no runs are pending.

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

## The central finding, measured 2026-07-20

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
See W3_PROPOSAL.md; the fix is per-step residual re-conditioning, not more
endpoint information.

Two consequences. Post-hoc correction is never free, not even at 2px, so those
numbers are a floor and not a solution. And because cost scales with correction
size, arrival has to be part of what the model is asked to do rather than a
fix applied afterwards.

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

Still unmeasured: the dense-125Hz family (CANDI's 0.752, the older CFM and DDPM
line) emits one value per 8 ms slot and rounds, so it never carries the exact
structure real paths do. Those numbers have not been read through this panel.

## Numbers are not currently comparable across model families

CANDI applies endpoint correction as standard (CANDI_CORRECT=rotate,
experiments/candi.py:235), so its 0.504 and 0.513 headline numbers are measured
on paths that arrive and include the cost of arriving. Every event-stream
number, including W0's 0.539 fallback and W1's 0.654 one-shot baseline, pays
none of that cost and is flattered by roughly the tax above. Do not compare
across families without stating which side pays.

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
  W3_groundwork_...ce210375, ...b7753a76. PLAN.md's original "W3 = SCALE" is
  superseded: scaling would spend money against the wrong diagnosis.

## Next steps, in order

1. Decision for L first. The packageable product today is the K=32 corrected
   fallback at 0.58 to 0.59 with exact pixel arrival, about a second per
   request. Every cheap improvement lever on top of the current model family
   has now been measured and is flat: P1 conditioning (six cycles), P2
   character latent plus guidance, pool mixing, correction schemes. Getting
   materially below 0.58 means designing and training a new architecture
   (P3), which is real design work and GPU time, not another tuning cycle.
   The fork: ship on 0.58, or fund P3.
2. If P3 goes ahead, the P1 and P2 post-mortems jointly specify it. The
   model must be natively endpoint-conditioned, because feedback channels
   bolted onto this trunk are ignored; the trunk plans the whole path from
   its static conditioning. And it must be dispersion-calibrated by
   construction, because a learned character latent works as a
   representation but this decoder squashes any character command about
   five to one, and guidance only partially undoes that before going
   off-manifold. Drafting the proposal costs nothing and needs no sign-off.
3. Anything needing cloud spend stops for L sign-off. Local GPU does not.

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

Nothing from this work is committed. PLAN.md is modified (new section "K=1
MANDATE AND THE ENDPOINT TAX"). HANDOFF_W1.md line 18 was reworded for
precision. Untracked: HANDOFF_W1.md, HANDOFF_W3.md,
research/w1_oneshot_score.py, research/w3_landing_price_results.json,
research/w3_landing_cache.pkl, research/autoloop/. Add files individually.

Note: ~120 tracked files show as modified in git status. That is a line-ending
artifact of the Windows mount, not real edits. Do not sweep them into a commit.

## Open questions for L

- Cloud budget ceiling for W3, if a training run eventually needs one.
- Whether to package the fallback as a usable product now, in parallel with
  research. ANSWERED 2026-07-21 (row W3_groundwork_...e7f67c96): with exact
  arrival enforced and selection run on the corrected candidates it reads
  about 0.58 at roughly 1s/request, so it is packageable; see W3_PROPOSAL.md.
