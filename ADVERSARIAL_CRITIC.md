# Learned Whole-Path Critic

Status as of 26 July 2026: CLOSED. Phase 0 passed, Phase 1 ran and failed, Phase 2 will not be funded. Everything below Phase 1 is kept as a record of what was designed, not as work that is queued.

The line that used to sit here said no training had been run. That was wrong by the time anyone read it, and it cost a later session a day of rediscovery. What has actually been run:

- Phase 0 (research/phase0_critic.py, 19 July). Critic reads the raw path. OOF AUC 0.632 against the RF's 0.757, and nearly blind to the RF's top tells. Fails the gate on its own.
- Phase 0b (research/phase0b_critic.py, 19 July). Same critic handed speed, acceleration, jerk, curvature and angular velocity as explicit channels, which removes the confound that a transformer cannot compute a second derivative from 4000 noisy samples. OOF AUC 0.792, above the RF. Passes.
- Coverage check (research/w3_critic_coverage.py, 26 July). Phase 0b's headline is a single pooled number, and the gap is not pooled: it lives in a quarter of human movement (see research/w3_missing_paths.py). Retrained against the event-model arm Phase 1 would fine-tune, the critic reads 0.843 on that quarter against 0.544 on the rest, so it is aimed at the deficit. The two teachers agree on only 44 percent of which paths are missing, so the critic is contributing its own signal rather than echoing the examiner.

- Phase 1 (26 July). Ran deliberately against the recommendation on record, because the rig was cheap and the prediction deserved a direct test. The generator was fine-tuned through the full K=200 differentiable sampler to fool the frozen Phase 0b critic, with a flow-matching anchor against drift. One 90 minute burst, 617 steps. Training telemetry looked like success: the fool loss halved and the critic's logits moved firmly toward the human side. On a fresh 2000-trajectory sample the RF detector read 0.766 against a 0.757 baseline, no better. The critic itself, tested out of sample on the new model's own output, still separated at 0.80. The apparent training-time win did not survive a clean test even against the judge it was optimised on. Full account in EXPERIMENTS.md; the first attempt at this run was invalid on a gradient clip inherited from the old harness and is recorded there too.

Phase 2 is not funded. Phase 1 is the fifth independent negative for fine-tuning this generator against a fixed target, after direct moment matching, two conditioning corrections, and the RL pilot. The design below is left in place because the next architecture will face the same stability questions, not because anyone should start it.

## Where this sits in the story

The pilot proved two things over five training bursts. One is a genuine positive we get to keep: the hard piece of machinery, training the model through its own full generation process on this laptop, works and is stable. The other is the negative that sent us here: every attempt to make the paths more human by matching a checklist of hand-measured features hits the same wall. Push the model to match one feature and the detector simply moves its attention to a different feature we were not matching. Some of those features have no smooth form the training can even reach. Matching features one by one tops out at roughly where we started.

There is exactly one idea on the list that was never properly tried, and it is the one that sidesteps that wall by design. Instead of us naming the features to match, we train a second network, the critic, to look at a whole path and learn for itself what makes ours look non-human. The generator then learns to fool it. Because the critic is not limited to a fixed checklist, the generator cannot escape by relocating its tell to an unlisted feature. The critic learns the new tell too.

This is the most interesting idea left and also the most dangerous one. It is close kin to the reinforcement-learning approach we already parked, and it fails in the same family of ways if built carelessly: the two networks can chase each other in circles, or the generator can find a cheap trick that fools the critic without actually looking more human. The whole design below is about earning the upside while spending real effort on not falling into those holes.

## The honest expectation

No published system that generates individual mouse paths reaches the undetectable mark we are aiming at. The best in the literature still gets caught most of the time. Our own selection-based delivery, the pool at 0.504, is already better than anything published. So this build is real research with a real chance of landing at 0.55 to 0.65 rather than at 0.50. That would still be a genuine result and a large improvement on the raw model, but it is not a sure thing and I will not pretence otherwise. The pool at 0.504 remains the working deliverable no matter how this lands.

## The plan, in phases, cheapest first

The single most important design choice here is to spend a little before spending a lot. A full adversarial build is weeks. We do not commit those weeks until a cheap early test says the gradient actually points somewhere useful.

### Phase 0: the critic alone, no adversarial loop yet

Freeze the generator. Generate a fixed pile of our paths, take a matching pile of human paths, and train just the critic to tell them apart. This is ordinary supervised training of one network, cheap and stable, no chasing.

Phase 0 answers the question that decides everything: does a learned critic looking at whole paths find a separation that the real detector also cares about? We check this two ways. First, can the critic separate our paths from human at all, and how confidently. Second, and this is the real test, when the critic points at what it dislikes, does fixing it move the number the real detector reports, or is the critic obsessed with something the real detector ignores. If the critic learns a tell that lives in a space the real scorer never looks at, the whole loop would optimise a mirage, and we stop here having spent a day, not a month.

Phase 0 is the go/no-go. It is a few hundred paths and one small network, on the order of a day.

### Phase 1: one short closed loop

Only if Phase 0 passes. Take the trained critic, unfreeze the generator, and do a small number of generator updates against it through the differentiable rig we already built. No alternating schedule yet, no full stability toolkit, just enough to confirm the generator can actually reduce the critic's score and that doing so moves the real detector number in the right direction on a proper measurement. This is a smoke test of the coupling, not the finished system. A day or two.

### Phase 2: the full adversarial build

Only if Phase 1 shows the coupling works. This is the multi-week piece: the two networks trained together with the full set of stability safeguards, tuned over many supervised bursts. Budgeted honestly as weeks, gated at the end by the same measurement we have used throughout. We do not start Phase 2 without a fresh look at the Phase 1 result together.

## The design decisions, settled

These are the choices that carry the build, recorded so Phase 0 starts clean.

- The critic reads the raw path directly, as a sequence, and learns its own features. Earlier critic attempts in this program only ever looked at the same 18 hand-measured numbers through a noisy connection, which is why they never worked. This one does not repeat that.
- The generator side is already built and proven. The critic's training signal flows back into the generator through the full-chain differentiable sampler from the pilot. That machinery is validated and stays exactly as it is, including every safety guard baked into it.
- The stability toolkit is not optional. Because this is the RL failure geometry, Phase 2 carries the established anti-collapse measures as a set, not a pick-one: batch-aware critic design so the generator cannot collapse to one clever path, a penalty that keeps the critic's gradients smooth, memory of past states so the pair cannot cycle, and a constant pull back toward the original model so it does not forget how to make a plausible path at all.
- The critic must judge in a space the real detector respects. The clearest lesson from the pilot is that the smooth path the model emits and the resampled path the detector actually measures are not the same thing, and a signal in one can be invisible in the other. So the critic is fed the path in the form the real scorer sees, not the raw model output. This is the specific trap that Phase 0 is built to catch early.

## The gate

Unchanged in spirit and measured exactly as before. On 2000 freshly generated single paths, judged against the in-training validation humans, never the held-out eval set, always at that size.

- Phase 0 go: the critic separates our paths from human, and the thing it objects to is visibly connected to the real detector's number, not an artefact the detector ignores.
- Phase 1 go: a short loop reduces the critic's score and moves the real detector number the right way on a proper N=2000 measurement.
- Phase 2 pass: the real detector number drops clearly below today's 0.76, with no single feature quietly collapsing to a new easy tell. The whole-pixel serving number is reported alongside every time, never hidden, but it remains a separate serving-layer matter and does not gate the mechanism.
- Fail at any phase: park the model direction, write up the negative cleanly, and put the effort into the serve-once pool logistics, which works today at 0.504.

## GPU safety rules, unchanged

- Supervised sessions only, never unattended, never overnight.
- 90-minute bursts with cooldowns, per-minute temperature log running unsandboxed, hard stop at 83C.
- This machine has bluescreened under sustained load. Checkpoints save every 25 to 50 steps. The original model file is copied before any run and checksum-verified untouched after.

## The safety net, unchanged

Nothing here touches the delivery path that already works: the serve-once pool at 0.504. It stays available regardless of how this build lands.

## What approval means

Approving this document authorizes Phase 0 only, the cheap critic-alone diagnostic, reported back with the numbers before anything else starts. Phase 1 and Phase 2 are separate go decisions, each taken after seeing the prior phase's result. This staging is deliberate: it caps what we can waste at roughly a day if the idea does not hold.
