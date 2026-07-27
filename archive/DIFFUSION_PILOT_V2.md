# Whole-Trajectory Diffusion Pilot, Version 2

Status: SCOPING, awaiting approval. No training has been run. This document replaces the training approach in DIFFUSION_PILOT.md; the gate and the safety net from that document still stand.

## Why there is a version 2

Version 1 added a whole-path curvature loss to the CANDI diffusion model and fine-tuned it. The training ran cleanly, but the model's actual generated trajectories did not change: curvature variety stayed at 0.58 of human, and detectability stayed at 0.77, both statistically unchanged from before.

The failure has a specific, understood cause. The loss measured curvature on a one-step shortcut estimate of the finished path, computed cheaply inside each training step. By that shortcut measure the model already looked human-like from the first epoch, so the loss concluded there was nothing to fix and pushed almost nothing. The real deficit does not live in the one-step shortcut. It builds up over the model's full multi-step generation process, which the shortcut never looks at. In plain terms: we graded the model on a practice test it was already passing, while the real exam went untouched.

This is not a tuning problem. Turning the loss strength up does nothing when the loss already believes it is passing. The fix has to make the training signal look at the genuinely generated path, not the shortcut.

## What the literature survey concluded

A survey of current techniques (2023 through 2026) for attaching a whole-path objective to a diffusion model's real output returned a clear front-runner and a clear ordering.

The front-runner is called truncated backpropagation, or a differentiable sampler tail. The idea is small and reuses almost everything already built. Generation runs in many small steps. Instead of the one-step shortcut, we let the loss watch the last handful of real generation steps (call it the last four to sixteen), and grade curvature on what those steps actually produce. The early steps run exactly as they do today with no change. Only the tail carries a training signal back into the model. This is the standard, published way to fine-tune a diffusion model against a goal measured on its output without paying to watch every single step, which would exceed this laptop's memory.

The same survey confirmed three things worth stating plainly:

- The closest published success story to version 1 (a motion-generation model that applies a geometric loss at every step) works precisely because its target is a local, single-step property. Ours is a whole-path property. That is the exact distinction that sank version 1, and it is a property of the statistic we chose, not a flaw in the diffusion approach.
- A richer, well-established family of whole-path distance measures (path signatures, and distribution-matching losses purpose-built for trajectory data) can be swapped in on the very same machinery at low extra cost, and is worth trying as a second variant.
- The reinforcement-learning route (reward the model for batch variety) is very likely to repeat the RL negative we already parked, for the same underlying reason, and is not recommended.

## The plan, concretely

### Phase 0: a diagnostic that needs no training

Before touching the training code, instrument the existing generator to report curvature variety not just at the finished path but at several intermediate points of its generation process, with the guidance machinery toggled on and off. This is a few hundred generated paths, no fine-tuning, no model changes, on the order of an hour of GPU.

Phase 0 answers the one question that decides whether the whole direction is worth building: where does the curvature collapse actually happen? If it concentrates in the last few generation steps, the differentiable-tail fix is well targeted and cheap. If it is smeared across the entire process, the tail would have to be impractically long, and the honest recommendation flips to either a heavier multi-week build or parking the model direction. Phase 0 is the real go/no-go, and it is nearly free.

### Phase 1: the pilot build (only if Phase 0 passes)

Make the last few generation steps differentiable, attach the existing curvature loss to what they actually produce instead of the one-step shortcut, and fine-tune in the usual supervised 90-minute bursts. Budget two to four days of build plus a handful of bursts. As a second variant on the same machinery, swap in the path-signature distance measure and compare.

Everything here reuses the current model, the current data, and the current evaluation. It is days of work, not weeks.

## The gate

Unchanged in spirit from version 1, measured the way you approved last time: on 2000 freshly generated single trajectories, judged against the in-training validation humans, never the held-out eval set, always at N=2000.

- Phase 0 go: the curvature collapse is concentrated enough in the late generation steps that a differentiable tail of practical length can reach it. This is a judgment read on the diagnostic curve, reported to you in plain terms with the numbers.
- Phase 1 pass: curvature variety recovers to at least 0.8 of human on generated output, and detectability drops clearly below today's 0.76. The whole-pixel serving number gets reported alongside every time, never hidden, but it is a separate serving-layer problem and does not gate the mechanism.
- Fail: variety does not move after the pilot bursts. Park the model direction, write up the negative, and put the effort into the serve-once pool logistics, which works today at 0.504.

## What is deliberately out of scope

Two credible but expensive directions are named and parked, to be raised only if the cheap pilot earns them:

- Distilling the model down to a few-step generator so the whole path becomes cheap to watch. A multi-week build with its own convergence risk. Only worth it if Phase 0 shows the collapse is too spread out for a short tail.
- A learned sequence-level adversarial critic, the one genuinely untried adversarial idea. Conceptually the most interesting, the most expensive, and the most prone to training instability. It also depends on the same differentiable-tail machinery the pilot builds first. Sequence it after the pilot, never instead of it.

## GPU safety rules, unchanged

- Supervised sessions only, never unattended, never overnight.
- 90-minute bursts with cooldowns, per-minute temperature log, hard stop at 83C.
- This machine has bluescreened under sustained load; checkpoints save every epoch, and the source model file is copied before any run and checksum-verified untouched after.

## The safety net, unchanged

Nothing here touches the delivery path that already works: the serve-once pool at 0.504. It stays available regardless of how the pilot lands.

## What approval means

Approving this document authorizes Phase 0, the no-training diagnostic, and if it passes, the Phase 1 differentiable-tail pilot build plus its supervised bursts, reported back after each. It does not authorize the distillation build, the adversarial critic, or any change to the evaluation protocol. Those remain separate decisions.
