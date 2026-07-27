# Whole-Trajectory Diffusion Pilot

Status: DRAFT, awaiting approval. No training has been run.

## Why this pilot exists

Four independent attempts to teach the current event-stream model to be less detectable have all failed the same way: distillation, DPO preference learning, RL against the detector, and conditioning-aim correction. The shared post-mortem is that the model is trained token by token, and a token-level objective cannot control statistics of the whole path. The detector's remaining edge lives in exactly those whole-path statistics, above all the variety of curvature along a trajectory (synthetic paths vary their curvature at roughly 0.55 to 0.60 of the human rate). The only thing that has ever closed that gap is selecting whole paths after the fact, which is why the 0.504 headline rests on selection, not on the model.

The fix that follows from the diagnosis: a model that generates the entire trajectory in one shot and is trained with losses that see the entire trajectory. Diffusion models do exactly this. They denoise the whole path at once and can predict the clean sample at every training step, which means a whole-path penalty (for example, "the curvature variety of this batch must match the human batch") attaches directly to the training signal. The per-token architecture has no place to attach such a penalty; a whole-path diffusion model does.

Honesty about the ceiling: no published mouse-trajectory generator reaches 0.50. The best published system still leaves detectors at 72 to 92 percent accuracy, and our current bare model already beats that literature. This pilot is genuine research with a real chance of landing at 0.55 to 0.65 rather than 0.50. Even that outcome has value: a stronger base model shrinks the pool a selection-based server needs. But nobody should read this plan as a recipe with a known answer.

## Why this is a pilot and not a full build

We are not starting from zero. The repo already contains a working whole-trajectory diffusion stack from earlier experiments:

- A whole-sequence denoiser with the best generative score in the repo (CANDI, AUC 0.752 unselected, polar representation), including two working samplers and a full 500-line training loop.
- A second whole-path transformer diffusion model and a flow-matching training loop, both functional.
- Ready-made training tensors: 3.74 million human trajectories, resampled and split, loading via memory-map.
- The evaluation harness needs zero changes; a new model plugs in as one experiment file.

Roughly 90 percent of the plumbing (noise schedules, samplers, conditioning, data loaders) is written and debugged. What the pilot actually builds is the new part: the whole-path auxiliary loss, and the training run that uses it. That is days of work, not weeks, and the gate below decides whether anything bigger ever gets built.

## The pilot, concretely

Starting point: the CANDI backbone with its polar representation (polar already beat cartesian 0.752 to 0.950 on this exact task). Retrain it with the standard denoising loss plus a batch-level curvature-moment loss: at each training step, compare the curvature mean and curvature spread of the batch of predicted clean paths against a matched batch of real human paths, and penalize the difference. This targets the diagnosed wall directly. The model must also emit whole-pixel coordinates, because we have proven the detector fingerprints any departure from the integer pixel lattice.

A learned latent space (the fancier version from the literature) is deliberately out of scope for the pilot. It is the one genuinely green-field component, and we should only pay for it if the simple version shows the auxiliary loss moves the needle.

Training cost per attempt is known from history: about 25 minutes per epoch on the 1 million trajectory subset, and CANDI reached its best score at epoch 21. So one pilot attempt is roughly 9 hours of GPU time, run as supervised 90-minute bursts. Budget two to three attempts (loss weight needs tuning): call it three to five supervised GPU days end to end.

## The gate

All pilot scoring uses the in-training validation humans, never the held-out eval set, and always N=2000 (small-N numbers read about 0.33 too low and are banned from decisions).

Measured after training, on 2000 unselected single trajectories:

- PASS: AUC at or below 0.62 (clearly beats the current bare model's 0.647 and the old CANDI's 0.752) AND the curvature-variety ratio recovers to at least 0.8 of human (from today's 0.55 to 0.60). Passing means the whole-path loss reaches the thing that killed every per-token attempt, and a full build (bigger model, full 4M corpus, possibly the latent version) is justified.
- FAIL: either condition missed after the tuning attempts above. We park the direction, write up the negative, and put the effort into the serve-once pool logistics, which works today at 0.504.

The gate is deliberately not 0.50. The pilot's job is to prove the mechanism (whole-path losses control whole-path statistics), not to hit the final number in one step.

## GPU safety rules (unchanged from the RL pilot)

- Supervised sessions only, never unattended, never overnight.
- 90-minute bursts with cooldowns, per-minute temperature log, hard stop at 83C.
- This machine has bluescreened under sustained load four times; checkpoints save every epoch so a crash costs at most 25 minutes.

## The safety net

Nothing in this pilot touches the delivery path that already works: the serve-once pool at 0.504 (selection over a large pre-generated pool, each member served once per detector window, integer translation only). That stays available regardless of how the pilot lands.

## What approval means

Approving this document authorizes: writing the training script changes (auxiliary loss on the CANDI backbone), then the first supervised GPU burst schedule, reported back after each burst. It does not authorize a multi-week build, a latent-space model, or any change to the evaluation protocol.
