# Autoloop Leaderboard

_Regenerated 2026-07-27T05:55:52+00:00_

**Best-of-N tier1 scores are selection-biased; only tier2 confirmations are quotable.**

Program metric: RF-OOB AUC vs data/human_val_features_grpo.npy (n_estimators=100, oob_score, random_state=42). Target is chance (0.50), not minimum -- ranked by distance from 0.50, not by raw value. `human_eval_features.npy` is never used inside this loop.

## W0

n_trials (tier1, status=ok): 3

| rank | run_id | tier1 AUC | gap-to-0.5 | tier2 status | tier2 AUC | collapse | notes |
|---|---|---|---|---|---|---|---|
| 1 | W0_2026-07-20T063006+0000_56691256 | 0.5391 | 0.0391 | UNCONFIRMED | n/a | ok | BACKFILL, no fresh compute. Per-item SIR judge floor at K=32 |
| 2 | W0_2026-07-20T063005+0000_9dacaec5 | 0.5646 | 0.0646 | UNCONFIRMED | n/a | ok | BACKFILL, no fresh compute. Per-item SIR judge floor at K=16 |
| 3 | W0_2026-07-20T063005+0000_abab31af | 0.5809 | 0.0809 | UNCONFIRMED | n/a | ok | BACKFILL, no fresh compute. Per-item SIR judge floor at K=8  |

## W1_scratch

n_trials (tier1, status=ok): 1

| rank | run_id | tier1 AUC | gap-to-0.5 | tier2 status | tier2 AUC | collapse | notes |
|---|---|---|---|---|---|---|---|
| 1 | W1_scratch_2026-07-21T033224+0000_694e74a5 | 0.8331 | 0.3331 | UNCONFIRMED | n/a | ok | W1 GATE = NEGATIVE. Pre-registered gate: one-shot N=2000 RF- |

## W2_stat_guided_probe

n_trials (tier1, status=ok): 3

| rank | run_id | tier1 AUC | gap-to-0.5 | tier2 status | tier2 AUC | collapse | notes |
|---|---|---|---|---|---|---|---|
| 1 | W2_stat_guided_probe_2026-07-20T071449+0000_68626b4c | 0.7253 | 0.2253 | UNCONFIRMED | n/a | ok | Tuning iteration testing batch=256 throughput. AUC improved  |
| 2 | W2_stat_guided_probe_2026-07-20T071449+0000_5ae1a299 | 0.7637 | 0.2637 | UNCONFIRMED | n/a | ok | First full smoke run with control+validation enabled (defaul |
| 3 | W2_stat_guided_probe_2026-07-20T071449+0000_eac56290 | 0.8134 | 0.3134 | UNCONFIRMED | n/a | ok | Tuning iteration (skip-control/skip-validation), lr=0.05 m_s |

## W3_P1

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## W3_P1_geoadv

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## W3_P2

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## W3_P3

n_trials (tier1, status=ok): 1

| rank | run_id | tier1 AUC | gap-to-0.5 | tier2 status | tier2 AUC | collapse | notes |
|---|---|---|---|---|---|---|---|
| 1 | W3_P3_2026-07-22T091532+0000_33e21a15 | 0.9815 | 0.4815 | UNCONFIRMED | n/a | ok | FAIL against the pre-registered gate by a wide margin (0.98  |

## W3_groundwork

n_trials (tier1, status=ok): 3

| rank | run_id | tier1 AUC | gap-to-0.5 | tier2 status | tier2 AUC | collapse | notes |
|---|---|---|---|---|---|---|---|
| 1 | W3_groundwork_2026-07-26T041909+0000_0893dd67 | 0.5833 | 0.0833 | UNCONFIRMED | n/a | ok | Degeneracy control on the 0.58 product number. Contract colu |
| 2 | W3_groundwork_2026-07-26T041909+0000_86be14c7 | 0.6500 | 0.1500 | UNCONFIRMED | n/a | ok | Degeneracy control on the arrival tax. Contract column repro |
| 3 | W3_groundwork_2026-07-26T055107+0000_426caf73 | 0.7525 | 0.2525 | UNCONFIRMED | n/a | ok | Degeneracy control on CANDI. The published 0.752 reproduces  |

## phase0b_critic_finetune

n_trials (tier1, status=ok): 2

| rank | run_id | tier1 AUC | gap-to-0.5 | tier2 status | tier2 AUC | collapse | notes |
|---|---|---|---|---|---|---|---|
| 1 | phase0b_critic_finetune_2026-07-20T063005+0000_1dd2b084 | 0.7573 | 0.2573 | UNCONFIRMED | n/a | ok | BACKFILL, no fresh compute. Published baseline reference for |
| 2 | phase0b_critic_finetune_2026-07-20T063005+0000_07c11584 | 0.7662 | 0.2662 | UNCONFIRMED | n/a | ok | BACKFILL, no fresh compute. Phase 1 fine-tune of candi_polar |

