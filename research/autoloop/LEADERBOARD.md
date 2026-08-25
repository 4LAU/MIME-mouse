# Autoloop Leaderboard

_Regenerated 2026-08-25T05:05:30+00:00_

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

## fc_v3_feat_film_train

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## jq_queued_mmd_train

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## phase0b_critic_finetune

n_trials (tier1, status=ok): 2

| rank | run_id | tier1 AUC | gap-to-0.5 | tier2 status | tier2 AUC | collapse | notes |
|---|---|---|---|---|---|---|---|
| 1 | phase0b_critic_finetune_2026-07-20T063005+0000_1dd2b084 | 0.7573 | 0.2573 | UNCONFIRMED | n/a | ok | BACKFILL, no fresh compute. Published baseline reference for |
| 2 | phase0b_critic_finetune_2026-07-20T063005+0000_07c11584 | 0.7662 | 0.2662 | UNCONFIRMED | n/a | ok | BACKFILL, no fresh compute. Phase 1 fine-tune of candi_polar |

## w3_conditional_gate

n_trials (tier1, status=ok): 1

| rank | run_id | tier1 AUC | gap-to-0.5 | tier2 status | tier2 AUC | collapse | notes |
|---|---|---|---|---|---|---|---|
| 1 | w3_conditional_gate_2026-07-27T073545+0000_f28ce8d0 | 0.5491 | 0.0491 | UNCONFIRMED | n/a | ok | POSITIVE, and the first localisation this session that survi |

## w3_coupling_gate

n_trials (tier1, status=ok): 2

| rank | run_id | tier1 AUC | gap-to-0.5 | tier2 status | tier2 AUC | collapse | notes |
|---|---|---|---|---|---|---|---|
| 1 | w3_coupling_gate_2026-07-27T072736+0000_76303ff4 | 0.6507 | 0.1507 | UNCONFIRMED | n/a | ok | NEGATIVE. Repairing every pairwise rank dependence to human, |
| 2 | w3_coupling_gate_2026-07-27T073545+0000_0f13f30b | 0.6507 | 0.1507 | UNCONFIRMED | n/a | ok | CORRECTED RERUN of row ...76303ff4, which is superseded. The |

## w3_duration_response

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w3_efficiency_gate

n_trials (tier1, status=ok): 1

| rank | run_id | tier1 AUC | gap-to-0.5 | tier2 status | tier2 AUC | collapse | notes |
|---|---|---|---|---|---|---|---|
| 1 | w3_efficiency_gate_2026-07-27T080921+0000_b8cfaca3 | 0.5479 | 0.0479 | UNCONFIRMED | n/a | ok | NEGATIVE for the single-feature story, and it decomposes the |

## w4

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_advmoment

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_advpath

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_advpath_noanchor

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_advtime

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_ar_v1

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_arrangement

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_attenuation

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_bracket

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_budget

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_bulktail

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_channels

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_command_ceiling

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_command_ceiling_v2

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_condshare

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_copula_order

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_cosse

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_coupling

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_critic

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_critic_ablate

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_deepq

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_detcap

n_trials (tier1, status=ok): 2

| rank | run_id | tier1 AUC | gap-to-0.5 | tier2 status | tier2 AUC | collapse | notes |
|---|---|---|---|---|---|---|---|
| 1 | w4_detcap_2026-08-18T062901+0000_5eea7fe2 | 0.5881 | 0.0881 | UNCONFIRMED | n/a | ok | SUPERSEDES w4_detcap_2026-08-18T052254+0000_a7a18a5d, whose  |
| 2 | w4_detcap_2026-08-18T052254+0000_a7a18a5d | 0.5982 | 0.0982 | UNCONFIRMED | n/a | ok | NEGLIGIBLE and POWERED. auc_rf_oob 0.5982 is the CONTRACT RU |

## w4_drawvar

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_dtstruct

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_dttilt

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_durmatch

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_durmech

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_dursrc

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_e1chan

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_ess

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_estimator

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_estimator_rel

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_evcount

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_evprice

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_expertgap

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_featcond

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_featmap2

n_trials (tier1, status=ok): 1

| rank | run_id | tier1 AUC | gap-to-0.5 | tier2 status | tier2 AUC | collapse | notes |
|---|---|---|---|---|---|---|---|
| 1 | w4_featmap2_2026-08-14T075118+0000_58dbd59b | 0.6099 | 0.1099 | UNCONFIRMED | n/a | ok | P1 REFUTED, P2 CONFIRMED, record 8 of 19. DECIDES the fork t |

## w4_firstev

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_firsthead

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_floor

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_gradsnr

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_gradsnr2

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_gradsnr_check

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_headcap

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_headmlp

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_histfeat

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_k0power

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_kfill

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_klanchor

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_lattice_emitter

n_trials (tier1, status=ok): 1

| rank | run_id | tier1 AUC | gap-to-0.5 | tier2 status | tier2 AUC | collapse | notes |
|---|---|---|---|---|---|---|---|
| 1 | w4_lattice_emitter_2026-07-27T180857+0000_842e1356 | 0.7320 | 0.2320 | UNCONFIRMED | n/a | ok | NEGATIVE, AND THE NULL ARM CARRIES THE FINDING. Tested the p |

## w4_manifold_projection

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_margfix

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_marginal

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_marginal_vs_coupling

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_margsurr

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_margtilt

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_mmd_alignment

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_mmd_blindness

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_mmd_queue

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_mmd_symmetric

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_mmd_term_balance

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_ms_lattice

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_mserve

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_nardiff

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_nodur

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_objective_vs_metric

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_order

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_order_resid

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_pairdep

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_pairq

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_pairsplit

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_pipeline_agreement

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_placebo

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_placebo2

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_placebo3

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_plan_space

n_trials (tier1, status=ok): 1

| rank | run_id | tier1 AUC | gap-to-0.5 | tier2 status | tier2 AUC | collapse | notes |
|---|---|---|---|---|---|---|---|
| 1 | w4_plan_space_2026-07-27T094052+0000_d9c80213 | 0.5595 | 0.0595 | UNCONFIRMED | n/a | ok | THE PLAN LAYER IS NOT WHERE THE DEFICIT IS, which kills the  |

## w4_poskl

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_prefix

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_prefixch

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_prefixcond

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_qladder

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_qwarm

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_redundancy

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_refine

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_renderer_endpoint

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_residual

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_robust_coverage

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_rollout_clock

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_rollout_critic

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_rollout_energy

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_rollout_token

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_rollout_zbuf

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_rowmap

n_trials (tier1, status=ok): 1

| rank | run_id | tier1 AUC | gap-to-0.5 | tier2 status | tier2 AUC | collapse | notes |
|---|---|---|---|---|---|---|---|
| 1 | w4_rowmap_2026-08-18T022511+0000_e78bbaad | 0.5895 | 0.0895 | UNCONFIRMED | n/a | ok | REPLICATION at an already recorded config, NOT a new candida |

## w4_scale_audit

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_seedvar

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_sf_mmd_pilot

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_shape

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_sharpness

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_single_trajectory

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_softdec_check

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_softdec_check2

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_softdec_clip

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_softdec_critic

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_softdec_pair

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_softdec_pair2

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_softdec_tail

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_softdec_tau

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_spikerate

n_trials (tier1, status=ok): 1

| rank | run_id | tier1 AUC | gap-to-0.5 | tier2 status | tier2 AUC | collapse | notes |
|---|---|---|---|---|---|---|---|
| 1 | w4_spikerate_2026-08-10T234356+0000_2690d04a | 0.6412 | 0.1412 | UNCONFIRMED | n/a | ok | NULL. The sub ms wait rate is the mechanism behind the 2.5x  |

## w4_statecoord

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_statevisit

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_step0

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_stillcal

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_stillprice

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_straight

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_submove

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_submovement_ceiling

n_trials (tier1, status=ok): 1

| rank | run_id | tier1 AUC | gap-to-0.5 | tier2 status | tier2 AUC | collapse | notes |
|---|---|---|---|---|---|---|---|
| 1 | w4_submovement_ceiling_2026-07-27T095747+0000_e6e3bff8 | 0.8335 | 0.3335 | UNCONFIRMED | n/a | ok | NO SMOOTH PARAMETERISATION AT ANY TRACTABLE SIZE PRESERVES W |

## w4_tailtest

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_tenseed

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_texcover

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_texture_sensitivity

n_trials (tier1, status=ok): 1

| rank | run_id | tier1 AUC | gap-to-0.5 | tier2 status | tier2 AUC | collapse | notes |
|---|---|---|---|---|---|---|---|
| 1 | w4_texture_sensitivity_2026-07-27T094052+0000_7613625c | 0.8287 | 0.3287 | UNCONFIRMED | n/a | ok | THE CONTRACT IS A TEXTURE INSTRUMENT, NOT A SHAPE INSTRUMENT |

## w4_threehead

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_ticknull

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_tickstruct

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_token_ceiling

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_train_serve_gap

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_trunkcap

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_ttp_repair

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_variety_mechanisms

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_variety_vs_steering

n_trials (tier1, status=ok): 1

| rank | run_id | tier1 AUC | gap-to-0.5 | tier2 status | tier2 AUC | collapse | notes |
|---|---|---|---|---|---|---|---|
| 1 | w4_variety_vs_steering_2026-07-27T185816+0000_1252df17 | 0.6624 | 0.1624 | UNCONFIRMED | n/a | ok | TWO FINDINGS, THE SECOND ONE REFRAMES THE PROGRAM. (1) The A |

## w4_views

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

## w4_warmtemp

n_trials (tier1, status=ok): 0

_(no completed tier1 runs)_

