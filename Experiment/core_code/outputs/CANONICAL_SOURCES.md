# Canonical Sources for Paper Tables and Figures

Maps each table/figure in `Publication/paper/main_v3.tex` to its data source.
All paths relative to `Experiment/core_code/`.

## Tables

| Paper Table | Label | Source | Collapse Set | Seeds |
|---|---|---|---|---|
| Tab 1 (Absorber pairs) | `absorber_pairs` | `outputs/per_class_collapse_tls22_monthly/collapse_classes.csv` | 12 (support≥50) | — |
| Tab 2 (TTA baselines GN) | `tta_baselines` | TTA rows: `outputs/tta_multiperiod/{M7,M9,M12}/baselines_group_metrics.csv`; CARE row: `outputs/care_multiperiod/{M_2022_7,M_2022_9,M_2022_12}/aggregated_mean_std.csv` | Auto-discovered per period (M7: 2, M9: 7, M12: 10 classes) | 1 (TTA) / 3 (CARE) |
| Tab 3 (TTA baselines BN) | `bn_tta` | `outputs/care_bn_strict/aggregated_mean_std.csv` + `outputs/bn_tta_baselines_M12/` | 12 (support≥50) | 3 (CARE) |
| Tab 4 (Detection ablation) | `unsup_detect` | `outputs/detection_ablation/ablation_summary.json` | 14 (support>0) | — |
| Tab 5 (Main results) | `main_results` | TTA: `outputs/baselines_group_metrics_M12/`; Unified AL (entropy/coreset/random/margin): `outputs/unified_al_baselines/{entropy,coreset,random,margin}/aggregated_mean_std.csv`; BADGE all-class: `outputs/badge_allreplay_5seeds/aggregated_mean_std.csv` | 12 (support≥50) | 5 |
| Tab trigger (Detection trigger) | `trigger_eval` | `outputs/detection_trigger_eval/{M7,M9,M11,M12}/detection_summary.json` | per-period ground truth | — |
| Tab detect_necessity (Replay ablation) | `detect_necessity` | `outputs/detection_necessity_ablation/` + `outputs/fair_replay_budget/{all_k5,detected_k30,all_k1}/` | 12 (fixed) | 5 |
| Tab 6 (Per-class recovery) | `per_class` | `outputs/care_seed_0/per_collapse_class_m12.csv` (non-strict, absorber_margin@1000, oracle) | 12 (support≥50) | seed 0 |
| Tab 7 (Ablation) | `ablation` | `outputs/unified_ablation/{ft_only,ft_replay_noKD,full_care}/aggregated_mean_std.csv` | 12 (support≥50) | 5 |
| Tab 8 (Auto absorber) | `auto_absorber` | Oracle: `outputs/care_5seeds_strict_cnn/` (absorber_margin, 12-fixed eval); Auto: `outputs/care_auto_absorber_m11_to_m12/` (11-probe eval); Margin: `outputs/care_5seeds_strict_cnn/` (margin, 12-fixed eval) | **MIXED**: Oracle/Margin=12-fixed, Auto=11-probe | 5 |
| Tab 9 (Multi-period) | `multiperiod` | `outputs/care_multiperiod_allreplay/{M_2022_7,M_2022_9,M_2022_11,M_2022_12}/aggregated_mean_std.csv` | 12 (fixed, support≥50) | 3 |
| Tab 10 (E2E autonomous) | `e2e` | Labeled probe: `outputs/care_5seeds_strict_cnn/`; Autonomous: `outputs/autonomous_5seeds_strict/care/aggregated_mean_std.csv` | 12 (fixed eval set) | 5/5 |
| Tab 11 (Proto replay) | `proto_replay` | `outputs/proto_replay_comparison/{real_replay,proto_replay}/aggregated_mean_std.csv` | 12 (support≥50) | 3 |
| Tab 12 (Architecture) | `arch` | CNN: `outputs/care_5seeds_strict_cnn/`; Transformer: `outputs/care_transformer_multiperiod/M_2022_12/aggregated_mean_std.csv` | 12 (support≥50) | 5/3 |
| TTA lr sweep (inline) | — | `outputs/tta_lr_sweep/{lr_1e-3,lr_1e-4,lr_1e-5}/` | 12 (support≥50) | 1 |
| Training seed audit (inline) | — | `outputs/multiseed_audit/{trainseed0,trainseed1,trainseed2}/care/` | 12 (support≥50) | 3×3 |
| FT depth (inline Discussion) | — | `outputs/full_ft_baseline/seed_{0..4}/results_by_budget.csv` | 12 (support≥50) | 5 |
| QUICEXT-25 generalization | `quicext25` | `outputs/care_quicext25_canonical/aggregated_mean_std.csv` | 4 (QUICEXT-25 collapse classes) | 5 |

## Figures

| Figure | File | Source |
|---|---|---|
| Fig 1 | `fig_collapse_timeline.pdf` | `outputs/per_class_collapse_tls22_monthly/collapse_timeline.csv` |
| Fig 2 | `fig_ablation_bar.pdf` | `outputs/unified_ablation/*/aggregated_mean_std.csv` |
| Fig 3 | `fig_tta_failure.pdf` | `outputs/tta_multiperiod/{M7,M9,M11,M12}/` + `outputs/care_multiperiod_allreplay/*/` |
| Fig 4 | `fig_strategy_comparison.pdf` | `outputs/unified_al_baselines/` + `outputs/badge_allreplay_5seeds/` |
| Fig 5 | `fig_budget_sweep.pdf` | `outputs/care_5seeds_strict_cnn/aggregated_mean_std.csv` |

## Collapse Set Definitions

- **12 classes** (main evaluation): recall < 0.1 AND support ≥ 50 at M-2022-12
- **14 classes** (detection ground truth): recall < 0.1 AND support > 0 at M-2022-12
- **Auto-discovered**: per-period probe-based discovery (class counts vary by period)

## Aggregation

All `aggregated_mean_std.csv` use sample std (ddof=1) via `scripts/aggregate_seeds.py`.

## Superseded directories (not used in paper)

- `autonomous_fixed12_eval/` — superseded by `autonomous_5seeds_strict/`
- `autonomous_pipeline_final/` — 3-seed version, superseded by `autonomous_5seeds_strict/`
- `unsupervised_care_pipeline/` — uses detected (not fixed-12) eval set; superseded
- `unsupervised_collapse_detection_v4/` — superseded by `detection_ablation/`
- `care_multiperiod/` — superseded by `care_multiperiod_allreplay/` (was stable_absorber replay + auto-discovered eval)
- `ablation_strict/` — superseded by `unified_ablation/` (clean single-variable ablation with all-class replay)
- `al_baselines_strict/` — superseded by `unified_al_baselines/` (unified pipeline, all selectors use same replay+KD)
- `badge_5seeds_strict/` — superseded by `badge_allreplay_5seeds/` (all-class replay)
- `care_quic22_strict/` — superseded by `care_quicext25_canonical/` (newer dataset, all-class proto replay)
