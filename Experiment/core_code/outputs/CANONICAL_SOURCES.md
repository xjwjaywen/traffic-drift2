# Canonical Sources for Paper Tables and Figures

Maps each table/figure in `Publication/paper/main_v2.tex` to its data source.
All paths relative to `Experiment/core_code/`.

## Tables

| Paper Table | Label | Source | Collapse Set | Seeds |
|---|---|---|---|---|
| Tab 1 (Absorber pairs) | `absorber_pairs` | `outputs/per_class_collapse_tls22_monthly/collapse_classes.csv` | 12 (support≥50) | — |
| Tab 2 (TTA baselines GN) | `tta_baselines` | TTA rows: `outputs/tta_multiperiod/{M7,M9,M12}/baselines_group_metrics.csv`; CARE row: `outputs/care_multiperiod/{M_2022_7,M_2022_9,M_2022_12}/aggregated_mean_std.csv` | Auto-discovered per period (M7: 2, M9: 8, M12: 12 classes) | 1 (TTA) / 3 (CARE) |
| Tab 3 (TTA baselines BN) | `bn_tta` | `outputs/care_bn_strict/aggregated_mean_std.csv` + `outputs/bn_tta_baselines_M12/` | 12 (support≥50) | 3 (CARE) |
| Tab 4 (Detection ablation) | `unsup_detect` | `outputs/detection_ablation/ablation_summary.json` | 14 (support>0) | — |
| Tab 5 (Main results) | `main_results` | TTA: `outputs/baselines_group_metrics_M12/`; FT-Head/FT+Replay: `outputs/ablation_strict/{ft_only,ft_replay}/`; Active: `outputs/al_baselines_strict/aggregated_mean_std.csv`; BADGE: `outputs/badge_5seeds_strict/aggregated_mean_std.csv` | 12 (support≥50) | 5 |
| Tab 6 (Per-class recovery) | `per_class` | `outputs/care_5seeds_strict_cnn/seed_0/per_collapse_class_m12.csv` | 12 (support≥50) | seed 0 |
| Tab 7 (Ablation) | `ablation` | `outputs/ablation_strict/{ft_only,ft_replay,full_care}/aggregated_mean_std.csv` | 12 (support≥50) | 5 |
| Tab 8 (Auto absorber) | `auto_absorber` | Oracle: `outputs/care_5seeds_strict_cnn/` (absorber_margin); Auto: `outputs/care_auto_absorber_m11_to_m12/`; Margin: `outputs/care_5seeds_strict_cnn/` (margin) | 12 (support≥50) | 5 |
| Tab 9 (Multi-period) | `multiperiod` | `outputs/care_multiperiod/{M_2022_7,M_2022_9,M_2022_11,M_2022_12}/aggregated_mean_std.csv` | Auto-discovered per period | 3 |
| Tab 10 (E2E autonomous) | `e2e` | Labeled probe: `outputs/care_5seeds_strict_cnn/`; Autonomous: `outputs/autonomous_pipeline_final/care/aggregated_mean_std.csv` | 12 (fixed eval set) | 5/3 |
| Tab 11 (Proto replay) | `proto_replay` | `outputs/proto_replay_comparison/{real_replay,proto_replay}/aggregated_mean_std.csv` | 12 (support≥50) | 3 |
| Tab 12 (Architecture) | `arch` | CNN: `outputs/care_5seeds_strict_cnn/`; Transformer: `outputs/care_transformer_multiperiod/M_2022_12/aggregated_mean_std.csv` | 12 (support≥50) | 5/3 |
| TTA lr sweep (inline) | — | `outputs/tta_lr_sweep/{lr_1e-3,lr_1e-4,lr_1e-5}/` | 12 (support≥50) | 1 |
| Training seed audit (inline) | — | `outputs/multiseed_audit/{trainseed0,trainseed1,trainseed2}/care/` | 12 (support≥50) | 3×3 |

## Figures

| Figure | File | Source |
|---|---|---|
| Fig 1 | `fig_collapse_timeline.pdf` | `outputs/per_class_collapse_tls22_monthly/collapse_timeline.csv` |
| Fig 2 | `fig_ablation_bar.pdf` | `outputs/ablation_strict/*/aggregated_mean_std.csv` |
| Fig 3 | `fig_tta_failure.pdf` | `outputs/baselines_group_metrics_*/` + `outputs/care_multiperiod/*/` |
| Fig 4 | `fig_strategy_comparison.pdf` | `outputs/al_baselines_strict/` + `outputs/badge_5seeds_strict/` |
| Fig 5 | `fig_budget_sweep.pdf` | `outputs/care_5seeds_strict_cnn/aggregated_mean_std.csv` |

## Collapse Set Definitions

- **12 classes** (main evaluation): recall < 0.1 AND support ≥ 50 at M-2022-12
- **14 classes** (detection ground truth): recall < 0.1 AND support > 0 at M-2022-12
- **Auto-discovered**: per-period probe-based discovery (class counts vary by period)

## Aggregation

All `aggregated_mean_std.csv` use sample std (ddof=1) via `scripts/aggregate_seeds.py`.

## Superseded directories (not used in paper)

- `autonomous_fixed12_eval/` — superseded by `autonomous_pipeline_final/`
- `unsupervised_care_pipeline/` — uses detected (not fixed-12) eval set; superseded
- `unsupervised_collapse_detection_v4/` — superseded by `detection_ablation/`
