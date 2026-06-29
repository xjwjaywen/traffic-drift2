# Canonical Sources for Paper Tables and Figures

Maps each table/figure in `Publication/paper/main_v4.tex` to its data source.
All paths relative to `Experiment/core_code/`.

## Tables

| Paper Table | Label | Source | Collapse Set | Seeds |
|---|---|---|---|---|
| Absorber pairs | `tab:absorber_pairs` | `outputs/per_class_collapse_tls22_monthly/collapse_classes.csv` | 12 (support≥50) | — |
| TTA baselines (GN) | `tab:tta` | TTA rows: `outputs/tta_multiperiod/{M7,M9,M12}/baselines_group_metrics.csv`; CARE row: `outputs/care_multiperiod_allreplay/{M_2022_7,M_2022_9,M_2022_12}/aggregated_mean_std.csv` | Auto-discovered per period | 1 (TTA) / 3 (CARE) |
| Main results | `tab:main_results` | Unified AL: `outputs/unified_al_baselines/{entropy,coreset,random,margin}/aggregated_mean_std.csv`; BADGE: `outputs/badge_allreplay_5seeds/aggregated_mean_std.csv` | 12 (support≥50) | 5 |
| Component ablation (7 configs) | `tab:ablation` | Current data from `outputs/unified_ablation/`, `outputs/ablation_v3/`, `outputs/ablation_strict/ft_distill/` (3 batches, same pipeline). Canonical re-run: `bash scripts/run_ablation_7config.sh` → `outputs/ablation_7config_canonical/`. KD is defined over D_rpl (§3.5): Rpl+KD uses all-class replay (890); FT+KD uses targeted stable+absorber replay (150); target_repeat matches per-config originals (1 or 2). | 12 (support≥50) | 5 |
| Replay ablation | `tab:replay_ablation` | `outputs/detection_necessity_ablation/` + `outputs/fair_replay_budget/{all_k5,detected_k30,all_k1}/` | 12 (fixed) | 5 |
| Multi-period | `tab:multiperiod` | `outputs/care_multiperiod_allreplay/{M_2022_7,M_2022_9,M_2022_11,M_2022_12}/aggregated_mean_std.csv` | 12 (fixed, support≥50) | 3 |
| Detection signals | `tab:detect_signals` | `outputs/detection_unified_metrics.json` | 12 (recall<0.1, support≥50) | — |
| Detection trigger | `tab:trigger` | `outputs/detection_unified_metrics.json` | 12 (recall<0.1, support≥50) | — |
| Fisher mechanism | `tab:fisher` | `outputs/fisher_mechanism_analysis.csv` | 10 with stable estimates (of 12) | — |
| BN architecture | `tab:bn` | `outputs/care_bn_strict/aggregated_mean_std.csv` + `outputs/bn_tta_baselines_M12/` | 12 (support≥50) | 3 (CARE) |
| Transformer | `tab:transformer` | CNN: `outputs/care_5seeds_strict_cnn/`; Transformer: `outputs/care_transformer_multiperiod/M_2022_12/aggregated_mean_std.csv` | 12 (support≥50) | 5/3 |
| Proto replay | `tab:proto_replay` | `outputs/proto_replay_comparison/{real_replay,proto_replay}/aggregated_mean_std.csv` | 12 (support≥50) | 3 |
| Per-class recovery | `tab:per_class_recovery` | Margin: `outputs/autonomous_allreplay_5seeds/care/seed_*/per_collapse_class_m12.csv`; BADGE: `outputs/badge_allreplay_5seeds/seed_*/per_collapse_class_m12.csv`; Static: `outputs/per_class_collapse_tls22_monthly/collapse_classes.csv`; Selected: `*/selected_class_counts.csv` | 12 (support≥50) | 5 |

### Inline results (not in numbered tables)

| Description | Source | Seeds |
|---|---|---|
| TTA lr sweep | `outputs/tta_lr_sweep/{lr_1e-3,lr_1e-4,lr_1e-5}/` | 1 |
| TTA hyperparam sweep | `outputs/tta_hyperparam_sweep/sweep_summary.csv` | 1 |
| Training seed audit | `outputs/multiseed_audit/{trainseed0,trainseed1,trainseed2}/care/` | 3×3 |
| FT depth (Discussion) | `outputs/full_ft_baseline_fair_kd/seed_{0..4}/results_by_budget.csv` | 5 |
| QUICEXT-25 generalization (not in current paper; data retained for future use) | `outputs/quicext25_care_v3/aggregated_mean_std.csv` | 5 |
| Significance tests | `outputs/significance_tests/significance_tests.json` | — |
| M3 vs M4 replay | `outputs/m3_vs_m4_replay_comparison.csv` | 5 |
| Label budget per-class coverage | `outputs/label_budget_perclass_coverage.csv` | 5 |

## Figures

| Figure | File | Source |
|---|---|---|
| Fig 1 | `fig_collapse_timeline.pdf` | `outputs/per_class_collapse_tls22_monthly/collapse_timeline.csv` |
| Fig 2 | `fig_tsne_multi_pair_m12.pdf` | `outputs/visualizations/` (t-SNE of victim-absorber pairs at M12) |
| Fig 3 | `fig_ablation_bar.pdf` | `outputs/unified_ablation/*/aggregated_mean_std.csv` |
| Fig 4 | `fig_strategy_comparison.pdf` | `outputs/unified_al_baselines/` + `outputs/badge_allreplay_5seeds/` |
| Fig 5 | `budget_curve.pdf` | `outputs/paper_experiments/budget_curve/aggregated_mean_std.csv` |

## Collapse Set Definitions

- **12 classes** (main evaluation + detection ground truth): recall < 0.1 AND support ≥ 50 at M-2022-12
- **3 classes** (QUICEXT-25): classes 17, 23, 25 (recall < 0.1 at M-2025-5)
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
- `care_quic22_strict/` — superseded by `quicext25_care_v3/` (newer dataset, all-class proto replay)
- `quicext25_care_v3_seed{0..4}/` — individual seed runs (M-2024-6 ref, M-2025-5 target, auto-discovered 4 classes [17,21,23,25]); aggregated into `quicext25_care_v3/aggregated_mean_std.csv` which uses the paper's 3-class definition {17,23,25}. Class 21 was excluded because it has recall=0.000 at training time (never correctly classified, not drift-induced collapse). Seed summaries report the auto-discovered 4-class set; canonical aggregated (3-class) is authoritative.
- `care_quicext25_m2025_5/`, `care_quicext25_seed{1..4}/`, `eval_quicext25/` — earlier QUICEXT iterations, superseded by `quicext25_care_v3/`
- `detection_ablation/ablation_summary.json` — uses 14-class GT (support>0); paper now uses `detection_unified_metrics.json` with 12-class GT (support≥50)
- `weight_sensitivity/weight_sensitivity_summary.json` — uses 14-class GT (n_actual_collapsed=14, F1=0.706); paper now uses `detection_unified_metrics.json` weight_sensitivity section with 12-class GT (F1=0.625)
- `full_ft_baseline/` — superseded by `full_ft_baseline_fair_kd/` (KD only on replay, fair comparison with head-only)
