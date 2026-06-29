# Reproducing Paper Results

All paper results in `Publication/paper/main_v4.tex` trace back to canonical CSV/JSON files
in `outputs/`. The authoritative mapping is `outputs/CANONICAL_SOURCES.md`.

## Pre-computed results (no GPU needed)

Most tables and figures can be verified by reading the canonical CSVs directly:

| Paper Element | Canonical Source |
|---|---|
| Table I (absorber pairs) | `outputs/per_class_collapse_tls22_monthly/collapse_classes.csv` |
| Table II (Fisher) | `outputs/fisher_mechanism_analysis.csv` |
| Table III (TTA baselines) | `outputs/tta_multiperiod/{M7,M9,M12}/baselines_group_metrics.csv` |
| Table IV (ablation, 7 configs) | `outputs/unified_ablation/` + `outputs/ablation_v3/` + `outputs/ablation_strict/ft_distill/` |
| Table V (main results) | `outputs/unified_al_baselines/` + `outputs/badge_allreplay_5seeds/` |
| Table VI (detection) | `outputs/detection_unified_metrics.json` |
| Table VII (per-class recovery) | `outputs/per_class_recovery_table.csv` |
| TTA LR sweep (inline) | `outputs/tta_hyperparam_sweep/sweep_summary.csv` |
| M3 vs M4 replay (inline) | `outputs/m3_vs_m4_replay_comparison.csv` |
| Label budget coverage (inline) | `outputs/label_budget_perclass_coverage.csv` |
| QUICEXT-25 (§6.7) | `outputs/quicext25_care_v3/aggregated_mean_std.csv` |

## Re-running experiments (GPU + dataset required)

Experiments require the CESNET-TLS-Year22 dataset and a CUDA GPU.

```bash
# 1. Train base model
python train.py --config configs/train_tls22_cnn.yaml

# 2. Main CARE experiments (margin + BADGE, 5 seeds)
bash scripts/run_v3_unified_experiments.sh

# 3. TTA baselines (LR sweep)
bash scripts/run_tta_hyperparam_sweep.sh

# 4. QUICEXT-25 generalization
# Requires CESNET-QUICEXT-25 dataset; set data_dir in configs/eval_quicext25.yaml
# See outputs/quicext25_care_v3/summary.json for config
```

## Collapse set definition

- **12 classes** (main evaluation): recall < 0.1 AND support >= 50 at M-2022-12
- Classes: 10, 26, 38, 47, 48, 56, 66, 69, 104, 109, 163, 174

## Aggregation

All `aggregated_mean_std.csv` use sample std (ddof=1) via `scripts/aggregate_seeds.py`.

## Superseded directories

See `outputs/CANONICAL_SOURCES.md` § "Superseded directories" for a full list of
old experiment directories that are NOT used in the current paper.
