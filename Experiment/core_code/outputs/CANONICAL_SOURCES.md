# Canonical Output Directories for Paper Tables

Each paper table is sourced from exactly one directory. Other directories
contain exploratory or superseded results and should not be cited.

| Paper Table | Source Directory | Description |
|---|---|---|
| Table 2 (TTA baselines GN) | `baselines_group_metrics_M12/` | Static + 5 TTA methods at M12 |
| Table 3 (TTA baselines BN) | `bn_tta_baselines_M12/` | BN architecture TTA eval |
| Table 4 (Detection ablation) | `detection_ablation/` | 4-config signal ablation |
| Table 5 (Main results) | `care_5seeds_strict_cnn/` + `al_baselines_strict/` | 5-seed strict eval |
| Table 6 (Per-class recovery) | `care_5seeds_strict_cnn/seed_0/` | Single seed, absorber-margin |
| Table 7 (Ablation) | `ablation_strict/{ft_only,ft_replay,full_care}/` | 5-seed component ablation |
| Table 8 (Auto absorber) | `care_5seeds_strict_cnn/` | Oracle vs auto vs margin |
| Table 9 (Multi-period) | `care_multiperiod/` | 3-seed, M7/M9/M11/M12 |
| Table 10 (E2E autonomous) | **`autonomous_pipeline_final/`** | Detected repair, fixed-12 eval |
| Table 11 (Proto replay) | `proto_replay_comparison/` | Real vs prototype replay |
| Table 12 (Architecture) | `care_5seeds_strict_cnn/` + `tls22_transformer/` | CNN vs Transformer |
| TTA lr sweep (inline) | `tta_lr_sweep/` | 3 learning rates |
| Training seed audit | `multiseed_audit/` | 3 train seeds × 3 FT seeds |

## Superseded directories (not used in paper)

- `autonomous_fixed12_eval/` — superseded by `autonomous_pipeline_final/`
- `unsupervised_care_pipeline/` — uses detected (not fixed-12) eval set; superseded
- `unsupervised_collapse_detection_v4/` — superseded by `detection_ablation/`
- `collapse_active_maintenance_tls22_*` — early exploratory runs
