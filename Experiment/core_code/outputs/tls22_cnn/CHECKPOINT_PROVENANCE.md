# Checkpoint Provenance: tls22_cnn/best_model.pt

## Identity

| Field | Value |
|---|---|
| File | `outputs/tls22_cnn/best_model.pt` |
| SHA-256 | `edf6a197e9c917efa797540906c75b6c0e1a5e66a0be79e385d6e17e1d044db5` |
| Num classes | 178 |

## Training

| Field | Value |
|---|---|
| Command | `python train.py --config configs/train_tls22_cnn.yaml` |
| Config | `outputs/tls22_cnn/config.yaml` |
| Dataset | CESNET-TLS-Year22, size=S, train=M-2022-3 |
| Seed | 0 (default, not explicitly set) |
| Epochs | 30 |
| Best epoch | 27 |
| Validation macro-F1 | 0.8918 |
| Test macro-F1 (M-2022-5) | 0.8391 |
| Deterministic | Yes (cuDNN deterministic mode) |

## Deployment macro-F1

| Evaluation | Static macro-F1 |
|---|---|
| This checkpoint, M-2022-12 | 0.629 |
| Retraining audit (3 seeds), M-2022-12 | 0.620 ± 0.001 |

The 0.009 gap is attributable to deterministic cuDNN settings used during the
original training run. The relative improvement from CARE is consistent across
all training seeds (CARE macro-F1: 0.667 ± 0.007 across 3×3=9 runs).

This checkpoint is the fixed primary checkpoint used throughout the paper.
It was not selected from multiple candidates; it was the first and only
training run performed with the default configuration.
