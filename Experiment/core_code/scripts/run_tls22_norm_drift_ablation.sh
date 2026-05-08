#!/usr/bin/env bash
set -euo pipefail

# Train/evaluate TLS22 normalization ablations and summarize by drift type.
#
# Default protocol:
#   - GN checkpoint: existing outputs/tls22_cnn/best_model.pt
#   - Train IN/BN/LN checkpoints if TRAIN=1
#   - Evaluate static predictions on M-2022-7/10/12
#   - Summarize stable vs abrupt/gradual/final-collapsed/degraded/absorber groups
#
# Usage from Experiment/core_code/:
#   bash scripts/run_tls22_norm_drift_ablation.sh
#
# Evaluation only with existing checkpoints:
#   TRAIN=0 bash scripts/run_tls22_norm_drift_ablation.sh

CONFIG_EVAL="${CONFIG_EVAL:-configs/eval_tls22.yaml}"
PERIODS="${PERIODS:-M-2022-7 M-2022-10 M-2022-12}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/norm_drift_type_ablation_tls22}"
TRAIN="${TRAIN:-1}"
TRAIN_NORMS="${TRAIN_NORMS:-in bn ln}"
BASELINE_NORM="${BASELINE_NORM:-gn}"
GN_CHECKPOINT="${GN_CHECKPOINT:-outputs/tls22_cnn/best_model.pt}"
IN_CHECKPOINT="${IN_CHECKPOINT:-outputs/tls22_cnn_in/best_model.pt}"
BN_CHECKPOINT="${BN_CHECKPOINT:-outputs/tls22_cnn_bn/best_model.pt}"
LN_CHECKPOINT="${LN_CHECKPOINT:-outputs/tls22_cnn_ln/best_model.pt}"
COLLAPSE_REPORT="${COLLAPSE_REPORT:-outputs/per_class_collapse_tls22_monthly/collapse_classes.csv}"

if [[ "${TRAIN}" == "1" ]]; then
  for norm in ${TRAIN_NORMS}; do
    echo "=== Training TLS22 CNN norm=${norm} ==="
    python train.py \
      --config "configs/train_tls22_cnn_${norm}.yaml" \
      --output-dir "outputs/tls22_cnn_${norm}"
  done
fi

python scripts/norm_drift_type_ablation_tls22.py \
  --config "${CONFIG_EVAL}" \
  --checkpoints \
    "gn=${GN_CHECKPOINT}" \
    "in=${IN_CHECKPOINT}" \
    "bn=${BN_CHECKPOINT}" \
    "ln=${LN_CHECKPOINT}" \
  --periods ${PERIODS} \
  --baseline-norm "${BASELINE_NORM}" \
  --collapse-report "${COLLAPSE_REPORT}" \
  --output-dir "${OUTPUT_DIR}"
