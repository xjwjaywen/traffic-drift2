#!/usr/bin/env bash
set -euo pipefail

# Collapse-aware active maintenance validation for TLS-Year22.
#
# Quick run from Experiment/core_code/:
#   bash scripts/run_tls22_collapse_active_maintenance.sh

CONFIG="${CONFIG:-configs/eval_tls22.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/tls22_cnn/best_model.pt}"
TARGET_PERIOD="${TARGET_PERIOD:-M-2022-12}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/collapse_active_maintenance_tls22_${TARGET_PERIOD}}"
BUDGETS="${BUDGETS:-50,100,200,500,1000}"
STRATEGIES="${STRATEGIES:-random,absorber_random,absorber_margin,oracle_collapse_random}"
SEED="${SEED:-0}"
FT_LR="${FT_LR:-0.001}"
FT_EPOCHS="${FT_EPOCHS:-30}"
FT_BATCH_SIZE="${FT_BATCH_SIZE:-64}"

python scripts/collapse_active_maintenance_tls22.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --target-period "${TARGET_PERIOD}" \
  --output-dir "${OUTPUT_DIR}" \
  --budgets "${BUDGETS}" \
  --strategies "${STRATEGIES}" \
  --seed "${SEED}" \
  --ft-lr "${FT_LR}" \
  --ft-epochs "${FT_EPOCHS}" \
  --ft-batch-size "${FT_BATCH_SIZE}"
