#!/usr/bin/env bash
set -euo pipefail

# Run QUIC22 class-collapse diagnosis across W-45..W-47.
# Usage from Experiment/core_code/:
#   bash scripts/run_quic22_collapse_diagnosis.sh

CONFIG="${CONFIG:-configs/eval_quic22.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/quic22_cnn/best_model.pt}"
PERIODS="${PERIODS:-W-2022-45 W-2022-46 W-2022-47}"
REFERENCE_PERIOD="${REFERENCE_PERIOD:-W-2022-45}"
FINAL_PERIOD="${FINAL_PERIOD:-W-2022-47}"
DRIFT_OUTPUT_DIR="${DRIFT_OUTPUT_DIR:-outputs/class_conditional_drift_quic22}"
COLLAPSE_OUTPUT_DIR="${COLLAPSE_OUTPUT_DIR:-outputs/per_class_collapse_quic22}"
MIN_SUPPORT="${MIN_SUPPORT:-50}"
QUICK_COLLAPSE_ONLY="${QUICK_COLLAPSE_ONLY:-1}"
COLLAPSE_EXTRA_ARGS="${COLLAPSE_EXTRA_ARGS:-}"

QUICK_ARGS=()
if [[ "${QUICK_COLLAPSE_ONLY}" == "1" ]]; then
  QUICK_ARGS+=(--quick-collapse-only)
fi

python scripts/class_conditional_drift.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --periods ${PERIODS} \
  --min-support "${MIN_SUPPORT}" \
  --output-dir "${DRIFT_OUTPUT_DIR}" \
  "${QUICK_ARGS[@]}"

python scripts/summarize_per_class_collapse.py \
  --input-dir "${DRIFT_OUTPUT_DIR}" \
  --output-dir "${COLLAPSE_OUTPUT_DIR}" \
  --reference-period "${REFERENCE_PERIOD}" \
  --final-period "${FINAL_PERIOD}" \
  --min-support "${MIN_SUPPORT}" \
  ${COLLAPSE_EXTRA_ARGS}
