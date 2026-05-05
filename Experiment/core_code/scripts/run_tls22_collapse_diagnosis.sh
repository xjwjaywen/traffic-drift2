#!/usr/bin/env bash
set -euo pipefail

# Run monthly TLS-Year22 class-collapse diagnosis from M-2022-4 to M-2022-12.
# Usage from Experiment/core_code/:
#   bash scripts/run_tls22_collapse_diagnosis.sh

CONFIG="${CONFIG:-configs/eval_tls22.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/tls22_cnn/best_model.pt}"
PERIODS="${PERIODS:-M-2022-4 M-2022-5 M-2022-6 M-2022-7 M-2022-8 M-2022-9 M-2022-10 M-2022-11 M-2022-12}"
DRIFT_OUTPUT_DIR="${DRIFT_OUTPUT_DIR:-outputs/class_conditional_drift_tls22_monthly}"
COLLAPSE_OUTPUT_DIR="${COLLAPSE_OUTPUT_DIR:-outputs/per_class_collapse_tls22_monthly}"
COLLAPSE_EXTRA_ARGS="${COLLAPSE_EXTRA_ARGS:-}"

python scripts/class_conditional_drift.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --periods ${PERIODS} \
  --output-dir "${DRIFT_OUTPUT_DIR}"

python scripts/summarize_per_class_collapse.py \
  --input-dir "${DRIFT_OUTPUT_DIR}" \
  --output-dir "${COLLAPSE_OUTPUT_DIR}" \
  --reference-period M-2022-4 \
  --final-period M-2022-12 \
  ${COLLAPSE_EXTRA_ARGS}
