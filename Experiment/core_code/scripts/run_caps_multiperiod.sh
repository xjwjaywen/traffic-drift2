#!/usr/bin/env bash
set -euo pipefail

# Run CAPS prototype-only MVP for three TLS-Year22 target periods.
# Usage from Experiment/core_code/:
#   bash scripts/run_caps_multiperiod.sh

CONFIG="${CONFIG:-configs/eval_tls22.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/tls22_cnn/best_model.pt}"
REFERENCE_PERIOD="${REFERENCE_PERIOD:-M-2022-4}"
PERIODS="${PERIODS:-M-2022-7 M-2022-10 M-2022-12}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"

for period in ${PERIODS}; do
  out_dir="${OUTPUT_ROOT}/caps_target_prototype_tls22_${period}"
  echo "=== CAPS ${period} -> ${out_dir} ==="
  python scripts/caps_target_prototype_tls22.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --reference-period "${REFERENCE_PERIOD}" \
    --target-period "${period}" \
    --output-dir "${out_dir}"
done

summary_inputs=()
for period in ${PERIODS}; do
  summary_inputs+=("${OUTPUT_ROOT}/caps_target_prototype_tls22_${period}")
done

python scripts/summarize_caps_experiments.py \
  --input-dirs "${summary_inputs[@]}" \
  --output-dir "${OUTPUT_ROOT}/caps_target_prototype_summary"
