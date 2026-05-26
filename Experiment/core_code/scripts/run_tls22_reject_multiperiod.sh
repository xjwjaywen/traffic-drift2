#!/usr/bin/env bash
set -euo pipefail

# Run reject-option ablations for multiple TLS-Year22 target periods.
# Usage from Experiment/core_code/:
#   CUDA_VISIBLE_DEVICES=3 bash scripts/run_tls22_reject_multiperiod.sh

CONFIG="${CONFIG:-configs/eval_tls22.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/tls22_cnn/best_model.pt}"
REFERENCE_PERIOD="${REFERENCE_PERIOD:-M-2022-4}"
PERIODS="${PERIODS:-M-2022-7 M-2022-10 M-2022-12}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
REJECT_EXTRA_ARGS="${REJECT_EXTRA_ARGS:-}"

summary_inputs=()
for period in ${PERIODS}; do
  out_dir="${OUTPUT_ROOT}/reject_option_ablation_tls22_${period}"
  echo "=== Reject option ${period} -> ${out_dir} ==="
  python scripts/reject_option_ablation_tls22.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --reference-period "${REFERENCE_PERIOD}" \
    --target-period "${period}" \
    --output-dir "${out_dir}" \
    ${REJECT_EXTRA_ARGS}
  summary_inputs+=("${out_dir}")
done

python scripts/summarize_reject_option_experiments.py \
  --input-dirs "${summary_inputs[@]}" \
  --output-dir "${OUTPUT_ROOT}/reject_option_ablation_tls22_summary"
