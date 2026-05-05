#!/usr/bin/env bash
set -euo pipefail

# Run CAPS++ adapter MVP for TLS-Year22 target periods.
# Usage from Experiment/core_code/:
#   bash scripts/run_capspp_adapter_multiperiod.sh
#
# Optional focused grid:
#   CAPSPP_EXTRA_ARGS="--alphas 5 --tau-confs 0.8 --momentums 0.9 --adapter-lrs 0.0003,0.001" \
#     bash scripts/run_capspp_adapter_multiperiod.sh

CONFIG="${CONFIG:-configs/eval_tls22.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/tls22_cnn/best_model.pt}"
REFERENCE_PERIOD="${REFERENCE_PERIOD:-M-2022-4}"
PERIODS="${PERIODS:-M-2022-7 M-2022-10 M-2022-12}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
CAPSPP_EXTRA_ARGS="${CAPSPP_EXTRA_ARGS:-}"

for period in ${PERIODS}; do
  out_dir="${OUTPUT_ROOT}/capspp_adapter_tls22_${period}"
  echo "=== CAPS++ adapter ${period} -> ${out_dir} ==="
  python scripts/capspp_adapter_tls22.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --reference-period "${REFERENCE_PERIOD}" \
    --target-period "${period}" \
    --output-dir "${out_dir}" \
    ${CAPSPP_EXTRA_ARGS}
done

summary_inputs=()
for period in ${PERIODS}; do
  summary_inputs+=("${OUTPUT_ROOT}/capspp_adapter_tls22_${period}")
done

python scripts/summarize_caps_experiments.py \
  --input-dirs "${summary_inputs[@]}" \
  --output-dir "${OUTPUT_ROOT}/capspp_adapter_summary"
