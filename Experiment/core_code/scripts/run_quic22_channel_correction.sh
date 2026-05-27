#!/usr/bin/env bash
set -euo pipefail

# QUIC22 channel-level drift check and input correction.
# This is the targeted follow-up to the channel diagnostic line:
#   1) quantify size / direction / IPT drift by period
#   2) evaluate frozen-model channel-specific quantile correction
#   3) summarize which channel/region helps most

CONFIG="${CONFIG:-configs/eval_quic22.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/quic22_cnn/best_model.pt}"
PERIODS="${PERIODS:-W-2022-46 W-2022-47}"
SETTINGS="${SETTINGS:-raw size_all direction_front_0_9 ipt_tail_20_29 size_direction_front_ipt_tail all}"
DRIFT_OUTPUT_DIR="${DRIFT_OUTPUT_DIR:-outputs/drift_quantification/eval_quic22}"
CORRECTION_OUTPUT_DIR="${CORRECTION_OUTPUT_DIR:-outputs/quantile_correction_quic22}"
MAX_DRIFT_BATCHES="${MAX_DRIFT_BATCHES:-0}"
MAX_SOURCE_BATCHES="${MAX_SOURCE_BATCHES:-1600}"
MAX_TEST_BATCHES="${MAX_TEST_BATCHES:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"

echo "=== QUIC22 channel drift quantification ==="
python scripts/quantify_drift.py \
  --config "${CONFIG}" \
  --output-dir "${DRIFT_OUTPUT_DIR}" \
  --max-batches "${MAX_DRIFT_BATCHES}"

echo "=== QUIC22 channel quantile correction ==="
CMD=(
  python scripts/quantile_correct_eval.py
  --config "${CONFIG}"
  --checkpoint "${CHECKPOINT}"
  --output-dir "${CORRECTION_OUTPUT_DIR}"
  --max-source-batches "${MAX_SOURCE_BATCHES}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --periods
)
for period in ${PERIODS}; do
  CMD+=("${period}")
done
CMD+=(--settings)
for setting in ${SETTINGS}; do
  CMD+=("${setting}")
done
if [[ -n "${MAX_TEST_BATCHES}" ]]; then
  CMD+=(--max-test-batches "${MAX_TEST_BATCHES}")
fi
"${CMD[@]}"

echo "=== QUIC22 channel correction summary ==="
python scripts/summarize_quic22_channel_correction.py \
  --correction-summary "${CORRECTION_OUTPUT_DIR}/summary.csv" \
  --drift-summary "${DRIFT_OUTPUT_DIR}/summary.csv" \
  --output-dir "${CORRECTION_OUTPUT_DIR}"

