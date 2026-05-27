#!/usr/bin/env bash
set -euo pipefail

# Train and evaluate a QUIC22 model with targeted direction-channel augmentation.
#
# Smoke test:
#   EPOCHS=2 MAX_STEPS_PER_EPOCH=100 bash scripts/run_quic22_training_channel_aug.sh
#
# Full run:
#   bash scripts/run_quic22_training_channel_aug.sh

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_quic22_cnn_direction_front_dropout.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/eval_quic22.yaml}"
BASELINE_RESULTS="${BASELINE_RESULTS:-outputs/eval_quic22/results_sequential.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/quic22_cnn_direction_front_drop002}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-outputs/eval_quic22_direction_front_drop002}"
EPOCHS="${EPOCHS:-}"
MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-0}"

TRAIN_CMD=(
  python train.py
  --config "${TRAIN_CONFIG}"
  --output-dir "${OUTPUT_DIR}"
  --max-steps-per-epoch "${MAX_STEPS_PER_EPOCH}"
)
if [[ -n "${EPOCHS}" ]]; then
  TRAIN_CMD+=(--epochs "${EPOCHS}")
fi

echo "=== Training QUIC22 channel-augmented model ==="
"${TRAIN_CMD[@]}"

echo "=== Evaluating static sequential performance ==="
python evaluate_tta.py \
  --config "${EVAL_CONFIG}" \
  --checkpoint "${OUTPUT_DIR}/best_model.pt" \
  --output-dir "${EVAL_OUTPUT_DIR}" \
  --mode sequential \
  --methods static \
  --output-suffix static

echo "=== Summarizing against baseline ==="
python scripts/summarize_quic22_training_aug.py \
  --baseline-results "${BASELINE_RESULTS}" \
  --aug-results "${EVAL_OUTPUT_DIR}/results_sequential_static.json" \
  --output-dir "${EVAL_OUTPUT_DIR}"

