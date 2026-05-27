#!/usr/bin/env bash
set -euo pipefail

# Multi-period validation for collapse-aware active maintenance with replay.
#
# Default setting follows the strongest current signal:
# all-class reference replay, low replay count per class, and target repeat.

CONFIG="${CONFIG:-configs/eval_tls22.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/tls22_cnn/best_model.pt}"
REFERENCE_PERIOD="${REFERENCE_PERIOD:-M-2022-4}"
PERIODS="${PERIODS:-M-2022-7 M-2022-10 M-2022-12}"
REPLAY_MODE="${REPLAY_MODE:-all}"
REPLAY_PER_CLASS="${REPLAY_PER_CLASS:-5}"
TARGET_REPEAT="${TARGET_REPEAT:-2}"
REPLAY_DISTILL_WEIGHT="${REPLAY_DISTILL_WEIGHT:-0}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
BUDGETS="${BUDGETS:-200,500,1000}"
STRATEGIES="${STRATEGIES:-random,margin,absorber_random,absorber_margin,absorber_margin_balanced}"
SEED="${SEED:-0}"
FT_LR="${FT_LR:-0.001}"
FT_EPOCHS="${FT_EPOCHS:-30}"
FT_BATCH_SIZE="${FT_BATCH_SIZE:-64}"
if [[ "${REPLAY_DISTILL_WEIGHT}" == "0" || "${REPLAY_DISTILL_WEIGHT}" == "0.0" ]]; then
  DEFAULT_SUMMARY_DIR="outputs/collapse_active_replay_tls22_summary_${REPLAY_MODE}_r${REPLAY_PER_CLASS}_tr${TARGET_REPEAT}"
else
  DEFAULT_SUMMARY_DIR="outputs/collapse_active_replay_tls22_summary_${REPLAY_MODE}_r${REPLAY_PER_CLASS}_tr${TARGET_REPEAT}_distill${REPLAY_DISTILL_WEIGHT}"
fi
SUMMARY_DIR="${SUMMARY_DIR:-${DEFAULT_SUMMARY_DIR}}"

SUMMARY_INPUTS=()

for PERIOD in ${PERIODS}; do
  if [[ "${REPLAY_DISTILL_WEIGHT}" == "0" || "${REPLAY_DISTILL_WEIGHT}" == "0.0" ]]; then
    PERIOD_OUTPUT_DIR="outputs/collapse_active_replay_tls22_${PERIOD}_${REPLAY_MODE}_r${REPLAY_PER_CLASS}_tr${TARGET_REPEAT}"
  else
    PERIOD_OUTPUT_DIR="outputs/collapse_active_replay_tls22_${PERIOD}_${REPLAY_MODE}_r${REPLAY_PER_CLASS}_tr${TARGET_REPEAT}_distill${REPLAY_DISTILL_WEIGHT}"
  fi
  echo "=== Active replay ${PERIOD} -> ${PERIOD_OUTPUT_DIR} ==="
  CONFIG="${CONFIG}" \
  CHECKPOINT="${CHECKPOINT}" \
  REFERENCE_PERIOD="${REFERENCE_PERIOD}" \
  TARGET_PERIOD="${PERIOD}" \
  OUTPUT_DIR="${PERIOD_OUTPUT_DIR}" \
  BUDGETS="${BUDGETS}" \
  STRATEGIES="${STRATEGIES}" \
  SEED="${SEED}" \
  FT_LR="${FT_LR}" \
  FT_EPOCHS="${FT_EPOCHS}" \
  FT_BATCH_SIZE="${FT_BATCH_SIZE}" \
  REPLAY_MODE="${REPLAY_MODE}" \
  REPLAY_PER_CLASS="${REPLAY_PER_CLASS}" \
  TARGET_REPEAT="${TARGET_REPEAT}" \
  REPLAY_DISTILL_WEIGHT="${REPLAY_DISTILL_WEIGHT}" \
  DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE}" \
  bash scripts/run_tls22_collapse_active_replay.sh
  SUMMARY_INPUTS+=(--input-dir "${PERIOD}:${PERIOD_OUTPUT_DIR}")
done

python scripts/summarize_active_replay_multiperiod.py \
  "${SUMMARY_INPUTS[@]}" \
  --output-dir "${SUMMARY_DIR}"
