#!/usr/bin/env bash
set -euo pipefail

# Multi-period QUIC22 validation after collapse diagnosis.
# Default periods skip W-45 because it is the reference period.

CONFIG="${CONFIG:-configs/eval_quic22.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/quic22_cnn/best_model.pt}"
REFERENCE_PERIOD="${REFERENCE_PERIOD:-W-2022-45}"
PERIODS="${PERIODS:-W-2022-46 W-2022-47}"
COLLAPSE_FINAL_PERIOD="${COLLAPSE_FINAL_PERIOD:-W-2022-47}"
DRIFT_OUTPUT_DIR="${DRIFT_OUTPUT_DIR:-outputs/class_conditional_drift_quic22}"
COLLAPSE_OUTPUT_DIR="${COLLAPSE_OUTPUT_DIR:-outputs/per_class_collapse_quic22}"
COLLAPSE_CSV="${COLLAPSE_CSV:-${COLLAPSE_OUTPUT_DIR}/collapse_classes.csv}"
RUN_DIAGNOSIS="${RUN_DIAGNOSIS:-0}"
MIN_SUPPORT="${MIN_SUPPORT:-50}"
REPLAY_MODE="${REPLAY_MODE:-all}"
REPLAY_PER_CLASS="${REPLAY_PER_CLASS:-5}"
TARGET_REPEAT="${TARGET_REPEAT:-2}"
REPLAY_DISTILL_WEIGHT="${REPLAY_DISTILL_WEIGHT:-0.5}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
BUDGETS="${BUDGETS:-200,500,1000}"
STRATEGIES="${STRATEGIES:-random,margin,absorber_random,absorber_margin,absorber_margin_balanced}"
SEED="${SEED:-0}"
FT_LR="${FT_LR:-0.001}"
FT_EPOCHS="${FT_EPOCHS:-30}"
FT_BATCH_SIZE="${FT_BATCH_SIZE:-64}"

if [[ "${REPLAY_DISTILL_WEIGHT}" == "0" || "${REPLAY_DISTILL_WEIGHT}" == "0.0" ]]; then
  DEFAULT_SUMMARY_DIR="outputs/collapse_active_replay_quic22_summary_${REPLAY_MODE}_r${REPLAY_PER_CLASS}_tr${TARGET_REPEAT}"
else
  DEFAULT_SUMMARY_DIR="outputs/collapse_active_replay_quic22_summary_${REPLAY_MODE}_r${REPLAY_PER_CLASS}_tr${TARGET_REPEAT}_distill${REPLAY_DISTILL_WEIGHT}"
fi
SUMMARY_DIR="${SUMMARY_DIR:-${DEFAULT_SUMMARY_DIR}}"

SUMMARY_INPUTS=()

if [[ "${RUN_DIAGNOSIS}" == "1" || ! -f "${COLLAPSE_CSV}" ]]; then
  CONFIG="${CONFIG}" \
  CHECKPOINT="${CHECKPOINT}" \
  REFERENCE_PERIOD="${REFERENCE_PERIOD}" \
  FINAL_PERIOD="${COLLAPSE_FINAL_PERIOD}" \
  DRIFT_OUTPUT_DIR="${DRIFT_OUTPUT_DIR}" \
  COLLAPSE_OUTPUT_DIR="${COLLAPSE_OUTPUT_DIR}" \
  MIN_SUPPORT="${MIN_SUPPORT}" \
  bash scripts/run_quic22_collapse_diagnosis.sh
fi

for PERIOD in ${PERIODS}; do
  if [[ "${REPLAY_DISTILL_WEIGHT}" == "0" || "${REPLAY_DISTILL_WEIGHT}" == "0.0" ]]; then
    PERIOD_OUTPUT_DIR="outputs/collapse_active_replay_quic22_${PERIOD}_${REPLAY_MODE}_r${REPLAY_PER_CLASS}_tr${TARGET_REPEAT}"
  else
    PERIOD_OUTPUT_DIR="outputs/collapse_active_replay_quic22_${PERIOD}_${REPLAY_MODE}_r${REPLAY_PER_CLASS}_tr${TARGET_REPEAT}_distill${REPLAY_DISTILL_WEIGHT}"
  fi
  echo "=== QUIC22 active replay ${PERIOD} -> ${PERIOD_OUTPUT_DIR} ==="
  CONFIG="${CONFIG}" \
  CHECKPOINT="${CHECKPOINT}" \
  REFERENCE_PERIOD="${REFERENCE_PERIOD}" \
  TARGET_PERIOD="${PERIOD}" \
  DRIFT_OUTPUT_DIR="${DRIFT_OUTPUT_DIR}" \
  COLLAPSE_OUTPUT_DIR="${COLLAPSE_OUTPUT_DIR}" \
  RUN_DIAGNOSIS="0" \
  MIN_SUPPORT="${MIN_SUPPORT}" \
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
  bash scripts/run_quic22_collapse_active_replay.sh
  SUMMARY_INPUTS+=(--input-dir "${PERIOD}:${PERIOD_OUTPUT_DIR}")
done

python scripts/summarize_active_replay_multiperiod.py \
  "${SUMMARY_INPUTS[@]}" \
  --output-dir "${SUMMARY_DIR}"
