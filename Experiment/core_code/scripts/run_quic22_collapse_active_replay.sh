#!/usr/bin/env bash
set -euo pipefail

# QUIC22 quick validation for collapse-aware active replay.
# Run scripts/run_quic22_collapse_diagnosis.sh first, or set RUN_DIAGNOSIS=1.

CONFIG="${CONFIG:-configs/eval_quic22.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/quic22_cnn/best_model.pt}"
REFERENCE_PERIOD="${REFERENCE_PERIOD:-W-2022-45}"
TARGET_PERIOD="${TARGET_PERIOD:-W-2022-47}"
DRIFT_OUTPUT_DIR="${DRIFT_OUTPUT_DIR:-outputs/class_conditional_drift_quic22}"
COLLAPSE_OUTPUT_DIR="${COLLAPSE_OUTPUT_DIR:-outputs/per_class_collapse_quic22}"
COLLAPSE_CSV="${COLLAPSE_CSV:-${COLLAPSE_OUTPUT_DIR}/collapse_classes.csv}"
SELECTED_STABLE_CSV="${SELECTED_STABLE_CSV:-${DRIFT_OUTPUT_DIR}/selected_stable_classes.csv}"
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
  DEFAULT_OUTPUT_DIR="outputs/collapse_active_replay_quic22_${TARGET_PERIOD}_${REPLAY_MODE}_r${REPLAY_PER_CLASS}_tr${TARGET_REPEAT}"
else
  DEFAULT_OUTPUT_DIR="outputs/collapse_active_replay_quic22_${TARGET_PERIOD}_${REPLAY_MODE}_r${REPLAY_PER_CLASS}_tr${TARGET_REPEAT}_distill${REPLAY_DISTILL_WEIGHT}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

if [[ "${RUN_DIAGNOSIS}" == "1" || ! -f "${COLLAPSE_CSV}" ]]; then
  CONFIG="${CONFIG}" \
  CHECKPOINT="${CHECKPOINT}" \
  REFERENCE_PERIOD="${REFERENCE_PERIOD}" \
  FINAL_PERIOD="${TARGET_PERIOD}" \
  DRIFT_OUTPUT_DIR="${DRIFT_OUTPUT_DIR}" \
  COLLAPSE_OUTPUT_DIR="${COLLAPSE_OUTPUT_DIR}" \
  MIN_SUPPORT="${MIN_SUPPORT}" \
  bash scripts/run_quic22_collapse_diagnosis.sh
fi

eval "$(
  python scripts/extract_collapse_active_classes.py \
    --collapse-csv "${COLLAPSE_CSV}" \
    --selected-stable-csv "${SELECTED_STABLE_CSV}" \
    --min-support "${MIN_SUPPORT}"
)"

echo "QUIC22 collapse classes: ${COLLAPSE_CLASSES}"
echo "QUIC22 absorber classes: ${ABSORBER_CLASSES}"
echo "QUIC22 stable classes: ${STABLE_CLASSES}"

if [[ -z "${COLLAPSE_CLASSES}" || -z "${ABSORBER_CLASSES}" || -z "${STABLE_CLASSES}" ]]; then
  echo "No usable QUIC22 collapse/absorber/stable class groups were extracted."
  echo "Inspect ${COLLAPSE_CSV} before running active maintenance."
  exit 2
fi

CONFIG="${CONFIG}" \
CHECKPOINT="${CHECKPOINT}" \
REFERENCE_PERIOD="${REFERENCE_PERIOD}" \
TARGET_PERIOD="${TARGET_PERIOD}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
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
COLLAPSE_CLASSES="${COLLAPSE_CLASSES}" \
STABLE_CLASSES="${STABLE_CLASSES}" \
ABSORBER_CLASSES="${ABSORBER_CLASSES}" \
bash scripts/run_tls22_collapse_active_maintenance.sh
