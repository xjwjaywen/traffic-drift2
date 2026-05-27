#!/usr/bin/env bash
set -euo pipefail

# Collapse-aware active maintenance validation for TLS-Year22.
#
# Quick run from Experiment/core_code/:
#   bash scripts/run_tls22_collapse_active_maintenance.sh

CONFIG="${CONFIG:-configs/eval_tls22.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/tls22_cnn/best_model.pt}"
REFERENCE_PERIOD="${REFERENCE_PERIOD:-M-2022-4}"
TARGET_PERIOD="${TARGET_PERIOD:-M-2022-12}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/collapse_active_maintenance_tls22_${TARGET_PERIOD}}"
BUDGETS="${BUDGETS:-50,100,200,500,1000}"
STRATEGIES="${STRATEGIES:-random,entropy,margin,absorber_random,absorber_margin,absorber_distance,absorber_proto_disagree,hybrid_risk,oracle_collapse_random}"
SEED="${SEED:-0}"
FT_LR="${FT_LR:-0.001}"
FT_EPOCHS="${FT_EPOCHS:-30}"
FT_BATCH_SIZE="${FT_BATCH_SIZE:-64}"
REPLAY_MODE="${REPLAY_MODE:-none}"
REPLAY_PER_CLASS="${REPLAY_PER_CLASS:-0}"
TARGET_REPEAT="${TARGET_REPEAT:-1}"
REPLAY_DISTILL_WEIGHT="${REPLAY_DISTILL_WEIGHT:-0}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
COLLAPSE_CLASSES="${COLLAPSE_CLASSES:-}"
STABLE_CLASSES="${STABLE_CLASSES:-}"
ABSORBER_CLASSES="${ABSORBER_CLASSES:-}"

CLASS_ARGS=()
if [[ -n "${COLLAPSE_CLASSES}" ]]; then
  CLASS_ARGS+=(--collapse-classes "${COLLAPSE_CLASSES}")
fi
if [[ -n "${STABLE_CLASSES}" ]]; then
  CLASS_ARGS+=(--stable-classes "${STABLE_CLASSES}")
fi
if [[ -n "${ABSORBER_CLASSES}" ]]; then
  CLASS_ARGS+=(--absorber-classes "${ABSORBER_CLASSES}")
fi

python scripts/collapse_active_maintenance_tls22.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --reference-period "${REFERENCE_PERIOD}" \
  --target-period "${TARGET_PERIOD}" \
  --output-dir "${OUTPUT_DIR}" \
  --budgets "${BUDGETS}" \
  --strategies "${STRATEGIES}" \
  --seed "${SEED}" \
  --ft-lr "${FT_LR}" \
  --ft-epochs "${FT_EPOCHS}" \
  --ft-batch-size "${FT_BATCH_SIZE}" \
  --replay-mode "${REPLAY_MODE}" \
  --replay-per-class "${REPLAY_PER_CLASS}" \
  --target-repeat "${TARGET_REPEAT}" \
  --replay-distill-weight "${REPLAY_DISTILL_WEIGHT}" \
  --distill-temperature "${DISTILL_TEMPERATURE}" \
  "${CLASS_ARGS[@]}"

python scripts/summarize_active_maintenance_results.py \
  --input-dir "${OUTPUT_DIR}"
