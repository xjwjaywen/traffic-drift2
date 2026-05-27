#!/usr/bin/env bash
set -euo pipefail

# Collapse-aware active maintenance with reference replay.
#
# This validates whether target-label maintenance can recover collapsed classes
# without damaging stable classes as much as target-only head fine-tuning.

CONFIG="${CONFIG:-configs/eval_tls22.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/tls22_cnn/best_model.pt}"
REFERENCE_PERIOD="${REFERENCE_PERIOD:-M-2022-4}"
TARGET_PERIOD="${TARGET_PERIOD:-M-2022-12}"
REPLAY_MODE="${REPLAY_MODE:-stable}"
REPLAY_PER_CLASS="${REPLAY_PER_CLASS:-25}"
TARGET_REPEAT="${TARGET_REPEAT:-2}"
REPLAY_DISTILL_WEIGHT="${REPLAY_DISTILL_WEIGHT:-0}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
if [[ "${REPLAY_DISTILL_WEIGHT}" == "0" || "${REPLAY_DISTILL_WEIGHT}" == "0.0" ]]; then
  DEFAULT_OUTPUT_DIR="outputs/collapse_active_replay_tls22_${TARGET_PERIOD}_${REPLAY_MODE}_r${REPLAY_PER_CLASS}_tr${TARGET_REPEAT}"
else
  DEFAULT_OUTPUT_DIR="outputs/collapse_active_replay_tls22_${TARGET_PERIOD}_${REPLAY_MODE}_r${REPLAY_PER_CLASS}_tr${TARGET_REPEAT}_distill${REPLAY_DISTILL_WEIGHT}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
BUDGETS="${BUDGETS:-50,100,200,500,1000}"
STRATEGIES="${STRATEGIES:-random,margin,absorber_random,absorber_margin,oracle_collapse_random}"
SEED="${SEED:-0}"
FT_LR="${FT_LR:-0.001}"
FT_EPOCHS="${FT_EPOCHS:-30}"
FT_BATCH_SIZE="${FT_BATCH_SIZE:-64}"

export CONFIG CHECKPOINT REFERENCE_PERIOD TARGET_PERIOD OUTPUT_DIR
export BUDGETS STRATEGIES SEED FT_LR FT_EPOCHS FT_BATCH_SIZE
export REPLAY_MODE REPLAY_PER_CLASS TARGET_REPEAT
export REPLAY_DISTILL_WEIGHT DISTILL_TEMPERATURE

bash scripts/run_tls22_collapse_active_maintenance.sh
