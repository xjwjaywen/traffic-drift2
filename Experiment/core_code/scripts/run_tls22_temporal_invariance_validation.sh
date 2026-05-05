#!/usr/bin/env bash
set -euo pipefail

# Minimal validation for temporal prototype invariance.
# Runs two training-time methods on the same historical periods:
#   1. period-balanced pooled ERM
#   2. pooled ERM + class-level temporal prototype invariance
#
# Usage from Experiment/core_code/:
#   bash scripts/run_tls22_temporal_invariance_validation.sh
#
# Quick smoke:
#   EPOCHS=2 MAX_STEPS_PER_EPOCH=50 bash scripts/run_tls22_temporal_invariance_validation.sh

CONFIG="${CONFIG:-configs/train_tls22_cnn.yaml}"
TRAIN_PERIODS="${TRAIN_PERIODS:-M-2022-1 M-2022-2 M-2022-3 M-2022-4 M-2022-5 M-2022-6}"
TEST_PERIODS="${TEST_PERIODS:-M-2022-7 M-2022-8 M-2022-9 M-2022-10 M-2022-11 M-2022-12}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/titc_validation_tls22}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
PER_PERIOD_BATCH_SIZE="${PER_PERIOD_BATCH_SIZE:-256}"
MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-0}"
LAMBDA_TEMPORAL="${LAMBDA_TEMPORAL:-0.1}"
MIN_PROTO_SAMPLES="${MIN_PROTO_SAMPLES:-2}"

COMMON_ARGS=(
  --config "${CONFIG}"
  --train-periods ${TRAIN_PERIODS}
  --test-periods ${TEST_PERIODS}
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --per-period-batch-size "${PER_PERIOD_BATCH_SIZE}"
  --max-steps-per-epoch "${MAX_STEPS_PER_EPOCH}"
  --min-proto-samples "${MIN_PROTO_SAMPLES}"
)

if [[ -n "${INIT_CHECKPOINT}" ]]; then
  COMMON_ARGS+=(--init-checkpoint "${INIT_CHECKPOINT}")
fi

echo "=== Training pooled ERM baseline ==="
python scripts/train_temporal_invariance_tls22.py \
  --method pooled_erm \
  --output-dir "${OUTPUT_ROOT}/pooled_erm" \
  "${COMMON_ARGS[@]}"

echo "=== Training temporal prototype invariance ==="
python scripts/train_temporal_invariance_tls22.py \
  --method temporal_proto \
  --lambda-temporal "${LAMBDA_TEMPORAL}" \
  --output-dir "${OUTPUT_ROOT}/temporal_proto" \
  "${COMMON_ARGS[@]}"

echo "=== Validation summaries ==="
python - "${OUTPUT_ROOT}" <<'PY'
import json
import os
import sys

root = sys.argv[1]
for name in ("pooled_erm", "temporal_proto"):
    path = os.path.join(root, name, "summary.json")
    with open(path) as f:
        s = json.load(f)
    print(
        f"{name}: mean_macro={s['test_mean_macro_f1']:.4f} "
        f"mean_collapse={s['test_mean_collapse_macro_f1']:.4f} "
        f"final_macro={s['final_macro_f1']:.4f} "
        f"final_collapse={s['final_collapse_macro_f1']:.4f} "
        f"final_stable={s['final_stable_macro_f1']:.4f} "
        f"collapsed={s['final_collapsed_count']} "
        f"severe={s['final_severe_collapsed_count']}"
    )
PY

echo "Saved outputs to: ${OUTPUT_ROOT}"
