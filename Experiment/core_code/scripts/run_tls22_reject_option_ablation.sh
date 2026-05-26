#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/eval_tls22.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/tls22_cnn/best_model.pt}"
REFERENCE_PERIOD="${REFERENCE_PERIOD:-M-2022-4}"
TARGET_PERIOD="${TARGET_PERIOD:-M-2022-12}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/reject_option_ablation_tls22_m12}"

python scripts/reject_option_ablation_tls22.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --reference-period "$REFERENCE_PERIOD" \
  --target-period "$TARGET_PERIOD" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
