#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/eval_tls22.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/tls22_cnn/best_model.pt}"
TARGET_PERIOD="${TARGET_PERIOD:-M-2022-12}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/collapse_visual_diagnostics_tls22}"

python scripts/collapse_confusion_tsne_tls22.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --target-period "$TARGET_PERIOD" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
