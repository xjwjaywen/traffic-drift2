#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/eval_tls22.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/tls22_cnn/best_model.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/tta_drift_type_ablation_tls22}"
METHODS="${METHODS:-static,eata,cotta,sar,tta_tc}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 1
fi

python scripts/tta_drift_type_ablation_tls22.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --methods "$METHODS" \
  --output-dir "$OUTPUT_DIR" \
  $EXTRA_ARGS

