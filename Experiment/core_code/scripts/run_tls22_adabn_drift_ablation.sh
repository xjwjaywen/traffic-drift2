#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/eval_tls22.yaml}"
BN_CHECKPOINT="${BN_CHECKPOINT:-outputs/tls22_cnn_bn/best_model.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/adabn_drift_type_ablation_tls22}"
PERIODS="${PERIODS:-M-2022-7 M-2022-10 M-2022-12}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ ! -f "$BN_CHECKPOINT" ]]; then
  echo "Missing BN checkpoint: $BN_CHECKPOINT" >&2
  echo "Train it first, e.g.:" >&2
  echo "  python train.py --config configs/train_tls22_cnn_bn.yaml" >&2
  exit 1
fi

python scripts/adabn_drift_type_ablation_tls22.py \
  --config "$CONFIG" \
  --checkpoint "$BN_CHECKPOINT" \
  --periods $PERIODS \
  --output-dir "$OUTPUT_DIR" \
  $EXTRA_ARGS

