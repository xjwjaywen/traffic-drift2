#!/usr/bin/env bash
set -euo pipefail

# Analyze whether final collapse pairs are already close in historical periods.
# Run after scripts/run_tls22_collapse_diagnosis.sh has produced collapse_classes.csv.
#
# Usage from Experiment/core_code/:
#   bash scripts/run_tls22_collapse_pair_distances.sh

CONFIG="${CONFIG:-configs/eval_tls22.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/tls22_cnn/best_model.pt}"
COLLAPSE_CSV="${COLLAPSE_CSV:-outputs/per_class_collapse_tls22_monthly/collapse_classes.csv}"
PROTOTYPE_PERIODS="${PROTOTYPE_PERIODS:-M-2022-1 M-2022-2 M-2022-3 M-2022-4 M-2022-12}"
SPLIT="${SPLIT:-train}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/collapse_pair_distances_tls22}"
PAIR_DISTANCE_EXTRA_ARGS="${PAIR_DISTANCE_EXTRA_ARGS:-}"

python scripts/analyze_collapse_pair_distances.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --collapse-csv "${COLLAPSE_CSV}" \
  --prototype-periods ${PROTOTYPE_PERIODS} \
  --split "${SPLIT}" \
  --pair-source final \
  --final-collapsed-only true \
  --output-dir "${OUTPUT_DIR}" \
  ${PAIR_DISTANCE_EXTRA_ARGS}
