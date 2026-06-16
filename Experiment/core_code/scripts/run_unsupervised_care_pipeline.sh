#!/bin/bash
# Full autonomous pipeline: unsupervised collapse detection → CARE repair
# No labeled probe period needed. Only 1000 target labels for repair.
#
# Usage: bash scripts/run_unsupervised_care_pipeline.sh <config> <checkpoint> <ref_period> <tgt_period> <output_dir>

set -euo pipefail

CONFIG="${1:?Usage: $0 <config> <checkpoint> <ref_period> <tgt_period> <output_dir>}"
CHECKPOINT="${2:?}"
REF_PERIOD="${3:?}"
TGT_PERIOD="${4:?}"
OUTPUT_DIR="${5:?}"

echo "=== Step 1: Unsupervised Collapse Detection ==="
python scripts/unsupervised_collapse_detection.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --reference-period "${REF_PERIOD}" \
    --target-period "${TGT_PERIOD}" \
    --output-dir "${OUTPUT_DIR}/detection"

DETECTION_FILE="${OUTPUT_DIR}/detection/detection_summary.json"
COLLAPSE_CLASSES=$(python -c "
import json
d = json.load(open('${DETECTION_FILE}'))
classes = d.get('detected_collapse', [])
print(','.join(str(c) for c in classes) if classes else '')
")
ABSORBER_CLASSES=$(python -c "
import json
d = json.load(open('${DETECTION_FILE}'))
classes = d.get('detected_absorbers', [])
print(','.join(str(c) for c in classes) if classes else '')
")

# Fixed 12-class eval set (for fair comparison across pipelines)
EVAL_COLLAPSE="56,163,174,48,38,69,104,47,66,10,109,26"

echo ""
echo "Detected collapse classes: ${COLLAPSE_CLASSES:-none}"
echo "Detected absorber classes: ${ABSORBER_CLASSES:-none}"
echo "Eval collapse classes (fixed): ${EVAL_COLLAPSE}"

echo ""
echo "=== Step 2: CARE Repair (3 seeds) ==="
echo "Using DETECTED classes for replay, FIXED 12 classes for evaluation"
for SEED in 0 1 2; do
    echo "--- Seed ${SEED} ---"
    EXTRA=""
    if [[ -n "${COLLAPSE_CLASSES}" ]]; then
        EXTRA="${EXTRA} --collapse-classes ${COLLAPSE_CLASSES}"
    fi
    if [[ -n "${ABSORBER_CLASSES}" ]]; then
        EXTRA="${EXTRA} --absorber-classes ${ABSORBER_CLASSES}"
    fi
    EXTRA="${EXTRA} --eval-collapse-classes ${EVAL_COLLAPSE}"

    python scripts/collapse_active_maintenance_tls22.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --reference-period "${REF_PERIOD}" \
        --target-period "${TGT_PERIOD}" \
        --strategies "margin" \
        --budgets "1000" \
        ${EXTRA} \
        --replay-mode stable_absorber \
        --replay-per-class 5 \
        --replay-distill-weight 0.5 \
        --target-repeat 2 \
        --seed "${SEED}" \
        --output-dir "${OUTPUT_DIR}/care/seed_${SEED}"
done

echo ""
echo "=== Step 3: Aggregate ==="
python scripts/aggregate_seeds.py --base-dir "${OUTPUT_DIR}/care" --num-seeds 3

echo ""
echo "=== Autonomous Pipeline Complete ==="
echo "Detection: ${OUTPUT_DIR}/detection/"
echo "Repair:    ${OUTPUT_DIR}/care/"
