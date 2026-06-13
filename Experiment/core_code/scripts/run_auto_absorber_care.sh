#!/bin/bash
# End-to-end deployable CARE pipeline:
#   1. Discover absorbers from a probe period (e.g., M-2022-11)
#   2. Run CARE on target period (e.g., M-2022-12) using discovered lists
#
# Usage:
#   bash scripts/run_auto_absorber_care.sh <config> <checkpoint> <probe_period> <target_period> <output_dir>
#
# Example:
#   bash scripts/run_auto_absorber_care.sh \
#     configs/eval_tls22.yaml outputs/tls22_cnn/best_model.pt \
#     M-2022-11 M-2022-12 outputs/care_auto_absorber

set -euo pipefail

CONFIG="${1:?Usage: $0 <config> <checkpoint> <probe> <target> <output>}"
CHECKPOINT="${2:?}"
PROBE_PERIOD="${3:?}"
TARGET_PERIOD="${4:?}"
OUTPUT_DIR="${5:?}"

DISCOVERY_DIR="${OUTPUT_DIR}/discovery"
PROBE_SLUG=$(echo "${PROBE_PERIOD}" | tr '-' '_')

echo "=== Step 1: Discover absorbers from ${PROBE_PERIOD} ==="
python scripts/discover_absorbers.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --probe-period "${PROBE_PERIOD}" \
    --recall-threshold 0.1 \
    --top-k-absorbers 10 \
    --output-dir "${DISCOVERY_DIR}"

# Parse discovered classes from JSON
DISCOVERY_FILE="${DISCOVERY_DIR}/discovered_${PROBE_SLUG}.json"
COLLAPSE_CLASSES=$(python -c "import json; d=json.load(open('${DISCOVERY_FILE}')); print(','.join(str(c) for c in d['collapse_classes']))")
ABSORBER_CLASSES=$(python -c "import json; d=json.load(open('${DISCOVERY_FILE}')); print(','.join(str(c) for c in d['absorber_classes']))")

echo ""
echo "Discovered collapse classes: ${COLLAPSE_CLASSES}"
echo "Discovered absorber classes: ${ABSORBER_CLASSES}"
echo ""

echo "=== Step 2: Run CARE on ${TARGET_PERIOD} with auto-discovered lists ==="
for SEED in 0 1 2 3 4; do
    echo "--- Seed ${SEED} ---"
    python scripts/collapse_active_maintenance_tls22.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --target-period "${TARGET_PERIOD}" \
        --strategies "margin,absorber_margin" \
        --budgets "200,500,1000" \
        --collapse-classes "${COLLAPSE_CLASSES}" \
        --absorber-classes "${ABSORBER_CLASSES}" \
        --replay-mode stable_absorber \
        --replay-per-class 5 \
        --replay-distill-weight 0.5 \
        --target-repeat 2 \
        --seed "${SEED}" \
        --output-dir "${OUTPUT_DIR}/seed_${SEED}"
    echo ""
done

echo "=== Step 3: Aggregate results ==="
python scripts/aggregate_seeds.py --base-dir "${OUTPUT_DIR}"

echo ""
echo "=== Done. Auto-absorber CARE results in ${OUTPUT_DIR}/ ==="
