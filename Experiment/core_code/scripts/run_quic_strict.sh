#!/bin/bash
# QUIC22 strict evaluation with 5 seeds.
# Uses auto-discovered absorbers from W-2022-46 to evaluate on W-2022-47.
#
# Usage: bash scripts/run_quic_strict.sh <output_base>

set -euo pipefail

OUTPUT_BASE="${1:-outputs/care_quic22_strict}"
CONFIG="configs/eval_quic22.yaml"
CHECKPOINT="outputs/quic22_cnn/best_model.pt"
PROBE_PERIOD="W-2022-46"
TARGET_PERIOD="W-2022-47"

echo "=== Step 1: Discover absorbers from ${PROBE_PERIOD} ==="
python scripts/discover_absorbers.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --probe-period "${PROBE_PERIOD}" \
    --recall-threshold 0.1 \
    --top-k-absorbers 10 \
    --output-dir "${OUTPUT_BASE}/discovery"

PROBE_SLUG=$(echo "${PROBE_PERIOD}" | tr '-' '_')
DISCOVERY_FILE="${OUTPUT_BASE}/discovery/discovered_${PROBE_SLUG}.json"
COLLAPSE_CLASSES=$(python -c "import json; d=json.load(open('${DISCOVERY_FILE}')); print(','.join(str(c) for c in d['collapse_classes']))")
ABSORBER_CLASSES=$(python -c "import json; d=json.load(open('${DISCOVERY_FILE}')); print(','.join(str(c) for c in d['absorber_classes']))")

echo "Discovered collapse: ${COLLAPSE_CLASSES}"
echo "Discovered absorbers: ${ABSORBER_CLASSES}"

echo ""
echo "=== Step 2: Run CARE with 5 seeds ==="
for SEED in 0 1 2 3 4; do
    echo "--- Seed ${SEED} ---"
    python scripts/collapse_active_maintenance_tls22.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --reference-period "W-2022-45" \
        --target-period "${TARGET_PERIOD}" \
        --strategies "random,margin,absorber_margin" \
        --budgets "200,500,1000" \
        --collapse-classes "${COLLAPSE_CLASSES}" \
        --absorber-classes "${ABSORBER_CLASSES}" \
        --replay-mode stable_absorber \
        --replay-per-class 5 \
        --replay-distill-weight 0.5 \
        --target-repeat 2 \
        --seed "${SEED}" \
        --output-dir "${OUTPUT_BASE}/seed_${SEED}"
    echo ""
done

echo "=== Step 3: Aggregate ==="
python scripts/aggregate_seeds.py --base-dir "${OUTPUT_BASE}"
echo "=== Done ==="
