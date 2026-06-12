#!/bin/bash
# Active learning baseline comparison (5 seeds).
# Compares: random, entropy, margin, coreset, badge
# All use full CARE (replay + distill) to isolate the selection strategy effect.
#
# Usage: bash scripts/run_al_baselines.sh <config> <checkpoint> <output_base>

set -euo pipefail

CONFIG="${1:?Usage: $0 <config> <checkpoint> <output_base>}"
CHECKPOINT="${2:?}"
OUTPUT_BASE="${3:?}"

STRATEGIES="random,entropy,margin,coreset,badge"
BUDGETS="200,500,1000"

for SEED in 0 1 2 3 4; do
    echo "=== Seed ${SEED} ==="
    python scripts/collapse_active_maintenance_tls22.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --strategies "${STRATEGIES}" \
        --budgets "${BUDGETS}" \
        --replay-mode stable_absorber \
        --replay-per-class 5 \
        --replay-distill-weight 0.5 \
        --target-repeat 2 \
        --seed "${SEED}" \
        --output-dir "${OUTPUT_BASE}/seed_${SEED}"
    echo ""
done

echo "=== Aggregate ==="
python scripts/aggregate_seeds.py --base-dir "${OUTPUT_BASE}"
echo "=== Done ==="
