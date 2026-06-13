#!/bin/bash
# Run CARE experiments with 5 seeds and strict evaluation (queried samples excluded).
# Usage: bash scripts/run_5seeds_strict.sh <config> <checkpoint> <output_base>
#
# Example (CNN):
#   bash scripts/run_5seeds_strict.sh configs/eval_tls22.yaml outputs/tls22_cnn/best_model.pt outputs/care_5seeds_strict_cnn
# Example (Transformer):
#   bash scripts/run_5seeds_strict.sh outputs/tls22_transformer/config.yaml outputs/tls22_transformer/best_model.pt outputs/care_5seeds_strict_transformer

set -euo pipefail

CONFIG="${1:?Usage: $0 <config> <checkpoint> <output_base>}"
CHECKPOINT="${2:?}"
OUTPUT_BASE="${3:?}"

STRATEGIES="random,margin,absorber_margin"
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

echo "=== All 5 seeds complete. Results in ${OUTPUT_BASE}/ ==="
