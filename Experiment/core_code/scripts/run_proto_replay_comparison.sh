#!/bin/bash
# Compare real replay vs prototype replay (exemplar-free).
# Proves CARE works without storing source training data.
#
# Usage: bash scripts/run_proto_replay_comparison.sh <config> <checkpoint> <output_base>

set -euo pipefail

CONFIG="${1:?Usage: $0 <config> <checkpoint> <output_base>}"
CHECKPOINT="${2:?}"
OUTPUT_BASE="${3:?}"

STRATEGIES="margin"
BUDGETS="200,500,1000"

# Config 1: Real replay (current CARE)
echo "========================================"
echo "=== Real Replay (stable_absorber) ==="
echo "========================================"
for SEED in 0 1 2; do
    echo "--- seed ${SEED} ---"
    python scripts/collapse_active_maintenance_tls22.py \
        --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
        --strategies "${STRATEGIES}" --budgets "${BUDGETS}" \
        --replay-mode stable_absorber --replay-per-class 5 \
        --replay-distill-weight 0.5 --target-repeat 2 \
        --seed "${SEED}" \
        --output-dir "${OUTPUT_BASE}/real_replay/seed_${SEED}"
done
python scripts/aggregate_seeds.py --base-dir "${OUTPUT_BASE}/real_replay" --num-seeds 3

# Config 2: Prototype replay (exemplar-free)
echo "========================================"
echo "=== Prototype Replay (proto_stable_absorber) ==="
echo "========================================"
for SEED in 0 1 2; do
    echo "--- seed ${SEED} ---"
    python scripts/collapse_active_maintenance_tls22.py \
        --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
        --strategies "${STRATEGIES}" --budgets "${BUDGETS}" \
        --replay-mode proto_stable_absorber --replay-per-class 5 \
        --replay-distill-weight 0.5 --target-repeat 2 \
        --seed "${SEED}" \
        --output-dir "${OUTPUT_BASE}/proto_replay/seed_${SEED}"
done
python scripts/aggregate_seeds.py --base-dir "${OUTPUT_BASE}/proto_replay" --num-seeds 3

echo ""
echo "=== Comparison Complete ==="
echo "Real replay: ${OUTPUT_BASE}/real_replay/aggregated_mean_std.csv"
echo "Proto replay: ${OUTPUT_BASE}/proto_replay/aggregated_mean_std.csv"
