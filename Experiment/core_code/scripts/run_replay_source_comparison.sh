#!/bin/bash
# Compare replay source variants: M3 replay vs M4 replay vs no replay.
#
# Runs collapse_active_maintenance_tls22.py three times with different
# reference/replay configurations to measure how replay source period
# affects CARE repair quality.
#
# Output structure:
#   outputs/replay_source_comparison/m3/seed_{0..4}/
#   outputs/replay_source_comparison/m4/seed_{0..4}/
#   outputs/replay_source_comparison/none/seed_{0..4}/
#
# Usage from Experiment/core_code/:
#   bash scripts/run_replay_source_comparison.sh [config] [checkpoint] [num_seeds]

set -euo pipefail

CONFIG="${1:-configs/eval_tls22.yaml}"
CHECKPOINT="${2:-outputs/tls22_cnn/best_model.pt}"
NUM_SEEDS="${3:-5}"
OUTPUT_BASE="outputs/replay_source_comparison"
TARGET_PERIOD="M-2022-12"

# Fixed 12 collapse classes and absorbers (same across all runs for fair comparison)
COLLAPSE_CLASSES="56,163,174,48,38,69,104,47,66,10,109,26"
ABSORBER_CLASSES="96,46,2,14,45,105,5,71,156,13"

echo "============================================================"
echo "Replay Source Comparison Experiment"
echo "============================================================"
echo "Config:        ${CONFIG}"
echo "Checkpoint:    ${CHECKPOINT}"
echo "Target period: ${TARGET_PERIOD}"
echo "Seeds:         ${NUM_SEEDS}"
echo "Output:        ${OUTPUT_BASE}/"
echo ""

# ---- Variant 1: M3 replay (reference=M-2022-3) ----
echo "============================================================"
echo "=== Variant 1: M3 replay (reference-period=M-2022-3) ==="
echo "============================================================"
for SEED in $(seq 0 $((NUM_SEEDS - 1))); do
    echo "--- M3 replay, seed ${SEED} ---"
    python scripts/collapse_active_maintenance_tls22.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --reference-period "M-2022-3" \
        --target-period "${TARGET_PERIOD}" \
        --strategies "margin" \
        --budgets "1000" \
        --collapse-classes "${COLLAPSE_CLASSES}" \
        --absorber-classes "${ABSORBER_CLASSES}" \
        --replay-mode stable_absorber \
        --replay-per-class 5 \
        --replay-distill-weight 0.5 \
        --target-repeat 2 \
        --seed "${SEED}" \
        --output-dir "${OUTPUT_BASE}/m3/seed_${SEED}"
done
echo ""
echo "Aggregating M3 results..."
python scripts/aggregate_seeds.py --base-dir "${OUTPUT_BASE}/m3" --num-seeds "${NUM_SEEDS}"

# ---- Variant 2: M4 replay (reference=M-2022-4, current default) ----
echo ""
echo "============================================================"
echo "=== Variant 2: M4 replay (reference-period=M-2022-4) ==="
echo "============================================================"
for SEED in $(seq 0 $((NUM_SEEDS - 1))); do
    echo "--- M4 replay, seed ${SEED} ---"
    python scripts/collapse_active_maintenance_tls22.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --reference-period "M-2022-4" \
        --target-period "${TARGET_PERIOD}" \
        --strategies "margin" \
        --budgets "1000" \
        --collapse-classes "${COLLAPSE_CLASSES}" \
        --absorber-classes "${ABSORBER_CLASSES}" \
        --replay-mode stable_absorber \
        --replay-per-class 5 \
        --replay-distill-weight 0.5 \
        --target-repeat 2 \
        --seed "${SEED}" \
        --output-dir "${OUTPUT_BASE}/m4/seed_${SEED}"
done
echo ""
echo "Aggregating M4 results..."
python scripts/aggregate_seeds.py --base-dir "${OUTPUT_BASE}/m4" --num-seeds "${NUM_SEEDS}"

# ---- Variant 3: No replay ----
echo ""
echo "============================================================"
echo "=== Variant 3: No replay (replay-mode=none) ==="
echo "============================================================"
for SEED in $(seq 0 $((NUM_SEEDS - 1))); do
    echo "--- No replay, seed ${SEED} ---"
    python scripts/collapse_active_maintenance_tls22.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --reference-period "M-2022-4" \
        --target-period "${TARGET_PERIOD}" \
        --strategies "margin" \
        --budgets "1000" \
        --collapse-classes "${COLLAPSE_CLASSES}" \
        --absorber-classes "${ABSORBER_CLASSES}" \
        --replay-mode none \
        --replay-per-class 0 \
        --target-repeat 1 \
        --seed "${SEED}" \
        --output-dir "${OUTPUT_BASE}/none/seed_${SEED}"
done
echo ""
echo "Aggregating no-replay results..."
python scripts/aggregate_seeds.py --base-dir "${OUTPUT_BASE}/none" --num-seeds "${NUM_SEEDS}"

# ---- Summary ----
echo ""
echo "============================================================"
echo "=== Replay Source Comparison Complete ==="
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  M3 replay:  ${OUTPUT_BASE}/m3/aggregated_mean_std.csv"
echo "  M4 replay:  ${OUTPUT_BASE}/m4/aggregated_mean_std.csv"
echo "  No replay:  ${OUTPUT_BASE}/none/aggregated_mean_std.csv"
echo ""
echo "Compare with:"
echo "  diff <(head -5 ${OUTPUT_BASE}/m3/aggregated_mean_std.csv) <(head -5 ${OUTPUT_BASE}/m4/aggregated_mean_std.csv)"
