#!/bin/bash
# Ablation study with strict evaluation (5 seeds each).
#
# Three cleanly isolated configurations (paper Table 7):
#   1. FT only:     target labels → CE, no replay, no distillation
#   2. FT + Replay: target labels + 150 replay samples → CE, no distillation
#   3. Full CARE:   target labels + replay → CE, + KL distillation on replay
#
# Note: "FT + Distill" (distillation without replay) is not included because
# distillation requires reference features as KL targets, making it impossible
# to cleanly separate from replay. The paper acknowledges this.
#
# ft_distill is retained in the script for historical reference but is NOT
# used in the paper's ablation table.
#
# Usage: bash scripts/run_ablation_strict.sh <config> <checkpoint> <output_base>

set -euo pipefail

CONFIG="${1:?Usage: $0 <config> <checkpoint> <output_base>}"
CHECKPOINT="${2:?}"
OUTPUT_BASE="${3:?}"

STRATEGIES="margin"
BUDGETS="200,500,1000"

declare -A CONFIGS
CONFIGS[ft_only]="--replay-mode none --replay-per-class 0 --replay-distill-weight 0"
CONFIGS[ft_replay]="--replay-mode stable_absorber --replay-per-class 5 --replay-distill-weight 0 --target-repeat 2"
CONFIGS[ft_distill]="--replay-mode stable_absorber --replay-per-class 5 --replay-distill-weight 0.5 --target-repeat 1"
CONFIGS[full_care]="--replay-mode stable_absorber --replay-per-class 5 --replay-distill-weight 0.5 --target-repeat 2"

for CONFIG_NAME in ft_only ft_replay ft_distill full_care; do
    EXTRA_ARGS="${CONFIGS[$CONFIG_NAME]}"
    echo "========================================"
    echo "=== Ablation: ${CONFIG_NAME} ==="
    echo "=== Args: ${EXTRA_ARGS} ==="
    echo "========================================"

    for SEED in 0 1 2 3 4; do
        echo "--- ${CONFIG_NAME} seed ${SEED} ---"
        python scripts/collapse_active_maintenance_tls22.py \
            --config "${CONFIG}" \
            --checkpoint "${CHECKPOINT}" \
            --strategies "${STRATEGIES}" \
            --budgets "${BUDGETS}" \
            ${EXTRA_ARGS} \
            --seed "${SEED}" \
            --output-dir "${OUTPUT_BASE}/${CONFIG_NAME}/seed_${SEED}"
    done
    echo ""

    echo "--- Aggregate ${CONFIG_NAME} ---"
    python scripts/aggregate_seeds.py --base-dir "${OUTPUT_BASE}/${CONFIG_NAME}"
    echo ""
done

echo "=== All ablation experiments complete ==="
