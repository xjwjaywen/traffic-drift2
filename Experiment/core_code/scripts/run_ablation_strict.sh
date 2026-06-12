#!/bin/bash
# Ablation study with strict evaluation (5 seeds each).
# Configurations:
#   1. FT only        (no replay, no distill)
#   2. FT + Replay    (replay, no distill)
#   3. FT + Distill   (no replay, distill on reference samples... but distill needs replay features)
#   4. Full CARE      (replay + distill)
#
# Note: "FT + Distill" uses stable_absorber replay for distillation targets
# but we separate the replay CE effect by setting target-repeat high to dilute replay.
# Actually, simpler: FT+Distill = replay + distill but replay-per-class=0 won't work
# because distill needs replay features. So we use two clean configs:
#   - FT only:     replay-mode=none, distill=0
#   - FT+Replay:   replay-mode=stable_absorber, distill=0
#   - FT+Distill:  replay-mode=stable_absorber, distill=0.5, replay-per-class=5
#                   (but we zero out replay CE by a workaround... not clean)
#
# Cleanest approach: 4 configs that clearly isolate components:
#   1. FT only:           replay=none,             distill=0
#   2. FT + Replay:       replay=stable_absorber,  distill=0
#   3. FT + Replay+Dist:  replay=stable_absorber,  distill=0.5
#   (FT + Distill alone is not cleanly separable since distill needs replay samples)
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
