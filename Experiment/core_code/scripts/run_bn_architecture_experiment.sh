#!/bin/bash
# BN Architecture Experiment: train CNN-BN, run TTA baselines, then CARE
# Purpose: distinguish "TTA fails on traffic" vs "TTA fails on GroupNorm"
#
# Usage: CUDA_VISIBLE_DEVICES=2 bash scripts/run_bn_architecture_experiment.sh

set -euo pipefail

CONFIG_TRAIN="configs/train_tls22_cnn_bn.yaml"
OUTPUT_MODEL="outputs/tls22_cnn_bn"
CONFIG_EVAL="configs/eval_tls22.yaml"

echo "============================================"
echo "=== Step 1: Train CNN-BN model ==="
echo "============================================"
python train.py --config "${CONFIG_TRAIN}" --output-dir "${OUTPUT_MODEL}"

echo ""
echo "============================================"
echo "=== Step 2: Evaluate TTA baselines on BN ==="
echo "============================================"
for METHOD in static bn_adapt tent eata sar cotta note; do
    echo "--- ${METHOD} ---"
    python scripts/eval_baselines_with_groups.py \
        --config "${CONFIG_EVAL}" \
        --checkpoint "${OUTPUT_MODEL}/best_model.pt" \
        --test-period M-2022-12 \
        --method "${METHOD}" \
        --output-dir "outputs/bn_tta_baselines_M12" \
        2>/dev/null || echo "SKIP ${METHOD} (not supported or error)"
done

echo ""
echo "============================================"
echo "=== Step 3: Run CARE on BN model (3 seeds) ==="
echo "============================================"
for SEED in 0 1 2; do
    echo "--- CARE seed ${SEED} ---"
    python scripts/collapse_active_maintenance_tls22.py \
        --config "${CONFIG_EVAL}" \
        --checkpoint "${OUTPUT_MODEL}/best_model.pt" \
        --strategies "margin" \
        --budgets "1000" \
        --replay-mode stable_absorber \
        --replay-per-class 5 \
        --replay-distill-weight 0.5 \
        --target-repeat 2 \
        --seed "${SEED}" \
        --output-dir "outputs/care_bn_strict/seed_${SEED}"
done
python scripts/aggregate_seeds.py --base-dir "outputs/care_bn_strict" --num-seeds 3

echo ""
echo "=== BN Architecture Experiment Complete ==="
