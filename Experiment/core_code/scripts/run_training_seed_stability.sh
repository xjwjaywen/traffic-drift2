#!/bin/bash
# Train models with different seeds and check collapse class stability.
# Purpose: verify absorber-collapse is a data property, not training artifact.
#
# Usage: CUDA_VISIBLE_DEVICES=2 bash scripts/run_training_seed_stability.sh

set -euo pipefail

CONFIG="configs/train_tls22_cnn.yaml"
EVAL_CONFIG="configs/eval_tls22.yaml"

echo "=== Training 3 models with different seeds ==="
for SEED in 0 1 2; do
    OUTPUT="outputs/tls22_cnn_seed${SEED}"
    if [ -f "${OUTPUT}/best_model.pt" ]; then
        echo "--- Seed ${SEED}: already trained, skipping ---"
    else
        echo "--- Training seed ${SEED} ---"
        python train.py --config "${CONFIG}" --output-dir "${OUTPUT}" --seed "${SEED}"
    fi
done

echo ""
echo "=== Evaluating static model at M12 for each seed ==="
for SEED in 0 1 2; do
    CHECKPOINT="outputs/tls22_cnn_seed${SEED}/best_model.pt"
    echo "--- Seed ${SEED} ---"
    python scripts/eval_baselines_with_groups.py \
        --config "${EVAL_CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --test-period M-2022-12 \
        --output-dir "outputs/seed_stability/seed${SEED}_static"
done

echo ""
echo "=== Checking collapse class overlap ==="
python -c "
import csv, os
all_collapsed = []
for seed in range(3):
    path = f'outputs/seed_stability/seed{seed}_static/baselines_group_metrics.csv'
    if not os.path.exists(path):
        # Try per-class report
        continue
    print(f'Seed {seed}: see {path}')

# Compare per-class recalls
for seed in range(3):
    ckpt = f'outputs/tls22_cnn_seed{seed}/best_model.pt'
    print(f'Seed {seed} checkpoint: {ckpt}')
print('Manual inspection needed: check which classes have recall < 0.1 for each seed')
"

echo "=== Done ==="
