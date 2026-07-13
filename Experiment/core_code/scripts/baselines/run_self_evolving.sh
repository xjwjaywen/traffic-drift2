#!/usr/bin/env bash
# Run self-evolving pseudo-label baseline (Chen et al. 2025 inspired)
# Sweeps confidence thresholds across 5 seeds
set -euo pipefail
cd "$(dirname "$0")/../.."

CHECKPOINT="outputs/tls22_cnn/best_model.pt"
CONFIG="configs/eval_tls22.yaml"
BASE_OUT="outputs/baselines/self_evolving"
THRESHOLDS="0.90,0.95,0.99,0.997"

echo "=== Self-Evolving Baseline (pure pseudo-label, no replay) ==="
for SEED in 0 1 2 3 4; do
    echo "--- Seed ${SEED} ---"
    python scripts/baselines/self_evolving_baseline.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --output-dir "${BASE_OUT}/seed_${SEED}" \
        --seed "${SEED}" \
        --thresholds "${THRESHOLDS}" \
        --replay-mode none
done

echo ""
echo "=== Aggregating seeds ==="
python scripts/aggregate_seeds.py --base-dir "${BASE_OUT}"

echo ""
echo "=== Self-Evolving + All-Class Replay + KD ==="
BASE_OUT_RPL="outputs/baselines/self_evolving_replay_kd"
for SEED in 0 1 2 3 4; do
    echo "--- Seed ${SEED} ---"
    python scripts/baselines/self_evolving_baseline.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --output-dir "${BASE_OUT_RPL}/seed_${SEED}" \
        --seed "${SEED}" \
        --thresholds "${THRESHOLDS}" \
        --replay-mode all \
        --replay-per-class 5 \
        --replay-distill-weight 0.5
done

echo ""
echo "=== Aggregating replay+kd seeds ==="
python scripts/aggregate_seeds.py --base-dir "${BASE_OUT_RPL}"

echo "Done."
