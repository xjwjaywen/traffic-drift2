#!/usr/bin/env bash
# Run self-evolving pseudo-label baseline (Chen et al. 2025 inspired)
# v2: Fixed protocol — holdout split, full FFT, Chen et al. hyperparameters,
#     strict evaluation, proper per-class stats.
set -euo pipefail
cd "$(dirname "$0")/../.."

CHECKPOINT="outputs/tls22_cnn/best_model.pt"
CONFIG="configs/eval_tls22.yaml"
THRESHOLDS="0.90,0.95,0.99,0.997"

# --- Chen et al. hyperparameters ---
LR=0.0025
EPOCHS=50
BATCH=500
HOLDOUT=0.2

echo "============================================================"
echo "=== Self-Evolving Baseline v2 (full FFT, holdout split)  ==="
echo "=== lr=${LR} epochs=${EPOCHS} batch=${BATCH} holdout=${HOLDOUT} ==="
echo "============================================================"

echo ""
echo "=== 1. Pure self-evolving (full FFT, no replay) ==="
BASE_OUT="outputs/baselines/self_evolving_v2"
for SEED in 0 1 2 3 4; do
    echo "--- Seed ${SEED} ---"
    python3 scripts/baselines/self_evolving_baseline.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --output-dir "${BASE_OUT}/seed_${SEED}" \
        --seed "${SEED}" \
        --thresholds "${THRESHOLDS}" \
        --holdout-ratio "${HOLDOUT}" \
        --ft-depth full \
        --ft-lr "${LR}" \
        --ft-epochs "${EPOCHS}" \
        --ft-batch-size "${BATCH}" \
        --replay-mode none
done
echo "=== Aggregating seeds ==="
python3 scripts/aggregate_seeds.py --base-dir "${BASE_OUT}"

echo ""
echo "=== 2. Self-Evolving + All-Class Replay + KD (full FFT) ==="
BASE_OUT_RPL="outputs/baselines/self_evolving_replay_kd_v2"
for SEED in 0 1 2 3 4; do
    echo "--- Seed ${SEED} ---"
    python3 scripts/baselines/self_evolving_baseline.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --output-dir "${BASE_OUT_RPL}/seed_${SEED}" \
        --seed "${SEED}" \
        --thresholds "${THRESHOLDS}" \
        --holdout-ratio "${HOLDOUT}" \
        --ft-depth full \
        --ft-lr "${LR}" \
        --ft-epochs "${EPOCHS}" \
        --ft-batch-size "${BATCH}" \
        --replay-mode all \
        --replay-per-class 5 \
        --replay-distill-weight 0.5
done
echo "=== Aggregating replay+kd seeds ==="
python3 scripts/aggregate_seeds.py --base-dir "${BASE_OUT_RPL}"

echo ""
echo "=== 3. Head-only ablation (for comparison with CARE head-only) ==="
BASE_OUT_HEAD="outputs/baselines/self_evolving_head_v2"
for SEED in 0 1 2 3 4; do
    echo "--- Seed ${SEED} ---"
    python3 scripts/baselines/self_evolving_baseline.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --output-dir "${BASE_OUT_HEAD}/seed_${SEED}" \
        --seed "${SEED}" \
        --thresholds "${THRESHOLDS}" \
        --holdout-ratio "${HOLDOUT}" \
        --ft-depth head \
        --ft-lr "${LR}" \
        --ft-epochs "${EPOCHS}" \
        --ft-batch-size "${BATCH}" \
        --replay-mode none
done
echo "=== Aggregating head-only seeds ==="
python3 scripts/aggregate_seeds.py --base-dir "${BASE_OUT_HEAD}"

echo ""
echo "Done. Results in outputs/baselines/self_evolving_v2/"
echo "                    outputs/baselines/self_evolving_replay_kd_v2/"
echo "                    outputs/baselines/self_evolving_head_v2/"
