#!/bin/bash
# ============================================================
# TTA-TC: Complete Experiment Pipeline
# ============================================================
# Runs all experiments in order:
#   1. Download data
#   2. Train source models
#   3. Evaluate all TTA methods
#   4. Run ablation studies
#   5. Run long-term evaluation (TLS-Year22)
#
# Usage:
#   bash scripts/run_all.sh [--skip-download] [--skip-train] [--gpu]
#
# Prerequisites:
#   conda activate tta-tc
# ============================================================
set -e

# Parse arguments
SKIP_DOWNLOAD=false
SKIP_TRAIN=false
GPU_FLAG=""
DATA_SIZE="S"   # S for dev, M for final experiments

for arg in "$@"; do
    case $arg in
        --skip-download) SKIP_DOWNLOAD=true ;;
        --skip-train) SKIP_TRAIN=true ;;
        --gpu) GPU_FLAG="--gpu" ;;
        --size-m) DATA_SIZE="M" ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================================"
echo "TTA-TC Experiment Pipeline"
echo "Working directory: $PROJECT_DIR"
echo "Data size: $DATA_SIZE"
echo "============================================================"

# ============================================================
# Step 0: Verify environment
# ============================================================
echo ""
echo ">>> Step 0: Verifying environment..."
python -c "import torch; import cesnet_datazoo; print(f'PyTorch {torch.__version__}, CESNET-DataZoo OK')"
echo "Environment verified."

# ============================================================
# Step 1: Download data
# ============================================================
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo ""
    echo ">>> Step 1: Downloading datasets..."
    bash scripts/download_data.sh --size "$DATA_SIZE" --data-dir ./data
else
    echo ""
    echo ">>> Step 1: Skipping download (--skip-download)"
fi

# ============================================================
# Step 2: Train source models
# ============================================================
if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo ">>> Step 2a: Training CNN model on CESNET-QUIC22..."
    python train.py --config configs/train_quic22_cnn.yaml

    echo ""
    echo ">>> Step 2b: Training Transformer model on CESNET-QUIC22..."
    python train.py --config configs/train_quic22_transformer.yaml

    echo ""
    echo ">>> Step 2c: Training CNN model on CESNET-TLS-Year22..."
    python train.py --config configs/train_tls22_cnn.yaml
else
    echo ""
    echo ">>> Step 2: Skipping training (--skip-train)"
fi

# ============================================================
# Step 3: Evaluate all methods on QUIC22 (single period)
# ============================================================
echo ""
echo ">>> Step 3: Evaluating all methods on QUIC22 (W-45)..."
python evaluate_tta.py \
    --config configs/eval_quic22.yaml \
    --checkpoint outputs/quic22_cnn/best_model.pt \
    --output-dir outputs/eval_quic22_single \
    --mode single

# ============================================================
# Step 4: Sequential evaluation on QUIC22 (W-45 -> W-46 -> W-47)
# ============================================================
echo ""
echo ">>> Step 4: Sequential evaluation on QUIC22 (3 weeks)..."
python evaluate_tta.py \
    --config configs/eval_quic22.yaml \
    --checkpoint outputs/quic22_cnn/best_model.pt \
    --output-dir outputs/eval_quic22_sequential \
    --mode sequential

# ============================================================
# Step 5: Ablation studies
# ============================================================
echo ""
echo ">>> Step 5a: Ablation - SSL task selection..."
python run_ablation.py \
    --ablation-config configs/ablation_ssl_tasks.yaml \
    --checkpoint outputs/quic22_cnn/best_model.pt \
    --output-dir outputs/ablations/ssl_tasks

echo ""
echo ">>> Step 5b: Ablation - Mask ratio..."
python run_ablation.py \
    --ablation-config configs/ablation_mask_ratio.yaml \
    --checkpoint outputs/quic22_cnn/best_model.pt \
    --output-dir outputs/ablations/mask_ratio

echo ""
echo ">>> Step 5c: Ablation - Adaptation depth..."
python run_ablation.py \
    --ablation-config configs/ablation_adapt_depth.yaml \
    --checkpoint outputs/quic22_cnn/best_model.pt \
    --output-dir outputs/ablations/adapt_depth

echo ""
echo ">>> Step 5d: Ablation - Anti-forgetting..."
python run_ablation.py \
    --ablation-config configs/ablation_anti_forgetting.yaml \
    --checkpoint outputs/quic22_cnn/best_model.pt \
    --output-dir outputs/ablations/anti_forgetting

# ============================================================
# Step 6: Long-term evaluation on TLS-Year22 (9 months)
# ============================================================
echo ""
echo ">>> Step 6: Sequential evaluation on TLS-Year22 (9 months)..."
python evaluate_tta.py \
    --config configs/eval_tls22.yaml \
    --checkpoint outputs/tls22_cnn/best_model.pt \
    --output-dir outputs/eval_tls22_sequential \
    --mode sequential

# ============================================================
# Step 7: Evaluate Transformer backbone
# ============================================================
echo ""
echo ">>> Step 7: Evaluating Transformer backbone on QUIC22..."
python evaluate_tta.py \
    --config configs/eval_quic22.yaml \
    --checkpoint outputs/quic22_transformer/best_model.pt \
    --output-dir outputs/eval_quic22_transformer \
    --mode sequential

# ============================================================
# Done
# ============================================================
echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "Results locations:"
echo "  Training:     outputs/quic22_cnn/train_results.json"
echo "  Single eval:  outputs/eval_quic22_single/results_single.json"
echo "  Sequential:   outputs/eval_quic22_sequential/results_sequential.json"
echo "  Ablations:    outputs/ablations/*/ablation_results.json"
echo "  TLS-Year22:   outputs/eval_tls22_sequential/results_sequential.json"
echo "  Transformer:  outputs/eval_quic22_transformer/results_sequential.json"
echo ""
echo "TensorBoard logs: outputs/*/tb_logs/"
echo "  tensorboard --logdir outputs/"
