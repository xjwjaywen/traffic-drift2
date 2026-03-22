#!/bin/bash
# ============================================================
# TTA-TC: Complete Experiment Pipeline
# ============================================================
# Runs all experiments in order:
#   0. Verify pipeline with synthetic data (no CESNET needed)
#   1. Download data
#   2. Train source models
#   3. Evaluate all TTA methods (single + sequential, QUIC22)
#   4. Run ablation studies
#   5. Long-term evaluation (TLS-Year22, 9 months)
#   6. Evaluate Transformer backbone
#   7. Aggregate results + generate figures + LaTeX tables
#
# Usage:
#   bash scripts/run_all.sh [--skip-verify] [--skip-download] \
#                           [--skip-train] [--size-m] [--gpu]
#
# Prerequisites:
#   conda activate tta-tc
# ============================================================
set -e

SKIP_VERIFY=false
SKIP_DOWNLOAD=false
SKIP_TRAIN=false
GPU_FLAG=""
DATA_SIZE="S"   # S for dev, M for final experiments

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-verify)   SKIP_VERIFY=true   ;;
        --skip-download) SKIP_DOWNLOAD=true ;;
        --skip-train)    SKIP_TRAIN=true    ;;
        --gpu)           GPU_FLAG="--gpu"   ;;
        --size-m)        DATA_SIZE="M"      ;;
    esac
    shift
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ANALYSIS_DIR="$(dirname "$PROJECT_DIR")/analysis"
cd "$PROJECT_DIR"

echo "============================================================"
echo "TTA-TC Experiment Pipeline"
echo "Working directory: $PROJECT_DIR"
echo "Analysis directory: $ANALYSIS_DIR"
echo "Data size: $DATA_SIZE"
echo "============================================================"

# ============================================================
# Step 0: Verify environment
# ============================================================
echo ""
echo ">>> Step 0: Verifying Python environment..."
python -c "import torch; print(f'PyTorch {torch.__version__}')"

if [ "$SKIP_VERIFY" = false ]; then
    echo ">>> Step 0b: Pipeline verification with synthetic data..."
    python scripts/verify_pipeline.py --output-dir outputs/verify
    echo "Verification passed."
fi

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
# Step 3: Evaluate all methods on QUIC22 (single period W-45)
# ============================================================
echo ""
echo ">>> Step 3: Evaluating all methods on QUIC22 (W-45, single period)..."
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
echo ">>> Step 5a: Ablation A1 — SSL task selection..."
python run_ablation.py \
    --ablation-config configs/ablation_ssl_tasks.yaml \
    --checkpoint outputs/quic22_cnn/best_model.pt \
    --output-dir outputs/ablations/ssl_tasks

echo ""
echo ">>> Step 5b: Ablation A2 — Mask ratio..."
python run_ablation.py \
    --ablation-config configs/ablation_mask_ratio.yaml \
    --checkpoint outputs/quic22_cnn/best_model.pt \
    --output-dir outputs/ablations/mask_ratio

echo ""
echo ">>> Step 5c: Ablation A3 — Adaptation depth..."
python run_ablation.py \
    --ablation-config configs/ablation_adapt_depth.yaml \
    --checkpoint outputs/quic22_cnn/best_model.pt \
    --output-dir outputs/ablations/adapt_depth

echo ""
echo ">>> Step 5d: Ablation A5 — Anti-forgetting mechanisms..."
python run_ablation.py \
    --ablation-config configs/ablation_anti_forgetting.yaml \
    --checkpoint outputs/quic22_cnn/best_model.pt \
    --output-dir outputs/ablations/anti_forgetting

echo ""
echo ">>> Step 5e: Ablation A8 — Normalization type..."
python run_ablation.py \
    --ablation-config configs/ablation_norm_type.yaml \
    --checkpoint outputs/quic22_cnn/best_model.pt \
    --output-dir outputs/ablations/norm_type

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
# Step 7: Transformer backbone evaluation
# ============================================================
echo ""
echo ">>> Step 7: Evaluating Transformer backbone on QUIC22..."
python evaluate_tta.py \
    --config configs/eval_quic22.yaml \
    --checkpoint outputs/quic22_transformer/best_model.pt \
    --output-dir outputs/eval_quic22_transformer \
    --mode sequential

# ============================================================
# Step 8: Aggregate results + figures + LaTeX tables
# ============================================================
echo ""
echo ">>> Step 8: Aggregating results..."
python "$ANALYSIS_DIR/aggregate_results.py" \
    --output-dir outputs/ \
    --save-dir "$ANALYSIS_DIR/results/"

echo ""
echo ">>> Step 8b: Generating figures..."
python "$ANALYSIS_DIR/visualize_results.py" \
    --outputs-dir outputs/ \
    --save-dir "$ANALYSIS_DIR/figures/"

echo ""
echo ">>> Step 8c: Generating LaTeX tables..."
python "$ANALYSIS_DIR/make_paper_tables.py" \
    --outputs-dir outputs/ \
    --save-dir "$ANALYSIS_DIR/tables/"

# ============================================================
# Done
# ============================================================
echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "Key output locations:"
echo "  Checkpoints:   outputs/quic22_cnn/best_model.pt"
echo "  Single eval:   outputs/eval_quic22_single/results_single.json"
echo "  Sequential:    outputs/eval_quic22_sequential/results_sequential.json"
echo "  Ablations:     outputs/ablations/*/ablation_results.json"
echo "  TLS-Year22:    outputs/eval_tls22_sequential/results_sequential.json"
echo "  Transformer:   outputs/eval_quic22_transformer/results_sequential.json"
echo ""
echo "Analysis outputs:"
echo "  Summary JSON:  $ANALYSIS_DIR/results/all_results.json"
echo "  Figures:       $ANALYSIS_DIR/figures/"
echo "  LaTeX tables:  $ANALYSIS_DIR/tables/"
echo ""
echo "TensorBoard:"
echo "  tensorboard --logdir outputs/"
