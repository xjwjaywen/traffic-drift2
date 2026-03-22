#!/bin/bash
# ============================================================
# TTA-TC Environment Setup
# ============================================================
# Usage: bash setup.sh [--gpu]
# --gpu : Install CUDA-enabled PyTorch (default: CPU/MPS)
# ============================================================
set -e

ENV_NAME="tta-tc"
PYTHON_VERSION="3.10"

echo "=== TTA-TC Environment Setup ==="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Install Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Parse args
USE_GPU=false
for arg in "$@"; do
    case $arg in
        --gpu) USE_GPU=true ;;
    esac
done

# Create conda environment
echo "Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Install PyTorch
if [ "$USE_GPU" = true ]; then
    echo "Installing PyTorch with CUDA 12.1 support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "Installing PyTorch (CPU/MPS)..."
    pip install torch torchvision
fi

# Install cesnet-datazoo and dependencies
echo "Installing cesnet-datazoo..."
pip install cesnet-datazoo

# Install other dependencies
echo "Installing remaining dependencies..."
pip install \
    numpy \
    pandas \
    scikit-learn \
    scipy \
    matplotlib \
    seaborn \
    tqdm \
    tensorboard \
    pyyaml \
    rich \
    jsonlines

echo ""
echo "=== Setup Complete ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo ""
echo "Next steps:"
echo "  1. bash scripts/download_data.sh"
echo "  2. bash scripts/run_all.sh"
