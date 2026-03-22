#!/bin/bash
# ============================================================
# Download CESNET datasets via cesnet-datazoo
# ============================================================
# Usage: bash scripts/download_data.sh [--size S|M|L] [--data-dir /path]
# ============================================================
set -e

SIZE="S"
DATA_DIR="./data"

for arg in "$@"; do
    case $arg in
        --size) shift; SIZE="$1" ;;
        --data-dir) shift; DATA_DIR="$1" ;;
    esac
    shift 2>/dev/null || true
done

mkdir -p "${DATA_DIR}"

echo "=== Downloading CESNET Datasets ==="
echo "Size: ${SIZE}, Directory: ${DATA_DIR}"

python -c "
from cesnet_datazoo.datasets import CESNET_QUIC22, CESNET_TLS22
import os

data_dir = '${DATA_DIR}'
size = '${SIZE}'

print('--- Downloading CESNET-QUIC22 ---')
try:
    dataset = CESNET_QUIC22(data_dir, size=size)
    print(f'CESNET-QUIC22 (size={size}) ready at {data_dir}')
except Exception as e:
    print(f'CESNET-QUIC22 download error: {e}')
    print('You may need to download manually from https://zenodo.org/')

print()
print('--- Downloading CESNET-TLS-Year22 ---')
try:
    dataset = CESNET_TLS22(data_dir, size=size)
    print(f'CESNET-TLS-Year22 (size={size}) ready at {data_dir}')
except Exception as e:
    print(f'CESNET-TLS-Year22 download error: {e}')
    print('You may need to download manually from https://zenodo.org/')

print()
print('=== Download Complete ===')
"
