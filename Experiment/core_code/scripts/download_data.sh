#!/bin/bash
# ============================================================
# Download CESNET datasets via cesnet-datazoo
# ============================================================
# Usage: bash scripts/download_data.sh [--size S|M|L] [--data-dir /path]
# ============================================================
set -e

SIZE="S"
DATA_DIR="./data"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --size)
            SIZE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            shift
            ;;
    esac
done

mkdir -p "${DATA_DIR}"

echo "=== Downloading CESNET Datasets ==="
echo "Size: ${SIZE}, Directory: ${DATA_DIR}"

python - <<PYEOF
import sys

data_dir = "${DATA_DIR}"
size = "${SIZE}"

print("--- Downloading CESNET-QUIC22 ---")
try:
    from cesnet_datazoo.datasets import CESNET_QUIC22
    dataset = CESNET_QUIC22(data_dir, size=size)
    print(f"CESNET-QUIC22 (size={size}) ready at {data_dir}")
except Exception as e:
    print(f"CESNET-QUIC22 download error: {e}")
    print("You may need to download manually from https://zenodo.org/")

print()
print("--- Downloading CESNET-TLS-Year22 ---")
try:
    try:
        from cesnet_datazoo.datasets import CESNET_TLS_Year22
        dataset = CESNET_TLS_Year22(data_dir, size=size)
    except ImportError:
        from cesnet_datazoo.datasets import CESNET_TLS22
        dataset = CESNET_TLS22(data_dir, size=size)
    print(f"CESNET-TLS-Year22 (size={size}) ready at {data_dir}")
except Exception as e:
    print(f"CESNET-TLS-Year22 download error: {e}")
    print("You may need to download manually from https://zenodo.org/")

print()
print("=== Download Complete ===")
PYEOF
