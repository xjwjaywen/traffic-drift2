#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" -u "$SCRIPT_DIR/stage0.py" "${1:-all}" --data-root "${TMA_DATA_ROOT:-/data/xjw/tma-care-data}"
