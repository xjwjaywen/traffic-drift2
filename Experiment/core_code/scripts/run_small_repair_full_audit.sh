#!/usr/bin/env bash
# Foreground CPU evaluation only; no training or new acquisition.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
mode="${1:-run}"
if (($#)); then shift; fi
case "$mode" in
    run|preflight) ;;
    *) echo 'Usage: run_small_repair_full_audit.sh {preflight|run} [options]' >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SMALL_FULL_THREADS:-8}"
export MKL_NUM_THREADS="${SMALL_FULL_THREADS:-8}"
python_bin="${PYTHON:-python}"
exec "$python_bin" "$script_dir/small_repair_full_audit.py" "$mode" \
    --source-dir "${SMALL_FULL_SOURCE_DIR:-outputs/small_repair_diagnostic_v1}" \
    --cache-dir "${SMALL_FULL_CACHE_DIR:-outputs/kbs_supplement_v1/cache}" \
    --checkpoint "${SMALL_FULL_CHECKPOINT:-outputs/tls22_cnn/best_model.pt}" \
    --output-dir "${SMALL_FULL_OUTPUT_DIR:-outputs/small_repair_full_audit_v1}" \
    --threads "${SMALL_FULL_THREADS:-8}" "$@"
