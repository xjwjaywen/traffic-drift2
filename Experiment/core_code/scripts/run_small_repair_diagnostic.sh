#!/usr/bin/env bash
# CPU only. Freeze all queries -> queried-label fitting -> separate offline evaluation.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
mode="${1:-run}"
if (($#)); then shift; fi
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SMALL_REPAIR_THREADS:-8}"
export MKL_NUM_THREADS="${SMALL_REPAIR_THREADS:-8}"
python_bin="${PYTHON:-python}"
common=(--cache-dir "${SMALL_REPAIR_CACHE_DIR:-outputs/kbs_supplement_v1/cache}"
        --checkpoint "${SMALL_REPAIR_CHECKPOINT:-outputs/tls22_cnn/best_model.pt}"
        --output-dir "${SMALL_REPAIR_OUTPUT_DIR:-outputs/small_repair_diagnostic_v1}"
        --seeds "${SMALL_REPAIR_SEEDS:-0,1,2}" --threads "${SMALL_REPAIR_THREADS:-8}")
case "$mode" in
    run)
        "$python_bin" "$script_dir/small_repair_diagnostic.py" select "${common[@]}" "$@"
        "$python_bin" "$script_dir/small_repair_diagnostic.py" train "${common[@]}" "$@"
        "$python_bin" "$script_dir/small_repair_diagnostic.py" evaluate "${common[@]}" "$@"
        ;;
    plan|preflight|select|train|evaluate)
        exec "$python_bin" "$script_dir/small_repair_diagnostic.py" "$mode" "${common[@]}" "$@"
        ;;
    *) echo 'Usage: run_small_repair_diagnostic.sh {plan|preflight|run|select|train|evaluate} [options]' >&2; exit 2 ;;
esac
