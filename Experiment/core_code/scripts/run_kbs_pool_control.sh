#!/usr/bin/env bash
# Fix the original risk pool; select all seeds before a separate label-audit process.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
mode="${1:-run}"
if (($#)); then shift; fi
python_bin="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
if [[ -n "${GPU_ID:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU_ID"; fi
study="${STUDY_DIR:-outputs/kbs_supplement_v1}"
common=(--study-dir "$study" --cache-dir "${CACHE_DIR:-$study/cache}"
        --pilot-dir "${PILOT_DIR:-outputs/kbs_acquisition_pilot_v1}"
        --output-dir "${POOL_OUTPUT_DIR:-outputs/kbs_pool_control_v1}"
        --checkpoint "${CHECKPOINT:-outputs/tls22_cnn/best_model.pt}")
if [[ -n "${POOL_SEEDS:-}" ]]; then common+=(--seeds "$POOL_SEEDS"); fi
if [[ -n "${DEVICE:-}" ]]; then common+=(--device "$DEVICE"); fi
case "$mode" in
    run)
        "$python_bin" "$script_dir/kbs_pool_control.py" select "${common[@]}" "$@"
        "$python_bin" "$script_dir/kbs_pool_control.py" evaluate "${common[@]}" "$@"
        ;;
    plan|preflight|select|evaluate)
        exec "$python_bin" "$script_dir/kbs_pool_control.py" "$mode" "${common[@]}" "$@"
        ;;
    *) echo "Usage: bash scripts/run_kbs_pool_control.sh {plan|preflight|run|select|evaluate}" >&2; exit 2 ;;
esac
