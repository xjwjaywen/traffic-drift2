#!/usr/bin/env bash
# Three separate processes: scout-only acquisition, queried-only repair, offline evaluation.
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
        --checkpoint "${CHECKPOINT:-outputs/tls22_cnn/best_model.pt}"
        --output-dir "${REPAIR_OUTPUT_DIR:-outputs/kbs_repair_aware_v1}"
        --seeds "${REPAIR_SEEDS:-0,1,2,3,4}" --device "${DEVICE:-cuda}")
case "$mode" in
    run)
        "$python_bin" "$script_dir/kbs_repair_aware.py" select "${common[@]}" "$@"
        "$python_bin" "$script_dir/kbs_repair_aware.py" train "${common[@]}" "$@"
        "$python_bin" "$script_dir/kbs_repair_aware.py" evaluate "${common[@]}" "$@"
        ;;
    plan|preflight|select|train|evaluate)
        exec "$python_bin" "$script_dir/kbs_repair_aware.py" "$mode" "${common[@]}" "$@"
        ;;
    *) echo 'Usage: run_kbs_repair_aware.sh {plan|preflight|run|select|train|evaluate} [options]' >&2; exit 2 ;;
esac
