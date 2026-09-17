#!/usr/bin/env bash
# Oracle selection, fixed-query training, common evaluation in separate processes.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
mode="${1:-run}"
if (($#)); then shift; fi
python_bin="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
if [[ -n "${GPU_ID:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU_ID"; fi
study="${STUDY_DIR:-outputs/kbs_supplement_v1}"
common=(--pilot-dir "${REPAIR_OUTPUT_DIR:-outputs/kbs_repair_aware_v1}"
        --study-dir "$study" --cache-dir "${CACHE_DIR:-$study/cache}"
        --checkpoint "${CHECKPOINT:-outputs/tls22_cnn/best_model.pt}"
        --output-dir "${ORACLE_OUTPUT_DIR:-outputs/kbs_oracle_protection_v1}"
        --seeds "${ORACLE_SEEDS:-0,1,2}")
case "$mode" in
    run)
        "$python_bin" "$script_dir/kbs_oracle_protection.py" select "${common[@]}" "$@"
        "$python_bin" "$script_dir/kbs_oracle_protection.py" train "${common[@]}" "$@"
        "$python_bin" "$script_dir/kbs_oracle_protection.py" evaluate "${common[@]}" "$@"
        ;;
    plan|preflight|select|train|evaluate)
        exec "$python_bin" "$script_dir/kbs_oracle_protection.py" "$mode" "${common[@]}" "$@"
        ;;
    *) echo 'Usage: run_kbs_oracle_protection.sh {plan|preflight|run|select|train|evaluate} [options]' >&2; exit 2 ;;
esac
