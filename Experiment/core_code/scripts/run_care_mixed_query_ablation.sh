#!/usr/bin/env bash
# Foreground CPU stages; queries are frozen before fitting, then independently evaluated.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
mode="${1:-run}"
if (($#)); then shift; fi
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${CARE_MIXED_THREADS:-8}"
export MKL_NUM_THREADS="${CARE_MIXED_THREADS:-8}"
python_bin="${PYTHON:-python}"
common=(--source-dir "${CARE_MIXED_SOURCE_DIR:-outputs/small_repair_diagnostic_v1}"
    --source-audit-dir "${CARE_MIXED_AUDIT_DIR:-outputs/small_repair_full_audit_v1}"
    --protection-dir "${CARE_MIXED_PROTECTION_DIR:-outputs/care_target_protection_v1}"
    --cache-dir "${CARE_MIXED_CACHE_DIR:-outputs/kbs_supplement_v1/cache}"
    --checkpoint "${CARE_MIXED_CHECKPOINT:-outputs/tls22_cnn/best_model.pt}"
    --output-dir "${CARE_MIXED_OUTPUT_DIR:-outputs/care_mixed_query_ablation_v1}"
    --threads "${CARE_MIXED_THREADS:-8}")
case "$mode" in
    run)
        "$python_bin" "$script_dir/care_mixed_query_ablation.py" select "${common[@]}" "$@"
        "$python_bin" "$script_dir/care_mixed_query_ablation.py" train "${common[@]}" "$@"
        "$python_bin" "$script_dir/care_mixed_query_ablation.py" evaluate "${common[@]}" "$@"
        ;;
    plan|preflight|select|train|evaluate)
        exec "$python_bin" "$script_dir/care_mixed_query_ablation.py" "$mode" "${common[@]}" "$@"
        ;;
    *) echo 'Usage: run_care_mixed_query_ablation.sh {plan|preflight|run|select|train|evaluate} [options]' >&2; exit 2 ;;
esac
