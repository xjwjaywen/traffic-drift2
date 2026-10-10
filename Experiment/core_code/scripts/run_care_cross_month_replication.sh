#!/usr/bin/env bash
# Foreground-only CPU feature extraction and M11 replication.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
mode="${1:-run}"
if (($#)); then shift; fi
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${CARE_M11_THREADS:-8}"
export MKL_NUM_THREADS="${CARE_M11_THREADS:-8}"
python_bin="${PYTHON:-python}"
common=(--source-dir "${CARE_M11_SOURCE_DIR:-outputs/small_repair_diagnostic_v1}"
    --source-audit-dir "${CARE_M11_AUDIT_DIR:-outputs/small_repair_full_audit_v1}"
    --protection-dir "${CARE_M11_PROTECTION_DIR:-outputs/care_target_protection_v1}"
    --ablation-dir "${CARE_M11_ABLATION_DIR:-outputs/care_mixed_query_ablation_v1}"
    --replication-dir "${CARE_M11_REPLICATION_DIR:-outputs/care_query_seed_robustness_v1}"
    --source-cache-dir "${CARE_M11_SOURCE_CACHE_DIR:-outputs/kbs_supplement_v1/cache}"
    --checkpoint "${CARE_M11_CHECKPOINT:-outputs/tls22_cnn/best_model.pt}"
    --cache-dir "${CARE_M11_CACHE_DIR:-outputs/care_m11_cache_v1}"
    --output-dir "${CARE_M11_OUTPUT_DIR:-outputs/care_m11_replication_v1}"
    --threads "${CARE_M11_THREADS:-8}")
if [[ -n "${CARE_M11_DATA_DIR:-}" ]]; then common+=(--data-dir "$CARE_M11_DATA_DIR"); fi
case "$mode" in
    run)
        for stage in prepare select train evaluate; do
            "$python_bin" "$script_dir/care_cross_month_replication.py" "$stage" "${common[@]}" "$@"
        done
        ;;
    plan|preflight|prepare|select|train|evaluate)
        exec "$python_bin" "$script_dir/care_cross_month_replication.py" "$mode" "${common[@]}" "$@"
        ;;
    *) echo 'Usage: run_care_cross_month_replication.sh {plan|preflight|run|prepare|select|train|evaluate} [options]' >&2; exit 2 ;;
esac
