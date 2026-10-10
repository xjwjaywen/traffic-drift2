#!/usr/bin/env bash
# Foreground CPU diagnosis of completed M11 heads; never fit or acquire labels.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
mode="${1:-run}"
if (($#)); then shift; fi
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${CARE_M11_DIAG_THREADS:-8}"
export MKL_NUM_THREADS="${CARE_M11_DIAG_THREADS:-8}"
python_bin="${PYTHON:-python}"
common=(--source-dir "${CARE_M11_DIAG_SOURCE_DIR:-outputs/small_repair_diagnostic_v1}"
    --source-audit-dir "${CARE_M11_DIAG_AUDIT_DIR:-outputs/small_repair_full_audit_v1}"
    --protection-dir "${CARE_M11_DIAG_PROTECTION_DIR:-outputs/care_target_protection_v1}"
    --ablation-dir "${CARE_M11_DIAG_ABLATION_DIR:-outputs/care_mixed_query_ablation_v1}"
    --replication-dir "${CARE_M11_DIAG_REPLICATION_DIR:-outputs/care_query_seed_robustness_v1}"
    --source-cache-dir "${CARE_M11_DIAG_SOURCE_CACHE_DIR:-outputs/kbs_supplement_v1/cache}"
    --cache-dir "${CARE_M11_DIAG_CACHE_DIR:-outputs/care_m11_cache_v1}"
    --checkpoint "${CARE_M11_DIAG_CHECKPOINT:-outputs/tls22_cnn/best_model.pt}"
    --month-dir "${CARE_M11_DIAG_MONTH_DIR:-outputs/care_m11_replication_v1}"
    --output-dir "${CARE_M11_DIAG_OUTPUT_DIR:-outputs/care_m11_mechanism_diagnostic_v1}"
    --threads "${CARE_M11_DIAG_THREADS:-8}")
if [[ -n "${CARE_M11_DIAG_DATA_DIR:-}" ]]; then common+=(--data-dir "$CARE_M11_DIAG_DATA_DIR"); fi
case "$mode" in
    run)
        for stage in audit geometry report; do
            "$python_bin" "$script_dir/care_m11_mechanism_diagnostic.py" "$stage" "${common[@]}" "$@"
        done
        ;;
    plan|preflight|audit|geometry|report)
        exec "$python_bin" "$script_dir/care_m11_mechanism_diagnostic.py" "$mode" "${common[@]}" "$@"
        ;;
    *) echo 'Usage: run_care_m11_mechanism_diagnostic.sh {plan|preflight|run|audit|geometry|report} [options]' >&2; exit 2 ;;
esac
