#!/usr/bin/env bash
# Foreground CPU replication; inventory is metadata-only, not cache validation.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
mode="${1:-run}"
if (($#)); then shift; fi
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${CARE_QUERY_THREADS:-8}"
export MKL_NUM_THREADS="${CARE_QUERY_THREADS:-8}"
python_bin="${PYTHON:-python}"
common=(--source-dir "${CARE_QUERY_SOURCE_DIR:-outputs/small_repair_diagnostic_v1}"
    --source-audit-dir "${CARE_QUERY_AUDIT_DIR:-outputs/small_repair_full_audit_v1}"
    --protection-dir "${CARE_QUERY_PROTECTION_DIR:-outputs/care_target_protection_v1}"
    --ablation-dir "${CARE_QUERY_ABLATION_DIR:-outputs/care_mixed_query_ablation_v1}"
    --cache-dir "${CARE_QUERY_CACHE_DIR:-outputs/kbs_supplement_v1/cache}"
    --checkpoint "${CARE_QUERY_CHECKPOINT:-outputs/tls22_cnn/best_model.pt}"
    --output-dir "${CARE_QUERY_OUTPUT_DIR:-outputs/care_query_seed_robustness_v1}"
    --scan-root "${CARE_QUERY_SCAN_ROOT:-outputs}"
    --threads "${CARE_QUERY_THREADS:-8}")
case "$mode" in
    run)
        "$python_bin" "$script_dir/care_query_seed_robustness.py" inventory "${common[@]}" "$@"
        "$python_bin" "$script_dir/care_query_seed_robustness.py" select "${common[@]}" "$@"
        "$python_bin" "$script_dir/care_query_seed_robustness.py" train "${common[@]}" "$@"
        "$python_bin" "$script_dir/care_query_seed_robustness.py" evaluate "${common[@]}" "$@"
        ;;
    plan|inventory|preflight|select|train|evaluate)
        exec "$python_bin" "$script_dir/care_query_seed_robustness.py" "$mode" "${common[@]}" "$@"
        ;;
    *) echo 'Usage: run_care_query_seed_robustness.sh {plan|inventory|preflight|run|select|train|evaluate} [options]' >&2; exit 2 ;;
esac
