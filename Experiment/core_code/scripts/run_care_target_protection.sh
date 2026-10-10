#!/usr/bin/env bash
# CPU foreground only. Freeze queries, fit queried labels, independently evaluate.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
mode="${1:-run}"
if (($#)); then shift; fi
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${CARE_PROTECT_THREADS:-8}"
export MKL_NUM_THREADS="${CARE_PROTECT_THREADS:-8}"
python_bin="${PYTHON:-python}"
common=(--source-dir "${CARE_PROTECT_SOURCE_DIR:-outputs/small_repair_diagnostic_v1}"
    --source-audit-dir "${CARE_PROTECT_AUDIT_DIR:-outputs/small_repair_full_audit_v1}"
    --cache-dir "${CARE_PROTECT_CACHE_DIR:-outputs/kbs_supplement_v1/cache}"
    --checkpoint "${CARE_PROTECT_CHECKPOINT:-outputs/tls22_cnn/best_model.pt}"
    --output-dir "${CARE_PROTECT_OUTPUT_DIR:-outputs/care_target_protection_v1}"
    --threads "${CARE_PROTECT_THREADS:-8}")
case "$mode" in
    run)
        "$python_bin" "$script_dir/care_target_protection.py" select "${common[@]}" "$@"
        "$python_bin" "$script_dir/care_target_protection.py" train "${common[@]}" "$@"
        "$python_bin" "$script_dir/care_target_protection.py" evaluate "${common[@]}" "$@"
        ;;
    plan|preflight|select|train|evaluate)
        exec "$python_bin" "$script_dir/care_target_protection.py" "$mode" "${common[@]}" "$@"
        ;;
    *) echo 'Usage: run_care_target_protection.sh {plan|preflight|run|select|train|evaluate} [options]' >&2; exit 2 ;;
esac
