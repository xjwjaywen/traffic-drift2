#!/usr/bin/env bash
# Fixed nested allocations; source oracle and BADGE fits are never rerun.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
mode="${1:-run}"
if (($#)); then shift; fi
python_bin="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
if [[ -n "${GPU_ID:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU_ID"; fi
study="${STUDY_DIR:-outputs/kbs_supplement_v1}"
common=(--oracle-dir "${ORACLE_OUTPUT_DIR:-outputs/kbs_oracle_protection_v1}"
        --pilot-dir "${REPAIR_OUTPUT_DIR:-outputs/kbs_repair_aware_v1}"
        --study-dir "$study" --cache-dir "${CACHE_DIR:-$study/cache}"
        --checkpoint "${CHECKPOINT:-outputs/tls22_cnn/best_model.pt}"
        --output-dir "${PROTECTION_BUDGET_OUTPUT_DIR:-outputs/kbs_protection_budget_v1}"
        --seeds "${PROTECTION_BUDGET_SEEDS:-0,1,2}")
case "$mode" in
    run)
        "$python_bin" "$script_dir/kbs_protection_budget.py" select "${common[@]}" "$@"
        "$python_bin" "$script_dir/kbs_protection_budget.py" train "${common[@]}" "$@"
        "$python_bin" "$script_dir/kbs_protection_budget.py" evaluate "${common[@]}" "$@"
        ;;
    plan|preflight|select|train|evaluate)
        exec "$python_bin" "$script_dir/kbs_protection_budget.py" "$mode" "${common[@]}" "$@"
        ;;
    *) echo 'Usage: run_kbs_protection_budget.sh {plan|preflight|run|select|train|evaluate} [options]' >&2; exit 2 ;;
esac
