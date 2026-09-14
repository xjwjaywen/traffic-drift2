#!/usr/bin/env bash
# Freeze query-confirmed relations for all seeds before a separate audit process.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
mode="${1:-run}"
if (($#)); then shift; fi
python_bin="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
common=(--study-dir "${STUDY_DIR:-outputs/kbs_supplement_v1}"
        --output-dir "${RELATION_OUTPUT_DIR:-outputs/kbs_relation_audit_v1}")
if [[ -n "${RELATION_SEEDS:-}" ]]; then common+=(--seeds "$RELATION_SEEDS"); fi
case "$mode" in
    run)
        "$python_bin" "$script_dir/kbs_relation_audit.py" freeze "${common[@]}" "$@"
        "$python_bin" "$script_dir/kbs_relation_audit.py" evaluate "${common[@]}" "$@"
        ;;
    plan|preflight|freeze|evaluate)
        exec "$python_bin" "$script_dir/kbs_relation_audit.py" "$mode" "${common[@]}" "$@"
        ;;
    *) echo "Usage: bash scripts/run_kbs_relation_audit.sh {plan|preflight|run|freeze|evaluate}" >&2; exit 2 ;;
esac
