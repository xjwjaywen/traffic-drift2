#!/usr/bin/env bash
# Read-only CPU diagnosis of completed learned and oracle-budget studies.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
mode="${1:-run}"
if (($#)); then shift; fi
case "$mode" in
    plan|preflight|run) ;;
    *) echo 'Usage: run_kbs_protection_gap_diagnosis.sh {plan|preflight|run} [options]' >&2; exit 2 ;;
esac
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${DIAG_THREADS:-8}"
export MKL_NUM_THREADS="${DIAG_THREADS:-8}"
study="${STUDY_DIR:-outputs/kbs_supplement_v1}"
exec "${PYTHON:-python}" "$script_dir/kbs_protection_gap_diagnosis.py" "$mode" \
    --learned-dir "${LEARNED_PROTECTION_OUTPUT_DIR:-outputs/kbs_learned_protection_v1}" \
    --allocation-dir "${PROTECTION_BUDGET_OUTPUT_DIR:-outputs/kbs_protection_budget_v1}" \
    --oracle-dir "${ORACLE_OUTPUT_DIR:-outputs/kbs_oracle_protection_v1}" \
    --pilot-dir "${REPAIR_OUTPUT_DIR:-outputs/kbs_repair_aware_v1}" \
    --study-dir "$study" --cache-dir "${CACHE_DIR:-$study/cache}" \
    --checkpoint "${CHECKPOINT:-outputs/tls22_cnn/best_model.pt}" \
    --output-dir "${PROTECTION_GAP_OUTPUT_DIR:-outputs/kbs_protection_gap_diagnosis_v1}" \
    --seeds "${PROTECTION_GAP_SEEDS:-0,1,2}" --threads "${DIAG_THREADS:-8}" "$@"
