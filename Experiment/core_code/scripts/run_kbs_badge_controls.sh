#!/usr/bin/env bash
# Existing controlled study and conda environment; no new feature extraction.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
mode="${1:-run}"
if (($#)); then shift; fi
case "$mode" in
    plan|preflight|run|summarize) ;;
    *) echo 'Usage: run_kbs_badge_controls.sh {plan|preflight|run|summarize} [Python CLI options]' >&2; exit 2 ;;
esac
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
if [[ -n "${GPU_ID:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU_ID"; fi
output_dir="${OUTPUT_DIR:-outputs/kbs_supplement_v1}"
common=(--config "${CONFIG:-configs/eval_tls22.yaml}"
        --checkpoint "${CHECKPOINT:-outputs/tls22_cnn/best_model.pt}"
        --cache-dir "${CACHE_DIR:-$output_dir/cache}"
        --output-dir "$output_dir" --device "${DEVICE:-cuda}")
# Do not inherit earlier SEEDS=0,1,2 or STEPS from other experiment launchers.
if [[ -n "${BADGE_CONTROL_SEEDS:-}" ]]; then common+=(--seeds "$BADGE_CONTROL_SEEDS"); fi
if [[ -n "${DATA_DIR:-}" ]]; then common+=(--data-dir "$DATA_DIR"); fi
exec "${PYTHON:-python}" "$script_dir/kbs_badge_controls.py" "$mode" "${common[@]}" "$@"
