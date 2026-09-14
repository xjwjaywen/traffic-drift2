#!/usr/bin/env bash
# Run from repository root or Experiment/core_code. Existing conda environment is used.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."

mode="${1:-primary}"
if (($#)); then shift; fi
python_bin="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
if [[ -n "${GPU_ID:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU_ID"; fi

output_dir="${OUTPUT_DIR:-outputs/kbs_supplement_v1}"
cache_dir="${CACHE_DIR:-$output_dir/cache}"
common=(
    --config "${CONFIG:-configs/eval_tls22.yaml}"
    --checkpoint "${CHECKPOINT:-outputs/tls22_cnn/best_model.pt}"
    --cache-dir "$cache_dir"
    --output-dir "$output_dir"
    --device "${DEVICE:-cuda}"
)
if [[ -n "${DATA_DIR:-}" ]]; then common+=(--data-dir "$DATA_DIR"); fi
if [[ -n "${SEEDS:-}" ]]; then common+=(--seeds "$SEEDS"); fi
if [[ -n "${STEPS:-}" ]]; then common+=(--steps "$STEPS"); fi

case "$mode" in
    smoke)
        exec "$python_bin" -m unittest discover -s "$script_dir/tests" -p 'test_kbs*.py'
        ;;
    badge-kd-plan|badge-kd-summarize)
        exec "$python_bin" "$script_dir/kbs_badge_kd_followup.py" "${mode#badge-kd-}" "${common[@]}" "$@"
        ;;
    badge-kd)
        mkdir -p "$output_dir"
        log="$output_dir/launcher-${mode}-$(date +%Y%m%d-%H%M%S)-$$.log"
        "$python_bin" "$script_dir/kbs_badge_kd_followup.py" run "${common[@]}" "$@" 2>&1 | tee -a "$log"
        ;;
    plan|summarize)
        exec "$python_bin" "$script_dir/kbs_supplement.py" "$mode" --suite "${SUITE:-primary}" "${common[@]}" "$@"
        ;;
    preflight|prepare)
        exec "$python_bin" "$script_dir/kbs_supplement.py" "$mode" "${common[@]}" "$@"
        ;;
    primary|sensitivity)
        mkdir -p "$output_dir"
        log="$output_dir/launcher-${mode}-$(date +%Y%m%d-%H%M%S)-$$.log"
        "$python_bin" "$script_dir/kbs_supplement.py" prepare --suite "$mode" "${common[@]}" "$@" 2>&1 | tee -a "$log"
        "$python_bin" "$script_dir/kbs_supplement.py" run --suite "$mode" "${common[@]}" "$@" 2>&1 | tee -a "$log"
        ;;
    *)
        echo "Usage: bash scripts/run_kbs_supplement.sh {preflight|smoke|plan|prepare|primary|sensitivity|summarize|badge-kd|badge-kd-plan|badge-kd-summarize} [Python CLI options]" >&2
        exit 2
        ;;
esac
