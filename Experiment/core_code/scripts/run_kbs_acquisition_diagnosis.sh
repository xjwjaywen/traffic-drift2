#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
mode="${1:-run}"
if (($#)); then shift; fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
if [[ -n "${GPU_ID:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU_ID"; fi
study="${STUDY_DIR:-outputs/kbs_supplement_v1}"
common=(--study-dir "$study" --cache-dir "${CACHE_DIR:-$study/cache}"
        --pilot-dir "${PILOT_DIR:-outputs/kbs_acquisition_pilot_v1}"
        --output-dir "${DIAG_OUTPUT_DIR:-outputs/kbs_acquisition_diagnosis_v1}"
        --checkpoint "${CHECKPOINT:-outputs/tls22_cnn/best_model.pt}" --device "${DEVICE:-cuda}")
if [[ -n "${DIAG_SEEDS:-}" ]]; then common+=(--seeds "$DIAG_SEEDS"); fi
exec "${PYTHON:-python}" "$script_dir/kbs_acquisition_diagnosis.py" "$mode" "${common[@]}" "$@"
