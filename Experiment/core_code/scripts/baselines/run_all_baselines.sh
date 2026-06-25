#!/bin/bash
# Run all 4 baseline methods for comparison with CARE.
# Each baseline uses the same: checkpoint, eval set, budget, strategy, seeds.
#
# Methods:
#   1. MEMENTO  — Full model + KD (no rectification)
#   2. ILETC   — GAN-generated replay features
#   3. CADE    — Contrastive drift detection + repair
#   4. Expanded Head — Wider classification head (capacity expansion)
#
# Output: outputs/baselines/{memento,iletc,cade,expanded_head}/seed_*/
# Usage: bash scripts/baselines/run_all_baselines.sh [--force]

set -euo pipefail
cd "$(dirname "$0")/../.."

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
    shift
fi

CONFIG="configs/eval_tls22.yaml"
CHECKPOINT="outputs/tls22_cnn/best_model.pt"
REF="M-2022-4"
TARGET="M-2022-12"
EVAL_COLLAPSE="56,163,174,48,38,69,104,47,66,10,109,26"
BUDGET=1000
STRATEGY="margin"
SEEDS=5
BASE="outputs/baselines"
SCRIPT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

run_method() {
    local METHOD="$1"
    local SCRIPT="$2"
    shift 2
    local EXTRA_ARGS=("$@")

    echo ""
    echo "=========================================="
    echo "  ${METHOD} (commit: ${SCRIPT_HASH})"
    echo "=========================================="

    for SEED in $(seq 0 $((SEEDS - 1))); do
        OUTDIR="${BASE}/${METHOD}/seed_${SEED}"
        if [ "${FORCE}" -eq 0 ] && [ -f "${OUTDIR}/summary.json" ]; then
            echo "  Seed ${SEED} exists, skipping (use --force to rerun)"
            continue
        fi
        if [ "${FORCE}" -eq 1 ] && [ -d "${OUTDIR}" ]; then
            echo "  Seed ${SEED}: removing old results (--force)"
            rm -rf "${OUTDIR}"
        fi
        echo "  Seed ${SEED}..."
        python "${SCRIPT}" \
            --config "${CONFIG}" \
            --checkpoint "${CHECKPOINT}" \
            --reference-period "${REF}" \
            --target-period "${TARGET}" \
            --eval-collapse-classes "${EVAL_COLLAPSE}" \
            --budget "${BUDGET}" \
            --strategy "${STRATEGY}" \
            --seed "${SEED}" \
            --output-dir "${OUTDIR}" \
            "${EXTRA_ARGS[@]}"
    done

    # Aggregate seeds
    if [ -d "${BASE}/${METHOD}" ]; then
        python scripts/aggregate_seeds.py \
            --base-dir "${BASE}/${METHOD}" \
            --num-seeds "${SEEDS}" \
            --require-complete
    fi
}

mkdir -p "${BASE}"

# 1. MEMENTO: full model + replay + KD
run_method "memento" "scripts/baselines/memento_baseline.py" \
    --ft-lr 1e-4 --ft-epochs 30 --replay-per-class 5 \
    --target-repeat 2 --distill-weight 0.5

# 2. ILETC: GAN replay (no KD)
run_method "iletc" "scripts/baselines/iletc_baseline.py" \
    --ft-lr 1e-3 --ft-epochs 30 --replay-per-class 5 \
    --target-repeat 2 --gan-epochs 50

# 3. CADE: contrastive detection + repair
run_method "cade" "scripts/baselines/cade_baseline.py" \
    --ft-lr 1e-3 --ft-epochs 30 --replay-per-class 5 \
    --target-repeat 2 --cae-epochs 100

# 4. Expanded Head: wider head with more capacity
run_method "expanded_head" "scripts/baselines/expanded_head_baseline.py" \
    --ft-lr 1e-3 --ft-epochs 30 --replay-per-class 5 \
    --target-repeat 2 --distill-weight 0.5 --hidden-dim 512

echo ""
echo "=========================================="
echo "  Summary (commit: ${SCRIPT_HASH})"
echo "=========================================="
for method in memento iletc cade expanded_head; do
    echo ""
    echo "  ${method}:"
    if [ -f "${BASE}/${method}/aggregated_mean_std.csv" ]; then
        head -1 "${BASE}/${method}/aggregated_mean_std.csv"
        grep -v "^method" "${BASE}/${method}/aggregated_mean_std.csv" | head -5
    elif [ -f "${BASE}/${method}/seed_0/summary.json" ]; then
        python -c "import json; d=json.load(open('${BASE}/${method}/seed_0/summary.json')); print(f'    seed_0: {d}')" 2>/dev/null || echo "    (parse error)"
    else
        echo "    not found"
    fi
done

echo ""
echo "Done. Results in ${BASE}/"
