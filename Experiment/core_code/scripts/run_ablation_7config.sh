#!/bin/bash
# Generate all 7 ablation configurations for Table IV (paper)
# All use margin@1000, strict eval, 5 seeds, same checkpoint and eval set.
#
# Configs:
#   1. Static (no repair)
#   2. Replay only (no labels, all-class replay k=5 + no FT/KD)
#   3. KD only (no labels, distillation only)
#   4. FT only (labels, head FT, no replay, no KD)
#   5. FT+KD (labels, head FT + KD, no replay)
#   6. FT+Replay (labels, head FT + all-class replay, no KD)
#   7. Full CARE (labels, head FT + all-class replay + KD)
#
# Output: outputs/ablation_7config_canonical/
# Usage: bash scripts/run_ablation_7config.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/eval_tls22.yaml"
CHECKPOINT="outputs/tls22_cnn/best_model.pt"
REF="M-2022-4"
EVAL_COLLAPSE="56,163,174,48,38,69,104,47,66,10,109,26"
SEEDS=5
BASE="outputs/ablation_7config_canonical"

run_config() {
    local LABEL="$1"
    local OUTDIR="$2"
    local STRATEGY="$3"
    local BUDGET="$4"
    local REPLAY_MODE="$5"
    local REPLAY_K="$6"
    local DISTILL_W="$7"

    echo "=== ${LABEL} ==="
    for SEED in $(seq 0 $((SEEDS - 1))); do
        if [ -f "${OUTDIR}/seed_${SEED}/results_by_budget.csv" ]; then
            echo "  Seed ${SEED} exists, skipping"
            continue
        fi
        echo "  Seed ${SEED}"
        python scripts/collapse_active_maintenance_tls22.py \
            --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
            --reference-period "${REF}" --target-period M-2022-12 \
            --strategies "${STRATEGY}" --budgets "${BUDGET}" \
            --eval-collapse-classes "${EVAL_COLLAPSE}" \
            --replay-mode "${REPLAY_MODE}" --replay-per-class "${REPLAY_K}" \
            --target-repeat 2 \
            --replay-distill-weight "${DISTILL_W}" \
            --distill-temperature 2.0 \
            --seed "${SEED}" \
            --output-dir "${OUTDIR}/seed_${SEED}"
    done
    python scripts/aggregate_seeds.py --base-dir "${OUTDIR}" --num-seeds "${SEEDS}"
}

mkdir -p "${BASE}"

# Without target labels (budget=0)
run_config "Replay only"  "${BASE}/replay_only"  "margin" "0" "all" "5" "0.0"
run_config "KD only"      "${BASE}/kd_only"      "margin" "0" "all" "0" "0.5"

# With target labels (budget=1000)
run_config "FT only"      "${BASE}/ft_only"      "margin" "1000" "none" "0" "0.0"
run_config "FT+KD"        "${BASE}/ft_kd"        "margin" "1000" "none" "0" "0.5"
run_config "FT+Replay"    "${BASE}/ft_replay"    "margin" "1000" "all"  "5" "0.0"
run_config "Full CARE"    "${BASE}/full_care"     "margin" "1000" "all"  "5" "0.5"

echo ""
echo "=== Summary ==="
for d in replay_only kd_only ft_only ft_kd ft_replay full_care; do
    echo "  ${d}:"
    if [ -f "${BASE}/${d}/aggregated_mean_std.csv" ]; then
        grep "margin" "${BASE}/${d}/aggregated_mean_std.csv" | \
            awk -F, '{printf "    macro=%.4f±%.4f collapse=%.4f±%.4f stable=%.4f±%.4f\n",$5,$6,$7,$8,$9,$10}' || echo "    parse error"
    else
        echo "    not found"
    fi
done
echo ""
echo "Done. Results in ${BASE}/"
