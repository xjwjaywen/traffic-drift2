#!/bin/bash
# Generate all 7 ablation configurations for Table IV (paper)
# Strict eval, 5 seeds, same checkpoint and eval set.
#
# KD is defined over D_rpl (§3.5), so configs with KD always include replay
# as the substrate for distillation. The paper table's "Rpl" column denotes
# whether all-class CE replay is the PRIMARY mechanism; KD-only and FT+KD
# mark Rpl=✗ because replay's role there is as input to distillation, not
# a standalone regularizer. See table caption footnote.
#
# Configs and their actual parameters:
#   1. Static (no repair) — baseline from eval
#   2. Replay only (B=0, all-class k=5, distill=0.0) — replay CE only
#   3. KD only (B=0, all-class k=5, distill=0.5) — replay CE + KD
#   4. FT only (B=1000, no replay, distill=0.0) — labels only
#   5. FT+KD (B=1000, stable_absorber k=5, distill=0.5) — labels + targeted replay as KD substrate
#   6. FT+Replay (B=1000, all-class k=5, distill=0.0) — labels + all-class replay CE
#   7. Full CARE (B=1000, all-class k=5, distill=0.5) — all components
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

# Without target labels (budget=0): both use all-class replay (890 samples)
# KD-only adds distillation on top of replay CE — "Rpl" column is ✗ because
# the test isolates KD's contribution, not replay's.
run_config "Replay only"  "${BASE}/replay_only"  "margin" "0" "all" "5" "0.0"
run_config "KD only"      "${BASE}/kd_only"      "margin" "0" "all" "5" "0.5"

# With target labels (budget=1000):
# FT+KD uses targeted (stable+absorber) replay as KD substrate (150 samples).
# FT+Rpl and Full CARE use all-class replay (890 samples).
run_config "FT only"      "${BASE}/ft_only"      "margin" "1000" "none"             "0" "0.0"
run_config "FT+KD"        "${BASE}/ft_kd"        "margin" "1000" "stable_absorber"  "5" "0.5"
run_config "FT+Replay"    "${BASE}/ft_replay"    "margin" "1000" "all"              "5" "0.0"
run_config "Full CARE"    "${BASE}/full_care"     "margin" "1000" "all"              "5" "0.5"

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
