#!/bin/bash
# V3 unified experiments: all rows use the SAME repair pipeline
# All-class replay k=5 (890 samples), target_repeat=2, strict eval, 5 seeds
# Only ONE variable changes per comparison
#
# Usage: bash scripts/run_v3_unified_experiments.sh [u1|u2|u3|all]
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/eval_tls22.yaml"
CHECKPOINT="outputs/tls22_cnn/best_model.pt"
REF="M-2022-4"
EVAL_COLLAPSE="56,163,174,48,38,69,104,47,66,10,109,26"
SEEDS=5

# Common CARE args (canonical v3 config)
CARE_ARGS="--replay-mode all --replay-per-class 5 --target-repeat 2"

run_care() {
    local LABEL="$1"
    local OUT="$2"
    local STRATEGY="$3"
    local BUDGET="$4"
    local DISTILL_W="$5"
    local EXTRA="${6:-}"

    echo "--- ${LABEL} ---"
    for SEED in $(seq 0 $((SEEDS - 1))); do
        echo "  Seed ${SEED}"
        python scripts/collapse_active_maintenance_tls22.py \
            --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
            --reference-period "${REF}" --target-period M-2022-12 \
            --strategies "${STRATEGY}" --budgets "${BUDGET}" \
            --eval-collapse-classes "${EVAL_COLLAPSE}" \
            ${CARE_ARGS} \
            --replay-distill-weight "${DISTILL_W}" \
            --distill-temperature 2.0 \
            --seed "${SEED}" \
            ${EXTRA} \
            --output-dir "${OUT}/seed_${SEED}"
    done
    python scripts/aggregate_seeds.py --base-dir "${OUT}" --num-seeds "${SEEDS}"
    echo "  Result:"
    grep "${STRATEGY}.*${BUDGET}" "${OUT}/aggregated_mean_std.csv" || true
}

# ============================================================
# U1: Clean ablation — all use margin@1000, all-class replay k=5
#     Only distillation weight varies
# ============================================================
run_u1() {
    echo "================================================================"
    echo "U1: Clean ablation (margin@1000, all-class k=5, 5 seeds)"
    echo "================================================================"
    local BASE="outputs/unified_ablation"

    # U1a: FT + all-class replay, NO distillation
    run_care "FT+Replay (no KD)" "${BASE}/ft_replay_noKD" "margin" "1000" "0.0"

    # U1b: FT + all-class replay + KD = Full CARE (should match P1)
    run_care "Full CARE (FT+Replay+KD)" "${BASE}/full_care" "margin" "1000" "0.5"

    # U1c: FT only — no replay, no KD (override CARE_ARGS replay settings)
    echo "--- FT only (no replay) ---"
    for SEED in $(seq 0 $((SEEDS - 1))); do
        echo "  Seed ${SEED}"
        python scripts/collapse_active_maintenance_tls22.py \
            --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
            --reference-period "${REF}" --target-period M-2022-12 \
            --strategies "margin" --budgets "1000" \
            --eval-collapse-classes "${EVAL_COLLAPSE}" \
            --replay-mode none --replay-per-class 0 --target-repeat 1 \
            --replay-distill-weight 0.0 \
            --seed "${SEED}" \
            --output-dir "${BASE}/ft_only/seed_${SEED}"
    done
    python scripts/aggregate_seeds.py --base-dir "${BASE}/ft_only" --num-seeds "${SEEDS}"
    echo "  Result:"
    grep "margin.*1000" "${BASE}/ft_only/aggregated_mean_std.csv" || true

    echo ""
    echo "=== U1 Summary ==="
    for d in ft_only ft_replay_noKD full_care; do
        echo "  $d:"
        grep "margin.*1000" "${BASE}/$d/aggregated_mean_std.csv" 2>/dev/null | \
            awk -F, '{printf "    macro=%.4f±%.4f collapse=%.4f±%.4f stable=%.4f±%.4f\n",$5,$6,$7,$8,$9,$10}' || echo "    not found"
    done
}

# ============================================================
# U2: AL baselines — all use all-class replay + KD (same CARE pipeline)
#     Only selection strategy varies
# ============================================================
run_u2() {
    echo "================================================================"
    echo "U2: AL baselines with unified pipeline (5 seeds)"
    echo "================================================================"
    local BASE="outputs/unified_al_baselines"

    for STRATEGY in random entropy coreset margin; do
        run_care "${STRATEGY}" "${BASE}/${STRATEGY}" "${STRATEGY}" "1000" "0.5"
    done

    echo ""
    echo "=== U2 Summary ==="
    for s in entropy coreset random margin; do
        echo "  $s:"
        grep "${s}.*1000" "${BASE}/${s}/aggregated_mean_std.csv" 2>/dev/null | \
            awk -F, '{printf "    macro=%.4f±%.4f collapse=%.4f±%.4f\n",$5,$6,$7,$8}' || echo "    not found"
    done
}

# ============================================================
# U3: Full fine-tuning baseline — unfreeze encoder + head
#     Uses fit_full_model() with fair KD (KD only on replay samples)
#     Results in outputs/full_ft_baseline_fair_kd/
# ============================================================
run_u3() {
    echo "================================================================"
    echo "U3: Full fine-tuning baseline (encoder+head, fair KD, 5 seeds)"
    echo "================================================================"
    local BASE="outputs/full_ft_baseline_fair_kd"

    for SEED in $(seq 0 $((SEEDS - 1))); do
        echo "  Seed ${SEED}"
        python scripts/collapse_active_maintenance_tls22.py \
            --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
            --reference-period "${REF}" --target-period M-2022-12 \
            --strategies "margin" --budgets "1000" \
            --eval-collapse-classes "${EVAL_COLLAPSE}" \
            ${CARE_ARGS} \
            --replay-distill-weight 0.5 \
            --distill-temperature 2.0 \
            --ft-depth full \
            --seed "${SEED}" \
            --output-dir "${BASE}/seed_${SEED}"
    done
    python scripts/aggregate_seeds.py --base-dir "${BASE}" --num-seeds "${SEEDS}"
    echo "  Result:"
    grep "margin.*1000" "${BASE}/aggregated_mean_std.csv" || true
}

# ============================================================
case "${1:-all}" in
    u1) run_u1 ;;
    u2) run_u2 ;;
    u3) run_u3 ;;
    all) run_u1; run_u2; run_u3 ;;
    *) echo "Usage: $0 [u1|u2|u3|all]"; exit 1 ;;
esac
