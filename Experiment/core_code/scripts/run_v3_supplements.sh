#!/bin/bash
# Supplement experiments for v3 paper submission
# Run from Experiment/core_code/
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/eval_tls22.yaml"
CHECKPOINT="outputs/tls22_cnn/best_model.pt"
REF="M-2022-4"
EVAL_COLLAPSE="56,163,174,48,38,69,104,47,66,10,109,26"
SEEDS=5

# ============================================================
# S1: Missing ablation rows — "Replay only" and "Distill only"
#     (no target labels, only source data)
# ============================================================
run_s1() {
    echo "================================================================"
    echo "S1: Missing ablation rows (no target labels)"
    echo "================================================================"

    # S1a: Replay only — head FT with ONLY replay samples, no target labels
    echo "--- S1a: Replay only (0 target labels, 890 replay) ---"
    for SEED in $(seq 0 $((SEEDS - 1))); do
        python scripts/collapse_active_maintenance_tls22.py \
            --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
            --reference-period "${REF}" --target-period M-2022-12 \
            --strategies "random" --budgets "0" \
            --eval-collapse-classes "${EVAL_COLLAPSE}" \
            --replay-mode all --replay-per-class 5 --target-repeat 1 \
            --replay-distill-weight 0.0 \
            --seed "${SEED}" \
            --output-dir "outputs/ablation_v3/replay_only/seed_${SEED}"
    done
    python scripts/aggregate_seeds.py --base-dir "outputs/ablation_v3/replay_only" --num-seeds "${SEEDS}" 2>/dev/null || true

    # S1b: Distill only — head FT with ONLY replay+distillation, no target labels
    echo "--- S1b: Distill only (0 target labels, 890 replay + KL distill) ---"
    for SEED in $(seq 0 $((SEEDS - 1))); do
        python scripts/collapse_active_maintenance_tls22.py \
            --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
            --reference-period "${REF}" --target-period M-2022-12 \
            --strategies "random" --budgets "0" \
            --eval-collapse-classes "${EVAL_COLLAPSE}" \
            --replay-mode all --replay-per-class 5 --target-repeat 1 \
            --replay-distill-weight 0.5 --distill-temperature 2.0 \
            --seed "${SEED}" \
            --output-dir "outputs/ablation_v3/distill_only/seed_${SEED}"
    done
    python scripts/aggregate_seeds.py --base-dir "outputs/ablation_v3/distill_only" --num-seeds "${SEEDS}" 2>/dev/null || true

    echo ""
    echo "S1 results:"
    for d in replay_only distill_only; do
        echo "  $d:"
        cat "outputs/ablation_v3/$d/aggregated_mean_std.csv" 2>/dev/null | tail -1 || echo "    not found"
    done
}

# ============================================================
# S2: M10 with 3 seeds (seed 0 already done)
# ============================================================
run_s2() {
    echo "================================================================"
    echo "S2: M10 seeds 1-2 + aggregate"
    echo "================================================================"
    for SEED in 1 2; do
        echo "--- Seed ${SEED} ---"
        python scripts/collapse_active_maintenance_tls22.py \
            --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
            --reference-period "${REF}" --target-period M-2022-10 \
            --strategies margin --budgets "1000" \
            --eval-collapse-classes "${EVAL_COLLAPSE}" \
            --replay-mode all --replay-per-class 5 --target-repeat 2 \
            --replay-distill-weight 0.5 \
            --seed "${SEED}" \
            --output-dir "outputs/care_allreplay_M10/seed_${SEED}"
    done
    python scripts/aggregate_seeds.py --base-dir "outputs/care_allreplay_M10" --num-seeds 3

    echo ""
    echo "S2 M10 results:"
    grep "margin.*1000" "outputs/care_allreplay_M10/aggregated_mean_std.csv"
}

# ============================================================
# S3: All-class replay budget sweep (B=200,500,1000)
# ============================================================
run_s3() {
    echo "================================================================"
    echo "S3: All-class replay budget sweep (${SEEDS} seeds)"
    echo "================================================================"
    for SEED in $(seq 0 $((SEEDS - 1))); do
        echo "--- Seed ${SEED} ---"
        python scripts/collapse_active_maintenance_tls22.py \
            --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
            --reference-period "${REF}" --target-period M-2022-12 \
            --strategies "margin" --budgets "200,500,1000" \
            --eval-collapse-classes "${EVAL_COLLAPSE}" \
            --replay-mode all --replay-per-class 5 --target-repeat 2 \
            --replay-distill-weight 0.5 \
            --seed "${SEED}" \
            --output-dir "outputs/budget_sweep_allreplay/seed_${SEED}"
    done
    python scripts/aggregate_seeds.py --base-dir "outputs/budget_sweep_allreplay" --num-seeds "${SEEDS}"

    echo ""
    echo "S3 budget sweep results:"
    cat "outputs/budget_sweep_allreplay/aggregated_mean_std.csv"
}

# ============================================================
# Main
# ============================================================
case "${1:-all}" in
    s1) run_s1 ;;
    s2) run_s2 ;;
    s3) run_s3 ;;
    all) run_s1; run_s2; run_s3 ;;
    *) echo "Usage: $0 [s1|s2|s3|all]"; exit 1 ;;
esac
