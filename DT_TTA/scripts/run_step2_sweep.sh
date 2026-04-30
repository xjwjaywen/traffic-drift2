#!/usr/bin/env bash
# DT-TTA Step 2: 6-setting sweep on QUIC22 + TLS22 with 3 seeds.
#
# Settings:
#   (0) static            — no adaptation (already in evaluate_tta)
#   (1) ft_head            — full classifier head fine-tune
#   (2) supervised_norm    — full GroupNorm γ/β
#   (3) selective_norm     — γ/β masked to drifted channels
#   (4) focal_strategy     — selective_norm + bias-only head
#   (5) diffuse_strategy   — full norm + full head
#
# Settings (3)-(5) require pre-computed source GroupNorm stats —
# run scripts/compute_source_stats.py first.
#
# Total: 6 settings × 2 datasets × 3 seeds = 36 sequential evaluations.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT/Experiment/core_code"

DATASETS=("quic22" "tls22")
SEEDS=(0 1 2)
NEED_STATS_METHODS=("selective_norm" "focal_strategy" "diffuse_strategy")
ALL_METHODS=("static" "ft_head" "supervised_norm"
             "selective_norm" "focal_strategy" "diffuse_strategy")

OUT_ROOT="../../DT_TTA/outputs/step2_sweep"
mkdir -p "$OUT_ROOT"

run_one () {
    local dataset=$1
    local method=$2
    local seed=$3
    local ckpt="outputs/${dataset}_cnn/best_model.pt"
    local cfg="configs/eval_${dataset}.yaml"
    local out_dir="${OUT_ROOT}/${dataset}"
    mkdir -p "$out_dir"
    local tag="${method}_seed${seed}"
    local log="${out_dir}/${tag}.log"

    local stats_arg=""
    for m in "${NEED_STATS_METHODS[@]}"; do
        if [[ "$m" == "$method" ]]; then
            stats_arg="--dt-source-stats ../../DT_TTA/outputs/source_stats/${dataset}_source_stats.pt"
        fi
    done

    echo "==> [${dataset}] ${tag}"
    python evaluate_tta.py \
        --config "$cfg" \
        --checkpoint "$ckpt" \
        --mode sequential \
        --methods "$method" \
        --sampler random \
        --seed "$seed" \
        --output-dir "$out_dir" \
        --output-suffix "$tag" \
        $stats_arg \
        2>&1 | tee "$log"
}

for dataset in "${DATASETS[@]}"; do
    for method in "${ALL_METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_one "$dataset" "$method" "$seed"
        done
    done
done

echo
echo "==> Sweep complete. Run: python ../../DT_TTA/scripts/aggregate_step2.py"
