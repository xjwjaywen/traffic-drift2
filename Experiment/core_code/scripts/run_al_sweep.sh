#!/usr/bin/env bash
# M1: Active TTA sampling-criterion sweep (precision design).
#
# Tier 3: tta_tc with 5 samplers × 3 seeds × 2 datasets   = 30 runs
# Tier 2: knn_labeled + ft_head with random × 3 seeds × 2 = 12 runs
# Total : 42 runs, ~3.5 hours on a single GPU.
#
# Run from Experiment/core_code/ :
#   bash scripts/run_al_sweep.sh
set -e

DATASETS=("quic22" "tls22")
SEEDS=(0 1 2)
TIER3_SAMPLERS=("random" "entropy" "margin" "coreset" "class_balanced")
TIER2_METHODS=("knn_labeled" "ft_head" "supervised_norm")

OUT_ROOT="outputs/al_sweep"
mkdir -p "$OUT_ROOT"

run_one () {
    local dataset=$1
    local method=$2
    local sampler=$3
    local seed=$4
    local ckpt="outputs/${dataset}_cnn/best_model.pt"
    local cfg="configs/eval_${dataset}.yaml"
    local out_dir="${OUT_ROOT}/${dataset}"
    mkdir -p "$out_dir"
    local tag="${method}_${sampler}_seed${seed}"
    local log="${out_dir}/${tag}.log"

    echo "==> [${dataset}] ${tag}"
    python evaluate_tta.py \
        --config "$cfg" \
        --checkpoint "$ckpt" \
        --mode sequential \
        --methods "$method" \
        --sampler "$sampler" \
        --seed "$seed" \
        --output-dir "$out_dir" \
        --output-suffix "$tag" \
        2>&1 | tee "$log"
}

for dataset in "${DATASETS[@]}"; do
    # Tier 3: 5 samplers × 3 seeds for tta_tc
    for sampler in "${TIER3_SAMPLERS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_one "$dataset" "tta_tc" "$sampler" "$seed"
        done
    done

    # Tier 2: random × 3 seeds for knn_labeled and ft_head
    for method in "${TIER2_METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_one "$dataset" "$method" "random" "$seed"
        done
    done
done

echo "==> Sweep complete. Run scripts/aggregate_al_sweep.py to compile results."
