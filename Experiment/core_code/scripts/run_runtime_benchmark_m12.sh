#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$CORE_DIR"

GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2 3 4}"
OUT="${OUT:-outputs/runtime_benchmark_m12_final}"
DRY_RUN="${DRY_RUN:-0}"
SUMMARIZE_ONLY="${SUMMARIZE_ONLY:-0}"
RUNTIME_CSV="${RUNTIME_CSV:-$OUT/runtime_runs.csv}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

CONFIG="configs/eval_tls22.yaml"
CHECKPOINT="outputs/tls22_cnn/best_model.pt"
EVAL_COLLAPSE="56,163,174,48,38,69,104,47,66,10,109,26"

summarize_results() {
  BENCH_CSV="$RUNTIME_CSV" \
  BENCH_SUMMARY="$OUT/runtime_summary.csv" \
  python - <<'PY'
import csv
import os
import statistics

with open(os.environ["BENCH_CSV"], newline="") as handle:
    rows = list(csv.DictReader(handle))

groups = {}
for row in rows:
    if int(row["exit_code"]) != 0:
        continue
    groups.setdefault(row["method"], []).append(row)

if not groups:
    raise SystemExit("no successful runtime rows to summarize")

fieldnames = [
    "method",
    "n",
    "wall_s_mean",
    "wall_s_sample_sd",
    "max_rss_mib_mean",
    "max_rss_mib_sample_sd",
]
with open(os.environ["BENCH_SUMMARY"], "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for method, method_rows in sorted(groups.items()):
        wall = [float(row["wall_s"]) for row in method_rows]
        rss = [float(row["max_rss_kb"]) / 1024.0 for row in method_rows]
        writer.writerow({
            "method": method,
            "n": len(method_rows),
            "wall_s_mean": statistics.mean(wall),
            "wall_s_sample_sd": statistics.stdev(wall) if len(wall) > 1 else 0.0,
            "max_rss_mib_mean": statistics.mean(rss),
            "max_rss_mib_sample_sd": statistics.stdev(rss) if len(rss) > 1 else 0.0,
        })

for method, method_rows in sorted(groups.items()):
    wall = [float(row["wall_s"]) for row in method_rows]
    mean = statistics.mean(wall)
    sd = statistics.stdev(wall) if len(wall) > 1 else 0.0
    print(f"{method}: n={len(wall)} wall_s={mean:.2f} +/- {sd:.2f}")
PY
}

mkdir -p "$OUT"

if [[ "$SUMMARIZE_ONLY" == "1" ]]; then
  summarize_results
  exit 0
fi

echo 'method,seed,command' > "$OUT/commands.csv"

if [[ "$DRY_RUN" != "1" ]]; then
  command -v python >/dev/null
  command -v nvidia-smi >/dev/null
  test -x /usr/bin/time
  test -f "$CONFIG"
  test -f "$CHECKPOINT"
  test -d data/tls22

  {
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "git_sha=$(git rev-parse HEAD)"
    echo "gpu_index=$GPU"
    echo "seeds=$SEEDS"
    echo "omp_num_threads=$OMP_NUM_THREADS"
    echo "mkl_num_threads=$MKL_NUM_THREADS"
    CUDA_VISIBLE_DEVICES="$GPU" python - <<'PY'
import platform
import torch

print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"cudnn={torch.backends.cudnn.version()}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available on the selected GPU")
PY
    nvidia-smi -i "$GPU" \
      --query-gpu=name,uuid,driver_version,memory.total \
      --format=csv,noheader
  } > "$OUT/environment.txt"

  echo 'method,seed,wall_s,max_rss_kb,exit_code' > "$RUNTIME_CSV"
fi

run_one() {
  local method="$1"
  local seed="$2"
  shift 2
  local -a command=("$@")
  local rendered
  printf -v rendered '%q ' "${command[@]}"
  printf '%s,%s,"%s"\n' "$method" "$seed" "$rendered" >> "$OUT/commands.csv"

  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi

  local run_dir="$OUT/$method/seed_$seed"
  mkdir -p "$run_dir/result"
  echo "[$(date --iso-8601=seconds)] START $method seed=$seed"

  CUDA_VISIBLE_DEVICES="$GPU" /usr/bin/time \
    -f "$method,$seed,%e,%M,%x" \
    -a -o "$RUNTIME_CSV" \
    "${command[@]}" > "$run_dir/run.log" 2>&1

  echo "[$(date --iso-8601=seconds)] END $method seed=$seed"
}

run_method() {
  local method="$1"
  local seed="$2"

  case "$method" in
    care_margin|care_badge)
      local strategy="margin"
      if [[ "$method" == "care_badge" ]]; then
        strategy="badge"
      fi
      run_one "$method" "$seed" \
        python scripts/collapse_active_maintenance_tls22.py \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" \
        --reference-period M-2022-4 --target-period M-2022-12 \
        --strategies "$strategy" --budgets 1000 \
        --eval-collapse-classes "$EVAL_COLLAPSE" \
        --replay-mode all --replay-per-class 5 --target-repeat 2 \
        --replay-distill-weight 0.5 --distill-temperature 2.0 \
        --ft-depth head --ft-lr 0.001 --ft-epochs 30 \
        --ft-batch-size 64 --ft-weight-decay 0.0001 \
        --holdout-ratio 0.2 --seed "$seed" \
        --output-dir "$OUT/$method/seed_$seed/result"
      ;;
    chen_p090|chen_p0997)
      local threshold="0.90"
      if [[ "$method" == "chen_p0997" ]]; then
        threshold="0.997"
      fi
      run_one "$method" "$seed" \
        python scripts/baselines/self_evolving_baseline.py \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" \
        --reference-period M-2022-4 --target-period M-2022-12 \
        --thresholds "$threshold" --holdout-ratio 0.2 \
        --ft-depth full --ft-lr 0.0025 --ft-epochs 50 \
        --ft-batch-size 500 --ft-weight-decay 0.0001 \
        --replay-mode none --seed "$seed" \
        --output-dir "$OUT/$method/seed_$seed/result"
      ;;
    *)
      echo "unknown method: $method" >&2
      return 2
      ;;
  esac
}

methods=(care_margin care_badge chen_p090 chen_p0997)
read -r -a seed_list <<< "$SEEDS"

# Rotate method order by seed so cold filesystem/page-cache effects are not
# assigned to the same method in every run.
for seed in "${seed_list[@]}"; do
  shift_by=$((seed % ${#methods[@]}))
  for ((offset = 0; offset < ${#methods[@]}; offset++)); do
    index=$(((shift_by + offset) % ${#methods[@]}))
    run_method "${methods[$index]}" "$seed"
  done
done

if [[ "$DRY_RUN" != "1" ]]; then
  summarize_results
  echo "Benchmark complete: $OUT"
fi
