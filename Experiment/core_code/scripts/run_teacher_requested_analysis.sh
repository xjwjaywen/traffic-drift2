#!/usr/bin/env bash
set -euo pipefail

# Run the advisor-requested post-processing:
#   1. optional AdaBN evaluation on a BN checkpoint;
#   2. TTA / normalization / collapse visualizations.
#
# Usage from Experiment/core_code/:
#   bash scripts/run_teacher_requested_analysis.sh
#
# To skip AdaBN and only regenerate figures:
#   RUN_ADABN=0 bash scripts/run_teacher_requested_analysis.sh

RUN_ADABN="${RUN_ADABN:-1}"

if [[ "$RUN_ADABN" == "1" ]]; then
  bash scripts/run_tls22_adabn_drift_ablation.sh
else
  echo "Skipping AdaBN evaluation because RUN_ADABN=0"
fi

python scripts/visualize_teacher_results.py \
  --output-dir outputs/teacher_result_visuals

echo
echo "Advisor-facing summary:"
echo "  outputs/teacher_result_visuals/teacher_result_visuals_summary.md"

