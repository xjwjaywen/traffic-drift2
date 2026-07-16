#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH="$CORE_DIR/scripts/run_runtime_benchmark_m12.sh"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

cd "$CORE_DIR"

DRY_RUN=1 \
GPU=3 \
SEEDS="0 1" \
OUT="$TMP_DIR/dry" \
bash "$BENCH"

COMMANDS="$TMP_DIR/dry/commands.csv"
test -f "$COMMANDS"
test "$(wc -l < "$COMMANDS" | tr -d ' ')" -eq 9

grep -F 'care_margin,0,' "$COMMANDS" >/dev/null
grep -F 'care_badge,1,' "$COMMANDS" >/dev/null
grep -F 'chen_p090,0,' "$COMMANDS" >/dev/null
grep -F 'chen_p0997,1,' "$COMMANDS" >/dev/null
grep -F -- '--holdout-ratio 0.2' "$COMMANDS" >/dev/null
grep -F -- '--thresholds 0.90' "$COMMANDS" >/dev/null
grep -F -- '--thresholds 0.997' "$COMMANDS" >/dev/null
grep -F -- '--eval-collapse-classes 56\,163\,174\,48\,38\,69\,104\,47\,66\,10\,109\,26' "$COMMANDS" >/dev/null

mkdir -p "$TMP_DIR/summary"

SUMMARIZE_ONLY=1 \
OUT="$TMP_DIR/summary" \
RUNTIME_CSV="$SCRIPT_DIR/fixtures/runtime_runs.csv" \
bash "$BENCH"

SUMMARY="$TMP_DIR/summary/runtime_summary.csv"
test -f "$SUMMARY"
grep -F 'care_margin,2,12.0,2.8284271247461903' "$SUMMARY" >/dev/null
grep -F 'chen_p090,2,32.0,2.8284271247461903' "$SUMMARY" >/dev/null

echo "runtime benchmark self-tests passed"
