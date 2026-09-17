#!/usr/bin/env bash
set -euo pipefail
tests_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
launcher="$tests_dir/../run_kbs_protection_gap_diagnosis.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cat > "$tmp/fake python" <<'PY'
#!/usr/bin/env python3
import json, os, sys
with open(os.environ['CALLS'], 'a') as f:
    f.write(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(),
                        'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
                        'omp': os.environ.get('OMP_NUM_THREADS')}) + '\n')
sys.exit(43 if os.environ.get('FAIL') == '1' else 0)
PY
chmod +x "$tmp/fake python"
export PYTHON="$tmp/fake python" CALLS="$tmp/calls.jsonl"
export STUDY_DIR="$tmp/study space" CACHE_DIR="$tmp/cache space"
export CHECKPOINT="$tmp/checkpoint space.pt" REPAIR_OUTPUT_DIR="$tmp/pilot input"
export LEARNED_PROTECTION_OUTPUT_DIR="$tmp/learned input" PROTECTION_BUDGET_OUTPUT_DIR="$tmp/allocation input"
export ORACLE_OUTPUT_DIR="$tmp/oracle input" PROTECTION_GAP_OUTPUT_DIR="$tmp/diagnosis output"
export DIAG_THREADS=3 PROTECTION_GAP_SEEDS=0,2
export GPU_ID=2 CUDA_VISIBLE_DEVICES=2 SEEDS=9 REPAIR_SEEDS=8 DEVICE=cuda OUTPUT_DIR=ignored
export LEARNED_PROTECTION_SEEDS=7 PROTECTION_BUDGET_SEEDS=6
for mode in plan preflight run; do
    bash "$launcher" "$mode" --threads 2
done
python3 - "$tests_dir/../.." <<'PY'
import json, os, pathlib, sys
rows = [json.loads(line) for line in open(os.environ['CALLS'])]
assert len(rows) == 3
for row, mode in zip(rows, ['plan', 'preflight', 'run']):
    args = row['args']
    assert args[1] == mode
    for flag, env in [('--study-dir', 'STUDY_DIR'), ('--cache-dir', 'CACHE_DIR'),
                      ('--checkpoint', 'CHECKPOINT'), ('--pilot-dir', 'REPAIR_OUTPUT_DIR'),
                      ('--learned-dir', 'LEARNED_PROTECTION_OUTPUT_DIR'),
                      ('--allocation-dir', 'PROTECTION_BUDGET_OUTPUT_DIR'), ('--oracle-dir', 'ORACLE_OUTPUT_DIR'),
                      ('--output-dir', 'PROTECTION_GAP_OUTPUT_DIR')]:
        assert args[args.index(flag) + 1] == os.environ[env]
    assert args[args.index('--seeds') + 1] == '0,2'
    assert '--device' not in args and args[-2:] == ['--threads', '2']
    assert pathlib.Path(row['cwd']) == pathlib.Path(sys.argv[1]).resolve()
    assert row['gpu'] == '' and row['omp'] == '3'
PY
set +e
FAIL=1 bash "$launcher" run
status=$?
set -e
test "$status" -eq 43
set +e
bash "$launcher" train
status=$?
set -e
test "$status" -eq 2
echo 'Protection gap diagnosis launcher tests passed.'
