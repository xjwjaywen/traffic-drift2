#!/usr/bin/env bash
set -euo pipefail
tests_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
launcher="$tests_dir/../run_kbs_repair_diagnosis.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cat > "$tmp/fake python" <<'PY'
#!/usr/bin/env python3
import json, os, sys
with open(os.environ['CALLS'], 'a') as f:
    f.write(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(),
                        'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}) + '\n')
sys.exit(43 if os.environ.get('FAIL') == '1' else 0)
PY
chmod +x "$tmp/fake python"
export PYTHON="$tmp/fake python" CALLS="$tmp/calls.jsonl"
export STUDY_DIR="$tmp/study space" CACHE_DIR="$tmp/cache space"
export CHECKPOINT="$tmp/checkpoint space.pt" REPAIR_OUTPUT_DIR="$tmp/pilot input"
export REPAIR_DIAG_OUTPUT_DIR="$tmp/diagnosis output" DIAG_THREADS=3
export GPU_ID=2 CUDA_VISIBLE_DEVICES=2 SEEDS=9 REPAIR_SEEDS=8 DEVICE=cuda OUTPUT_DIR=ignored
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
                      ('--output-dir', 'REPAIR_DIAG_OUTPUT_DIR')]:
        assert args[args.index(flag) + 1] == os.environ[env]
    assert '--seeds' not in args and '--device' not in args
    assert args[-2:] == ['--threads', '2']
    assert pathlib.Path(row['cwd']) == pathlib.Path(sys.argv[1]).resolve()
    assert row['gpu'] == ''
PY
set +e
FAIL=1 bash "$launcher" run
status=$?
set -e
test "$status" -eq 43
echo 'Repair diagnosis launcher tests passed.'
