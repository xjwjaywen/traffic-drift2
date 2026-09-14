#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
cat > "$scratch/fake-python" <<'PY'
#!/usr/bin/env python3
import json, os, sys
with open(os.environ['CALLS'], 'a') as stream:
    stream.write(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(),
                             'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}) + '\n')
if os.environ.get('FAIL_RUN') == '1':
    sys.exit(43)
PY
chmod +x "$scratch/fake-python"
export PYTHON="$scratch/fake-python" CALLS="$scratch/calls.jsonl"
export OUTPUT_DIR="$scratch/study space" CACHE_DIR="$scratch/cache space" DATA_DIR="$scratch/data space"
export BADGE_CONTROL_SEEDS=0,1 GPU_ID=2 SEEDS=9 STEPS=3 AUDIT_SEEDS=8
for mode in plan preflight run summarize; do
    bash "$script_dir/run_kbs_badge_controls.sh" "$mode" --batch-size 16
done
python3 - "$CALLS" "$OUTPUT_DIR" "$CACHE_DIR" "$DATA_DIR" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
assert [r['args'][1] for r in rows] == ['plan', 'preflight', 'run', 'summarize']
for row in rows:
    args = row['args']
    assert args[args.index('--output-dir') + 1] == sys.argv[2]
    assert args[args.index('--cache-dir') + 1] == sys.argv[3]
    assert args[args.index('--data-dir') + 1] == sys.argv[4]
    assert args[args.index('--seeds') + 1] == '0,1'
    assert '--steps' not in args and args[-2:] == ['--batch-size', '16']
    assert row['cwd'].endswith('/Experiment/core_code') and row['gpu'] == '2'
PY
export FAIL_RUN=1
set +e
bash "$script_dir/run_kbs_badge_controls.sh" run
code=$?
set -e
[[ "$code" == 43 ]]
unset FAIL_RUN BADGE_CONTROL_SEEDS
: > "$CALLS"
bash "$script_dir/run_kbs_badge_controls.sh" plan --seeds 2,3
python3 - "$CALLS" <<'PY'
import json, sys
args = json.loads(open(sys.argv[1]).readline())['args']
assert args.count('--seeds') == 1 and args[-2:] == ['--seeds', '2,3']
PY
echo 'BADGE controls launcher tests passed'
