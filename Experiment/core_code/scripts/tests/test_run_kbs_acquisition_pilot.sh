#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
cat > "$scratch/fake-python" <<'PY'
#!/usr/bin/env python3
import json, os, sys
with open(os.environ['CALLS'], 'a') as stream:
    stream.write(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}) + '\n')
if 'select' in sys.argv and os.environ.get('FAIL_SELECT') == '1':
    sys.exit(47)
PY
chmod +x "$scratch/fake-python"
export PYTHON="$scratch/fake-python" CALLS="$scratch/calls.jsonl"
export STUDY_DIR="$scratch/study space" PILOT_OUTPUT_DIR="$scratch/pilot space" PILOT_SEEDS=0,1 GPU_ID=2
bash "$script_dir/run_kbs_acquisition_pilot.sh" run --batch-size 123
python3 - "$CALLS" "$PILOT_OUTPUT_DIR" "$STUDY_DIR" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
assert [r['args'][1] for r in rows] == ['select', 'evaluate']
for r in rows:
    args = r['args']
    assert args[args.index('--output-dir') + 1] == sys.argv[2]
    assert args[args.index('--cache-dir') + 1] == sys.argv[3] + '/cache'
    assert args[args.index('--seeds') + 1] == '0,1'
    assert args[-2:] == ['--batch-size', '123']
    assert r['gpu'] == '2' and r['cwd'].endswith('/Experiment/core_code')
PY
: > "$CALLS"
export FAIL_SELECT=1
set +e
bash "$script_dir/run_kbs_acquisition_pilot.sh" run
code=$?
set -e
[[ "$code" == 47 ]]
[[ "$(wc -l < "$CALLS" | tr -d ' ')" == 1 ]]
echo 'Acquisition launcher tests passed'
