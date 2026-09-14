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
sys.exit(int(os.environ.get('RETURN_CODE', '0')))
PY
chmod +x "$scratch/fake-python"
export PYTHON="$scratch/fake-python" CALLS="$scratch/calls.jsonl" GPU_ID=3
export STUDY_DIR="$scratch/study space" PILOT_DIR="$scratch/pilot space" DIAG_OUTPUT_DIR="$scratch/diagnosis space" DIAG_SEEDS=0,2
bash "$script_dir/run_kbs_acquisition_diagnosis.sh" run --batch-size 256
python3 - "$CALLS" "$PILOT_DIR" "$DIAG_OUTPUT_DIR" <<'PY'
import json, sys
rows = [json.loads(s) for s in open(sys.argv[1])]
assert len(rows) == 1
r = rows[0]
args = r['args']
assert args[1] == 'run' and args[0].endswith('kbs_acquisition_diagnosis.py')
assert args[args.index('--pilot-dir') + 1] == sys.argv[2]
assert args[args.index('--output-dir') + 1] == sys.argv[3]
assert args[args.index('--seeds') + 1] == '0,2'
assert r['gpu'] == '3' and r['cwd'].endswith('/Experiment/core_code')
assert args[-2:] == ['--batch-size', '256']
PY
set +e
RETURN_CODE=48 bash "$script_dir/run_kbs_acquisition_diagnosis.sh" preflight
code=$?
set -e
[[ "$code" == 48 ]]
echo 'Diagnosis launcher tests passed'
