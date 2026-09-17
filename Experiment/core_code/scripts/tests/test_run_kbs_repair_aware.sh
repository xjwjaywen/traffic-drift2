#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
cat > "$scratch/fake-python" <<'PY'
#!/usr/bin/env python3
import json, os, sys
with open(os.environ['CALLS'], 'a') as f:
    f.write(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}) + '\n')
if sys.argv[2] == os.environ.get('FAIL_STAGE'):
    sys.exit(43)
PY
chmod +x "$scratch/fake-python"
export PYTHON="$scratch/fake-python" CALLS="$scratch/calls.jsonl"
export STUDY_DIR="$scratch/study space" CACHE_DIR="$scratch/cache space"
export REPAIR_OUTPUT_DIR="$scratch/output space" REPAIR_SEEDS=0,1 GPU_ID=2
export SEEDS=9 STEPS=4 PILOT_SEEDS=8 OUTPUT_DIR=ignored
bash "$script_dir/run_kbs_repair_aware.sh" run --batch-size 17
python3 - "$CALLS" "$STUDY_DIR" "$CACHE_DIR" "$REPAIR_OUTPUT_DIR" <<'PY'
import json, sys
rows = [json.loads(s) for s in open(sys.argv[1])]
assert [r['args'][1] for r in rows] == ['select', 'train', 'evaluate']
for r in rows:
    a = r['args']
    for flag, value in zip(['--study-dir', '--cache-dir', '--output-dir'], sys.argv[2:]):
        assert a[a.index(flag)+1] == value
    assert a[a.index('--seeds')+1] == '0,1'
    assert '--steps' not in a and a[-2:] == ['--batch-size', '17']
    assert r['cwd'].endswith('/Experiment/core_code') and r['gpu'] == '2'
PY
for stage in select train; do
    export FAIL_STAGE="$stage"
    : > "$CALLS"
    set +e
    bash "$script_dir/run_kbs_repair_aware.sh" run
    code=$?
    set -e
    [[ "$code" == 43 ]]
    python3 - "$CALLS" "$stage" <<'PY'
import json, sys
modes = [json.loads(s)['args'][1] for s in open(sys.argv[1])]
assert modes == (['select'] if sys.argv[2] == 'select' else ['select', 'train'])
PY
done
unset FAIL_STAGE
for mode in plan preflight select train evaluate; do
    bash "$script_dir/run_kbs_repair_aware.sh" "$mode"
done
echo 'Repair-aware launcher tests passed'
