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
export CHECKPOINT="$scratch/model space.pt" REPAIR_OUTPUT_DIR="$scratch/pilot input"
export ORACLE_OUTPUT_DIR="$scratch/oracle output" GPU_ID=2
export SEEDS=9 REPAIR_SEEDS=8 DEVICE=cpu STEPS=4 OUTPUT_DIR=ignored
unset ORACLE_SEEDS
bash "$script_dir/run_kbs_oracle_protection.sh" run --seeds 0,1
python3 - "$CALLS" <<'PY'
import json, os, sys
rows = [json.loads(s) for s in open(sys.argv[1])]
assert [r['args'][1] for r in rows] == ['select', 'train', 'evaluate']
for r in rows:
    a = r['args']
    for flag, env in [('--study-dir', 'STUDY_DIR'), ('--cache-dir', 'CACHE_DIR'),
                      ('--pilot-dir', 'REPAIR_OUTPUT_DIR'), ('--output-dir', 'ORACLE_OUTPUT_DIR'),
                      ('--checkpoint', 'CHECKPOINT')]:
        assert a[a.index(flag)+1] == os.environ[env]
    assert a[a.index('--seeds')+1] == '0,1,2' and a[-2:] == ['--seeds', '0,1']
    assert '--device' not in a and '--steps' not in a
    assert r['cwd'].endswith('/Experiment/core_code') and r['gpu'] == '2'
PY
for stage in select train; do
    export FAIL_STAGE="$stage"
    : > "$CALLS"
    set +e
    bash "$script_dir/run_kbs_oracle_protection.sh" run
    status=$?
    set -e
    [[ "$status" == 43 ]]
    python3 - "$CALLS" "$stage" <<'PY'
import json, sys
modes = [json.loads(s)['args'][1] for s in open(sys.argv[1])]
assert modes == (['select'] if sys.argv[2] == 'select' else ['select', 'train'])
PY
done
unset FAIL_STAGE
export ORACLE_SEEDS=1,2
: > "$CALLS"
for mode in plan preflight select train evaluate; do
    bash "$script_dir/run_kbs_oracle_protection.sh" "$mode"
done
python3 - "$CALLS" <<'PY'
import json, sys
for r in map(json.loads, open(sys.argv[1])):
    a = r['args']
    assert a[a.index('--seeds') + 1] == '1,2'
PY
echo 'Oracle protection launcher tests passed'
