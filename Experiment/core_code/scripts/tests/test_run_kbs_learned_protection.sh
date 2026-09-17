#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
cat > "$scratch/fake-python" <<'PY'
#!/usr/bin/env python3
import json, os, sys
with open(os.environ['CALLS'], 'a') as f:
    f.write(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')})+'\n')
if sys.argv[2] == os.environ.get('FAIL_STAGE'):
    sys.exit(43)
PY
chmod +x "$scratch/fake-python"
export PYTHON="$scratch/fake-python" CALLS="$scratch/calls.jsonl"
export STUDY_DIR="$scratch/study space" CACHE_DIR="$scratch/cache space" CHECKPOINT="$scratch/checkpoint space.pt"
export REPAIR_OUTPUT_DIR="$scratch/pilot source" LEARNED_PROTECTION_OUTPUT_DIR="$scratch/new output"
export GPU_ID=2 SEEDS=9 REPAIR_SEEDS=8 ORACLE_SEEDS=7 PROTECTION_BUDGET_SEEDS=6
export OUTPUT_DIR=ignored DEVICE=cpu STEPS=1 ORACLE_OUTPUT_DIR=ignored PROTECTION_BUDGET_OUTPUT_DIR=ignored
unset LEARNED_PROTECTION_SEEDS
bash "$script_dir/run_kbs_learned_protection.sh" run --seeds 0,1
python3 - "$CALLS" <<'PY'
import json, os, sys
rows = [json.loads(s) for s in open(sys.argv[1])]
assert [r['args'][1] for r in rows] == ['select','train','evaluate']
for r in rows:
    a = r['args']
    for flag, env in [('--study-dir','STUDY_DIR'), ('--cache-dir','CACHE_DIR'), ('--checkpoint','CHECKPOINT'),
                      ('--pilot-dir','REPAIR_OUTPUT_DIR'), ('--output-dir','LEARNED_PROTECTION_OUTPUT_DIR')]:
        assert a[a.index(flag)+1] == os.environ[env]
    assert a[a.index('--seeds')+1] == '0,1,2' and a[-2:] == ['--seeds','0,1']
    assert not {'--device','--steps','--oracle-dir'} & set(a)
    assert r['gpu'] == '2' and r['cwd'].endswith('/Experiment/core_code')
PY
for stage in select train; do
    export FAIL_STAGE="$stage"
    : > "$CALLS"
    set +e
    bash "$script_dir/run_kbs_learned_protection.sh" run
    status=$?
    set -e
    [[ "$status" == 43 ]]
    python3 - "$CALLS" "$stage" <<'PY'
import json, sys
modes = [json.loads(s)['args'][1] for s in open(sys.argv[1])]
assert modes == (['select'] if sys.argv[2] == 'select' else ['select','train'])
PY
done
unset FAIL_STAGE
export LEARNED_PROTECTION_SEEDS=1,2
: > "$CALLS"
for mode in plan preflight select train evaluate; do
    bash "$script_dir/run_kbs_learned_protection.sh" "$mode"
done
python3 - "$CALLS" <<'PY'
import json, sys
for r in map(json.loads, open(sys.argv[1])):
    a = r['args']; assert a[a.index('--seeds')+1] == '1,2'
PY
echo 'Learned protection launcher tests passed'
