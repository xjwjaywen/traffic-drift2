#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
cat > "$scratch/fake-python" <<'PY'
#!/usr/bin/env python3
import json, os, sys
with open(os.environ['CALLS'], 'a') as stream:
    stream.write(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd()}) + '\n')
if 'freeze' in sys.argv and os.environ.get('FAIL_FREEZE') == '1':
    sys.exit(43)
PY
chmod +x "$scratch/fake-python"
export PYTHON="$scratch/fake-python" CALLS="$scratch/calls.jsonl"
export STUDY_DIR="$scratch/study space" RELATION_OUTPUT_DIR="$scratch/relation space"
export RELATION_SEEDS=0,1 SEEDS=9 AUDIT_SEEDS=8 POOL_SEEDS=7
bash "$script_dir/run_kbs_relation_audit.sh" run
python3 - "$CALLS" "$RELATION_OUTPUT_DIR" "$STUDY_DIR" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
assert [r['args'][1] for r in rows] == ['freeze', 'evaluate']
for r in rows:
    args = r['args']
    assert args[args.index('--output-dir') + 1] == sys.argv[2]
    assert args[args.index('--study-dir') + 1] == sys.argv[3]
    assert args[args.index('--seeds') + 1] == '0,1'
    assert r['cwd'].endswith('/Experiment/core_code')
    assert '--device' not in args and '--cache-dir' not in args
PY
: > "$CALLS"
export FAIL_FREEZE=1
set +e
bash "$script_dir/run_kbs_relation_audit.sh" run
code=$?
set -e
[[ "$code" == 43 ]]
[[ "$(wc -l < "$CALLS" | tr -d ' ')" == 1 ]]
unset RELATION_SEEDS
: > "$CALLS"
bash "$script_dir/run_kbs_relation_audit.sh" plan --seeds 2,3
python3 - "$CALLS" <<'PY'
import json, sys
args = json.loads(open(sys.argv[1]).readline())['args']
assert args.count('--seeds') == 1 and args[-2:] == ['--seeds', '2,3']
PY
echo 'Relation audit launcher tests passed'
