#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "$tmp_dir"' EXIT
mkdir -p "$tmp_dir/path with spaces/scripts"
cp "$script_dir/../run_care_target_protection.sh" "$tmp_dir/path with spaces/scripts/"
cat > "$tmp_dir/mock python" <<'SH'
#!/usr/bin/env bash
python - "$@" <<'PY'
import json, os, sys
with open(os.environ['MOCK_RECORD'], 'a') as f:
    f.write(json.dumps({'argv': sys.argv[1:], 'cuda': os.environ.get('CUDA_VISIBLE_DEVICES'),
                       'omp': os.environ.get('OMP_NUM_THREADS'), 'mkl': os.environ.get('MKL_NUM_THREADS')}) + '\n')
sys.exit(17 if os.environ.get('MOCK_FAIL') == sys.argv[2] else 0)
PY
SH
chmod +x "$tmp_dir/mock python"
export PYTHON="$tmp_dir/mock python" MOCK_RECORD="$tmp_dir/record.jsonl"
export SEEDS=99 OUTPUT_DIR=old SMALL_REPAIR_OUTPUT_DIR=old DEVICE=cuda CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=128
export CARE_PROTECT_SOURCE_DIR='outputs/old results' CARE_PROTECT_OUTPUT_DIR='outputs/new protected' CARE_PROTECT_THREADS=4
bash "$tmp_dir/path with spaces/scripts/run_care_target_protection.sh" run --protection-budget 20
python - <<'PY'
import json, os
rows = [json.loads(x) for x in open(os.environ['MOCK_RECORD'])]
assert [r['argv'][1] for r in rows] == ['select', 'train', 'evaluate'], rows
for r in rows:
    a = r['argv']
    assert r['cuda'] == '' and r['omp'] == '4' and r['mkl'] == '4', r
    assert a[a.index('--source-dir')+1] == 'outputs/old results', a
    assert a[a.index('--output-dir')+1] == 'outputs/new protected', a
    assert a[a.index('--threads')+1] == '4', a
    assert a[-2:] == ['--protection-budget', '20'], a
PY
export MOCK_FAIL=train
set +e
bash "$tmp_dir/path with spaces/scripts/run_care_target_protection.sh" run
status=$?
set -e
test "$status" -eq 17
python - <<'PY'
import json, os
rows = [json.loads(x) for x in open(os.environ['MOCK_RECORD'])]
assert [r['argv'][1] for r in rows[3:]] == ['select', 'train'], rows
PY
echo 'CARE target protection launcher checks passed.'
