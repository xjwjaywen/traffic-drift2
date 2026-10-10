#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "$tmp_dir"' EXIT
mkdir -p "$tmp_dir/path with spaces/scripts"
cp "$script_dir/../run_small_repair_diagnostic.sh" "$tmp_dir/path with spaces/scripts/"
cat > "$tmp_dir/mock python" <<'SH'
#!/usr/bin/env bash
python - "$@" <<'PY'
import json, os, sys
with open(os.environ['MOCK_RECORD'], 'a') as f:
    f.write(json.dumps({'argv': sys.argv[1:], 'cuda': os.environ.get('CUDA_VISIBLE_DEVICES'),
                        'omp': os.environ.get('OMP_NUM_THREADS'), 'mkl': os.environ.get('MKL_NUM_THREADS')}) + '\n')
sys.exit(17 if sys.argv[2] == os.environ.get('MOCK_FAIL') else 0)
PY
SH
chmod +x "$tmp_dir/mock python"
export PYTHON="$tmp_dir/mock python" MOCK_RECORD="$tmp_dir/record.jsonl"
export SEEDS=99 OUTPUT_DIR=old DEVICE=cuda GPU_ID=3 CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=128
export SMALL_REPAIR_OUTPUT_DIR='outputs/custom result' SMALL_REPAIR_SEEDS=0,2 SMALL_REPAIR_THREADS=4
bash "$tmp_dir/path with spaces/scripts/run_small_repair_diagnostic.sh" run --eval-size 200
python - <<'PY'
import json, os
rows = [json.loads(x) for x in open(os.environ['MOCK_RECORD'])]
assert [r['argv'][1] for r in rows] == ['select', 'train', 'evaluate'], rows
for r in rows:
    a = r['argv']
    assert r['cuda'] == '' and r['omp'] == '4' and r['mkl'] == '4', r
    assert a[a.index('--seeds')+1] == '0,2', a
    assert a[a.index('--output-dir')+1] == 'outputs/custom result', a
    assert a[-2:] == ['--eval-size', '200'], a
PY
export MOCK_FAIL=train
set +e
bash "$tmp_dir/path with spaces/scripts/run_small_repair_diagnostic.sh" run
status=$?
set -e
test "$status" -eq 17
python - <<'PY'
import json, os
rows = [json.loads(x) for x in open(os.environ['MOCK_RECORD'])]
assert [r['argv'][1] for r in rows[3:]] == ['select', 'train'], rows
PY
echo 'Small repair launcher checks passed.'
