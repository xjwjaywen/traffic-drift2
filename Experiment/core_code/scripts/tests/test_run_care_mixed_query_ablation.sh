#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "$tmp_dir"' EXIT
mkdir -p "$tmp_dir/path with spaces/scripts"
cp "$script_dir/../run_care_mixed_query_ablation.sh" "$tmp_dir/path with spaces/scripts/"
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
export SEEDS=99 OUTPUT_DIR=old CARE_PROTECT_OUTPUT_DIR=old DEVICE=cuda CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=128
export CARE_MIXED_PROTECTION_DIR='outputs/old protection' CARE_MIXED_OUTPUT_DIR='outputs/new mixed' CARE_MIXED_THREADS=4
bash "$tmp_dir/path with spaces/scripts/run_care_mixed_query_ablation.sh" run --eval-size 20
python - <<'PY'
import json, os
rows = [json.loads(x) for x in open(os.environ['MOCK_RECORD'])]
assert [r['argv'][1] for r in rows] == ['select', 'train', 'evaluate'], rows
for r in rows:
    a = r['argv']
    assert r['cuda'] == '' and r['omp'] == '4' and r['mkl'] == '4', r
    assert a[a.index('--protection-dir')+1] == 'outputs/old protection', a
    assert a[a.index('--output-dir')+1] == 'outputs/new mixed', a
    assert a[a.index('--threads')+1] == '4', a
    assert a[-2:] == ['--eval-size', '20'], a
PY
export MOCK_FAIL=train
set +e
bash "$tmp_dir/path with spaces/scripts/run_care_mixed_query_ablation.sh" run
status=$?
set -e
test "$status" -eq 17
python - <<'PY'
import json, os
rows = [json.loads(x) for x in open(os.environ['MOCK_RECORD'])]
assert [r['argv'][1] for r in rows[3:]] == ['select', 'train'], rows
PY
echo 'CARE mixed query ablation foreground launcher checks passed.'
