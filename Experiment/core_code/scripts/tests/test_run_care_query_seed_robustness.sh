#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "$tmp_dir"' EXIT
mkdir -p "$tmp_dir/path with spaces/scripts"
cp "$script_dir/../run_care_query_seed_robustness.sh" "$tmp_dir/path with spaces/scripts/"
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
export SEEDS=99 OUTPUT_DIR=old CARE_MIXED_OUTPUT_DIR=old QUERY_SEEDS=1,2 DEVICE=cuda CUDA_VISIBLE_DEVICES=3
export CARE_QUERY_ABLATION_DIR='outputs/old ablation' CARE_QUERY_OUTPUT_DIR='outputs/new robustness' CARE_QUERY_THREADS=4
bash "$tmp_dir/path with spaces/scripts/run_care_query_seed_robustness.sh" run --query-seeds 6174,6175
python - <<'PY'
import json, os
rows = [json.loads(x) for x in open(os.environ['MOCK_RECORD'])]
assert [r['argv'][1] for r in rows] == ['inventory', 'select', 'train', 'evaluate'], rows
for r in rows:
    a = r['argv']
    assert r['cuda'] == '' and r['omp'] == '4' and r['mkl'] == '4', r
    assert a[a.index('--ablation-dir')+1] == 'outputs/old ablation', a
    assert a[a.index('--output-dir')+1] == 'outputs/new robustness', a
    assert a[a.index('--threads')+1] == '4', a
    assert a[-2:] == ['--query-seeds', '6174,6175'], a
PY
export MOCK_FAIL=train
set +e
bash "$tmp_dir/path with spaces/scripts/run_care_query_seed_robustness.sh" run
status=$?
set -e
test "$status" -eq 17
python - <<'PY'
import json, os
rows = [json.loads(x) for x in open(os.environ['MOCK_RECORD'])]
assert [r['argv'][1] for r in rows[4:]] == ['inventory', 'select', 'train'], rows
PY
echo 'CARE query-seed robustness foreground launcher checks passed.'
