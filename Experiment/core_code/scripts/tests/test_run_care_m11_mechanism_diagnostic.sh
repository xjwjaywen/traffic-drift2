#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "$tmp_dir"' EXIT
mkdir -p "$tmp_dir/path with spaces/scripts"
cp "$script_dir/../run_care_m11_mechanism_diagnostic.sh" "$tmp_dir/path with spaces/scripts/"
cat > "$tmp_dir/mock python" <<'SH'
#!/usr/bin/env bash
python - "$@" <<'PY'
import json, os, sys
with open(os.environ['MOCK_RECORD'], 'a') as f:
    f.write(json.dumps({'argv': sys.argv[1:], 'cuda': os.environ.get('CUDA_VISIBLE_DEVICES'),
                       'omp': os.environ.get('OMP_NUM_THREADS'), 'mkl': os.environ.get('MKL_NUM_THREADS')}) + '\n')
sys.exit(19 if os.environ.get('MOCK_FAIL') == sys.argv[2] else 0)
PY
SH
chmod +x "$tmp_dir/mock python"
export PYTHON="$tmp_dir/mock python" MOCK_RECORD="$tmp_dir/record.jsonl"
export SEEDS=99 OUTPUT_DIR=old QUERY_SEEDS=1,2 STEPS=1 DEVICE=cuda CUDA_VISIBLE_DEVICES=3
export CARE_M11_OUTPUT_DIR=old CARE_M11_THREADS=32 CARE_QUERY_OUTPUT_DIR=old
export CARE_M11_DIAG_OUTPUT_DIR='outputs/new diagnosis' CARE_M11_DIAG_MONTH_DIR='outputs/old m11'
export CARE_M11_DIAG_THREADS=4 CARE_M11_DIAG_CACHE_DIR='outputs/cache m11'
bash "$tmp_dir/path with spaces/scripts/run_care_m11_mechanism_diagnostic.sh" run --probe-cap 3
python - <<'PY'
import json, os
rows = [json.loads(x) for x in open(os.environ['MOCK_RECORD'])]
assert [r['argv'][1] for r in rows] == ['audit', 'geometry', 'report'], rows
for r in rows:
    a = r['argv']
    assert r['cuda'] == '' and r['omp'] == '4' and r['mkl'] == '4', r
    assert a[a.index('--month-dir')+1] == 'outputs/old m11', a
    assert a[a.index('--output-dir')+1] == 'outputs/new diagnosis', a
    assert a[a.index('--cache-dir')+1] == 'outputs/cache m11', a
    assert '--query-seeds' not in a and '--steps' not in a and '--device' not in a, a
    assert a[-2:] == ['--probe-cap', '3'], a
PY
export MOCK_FAIL=geometry
set +e
bash "$tmp_dir/path with spaces/scripts/run_care_m11_mechanism_diagnostic.sh" run
status=$?
set -e
test "$status" -eq 19
python - <<'PY'
import json, os
rows = [json.loads(x) for x in open(os.environ['MOCK_RECORD'])]
assert [r['argv'][1] for r in rows[3:]] == ['audit', 'geometry'], rows
PY
echo 'CARE M11 mechanism foreground launcher checks passed.'
