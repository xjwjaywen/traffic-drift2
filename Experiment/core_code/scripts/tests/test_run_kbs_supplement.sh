#!/usr/bin/env bash
# Launcher behavior only. Never calls CESNET or a GPU.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

cat > "$scratch/fake-python" <<'PY'
#!/usr/bin/env python3
import json, os, sys
with open(os.environ["CALLS"], "a") as stream:
    stream.write(json.dumps({"args": sys.argv[1:], "cwd": os.getcwd(),
                             "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\n")
if "prepare" in sys.argv and os.environ.get("FAIL_PREPARE") == "1":
    sys.exit(42)
PY
chmod +x "$scratch/fake-python"
export PYTHON="$scratch/fake-python" CALLS="$scratch/calls.jsonl"
export DATA_DIR="$scratch/data with spaces" OUTPUT_DIR="$scratch/output with spaces" GPU_ID=2
export SEEDS=0,1
bash "$script_dir/run_kbs_supplement.sh" primary --steps 2 > "$scratch/log" 2>&1
python3 - "$CALLS" "$DATA_DIR" <<'PY'
import json, sys
rows = [json.loads(s) for s in open(sys.argv[1])]
assert len(rows) == 2
assert rows[0]["args"][1] == "prepare"
assert rows[1]["args"][1] == "run"
assert all(r["args"][r["args"].index("--data-dir") + 1] == sys.argv[2] for r in rows)
assert all(r["gpu"] == "2" and r["cwd"].endswith("Experiment/core_code") for r in rows)
PY

: > "$CALLS"
export FAIL_PREPARE=1
set +e
bash "$script_dir/run_kbs_supplement.sh" primary > "$scratch/log" 2>&1
code=$?
set -e
[[ "$code" == 42 ]]
[[ "$(wc -l < "$CALLS" | tr -d ' ')" == 1 ]]
echo "KBS launcher tests passed"
