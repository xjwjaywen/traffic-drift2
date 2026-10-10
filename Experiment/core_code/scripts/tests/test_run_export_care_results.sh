#!/usr/bin/env bash
set -euo pipefail
tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "$tmp_dir"' EXIT
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$tmp_dir/path with spaces/scripts"
cp "$root/run_export_care_results.sh" "$tmp_dir/path with spaces/scripts/"
cat > "$tmp_dir/fake python" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$@" > "$RECORD"
test "$PYTHONUNBUFFERED" = 1
exit "${FAIL:-0}"
SH
chmod +x "$tmp_dir/fake python"
export PYTHON="$tmp_dir/fake python" RECORD="$tmp_dir/record"
bash "$tmp_dir/path with spaces/scripts/run_export_care_results.sh"
test "$(sed -n '2p' "$RECORD")" = publish
bash "$tmp_dir/path with spaces/scripts/run_export_care_results.sh" export --outputs-dir 'data with spaces'
test "$(sed -n '2p' "$RECORD")" = export
test "$(sed -n '4p' "$RECORD")" = 'data with spaces'
export FAIL=17
set +e
bash "$tmp_dir/path with spaces/scripts/run_export_care_results.sh" publish
status=$?
set -e
test "$status" -eq 17
echo 'CARE results foreground launcher checks passed.'
