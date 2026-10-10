#!/usr/bin/env bash
# Foreground result export/upload; Python standard library only.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUNBUFFERED=1
exec "${PYTHON:-python}" "$script_dir/export_care_results.py" "${1:-publish}" "${@:2}"
