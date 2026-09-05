#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

unset ALL_PROXY all_proxy
ulimit -n "$(ulimit -Hn)"

exec uv run --no-sync python temp/main_g1j.py "$@"
