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

child_pid=0
forward_signal() {
  trap - TERM INT
  [[ "$child_pid" -ne 0 ]] && kill -s "$1" "$child_pid" 2>/dev/null
  [[ "$child_pid" -ne 0 ]] && wait "$child_pid" 2>/dev/null
  exit "$2"
}
trap 'forward_signal TERM 143' TERM
trap 'forward_signal INT 130' INT

max_restarts="${RWKV_EVAL_MAX_RESTARTS:-5}"
delay="${RWKV_EVAL_RESTART_DELAY:-15}"
restarts=0
while true; do
  set +e
  uv run --no-sync python temp/main_g1j.py "$@" &
  child_pid=$!
  wait "$child_pid"
  status=$?
  set -e
  child_pid=0
  [[ "$status" -eq 0 ]] && exit "$status"
  ((restarts += 1))
  [[ "$restarts" -gt "$max_restarts" ]] && exit "$status"
  sleep "$delay"
done
