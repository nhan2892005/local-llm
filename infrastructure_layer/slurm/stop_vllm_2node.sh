#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <RUN_ID>"
  exit 2
fi

SESSION="$PWD/logs/vllm-run/$1"
[[ -f "$SESSION/jobs.env" ]] || { echo "missing $SESSION/jobs.env"; exit 2; }

source "$SESSION/jobs.env"
touch "$SESSION/STOP"
scancel "$HEAD_JOB" "$WORKER_JOB" 2>/dev/null || true
echo "Stopped HEAD_JOB=$HEAD_JOB WORKER_JOB=$WORKER_JOB"
