#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs/gloo-pair

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}"

echo "RUN_ID=$RUN_ID"

HEAD_SUBMIT="$(sbatch --export=ALL,RUN_ID="$RUN_ID" slurm/gloo_head_no_ssh.sbatch)"
HEAD_JOB="$(awk '{print $NF}' <<<"$HEAD_SUBMIT")"

WORKER_SUBMIT="$(sbatch --export=ALL,RUN_ID="$RUN_ID" slurm/gloo_worker_no_ssh.sbatch)"
WORKER_JOB="$(awk '{print $NF}' <<<"$WORKER_SUBMIT")"

echo "HEAD_JOB=$HEAD_JOB"
echo "WORKER_JOB=$WORKER_JOB"
echo
echo "Watch:"
echo "  squeue -j $HEAD_JOB,$WORKER_JOB"
echo
echo "Logs:"
echo "  tail -f logs/gloo-pair-gloo-head-$HEAD_JOB.out"
echo "  tail -f logs/gloo-pair-gloo-worker-$WORKER_JOB.out"
echo
echo "Session:"
echo "  logs/gloo-pair/$RUN_ID"
