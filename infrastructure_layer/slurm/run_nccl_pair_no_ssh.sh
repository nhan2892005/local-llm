#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs/nccl-pair

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}"

echo "RUN_ID=$RUN_ID"

HEAD_SUBMIT="$(sbatch --export=ALL,RUN_ID="$RUN_ID" slurm/nccl_head_no_ssh.sbatch)"
HEAD_JOB="$(awk '{print $NF}' <<<"$HEAD_SUBMIT")"

WORKER_SUBMIT="$(sbatch --export=ALL,RUN_ID="$RUN_ID" slurm/nccl_worker_no_ssh.sbatch)"
WORKER_JOB="$(awk '{print $NF}' <<<"$WORKER_SUBMIT")"

echo "HEAD_JOB=$HEAD_JOB"
echo "WORKER_JOB=$WORKER_JOB"
echo
echo "Watch:"
echo "  squeue -j $HEAD_JOB,$WORKER_JOB"
echo
echo "Logs:"
echo "  tail -f logs/nccl-pair-nccl-head-$HEAD_JOB.out"
echo "  tail -f logs/nccl-pair-nccl-worker-$WORKER_JOB.out"
echo
echo "Errors:"
echo "  cat logs/nccl-pair-nccl-head-$HEAD_JOB.err"
echo "  cat logs/nccl-pair-nccl-worker-$WORKER_JOB.err"
echo
echo "Socket traces:"
echo "  logs/nccl-pair/$RUN_ID/head-listen-ports.log"
echo "  logs/nccl-pair/$RUN_ID/worker-listen-ports.log"
