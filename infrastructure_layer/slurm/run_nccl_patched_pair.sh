#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs/nccl-patched

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}"

H="$(sbatch --export=ALL,RUN_ID="$RUN_ID" slurm/nccl_patched_head.sbatch)"
HID="$(awk '{print $NF}' <<<"$H")"

W="$(sbatch --export=ALL,RUN_ID="$RUN_ID" slurm/nccl_patched_worker.sbatch)"
WID="$(awk '{print $NF}' <<<"$W")"

echo "RUN_ID=$RUN_ID"
echo "HEAD_JOB=$HID"
echo "WORKER_JOB=$WID"
echo
echo "cat logs/nccl-patched-ncclp-head-$HID.out"
echo "cat logs/nccl-patched-ncclp-worker-$WID.out"
echo "cat logs/nccl-patched-ncclp-head-$HID.err"
echo "cat logs/nccl-patched-ncclp-worker-$WID.err"
