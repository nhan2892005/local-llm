#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs/vllm-run

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}"
SESSION="$PWD/logs/vllm-run/$RUN_ID"
mkdir -p "$SESSION"

echo "RUN_ID=$RUN_ID"

# Submit worker first; it will wait for the head's shared ray.env.
W_SUBMIT="$(sbatch --export=ALL,RUN_ID="$RUN_ID" slurm/vllm_worker_no_ssh.sbatch)"
W_JOB="$(awk '{print $NF}' <<<"$W_SUBMIT")"

H_SUBMIT="$(sbatch --export=ALL,RUN_ID="$RUN_ID" slurm/vllm_head_no_ssh.sbatch)"
H_JOB="$(awk '{print $NF}' <<<"$H_SUBMIT")"

cat > "$SESSION/jobs.env" <<EOF
RUN_ID=$RUN_ID
HEAD_JOB=$H_JOB
WORKER_JOB=$W_JOB
EOF

echo "HEAD_JOB=$H_JOB"
echo "WORKER_JOB=$W_JOB"
echo "SESSION=$SESSION"
echo
echo "Watch jobs:"
echo "  squeue -j $H_JOB,$W_JOB"
echo
echo "Watch head:"
echo "  tail -f logs/vllm-head-$H_JOB.out"
echo
echo "Watch vLLM server log after Ray is ready:"
echo "  tail -f $SESSION/vllm-server.log"
echo
echo "When API is ready:"
echo "  cat $SESSION/api.env"
echo
echo "Stop both:"
echo "  scancel $H_JOB $W_JOB"
