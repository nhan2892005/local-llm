#!/bin/bash
#SBATCH --job-name=ncclp-head
#SBATCH --partition=gpu-queue
#SBATCH --qos=gpu-q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpunode1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --output=logs/nccl-patched-%x-%j.out
#SBATCH --error=logs/nccl-patched-%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
: "${RUN_ID:?RUN_ID required}"

IMAGE="${IMAGE:-$PWD/build/vllm-gloo-nccl-patched.sandbox}"
SESSION="$PWD/logs/nccl-patched/$RUN_ID"
mkdir -p "$SESSION"

CONTROL_IF="${CONTROL_IF:-ens11f0}"
NCCL_PORT_MIN="${NCCL_PORT_MIN:-50600}"
NCCL_PORT_MAX="${NCCL_PORT_MAX:-51000}"

HEAD_IP="$(getent ahostsv4 "$(hostname -s)" | awk 'NR==1{print $1}')"
MASTER_PORT=$((50000 + SLURM_JOB_ID % 400))

cat > "$SESSION/endpoint.env.tmp" <<EOF
MASTER_ADDR=$HEAD_IP
MASTER_PORT=$MASTER_PORT
EOF
mv "$SESSION/endpoint.env.tmp" "$SESSION/endpoint.env"

echo "host=$(hostname -f)"
echo "master=$HEAD_IP:$MASTER_PORT"
echo "nccl_range=$NCCL_PORT_MIN-$NCCL_PORT_MAX"

apptainer exec \
  --nv \
  --cleanenv \
  --env "MASTER_ADDR=$HEAD_IP" \
  --env "MASTER_PORT=$MASTER_PORT" \
  --env "RANK=0" \
  --env "CUDA_VISIBLE_DEVICES=0" \
  --env "NCCL_NET=Socket" \
  --env "NCCL_IB_DISABLE=1" \
  --env "NCCL_SOCKET_IFNAME==$CONTROL_IF" \
  --env "NCCL_SOCKET_FAMILY=AF_INET" \
  --env "NCCL_PORT_MIN=$NCCL_PORT_MIN" \
  --env "NCCL_PORT_MAX=$NCCL_PORT_MAX" \
  --env "NCCL_DEBUG=INFO" \
  --env "NCCL_DEBUG_SUBSYS=INIT,NET,ENV" \
  --bind "$PWD:$PWD" \
  "$IMAGE" \
  python3 "$PWD/slurm/nccl_pair_test_patched.py"
