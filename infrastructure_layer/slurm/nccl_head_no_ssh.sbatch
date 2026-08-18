#!/bin/bash
#SBATCH --job-name=nccl-head
#SBATCH --partition=gpu-queue
#SBATCH --qos=gpu-q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpunode1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --output=logs/nccl-pair-%x-%j.out
#SBATCH --error=logs/nccl-pair-%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

: "${RUN_ID:?RUN_ID is required}"

IMAGE="${IMAGE:-$PWD/build/vllm-gloo-patched.sandbox}"
CONTROL_IF="${CONTROL_IF:-ens11f0}"

SESSION_DIR="$PWD/logs/nccl-pair/$RUN_ID"
mkdir -p "$SESSION_DIR"
rm -f "$SESSION_DIR/endpoint.env"

HEAD_IP="$(getent ahostsv4 "$(hostname -s)" | awk 'NR==1{print $1}')"
MASTER_PORT=$((50000 + SLURM_JOB_ID % 650))
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cat > "$SESSION_DIR/endpoint.env.tmp" <<EOF
MASTER_ADDR=$HEAD_IP
MASTER_PORT=$MASTER_PORT
EOF
mv "$SESSION_DIR/endpoint.env.tmp" "$SESSION_DIR/endpoint.env"

echo "============================================================"
echo " NCCL SOCKET HEAD / NO SSH"
echo "============================================================"
echo "run_id=$RUN_ID"
echo "job=$SLURM_JOB_ID"
echo "host=$(hostname -f)"
echo "image=$IMAGE"
echo "master=$HEAD_IP:$MASTER_PORT"
echo "control_if=$CONTROL_IF"
echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"
echo "kernel_ephemeral=$(cat /proc/sys/net/ipv4/ip_local_port_range)"
echo "NCCL_NET=Socket"
echo "NCCL_IB_DISABLE=1"
echo "============================================================"

[[ -d "$IMAGE" ]] || { echo "ERROR: sandbox missing: $IMAGE"; exit 2; }

PORT_TRACE="$SESSION_DIR/head-listen-ports.log"
: > "$PORT_TRACE"

monitor_ports() {
  while true; do
    {
      date +%s.%N
      ss -H -lnt4 2>/dev/null | awk '{print $4}' | sed -n 's/.*:\([0-9][0-9]*\)$/\1/p' | sort -n | tr '\n' ' '
      echo
    } >> "$PORT_TRACE"
    sleep 0.2
  done
}
monitor_ports &
MON_PID=$!

cleanup() {
  set +e
  kill "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

set +e
timeout --signal=TERM 150s \
apptainer exec \
  --nv \
  --cleanenv \
  --env "MASTER_ADDR=$HEAD_IP" \
  --env "MASTER_PORT=$MASTER_PORT" \
  --env "RANK=0" \
  --env "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES" \
  --env "NCCL_NET=Socket" \
  --env "NCCL_IB_DISABLE=1" \
  --env "NCCL_SOCKET_IFNAME==$CONTROL_IF" \
  --env "NCCL_SOCKET_FAMILY=AF_INET" \
  --env "NCCL_DEBUG=INFO" \
  --env "NCCL_DEBUG_SUBSYS=INIT,NET" \
  --bind "$PWD:$PWD" \
  "$IMAGE" \
  python3 "$PWD/slurm/nccl_pair_test.py"
RC=$?
set -e

cleanup
trap - EXIT INT TERM

echo
echo "=== UNIQUE LISTEN PORT SNAPSHOTS / HEAD ==="
awk 'NR%2==0' "$PORT_TRACE" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -nu | tr '\n' ' '
echo

if [[ "$RC" -ne 0 ]]; then
  echo "NCCL HEAD FAILED rc=$RC"
  exit "$RC"
fi

echo "NCCL HEAD PASS"
