#!/bin/bash
#SBATCH --job-name=gloo-port-smoke
#SBATCH --partition=gpu-queue
#SBATCH --qos=gpu-q
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/gloo-port-smoke-%j.out
#SBATCH --error=logs/gloo-port-smoke-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

IMAGE="${IMAGE:-$PWD/build/vllm-gloo-patched.sandbox}"
CONTROL_IF="${CONTROL_IF:-ens11f0}"
GLOO_PORT_MIN="${GLOO_PORT_MIN:-50700}"
GLOO_PORT_MAX="${GLOO_PORT_MAX:-51000}"

# Rendezvous stays below the patched Gloo listener range.
MASTER_PORT="${MASTER_PORT:-$((50000 + SLURM_JOB_ID % 650))}"

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ "${#NODES[@]}" -eq 2 ]] || {
  echo "ERROR: expected 2 nodes"
  exit 2
}

HEAD_NODE="${NODES[0]}"
WORKER_NODE="${NODES[1]}"

HEAD_IP="$(getent ahostsv4 "$HEAD_NODE" | awk 'NR==1{print $1}')"
WORKER_IP="$(getent ahostsv4 "$WORKER_NODE" | awk 'NR==1{print $1}')"

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=3
)

echo "============================================================"
echo " GLOO PATCH 2-NODE SMOKE TEST"
echo "============================================================"
echo "job=$SLURM_JOB_ID"
echo "image=$IMAGE"
echo "head=$HEAD_NODE $HEAD_IP"
echo "worker=$WORKER_NODE $WORKER_IP"
echo "master=$HEAD_IP:$MASTER_PORT"
echo "if=$CONTROL_IF"
echo "gloo_range=$GLOO_PORT_MIN-$GLOO_PORT_MAX"
echo "============================================================"

[[ -d "$IMAGE" ]] || {
  echo "ERROR: patched sandbox not found: $IMAGE"
  exit 2
}

ssh "${SSH_OPTS[@]}" "$WORKER_NODE" "test -d '$IMAGE'"

TEST_PY="$PWD/logs/gloo-smoke-${SLURM_JOB_ID}.py"

cat >"$TEST_PY" <<'PY'
import os
import socket
import sys
import time

import torch
import torch.distributed as dist

rank = int(sys.argv[1])

print(
    f"START host={socket.getfqdn()} rank={rank} "
    f"IF={os.getenv('GLOO_SOCKET_IFNAME')} "
    f"MIN={os.getenv('GLOO_PORT_MIN')} "
    f"MAX={os.getenv('GLOO_PORT_MAX')}",
    flush=True,
)

dist.init_process_group(
    backend="gloo",
    init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
    rank=rank,
    world_size=2,
)

x = torch.tensor([rank + 1], dtype=torch.int64)
dist.all_reduce(x)

print(
    f"GLOO_OK host={socket.getfqdn()} "
    f"rank={rank} all_reduce={x.item()}",
    flush=True,
)

# Keep listeners around briefly so ss can observe them.
dist.barrier()
time.sleep(5)
dist.destroy_process_group()
PY

HEAD_PID=""
WORKER_PID=""

cleanup() {
  rc=$?
  trap - EXIT INT TERM
  set +e

  [[ -n "$HEAD_PID" ]] && kill "$HEAD_PID" 2>/dev/null || true
  [[ -n "$WORKER_PID" ]] && kill "$WORKER_PID" 2>/dev/null || true

  [[ -n "$HEAD_PID" ]] && wait "$HEAD_PID" 2>/dev/null || true
  [[ -n "$WORKER_PID" ]] && wait "$WORKER_PID" 2>/dev/null || true

  exit "$rc"
}
trap cleanup EXIT INT TERM

echo
echo "=== Start rank 0 / HEAD ==="

apptainer exec \
  --cleanenv \
  --env "MASTER_ADDR=$HEAD_IP" \
  --env "MASTER_PORT=$MASTER_PORT" \
  --env "GLOO_SOCKET_IFNAME=$CONTROL_IF" \
  --env "GLOO_PORT_MIN=$GLOO_PORT_MIN" \
  --env "GLOO_PORT_MAX=$GLOO_PORT_MAX" \
  --bind "$PWD:$PWD" \
  "$IMAGE" \
  python3 "$TEST_PY" 0 \
  >"logs/gloo-rank0-${SLURM_JOB_ID}.log" 2>&1 &

HEAD_PID=$!

sleep 2

echo
echo "=== Start rank 1 / WORKER ==="

ssh "${SSH_OPTS[@]}" "$WORKER_NODE" \
  "cd '$PWD' && \
   apptainer exec \
     --cleanenv \
     --env 'MASTER_ADDR=$HEAD_IP' \
     --env 'MASTER_PORT=$MASTER_PORT' \
     --env 'GLOO_SOCKET_IFNAME=$CONTROL_IF' \
     --env 'GLOO_PORT_MIN=$GLOO_PORT_MIN' \
     --env 'GLOO_PORT_MAX=$GLOO_PORT_MAX' \
     --bind '$PWD:$PWD' \
     '$IMAGE' \
     python3 '$TEST_PY' 1" \
  >"logs/gloo-rank1-${SLURM_JOB_ID}.log" 2>&1 &

WORKER_PID=$!

sleep 3

echo
echo "=== TCP listeners while Gloo is alive ==="

echo "--- HEAD ---"
ss -lntp 2>/dev/null \
  | awk '$4 ~ /:50[0-9][0-9][0-9]$/ {print}' \
  || true

echo "--- WORKER ---"
ssh "${SSH_OPTS[@]}" "$WORKER_NODE" \
  "ss -lntp 2>/dev/null | awk '\$4 ~ /:50[0-9][0-9][0-9]\$/ {print}'" \
  || true

set +e
wait "$HEAD_PID"
R0=$?
wait "$WORKER_PID"
R1=$?
set -e

echo
echo "=== RANK 0 LOG ==="
cat "logs/gloo-rank0-${SLURM_JOB_ID}.log"

echo
echo "=== RANK 1 LOG ==="
cat "logs/gloo-rank1-${SLURM_JOB_ID}.log"

if [[ "$R0" -ne 0 || "$R1" -ne 0 ]]; then
  echo "ERROR: rank0=$R0 rank1=$R1"
  exit 10
fi

grep -q 'GLOO_OK' "logs/gloo-rank0-${SLURM_JOB_ID}.log"
grep -q 'GLOO_OK' "logs/gloo-rank1-${SLURM_JOB_ID}.log"

grep -q 'all_reduce=3' "logs/gloo-rank0-${SLURM_JOB_ID}.log"
grep -q 'all_reduce=3' "logs/gloo-rank1-${SLURM_JOB_ID}.log"

echo
echo "============================================================"
echo " GLOO 2-NODE SMOKE TEST PASSED"
echo " all_reduce=3"
echo " configured listener range=$GLOO_PORT_MIN-$GLOO_PORT_MAX"
echo "============================================================"
