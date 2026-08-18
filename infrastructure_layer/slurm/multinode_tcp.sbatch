#!/bin/bash
#SBATCH --job-name=vllm-2n8g-tcp
#SBATCH --partition=gpu-queue
#SBATCH --qos=gpu-q
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/vllm-multinode-tcp-%j.out
#SBATCH --error=logs/vllm-multinode-tcp-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

IMAGE="${IMAGE:-$PWD/build/vllm.sif}"
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:-$HOME/.cache}/llm-serving}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

TP="${TP:-4}"
PP="${PP:-2}"
DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

RAY_ONLY="${RAY_ONLY:-0}"
NCCL_DEBUG="${NCCL_DEBUG:-TRACE}"
CONTROL_IF="${CONTROL_IF:-ens11f0}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"

# Admin opened inter-node TCP 50000-51000.
# Use a 50-port job block, always within 50000-50999.
PORT_SLOT=$((SLURM_JOB_ID % 20))
PORT_BASE=$((50000 + PORT_SLOT * 50))

RAY_GCS_PORT=$((PORT_BASE + 0))
RAY_NODE_MANAGER_PORT=$((PORT_BASE + 1))
RAY_OBJECT_MANAGER_PORT=$((PORT_BASE + 2))
RAY_RUNTIME_ENV_PORT=$((PORT_BASE + 3))
RAY_DASH_AGENT_GRPC_PORT=$((PORT_BASE + 4))
RAY_DASH_AGENT_HTTP_PORT=$((PORT_BASE + 5))
RAY_METRICS_PORT=$((PORT_BASE + 6))
RAY_DASHBOARD_PORT=$((PORT_BASE + 8))
RAY_MIN_WORKER_PORT=$((PORT_BASE + 10))
RAY_MAX_WORKER_PORT=$((PORT_BASE + 34))
TCP_TEST_PORT=$((PORT_BASE + 39))
VLLM_INTERNAL_PORT=$((PORT_BASE + 40))

mkdir -p logs
mkdir -p "$CACHE_ROOT"/{huggingface,vllm,xdg}

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ "${#NODES[@]}" -eq 2 ]] || { echo "ERROR: need exactly 2 nodes"; exit 2; }

HEAD_NODE="${NODES[0]}"
WORKER_NODE="${NODES[1]}"

HEAD_IP="$(getent ahostsv4 "$HEAD_NODE" | awk 'NR==1 {print $1}')"
WORKER_IP="$(getent ahostsv4 "$WORKER_NODE" | awk 'NR==1 {print $1}')"

[[ -n "$HEAD_IP" && -n "$WORKER_IP" ]] || { echo "ERROR: cannot resolve node IP"; exit 2; }
[[ $((TP * PP)) -eq 8 ]] || { echo "ERROR: TP*PP must equal 8"; exit 2; }
[[ -f "$IMAGE" ]] || { echo "ERROR: image missing: $IMAGE"; exit 2; }

MODEL_CACHE_DIR="$CACHE_ROOT/huggingface/hub/models--${MODEL//\//--}"
[[ -d "$MODEL_CACHE_DIR" ]] || { echo "ERROR: cache missing: $MODEL_CACHE_DIR"; exit 3; }

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout="$SSH_CONNECT_TIMEOUT"
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=3
)

echo "============================================================"
echo "PROFILE=TCP / SOCKET"
date
echo "job_id=$SLURM_JOB_ID"
echo "head=$HEAD_NODE $HEAD_IP"
echo "worker=$WORKER_NODE $WORKER_IP"
echo "model=$MODEL"
echo "TP=$TP PP=$PP"
echo "control_if=$CONTROL_IF"
echo "port_base=$PORT_BASE"
echo "ray_gcs=$RAY_GCS_PORT"
echo "ray_workers=$RAY_MIN_WORKER_PORT-$RAY_MAX_WORKER_PORT"
echo "vllm_internal=$VLLM_INTERNAL_PORT"
echo "api_port=$PORT"
echo "============================================================"

echo
echo "=== SSH + GPU + cache preflight ==="
ssh "${SSH_OPTS[@]}" "$WORKER_NODE" 'true'
test "$(nvidia-smi -L | wc -l)" -eq 4
du -sh "$MODEL_CACHE_DIR"

ssh "${SSH_OPTS[@]}" "$WORKER_NODE" \
  "MODEL_CACHE_DIR='$MODEL_CACHE_DIR' bash -s" <<'REMOTE'
set -euo pipefail
hostname -f
test "$(nvidia-smi -L | wc -l)" -eq 4
nvidia-smi -L
du -sh "$MODEL_CACHE_DIR"
REMOTE

echo
echo "=== Ray fixed-port CLI preflight ==="
RAY_HELP="$(apptainer exec --cleanenv --bind "$CACHE_ROOT:/cache" "$IMAGE" ray start --help)"
for flag in \
  --node-manager-port \
  --object-manager-port \
  --runtime-env-agent-port \
  --dashboard-agent-grpc-port \
  --dashboard-agent-listen-port \
  --metrics-export-port \
  --min-worker-port \
  --max-worker-port; do
  grep -q -- "$flag" <<<"$RAY_HELP" || {
    echo "ERROR: Ray does not support $flag"
    exit 4
  }
done

echo
echo "=== TCP firewall test WORKER -> HEAD on $TCP_TEST_PORT ==="
python3 - "$HEAD_IP" "$TCP_TEST_PORT" <<'PY' &
import socket, sys
ip, port = sys.argv[1], int(sys.argv[2])
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((ip, port))
s.listen(1)
s.settimeout(20)
try:
    c, _ = s.accept()
    c.close()
finally:
    s.close()
PY
TEST_PID=$!
sleep 1
if ! ssh "${SSH_OPTS[@]}" "$WORKER_NODE" \
    "timeout 5 bash -c '</dev/tcp/$HEAD_IP/$TCP_TEST_PORT'"; then
  kill "$TEST_PID" 2>/dev/null || true
  wait "$TEST_PID" 2>/dev/null || true
  echo "ERROR: opened TCP range is not reachable worker -> head"
  exit 5
fi
wait "$TEST_PID"
echo "TCP range test OK"

echo
echo "=== Reverse TCP firewall test HEAD -> WORKER on $TCP_TEST_PORT ==="

ssh "${SSH_OPTS[@]}" "$WORKER_NODE" \
  "python3 - '$WORKER_IP' '$TCP_TEST_PORT' <<'PY'
import socket, sys
ip, port = sys.argv[1], int(sys.argv[2])
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((ip, port))
s.listen(1)
s.settimeout(20)
try:
    c, _ = s.accept()
    c.close()
finally:
    s.close()
PY" &
REVERSE_TEST_PID=$!

sleep 1

if ! timeout 5 bash -c "</dev/tcp/$WORKER_IP/$TCP_TEST_PORT"; then
    kill "$REVERSE_TEST_PID" 2>/dev/null || true
    wait "$REVERSE_TEST_PID" 2>/dev/null || true
    echo "ERROR: HEAD cannot TCP-connect to WORKER inside opened range"
    exit 5
fi

wait "$REVERSE_TEST_PID"
echo "Reverse TCP range test OK"

echo
echo "=== Kernel ephemeral TCP port range ==="
echo -n "HEAD   : "
cat /proc/sys/net/ipv4/ip_local_port_range
echo -n "WORKER : "
ssh "${SSH_OPTS[@]}" "$WORKER_NODE" \
  'cat /proc/sys/net/ipv4/ip_local_port_range'

echo
echo "NOTE: Ray is pinned to 50000-50999, but PyTorch Gloo may create"
echo "additional full-mesh TCP sockets using OS-selected ports."
echo "GLOO_SOCKET_IFNAME is forced to $CONTROL_IF in this profile."

echo
echo "=== Transport setup: TCP / Socket ==="

HEAD_NCCL_ENV=(
  --env "NCCL_DEBUG=$NCCL_DEBUG"
  --env "NCCL_DEBUG_SUBSYS=INIT,NET"
  --env "NCCL_NET=Socket"
  --env "NCCL_IB_DISABLE=1"
  --env "NCCL_SOCKET_IFNAME==$CONTROL_IF"
  --env "NCCL_SOCKET_FAMILY=AF_INET"
  --env "GLOO_SOCKET_IFNAME=$CONTROL_IF"
)

RAY_HEAD_PID=""
RAY_WORKER_SSH_PID=""

cleanup() {
  rc=$?
  trap - EXIT INT TERM
  set +e

  [[ -n "${RAY_WORKER_SSH_PID:-}" ]] && kill "$RAY_WORKER_SSH_PID" 2>/dev/null || true
  [[ -n "${RAY_HEAD_PID:-}" ]] && kill "$RAY_HEAD_PID" 2>/dev/null || true

  ssh "${SSH_OPTS[@]}" "$WORKER_NODE" \
    "apptainer exec --cleanenv --bind '$CACHE_ROOT:/cache' '$IMAGE' ray stop --force" \
    >/dev/null 2>&1 || true

  apptainer exec --cleanenv --bind "$CACHE_ROOT:/cache" "$IMAGE" \
    ray stop --force >/dev/null 2>&1 || true

  [[ -n "${RAY_WORKER_SSH_PID:-}" ]] && wait "$RAY_WORKER_SSH_PID" 2>/dev/null || true
  [[ -n "${RAY_HEAD_PID:-}" ]] && wait "$RAY_HEAD_PID" 2>/dev/null || true
  exit "$rc"
}
trap cleanup EXIT INT TERM

echo
echo "=== Starting Ray HEAD ==="

HEAD_ENV=(
  --env "CUDA_VISIBLE_DEVICES=0,1,2,3"
  --env "HF_HUB_OFFLINE=1"
  --env "TRANSFORMERS_OFFLINE=1"
  --env "VLLM_HOST_IP=$HEAD_IP"
  --env "VLLM_PORT=$VLLM_INTERNAL_PORT"
  "${HEAD_NCCL_ENV[@]}"
)

apptainer exec \
  --nv \
  --cleanenv \
  "${HEAD_ENV[@]}" \
  --bind "$CACHE_ROOT:/cache" \
  "$IMAGE" \
  ray start \
    --head \
    --node-ip-address="$HEAD_IP" \
    --port="$RAY_GCS_PORT" \
    --node-manager-port="$RAY_NODE_MANAGER_PORT" \
    --object-manager-port="$RAY_OBJECT_MANAGER_PORT" \
    --runtime-env-agent-port="$RAY_RUNTIME_ENV_PORT" \
    --dashboard-agent-grpc-port="$RAY_DASH_AGENT_GRPC_PORT" \
    --dashboard-agent-listen-port="$RAY_DASH_AGENT_HTTP_PORT" \
    --metrics-export-port="$RAY_METRICS_PORT" \
    --dashboard-port="$RAY_DASHBOARD_PORT" \
    --min-worker-port="$RAY_MIN_WORKER_PORT" \
    --max-worker-port="$RAY_MAX_WORKER_PORT" \
    --num-cpus=16 \
    --num-gpus=4 \
    --include-dashboard=false \
    --disable-usage-stats \
    --block \
  >"logs/ray-head-tcp-${SLURM_JOB_ID}.log" 2>&1 &

RAY_HEAD_PID=$!

HEAD_READY=0
for attempt in $(seq 1 45); do
  if timeout 2 bash -c "</dev/tcp/$HEAD_IP/$RAY_GCS_PORT" 2>/dev/null; then
    HEAD_READY=1
    break
  fi
  if ! kill -0 "$RAY_HEAD_PID" 2>/dev/null; then
    cat "logs/ray-head-tcp-${SLURM_JOB_ID}.log" || true
    exit 6
  fi
  sleep 2
done
[[ "$HEAD_READY" -eq 1 ]] || { echo "ERROR: Ray HEAD not reachable"; exit 6; }

echo "Ray HEAD reachable: $HEAD_IP:$RAY_GCS_PORT"

echo
echo "=== Starting Ray WORKER via SSH ==="

ssh "${SSH_OPTS[@]}" "$WORKER_NODE" \
  "HEAD_IP='$HEAD_IP' \
   WORKER_IP='$WORKER_IP' \
   IMAGE='$IMAGE' \
   CACHE_ROOT='$CACHE_ROOT' \
   CONTROL_IF='$CONTROL_IF' \
   NCCL_DEBUG='$NCCL_DEBUG' \
   VLLM_INTERNAL_PORT='$VLLM_INTERNAL_PORT' \
   RAY_GCS_PORT='$RAY_GCS_PORT' \
   RAY_NODE_MANAGER_PORT='$RAY_NODE_MANAGER_PORT' \
   RAY_OBJECT_MANAGER_PORT='$RAY_OBJECT_MANAGER_PORT' \
   RAY_RUNTIME_ENV_PORT='$RAY_RUNTIME_ENV_PORT' \
   RAY_DASH_AGENT_GRPC_PORT='$RAY_DASH_AGENT_GRPC_PORT' \
   RAY_DASH_AGENT_HTTP_PORT='$RAY_DASH_AGENT_HTTP_PORT' \
   RAY_METRICS_PORT='$RAY_METRICS_PORT' \
   RAY_MIN_WORKER_PORT='$RAY_MIN_WORKER_PORT' \
   RAY_MAX_WORKER_PORT='$RAY_MAX_WORKER_PORT' \
    \
   bash -s" \
  >"logs/ray-worker-tcp-${SLURM_JOB_ID}.log" 2>&1 <<'REMOTE' &

set -euo pipefail

cleanup_remote() {
  set +e
  apptainer exec --cleanenv --bind "$CACHE_ROOT:/cache" "$IMAGE" \
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup_remote EXIT INT TERM

REMOTE_NCCL_ENV=(
  --env "NCCL_DEBUG=$NCCL_DEBUG"
  --env "NCCL_DEBUG_SUBSYS=INIT,NET"
  --env "NCCL_NET=Socket"
  --env "NCCL_IB_DISABLE=1"
  --env "NCCL_SOCKET_IFNAME==$CONTROL_IF"
  --env "NCCL_SOCKET_FAMILY=AF_INET"
  --env "GLOO_SOCKET_IFNAME=$CONTROL_IF"
)

REMOTE_ENV=(
  --env "CUDA_VISIBLE_DEVICES=0,1,2,3"
  --env "HF_HUB_OFFLINE=1"
  --env "TRANSFORMERS_OFFLINE=1"
  --env "VLLM_HOST_IP=$WORKER_IP"
  --env "VLLM_PORT=$VLLM_INTERNAL_PORT"
  "${REMOTE_NCCL_ENV[@]}"
)

apptainer exec \
  --nv \
  --cleanenv \
  "${REMOTE_ENV[@]}" \
  --bind "$CACHE_ROOT:/cache" \
  "$IMAGE" \
  ray start \
    --address="$HEAD_IP:$RAY_GCS_PORT" \
    --node-ip-address="$WORKER_IP" \
    --node-manager-port="$RAY_NODE_MANAGER_PORT" \
    --object-manager-port="$RAY_OBJECT_MANAGER_PORT" \
    --runtime-env-agent-port="$RAY_RUNTIME_ENV_PORT" \
    --dashboard-agent-grpc-port="$RAY_DASH_AGENT_GRPC_PORT" \
    --dashboard-agent-listen-port="$RAY_DASH_AGENT_HTTP_PORT" \
    --metrics-export-port="$RAY_METRICS_PORT" \
    --min-worker-port="$RAY_MIN_WORKER_PORT" \
    --max-worker-port="$RAY_MAX_WORKER_PORT" \
    --num-cpus=16 \
    --num-gpus=4 \
    --disable-usage-stats \
    --block
REMOTE

RAY_WORKER_SSH_PID=$!

echo
echo "=== Waiting for Ray: 2 nodes / 8 GPUs ==="

RAY_READY=0
for attempt in $(seq 1 60); do
  RESULT="$(
    timeout 20s apptainer exec \
      --cleanenv \
      --bind "$CACHE_ROOT:/cache" \
      "$IMAGE" \
      python3 - "$HEAD_IP:$RAY_GCS_PORT" <<'PY' 2>/dev/null || true
import sys, ray
try:
    ray.init(address=sys.argv[1], logging_level="ERROR")
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    r = ray.cluster_resources()
    print("NODES=%d GPUS=%d CPUS=%d" %
          (len(nodes), int(r.get("GPU", 0)), int(r.get("CPU", 0))))
    for n in sorted(nodes, key=lambda x: x.get("NodeManagerAddress", "")):
        rr = n.get("Resources", {})
        print("NODE", n.get("NodeManagerAddress"),
              "GPU=" + str(rr.get("GPU", 0)),
              "CPU=" + str(rr.get("CPU", 0)))
    ray.shutdown()
except Exception as e:
    print("NOT_READY", type(e).__name__, str(e))
PY
  )"

  echo "[ray wait $attempt/60]"
  echo "$RESULT"

  if grep -q "NODES=2 GPUS=8" <<<"$RESULT"; then
    RAY_READY=1
    break
  fi

  kill -0 "$RAY_HEAD_PID" 2>/dev/null || {
    tail -150 "logs/ray-head-tcp-${SLURM_JOB_ID}.log" || true
    exit 7
  }
  kill -0 "$RAY_WORKER_SSH_PID" 2>/dev/null || {
    tail -150 "logs/ray-worker-tcp-${SLURM_JOB_ID}.log" || true
    exit 7
  }
  sleep 5
done

[[ "$RAY_READY" -eq 1 ]] || {
  echo "ERROR: Ray never reached 2 nodes / 8 GPUs"
  tail -150 "logs/ray-head-tcp-${SLURM_JOB_ID}.log" || true
  tail -150 "logs/ray-worker-tcp-${SLURM_JOB_ID}.log" || true
  exit 7
}

echo "============================================================"
echo "RAY CLUSTER READY: 2 NODES / 8 GPUS"
echo "============================================================"

if [[ "$RAY_ONLY" == "1" ]]; then
  echo "RAY_ONLY=1 SUCCESS; keeping cluster alive for 60 seconds."
  sleep 60
  exit 0
fi

echo
echo "=== Starting vLLM ==="
echo "API: http://$HEAD_IP:$PORT"
echo "model=$MODEL TP=$TP PP=$PP"
echo "Expected NCCL transport: NET/Socket"

VLLM_ENV=(
  --env "CUDA_VISIBLE_DEVICES=0,1,2,3"
  --env "HF_HUB_OFFLINE=1"
  --env "TRANSFORMERS_OFFLINE=1"
  --env "RAY_ADDRESS=$HEAD_IP:$RAY_GCS_PORT"
  --env "VLLM_HOST_IP=$HEAD_IP"
  --env "VLLM_PORT=$VLLM_INTERNAL_PORT"
  "${HEAD_NCCL_ENV[@]}"
)

apptainer run \
  --nv \
  --cleanenv \
  "${VLLM_ENV[@]}" \
  --bind "$CACHE_ROOT:/cache" \
  "$IMAGE" \
  serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --distributed-executor-backend ray \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enforce-eager
