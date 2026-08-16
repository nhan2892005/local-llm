#!/bin/bash
#SBATCH --job-name=vllm-qwen3-235b-2n8g
#SBATCH --partition=gpu-queue
#SBATCH --qos=gpu-q
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/vllm-multinode-%j.out
#SBATCH --error=logs/vllm-multinode-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

IMAGE="${IMAGE:-$PWD/build/vllm.sif}"
MODEL="${MODEL:-Qwen/Qwen3-235B-A22B-GPTQ-Int4}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:-$HOME/.cache}/llm-serving}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Two nodes x four GPUs.
TP="${TP:-4}"
PP="${PP:-2}"

# V100: FP16 activations. GPTQ weights remain INT4.
DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

RAY_PORT="${RAY_PORT:-6379}"
NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

# Optional: set this after inspecting `ip -br addr` / NCCL logs.
# Example: sbatch --export=ALL,NCCL_SOCKET_IFNAME=eno1 ...
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"

mkdir -p "$CACHE_ROOT"/{huggingface,vllm,xdg}

if [[ ! -f "$IMAGE" ]]; then
    echo "ERROR: image not found: $IMAGE"
    exit 2
fi

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ "${#NODES[@]}" -ne 2 ]]; then
    echo "ERROR: expected exactly 2 nodes, got ${#NODES[@]}: ${NODES[*]}"
    exit 2
fi

HEAD_NODE="${NODES[0]}"
WORKER_NODE="${NODES[1]}"

HEAD_IP="$(
    srun --nodes=1 --ntasks=1 --nodelist="$HEAD_NODE" \
      bash -lc 'getent ahostsv4 "$(hostname -f)" | awk "NR==1 {print \$1}"'
)"
WORKER_IP="$(
    srun --nodes=1 --ntasks=1 --nodelist="$WORKER_NODE" \
      bash -lc 'getent ahostsv4 "$(hostname -f)" | awk "NR==1 {print \$1}"'
)"

if [[ -z "$HEAD_IP" || -z "$WORKER_IP" ]]; then
    echo "ERROR: failed to resolve node IPs"
    exit 2
fi

echo "============================================================"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "nodes=${NODES[*]}"
echo "head_node=$HEAD_NODE"
echo "head_ip=$HEAD_IP"
echo "worker_node=$WORKER_NODE"
echo "worker_ip=$WORKER_IP"
echo "image=$IMAGE"
echo "model=$MODEL"
echo "TP=$TP"
echo "PP=$PP"
echo "world_size=$((TP * PP))"
echo "dtype=$DTYPE"
echo "max_model_len=$MAX_MODEL_LEN"
echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
echo "cache_root=$CACHE_ROOT"
echo "ray_address=$HEAD_IP:$RAY_PORT"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<auto>}"
echo "============================================================"

if [[ $((TP * PP)) -ne 8 ]]; then
    echo "ERROR: TP*PP must equal 8 for this 2-node x 4-GPU job."
    exit 2
fi

echo
echo "=== GPUs on both nodes ==="
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  bash -lc '
    echo "--- $(hostname -f) ---"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
  '

echo
echo "=== Local GPU topology on both nodes ==="
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  bash -lc '
    echo "--- $(hostname -f) ---"
    nvidia-smi topo -m || true
  '

echo
echo "=== Network summary ==="
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  bash -lc '
    echo "--- $(hostname -f) ---"
    hostname -I || true
    ip -br addr 2>/dev/null || true
    command -v ibstat >/dev/null 2>&1 && ibstat || true
  '

echo
echo "=== Preflight: Ray + vLLM available in SIF on both nodes ==="
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  apptainer exec \
    --nv \
    --cleanenv \
    --bind "$CACHE_ROOT:/cache" \
    "$IMAGE" \
    python3 -c '
import socket
import ray
import vllm
print(socket.getfqdn(), "ray=", ray.__version__, "vllm=", vllm.__version__)
'

echo
echo "=== Check shared model cache ==="
MODEL_CACHE_DIR="$CACHE_ROOT/huggingface/hub/models--${MODEL//\//--}"
if [[ ! -d "$MODEL_CACHE_DIR" ]]; then
    echo "ERROR: model is not present in shared HF cache:"
    echo "  $MODEL_CACHE_DIR"
    echo "Download it on compute first; do not hold 8 GPUs while downloading."
    exit 3
fi
du -sh "$MODEL_CACHE_DIR" || true

CONFIG_COUNT="$(
    find "$MODEL_CACHE_DIR/snapshots" -maxdepth 2 -name config.json 2>/dev/null | wc -l
)"
if [[ "$CONFIG_COUNT" -lt 1 ]]; then
    echo "ERROR: no snapshot config.json found under $MODEL_CACHE_DIR"
    exit 3
fi

echo
echo "=== Starting Ray cluster ==="

export HEAD_NODE HEAD_IP RAY_PORT IMAGE CACHE_ROOT NCCL_DEBUG NCCL_SOCKET_IFNAME

srun \
  --nodes=2 \
  --ntasks=2 \
  --ntasks-per-node=1 \
  --cpus-per-task=16 \
  --gres=gpu:4 \
  bash -lc '
    set -euo pipefail

    NODE_FQDN="$(hostname -f)"
    NODE_IP="$(getent ahostsv4 "$NODE_FQDN" | awk "NR==1 {print \$1}")"

    echo "[ray] node=$NODE_FQDN ip=$NODE_IP CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

    APPTAINER_ENV=(
      --env "HF_HUB_OFFLINE=1"
      --env "TRANSFORMERS_OFFLINE=1"
      --env "VLLM_HOST_IP=$NODE_IP"
      --env "NCCL_DEBUG=$NCCL_DEBUG"
    )
    if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
      APPTAINER_ENV+=(--env "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME")
    fi

    if [[ "$(hostname -s)" == "${HEAD_NODE%%.*}" ]]; then
      echo "[ray] starting HEAD at $HEAD_IP:$RAY_PORT"
      exec apptainer exec \
        --nv \
        --cleanenv \
        "${APPTAINER_ENV[@]}" \
        --bind "$CACHE_ROOT:/cache" \
        "$IMAGE" \
        ray start \
          --head \
          --node-ip-address="$HEAD_IP" \
          --port="$RAY_PORT" \
          --num-gpus=4 \
          --num-cpus=16 \
          --include-dashboard=false \
          --block
    else
      echo "[ray] starting WORKER; head=$HEAD_IP:$RAY_PORT"
      exec apptainer exec \
        --nv \
        --cleanenv \
        "${APPTAINER_ENV[@]}" \
        --bind "$CACHE_ROOT:/cache" \
        "$IMAGE" \
        ray start \
          --address="$HEAD_IP:$RAY_PORT" \
          --node-ip-address="$NODE_IP" \
          --num-gpus=4 \
          --num-cpus=16 \
          --block
    fi
  ' &

RAY_SRUN_PID=$!

cleanup() {
    rc=$?
    echo
    echo "=== Cleanup ==="
    if [[ -n "${RAY_SRUN_PID:-}" ]] && kill -0 "$RAY_SRUN_PID" 2>/dev/null; then
        echo "Stopping Ray Slurm step PID=$RAY_SRUN_PID"
        kill "$RAY_SRUN_PID" 2>/dev/null || true
        wait "$RAY_SRUN_PID" 2>/dev/null || true
    fi
    exit "$rc"
}
trap cleanup EXIT INT TERM

echo
echo "=== Waiting for Ray: 2 nodes / 8 GPUs ==="
RAY_READY=0

for attempt in $(seq 1 60); do
    RESULT="$(
      srun \
        --overlap \
        --nodes=1 \
        --ntasks=1 \
        --nodelist="$HEAD_NODE" \
        apptainer exec \
          --nv \
          --cleanenv \
          --env "RAY_ADDRESS=$HEAD_IP:$RAY_PORT" \
          --env "VLLM_HOST_IP=$HEAD_IP" \
          --bind "$CACHE_ROOT:/cache" \
          "$IMAGE" \
          python3 - "$HEAD_IP:$RAY_PORT" <<'PY' 2>/dev/null || true
import sys
import ray

address = sys.argv[1]
try:
    ray.init(address=address, logging_level="ERROR")
    resources = ray.cluster_resources()
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    print(f"NODES={len(nodes)} GPUS={int(resources.get('GPU', 0))}")
    ray.shutdown()
except Exception as exc:
    print(f"NOT_READY {type(exc).__name__}: {exc}")
PY
    )"

    echo "[ray wait $attempt/60] $RESULT"
    if grep -q "NODES=2 GPUS=8" <<<"$RESULT"; then
        RAY_READY=1
        break
    fi
    sleep 5
done

if [[ "$RAY_READY" -ne 1 ]]; then
    echo "ERROR: Ray cluster did not become ready with 2 nodes / 8 GPUs."
    exit 4
fi

echo
echo "=== Ray cluster ready ==="
srun \
  --overlap \
  --nodes=1 \
  --ntasks=1 \
  --nodelist="$HEAD_NODE" \
  apptainer exec \
    --nv \
    --cleanenv \
    --env "RAY_ADDRESS=$HEAD_IP:$RAY_PORT" \
    --bind "$CACHE_ROOT:/cache" \
    "$IMAGE" \
    ray status --address="$HEAD_IP:$RAY_PORT" || true

echo
echo "============================================================"
echo "Starting vLLM API server"
echo "head node : $HEAD_NODE"
echo "API       : http://$HEAD_IP:$PORT"
echo "test      : ssh $HEAD_NODE 'curl http://127.0.0.1:$PORT/v1/models'"
echo "============================================================"
echo

VLLM_ENV=(
  --env "HF_HUB_OFFLINE=1"
  --env "TRANSFORMERS_OFFLINE=1"
  --env "RAY_ADDRESS=$HEAD_IP:$RAY_PORT"
  --env "VLLM_HOST_IP=$HEAD_IP"
  --env "NCCL_DEBUG=$NCCL_DEBUG"
)
if [[ -n "$NCCL_SOCKET_IFNAME" ]]; then
    VLLM_ENV+=(--env "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME")
fi

# One API/driver process only; Ray places 8 model workers across both nodes.
srun \
  --overlap \
  --nodes=1 \
  --ntasks=1 \
  --nodelist="$HEAD_NODE" \
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
