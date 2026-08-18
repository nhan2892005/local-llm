#!/bin/bash
#SBATCH --job-name=vllm-2n8g-ssh
#SBATCH --partition=gpu-queue
#SBATCH --qos=gpu-q
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/vllm-multinode-%j.out
#SBATCH --error=logs/vllm-multinode-%j.err

set -euo pipefail

# ============================================================
# vLLM multi-node for this cluster:
#
#   Slurm:
#     - reserves 2 nodes
#     - reserves 4 GPUs/node
#
#   Process launch:
#     - batch shell runs on HEAD node
#     - Ray HEAD runs directly on the batch/head node
#     - Ray WORKER runs on the second allocated node via SSH
#     - NO child srun is used
#
# This is a cluster-specific workaround because child srun steps
# launched from the batch shell were hanging when targeting node 2.
#
# Default topology:
#     2 nodes x 4 V100 = 8 GPUs
#     TP=4 (inside each node)
#     PP=2 (across the two nodes)
#
# Default smoke-test model:
#     Qwen/Qwen2.5-32B-Instruct
#
# Ray-only test:
#   sbatch --export=ALL,RAY_ONLY=1 slurm/multi_node_deploy.sbatch
#
# Full model:
#   sbatch slurm/multi_node_deploy.sbatch
#
# Override model:
#   sbatch \
#     --export=ALL,MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
#     slurm/multi_node_deploy.sbatch
# ============================================================

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
RAY_PORT="${RAY_PORT:-$((50000 + SLURM_JOB_ID % 1000))}"

NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"

SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"

mkdir -p logs
mkdir -p "$CACHE_ROOT"/{huggingface,vllm,xdg}

echo "============================================================"
echo " MULTINODE SSH/RAY JOB STARTED"
echo "============================================================"
date
echo "job_id=$SLURM_JOB_ID"
echo "batch_host=$(hostname -f)"
echo "nodelist=$SLURM_JOB_NODELIST"
echo

if [[ ! -f "$IMAGE" ]]; then
    echo "ERROR: image not found: $IMAGE"
    exit 2
fi

if [[ $((TP * PP)) -ne 8 ]]; then
    echo "ERROR: TP*PP must equal 8 for this 2-node x 4-GPU job."
    echo "TP=$TP PP=$PP"
    exit 2
fi

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")

if [[ "${#NODES[@]}" -ne 2 ]]; then
    echo "ERROR: expected exactly 2 allocated nodes."
    printf 'allocated nodes: %s\n' "${NODES[*]}"
    exit 2
fi

HEAD_NODE="${NODES[0]}"
WORKER_NODE="${NODES[1]}"

HEAD_IP="$(getent ahostsv4 "$HEAD_NODE" | awk 'NR==1 {print $1}')"
WORKER_IP="$(getent ahostsv4 "$WORKER_NODE" | awk 'NR==1 {print $1}')"

if [[ -z "$HEAD_IP" || -z "$WORKER_IP" ]]; then
    echo "ERROR: failed to resolve node IPs."
    echo "HEAD_NODE=$HEAD_NODE HEAD_IP=$HEAD_IP"
    echo "WORKER_NODE=$WORKER_NODE WORKER_IP=$WORKER_IP"
    exit 2
fi

MODEL_CACHE_NAME="models--${MODEL//\//--}"
MODEL_CACHE_DIR="$CACHE_ROOT/huggingface/hub/$MODEL_CACHE_NAME"

echo "============================================================"
echo "head_node=$HEAD_NODE"
echo "head_ip=$HEAD_IP"
echo "worker_node=$WORKER_NODE"
echo "worker_ip=$WORKER_IP"
echo "image=$IMAGE"
echo "model=$MODEL"
echo "model_cache=$MODEL_CACHE_DIR"
echo "TP=$TP"
echo "PP=$PP"
echo "world_size=$((TP * PP))"
echo "dtype=$DTYPE"
echo "max_model_len=$MAX_MODEL_LEN"
echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
echo "ray_address=$HEAD_IP:$RAY_PORT"
echo "NCCL_DEBUG=$NCCL_DEBUG"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<auto>}"
echo "============================================================"

# ============================================================
# Model cache
# ============================================================

echo
echo "=== Model cache preflight on HEAD ==="

if [[ ! -d "$MODEL_CACHE_DIR" ]]; then
    echo "ERROR: model cache not found:"
    echo "  $MODEL_CACHE_DIR"
    exit 3
fi

du -sh "$MODEL_CACHE_DIR" || true

if ! find "$MODEL_CACHE_DIR/snapshots" \
        -maxdepth 2 \
        -name config.json \
        -print -quit 2>/dev/null | grep -q .; then
    echo "ERROR: no config.json found under model snapshots."
    exit 3
fi

# ============================================================
# SSH preflight
# ============================================================

SSH_OPTS=(
    -o BatchMode=yes
    -o ConnectTimeout="$SSH_CONNECT_TIMEOUT"
    -o ServerAliveInterval=15
    -o ServerAliveCountMax=3
)

echo
echo "=== SSH preflight to allocated worker ==="
echo "worker=$WORKER_NODE"

if ! ssh "${SSH_OPTS[@]}" "$WORKER_NODE" 'true'; then
    echo "ERROR: non-interactive SSH from $HEAD_NODE to $WORKER_NODE failed."
    echo "The batch job cannot prompt for an SSH password."
    echo "Make sure SSH keys / host trust work between compute nodes."
    exit 4
fi

echo "SSH to worker: OK"

# ============================================================
# HEAD preflight
# ============================================================

echo
echo "============================================================"
echo " HEAD PREFLIGHT: $HEAD_NODE"
echo "============================================================"

echo "node=$(hostname -f)"
echo "ip=$(hostname -I)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

HEAD_GPU_COUNT="$(nvidia-smi -L | wc -l)"
echo "visible_gpu_count=$HEAD_GPU_COUNT"

if [[ "$HEAD_GPU_COUNT" -ne 4 ]]; then
    echo "ERROR: expected 4 GPUs on HEAD."
    exit 5
fi

nvidia-smi \
    --query-gpu=index,name,driver_version,memory.total \
    --format=csv,noheader

echo
echo "--- HEAD topology ---"
nvidia-smi topo -m || true

echo
echo "--- HEAD container CUDA/Ray/vLLM ---"

apptainer exec \
    --nv \
    --cleanenv \
    --env "CUDA_VISIBLE_DEVICES=0,1,2,3" \
    --bind "$CACHE_ROOT:/cache" \
    "$IMAGE" \
    python3 - <<'PY'
import socket
import torch
import ray
import vllm

print("host =", socket.getfqdn())
print("vllm =", vllm.__version__)
print("ray =", ray.__version__)
print("torch_cuda =", torch.cuda.is_available())
print("cuda_devices =", torch.cuda.device_count())
PY

# ============================================================
# WORKER preflight via SSH
# ============================================================

echo
echo "============================================================"
echo " WORKER PREFLIGHT VIA SSH: $WORKER_NODE"
echo "============================================================"

ssh "${SSH_OPTS[@]}" "$WORKER_NODE" \
    "IMAGE='$IMAGE' CACHE_ROOT='$CACHE_ROOT' MODEL_CACHE_DIR='$MODEL_CACHE_DIR' bash -s" <<'REMOTE'
set -euo pipefail

echo "node=$(hostname -f)"
echo "ip=$(hostname -I)"

GPU_COUNT="$(nvidia-smi -L | wc -l)"
echo "visible_gpu_count=$GPU_COUNT"

if [[ "$GPU_COUNT" -ne 4 ]]; then
    echo "ERROR: expected 4 GPUs on WORKER."
    exit 20
fi

nvidia-smi \
    --query-gpu=index,name,driver_version,memory.total \
    --format=csv,noheader

echo
echo "--- WORKER topology ---"
nvidia-smi topo -m || true

echo
echo "--- WORKER shared model cache ---"
test -d "$MODEL_CACHE_DIR"
du -sh "$MODEL_CACHE_DIR"

echo
echo "--- WORKER container CUDA/Ray/vLLM ---"

apptainer exec \
    --nv \
    --cleanenv \
    --env "CUDA_VISIBLE_DEVICES=0,1,2,3" \
    --bind "$CACHE_ROOT:/cache" \
    "$IMAGE" \
    python3 - <<'PY'
import socket
import torch
import ray
import vllm

print("host =", socket.getfqdn())
print("vllm =", vllm.__version__)
print("ray =", ray.__version__)
print("torch_cuda =", torch.cuda.is_available())
print("cuda_devices =", torch.cuda.device_count())
PY
REMOTE

echo
echo "============================================================"
echo " PREFLIGHT PASSED ON BOTH NODES"
echo "============================================================"

# ============================================================
# Lifecycle / cleanup
# ============================================================

RAY_HEAD_PID=""
RAY_WORKER_SSH_PID=""

cleanup() {
    rc=$?
    trap - EXIT INT TERM
    set +e

    echo
    echo "============================================================"
    echo " CLEANUP"
    echo "============================================================"

    if [[ -n "${RAY_WORKER_SSH_PID:-}" ]]; then
        echo "stopping worker SSH session pid=$RAY_WORKER_SSH_PID"
        kill "$RAY_WORKER_SSH_PID" 2>/dev/null || true
    fi

    if [[ -n "${RAY_HEAD_PID:-}" ]]; then
        echo "stopping Ray head process pid=$RAY_HEAD_PID"
        kill "$RAY_HEAD_PID" 2>/dev/null || true
    fi

    # Best-effort cleanup of Ray processes started for this job.
    # The worker remote wrapper also has its own EXIT trap.
    ssh "${SSH_OPTS[@]}" "$WORKER_NODE" \
        "apptainer exec --cleanenv --bind '$CACHE_ROOT:/cache' '$IMAGE' ray stop --force" \
        >/dev/null 2>&1 || true

    apptainer exec \
        --cleanenv \
        --bind "$CACHE_ROOT:/cache" \
        "$IMAGE" \
        ray stop --force \
        >/dev/null 2>&1 || true

    [[ -n "${RAY_WORKER_SSH_PID:-}" ]] && \
        wait "$RAY_WORKER_SSH_PID" 2>/dev/null || true

    [[ -n "${RAY_HEAD_PID:-}" ]] && \
        wait "$RAY_HEAD_PID" 2>/dev/null || true

    echo "cleanup complete"
    exit "$rc"
}

trap cleanup EXIT INT TERM

# ============================================================
# Ray HEAD: direct process on batch node
# ============================================================

echo
echo "============================================================"
echo " STARTING RAY HEAD"
echo "============================================================"
echo "node=$HEAD_NODE"
echo "ip=$HEAD_IP"
echo "port=$RAY_PORT"

HEAD_ENV=(
    --env "CUDA_VISIBLE_DEVICES=0,1,2,3"
    --env "HF_HUB_OFFLINE=1"
    --env "TRANSFORMERS_OFFLINE=1"
    --env "VLLM_HOST_IP=$HEAD_IP"
    --env "NCCL_DEBUG=$NCCL_DEBUG"
)

if [[ -n "$NCCL_SOCKET_IFNAME" ]]; then
    HEAD_ENV+=(--env "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME")
fi

apptainer exec \
    --nv \
    --cleanenv \
    "${HEAD_ENV[@]}" \
    --bind "$CACHE_ROOT:/cache" \
    "$IMAGE" \
    ray start \
      --head \
      --node-ip-address="$HEAD_IP" \
      --port="$RAY_PORT" \
      --num-cpus=16 \
      --num-gpus=4 \
      --include-dashboard=false \
      --disable-usage-stats \
      --block \
    >"logs/ray-head-${SLURM_JOB_ID}.log" 2>&1 &

RAY_HEAD_PID=$!
echo "ray_head_pid=$RAY_HEAD_PID"

echo
echo "=== Waiting for Ray HEAD socket ==="

HEAD_READY=0

for attempt in $(seq 1 45); do
    if timeout 2 bash -c "</dev/tcp/$HEAD_IP/$RAY_PORT" 2>/dev/null; then
        HEAD_READY=1
        echo "Ray HEAD socket reachable."
        break
    fi

    if ! kill -0 "$RAY_HEAD_PID" 2>/dev/null; then
        echo "ERROR: Ray HEAD exited."
        cat "logs/ray-head-${SLURM_JOB_ID}.log" || true
        exit 6
    fi

    echo "[head wait $attempt/45]"
    sleep 2
done

if [[ "$HEAD_READY" -ne 1 ]]; then
    echo "ERROR: Ray HEAD did not open $HEAD_IP:$RAY_PORT"
    cat "logs/ray-head-${SLURM_JOB_ID}.log" || true
    exit 6
fi

# ============================================================
# Ray WORKER via SSH
# ============================================================

echo
echo "============================================================"
echo " STARTING RAY WORKER VIA SSH"
echo "============================================================"
echo "node=$WORKER_NODE"
echo "ip=$WORKER_IP"
echo "head=$HEAD_IP:$RAY_PORT"

# The remote shell has a cleanup trap so terminating the SSH session
# also stops the Ray processes it started on that worker.
ssh "${SSH_OPTS[@]}" "$WORKER_NODE" \
    "HEAD_IP='$HEAD_IP' \
     RAY_PORT='$RAY_PORT' \
     WORKER_IP='$WORKER_IP' \
     IMAGE='$IMAGE' \
     CACHE_ROOT='$CACHE_ROOT' \
     NCCL_DEBUG='$NCCL_DEBUG' \
     NCCL_SOCKET_IFNAME='$NCCL_SOCKET_IFNAME' \
     bash -s" \
    >"logs/ray-worker-${SLURM_JOB_ID}.log" 2>&1 <<'REMOTE' &

set -euo pipefail

cleanup_remote() {
    set +e
    apptainer exec \
        --cleanenv \
        --bind "$CACHE_ROOT:/cache" \
        "$IMAGE" \
        ray stop --force \
        >/dev/null 2>&1 || true
}
trap cleanup_remote EXIT INT TERM

RAY_ENV=(
    --env "CUDA_VISIBLE_DEVICES=0,1,2,3"
    --env "HF_HUB_OFFLINE=1"
    --env "TRANSFORMERS_OFFLINE=1"
    --env "VLLM_HOST_IP=$WORKER_IP"
    --env "NCCL_DEBUG=$NCCL_DEBUG"
)

if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
    RAY_ENV+=(--env "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME")
fi

apptainer exec \
    --nv \
    --cleanenv \
    "${RAY_ENV[@]}" \
    --bind "$CACHE_ROOT:/cache" \
    "$IMAGE" \
    ray start \
      --address="$HEAD_IP:$RAY_PORT" \
      --node-ip-address="$WORKER_IP" \
      --num-cpus=16 \
      --num-gpus=4 \
      --disable-usage-stats \
      --block
REMOTE

RAY_WORKER_SSH_PID=$!
echo "ray_worker_ssh_pid=$RAY_WORKER_SSH_PID"

# ============================================================
# Wait for Ray resources
# ============================================================

echo
echo "=== Waiting for Ray cluster: 2 nodes / 8 GPUs ==="

RAY_READY=0

for attempt in $(seq 1 60); do

    RESULT="$(
        timeout 20s apptainer exec \
          --cleanenv \
          --bind "$CACHE_ROOT:/cache" \
          "$IMAGE" \
          python3 - "$HEAD_IP:$RAY_PORT" <<'PY' 2>/dev/null || true
import sys
import ray

address = sys.argv[1]

try:
    ray.init(address=address, logging_level="ERROR")

    nodes = [n for n in ray.nodes() if n.get("Alive")]
    resources = ray.cluster_resources()

    print(
        "NODES=%d GPUS=%d CPUS=%d"
        % (
            len(nodes),
            int(resources.get("GPU", 0)),
            int(resources.get("CPU", 0)),
        )
    )

    for node in sorted(nodes, key=lambda n: n.get("NodeManagerAddress", "")):
        r = node.get("Resources", {})
        print(
            "NODE",
            node.get("NodeManagerAddress"),
            "GPU=" + str(r.get("GPU", 0)),
            "CPU=" + str(r.get("CPU", 0)),
        )

    ray.shutdown()

except Exception as exc:
    print("NOT_READY", type(exc).__name__, str(exc))
PY
    )"

    echo "[ray wait $attempt/60]"
    echo "$RESULT"

    if grep -q "NODES=2 GPUS=8" <<<"$RESULT"; then
        RAY_READY=1
        break
    fi

    if ! kill -0 "$RAY_HEAD_PID" 2>/dev/null; then
        echo "ERROR: Ray HEAD died."
        echo
        tail -150 "logs/ray-head-${SLURM_JOB_ID}.log" || true
        exit 7
    fi

    if ! kill -0 "$RAY_WORKER_SSH_PID" 2>/dev/null; then
        echo "ERROR: Ray WORKER SSH session died."
        echo
        tail -150 "logs/ray-worker-${SLURM_JOB_ID}.log" || true
        exit 7
    fi

    sleep 5
done

if [[ "$RAY_READY" -ne 1 ]]; then
    echo "ERROR: Ray did not reach 2 nodes / 8 GPUs."
    echo
    echo "=== Ray HEAD log ==="
    tail -150 "logs/ray-head-${SLURM_JOB_ID}.log" || true
    echo
    echo "=== Ray WORKER log ==="
    tail -150 "logs/ray-worker-${SLURM_JOB_ID}.log" || true
    exit 7
fi

echo
echo "============================================================"
echo " RAY CLUSTER READY: 2 NODES / 8 GPUS"
echo "============================================================"

echo
echo "=== Ray cluster status ==="

apptainer exec \
    --cleanenv \
    --bind "$CACHE_ROOT:/cache" \
    "$IMAGE" \
    ray status --address="$HEAD_IP:$RAY_PORT" || true

# ============================================================
# Ray-only smoke test
# ============================================================

if [[ "$RAY_ONLY" == "1" ]]; then
    echo
    echo "============================================================"
    echo " RAY_ONLY=1 SUCCESS"
    echo "============================================================"
    echo
    echo "Ray is healthy across:"
    echo "  $HEAD_NODE ($HEAD_IP)   : 4 GPUs"
    echo "  $WORKER_NODE ($WORKER_IP): 4 GPUs"
    echo
    echo "Logs:"
    echo "  logs/ray-head-${SLURM_JOB_ID}.log"
    echo "  logs/ray-worker-${SLURM_JOB_ID}.log"
    echo
    echo "Keeping the cluster alive for 60 seconds for inspection."
    sleep 60
    exit 0
fi

# ============================================================
# vLLM driver / API server on HEAD
# ============================================================

echo
echo "============================================================"
echo " STARTING vLLM"
echo "============================================================"
echo "head_node=$HEAD_NODE"
echo "head_ip=$HEAD_IP"
echo "port=$PORT"
echo "model=$MODEL"
echo "TP=$TP"
echo "PP=$PP"
echo
echo "When startup completes:"
echo "  curl http://127.0.0.1:$PORT/v1/models"
echo
echo "or from another internal node:"
echo "  curl http://$HEAD_IP:$PORT/v1/models"
echo "============================================================"

VLLM_ENV=(
    --env "CUDA_VISIBLE_DEVICES=0,1,2,3"
    --env "HF_HUB_OFFLINE=1"
    --env "TRANSFORMERS_OFFLINE=1"
    --env "RAY_ADDRESS=$HEAD_IP:$RAY_PORT"
    --env "VLLM_HOST_IP=$HEAD_IP"
    --env "NCCL_DEBUG=$NCCL_DEBUG"
)

if [[ -n "$NCCL_SOCKET_IFNAME" ]]; then
    VLLM_ENV+=(--env "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME")
fi

# One vLLM API/driver runs on the head node.
# Ray places the model workers over both nodes.
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
