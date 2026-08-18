#!/bin/bash
#SBATCH --job-name=vllm-head
#SBATCH --partition=gpu-queue
#SBATCH --qos=gpu-q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpunode1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/vllm-head-%j.out
#SBATCH --error=logs/vllm-head-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

: "${RUN_ID:?RUN_ID is required}"

source "$PWD/slurm/vllm_2node_common.sh"

SESSION="$PWD/logs/vllm-run/$RUN_ID"
mkdir -p "$SESSION"
rm -f "$SESSION/STOP" "$SESSION/API_READY" "$SESSION/api.env"

HEAD_IP="$(getent ahostsv4 "$(hostname -s)" | awk 'NR==1{print $1}')"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

echo "============================================================"
echo " vLLM HEAD"
echo "============================================================"
echo "run_id=$RUN_ID"
echo "job=$SLURM_JOB_ID"
echo "host=$(hostname -f)"
echo "ip=$HEAD_IP"
echo "image=$IMAGE"
echo "model=$MODEL"
echo "tp=$TP pp=$PP"
echo "ray=$HEAD_IP:$RAY_GCS_PORT"
echo "api=http://$HEAD_IP:$API_PORT"
echo "ray_ports=50000-50399"
echo "gloo_ports=$GLOO_PORT_MIN-$GLOO_PORT_MAX"
echo "nccl_ports=$NCCL_PORT_MIN-$NCCL_PORT_MAX"
echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"
echo "============================================================"

[[ -d "$IMAGE" ]] || { echo "ERROR: sandbox missing: $IMAGE"; exit 2; }

mapfile -t ENV_ARGS < <(container_env_args "$HEAD_IP")

cleanup() {
  set +e
  touch "$SESSION/STOP"
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
  apptainer exec --nv --cleanenv \
    --env "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES" \
    "${ENV_ARGS[@]}" \
    "$IMAGE" ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# Kill stale Ray owned by this user on the allocated node.
apptainer exec --nv --cleanenv \
  --env "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES" \
  "${ENV_ARGS[@]}" \
  "$IMAGE" ray stop --force >/dev/null 2>&1 || true

echo "Starting Ray HEAD..."
apptainer exec --nv --cleanenv \
  --env "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES" \
  "${ENV_ARGS[@]}" \
  "$IMAGE" \
  ray start \
    --head \
    --node-ip-address="$HEAD_IP" \
    --port="$RAY_GCS_PORT" \
    --node-manager-port="$RAY_NODE_MANAGER_PORT" \
    --object-manager-port="$RAY_OBJECT_MANAGER_PORT" \
    --runtime-env-agent-port="$RAY_RUNTIME_ENV_PORT" \
    --dashboard-agent-grpc-port="$RAY_DASHBOARD_AGENT_GRPC_PORT" \
    --dashboard-agent-listen-port="$RAY_DASHBOARD_AGENT_HTTP_PORT" \
    --metrics-export-port="$RAY_METRICS_PORT" \
    --ray-client-server-port="$RAY_CLIENT_PORT" \
    --dashboard-port="$RAY_DASHBOARD_PORT" \
    --include-dashboard=false \
    --min-worker-port="$RAY_WORKER_MIN_PORT" \
    --max-worker-port="$RAY_WORKER_MAX_PORT" \
    --num-cpus=16 \
    --num-gpus=4

cat > "$SESSION/ray.env.tmp" <<EOF
RAY_HEAD_IP=$HEAD_IP
RAY_GCS_PORT=$RAY_GCS_PORT
EOF
mv "$SESSION/ray.env.tmp" "$SESSION/ray.env"

echo "Ray endpoint published: $HEAD_IP:$RAY_GCS_PORT"
echo "Waiting for worker / 8 GPUs..."

apptainer exec --nv --cleanenv \
  --env "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES" \
  --env "RAY_ADDRESS=$HEAD_IP:$RAY_GCS_PORT" \
  --env "RAY_WAIT_TIMEOUT=1200" \
  "${ENV_ARGS[@]}" \
  --bind "$PWD:$PWD" \
  "$IMAGE" \
  python3 "$PWD/slurm/wait_ray_2node.py"

echo
echo "=== Ray resources ==="
apptainer exec --nv --cleanenv \
  --env "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES" \
  --env "RAY_ADDRESS=$HEAD_IP:$RAY_GCS_PORT" \
  "${ENV_ARGS[@]}" \
  "$IMAGE" ray status --address="$HEAD_IP:$RAY_GCS_PORT" || true

echo
echo "Starting vLLM..."

VLLM_LOG="$SESSION/vllm-server.log"

apptainer exec --nv --cleanenv \
  --env "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES" \
  --env "RAY_ADDRESS=$HEAD_IP:$RAY_GCS_PORT" \
  "${ENV_ARGS[@]}" \
  --bind "$PWD:$PWD" \
  "$IMAGE" \
  vllm serve "$MODEL" \
    --served-model-name "$SERVED_MODEL" \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --dtype half \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --distributed-executor-backend ray \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --disable-custom-all-reduce \
    --generation-config vllm \
    >"$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vllm_pid=$VLLM_PID"
echo "vllm_log=$VLLM_LOG"

echo "Waiting for API health..."
READY=0
for i in $(seq 1 180); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vLLM exited before becoming ready"
    tail -n 200 "$VLLM_LOG" || true
    exit 20
  fi

  if curl -fsS "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
    READY=1
    break
  fi

  if (( i % 10 == 0 )); then
    echo "still loading... $((i*5))s"
    tail -n 8 "$VLLM_LOG" || true
  fi
  sleep 5
done

if [[ "$READY" != 1 ]]; then
  echo "ERROR: API did not become healthy"
  tail -n 250 "$VLLM_LOG" || true
  exit 21
fi

cat > "$SESSION/api.env.tmp" <<EOF
API_HOST=$HEAD_IP
API_PORT=$API_PORT
API_URL=http://$HEAD_IP:$API_PORT
SERVED_MODEL=$SERVED_MODEL
HEAD_JOB=$SLURM_JOB_ID
EOF
mv "$SESSION/api.env.tmp" "$SESSION/api.env"
touch "$SESSION/API_READY"

echo
echo "============================================================"
echo " VLLM READY"
echo "============================================================"
echo "API_URL=http://$HEAD_IP:$API_PORT"
echo "MODEL=$SERVED_MODEL"
echo "session=$SESSION"
echo
echo "Test from node16:"
echo "curl http://$HEAD_IP:$API_PORT/v1/models"
echo "============================================================"

wait "$VLLM_PID"
