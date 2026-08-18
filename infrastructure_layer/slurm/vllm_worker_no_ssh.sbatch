#!/bin/bash
#SBATCH --job-name=vllm-worker
#SBATCH --partition=gpu-queue
#SBATCH --qos=gpu-q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpunode3
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/vllm-worker-%j.out
#SBATCH --error=logs/vllm-worker-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

: "${RUN_ID:?RUN_ID is required}"

source "$PWD/slurm/vllm_2node_common.sh"

SESSION="$PWD/logs/vllm-run/$RUN_ID"
mkdir -p "$SESSION"

WORKER_IP="$(getent ahostsv4 "$(hostname -s)" | awk 'NR==1{print $1}')"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

echo "============================================================"
echo " vLLM WORKER"
echo "============================================================"
echo "run_id=$RUN_ID"
echo "job=$SLURM_JOB_ID"
echo "host=$(hostname -f)"
echo "ip=$WORKER_IP"
echo "image=$IMAGE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"
echo "============================================================"

[[ -d "$IMAGE" ]] || { echo "ERROR: sandbox missing: $IMAGE"; exit 2; }

mapfile -t ENV_ARGS < <(container_env_args "$WORKER_IP")

cleanup() {
  set +e
  apptainer exec --nv --cleanenv \
    --env "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES" \
    "${ENV_ARGS[@]}" \
    "$IMAGE" ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

apptainer exec --nv --cleanenv \
  --env "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES" \
  "${ENV_ARGS[@]}" \
  "$IMAGE" ray stop --force >/dev/null 2>&1 || true

echo "Waiting for HEAD Ray endpoint..."
for i in $(seq 1 1200); do
  [[ -s "$SESSION/ray.env" ]] && break
  [[ -e "$SESSION/STOP" ]] && { echo "HEAD stopped before Ray endpoint"; exit 3; }
  sleep 1
done

[[ -s "$SESSION/ray.env" ]] || { echo "ERROR: timed out waiting for ray.env"; exit 4; }
source "$SESSION/ray.env"

[[ ! -e "$SESSION/STOP" ]] || { echo "HEAD already stopped"; exit 0; }

echo "Joining Ray at $RAY_HEAD_IP:$RAY_GCS_PORT ..."

apptainer exec --nv --cleanenv \
  --env "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES" \
  "${ENV_ARGS[@]}" \
  "$IMAGE" \
  ray start \
    --address="$RAY_HEAD_IP:$RAY_GCS_PORT" \
    --node-ip-address="$WORKER_IP" \
    --node-manager-port="$RAY_NODE_MANAGER_PORT" \
    --object-manager-port="$RAY_OBJECT_MANAGER_PORT" \
    --runtime-env-agent-port="$RAY_RUNTIME_ENV_PORT" \
    --dashboard-agent-grpc-port="$RAY_DASHBOARD_AGENT_GRPC_PORT" \
    --dashboard-agent-listen-port="$RAY_DASHBOARD_AGENT_HTTP_PORT" \
    --metrics-export-port="$RAY_METRICS_PORT" \
    --min-worker-port="$RAY_WORKER_MIN_PORT" \
    --max-worker-port="$RAY_WORKER_MAX_PORT" \
    --num-cpus=16 \
    --num-gpus=4

touch "$SESSION/worker-ray-started"
echo "RAY WORKER JOIN COMMAND SUCCEEDED"

# Keep allocation alive as long as the head/API is alive.
while [[ ! -e "$SESSION/STOP" ]]; do
  sleep 5
done

echo "STOP marker received"
