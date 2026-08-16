#!/bin/bash
#SBATCH --job-name=vllm-qwen32b-tp4
#SBATCH --partition=gpu-queue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --qos=gpu-q
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/vllm-%x-%j.out
#SBATCH --error=logs/vllm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

IMAGE="${IMAGE:-$SLURM_SUBMIT_DIR/build/vllm.sif}"
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:-$HOME/.cache}/llm-serving}"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
TP="${TP:-4}"
DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

mkdir -p "$CACHE_ROOT"/{huggingface,vllm,xdg}

echo "============================================================"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "node=$(hostname -f)"
echo "image=$IMAGE"
echo "model=$MODEL"
echo "tp=$TP"
echo "dtype=$DTYPE"
echo "max_model_len=$MAX_MODEL_LEN"
echo "cache_root=$CACHE_ROOT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "============================================================"

echo
echo "=== GPUs allocated by Slurm ==="
nvidia-smi \
  --query-gpu=index,name,driver_version,memory.total \
  --format=csv,noheader

echo
echo "=== GPU topology ==="
nvidia-smi topo -m || true

echo
echo "=== Check downloaded model ==="
du -sh \
  "$CACHE_ROOT/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct"

echo
echo "=== Cache env inside container ==="
apptainer exec \
  --cleanenv \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --bind "$CACHE_ROOT:/cache" \
  "$IMAGE" \
  /bin/bash -lc '
    env | grep -E "^(HF_HOME|HUGGINGFACE_HUB_CACHE|TRANSFORMERS_CACHE|VLLM_CACHE_ROOT|XDG_CACHE_HOME|HF_HUB_OFFLINE|TRANSFORMERS_OFFLINE)=" || true
  '

echo
echo "=== Starting vLLM TP=$TP ==="

srun --ntasks=1 apptainer run \
  --nv \
  --cleanenv \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --bind "$CACHE_ROOT:/cache" \
  "$IMAGE" \
  serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN"
