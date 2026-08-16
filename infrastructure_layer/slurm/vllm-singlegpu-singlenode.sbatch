#!/bin/bash
#SBATCH --job-name=vllm-smoke
#SBATCH --partition=gpu-queue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-q
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/vllm-%j.out
#SBATCH --error=logs/vllm-%j.err

set -euo pipefail

# Override these from sbatch --export=ALL,IMAGE=...,MODEL=...,CACHE_ROOT=...
IMAGE=${IMAGE:-$PWD/build/vllm.sif}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
CACHE_ROOT=${CACHE_ROOT:-${SCRATCH:-$HOME/.cache}/llm-serving}
PORT=${PORT:-8000}
TP=${TP:-1}

mkdir -p "$CACHE_ROOT"/{huggingface,vllm,xdg}

echo "node=$(hostname)"
echo "image=$IMAGE"
echo "model=$MODEL"
echo "tp=$TP"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# V100 does not support BF16. Start with FP16 for the baseline.
# Keep context modest for the first smoke test; increase during benchmark runs.
apptainer run --nv --cleanenv \
  --bind "$CACHE_ROOT:/cache" \
  "$IMAGE" \
  serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype float16 \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096