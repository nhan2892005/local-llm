#!/bin/bash
#SBATCH --job-name=hf-qwen32b-download
#SBATCH --partition=compute
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/hf-download-%j.out
#SBATCH --error=logs/hf-download-%j.err

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

IMAGE="${IMAGE:-$PWD/build/vllm.sif}"
MODEL="${MODEL:-Qwen/Qwen3-235B-A22B-GPTQ-Int4}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:-$HOME/.cache}/llm-serving}"

mkdir -p "$CACHE_ROOT"/{huggingface,vllm,xdg}

if [[ ! -x "$CACHE_ROOT/hf-downloader/venv/bin/python" ]]; then
  echo "ERROR: downloader venv not found:"
  echo "  $CACHE_ROOT/hf-downloader/venv/bin/python"
  exit 2
fi

echo "============================================================"
echo "job_id=${SLURM_JOB_ID:-}"
echo "node=$(hostname -f)"
echo "partition=${SLURM_JOB_PARTITION:-}"
echo "model=$MODEL"
echo "image=$IMAGE"
echo "cache_root=$CACHE_ROOT"
echo "============================================================"

apptainer exec \
  --cleanenv \
  --env HF_HUB_DISABLE_XET=1 \
  --env HF_HUB_DOWNLOAD_TIMEOUT=1800 \
  --env HF_HUB_ETAG_TIMEOUT=120 \
  --env HF_HOME=/cache/huggingface \
  --env HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
  --bind "$CACHE_ROOT:/cache" \
  "$IMAGE" \
  /cache/hf-downloader/venv/bin/python - "$MODEL" <<'PY'
import os
import sys
import time
from pathlib import Path

import huggingface_hub
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils._runtime import is_xet_available

repo_id = sys.argv[1]
cache_dir = "/cache/huggingface/hub"

print("huggingface_hub =", huggingface_hub.__version__, flush=True)
print("HF_HUB_DISABLE_XET =", os.getenv("HF_HUB_DISABLE_XET"), flush=True)
print("is_xet_available =", is_xet_available(), flush=True)
print("repo_id =", repo_id, flush=True)
print("cache_dir =", cache_dir, flush=True)

api = HfApi()

while True:
    try:
        files = api.list_repo_files(repo_id=repo_id)
        break
    except Exception as e:
        print(f"[metadata retry] {type(e).__name__}: {e}", flush=True)
        time.sleep(60)

print(f"repository files = {len(files)}", flush=True)

for i, filename in enumerate(files, 1):
    attempt = 0

    while True:
        attempt += 1
        print(f"\n[{i}/{len(files)}] {filename} attempt={attempt}", flush=True)

        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
            )
            print(f"[OK] {filename}", flush=True)
            print(path, flush=True)
            break

        except KeyboardInterrupt:
            raise

        except Exception as e:
            print(f"[RETRY] {type(e).__name__}: {e}", flush=True)
            print("sleep 60s; existing cache is kept", flush=True)
            time.sleep(60)

snapshot = snapshot_download(
    repo_id=repo_id,
    cache_dir=cache_dir,
    local_files_only=True,
)

print("\n============================================================", flush=True)
print("MODEL DOWNLOAD COMPLETE", flush=True)
print("snapshot =", snapshot, flush=True)
print("============================================================", flush=True)

marker = Path("/cache/huggingface") / (repo_id.replace("/", "--") + ".READY")
marker.write_text(snapshot + "\n")
print("ready marker =", marker, flush=True)
PY

status=$?

echo
du -sh "$CACHE_ROOT/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct" 2>/dev/null || true

exit "$status"
