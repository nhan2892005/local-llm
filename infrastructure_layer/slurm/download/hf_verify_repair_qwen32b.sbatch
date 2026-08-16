#!/bin/bash
#SBATCH --job-name=hf-repair-qwen32b
#SBATCH --partition=compute
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/hf-repair-%j.out
#SBATCH --error=logs/hf-repair-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

IMAGE="${IMAGE:-$PWD/build/vllm.sif}"
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:-$HOME/.cache}/llm-serving}"

mkdir -p "$CACHE_ROOT"/{huggingface,vllm,xdg}

DOWNLOADER_HOST="$CACHE_ROOT/hf-downloader/venv/bin/python"
DOWNLOADER_CONTAINER="/cache/hf-downloader/venv/bin/python"

if [[ ! -x "$DOWNLOADER_HOST" ]]; then
  echo "ERROR: downloader venv not found:"
  echo "  $DOWNLOADER_HOST"
  exit 2
fi

echo "============================================================"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "node=$(hostname -f)"
echo "partition=${SLURM_JOB_PARTITION:-}"
echo "model=$MODEL"
echo "image=$IMAGE"
echo "cache_root=$CACHE_ROOT"
echo "repair_shards=10..16"
echo "============================================================"

echo
echo "=== Step 1: force re-download ONLY corrupted shards 10..16 ==="

apptainer exec \
  --cleanenv \
  --env HF_HUB_DISABLE_XET=1 \
  --env HF_HUB_DOWNLOAD_TIMEOUT=1800 \
  --env HF_HUB_ETAG_TIMEOUT=120 \
  --env HF_HOME=/cache/huggingface \
  --env HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
  --bind "$CACHE_ROOT:/cache" \
  "$IMAGE" \
  "$DOWNLOADER_CONTAINER" - "$MODEL" <<'PY'
import sys
import time

from huggingface_hub import hf_hub_download

repo_id = sys.argv[1]
cache_dir = "/cache/huggingface/hub"

bad_shards = [
    f"model-{i:05d}-of-00017.safetensors"
    for i in range(10, 17)
]

print("Will repair:", flush=True)
for name in bad_shards:
    print(" -", name, flush=True)

for idx, name in enumerate(bad_shards, 1):
    attempt = 0

    while True:
        attempt += 1
        print(
            f"\n[{idx}/{len(bad_shards)}] force-download {name} "
            f"attempt={attempt}",
            flush=True,
        )

        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=name,
                cache_dir=cache_dir,
                force_download=True,
            )
            print(f"[DOWNLOADED] {name}", flush=True)
            print(path, flush=True)
            break

        except KeyboardInterrupt:
            raise

        except Exception as e:
            print(
                f"[RETRY] {name}: {type(e).__name__}: {e}",
                flush=True,
            )
            print("sleep 60s and retry same shard", flush=True)
            time.sleep(60)

print("\nAll requested shards were re-downloaded.", flush=True)
PY

echo
echo "=== Step 2: verify ALL 17 safetensors shards with vLLM Python ==="

apptainer exec \
  --cleanenv \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --bind "$CACHE_ROOT:/cache" \
  "$IMAGE" \
  python3 - "$MODEL" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open

repo_id = sys.argv[1]
cache_dir = "/cache/huggingface/hub"

snapshot = Path(
    snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        local_files_only=True,
    )
)

print("snapshot =", snapshot, flush=True)

files = sorted(snapshot.glob("model-*.safetensors"))
print("found shards =", len(files), flush=True)

if len(files) != 17:
    print(f"ERROR: expected 17 shards, found {len(files)}", flush=True)
    sys.exit(3)

bad = []

for f in files:
    try:
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            n_tensors = len(list(sf.keys()))

        print(
            f"[OK]  {f.name} "
            f"size={f.stat().st_size} "
            f"tensors={n_tensors}",
            flush=True,
        )

    except Exception as e:
        print(
            f"[BAD] {f.name}: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )
        bad.append(f.name)

if bad:
    print("\nVERIFY FAILED. Still bad:", flush=True)
    for name in bad:
        print(" -", name, flush=True)
    sys.exit(4)

print("\n============================================================", flush=True)
print("MODEL CACHE VERIFIED: ALL 17 SHARDS ARE VALID", flush=True)
print("============================================================", flush=True)

marker = Path("/cache/huggingface/Qwen--Qwen2.5-32B-Instruct.VERIFIED")
marker.write_text(str(snapshot) + "\n")
print("verified marker =", marker, flush=True)
PY

echo
echo "=== Cache size ==="
du -sh \
  "$CACHE_ROOT/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct" \
  2>/dev/null || true

echo
echo "Repair + verification completed successfully."
