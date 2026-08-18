#!/bin/bash
#SBATCH --job-name=gloo-head
#SBATCH --partition=gpu-queue
#SBATCH --qos=gpu-q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpunode1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/gloo-pair-%x-%j.out
#SBATCH --error=logs/gloo-pair-%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

: "${RUN_ID:?RUN_ID is required}"

IMAGE="${IMAGE:-$PWD/build/vllm-gloo-patched.sandbox}"
CONTROL_IF="${CONTROL_IF:-ens11f0}"
GLOO_PORT_MIN="${GLOO_PORT_MIN:-50700}"
GLOO_PORT_MAX="${GLOO_PORT_MAX:-51000}"

SESSION_DIR="$PWD/logs/gloo-pair/$RUN_ID"
mkdir -p "$SESSION_DIR"
rm -f "$SESSION_DIR/endpoint.env" "$SESSION_DIR/head.ok" "$SESSION_DIR/worker.ok"

HEAD_IP="$(getent ahostsv4 "$(hostname -s)" | awk 'NR==1{print $1}')"
MASTER_PORT=$((50000 + SLURM_JOB_ID % 650))

cat > "$SESSION_DIR/endpoint.env.tmp" <<EOF
MASTER_ADDR=$HEAD_IP
MASTER_PORT=$MASTER_PORT
EOF
mv "$SESSION_DIR/endpoint.env.tmp" "$SESSION_DIR/endpoint.env"

echo "============================================================"
echo " GLOO HEAD / NO SSH"
echo "============================================================"
echo "run_id=$RUN_ID"
echo "job=$SLURM_JOB_ID"
echo "host=$(hostname -f)"
echo "image=$IMAGE"
echo "master=$HEAD_IP:$MASTER_PORT"
echo "gloo_range=$GLOO_PORT_MIN-$GLOO_PORT_MAX"
echo "session=$SESSION_DIR"
echo "============================================================"

[[ -d "$IMAGE" ]] || {
  echo "ERROR: sandbox missing: $IMAGE"
  exit 2
}

export MASTER_ADDR="$HEAD_IP"
export MASTER_PORT
export RANK=0
export GLOO_SOCKET_IFNAME="$CONTROL_IF"
export GLOO_PORT_MIN
export GLOO_PORT_MAX

apptainer exec \
  --cleanenv \
  --env "MASTER_ADDR=$MASTER_ADDR" \
  --env "MASTER_PORT=$MASTER_PORT" \
  --env "RANK=0" \
  --env "GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME" \
  --env "GLOO_PORT_MIN=$GLOO_PORT_MIN" \
  --env "GLOO_PORT_MAX=$GLOO_PORT_MAX" \
  --bind "$PWD:$PWD" \
  "$IMAGE" \
  python3 "$PWD/slurm/gloo_pair_test.py"

touch "$SESSION_DIR/head.ok"

echo "HEAD PASS"
