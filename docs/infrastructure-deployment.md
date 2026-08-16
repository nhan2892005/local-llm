# Infrastructure Deployment & Slurm Guide

This guide details how to build containers, download models, and deploy high-performance LLM inference servers using Slurm, Apptainer/Singularity, and vLLM on an HPC cluster.

---

## 1. Prerequisites & Environment

- **HPC Scheduler**: Slurm Workload Manager.
- **Container Engine**: Apptainer / Singularity (unprivileged runtime with GPU support `--nv`).
- **Hardware**: NVIDIA Volta V100 / Ampere A100 / Hopper H100 GPUs.
- **Storage**: Fast shared scratch or NFS file system accessible across compute nodes.

---

## 2. Building the Apptainer Container

Build the vLLM container image from the definition file on a build node or login node with Apptainer access:

```bash
cd infrastructure_layer
mkdir -p build

apptainer build build/vllm.sif defs/vllm.def
```

*Note: The definition file `defs/vllm.def` builds upon `vllm/vllm-openai:v0.8.5.post1` and configures environment variables suited for NVIDIA Volta V100 (`VLLM_USE_V1=0`, `NVIDIA_TF32_OVERRIDE=0`, `dtype float16`).*

---

## 3. Downloading Hugging Face Models

To prevent holding GPU allocations idle while downloading massive weights, use compute-node download scripts:

```bash
cd infrastructure_layer

# Submit job to download Qwen2.5-32B-Instruct
sbatch slurm/download/hf_download_qwen32b.sbatch

# Submit job to download Qwen3-235B (GPTQ INT4)
sbatch slurm/download/hf_download_qwen235b.sbatch
```

To verify or repair model snapshot integrity:

```bash
sbatch slurm/download/hf_verify_repair_qwen32b.sbatch
```

---

## 4. Submitting Slurm Inference Jobs

### 4.1 Single GPU / Single Node
For smaller models or development testing:

```bash
sbatch infrastructure_layer/slurm/vllm-singlegpu-singlenode.sbatch
```

### 4.2 Multi-GPU Single Node (Tensor Parallelism TP=4)
For models such as `Qwen/Qwen2.5-32B-Instruct` requiring multiple GPUs on 1 node:

```bash
sbatch infrastructure_layer/slurm/vllm-multigpu-singlenode.sbatch
```

Job specifications:
- **GPUs**: 4 GPUs (`--gres=gpu:4`)
- **CPUs**: 16 cores (`--cpus-per-task=16`)
- **RAM**: 128 GB
- **Tensor Parallelism**: `TP=4`

### 4.3 Multi-Node Deployment (Ray Cluster: TP=4, PP=2)
For large-scale models (e.g. `Qwen/Qwen3-235B-A22B-GPTQ-Int4` across 2 nodes x 4 GPUs = 8 GPUs total):

```bash
sbatch infrastructure_layer/slurm/multi_node_deploy.sbatch
```

How Multi-Node Deployment Works:
1. Slurm allocates 2 compute nodes.
2. Script resolves Head node and Worker node IP addresses.
3. Launches a Ray head cluster on Node 0 and connects Node 1 as a Ray worker node.
4. Polls Ray cluster readiness (`NODES=2 GPUS=8`).
5. Launches `vllm serve` with `--distributed-executor-backend ray --tensor-parallel-size 4 --pipeline-parallel-size 2`.

---

## 5. Monitoring & Troubleshooting

Check running jobs and locate compute node names:

```bash
squeue -u $USER
```

Inspect Slurm logs:

```bash
tail -f infrastructure_layer/logs/vllm-*.out
tail -f infrastructure_layer/logs/vllm-*.err
```

Test endpoint directly on compute node:

```bash
ssh <compute-node-name> 'curl http://127.0.0.1:8000/v1/models'
```
