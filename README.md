# Local HPC LLM Architecture & Serving Framework

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![vLLM](https://img.shields.io/badge/vLLM-v0.8.5+-blue.svg)](https://github.com/vllm-project/vllm)
[![Apptainer](https://img.shields.io/badge/Apptainer-Supported-green.svg)](https://apptainer.org/)
[![Slurm](https://img.shields.io/badge/Slurm-Workload_Manager-orange.svg)](https://slurm.schedmd.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end local Large Language Model (LLM) serving stack designed for high-performance computing (HPC) environments equipped with GPU clusters (NVIDIA V100/A100/H100) and Slurm workload management. 

This repository provides a multi-layer solution:
1. **Infrastructure Layer**: Containerized vLLM engine, automated Hugging Face model downloads, single/multi-GPU Tensor Parallelism (TP=4), and multi-node Ray cluster orchestration (TP=4, PP=2).
2. **Application Layer**: An OpenAI-compatible FastAPI Gateway proxy with a built-in Web UI dashboard, API key authentication (`sk-hpc-...`), SHA-256 hashed SQLite database storage, Model ACLs, and sliding window rate limiting.

---

## 🏗️ Architecture Overview

```text
+-------------------------------------------------------------------------------+
|                                CLIENT LAYER                                   |
|   Python SDK (openai)    |    cURL / REST Clients    |    Web UI Dashboard    |
+---------------------------------------+---------------------------------------+
                                        | (HTTP http://127.0.0.1:9000/v1)
                                        v
+-------------------------------------------------------------------------------+
|                          APPLICATION LAYER (GATEWAY)                          |
|  FastAPI OpenAI Proxy  -  API Key ACL & Hash Storage  -  RPM Rate Limiting    |
+---------------------------------------+---------------------------------------+
                                        | (SSH Tunnel 127.0.0.1:18000 -> Node:8000)
                                        v
+-------------------------------------------------------------------------------+
|                           INFRASTRUCTURE LAYER (HPC)                          |
|  Slurm Jobs  -  Apptainer Container (vllm.sif)  -  Ray Distributed Engine     |
|  GPUs: NVIDIA V100/A100/H100  |  Models: Qwen2.5-32B, Qwen3-235B (TP4, PP2)   |
+-------------------------------------------------------------------------------+
```

Detailed architectural documentation is available in [docs/architecture.md](file:///home/phucnhan/codespace/hpc/local-llm/docs/architecture.md).

---

## ✨ Features

- **HPC Cluster Optimization**: Pre-configured for Slurm job submission with isolated Apptainer/Singularity container profiles (`defs/vllm.def`).
- **Distributed Inference**: Support for single-GPU, multi-GPU Tensor Parallelism (`TP=4`), and multi-node Ray pipeline execution (`TP=4, PP=2`).
- **Zero-Downtime Model Caching**: Standalone sbatch download jobs with hash verification and auto-repair (`slurm/download/`).
- **OpenAI-Compatible Gateway**: Serve standard `/v1/chat/completions`, `/v1/completions`, and `/v1/models` endpoints.
- **Secure Key Management**: Issue `sk-hpc-...` API keys stored as SHA-256 hashes in SQLite.
- **Access Control & Rate Limiting**: Model-level ACL per key and customizable Requests-Per-Minute (RPM) enforcement.
- **Web UI Dashboard**: Administrative dashboard for generating keys, tracking metrics, and revoking tokens.

---

## 📁 Repository Structure

```text
local-llm/
├── .gitignore                          # Ignored secrets, databases, caches & logs
├── README.md                           # Root documentation & getting started
├── docs/                               # Detailed system documentation
│   ├── api-gateway.md                  # Application gateway & dashboard guide
│   ├── architecture.md                 # Technical architecture & request flows
│   └── infrastructure-deployment.md    # HPC, Slurm, Ray & Apptainer guide
├── application_layer/                  # Application Layer Gateway Proxy
│   └── onenode/                        # Single-gateway service deployment
│       ├── app/                        # FastAPI application source code
│       │   ├── __init__.py
│       │   └── main.py                 # Core proxy server, key management & Web UI
│       ├── config/                     # Model routing configuration
│       │   └── models.json
│       ├── examples/                   # Usage code snippets
│       │   ├── curl.sh
│       │   └── python_client.py
│       ├── scripts/                    # Deployment & helper scripts
│       │   ├── run.sh                  # Virtualenv setup & gateway launcher
│       │   └── ssh-tunnel.sh           # Local SSH port forwarding to HPC node
│       ├── .env.example                # Template configuration file
│       ├── README.md                   # Application layer detailed readme
│       └── requirements.txt            # Python dependencies
└── infrastructure_layer/               # Infrastructure Layer (HPC / Slurm)
    ├── build/                          # Apptainer build directory & notes
    ├── defs/                           # Container definition files
    │   └── vllm.def                    # Apptainer vLLM profile for NVIDIA GPUs
    ├── logs/                           # Slurm execution logs (.out / .err)
    └── slurm/                          # Slurm batch submission scripts
        ├── download/                   # HF model download & repair jobs
        │   ├── hf_download_qwen235b.sbatch
        │   ├── hf_download_qwen32b.sbatch
        │   └── hf_verify_repair_qwen32b.sbatch
        ├── multi_node_deploy.sbatch    # 2-node 8-GPU Ray + vLLM deployment
        ├── vllm-multigpu-singlenode.sbatch # 4-GPU single-node TP4 deployment
        └── vllm-singlegpu-singlenode.sbatch # Single GPU single-node deployment
```

---

## 🚀 Quick Start Guide

### 1. Build Container & Download Model (HPC Cluster)
On the HPC login/build node:

```bash
cd infrastructure_layer

# Build Apptainer container
apptainer build build/vllm.sif defs/vllm.def

# Submit model download job
sbatch slurm/download/hf_download_qwen32b.sbatch
```

### 2. Launch Inference Server (Slurm)
Submit a Slurm job to start the vLLM engine:

```bash
# 4-GPU Single Node (Qwen2.5-32B)
sbatch infrastructure_layer/slurm/vllm-multigpu-singlenode.sbatch
```

### 3. Open SSH Tunnel & Launch Gateway (Local Machine)
Connect your local environment to the active Slurm compute node:

```bash
# Terminal 1: Open SSH Tunnel
cd application_layer/onenode
GPU_NODE=gpunode1.gitc.hpc ./scripts/ssh-tunnel.sh

# Terminal 2: Start Gateway Proxy API
cd application_layer/onenode
./scripts/run.sh
```

### 4. Create API Key & Send Request
1. Open `http://127.0.0.1:9000/` in your browser.
2. Enter the `ADMIN_TOKEN` found in `application_layer/onenode/.env`.
3. Generate a new API key (e.g. `sk-hpc-samplekey...`).
4. Execute requests using OpenAI Python SDK or cURL:

```python
import openai

client = openai.OpenAI(
    base_url="http://127.0.0.1:9000/v1",
    api_key="sk-hpc-YOUR_KEY"
)

response = client.chat.completions.create(
    model="qwen2.5-32b",
    messages=[{"role": "user", "content": "Hello vLLM on HPC!"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

---

## 📚 Documentation Index

- [Architecture & System Design](file:///home/phucnhan/codespace/hpc/local-llm/docs/architecture.md)
- [Infrastructure & Slurm Deployment Guide](file:///home/phucnhan/codespace/hpc/local-llm/docs/infrastructure-deployment.md)
- [API Gateway & Dashboard Guide](file:///home/phucnhan/codespace/hpc/local-llm/docs/api-gateway.md)

---

## 🔐 Security & Data Hygiene

- `.env` files contain sensitive tokens and secrets and are ignored by `.gitignore`.
- SQLite database files (`gateway.db`) store only hashed key representations.
- Model cache directories and container images (`*.sif`) are excluded from version control.
