# Local HPC LLM Architecture Overview

This document describes the high-level system architecture, design decisions, and component interactions for the **Local LLM** infrastructure and application stack.

---

## 1. System Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                                  CLIENT LAYER                                     |
|  +------------------------+      +------------------------+      +---------------+  |
|  |  Python Client (SDK)   |      |   cURL / REST Clients  |      | Web Dashboard |  |
|  +-----------+------------+      +-----------+------------+      +-------+-------+  |
+--------------|-------------------------------|---------------------------|--------+
               |                               |                           |
               +-------------------------------+---------------------------+
                                               |
                                               v (HTTP http://127.0.0.1:9000/v1)
+-----------------------------------------------------------------------------------+
|                             APPLICATION LAYER (GATEWAY)                           |
|  +-----------------------------------------------------------------------------+  |
|  | FastAPI Gateway Proxy Server                                                 |  |
|  |  - OpenAI-Compatible REST API (/v1/chat/completions, /v1/models, etc.)        |  |
|  |  - API Key Authentication & Prefix Validation (sk-hpc-...)                  |  |
|  |  - Model ACL & Rate Limiting (RPM tracking per key)                         |  |
|  |  - SQLite Storage (SHA-256 key hashing, usage metrics)                       |  |
|  |  - Web UI Admin Dashboard for key generation & management                   |  |
|  +-------------------------------------+---------------------------------------+  |
+----------------------------------------|------------------------------------------+
                                         |
                                         v (Forwarded to SSH Tunnel: 127.0.0.1:18000)
+-----------------------------------------------------------------------------------+
|                                 NETWORKING LAYER                                  |
|  +-----------------------------------------------------------------------------+  |
|  | SSH Local Port Forwarding Tunnel                                            |  |
|  | 127.0.0.1:18000  ===>  HPC Compute Node (e.g., gpunode1.gitc.hpc:8000)        |  |
|  +-------------------------------------+---------------------------------------+  |
+----------------------------------------|------------------------------------------+
                                         |
                                         v (HTTP / Native IPC)
+-----------------------------------------------------------------------------------+
|                            INFRASTRUCTURE LAYER (HPC)                             |
|                                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  | Slurm Workload Manager                                                      |  |
|  |  - Allocates compute nodes & GPU resources (NVIDIA V100/A100/H100)            |  |
|  +-------------------------------------+---------------------------------------+  |
|                                        |                                          |
|  +-------------------------------------v---------------------------------------+  |
|  | Apptainer / Singularity Container (vllm.sif)                                |  |
|  |  - Pinned CUDA & vLLM Runtime (v0.8.5+)                                       |  |
|  |  - Isolated PyTorch & FlashAttention environment                              |  |
|  +-------------------------------------+---------------------------------------+  |
|                                        |                                          |
|  +-------------------------------------v---------------------------------------+  |
|  | vLLM Serving Engine (Single-Node / Multi-Node Ray Cluster)                  |  |
|  |  - Tensor Parallelism (TP=4) & Pipeline Parallelism (PP=2)                    |  |
|  |  - High-throughput PagedAttention inference                                  |  |
|  |  - Model Weights (Qwen2.5-32B, Qwen3-235B-GPTQ, etc.)                        |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## 2. Key Components

### 2.1 Infrastructure Layer (`infrastructure_layer/`)
- **Apptainer Containerization (`defs/vllm.def`)**: Provides an isolated, reproducible container environment built on top of `vllm/vllm-openai`, configured for HPC cluster execution without root privileges.
- **Slurm Integration (`slurm/`)**: Standardized batch job scripts for single-GPU, multi-GPU Tensor Parallelism (TP4), and multi-node Ray cluster orchestration (TP4 + PP2).
- **Hugging Face Downloader (`slurm/download/`)**: Compute-node download scripts with hash verification and repair mechanisms to ensure zero GPU idle time during large model downloads.

### 2.2 Application Layer (`application_layer/`)
- **FastAPI Gateway Proxy (`app/main.py`)**: A lightweight gateway serving an OpenAI-compatible API spec.
- **Key & ACL Management**: Issues API keys formatted as `sk-hpc-...`. Raw keys are shown only once upon creation; only SHA-256 hashes are stored in SQLite (`gateway.db`).
- **Rate Limiting & ACL**: Per-key Requests Per Minute (RPM) enforcement using in-memory sliding windows, and per-key model access control lists (ACL).
- **Admin Dashboard**: Built-in HTML UI for token administration, key creation, usage monitoring, and key revocation.

---

## 3. End-to-End Request Flow

1. **Request Initiation**: Client sends an OpenAI-compatible HTTP request (e.g. `POST /v1/chat/completions`) to `http://127.0.0.1:9000/v1` with `Authorization: Bearer sk-hpc-...`.
2. **Gateway Processing**:
   - Computes SHA-256 hash of key and verifies against SQLite database.
   - Validates that the requested model is permitted under the key's ACL.
   - Checks rate limits (RPM). If exceeded, returns `429 Too Many Requests`.
3. **Upstream Forwarding**:
   - Resolves model alias to backend URL from `config/models.json` (e.g., `http://127.0.0.1:18000/v1`).
   - Streams or forwards HTTP payload through the local SSH tunnel to the active Slurm compute node running vLLM.
4. **Inference Execution**:
   - vLLM executes PagedAttention inference across allocated GPUs.
   - Streams response tokens back to Gateway proxy.
5. **Response Delivery**:
   - Gateway passes streaming response (Server-Sent Events) back to client while logging request metrics.
