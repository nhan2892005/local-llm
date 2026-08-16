# Application Layer Gateway API & Dashboard Guide

The application layer contains an OpenAI-compatible FastAPI gateway proxy (`application_layer/onenode/app/main.py`) that acts as a secure, feature-rich interface between client applications and the backend HPC vLLM cluster.

---

## 1. Setup & SSH Tunneling

### Step 1: Open SSH Tunnel to HPC Compute Node
Run the SSH tunnel script to securely bridge local port `18000` to port `8000` on the active Slurm compute node:

```bash
cd application_layer/onenode
./scripts/ssh-tunnel.sh
```

Override GPU node hostname or local port if required:

```bash
GPU_NODE=gpunode2.gitc.hpc LOCAL_PORT=18000 ./scripts/ssh-tunnel.sh
```

Verify connection:

```bash
curl http://127.0.0.1:18000/v1/models
```

### Step 2: Launch Local Gateway API

```bash
cd application_layer/onenode
./scripts/run.sh
```

The script automatically:
1. Creates a Python virtual environment (`.venv`) if missing.
2. Installs requirements from `requirements.txt`.
3. Creates `.env` with a securely generated `ADMIN_TOKEN` if not present.
4. Starts the FastAPI server on `127.0.0.1:9000`.

---

## 2. Configuration (`.env` and `models.json`)

### `.env` Parameters
```env
ADMIN_TOKEN=a_long_secure_random_secret_token
MODEL_CONFIG=./config/models.json
KEY_DB=./data/gateway.db
UPSTREAM_TIMEOUT=3600
LOG_LEVEL=INFO
```

### Model Registry (`config/models.json`)
Maps user-facing model IDs to upstream vLLM endpoints:

```json
{
  "models": [
    {
      "id": "qwen2.5-32b",
      "owned_by": "hpc-selfhosted",
      "base_url": "http://127.0.0.1:18000/v1",
      "upstream_model": "Qwen/Qwen2.5-32B-Instruct",
      "api_key": ""
    }
  ]
}
```

---

## 3. Web UI Dashboard & Admin Management

Navigate to `http://127.0.0.1:9000/` in your web browser to open the built-in Gateway Dashboard.

### Features:
- **Admin Authentication**: Enter `ADMIN_TOKEN` to access management actions.
- **Key Generation**: Create new API keys with specific names, allowed model lists, and custom RPM limits.
- **Key Revocation**: Instantly enable or disable API keys.
- **Usage Overview**: Monitor total API keys created and active key statistics.

---

## 4. Database Schema (`gateway.db`)

API keys are stored securely using SHA-256 hashes in SQLite.

```sql
CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key_hash TEXT NOT NULL UNIQUE,
    prefix TEXT NOT NULL,
    name TEXT NOT NULL,
    allowed_models TEXT NOT NULL,  -- JSON string or comma-separated list
    rpm INTEGER NOT NULL,          -- Requests Per Minute limit
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL
);
```

---

## 5. Supported API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/` | None | Web UI Dashboard |
| `GET` | `/v1/models` | Bearer Key | List available models permitted by key's ACL |
| `POST` | `/v1/chat/completions` | Bearer Key | OpenAI-compatible chat completion (streaming supported) |
| `POST` | `/v1/completions` | Bearer Key | OpenAI-compatible text completion |
| `GET` | `/admin/keys` | `X-Admin-Token` | List all registered API keys |
| `POST` | `/admin/keys` | `X-Admin-Token` | Issue a new `sk-hpc-...` API key |
| `POST` | `/admin/keys/revoke` | `X-Admin-Token` | Revoke/disable an API key |

---

## 6. Client Code Examples

### Python (using OpenAI SDK)
```python
import openai

client = openai.OpenAI(
    base_url="http://127.0.0.1:9000/v1",
    api_key="sk-hpc-YOUR_GENERATED_KEY"
)

response = client.chat.completions.create(
    model="qwen2.5-32b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant running on HPC."},
        {"role": "user", "content": "Explain parallel computing in 2 sentences."}
    ],
    temperature=0.7,
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

### cURL
```bash
curl http://127.0.0.1:9000/v1/chat/completions \
  -H "Authorization: Bearer sk-hpc-YOUR_GENERATED_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-32b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7
  }'
```
