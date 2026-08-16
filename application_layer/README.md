# Local HPC LLM API

Một gateway OpenAI-compatible chạy trên laptop của bạn.

## Kiến trúc

```text
Python app / browser
        |
        | http://127.0.0.1:9000/v1
        | Authorization: Bearer sk-hpc-...
        v
Local HPC LLM API (FastAPI)
        |
        | http://127.0.0.1:18000/v1
        v
SSH tunnel
        |
        v
gpunode1.gitc.hpc:8000
        |
        v
vLLM -> Qwen2.5-32B -> 4 x V100
```

## Chức năng

- Dashboard tạo API key
- API key dạng `sk-hpc-...`
- SQLite chỉ lưu SHA-256 hash của key
- Model ACL theo API key
- Rate limit RPM theo API key
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- Streaming passthrough
- OpenAI Python SDK compatible
- Request ID và access log
- Revoke key

## 1. Mở SSH tunnel

Terminal 1:

```bash
./scripts/ssh-tunnel.sh
```

Mặc định dùng SSH alias `pnhan`:

```text
127.0.0.1:18000 -> gpunode1.gitc.hpc:8000
```

Test:

```bash
curl http://127.0.0.1:18000/v1/models
```

Nếu model server ở node khác:

```bash
GPU_NODE=gpunode3.gitc.hpc ./scripts/ssh-tunnel.sh
```

## 2. Start local API

Terminal 2:

```bash
./scripts/run.sh
```

Lần đầu script sẽ:

- tạo `.venv`
- cài dependencies
- tạo `.env`
- generate `ADMIN_TOKEN`
- start gateway ở `127.0.0.1:9000`

Mở:

```text
http://127.0.0.1:9000/
```

## 3. Tạo API key trên dashboard

Xem admin token:

```bash
grep ADMIN_TOKEN .env
```

Paste token vào phần Admin của dashboard.

Ví dụ:

```text
Name: my-python-client
Models: qwen2.5-32b
RPM: 60
```

Bấm **Tạo API key**.

Bạn sẽ nhận:

```text
sk-hpc-...
```

Raw key chỉ xuất hiện một lần.

## 4. Test curl

```bash
export HPC_API_KEY='sk-hpc-...'
./examples/curl.sh
```

## 5. Python OpenAI SDK

```bash
export HPC_API_KEY='sk-hpc-...'
source .venv/bin/activate
python examples/python_client.py
```

Code application:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:9000/v1",
    api_key="sk-hpc-...",
)

response = client.chat.completions.create(
    model="qwen2.5-32b",
    messages=[
        {"role": "user", "content": "Hello"}
    ],
)

print(response.choices[0].message.content)
```

## 6. Thêm model khác

Sửa `config/models.json`:

```json
{
  "models": [
    {
      "id": "qwen2.5-32b",
      "base_url": "http://127.0.0.1:18000/v1",
      "upstream_model": "Qwen/Qwen2.5-32B-Instruct",
      "api_key": ""
    },
    {
      "id": "llama-70b",
      "base_url": "http://127.0.0.1:18001/v1",
      "upstream_model": "meta-llama/Llama-3.3-70B-Instruct",
      "api_key": ""
    }
  ]
}
```

Mỗi backend HPC có thể dùng một SSH tunnel local port khác nhau.

## 7. Endpoint

```text
GET    /
GET    /healthz
GET    /readyz

POST   /admin/keys
GET    /admin/keys
DELETE /admin/keys/{id}

GET    /v1/models
POST   /v1/chat/completions
POST   /v1/completions
```

Admin endpoints dùng:

```text
X-Admin-Token: <ADMIN_TOKEN>
```

Inference endpoints dùng:

```text
Authorization: Bearer sk-hpc-...
```

## Lưu ý production

Bản này là POC production-shaped chạy local trên một laptop.

Nếu public Internet thật:
- đặt HTTPS reverse proxy ở trước gateway;
- không bind gateway trực tiếp ra `0.0.0.0` nếu chưa có firewall/TLS;
- dùng Redis cho distributed rate limiting;
- dùng PostgreSQL nếu chạy nhiều gateway replicas;
- dùng secret manager cho admin token;
- không expose thẳng vLLM port 8000 ra Internet.
