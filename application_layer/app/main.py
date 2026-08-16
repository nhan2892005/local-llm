import hashlib
import json
import logging
import os
import secrets
import sqlite3
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("KEY_DB", str(APP_DIR / "data" / "gateway.db")))
MODEL_CONFIG = Path(os.getenv("MODEL_CONFIG", str(APP_DIR / "config" / "models.json")))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
UPSTREAM_TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT", "3600"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("local-llm-api")


def load_models() -> dict[str, dict[str, Any]]:
    data = json.loads(MODEL_CONFIG.read_text())
    models = {}
    for item in data.get("models", []):
        models[item["id"]] = item
    if not models:
        raise RuntimeError(f"No models configured in {MODEL_CONFIG}")
    return models


MODELS = load_models()


def db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT NOT NULL UNIQUE,
                prefix TEXT NOT NULL,
                name TEXT NOT NULL,
                allowed_models TEXT NOT NULL,
                rpm INTEGER NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at INTEGER NOT NULL
            )
            """
        )
        conn.commit()


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def bearer(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Missing Bearer API key",
                    "type": "authentication_error",
                }
            },
        )
    return authorization.split(" ", 1)[1].strip()


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="ADMIN_TOKEN is not configured")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid admin token")


class Identity(BaseModel):
    id: int
    prefix: str
    name: str
    allowed_models: list[str]
    rpm: int


def require_api_key(authorization: str | None = Header(default=None)) -> Identity:
    raw = bearer(authorization)
    with db() as conn:
        row = conn.execute(
            """
            SELECT id, prefix, name, allowed_models, rpm
            FROM api_keys
            WHERE key_hash = ? AND enabled = 1
            """,
            (sha256(raw),),
        ).fetchone()

    if not row:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid API key",
                    "type": "authentication_error",
                }
            },
        )

    return Identity(
        id=row["id"],
        prefix=row["prefix"],
        name=row["name"],
        allowed_models=json.loads(row["allowed_models"]),
        rpm=row["rpm"],
    )


WINDOWS: dict[int, deque[float]] = defaultdict(deque)


def enforce_rpm(identity: Identity) -> None:
    now = time.monotonic()
    q = WINDOWS[identity.id]
    while q and now - q[0] >= 60:
        q.popleft()
    if len(q) >= identity.rpm:
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "message": f"Rate limit exceeded: {identity.rpm} requests/minute",
                    "type": "rate_limit_error",
                }
            },
            headers={"Retry-After": "60"},
        )
    q.append(now)


def allowed(identity: Identity, model_id: str) -> bool:
    return "*" in identity.allowed_models or model_id in identity.allowed_models


class CreateKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    allowed_models: list[str] = Field(default_factory=lambda: ["*"])
    rpm: int = Field(default=60, ge=1, le=100000)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    log.info("started db=%s models=%s", DB_PATH, list(MODELS))
    yield


app = FastAPI(
    title="Local HPC LLM API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    results = {}
    async with httpx.AsyncClient(timeout=5) as client:
        for model_id, cfg in MODELS.items():
            try:
                r = await client.get(cfg["base_url"].rstrip("/") + "/models")
                results[model_id] = {
                    "reachable": r.status_code == 200,
                    "status_code": r.status_code,
                }
            except Exception as exc:
                results[model_id] = {
                    "reachable": False,
                    "error": type(exc).__name__,
                }
    return {"status": "ready", "backends": results}


@app.post("/admin/keys", dependencies=[Depends(require_admin)])
def create_key(body: CreateKeyRequest):
    for m in body.allowed_models:
        if m != "*" and m not in MODELS:
            raise HTTPException(status_code=400, detail=f"Unknown model: {m}")

    raw = "sk-hpc-" + secrets.token_urlsafe(32)
    prefix = raw[:16]

    with db() as conn:
        cur = conn.execute(
            """
            INSERT INTO api_keys
                (key_hash, prefix, name, allowed_models, rpm, enabled, created_at)
            VALUES (?, ?, ?, ?, ?, 1, ?)
            """,
            (
                sha256(raw),
                prefix,
                body.name,
                json.dumps(body.allowed_models),
                body.rpm,
                int(time.time()),
            ),
        )
        conn.commit()
        key_id = cur.lastrowid

    return {
        "id": key_id,
        "api_key": raw,
        "prefix": prefix,
        "name": body.name,
        "allowed_models": body.allowed_models,
        "rpm": body.rpm,
        "note": "Save this key now. The raw key is not stored and cannot be shown again.",
    }


@app.get("/admin/keys", dependencies=[Depends(require_admin)])
def list_keys():
    with db() as conn:
        rows = conn.execute(
            """
            SELECT id, prefix, name, allowed_models, rpm, enabled, created_at
            FROM api_keys ORDER BY id DESC
            """
        ).fetchall()

    return {
        "data": [
            {
                "id": r["id"],
                "prefix": r["prefix"],
                "name": r["name"],
                "allowed_models": json.loads(r["allowed_models"]),
                "rpm": r["rpm"],
                "enabled": bool(r["enabled"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]
    }


@app.delete("/admin/keys/{key_id}", dependencies=[Depends(require_admin)])
def revoke_key(key_id: int):
    with db() as conn:
        cur = conn.execute("UPDATE api_keys SET enabled = 0 WHERE id = ?", (key_id,))
        conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Key not found")
    return {"id": key_id, "revoked": True}


@app.get("/v1/models")
def models(identity: Identity = Depends(require_api_key)):
    data = []
    for model_id, cfg in MODELS.items():
        if allowed(identity, model_id):
            data.append(
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": cfg.get("owned_by", "self-hosted"),
                }
            )
    return {"object": "list", "data": data}


async def proxy_openai(
    request: Request,
    endpoint: str,
    identity: Identity,
    x_request_id: str | None,
):
    enforce_rpm(identity)

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    public_model = payload.get("model")
    if not public_model or public_model not in MODELS:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Unknown model: {public_model}",
                    "type": "invalid_request_error",
                }
            },
        )
    if not allowed(identity, public_model):
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "message": f"API key cannot access model '{public_model}'",
                    "type": "permission_error",
                }
            },
        )

    cfg = MODELS[public_model]
    rid = x_request_id or f"req_{secrets.token_hex(12)}"
    upstream_payload = dict(payload)
    upstream_payload["model"] = cfg["upstream_model"]

    upstream_url = cfg["base_url"].rstrip("/") + endpoint
    headers = {"Content-Type": "application/json", "X-Request-ID": rid}
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg['api_key']}"

    started = time.perf_counter()
    is_stream = bool(payload.get("stream", False))

    log.info(
        "start request_id=%s key=%s model=%s endpoint=%s stream=%s",
        rid, identity.prefix, public_model, endpoint, is_stream,
    )

    if not is_stream:
        try:
            async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
                resp = await client.post(upstream_url, json=upstream_payload, headers=headers)
        except httpx.HTTPError as exc:
            log.exception("upstream error request_id=%s", rid)
            return JSONResponse(
                status_code=502,
                headers={"X-Request-ID": rid},
                content={
                    "error": {
                        "message": f"Upstream unavailable: {type(exc).__name__}",
                        "type": "upstream_error",
                    }
                },
            )

        elapsed = (time.perf_counter() - started) * 1000
        log.info(
            "end request_id=%s key=%s model=%s status=%s latency_ms=%.1f",
            rid, identity.prefix, public_model, resp.status_code, elapsed,
        )

        try:
            body = resp.json()
        except Exception:
            body = {
                "error": {
                    "message": resp.text,
                    "type": "upstream_error",
                }
            }

        if isinstance(body, dict) and body.get("model"):
            body["model"] = public_model

        return JSONResponse(
            status_code=resp.status_code,
            headers={"X-Request-ID": rid},
            content=body,
        )

    client = httpx.AsyncClient(timeout=None)
    upstream_request = client.build_request(
        "POST", upstream_url, json=upstream_payload, headers=headers
    )

    try:
        resp = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        await client.aclose()
        return JSONResponse(
            status_code=502,
            headers={"X-Request-ID": rid},
            content={
                "error": {
                    "message": f"Upstream unavailable: {type(exc).__name__}",
                    "type": "upstream_error",
                }
            },
        )

    if resp.status_code >= 400:
        raw = await resp.aread()
        await resp.aclose()
        await client.aclose()
        try:
            body = json.loads(raw)
        except Exception:
            body = {"error": {"message": raw.decode("utf-8", "replace")}}
        return JSONResponse(
            status_code=resp.status_code,
            headers={"X-Request-ID": rid},
            content=body,
        )

    async def stream_iter():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()
            elapsed = (time.perf_counter() - started) * 1000
            log.info(
                "stream_end request_id=%s key=%s model=%s latency_ms=%.1f",
                rid, identity.prefix, public_model, elapsed,
            )

    return StreamingResponse(
        stream_iter(),
        media_type=resp.headers.get("content-type", "text/event-stream"),
        headers={
            "X-Request-ID": rid,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    identity: Identity = Depends(require_api_key),
    x_request_id: str | None = Header(default=None),
):
    return await proxy_openai(request, "/chat/completions", identity, x_request_id)


@app.post("/v1/completions")
async def completions(
    request: Request,
    identity: Identity = Depends(require_api_key),
    x_request_id: str | None = Header(default=None),
):
    return await proxy_openai(request, "/completions", identity, x_request_id)


DASHBOARD_HTML = r"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Local HPC LLM API</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 1050px; margin: 32px auto; padding: 0 18px; background:#fafafa; color:#111; }
.card { background:white; border:1px solid #ddd; border-radius:12px; padding:18px; margin-bottom:18px; }
input, select, textarea, button { width:100%; box-sizing:border-box; padding:10px; margin:6px 0 12px; font:inherit; }
button { cursor:pointer; }
.row { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
pre { white-space:pre-wrap; word-break:break-word; background:#f3f3f3; padding:14px; border-radius:8px; }
small { color:#666; }
code { background:#eee; padding:2px 5px; border-radius:4px; }
</style>
</head>
<body>
<h1>Local HPC LLM API</h1>
<p>Gateway chạy trên laptop, inference chạy trên HPC qua SSH tunnel.</p>

<div class="card">
<h2>1. Admin</h2>
<label>ADMIN_TOKEN</label>
<input id="adminToken" type="password" placeholder="admin token từ file .env">
<div class="row">
  <div>
    <label>Tên API key</label>
    <input id="keyName" value="my-python-client">
  </div>
  <div>
    <label>RPM</label>
    <input id="rpm" type="number" value="60">
  </div>
</div>
<label>Models</label>
<input id="allowedModels" value="qwen2.5-32b">
<button onclick="createKey()">Tạo API key</button>
<button onclick="listKeys()">Xem keys</button>
<pre id="adminOut">Chưa có dữ liệu.</pre>
<small>Raw key chỉ được hiển thị một lần khi tạo.</small>
</div>

<div class="card">
<h2>2. Playground</h2>
<label>API key</label>
<input id="apiKey" type="password" placeholder="sk-hpc-...">
<button onclick="loadModels()">Load models</button>
<label>Model</label>
<select id="model"></select>
<label>Prompt</label>
<textarea id="prompt">Giải thích Tensor Parallelism trong 5 câu.</textarea>
<button onclick="sendChat()">Gửi request</button>
<pre id="chatOut">Chưa có response.</pre>
</div>

<div class="card">
<h2>3. Python</h2>
<pre>from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:9000/v1",
    api_key="sk-hpc-...",
)

r = client.chat.completions.create(
    model="qwen2.5-32b",
    messages=[{"role": "user", "content": "Hello"}],
)

print(r.choices[0].message.content)</pre>
</div>

<script>
function adminHeaders() {
  return {
    "Content-Type": "application/json",
    "X-Admin-Token": document.getElementById("adminToken").value
  };
}

async function createKey() {
  const models = document.getElementById("allowedModels").value
    .split(",").map(x => x.trim()).filter(Boolean);
  const body = {
    name: document.getElementById("keyName").value,
    allowed_models: models,
    rpm: parseInt(document.getElementById("rpm").value || "60")
  };
  const r = await fetch("/admin/keys", {
    method: "POST",
    headers: adminHeaders(),
    body: JSON.stringify(body)
  });
  const data = await r.json();
  document.getElementById("adminOut").textContent = JSON.stringify(data, null, 2);
  if (data.api_key) {
    document.getElementById("apiKey").value = data.api_key;
  }
}

async function listKeys() {
  const r = await fetch("/admin/keys", { headers: adminHeaders() });
  const data = await r.json();
  document.getElementById("adminOut").textContent = JSON.stringify(data, null, 2);
}

async function loadModels() {
  const key = document.getElementById("apiKey").value;
  const r = await fetch("/v1/models", {
    headers: { "Authorization": `Bearer ${key}` }
  });
  const data = await r.json();
  const out = document.getElementById("chatOut");
  if (!r.ok) {
    out.textContent = JSON.stringify(data, null, 2);
    return;
  }
  const select = document.getElementById("model");
  select.innerHTML = "";
  for (const m of data.data) {
    const o = document.createElement("option");
    o.value = m.id;
    o.textContent = m.id;
    select.appendChild(o);
  }
  out.textContent = "Models loaded.";
}

async function sendChat() {
  const key = document.getElementById("apiKey").value;
  const model = document.getElementById("model").value;
  const prompt = document.getElementById("prompt").value;
  const out = document.getElementById("chatOut");
  out.textContent = "...";
  const r = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${key}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model,
      messages: [{role:"user", content:prompt}],
      temperature:0.2,
      max_tokens:512
    })
  });
  const data = await r.json();
  out.textContent = r.ok
    ? (data.choices?.[0]?.message?.content ?? JSON.stringify(data, null, 2))
    : JSON.stringify(data, null, 2);
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML
