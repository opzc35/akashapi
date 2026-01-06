import os
import time
import uuid
import json
from typing import Any, Dict, List, Generator, Tuple

import requests
from flask import Flask, request, jsonify, Response


# =========================
# Config
# =========================

AKASH_SESSION_URL = "https://chat.akash.network/api/auth/session/"
AKASH_CHAT_URL = "https://chat.akash.network/api/chat/"

# 外部调用你的代理时使用的 Bearer Key
OPENAI_PROXY_API_KEY = 'default_key'

# SOCKS 代理：通过环境变量配置
# 示例：
#   export SOCKS_PROXY="socks5h://127.0.0.1:7890"
# 或带账号密码：
#   export SOCKS_PROXY="socks5h://user:pass@127.0.0.1:7890"
SOCKS_PROXY = os.environ.get("SOCKS_PROXY", "").strip()

# Flask
app = Flask(__name__)

# requests Session（keep-alive）
_http = requests.Session()

# 如果配置了 SOCKS 代理，就启用
if SOCKS_PROXY:
    _http.proxies.update({
        "http": SOCKS_PROXY,
        "https": SOCKS_PROXY,
    })


# =========================
# Models (static)
# =========================
OPENAI_MODELS = [
    {"id": "DeepSeek-V3.2", "object": "model", "created": 0, "owned_by": "akash-chat"},
    {"id": "DeepSeek-V3.2-Speciale", "object": "model", "created": 0, "owned_by": "akash-chat"},
    {"id": "Qwen/Qwen3-30B-A3B", "object": "model", "created": 0, "owned_by": "akash-chat"},
    {"id": "DeepSeek-V3.1", "object": "model", "created": 0, "owned_by": "akash-chat"},
    {"id": "Meta-Llama-3-3-70B-Instruct", "object": "model", "created": 0, "owned_by": "akash-chat"},
    {"id": "AkashGen", "object": "model", "created": 0, "owned_by": "akash-chat"},
]
SUPPORTED_MODEL_IDS = {m["id"] for m in OPENAI_MODELS}


# =========================
# OpenAI-style error
# =========================

def openai_error(message: str, status_code: int = 400,
                 err_type: str = "invalid_request_error",
                 code: str = "invalid_request_error",
                 param: str = ""):
    return jsonify({
        "error": {
            "message": message,
            "type": err_type,
            "param": param,
            "code": code
        }
    }), status_code


# =========================
# Bearer auth
# =========================

def require_bearer_auth():
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return openai_error(
            "Missing Authorization header. Expected: Authorization: Bearer <API_KEY>",
            401,
            err_type="invalid_request_error",
            code="invalid_api_key",
        )
    key = auth[len("Bearer "):].strip()
    if not key or key != OPENAI_PROXY_API_KEY:
        return openai_error(
            "Invalid API key",
            401,
            err_type="invalid_request_error",
            code="invalid_api_key",
        )
    return None


# =========================
# Request cleaning
# =========================

def is_undefined(v: Any) -> bool:
    return v is None or v == "" or v == "[undefined]"


def pick_float(req: Dict[str, Any], key: str, default: float) -> float:
    v = req.get(key, default)
    if is_undefined(v):
        return default
    try:
        return float(v)
    except Exception:
        return default


# =========================
# Akash auth: ALWAYS refresh session_token per request
# =========================

def get_session_token_every_time() -> str:
    """
    每次请求都获取新的 session_token。
    注意：这个请求也会走 SOCKS_PROXY（如果你设置了）。
    """
    res = _http.get(AKASH_SESSION_URL, timeout=30)
    res.raise_for_status()
    token = res.cookies.get("session_token") or _http.cookies.get("session_token")
    if not token:
        raise RuntimeError("session_token not found in Akash session response")
    return token


def build_akash_headers(session_token: str) -> Dict[str, str]:
    return {
        "accept": "*/*",
        "Accept-Language": "en-US",
        "Content-Type": "application/json",

        "sec-ch-ua": "\"Google Chrome\";v=\"143\", \"Chromium\";v=\"143\", \"Not A(Brand\";v=\"24\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",

        "Origin": "https://chat.akash.network",
        "Referer": "https://chat.akash.network/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",

        "Cookie": f"session_token={session_token};",
    }


# =========================
# OpenAI -> Akash body
# =========================

def openai_messages_to_akash_messages(openai_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    akash_messages: List[Dict[str, Any]] = []
    for m in openai_messages:
        role = m.get("role", "")
        content = m.get("content", "")

        # OpenAI 多模态兼容：content=list
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            content = "\n".join(text_parts)

        msg = {"role": role, "content": content, "parts": []}
        if role == "assistant":
            msg["parts"].append({"type": "step-start"})
            msg["parts"].append({"type": "text", "text": content})
        else:
            msg["parts"].append({"type": "text", "text": content})
        akash_messages.append(msg)
    return akash_messages


def extract_system_text(openai_req: Dict[str, Any]) -> str:
    if isinstance(openai_req.get("system"), str) and openai_req["system"]:
        return openai_req["system"]
    for m in openai_req.get("messages", []):
        if m.get("role") == "system":
            return m.get("content", "") or ""
    return ""


def openai_to_akash_body(openai_req: Dict[str, Any]) -> Dict[str, Any]:
    model = openai_req.get("model", "DeepSeek-V3.2")
    temperature = pick_float(openai_req, "temperature", 1.0)
    top_p = pick_float(openai_req, "top_p", pick_float(openai_req, "topP", 0.95))

    return {
        "id": openai_req.get("id") or uuid.uuid4().hex[:16],
        "messages": openai_messages_to_akash_messages(openai_req.get("messages", [])),
        "model": model,
        "system": extract_system_text(openai_req),
        "temperature": f"{float(temperature):.2f}",
        "topP": f"{float(top_p):.2f}",
        "context": openai_req.get("context", []),
    }


# =========================
# Akash stream parser
# =========================

def parse_akash_line(line: str) -> Tuple[str, Any]:
    line = line.strip()
    if not line:
        return ("unknown", None)

    if line.startswith("f:"):
        raw = line[2:].strip()
        try:
            return ("start", json.loads(raw))
        except Exception:
            return ("start", raw)

    if line.startswith("0:"):
        raw = line[2:].strip()
        try:
            tok = json.loads(raw)
            if isinstance(tok, str):
                return ("token", tok)
            return ("token", str(tok))
        except Exception:
            return ("token", raw.strip("\""))

    if line.startswith("d:"):
        raw = line[2:].strip()
        try:
            return ("done", json.loads(raw))
        except Exception:
            return ("done", raw)

    return ("unknown", line)


def sse_event(obj: Any) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def openai_chunk(chunk_id: str, model: str, delta: Dict[str, Any], finish_reason: Any = None,
                 created: Optional[int] = None) -> Dict[str, Any]:
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created or int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason
        }]
    }


def akash_stream_to_openai_sse(upstream: requests.Response, model: str) -> Generator[str, None, None]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    # role chunk
    yield sse_event(openai_chunk(chunk_id, model, {"role": "assistant"}, None, created))

    for raw in upstream.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.strip()
        if not line:
            continue

        kind, payload = parse_akash_line(line)

        if kind == "token":
            yield sse_event(openai_chunk(chunk_id, model, {"content": payload}, None, created))

        elif kind == "done":
            yield sse_event(openai_chunk(chunk_id, model, {}, "stop", created))
            yield "data: [DONE]\n\n"
            return

    # upstream ended unexpectedly
    yield sse_event(openai_chunk(chunk_id, model, {}, "stop", created))
    yield "data: [DONE]\n\n"


# =========================
# Routes
# =========================

@app.get("/v1/models")
def v1_models():
    auth_err = require_bearer_auth()
    if auth_err:
        return auth_err
    return jsonify({"object": "list", "data": OPENAI_MODELS})


@app.post("/v1/chat/completions")
def v1_chat_completions():
    auth_err = require_bearer_auth()
    if auth_err:
        return auth_err

    openai_req = request.get_json(force=True, silent=False)

    model = openai_req.get("model", "DeepSeek-V3.2")
    if model not in SUPPORTED_MODEL_IDS:
        return openai_error(
            f"Model '{model}' not found. Supported models: {sorted(SUPPORTED_MODEL_IDS)}",
            400,
            err_type="invalid_request_error",
            code="model_not_found",
            param="model"
        )

    want_stream = bool(openai_req.get("stream", False))

    try:
        # 每次请求都刷新 session_token
        token = get_session_token_every_time()
        headers = build_akash_headers(token)
        akash_body = openai_to_akash_body(openai_req)

        upstream = _http.post(
            AKASH_CHAT_URL,
            headers=headers,
            json=akash_body,
            stream=True,          # 关键：上游流式
            timeout=120
        )
        upstream.raise_for_status()

        if want_stream:
            gen = akash_stream_to_openai_sse(upstream, model)
            return Response(
                gen,
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                }
            )

        # 如果客户端不想 stream，就收集拼接（这里简单处理：读取完再返回）
        # 你也可以后续加更好的 usage 统计对齐
        parts = []
        for raw in upstream.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            line = raw.strip()
            if not line:
                continue
            kind, payload = parse_akash_line(line)
            if kind == "token":
                parts.append(payload)
            elif kind == "done":
                break

        full_text = "".join(parts)
        resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": "stop"}
            ],
            "usage": None
        }
        return jsonify(resp)

    except requests.HTTPError as e:
        return openai_error(
            f"Akash upstream HTTP error: {str(e)}",
            502,
            err_type="server_error",
            code="upstream_error"
        )
    except Exception as e:
        return openai_error(
            str(e),
            500,
            err_type="server_error",
            code="internal_error"
        )


# =========================
# Run
# =========================

if __name__ == "__main__":
    # 生产建议 debug=False
    app.run(host="0.0.0.0", port=10006, debug=True, threaded=True)
