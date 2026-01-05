import os
import time
import uuid
import json
from typing import Any, Dict, List, Optional

import requests
from flask import Flask, request, jsonify


# =========================
# Configuration
# =========================

# 你自己的 OpenAI 风格 API Key（外部客户端调用你代理时用）
# 建议用环境变量：export OPENAI_PROXY_API_KEY="sk-xxx"
OPENAI_PROXY_API_KEY = os.environ.get("OPENAI_PROXY_API_KEY", "sk-your-secret-key-change-me")

AKASH_SESSION_URL = "https://chat.akash.network/api/auth/session/"
AKASH_CHAT_URL = "https://chat.akash.network/api/chat/"

# 静态模型列表（如果你发现 Akash 有真实 models 接口，可再做转发）
STATIC_MODELS = [
    {"id": "DeepSeek-V3.2", "object": "model", "created": 0, "owned_by": "akash-chat"},
    {"id": "DeepSeek-R1", "object": "model", "created": 0, "owned_by": "akash-chat"},
]


app = Flask(__name__)


# =========================
# Auth helper
# =========================

def openai_error(message: str, status_code: int = 401, err_type: str = "invalid_request_error", code: str = "invalid_api_key"):
    """
    返回 OpenAI 风格错误结构
    """
    return jsonify({
        "error": {
            "message": message,
            "type": err_type,
            "param": None,
            "code": code
        }
    }), status_code


def require_bearer_auth():
    """
    校验 Authorization: Bearer <key>
    """
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return openai_error("Missing Authorization header. Expected: Authorization: Bearer <API_KEY>", 401)

    key = auth[len("Bearer "):].strip()
    if not key or key != OPENAI_PROXY_API_KEY:
        return openai_error("Invalid API key", 401)

    return None


# =========================
# Akash session token cache
# =========================

_session = requests.Session()
_session_token: Optional[str] = None
_session_token_ts: float = 0.0


def get_session_token(force_refresh: bool = False) -> str:
    global _session_token, _session_token_ts

    # 缓存 10 分钟（可以调）
    if (not force_refresh) and _session_token and (time.time() - _session_token_ts) < 600:
        return _session_token

    res = _session.get(AKASH_SESSION_URL, timeout=30)
    res.raise_for_status()

    token = res.cookies.get("session_token") or _session.cookies.get("session_token")
    if not token:
        raise RuntimeError("Failed to obtain session_token from Akash session endpoint")

    _session_token = token
    _session_token_ts = time.time()
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
# OpenAI -> Akash conversion
# =========================

def openai_messages_to_akash_messages(openai_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    akash_messages: List[Dict[str, Any]] = []

    for m in openai_messages:
        role = m.get("role", "")
        content = m.get("content", "")

        # 兼容 OpenAI 多模态 content=list
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
    temperature = openai_req.get("temperature", 1.0)
    top_p = openai_req.get("top_p", openai_req.get("topP", 0.95))

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
# Akash -> OpenAI conversion
# =========================

def extract_assistant_text_from_akash(akash_resp: Any) -> str:
    if isinstance(akash_resp, str):
        return akash_resp
    if not isinstance(akash_resp, dict):
        return str(akash_resp)

    for key in ("content", "text", "message", "output", "response"):
        if key in akash_resp and isinstance(akash_resp[key], str):
            return akash_resp[key]

    if "messages" in akash_resp and isinstance(akash_resp["messages"], list):
        for m in reversed(akash_resp["messages"]):
            if isinstance(m, dict) and m.get("role") == "assistant":
                if isinstance(m.get("content"), str) and m["content"]:
                    return m["content"]
                parts = m.get("parts", [])
                if isinstance(parts, list):
                    texts = []
                    for p in parts:
                        if isinstance(p, dict) and p.get("type") == "text":
                            texts.append(p.get("text", ""))
                    if texts:
                        return "\n".join(texts)

    return json.dumps(akash_resp, ensure_ascii=False)


def make_openai_chat_completion_response(model: str, assistant_text: str, raw_upstream: Any) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": assistant_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None
        },
        "akash_raw": raw_upstream
    }


# =========================
# Flask routes
# =========================

@app.get("/v1/models")
def v1_models():
    # 鉴权
    auth_err = require_bearer_auth()
    if auth_err:
        return auth_err
    return jsonify({"object": "list", "data": STATIC_MODELS})


@app.post("/v1/chat/completions")
def v1_chat_completions():
    # 鉴权
    auth_err = require_bearer_auth()
    if auth_err:
        return auth_err

    openai_req = request.get_json(force=True, silent=False)

    # 先不实现 stream（你要我也可以加 SSE stream）
    if openai_req.get("stream"):
        return openai_error("stream is not supported yet (Flask minimal version)", 400, code="unsupported")

    try:
        token = get_session_token()
        headers = build_akash_headers(token)
        akash_body = openai_to_akash_body(openai_req)

        upstream = _session.post(AKASH_CHAT_URL, headers=headers, json=akash_body, timeout=120)

        if upstream.status_code in (401, 403):
            token = get_session_token(force_refresh=True)
            headers = build_akash_headers(token)
            upstream = _session.post(AKASH_CHAT_URL, headers=headers, json=akash_body, timeout=120)

        upstream.raise_for_status()

        try:
            akash_resp = upstream.json()
        except Exception:
            akash_resp = upstream.text

        assistant_text = extract_assistant_text_from_akash(akash_resp)
        model = openai_req.get("model", "DeepSeek-V3.2")

        return jsonify(make_openai_chat_completion_response(model, assistant_text, akash_resp))

    except requests.HTTPError as e:
        return openai_error(f"Akash upstream HTTP error: {str(e)}", 502, err_type="server_error", code="upstream_error")
    except Exception as e:
        return openai_error(str(e), 500, err_type="server_error", code="internal_error")


@app.post("/v1/completions")
def v1_completions():
    # 鉴权
    auth_err = require_bearer_auth()
    if auth_err:
        return auth_err

    req_json = request.get_json(force=True, silent=False)

    prompt = req_json.get("prompt", "")
    model = req_json.get("model", "DeepSeek-V3.2")
    temperature = req_json.get("temperature", 1.0)
    top_p = req_json.get("top_p", 0.95)

    if isinstance(prompt, list):
        prompt = "\n".join([str(x) for x in prompt])

    # completions -> chat/completions
    openai_chat_req = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
    }

    # 复用 chat/completions 逻辑
    try:
        token = get_session_token()
        headers = build_akash_headers(token)
        akash_body = openai_to_akash_body(openai_chat_req)

        upstream = _session.post(AKASH_CHAT_URL, headers=headers, json=akash_body, timeout=120)

        if upstream.status_code in (401, 403):
            token = get_session_token(force_refresh=True)
            headers = build_akash_headers(token)
            upstream = _session.post(AKASH_CHAT_URL, headers=headers, json=akash_body, timeout=120)

        upstream.raise_for_status()

        try:
            akash_resp = upstream.json()
        except Exception:
            akash_resp = upstream.text

        assistant_text = extract_assistant_text_from_akash(akash_resp)

        return jsonify({
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "text": assistant_text, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None
            },
            "akash_raw": akash_resp
        })

    except requests.HTTPError as e:
        return openai_error(f"Akash upstream HTTP error: {str(e)}", 502, err_type="server_error", code="upstream_error")
    except Exception as e:
        return openai_error(str(e), 500, err_type="server_error", code="internal_error")


# =========================
# Run
# =========================

if __name__ == "__main__":
    # debug=True 仅本地调试用，上线建议关掉
    app.run(host="0.0.0.0", port=10006, debug=True)
