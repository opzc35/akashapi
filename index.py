import os
import time
import uuid
import json
from typing import Any, Dict, List, Optional

import requests
from flask import Flask, request, jsonify


# =========================
# Config
# =========================

AKASH_SESSION_URL = "https://chat.akash.network/api/auth/session/"
AKASH_CHAT_URL = "https://chat.akash.network/api/chat/"

# 外部调用你这个代理时使用的 API Key（OpenAI 风格）
# 建议用环境变量：
#   export OPENAI_PROXY_API_KEY="sk-xxxx"
OPENAI_PROXY_API_KEY = os.environ.get("OPENAI_PROXY_API_KEY", "sk-your-secret-key-change-me")

# Flask app
app = Flask(__name__)

# shared session for keep-alive
_http = requests.Session()


# =========================
# Akash model list (static, from your JSON)
# =========================

AKASH_MODELS_RAW = [
    {
        "id": "DeepSeek-V3.2",
        "model_id": "DeepSeek-V3.2",
        "api_id": "deepseek-ai/DeepSeek-V3.2",
        "name": "DeepSeek V3.2",
        "description": "Advanced reasoning model with tool-calling and agentic capabilities",
        "tier_requirement": "permissionless",
        "available": True,
        "temperature": "1.00",
        "top_p": "0.95",
        "token_limit": 64000,
        "owned_by": None,
        "parameters": "685B",
        "architecture": "Mixture-of-Experts with DeepSeek Sparse Attention",
        "hf_repo": "deepseek-ai/DeepSeek-V3.2",
        "created_at": "2025-12-03T19:18:52.612Z",
        "updated_at": "2025-12-05T11:22:28.877Z"
    },
    {
        "id": "DeepSeek-V3.2-Speciale",
        "model_id": "DeepSeek-V3.2-Speciale",
        "api_id": "deepseek-ai/DeepSeek-V3.2-Speciale",
        "name": "DeepSeek V3.2 Speciale",
        "description": "Specialized deep reasoning model designed for complex mathematical and logical tasks",
        "tier_requirement": "permissionless",
        "available": True,
        "temperature": "1.00",
        "top_p": "0.95",
        "token_limit": 64000,
        "owned_by": None,
        "parameters": "685B",
        "architecture": "Transformer with DeepSeek Sparse Attention",
        "hf_repo": "deepseek-ai/DeepSeek-V3.2-Speciale",
        "created_at": "2025-12-03T13:43:01.825Z",
        "updated_at": "2025-12-05T11:22:22.327Z"
    },
    {
        "id": "Qwen/Qwen3-30B-A3B",
        "model_id": "Qwen/Qwen3-30B-A3B",
        "api_id": None,
        "name": "Qwen3 30B A3B",
        "description": "Efficient MoE model with 30.5B parameters (3.3B active)",
        "tier_requirement": "permissionless",
        "available": True,
        "temperature": "0.60",
        "top_p": "0.95",
        "token_limit": 32768,
        "owned_by": None,
        "parameters": None,
        "architecture": None,
        "hf_repo": "Qwen/Qwen3-30B-A3B",
        "created_at": "2025-11-24T18:08:54.575Z",
        "updated_at": "2025-12-04T17:57:18.081Z"
    },
    {
        "id": "DeepSeek-V3.1",
        "model_id": "DeepSeek-V3.1",
        "api_id": "deepseek-ai/DeepSeek-V3.1",
        "name": "DeepSeek V3.1",
        "description": "Next-generation reasoning model with enhanced capabilities",
        "tier_requirement": "permissionless",
        "available": True,
        "temperature": "0.60",
        "top_p": "0.95",
        "token_limit": 64000,
        "owned_by": None,
        "parameters": "685B",
        "architecture": "Mixture-of-Experts",
        "hf_repo": "deepseek-ai/DeepSeek-V3.1",
        "created_at": "2025-09-19T15:18:15.072Z",
        "updated_at": "2025-09-19T15:18:15.072Z"
    },
    {
        "id": "Meta-Llama-3-3-70B-Instruct",
        "model_id": "Meta-Llama-3-3-70B-Instruct",
        "api_id": "meta-llama/Llama-3.3-70B-Instruct",
        "name": "Llama 3.3 70B",
        "description": "Well-rounded model with strong capabilities",
        "tier_requirement": "permissionless",
        "available": True,
        "temperature": "0.60",
        "top_p": "0.90",
        "token_limit": 128000,
        "owned_by": None,
        "parameters": "70B",
        "architecture": "Transformer",
        "hf_repo": "meta-llama/Llama-3.3-70B-Instruct",
        "created_at": "2025-09-19T15:18:16.294Z",
        "updated_at": "2025-09-19T15:18:16.294Z"
    },
    {
        "id": "AkashGen",
        "model_id": "AkashGen",
        "api_id": "meta-llama/Llama-3.3-70B-Instruct",
        "name": "AkashGen",
        "description": "Generate images using AkashGen",
        "tier_requirement": "permissionless",
        "available": True,
        "temperature": "0.85",
        "top_p": "1.00",
        "token_limit": 12000,
        "owned_by": None,
        "parameters": None,
        "architecture": None,
        "hf_repo": None,
        "created_at": "2025-09-19T15:18:17.022Z",
        "updated_at": "2025-12-05T13:24:00.355Z"
    }
]

OPENAI_MODELS = [
    {
        "id": m["id"],
        "object": "model",
        "created": 0,
        "owned_by": m.get("owned_by") or "akash-chat",
    }
    for m in AKASH_MODELS_RAW
    if m.get("available") is True
]

SUPPORTED_MODEL_IDS = {m["id"] for m in OPENAI_MODELS}


# =========================
# OpenAI-style error response
# =========================

def openai_error(message: str, status_code: int = 400,
                err_type: str = "invalid_request_error",
                code: str = "invalid_request_error"):
    return jsonify({
        "error": {
            "message": message,
            "type": err_type,
            "param": None,
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
            code="invalid_api_key"
        )
    key = auth[len("Bearer "):].strip()
    if not key or key != OPENAI_PROXY_API_KEY:
        return openai_error(
            "Invalid API key",
            401,
            err_type="invalid_request_error",
            code="invalid_api_key"
        )
    return None


# =========================
# Akash auth: ALWAYS refresh session_token per request
# =========================

def get_session_token_every_time() -> str:
    """
    每次请求都重新获取 session_token（你要求的）
    """
    res = _http.get(AKASH_SESSION_URL, timeout=30)
    res.raise_for_status()

    token = res.cookies.get("session_token")
    if not token:
        # 兜底：从 session cookies 里取
        token = _http.cookies.get("session_token")

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
# OpenAI -> Akash conversion
# =========================

def openai_messages_to_akash_messages(openai_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    akash_messages: List[Dict[str, Any]] = []

    for m in openai_messages:
        role = m.get("role", "")
        content = m.get("content", "")

        # OpenAI 多模态兼容：content 为 list
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
    """
    尽可能抽取 assistant 输出
    你后续贴 Akash /api/chat 的真实返回结构，我可以把它改成完全精准。
    """
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

    # 模型校验（避免用户传错）
    model = openai_req.get("model", "")
    if model and model not in SUPPORTED_MODEL_IDS:
        return openai_error(
            f"Model '{model}' not found. Supported models: {sorted(SUPPORTED_MODEL_IDS)}",
            400,
            err_type="invalid_request_error",
            code="model_not_found"
        )

    if openai_req.get("stream"):
        return openai_error(
            "stream is not supported in this version (Flask minimal). If you want, I can add SSE streaming.",
            400,
            err_type="invalid_request_error",
            code="unsupported"
        )

    try:
        # 每次请求都刷新 session_token（你要求）
        token = get_session_token_every_time()
        headers = build_akash_headers(token)
        akash_body = openai_to_akash_body(openai_req)

        upstream = _http.post(AKASH_CHAT_URL, headers=headers, json=akash_body, timeout=120)
        upstream.raise_for_status()

        try:
            akash_resp = upstream.json()
        except Exception:
            akash_resp = upstream.text

        assistant_text = extract_assistant_text_from_akash(akash_resp)
        return jsonify(make_openai_chat_completion_response(model or "DeepSeek-V3.2", assistant_text, akash_resp))

    except requests.HTTPError as e:
        # 返回 OpenAI 风格的上游错误
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


@app.post("/v1/completions")
def v1_completions():
    """
    旧版 completions：把 prompt 转成 chat/completions
    """
    auth_err = require_bearer_auth()
    if auth_err:
        return auth_err

    req_json = request.get_json(force=True, silent=False)

    prompt = req_json.get("prompt", "")
    model = req_json.get("model", "DeepSeek-V3.2")
    temperature = req_json.get("temperature", 1.0)
    top_p = req_json.get("top_p", 0.95)

    if model and model not in SUPPORTED_MODEL_IDS:
        return openai_error(
            f"Model '{model}' not found. Supported models: {sorted(SUPPORTED_MODEL_IDS)}",
            400,
            err_type="invalid_request_error",
            code="model_not_found"
        )

    if isinstance(prompt, list):
        prompt = "\n".join([str(x) for x in prompt])

    openai_chat_req = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
    }

    try:
        token = get_session_token_every_time()
        headers = build_akash_headers(token)
        akash_body = openai_to_akash_body(openai_chat_req)

        upstream = _http.post(AKASH_CHAT_URL, headers=headers, json=akash_body, timeout=120)
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
    # debug=True 仅用于本地调试，上线建议关闭
    app.run(host="0.0.0.0", port=10006, debug=True)
