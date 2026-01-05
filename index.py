import time
import uuid
import json
from typing import Any, Dict, List, Optional

import requests
from flask import Flask, request, jsonify, Response


AKASH_SESSION_URL = "https://chat.akash.network/api/auth/session/"
AKASH_CHAT_URL = "https://chat.akash.network/api/chat/"


# 你可以把模型维护在这里（如果后续你发现 Akash 有 models API，也可以改成转发真实接口）
STATIC_MODELS = [
    {
        "id": "DeepSeek-V3.2",
        "object": "model",
        "created": 0,
        "owned_by": "akash-chat",
    }
]


app = Flask(__name__)


# -----------------------
# Akash session token cache
# -----------------------
_session = requests.Session()
_session_token: Optional[str] = None
_session_token_ts: float = 0.0


def get_session_token(force_refresh: bool = False) -> str:
    """
    获取 Akash session_token（带缓存）
    """
    global _session_token, _session_token_ts

    # 缓存 10 分钟（可自行调整）
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
    """
    构造 Akash headers（包含你截图里的字段）+ Cookie
    """
    return {
        "accept": "*/*",
        "Accept-Language": "en-US",
        "Content-Type": "application/json",

        "sec-ch-ua": "\"Google Chrome\";v=\"143\", \"Chromium\";v=\"143\", \"Not A(Brand\";v=\"24\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",

        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",

        # 推荐加上，更像浏览器（避免 403）
        "Origin": "https://chat.akash.network",
        "Referer": "https://chat.akash.network/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",

        # 关键 Cookie
        "Cookie": f"session_token={session_token};",
    }


# -----------------------
# OpenAI -> Akash conversion
# -----------------------

def openai_messages_to_akash_messages(openai_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    OpenAI messages -> Akash messages(parts)
    """
    akash_messages: List[Dict[str, Any]] = []

    for m in openai_messages:
        role = m.get("role", "")
        content = m.get("content", "")

        # 兼容 OpenAI 多模态: content 是 list
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            content = "\n".join(text_parts)

        msg = {
            "role": role,
            "content": content,
            "parts": []
        }

        if role == "assistant":
            msg["parts"].append({"type": "step-start"})
            msg["parts"].append({"type": "text", "text": content})
        else:
            msg["parts"].append({"type": "text", "text": content})

        akash_messages.append(msg)

    return akash_messages


def extract_system_text(openai_req: Dict[str, Any]) -> str:
    """
    system 文本来源：
    - openai_req["system"]（如果你传）
    - 或 messages 里 role=system 的第一条
    """
    if isinstance(openai_req.get("system"), str) and openai_req["system"]:
        return openai_req["system"]

    for m in openai_req.get("messages", []):
        if m.get("role") == "system":
            return m.get("content", "") or ""

    return ""


def openai_to_akash_body(openai_req: Dict[str, Any]) -> Dict[str, Any]:
    """
    OpenAI chat/completions -> Akash /api/chat/ body
    """
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


# -----------------------
# Akash -> OpenAI conversion
# -----------------------

def extract_assistant_text_from_akash(akash_resp: Any) -> str:
    """
    尽可能从 Akash 响应抽取 assistant 文本
    你如果贴真实返回结构，我可以把这里改到 100% 精确。
    """
    if isinstance(akash_resp, str):
        return akash_resp
    if not isinstance(akash_resp, dict):
        return str(akash_resp)

    # 常见字段
    for key in ("content", "text", "message", "output", "response"):
        if key in akash_resp and isinstance(akash_resp[key], str):
            return akash_resp[key]

    # 尝试从 messages 里取最后一个 assistant
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

    # fallback：把整个 dict dump
    return json.dumps(akash_resp, ensure_ascii=False)


def make_openai_chat_completion_response(model: str, assistant_text: str, raw_upstream: Any) -> Dict[str, Any]:
    """
    构造 OpenAI chat/completions 标准返回
    """
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
        # 方便调试：你可以删掉
        "akash_raw": raw_upstream
    }


# -----------------------
# Flask endpoints (OpenAI-compatible)
# -----------------------

@app.get("/v1/models")
def v1_models():
    return jsonify({"object": "list", "data": STATIC_MODELS})


@app.post("/v1/chat/completions")
def v1_chat_completions():
    openai_req = request.get_json(force=True, silent=False)

    # 先不支持 stream（要我也能给你实现 SSE 流式）
    if openai_req.get("stream"):
        return jsonify({"error": {"message": "stream is not supported in this Flask minimal version yet"}}), 400

    try:
        token = get_session_token()
        headers = build_akash_headers(token)
        akash_body = openai_to_akash_body(openai_req)

        upstream = _session.post(AKASH_CHAT_URL, headers=headers, json=akash_body, timeout=120)

        # token 失效：强制刷新重试一次
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
        return jsonify({"error": {"message": f"Akash upstream HTTP error: {str(e)}"}}), 502
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 500


@app.post("/v1/completions")
def v1_completions():
    """
    旧版 completions 接口（很多 OpenAI 客户端还会用）
    我们将 prompt 转为 chat/completions：
    - system: 可选
    - user: prompt
    """
    req_json = request.get_json(force=True, silent=False)

    prompt = req_json.get("prompt", "")
    model = req_json.get("model", "DeepSeek-V3.2")
    temperature = req_json.get("temperature", 1.0)
    top_p = req_json.get("top_p", 0.95)

    # prompt 可能是 list
    if isinstance(prompt, list):
        prompt = "\n".join([str(x) for x in prompt])

    openai_chat_req = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
    }

    # 直接复用 chat/completions 逻辑
    request.environ["openai_chat_req_override"] = openai_chat_req
    return v1_chat_completions_from_override(openai_chat_req)


def v1_chat_completions_from_override(openai_req: Dict[str, Any]):
    """
    给 /v1/completions 复用的内部函数
    """
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

        # completions 的返回格式稍微不同
        return jsonify({
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": openai_req.get("model", "DeepSeek-V3.2"),
            "choices": [
                {
                    "index": 0,
                    "text": assistant_text,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None
            },
            "akash_raw": akash_resp
        })

    except requests.HTTPError as e:
        return jsonify({"error": {"message": f"Akash upstream HTTP error: {str(e)}"}}), 502
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 500


# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    # 0.0.0.0 表示允许局域网访问
    app.run(host="0.0.0.0", port=8000, debug=True)
