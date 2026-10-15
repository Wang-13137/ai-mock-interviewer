# -*- coding: utf-8 -*-
"""LLM 客户端测试：离线替身 + 用 MockTransport 模拟 OpenAI 兼容协议（不发真实请求）。"""

import json

import httpx
import pytest

from src.clients.llm import EchoClient, LLMError, OpenAICompatClient

SSE_BODY = (
    'data: {"choices":[{"delta":{"content":"你"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"好"}}]}\n\n'
    "data: [DONE]\n\n"
)


def _json_response(payload: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


async def test_echo_client_chat():
    client = EchoClient()
    res = await client.achat([{"role": "user", "content": "我想做 LLM 应用"}])
    assert "想做 LLM 应用" in res.content
    assert res.usage is not None and res.usage.total_tokens > 0
    assert client.ready is False


async def test_echo_client_stream_concatenates():
    client = EchoClient()
    chunks = [c async for c in client.astream([{"role": "user", "content": "自我介绍"}])]
    text = "".join(chunks)
    assert len(chunks) > 1, "应当分片产出，验证前端流式渲染逻辑"
    assert text.startswith("[离线替身]")


async def test_openai_compat_non_stream():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return _json_response(
            {
                "choices": [{"message": {"content": "请介绍一下你自己"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 7},
            }
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = OpenAICompatClient(
            model="glm-4-flash", base_url="https://example.com/v1", api_key="k1", client=http
        )
        res = await client.achat([{"role": "user", "content": "hi"}])

    assert res.content == "请介绍一下你自己"
    assert res.usage.prompt_tokens == 12
    assert captured["body"]["stream"] is False
    assert captured["auth"] == "Bearer k1"


async def test_openai_compat_stream_parses_sse():
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, text=SSE_BODY, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = OpenAICompatClient(
            model="glm-4-flash", base_url="https://example.com/v1", api_key="k1", client=http
        )
        chunks = [c async for c in client.astream([{"role": "user", "content": "hi"}])]

    assert chunks == ["你", "好"]


async def test_http_error_is_wrapped():
    transport = httpx.MockTransport(lambda req: httpx.Response(401, json={"error": "bad key"}))
    async with httpx.AsyncClient(transport=transport) as http:
        client = OpenAICompatClient(
            model="m", base_url="https://example.com/v1", api_key="wrong", client=http
        )
        with pytest.raises(LLMError) as exc:
            await client.achat([{"role": "user", "content": "hi"}])
    assert exc.value.status_code == 401
    assert "Key" in str(exc.value) or "鉴权" in str(exc.value)


def test_missing_base_url_rejected():
    with pytest.raises(ValueError):
        OpenAICompatClient(model="m", base_url="", api_key="k")
