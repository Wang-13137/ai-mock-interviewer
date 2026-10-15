# -*- coding: utf-8 -*-
"""基准脚本自身的冒烟测试：保证 scripts/bench.py 不会悄悄坏掉。"""

import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bench import CallStat, call_plain, call_stream, summarize_mode  # noqa: E402

from src.app import build_client, create_app  # noqa: E402
from src.core.config import LLMConfig, MetricsConfig, ServerConfig, Settings  # noqa: E402
from src.metrics import JsonlSink  # noqa: E402

PAYLOAD = {"session_id": "t", "messages": [{"role": "user", "content": "介绍一下你自己"}]}


@pytest.fixture()
def bench_client():
    """进程内直连 app 的异步客户端（不发真实网络请求）。"""
    settings = Settings(
        env="test",
        llm=LLMConfig(provider="mock"),
        metrics=MetricsConfig(enabled=False),
        server=ServerConfig(),
    )
    app = create_app(settings)
    app.state.settings = settings
    app.state.client = build_client(settings)
    app.state.sink = JsonlSink("logs/test-metrics.jsonl", enabled=False)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://inline")


async def test_call_plain_collects_stats(bench_client):
    st = await call_plain(bench_client, PAYLOAD, "q01")
    assert isinstance(st, CallStat)
    assert st.ok is True
    assert st.total_ms >= 0
    assert st.chars > 0
    assert st.ttft_ms is None, "非流式在客户端拿不到独立的首字时间"


async def test_call_stream_collects_ttft(bench_client):
    st = await call_stream(bench_client, PAYLOAD, "q01")
    assert st.ok is True
    assert st.ttft_ms is not None
    assert st.total_ms >= st.ttft_ms
    assert st.chars > 0


# ---------------- 异常分支 ---------------- #
class _FakeResp:
    def __init__(self, status: int, lines=None):
        self.status_code = status
        self._lines = lines or []

    async def aread(self) -> bytes:
        return b"boom"

    def aiter_lines(self):
        async def gen():
            for line in self._lines:
                yield line
        return gen()


class _FakeStreamCtx:
    def __init__(self, resp: _FakeResp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


class _FakeClient:
    def __init__(self, resp: _FakeResp):
        self._resp = resp

    def stream(self, *args, **kwargs):
        return _FakeStreamCtx(self._resp)


async def test_stream_bad_status_is_captured():
    st = await call_stream(_FakeClient(_FakeResp(500)), PAYLOAD, "q01")
    assert st.ok is False
    assert "500" in st.error


async def test_stream_error_frame_is_captured():
    frame = "data: " + json.dumps({"type": "error", "error": "鉴权失败"}) + "\n\n"
    st = await call_stream(_FakeClient(_FakeResp(200, [frame])), PAYLOAD, "q01")
    assert st.ok is False
    assert "鉴权" in st.error


# ---------------- 统计 ---------------- #
def test_summarize_mode_handles_mixed():
    stats = [
        CallStat("stream", "q1", True, ttft_ms=10.0, total_ms=100.0, chars=50),
        CallStat("stream", "q2", True, ttft_ms=20.0, total_ms=200.0, chars=60),
        CallStat("stream", "q3", False, error="boom"),
    ]
    s = summarize_mode(stats)
    assert s["n"] == 3 and s["success"] == 2 and s["failed"] == 1
    assert s["client_ttft"]["p50"] == 15.0
    assert s["client_e2e"]["p50"] == 150.0
    assert s["chars_mean"] == 55.0


def test_summarize_mode_empty():
    s = summarize_mode([])
    assert s["n"] == 0 and s["client_ttft"]["p50"] is None
