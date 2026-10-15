# -*- coding: utf-8 -*-
"""FastAPI 端点的端到端测试（跑在离线替身上）。"""


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["provider"] == "mock"
    assert body["ready"] is False


def test_chat_non_stream(client):
    r = client.post(
        "/v1/chat",
        json={
            "session_id": "demo",
            "messages": [{"role": "user", "content": "你好，我想应聘 AI 应用工程师"}],
            "profile": {"position": "AI 应用工程师", "style": "专业严谨"},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "demo"
    assert body["reply"]
    assert body["trace_id"]
    assert body["latency_ms"] >= 0


def test_chat_rejects_non_user_tail(client):
    r = client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "hi"},
                           {"role": "assistant", "content": "hello"}]},
    )
    assert r.status_code == 422


def test_chat_stream_emits_sse(client):
    with client.stream(
        "POST",
        "/v1/chat/stream",
        json={"messages": [{"role": "user", "content": "介绍一下你自己"}]},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        raw = resp.read().decode("utf-8")

    lines = [ln for ln in raw.splitlines() if ln.startswith("data:")]
    assert len(lines) >= 3
    assert lines[-1].strip() == "data: [DONE]"

    deltas = [ln for ln in lines if '"type":"delta"' in ln.replace(" ", "")]
    assert deltas, "至少应产出一个增量帧"
    done_frame = [ln for ln in lines if '"type":"done"' in ln.replace(" ", "")]
    assert done_frame, "结束时应带一帧含 TTFT / 总耗时的汇总"
    assert "ttft_ms" in done_frame[0]


def test_system_prompt_is_built_from_profile(client):
    """profile 的变化应体现在最终送给模型的 system 消息里。"""
    from src.prompts.interviewer import build_messages

    msgs = build_messages(
        [{"role": "user", "content": "hi"}],
        {"position": "生信分析师", "level": "校招", "style": "压力面试", "company": "某某院"},
    )
    assert msgs[0]["role"] == "system"
    assert "生信分析师" in msgs[0]["content"]
    assert "压力面试" in msgs[0]["content"]
    assert "某某院的面试官" in msgs[0]["content"]


def test_retrieved_context_is_injected(offline_settings):
    from src.prompts.interviewer import build_messages

    msgs = build_messages(
        [{"role": "user", "content": "hi"}], {"position": "p"}, retrieved="题库片段：RAG 常见追问"
    )
    assert "题库片段" in msgs[0]["content"]
