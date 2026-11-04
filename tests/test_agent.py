# -*- coding: utf-8 -*-
"""Agent 模块的测试：协议解析、工具执行、循环收敛三块。

全部跑在离线替身 / 假客户端上，CI 无需任何密钥。
"""

from __future__ import annotations

import json

import pytest

from src.agent.loop import AgentLoop, summarize_runs
from src.agent.protocol import (
    ToolCallParseError,
    parse_tool_call,
    tool_specs_to_openai,
    try_parse_tool_call,
)
from src.agent.state import SessionStore
from src.agent.tools import ToolContext, ToolRegistry, ToolSpec, build_default_registry, load_question_bank
from src.clients.llm import BaseLLMClient, LLMResult, LLMUsage, OpenAICompatClient
from src.schemas.agent import AgentStep, ToolCall


# --------------------------------------------------------------------------- #
# 假客户端：按脚本返回文本，模拟模型行为（含各种「坏」模型）
# --------------------------------------------------------------------------- #
class ScriptedClient(BaseLLMClient):
    provider_name = "scripted"
    ready = True

    def __init__(self, replies: list[str], *, model: str = "scripted-1"):
        self._replies = list(replies)
        self.model = model
        self.prompts: list[list[dict]] = []

    async def achat(self, messages, *, temperature=None, max_tokens=None) -> LLMResult:
        self.prompts.append([dict(m) for m in messages])
        text = self._replies.pop(0) if self._replies else "好的，我们换个问题。"
        return LLMResult(
            content=text,
            finish_reason="stop",
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5),
        )

    async def astream(self, messages, *, temperature=None, max_tokens=None):
        yield ""  # pragma: no cover


def tool_block(name: str, **kwargs) -> str:
    return "```tool\n" + json.dumps({"name": name, "arguments": kwargs}, ensure_ascii=False) + "\n```"


@pytest.fixture()
def bank():
    return load_question_bank("config/question_bank.json")


# --------------------------------------------------------------------------- #
# 一、协议解析
# --------------------------------------------------------------------------- #
def test_parse_fenced_tool_call():
    call = parse_tool_call(tool_block("search_question_bank", query="RAG 重排", top_k=2))
    assert call is not None and call.name == "search_question_bank"
    assert call.arguments["top_k"] == 2


def test_parse_prose_around_call():
    text = "我先查一下题库。\n" + tool_block("search_question_bank", query="LaTeX") + "\n拿到结果再说"
    call = parse_tool_call(text)
    assert call is not None and call.arguments["query"] == "LaTeX"


def test_plain_text_is_final_answer():
    call, err = try_parse_tool_call("这个问题我直接回答：先讲切分策略。")
    assert call is None and err is None


def test_malformed_call_raises_parse_error():
    bad = '```tool\n{"name": "search_question_bank", "arguments": {"query": }\n```'
    call, err = try_parse_tool_call(bad)
    assert call is None and isinstance(err, ToolCallParseError)
    assert "JSON" in err.detail


def test_lenient_parse_chinese_quotes_and_trailing_comma():
    bad = '```json\n{“name”: “search_question_bank”, "arguments": {"query": "RAG",}}```'
    call, err = try_parse_tool_call(bad)
    assert err is None, err.detail if err else ""
    assert call is not None and call.name == "search_question_bank"


def test_json_without_name_treated_as_final():
    call, err = try_parse_tool_call('```json\n{"foo": 1}\n```')
    assert call is None
    assert err is None or isinstance(err, ToolCallParseError)  # 两种都允许，但不能崩


def test_signature_for_loop_detection():
    a = parse_tool_call(tool_block("search_question_bank", query="RAG"))
    b = parse_tool_call(tool_block("search_question_bank", query="RAG"))
    c = parse_tool_call(tool_block("search_question_bank", query="SSE"))
    assert a.signature == b.signature
    assert a.signature != c.signature


# --------------------------------------------------------------------------- #
# 二、工具执行
# --------------------------------------------------------------------------- #
async def _ctx(bank) -> ToolContext:
    return ToolContext(store=SessionStore(), bank=bank, profile={})


async def test_search_returns_relevant_question(bank):
    ctx = await _ctx(bank)
    reg = build_default_registry()
    res = await reg.call(
        parse_tool_call(tool_block("search_question_bank", query="重排 消融实验", top_k=2)), ctx
    )
    assert res.ok, res.error
    assert "检索到" in res.content
    assert "相关度" in res.content


async def test_search_no_hit_is_still_ok(bank):
    """无命中不是错误，要给模型「换角度」的可执行建议。"""
    ctx = await _ctx(bank)
    reg = build_default_registry()
    res = await reg.call(parse_tool_call(tool_block("search_question_bank", query="zzzz 量子佛学")), ctx)
    assert res.ok
    assert "换一个角度" in res.content


async def test_unknown_tool_is_recoverable(bank):
    ctx = await _ctx(bank)
    reg = build_default_registry()
    res = await reg.call(parse_tool_call(tool_block("delete_everything")), ctx)
    assert not res.ok
    assert "可用工具只有" in res.error


async def test_missing_required_argument(bank):
    ctx = await _ctx(bank)
    reg = build_default_registry()
    res = await reg.call(parse_tool_call(tool_block("search_question_bank", top_k=3)), ctx)
    assert not res.ok and "缺少必填参数" in res.error


async def test_top_k_out_of_range(bank):
    ctx = await _ctx(bank)
    reg = build_default_registry()
    res = await reg.call(parse_tool_call(tool_block("search_question_bank", query="RAG", top_k=99)), ctx)
    assert not res.ok and "1-5" in res.error


async def test_note_and_progress_roundtrip(bank):
    from src.schemas.agent import ToolCall

    ctx = await _ctx(bank)
    reg = build_default_registry()
    await reg.call(ToolCall(name="save_candidate_note", arguments={"note": "候选人提到做过 McNemar 检验", "dimension": "数据与统计"}), ctx)
    res = await reg.call(ToolCall(name="get_interview_progress", arguments={}), ctx)
    assert res.ok
    assert "McNemar" in res.content
    assert "20%" in res.content  # 1/5 维度覆盖率


async def test_finish_interview_marks_state(bank):
    from src.schemas.agent import ToolCall

    store = SessionStore()
    ctx = ToolContext(store=store, bank=bank, profile={})
    reg = build_default_registry()
    res = await reg.call(ToolCall(name="finish_interview", arguments={"summary": "整体不错，需补分布式"}), ctx)
    assert res.ok
    st = await store.get("anonymous")
    assert st.finished and "分布式" in st.summary


async def test_empty_note_rejected(bank):
    from src.schemas.agent import ToolCall

    ctx = await _ctx(bank)
    reg = build_default_registry()
    res = await reg.call(ToolCall(name="save_candidate_note", arguments={"note": "  "}), ctx)
    assert not res.ok and "不能为空" in res.error


# --------------------------------------------------------------------------- #
# 三、循环收敛
# --------------------------------------------------------------------------- #
async def test_loop_call_tool_then_final(bank):
    client = ScriptedClient(
        [
            tool_block("search_question_bank", query="RAG"),
            "你刚才提到加了重排，那你怎么证明它真的有效？",
        ]
    )
    loop = AgentLoop(client, build_default_registry(), SessionStore(), bank, max_steps=3)
    run = await loop.run([{"role": "user", "content": "我做了一个 RAG 问答系统"}], session_id="s1")

    assert run.stopped == "final"
    assert run.reply.startswith("你刚才提到")
    assert run.tool_calls == 1 and run.tool_errors == 0
    assert run.model_calls == 2
    assert [s.kind for s in run.steps].count("tool_result") == 1
    # observation 必须回喂给模型，否则下一轮看不到工具结果
    assert any("tool_result" in m["content"] for m in client.prompts[-1])


async def test_loop_recovers_from_parse_error(bank):
    client = ScriptedClient(
        [
            '```tool\n{"name": , "arguments": {}}\n```',  # 坏格式
            tool_block("get_interview_progress"),
            "这是第 1 轮，我们先从你的项目聊起。",
        ]
    )
    loop = AgentLoop(client, build_default_registry(), SessionStore(), bank, max_steps=3)
    run = await loop.run([{"role": "user", "content": "开始面试吧"}], session_id="s1")
    assert run.parse_errors == 1
    assert run.tool_calls == 1
    assert run.stopped == "final"


async def test_loop_guard_stops_repeated_calls(bank):
    same = tool_block("search_question_bank", query="RAG")
    client = ScriptedClient([same, same, same, same])
    loop = AgentLoop(client, build_default_registry(), SessionStore(), bank, max_steps=5)
    run = await loop.run([{"role": "user", "content": "聊聊 RAG"}], session_id="s1")
    assert run.stopped == "loop_guard"
    assert any(s.kind == "loop_guard" for s in run.steps)


async def test_final_answer_does_not_consume_tool_budget(bank):
    """max_steps=3 表示允许 3 次工具调用，之后还应有一轮产出自然语言终答。

    踩过的坑：早期实现把「回合数」当成预算，导致用完 3 次工具后直接走兜底，
    stopped 被误报成 max_steps_recovered —— 指标口径失真会让后面所有评测结论失效。
    """
    client = ScriptedClient(
        [
            tool_block("search_question_bank", query="重排"),
            tool_block("save_candidate_note", note="提到 Top-5 召回率", dimension="LLM 工程能力"),
            tool_block("get_interview_progress"),
            "你说 Top-5 召回率提升了，这个结论做过显著性检验吗？",
        ]
    )
    loop = AgentLoop(client, build_default_registry(), SessionStore(), bank, max_steps=3)
    run = await loop.run([{"role": "user", "content": "我加了重排"}], session_id="s2")
    assert run.stopped == "final"
    assert run.tool_calls == 3 and run.model_calls == 4
    assert run.reply.startswith("你说 Top-5")


async def test_loop_max_steps_recovered(bank):
    client = ScriptedClient([tool_block("search_question_bank", query=str(i)) for i in range(6)])
    loop = AgentLoop(client, build_default_registry(), SessionStore(), bank, max_steps=2)
    run = await loop.run([{"role": "user", "content": "开始"}], session_id="s1")
    assert run.stopped == "max_steps_recovered"
    # 3 次循环回合（最后一次发出工具块时预算已尽，仍消耗一次模型调用）+ 1 次兜底终答
    assert run.model_calls == 4
    assert run.reply


async def test_loop_without_registry_is_plain_chat(bank):
    client = ScriptedClient(["那聊聊你的项目。"])
    loop = AgentLoop(client, None, SessionStore(), bank, max_steps=3)
    run = await loop.run([{"role": "user", "content": "你好"}], session_id="s1")
    assert run.tool_calls == 0 and run.stopped == "final"
    assert "可用工具" not in client.prompts[0][0]["content"]


async def test_tool_failure_does_not_break_loop(bank):
    """工具报错要回喂给模型，让它自纠；loop 本身不许崩。"""
    client = ScriptedClient(
        [
            tool_block("search_question_bank", query="RAG", top_k=99),  # 参数越界
            "抱歉刚才的参数有问题，我直接问：你的重排怎么验证的？",
        ]
    )
    loop = AgentLoop(client, build_default_registry(), SessionStore(), bank, max_steps=3)
    run = await loop.run([{"role": "user", "content": "RAG"}], session_id="s1")
    assert run.tool_errors == 1
    assert run.stopped == "final"
    assert any(not s.ok and s.kind == "tool_result" for s in run.steps)


async def test_session_turn_increments(bank):
    store = SessionStore()
    client = ScriptedClient(["好的。"])
    loop = AgentLoop(client, build_default_registry(), store, bank, max_steps=1)
    await loop.run([{"role": "user", "content": "你好"}], session_id="s9")
    st = await store.get("s9")
    assert st.turns == 1


def test_summarize_runs_metrics():
    from src.schemas.agent import AgentRun

    runs = [
        AgentRun(reply="a", steps=[AgentStep(index=1, kind="llm")], tool_calls=2, tool_errors=0,
                 model_calls=3, stopped="final"),
        AgentRun(reply="b", steps=[AgentStep(index=1, kind="llm")], tool_calls=1, tool_errors=1,
                 model_calls=2, stopped="max_steps_recovered"),
    ]
    s = summarize_runs(runs)
    assert s["n"] == 2
    assert s["tool_calls"] == 3 and s["tool_errors"] == 1
    assert s["tool_success_rate"] == pytest.approx(2 / 3, 0.001)
    assert s["completion_rate"] == 0.5
    assert s["avg_model_calls"] == 2.5
    assert summarize_runs([]) == {"n": 0}


# --------------------------------------------------------------------------- #
# 四、HTTP 层
# --------------------------------------------------------------------------- #
def test_app_tools_endpoint(client):
    specs = client.get("/v1/tools").json()
    names = {s["name"] for s in specs}
    assert names == {"search_question_bank", "get_interview_progress", "save_candidate_note", "finish_interview"}
    search = next(s for s in specs if s["name"] == "search_question_bank")
    assert search["parameters"]["required"] == ["query"]


def test_app_health_reports_agent(client):
    h = client.get("/health").json()
    assert h["agent_enabled"] is True
    assert len(h["tools"]) == 4


def test_app_agent_endpoint_returns_trace(client):
    """替身客户端会回显，所以这里验证的是「链路与字段完整」，不是业务正确性。"""
    resp = client.post(
        "/v1/chat/agent",
        json={"session_id": "http-1", "messages": [{"role": "user", "content": "聊聊 RAG"}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "http-1"
    assert isinstance(data["steps"], list) and data["steps"]
    assert data["stopped"] in {"final", "max_steps_recovered", "loop_guard", "error"}
    assert data["tool_calls"] >= 0 and data["parse_errors"] >= 0


def test_registry_rejects_duplicate_name():
    reg = ToolRegistry()
    spec = ToolSpec(name="x", description="d", parameters={})
    reg.register(spec, lambda a, c: "ok")
    with pytest.raises(ValueError):
        reg.register(spec, lambda a, c: "dup")


# --------------------------------------------------------------------------- #
# 三、原生 function calling 路径（native protocol）
# --------------------------------------------------------------------------- #
class NativeScriptedClient(BaseLLMClient):
    """假客户端：直接回吐预设的 LLMResult（含原生 tool_calls），用于验证 native 路径。"""

    provider_name = "native-scripted"
    ready = True

    def __init__(self, results):
        self._results = list(results)
        self.prompts = []
        self.last_tools = None

    async def achat(self, messages, *, temperature=None, max_tokens=None, tools=None) -> LLMResult:
        self.prompts.append([dict(m) for m in messages])
        self.last_tools = tools
        return self._results.pop(0) if self._results else LLMResult(content="好的，继续。", finish_reason="stop")

    async def astream(self, messages, *, temperature=None, max_tokens=None, tools=None):
        yield ""


def test_tool_specs_to_openai():
    out = tool_specs_to_openai(
        [ToolSpec(name="demo", description="d", parameters={"properties": {"q": {"type": "string"}}})]
    )
    assert out[0] == {
        "type": "function",
        "function": {
            "name": "demo",
            "description": "d",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }


def test_parse_tool_calls_from_message():
    raw = [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "search_question_bank", "arguments": '{"query": "RAG"}'},
        }
    ]
    calls = OpenAICompatClient._parse_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0].name == "search_question_bank"
    assert calls[0].arguments == {"query": "RAG"}
    # 缺 name 的条目被跳过；空 / None 返回空列表
    assert OpenAICompatClient._parse_tool_calls(None) == []
    assert OpenAICompatClient._parse_tool_calls([]) == []
    # arguments 解析失败退化为空 dict（交给 registry.validate 报缺参并自纠），不直接崩
    bad = OpenAICompatClient._parse_tool_calls([{"function": {"name": "x", "arguments": "{坏"}}])
    assert bad[0].name == "x" and bad[0].arguments == {}


async def test_agent_native_executes_tool_call(bank):
    # 第一轮：模型以原生 tool_calls 调用 save_candidate_note；第二轮：给终答
    res1 = LLMResult(
        content="",
        finish_reason="tool_calls",
        tool_calls=[ToolCall(name="save_candidate_note", arguments={"note": "候选人提到 McNemar 检验"})],
    )
    res2 = LLMResult(content="我们深入聊聊这个项目。", finish_reason="stop")
    client = NativeScriptedClient([res1, res2])
    loop = AgentLoop(client, build_default_registry(), SessionStore(), bank, max_steps=3, protocol="native")
    run = await loop.run([{"role": "user", "content": "我叫小王"}], session_id="nat-1")
    assert run.tool_calls == 1
    assert run.tool_errors == 0
    assert run.stopped == "final"
    # native 模式下工具清单确实以 tools 参数下发了
    assert client.last_tools is not None
    assert any(t["function"]["name"] == "save_candidate_note" for t in client.last_tools)


async def test_agent_native_no_tool_call_is_final(bank):
    # 模型不调工具、直接给自然语言终答，native 模式应正常结束（不触发手写解析）
    res = LLMResult(content="你好，我们开始面试。", finish_reason="stop")
    client = NativeScriptedClient([res])
    loop = AgentLoop(client, build_default_registry(), SessionStore(), bank, max_steps=3, protocol="native")
    run = await loop.run([{"role": "user", "content": "hi"}], session_id="nat-2")
    assert run.tool_calls == 0
    assert run.stopped == "final"
