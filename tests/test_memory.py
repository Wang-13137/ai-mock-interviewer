# -*- coding: utf-8 -*-
"""W5 记忆管理测试。

替身红线同样适用于这里：extractive fallback 只验证链路，不验证「摘要质量」。
"""

from __future__ import annotations

import pytest

from src.agent.loop import AgentLoop
from src.agent.memory import MemoryManager, MemoryPolicy, estimate_tokens, history_tokens
from src.agent.state import SessionState
from src.clients.llm import EchoClient


def _long_history(n: int, char_each: int = 400) -> list[dict[str, str]]:
    out = []
    for i in range(n):
            out.append({"role": "user", "content": f"第{i}轮候选人回答 " + "答" * char_each})
            out.append({"role": "assistant", "content": f"第{i}轮面试官追问 " + "问" * char_each})
    return out


# --------------------------------------------------------------------------- #
# token 估算
# --------------------------------------------------------------------------- #
def test_estimate_tokens_counts_cjk_per_char():
    assert estimate_tokens("中文四个") == 4
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("ab中文") == 2 + 1  # 2 CJK + 2 ascii→1


# --------------------------------------------------------------------------- #
# MemoryManager.fit
# --------------------------------------------------------------------------- #
async def test_short_history_untouched():
    mm = MemoryManager(client=EchoClient(), policy=MemoryPolicy(max_tokens=3000))
    h = _long_history(2, 50)
    fitted, meta = await mm.fit(h, SessionState(session_id="s1"))
    assert fitted is h and meta["compressed"] is False


async def test_long_history_compressed_with_extractive_fallback():
    """EchoClient ready=False → 必须走抽取式兜底，且链路和字段都完整。"""
    mm = MemoryManager(client=EchoClient(), policy=MemoryPolicy(max_tokens=200, keep_recent=4))
    st = SessionState(session_id="s2")
    h = _long_history(10)
    fitted, meta = await mm.fit(h, st)
    assert meta["compressed"] is True
    assert meta["summarizer"] == "extractive-fallback"
    assert meta["tokens_after"] < meta["tokens_before"]
    assert len(fitted) == 2 + 4  # 摘要问答对 + 最近 4 条原文
    assert fitted[0]["content"].startswith("【前情摘要】")
    assert fitted[2:] == h[-4:]
    assert st.memory_summary  # 摘要落进了会话状态


async def test_summary_rolls_forward():
    """第二次压缩应基于旧摘要滚动更新，而不是丢掉第一段历史。"""
    mm = MemoryManager(client=EchoClient(), policy=MemoryPolicy(max_tokens=200, keep_recent=4))
    st = SessionState(session_id="s3")
    _, _ = await mm.fit(_long_history(8), st)
    first = st.memory_summary
    _, _ = await mm.fit(_long_history(12), st)
    assert st.memory_summary != first
    assert first[:30] in st.memory_summary  # 旧摘要被继承


async def test_llm_summary_used_when_client_ready():
    """真实 client（ready=True）时走 LLM 摘要路径。脚本化假模型固定回一句。"""

    class FakeClient(EchoClient):
        ready = True

        async def achat(self, messages, **kw):
            from src.clients.llm import LLMResult  # 延迟导入避免循环

            return LLMResult(content="LLM摘要：已考察项目深挖维度", usage=None)

    mm = MemoryManager(client=FakeClient(), policy=MemoryPolicy(max_tokens=200, keep_recent=4))
    st = SessionState(session_id="s4")
    fitted, meta = await mm.fit(_long_history(10), st)
    assert meta["summarizer"] == "llm"
    assert "LLM摘要" in st.memory_summary


async def test_disabled_policy_is_passthrough():
    mm = MemoryManager(client=EchoClient(), policy=MemoryPolicy(enabled=False, max_tokens=10))
    h = _long_history(10)
    fitted, meta = await mm.fit(h, SessionState(session_id="s5"))
    assert fitted is h and meta["compressed"] is False


# --------------------------------------------------------------------------- #
# AgentLoop 集成
# --------------------------------------------------------------------------- #
async def test_agent_loop_records_memory_step():
    """压缩发生时，AgentRun.steps 里应留一条 kind=memory 的痕迹。"""
    loop = AgentLoop(
        client=EchoClient(),
        memory=MemoryManager(client=EchoClient(), policy=MemoryPolicy(max_tokens=200, keep_recent=4)),
    )
    run = await loop.run(_long_history(10), session_id="mem-it")
    assert run.reply
    mem_steps = [s for s in run.steps if s.kind == "memory"]
    assert len(mem_steps) == 1
    assert "tokens" in mem_steps[0].content


async def test_agent_loop_without_memory_unchanged():
    """不传 memory 时行为与旧版一致（回归护栏）。"""
    loop = AgentLoop(client=EchoClient())
    run = await loop.run([{"role": "user", "content": "你好"}], session_id="plain")
    assert run.reply
    assert all(s.kind != "memory" for s in run.steps)
