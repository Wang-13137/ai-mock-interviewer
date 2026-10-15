# -*- coding: utf-8 -*-
"""会话记忆管理（W5）。

问题：一场模拟面试 20 轮起步，原文全塞进上下文，token 成本线性涨、
模型还会被早期内容带偏。标准解法是**滚动摘要**：超过预算后，把最老的一批对话
压成一段摘要，只保留最近 N 轮原文。

三条刻意的设计决策：
1. **摘要是状态的一部分**，存在 `SessionState.memory_summary` 里滚动更新
   （新摘要 = 旧摘要 + 本批压缩内容），不是每轮重算全文——否则摘要本身也要花 O(n) 的钱。
2. **离线必须可用**：没有真实 LLM 时退化为抽取式摘要（截取每轮开头），
   链路可测、可 CI；真实模型摘要只在 client.ready 时启用。替身摘要的"质量"
   不对外引用，同全项目红线。
3. **压缩是渐进的**：只压缩「最老的、且还没被压缩过」的部分，最近
   `keep_recent` 条永远原样保留——面试官对刚说过的话最敏感。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..clients.llm import BaseLLMClient

log = logging.getLogger("aimi.agent.memory")

# 摘要注入用的固定问答对，让模型把摘要当「已发生的对话」而不是新指令
SUMMARY_USER = "【前情摘要】以下是本场面试到目前为止的要点：\n{summary}"
SUMMARY_ACK = "好的，我会基于以上摘要继续，不再重复追问已覆盖的内容。"

SUMMARY_PROMPT = (
    "把以下面试对话压缩成不超过 {max_chars} 字的要点摘要，保留：已考察的维度、"
    "候选人答到的关键点、面试官欠下的追问。直接输出摘要正文，不要解释。\n\n{transcript}"
)


@dataclass(frozen=True)
class MemoryPolicy:
    enabled: bool = True
    max_tokens: int = 3000      # 历史（不含 system）估算 token 预算
    keep_recent: int = 6        # 最近多少条消息原文保留
    summary_max_chars: int = 600


def estimate_tokens(text: str) -> int:
    """粗估 token 数：CJK 一字符 ≈ 1 token，其余 ≈ 4 字符 1 token。

    不引入 tiktoken 的理由：估预算只需要数量级正确，多一个重依赖不值。
    """
    cjk = sum(1 for ch in text if ord(ch) > 0x2E7F)
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def history_tokens(history: List[Dict[str, str]]) -> int:
    return sum(estimate_tokens(m.get("content", "")) for m in history)


def _render_transcript(messages: List[Dict[str, str]]) -> str:
    role_map = {"user": "候选人", "assistant": "面试官", "system": "系统"}
    return "\n".join(f"{role_map.get(m.get('role'), m.get('role'))}: {m.get('content', '')}" for m in messages)


def _extractive_summary(messages: List[Dict[str, str]], max_chars: int) -> str:
    """离线替身摘要：每轮取前 40 字拼接截断。**只保证链路通，不是真摘要**。"""
    parts = []
    for m in messages:
        head = m.get("content", "").strip().replace("\n", " ")[:40]
        if head:
            parts.append(f"- {head}")
    return "\n".join(parts)[:max_chars]


class MemoryManager:
    def __init__(self, client: Optional[BaseLLMClient] = None, policy: Optional[MemoryPolicy] = None) -> None:
        self.client = client
        self.policy = policy or MemoryPolicy()

    async def fit(
        self, history: List[Dict[str, str]], state: Any
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """返回 (新 history, 元信息)。不超预算时原样返回。"""
        meta: Dict[str, Any] = {"compressed": False, "tokens_before": history_tokens(history)}
        if not self.policy.enabled or meta["tokens_before"] <= self.policy.max_tokens:
            return history, meta
        if len(history) <= self.policy.keep_recent + 2:
            return history, meta  # 消息太少，压了也没意义

        tail = history[-self.policy.keep_recent:] if self.policy.keep_recent else []
        head = history[: len(history) - len(tail)]

        prior = getattr(state, "memory_summary", "") or ""
        summary = await self._summarize(prior, head)
        state.memory_summary = summary

        fitted = [
            {"role": "user", "content": SUMMARY_USER.format(summary=summary)},
            {"role": "assistant", "content": SUMMARY_ACK},
            *tail,
        ]
        meta.update(
            compressed=True,
            tokens_after=history_tokens(fitted),
            summarized_messages=len(head),
            summarizer="llm" if self._llm_available() else "extractive-fallback",
        )
        log.info("会话 %s 记忆压缩：%d -> %d tokens", getattr(state, "session_id", "?"),
                 meta["tokens_before"], meta["tokens_after"])
        return fitted, meta

    def _llm_available(self) -> bool:
        return bool(self.client is not None and getattr(self.client, "ready", False))

    async def _summarize(self, prior: str, head: List[Dict[str, str]]) -> str:
        transcript = _render_transcript(head)
        if prior:
            transcript = f"已有摘要：{prior}\n---\n新增对话：\n{transcript}"
        if self._llm_available():
            try:
                res = await self.client.achat(
                    [{"role": "user", "content": SUMMARY_PROMPT.format(
                        max_chars=self.policy.summary_max_chars, transcript=transcript)}],
                    temperature=0.2,
                    max_tokens=self.policy.summary_max_chars * 2,
                )
                text = (res.content or "").strip()
                if text:
                    return text[: self.policy.summary_max_chars]
            except Exception as exc:  # 摘要失败不该弄崩面试：退回抽取式
                log.warning("LLM 摘要失败，退回抽取式：%s", exc)
        merged = (prior + "\n" if prior else "") + _extractive_summary(head, self.policy.summary_max_chars)
        return merged[: self.policy.summary_max_chars]
