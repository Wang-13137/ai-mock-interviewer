# -*- coding: utf-8 -*-
"""手写 tool-call 循环（ReAct 风格）。

一次 `run()` 的流程：

```
组装带工具说明的 messages
└─ step 1..N：调模型 → 解析输出
      ├─ 没有工具块      → 得到终答，结束（stopped=final）
      ├─ 工具块解析失败  → 把错误回喂模型，让它重来（parse_errors+1）
      ├─ 检测到重复调用  → 回喂警告；再犯则强制收敛（stopped=loop_guard）
      └─ 执行工具        → 结果作为 observation 回喂，继续下一轮
└─ 触到步数上限 → 追加一条「禁止再调工具」的指令，再问一次（stopped=max_steps_recovered）
```

三个防御是每个真实 Agent 都必须有的：步数上限、死循环检测、工具异常不外溢。
它们同时也产出可统计的字段——**「完成率 / 工具成功率 / 平均步数」就是 W3 要往简历里填的那组数字**。
"""

from __future__ import annotations

import json
import time
from collections import Counter
from typing import Any, Dict, List, Optional

from ..clients.llm import BaseLLMClient, LLMError
from ..prompts.interviewer import build_messages
from ..schemas.agent import AgentRun, AgentStep, AgentUsage, StopReason, ToolResult
from .protocol import (
    FINAL_FORCE_MESSAGE,
    LOOP_GUARD_MESSAGE,
    ToolCallParseError,
    render_observation,
    render_tools_prompt,
    tool_specs_to_openai,
    try_parse_tool_call,
)
from .memory import MemoryManager
from .state import SessionStore
from .tools import ToolContext, ToolRegistry

LOOP_FALLBACK = "（本次回答中工具调用重复无效，已被系统中断。我们换一个问题继续。）"
FORCE_STOP_REPLY = "（本次回答已达工具调用上限，未能给出完整回答，我们换个问题继续。）"


class AgentLoop:
    def __init__(
        self,
        client: BaseLLMClient,
        registry: Optional[ToolRegistry] = None,
        store: Optional[SessionStore] = None,
        bank: Optional[Dict[str, Any]] = None,
        max_steps: int = 3,
        memory: Optional[MemoryManager] = None,
        protocol: str = "handwritten",
    ) -> None:
        self.client = client
        self.registry = registry
        self.store = store or SessionStore()
        self.bank = bank if bank is not None else {}
        self.max_steps = max(1, int(max_steps))
        self.memory = memory
        # protocol: "handwritten"（默认，把工具写成 ```tool 代码块塞进 system）
        #            "native"（把工具以 OpenAI tools 参数下发，由模型自行决定是否调用）。
        # 两者共用同一套工具执行 / 死循环检测 / 步数上限逻辑，只在「怎么拿到 ToolCall」上分叉。
        self.protocol = protocol
        # native 模式下预先把工具清单转成 OpenAI schema；handwritten 模式下为 None。
        self._openai_tools: Optional[List[Dict[str, Any]]] = (
            tool_specs_to_openai(registry.specs()) if (protocol == "native" and registry) else None
        )

    # ---------------- 主循环 ---------------- #
    async def run(
        self,
        history: List[Dict[str, str]],
        profile: Optional[Dict[str, Any]] = None,
        session_id: str = "anonymous",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AgentRun:
        messages = self._build_messages(history, profile)
        steps: List[AgentStep] = []
        usage = AgentUsage()
        started = time.perf_counter()
        st = await self._bump_turn(session_id)  # 用户刚说完话 = 一轮开始，工具要能立刻读到

        # W5 记忆压缩：超预算时把老对话滚进摘要。注意压的是「带 system 的 messages」，
        # 压缩完要重建，因为 build_messages 还会往里拼工具说明。
        if self.memory is not None:
            fitted, meta = await self.memory.fit(messages[1:], st)  # 跳过 system
            if meta.get("compressed"):
                steps.append(AgentStep(
                    index=1, kind="memory", ok=True,
                    content=f"tokens {meta['tokens_before']}->{meta['tokens_after']} "
                            f"（{meta['summarizer']}，压缩 {meta['summarized_messages']} 条）",
                    elapsed_ms=0.0,
                ))
                messages = self._build_messages(fitted, profile)

        seen_signatures: set[str] = set()
        warned = False
        tool_calls_done = 0
        reason: StopReason = "max_steps_recovered"
        # max_steps 是「允许的工具步数」，产出自然语言终答的那一轮不占预算
        max_turns = self.max_steps + 1
        # native 模式下随轮下发的工具清单；finish_interview 之后清空，避免收尾轮仍被诱导调工具。
        current_tools = self._openai_tools

        for _ in range(1, max_turns + 1):
            t0 = time.perf_counter()
            try:
                result = await self.client.achat(
                    messages, temperature=temperature, max_tokens=max_tokens,
                    **self._tools_kw(current_tools),
                )
            except LLMError as exc:
                steps.append(
                    AgentStep(index=len(steps) + 1, kind="llm", ok=False, content=str(exc),
                              elapsed_ms=_ms(t0))
                )
                return self._done(steps, usage, "（模型服务暂时不可用，请稍后再试。）", "error", started)
            elapsed = _ms(t0)
            self._accumulate(usage, result.usage)
            usage.model_calls += 1
            text = (result.content or "").strip()

            steps.append(AgentStep(index=len(steps) + 1, kind="llm", content=text[:400], elapsed_ms=elapsed))

            call, perr = self._extract_call(result, text)
            if perr is not None:
                steps.append(
                    AgentStep(index=len(steps) + 1, kind="parse_error", ok=False,
                              content=perr.detail, elapsed_ms=0.0)
                )
                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {"role": "user", "content": f"工具调用格式有误：{perr.detail}。请按规定的 ```tool 代码块格式重新输出。"}
                )
                continue

            if call is None:
                steps.append(AgentStep(index=len(steps) + 1, kind="final", content=text, elapsed_ms=0.0))
                return self._done(steps, usage, text, "final", started)

            # 工具预算已用尽：保留这条发言，交给下面的兜底回合收敛（不再执行工具）
            if tool_calls_done >= self.max_steps:
                messages.append({"role": "assistant", "content": text})
                break

            # ---- 死循环检测 ---- #
            sig = call.signature
            if sig in seen_signatures:
                steps.append(
                    AgentStep(index=len(steps) + 1, kind="loop_guard", tool_name=call.name,
                              arguments=call.arguments, ok=False, content="重复调用已出现过一次")
                )
                if warned:
                    # 二次重复：直接收敛。注意不能把模型的工具代码块原样吐给候选人
                    return self._done(steps, usage, LOOP_FALLBACK, "loop_guard", started)
                warned = True
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": LOOP_GUARD_MESSAGE})
                continue
            seen_signatures.add(sig)

            # ---- 执行工具 ---- #
            steps.append(
                AgentStep(
                    index=len(steps) + 1,
                    kind="tool_call",
                    tool_name=call.name,
                    arguments=call.arguments,
                    content=_dump_args(call.arguments),
                    elapsed_ms=0.0,
                )
            )
            tool_calls_done += 1
            tool_result, dt = await self._execute(call, session_id, profile)
            steps.append(
                AgentStep(index=len(steps) + 1, kind="tool_result", tool_name=call.name,
                          content=(tool_result.content if tool_result.ok else tool_result.error)[:400],
                          ok=tool_result.ok, elapsed_ms=dt)
            )
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": render_observation(tool_result)})

            if tool_result.ok and call.name == "finish_interview":
                # 终局工具：再问一次拿自然语言收尾，但不再允许工具
                messages = _strip_tools(messages)
                current_tools = None  # native 模式同步收掉 tools 参数

        # ---- 触到步数上限：追加禁止再调工具的指令，再问一次 ---- #
        messages.append({"role": "user", "content": FINAL_FORCE_MESSAGE})
        t0 = time.perf_counter()
        try:
            last = await self.client.achat(
                messages, temperature=temperature, max_tokens=max_tokens,
                **self._tools_kw(current_tools),
            )
            elapsed = _ms(t0)
            self._accumulate(usage, last.usage)
            usage.model_calls += 1
            final_text = (last.content or "").strip()
            call2, _ = try_parse_tool_call(final_text)
            if call2 is not None:  # 仍执意调工具：直接收敛，不要把它的话术块端给候选人
                final_text = FORCE_STOP_REPLY
        except LLMError:
            elapsed = _ms(t0)
            final_text = FORCE_STOP_REPLY

        steps.append(AgentStep(index=len(steps) + 1, kind="final", content=final_text[:400], elapsed_ms=elapsed))
        return self._done(steps, usage, final_text, "max_steps_recovered", started)

    # ---------------- 内部 ---------------- #
    def _build_messages(self, history: List[Dict[str, str]], profile: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
        messages = build_messages(history, profile)
        if self.registry is None:
            return messages
        # native 模式下工具通过 API 的 tools 参数下发，不该再写进 system；
        # 只有 handwritten 模式才把工具说明渲染进 system 提示词。
        if self.protocol == "handwritten":
            tools_prompt = render_tools_prompt(self.registry.specs())
            if tools_prompt and messages and messages[0].get("role") == "system":
                messages = [dict(m) for m in messages]
                messages[0] = {
                    "role": "system",
                    "content": messages[0]["content"].rstrip() + "\n\n" + tools_prompt,
                }
        return messages

    # ---------------- 工具调用提取（原生 / 手写两条路径归一成 ToolCall） ---------------- #
    def _extract_call(
        self, result, text: str
    ) -> Tuple[Optional["ToolCall"], Optional[ToolCallParseError]]:
        """从一次模型响应里取出要执行的工具调用。

        - native 模式：优先用 LLMResult.tool_calls（API 保证格式合法，参数解析失败退化为空 dict）；
        - 否则（handwritten 模式，或 native 但模型没调工具）回退到文本里的 ```tool 代码块解析。
        返回 (ToolCall, None) / (None, None) / (None, ParseError)，语义与 try_parse_tool_call 一致。
        """
        if result.tool_calls:
            # 我们的协议一轮只调一个工具，取第一个；其余忽略（模型若多调会在下一轮继续）。
            return result.tool_calls[0], None
        return try_parse_tool_call(text)

    def _tools_kw(self, current_tools) -> Dict[str, Any]:
        """只有 native 模式且当前还允许用工具时才下发 tools 参数；handwritten 模式绝不传，
        以免手写测试用的假客户端（achat 不接受 tools 关键字）报 TypeError。"""
        return {"tools": current_tools} if (self.protocol == "native" and current_tools) else {}

    async def _execute(self, call, session_id: str, profile: Optional[Dict[str, Any]]):
        t0 = time.perf_counter()
        if self.registry is None:
            return ToolResult(ok=False, tool_name=call.name, error="当前未启用工具，请直接回答候选人。"), _ms(t0)
        args = dict(call.arguments)
        args.setdefault("session_id", session_id)
        ctx = ToolContext(store=self.store, bank=self.bank, profile=profile or {})
        result = await self.registry.call(call.model_copy(update={"arguments": args}), ctx)
        return result, _ms(t0)

    async def _bump_turn(self, session_id: str):
        st = await self.store.get(session_id)
        st.turns += 1
        return st

    @staticmethod
    def _accumulate(usage: AgentUsage, u) -> None:
        if u is None:
            return
        usage.prompt_tokens = (usage.prompt_tokens or 0) + (u.prompt_tokens or 0)
        usage.completion_tokens = (usage.completion_tokens or 0) + (u.completion_tokens or 0)
        usage.total_tokens = (usage.total_tokens or 0) + ((u.total_tokens or 0))

    @staticmethod
    def _done(
        steps: List[AgentStep],
        usage: AgentUsage,
        reply: str,
        stopped: StopReason,
        started: float,
    ) -> AgentRun:
        return AgentRun(
            reply=reply,
            steps=steps,
            tool_calls=sum(1 for s in steps if s.kind == "tool_call"),
            tool_errors=sum(1 for s in steps if s.kind == "tool_result" and not s.ok),
            parse_errors=sum(1 for s in steps if s.kind == "parse_error"),
            model_calls=usage.model_calls,
            stopped=stopped,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            usage=usage,
        )


def _dump_args(arguments: Dict[str, Any]) -> str:
    try:
        return json.dumps(arguments, ensure_ascii=False)
    except Exception:
        return repr(arguments)


def _strip_tools(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """移除 system 里的工具说明，用于「禁止再调工具」的收尾回合。"""
    out = [dict(m) for m in messages]
    if out and out[0].get("role") == "system":
        marker = "## 可用工具"
        sys_txt = out[0]["content"]
        if marker in sys_txt:
            out[0] = {"role": "system", "content": sys_txt.split(marker)[0].rstrip()}
    return out


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


# --------------------------------------------------------------------------- #
# 批量统计：给 W3 的评测报告用
# --------------------------------------------------------------------------- #
def summarize_runs(runs: List[AgentRun]) -> Dict[str, Any]:
    """把一批 AgentRun 压成一组指标——和 eval/stats.py 的思路一致：多给几个互补口径。"""
    if not runs:
        return {"n": 0}

    tool_calls = sum(r.tool_calls for r in runs)
    tool_errors = sum(r.tool_errors for r in runs)
    parse_errors = sum(r.parse_errors for r in runs)
    n = len(runs)

    def _avg(key: str) -> float:
        vals = [getattr(r, key) for r in runs]
        return round(sum(vals) / n, 3)

    stops = Counter(r.stopped for r in runs)
    return {
        "n": n,
        "stopped_distribution": dict(stops),
        "completion_rate": round(stops.get("final", 0) / n, 4),  # 未靠兜底收尾的比例
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "parse_errors": parse_errors,
        "tool_success_rate": round((tool_calls - tool_errors) / tool_calls, 4) if tool_calls else None,
        "parse_error_rate": round(parse_errors / sum(r.model_calls for r in runs), 4)
        if sum(r.model_calls for r in runs)
        else None,
        "avg_model_calls": _avg("model_calls"),
        "avg_steps": round(sum(len(r.steps) for r in runs) / n, 3),
        "avg_elapsed_ms": _avg("elapsed_ms"),
    }
