# -*- coding: utf-8 -*-
"""工具注册表与四个种子工具。

四个工具覆盖了面试 Agent 需要的最少能力，也正好对上 JD 里那几条：
- `search_question_bank`  —— 检索（现在是关键词打分，**W2 换成向量检索，契约不变**）
- `get_interview_progress`—— 读会话状态（面试是有状态的，这是和普通 chatbot 的分界线）
- `save_candidate_note`   —— 写会话状态
- `finish_interview`      —— 终局动作，产出结构化收尾

参数校验的错误信息是刻意写得很「可执行」的（`top_k 必须在 1-5 之间`），
因为这条消息会被回喂给模型让它自纠——**错误文案也是工程的一部分**，
直接决定 explore 一轮后的工具调用成功率。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..schemas.agent import ToolCall, ToolResult, ToolSpec
from .state import SessionStore

log = logging.getLogger("aimi.agent.tools")

REPO_ROOT = Path(__file__).resolve().parents[2]


class ToolExecutionError(RuntimeError):
    """工具执行失败。message 会给回模型看。"""


@dataclass
class ToolContext:
    """工具执行期的上下文。工具不应该自己去读全局配置。"""

    store: SessionStore
    bank: Dict[str, Any]
    profile: Dict[str, Any]


ToolFn = Callable[[Dict[str, Any], ToolContext], Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, tuple[ToolSpec, ToolFn]] = {}

    def register(self, spec: ToolSpec, func: ToolFn) -> None:
        if spec.name in self._tools:
            raise ValueError(f"工具重名: {spec.name}")
        self._tools[spec.name] = (spec, func)

    def specs(self) -> List[ToolSpec]:
        return [spec for spec, _ in self._tools.values()]

    def names(self) -> List[str]:
        return list(self._tools)

    def has(self, name: str) -> bool:
        return name in self._tools

    # ---------------- 参数校验 ---------------- #
    @staticmethod
    def validate(spec: ToolSpec, arguments: Dict[str, Any]) -> None:
        props: Dict[str, Any] = (spec.parameters or {}).get("properties", {}) or {}
        required: List[str] = (spec.parameters or {}).get("required", []) or []

        missing = [k for k in required if k not in arguments or arguments[k] in (None, "")]
        if missing:
            raise ToolExecutionError(f"缺少必填参数: {', '.join(missing)}")

        unknown = [k for k in arguments if k not in props and k not in INJECTED_PARAMS]
        if unknown:
            # 不算致命错误：小模型常拼错参数名。给提示让它自纠。
            raise ToolExecutionError(
                f"未知参数: {', '.join(unknown)}；{spec.name} 只接受 {', '.join(props) or '（无参数）'}"
            )

        top_k = arguments.get("top_k")
        if top_k is not None:
            if isinstance(top_k, str) and top_k.isdigit():
                arguments["top_k"] = int(top_k)
            if not isinstance(arguments["top_k"], int):
                raise ToolExecutionError("top_k 必须是整数")
            if not 1 <= arguments["top_k"] <= 5:
                raise ToolExecutionError("top_k 必须在 1-5 之间")

    # ---------------- 执行 ---------------- #
    async def call(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        entry = self._tools.get(call.name)
        if entry is None:
            known = "、".join(self.names())
            return ToolResult(
                ok=False,
                tool_name=call.name,
                error=f"没有名为 {call.name} 的工具。可用工具只有：{known}。请从中选择一个。",
            )

        spec, func = entry
        try:
            self.validate(spec, dict(call.arguments))
        except ToolExecutionError as exc:
            return ToolResult(ok=False, tool_name=call.name, error=str(exc))

        try:
            raw = func(dict(call.arguments), ctx)
            if isinstance(raw, Awaitable):
                raw = await raw
        except ToolExecutionError as exc:  # 业务性失败，可自纠
            return ToolResult(ok=False, tool_name=call.name, error=str(exc))
        except Exception as exc:  # 兜底：绝不让整个 loop 崩掉
            log.exception("工具 %s 内部异常", call.name)
            return ToolResult(ok=False, tool_name=call.name, error=f"工具内部错误：{exc}")

        return ToolResult(ok=True, tool_name=call.name, content=str(raw))


# --------------------------------------------------------------------------- #
# 检索实现（关键词版）
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#._-]*")

# 由执行层注入、但不暴露给模型的参数。校验时要放行，否则模型每调一次都会撞「未知参数」
INJECTED_PARAMS = {"session_id"}


def load_question_bank(path: str | Path = "config/question_bank.json") -> Dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        log.warning("题库文件不存在，检索工具将返回空结果: %s", p)
        return {"version": "empty", "items": []}
    return json.loads(p.read_text(encoding="utf-8"))


def _tokens(text: str) -> List[str]:
    """英文数字按词切，中文按 bigram 切。

    刻意**不做**中文单字匹配：单字作为 token 会让「的」「学」这类字命中几乎所有条目，
    检索退化成噪声（踩过：query「量子佛学」能匹配到向量库那题）。零宽先验带来的假命中
    比漏召回更伤，因为模型会被一堆无关题目带偏。
    """
    out: List[str] = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    han = "".join(re.findall(r"[\u4e00-\u9fff]", text or ""))
    if len(han) == 1:  # 单字查询退化为 unigram，否则永远召回空
        out.append(han)
    else:
        out += [han[i:i + 2] for i in range(len(han) - 1)]
    return out


def score_item(item: Dict[str, Any], query_tokens: List[str], boost_dimensions: List[str]) -> float:
    """BM25 的极简替身：标签权重 > 维度 > 题干；未覆盖维度加权。"""
    if not query_tokens:
        return 0.0
    tags = " ".join(item.get("tags", []) or []).lower()
    dim = (item.get("dimension") or "").lower()
    q = (item.get("question") or "").lower()
    score = 0.0
    for t in set(query_tokens):
        if t in tags:
            score += 3.0
        elif t in dim:
            score += 2.0
        elif t in q:
            score += 1.0
    # 只有真正命中了关键词才允许加权。否则难度加分会给所有条目一个「低保」，
    # 检索退化成「按难度排序」，跟 query 无关。
    if score <= 0:
        return 0.0
    if item.get("dimension") in boost_dimensions:
        score *= 1.5
    score += max(0.0, 0.2 * (int(item.get("difficulty", 2)) - 1))  # 难题略微优先
    return score


# --------------------------------------------------------------------------- #
# 四个种子工具
# --------------------------------------------------------------------------- #
SEARCH_SPEC = ToolSpec(
    name="search_question_bank",
    description="在本地题库里检索与给定主题相关的面试问题，用于在追问前挑选更贴题的问法",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索关键词，例如「RAG 重排」或「批次效应」"},
            "top_k": {"type": "integer", "description": "返回条数，1-5，默认 3"},
        },
        "required": ["query"],
    },
)

PROGRESS_SPEC = ToolSpec(
    name="get_interview_progress",
    description="查询本场面试已经覆盖的能力维度、已经记录的要点，以及建议下一步提问的方向",
    parameters={"type": "object", "properties": {"session_id": {"type": "string", "description": "会话 ID"}}, "required": []},
)

NOTE_SPEC = ToolSpec(
    name="save_candidate_note",
    description="记录候选人回答中的关键事实或疑点，供后续追问与收尾使用",
    parameters={
        "type": "object",
        "properties": {
            "note": {"type": "string", "description": "要记录的要点，一句话即可"},
            "dimension": {"type": "string", "description": "该要点所属的能力维度，可留空"},
        },
        "required": ["note"],
    },
)

FINISH_SPEC = ToolSpec(
    name="finish_interview",
    description="结束本次模拟面试，并给出一段对候选人的总结评价",
    parameters={
        "type": "object",
        "properties": {"summary": {"type": "string", "description": "对候选人表现的整体评价，100 字以内"}},
        "required": ["summary"],
    },
)


def _search(arguments: Dict[str, Any], ctx: ToolContext) -> str:
    import asyncio

    query = str(arguments.get("query", "")).strip()
    top_k = int(arguments.get("top_k", 3) or 3)
    items: List[Dict[str, Any]] = list(ctx.bank.get("items", []) or [])
    if not items:
        return "题库为空（未加载配置文件）。请直接凭专业知识向候选人提问。"

    session_id = str(arguments.get("session_id", "") or "")
    boost: List[str] = []
    if session_id:
        # 这里是同步上下文，取不到 await 版本的状态；由上层注入 remaining
        st = ctx.store.peek(session_id)
        if st:
            boost = st.remaining_dimensions

    tokens = _tokens(query)
    scored = [(score_item(it, tokens, boost), it) for it in items]
    scored = [(s, it) for s, it in scored if s > 0]
    if not scored:
        return f"题库中没有与「{query}」相关的问题。请换一个角度提问，或直接凭常识追问。"
    scored.sort(key=lambda x: (-x[0], x[1].get("id", "")))

    lines = [f"检索到 {min(top_k, len(scored))} 条相关问题："]
    for idx, (s, it) in enumerate(scored[:top_k], 1):
        lines.append(
            f"{idx}. [{it.get('dimension','未分类')}] {it.get('question','')}"
            f"（难度 {it.get('difficulty', 2)}/3，相关度 {s:.1f}）"
        )
    lines.append("请挑选其中最贴题的一条，结合候选人刚刚的回答改写成追问。")
    return "\n".join(lines)


async def _progress(arguments: Dict[str, Any], ctx: ToolContext) -> str:
    session_id = str(arguments.get("session_id", "") or "anonymous")
    st = await ctx.store.get(session_id)
    lines = [
        f"会话 {session_id}：已进行 {st.turns} 轮候选人发言，维度覆盖率 {st.coverage:.0%}。",
        f"已覆盖维度：{'、'.join(st.asked_dimensions) or '（无）'}",
        f"尚未覆盖：{'、'.join(st.remaining_dimensions) or '（已全部覆盖）'}",
    ]
    if st.notes:
        lines.append("已记录的要点：")
        lines += [f"  - {n}" for n in st.notes[-5:]]
    if st.remaining_dimensions:
        lines.append(f"建议下一步优先考察：{st.remaining_dimensions[0]}")
    return "\n".join(lines)


async def _note(arguments: Dict[str, Any], ctx: ToolContext) -> str:
    session_id = str(arguments.get("session_id", "") or "anonymous")
    note = str(arguments.get("note", "")).strip()
    dimension = str(arguments.get("dimension", "") or "").strip()
    if not note:
        raise ToolExecutionError("note 不能为空")
    st = await ctx.store.get(session_id)
    st.notes.append(note)
    if dimension:
        st.asked_dimensions.append(dimension)
        st.asked_dimensions = list(dict.fromkeys(st.asked_dimensions))
    return f"已记录（当前共 {len(st.notes)} 条要点，覆盖率 {st.coverage:.0%}）。"


async def _finish(arguments: Dict[str, Any], ctx: ToolContext) -> str:
    session_id = str(arguments.get("session_id", "") or "anonymous")
    summary = str(arguments.get("summary", "")).strip()
    if not summary:
        raise ToolExecutionError("summary 不能为空")
    st = await ctx.store.get(session_id)
    st.finished = True
    st.summary = summary
    return f"面试已标记结束。请向候选人复述这段总结：{summary}"


def build_default_registry() -> ToolRegistry:
    """默认工具集。`session_id` 由执行层注入 arguments，工具函数本身不必感知 HTTP 层。"""
    reg = ToolRegistry()
    reg.register(SEARCH_SPEC, _search)
    reg.register(PROGRESS_SPEC, _progress)
    reg.register(NOTE_SPEC, _note)
    reg.register(FINISH_SPEC, _finish)
    return reg
