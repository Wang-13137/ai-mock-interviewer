# -*- coding: utf-8 -*-
"""手写 tool-call 协议：模型怎么「表达」一次工具调用，我们怎么把它解析回来。

约定（写在 system prompt 里交给模型）：

```
```tool
{"name": "search_question_bank", "arguments": {"query": "RAG 重排", "top_k": 3}}
```
```

不输出工具代码块 = 直接给出面向候选人的自然语言回答。

为什么用代码块而不是原生 tools 参数：
- 任何 OpenAI 兼容模型都吃这一套，包括免费额度那几家；
- 解析失败可以被捕获并**回喂给模型纠正**，这本身就是一条可观测指标（parse_errors）；
- 失败模式可枚举 → 可以做测试用例 → 可以做故障率统计。

解析器的健壮性分层（从严格到宽松）：
1. ```tool 围栏里的 JSON；
2. 任意围栏里的、带 name 字段的 JSON；
3. 全文扫描最外层 `{...}` 且含 name 字段。
三层都失败：没有 JSON → 视为终答；有 JSON 但坏 → 抛 ToolCallParseError，回喂模型。
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

from ..schemas.agent import ToolCall, ToolSpec

TOOL_FENCE = "tool"
_OPEN_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\s*")
_FENCE_CONTENT_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\s*(.*?)```", re.DOTALL)


class ToolCallParseError(ValueError):
    """模型确实想调用工具，但格式坏了。错误详情要回喂给模型让它重试。"""

    def __init__(self, detail: str, raw: str = ""):
        super().__init__(detail)
        self.detail = detail
        self.raw = raw[:400]


# --------------------------------------------------------------------------- #
# 指令渲染
# --------------------------------------------------------------------------- #
def render_tools_prompt(specs: List[ToolSpec]) -> str:
    """把工具清单渲染成提示词片段。保持每行短、字段少，小模型也能照做。"""
    if not specs:
        return ""
    lines = [
        "## 可用工具",
        "需要查资料或记录进度时，**只**输出如下格式的一个代码块，不要同时输出别的文字：",
        "",
        "```tool",
        '{"name": "工具名", "arguments": {"参数名": "参数值"}}',
        "```",
        "",
        "工具清单：",
    ]
    for spec in specs:
        params = spec.parameters.get("properties", {}) or {}
        required = spec.parameters.get("required", []) or []
        param_desc = "，".join(
            f"{k}{'（必填）' if k in required else ''}: {v.get('description', '')}"
            for k, v in params.items()
        )
        lines.append(f"- **{spec.name}**：{spec.description}" + (f"（参数：{param_desc}）" if param_desc else ""))
    lines += [
        "",
        "## 调用规则",
        "1. 一次只调用一个工具，拿到结果后再决定下一步。",
        "2. 不需要工具时，直接给出面向候选人的自然语言回答，**不要**输出代码块。",
        "3. 工具返回结果里写明失败时，修正参数重试；重试两次仍失败就直接凭常识回答，并向候选人说明。",
        "4. 严禁用同样的参数重复调用同一个工具。",
    ]
    return "\n".join(lines)


OBSERVATION_TEMPLATE = """<tool_result name="{name}" ok="{ok}">
{content}
</tool_result>
以上是工具返回的结果。请据此继续面试；如确有需要可再调用一次工具，否则直接回答候选人。"""


def render_observation(result) -> str:
    """把工具结果渲染成喂回模型的 observation。"""
    body = result.content if result.ok else f"[调用失败] {result.error}"
    return OBSERVATION_TEMPLATE.format(name=result.tool_name, ok=str(result.ok).lower(), content=body)


LOOP_GUARD_MESSAGE = (
    "你刚刚用相同的参数重复调用了同一个工具，这属于无效循环。"
    "请基于已经拿到的信息直接回答候选人，不要再调用工具。"
)

FINAL_FORCE_MESSAGE = (
    "已达到最大工具调用步数。请立即停止调用工具，用自然语言给出你的回答。"
)


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #
def _looks_like_tool_call(obj: object) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("name"), str) and bool(obj.get("name"))


def _build(obj: dict) -> ToolCall:
    args = obj.get("arguments")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ToolCallParseError("arguments 必须是对象类型", raw=json.dumps(obj, ensure_ascii=False))
    return ToolCall(name=str(obj["name"]), arguments=args)


def _extract_json_candidates(text: str) -> List[Tuple[str, str]]:
    """返回 [(来源标签, 候选 JSON 文本)]，按优先级从高到低。"""
    cands: List[Tuple[str, str]] = []

    # 1) ```tool / ```json / 任意围栏
    for lang, body in _FENCE_CONTENT_RE.findall(text):
        body = body.strip()
        if not body:
            continue
        tag = "tool" if lang.lower() == TOOL_FENCE else "fence"
        cands.append((tag, body))

    # 2) 全文里所有看起来像 {...} 的最外层对象（围栏外的裸 JSON）
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    cands.append(("bare", text[start:i + 1]))
                    start = -1
    return cands


def try_parse_tool_call(text: str) -> Tuple[Optional[ToolCall], Optional[ToolCallParseError]]:
    """宽松解析。

    返回：
        (ToolCall, None)        —— 解析成功
        (None, None)            —— 文本里没有工具调用，视为终答
        (None, ParseError)      —— 有调用意图但格式坏了
    """
    if not text or not text.strip():
        return None, None

    stripped = text.strip()
    saw_candidate = False
    last_detail = ""

    for tag, raw in _extract_json_candidates(stripped):
        obj = _load_lenient(raw)
        if obj is None:
            saw_candidate = True
            last_detail = f"JSON 无法解析：{raw[:80]}"
            continue
        if _looks_like_tool_call(obj):
            try:
                return _build(obj), None
            except ToolCallParseError as exc:
                return None, ToolCallParseError(exc.detail, raw=raw)
        saw_candidate = True
        last_detail = "JSON 缺少 name 字段（或 name 不是字符串）"

    if saw_candidate:
        return None, ToolCallParseError(last_detail or "未识别的工具调用格式", raw=stripped)
    return None, None


def _load_lenient(raw: str):
    """尽力解析 JSON：失败时返回 None（不抛，交给上层决定是否回喂模型）。"""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # 常见坏味修一次：中文引号、尾部多余逗号、代码块里混了说明文字
    fixed = raw.replace("“", '"').replace("”", '"').replace("，", ",")
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    fixed = _OPEN_FENCE_RE.sub("", fixed).strip().rstrip("`").strip()
    m = re.search(r"\{.*\}", fixed, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def parse_tool_call(text: str) -> Optional[ToolCall]:
    """严格入口：有意图但格式坏时抛错（测试与评测脚本用）。"""
    call, err = try_parse_tool_call(text)
    if err:
        raise err
    return call
