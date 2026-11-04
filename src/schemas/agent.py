# -*- coding: utf-8 -*-
"""Agent（工具调用）相关的数据契约。

放在 schemas 层是为了避免循环依赖：src/agent/* 可以 import 这里，
API 层（src/app.py）也能直接把 AgentRun 作为响应体返回。

设计取舍：**手写结构化协议 + 厂商原生 function calling 双轨并存**。
- 手写协议（```tool 代码块）保证对免费额度模型的兼容性与可观测性（格式怎么坏都能统计）；
- native 协议把工具以 OpenAI `tools` 参数下发，由模型自行决定是否调用，格式由 API 保证合法；
- 两条路径在 loop 里统一归一成 `ToolCall`，共用同一套执行 / 死循环检测 / 步数上限逻辑。
`AgentConfig.protocol` 字段切换默认走哪条；默认 native（更稳定），handwritten 作为兜底与评测对照。
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# 停止原因：
#   final                —— 模型主动给出自然语言终答（最理想）
#   max_steps_recovered  —— 触到步数上限，兜底回合拿到了终答
#   loop_guard           —— 检测到重复调用，强制收敛
#   error                —— LLM 调用本身失败
StopReason = Literal["final", "max_steps_recovered", "loop_guard", "error"]


class ToolCall(BaseModel):
    """一次工具调用请求（从模型输出里解析出来的）。"""

    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)

    @property
    def signature(self) -> str:
        """用于死循环检测：同名+同参数视为重复。"""
        import json

        try:
            args = json.dumps(self.arguments, ensure_ascii=False, sort_keys=True)
        except Exception:  # 不可序列化时退化为 repr，仍能检测重复
            args = repr(self.arguments)
        return f"{self.name}|{args}"


class ToolResult(BaseModel):
    """工具执行结果。失败时 ok=False 且 error 非空，错误要**回喂给模型**让它自纠。"""

    ok: bool = True
    content: str = ""
    error: str = ""
    tool_name: str = ""


class ToolSpec(BaseModel):
    """工具的对外描述（喂给模型的那份），同时也是参数的 JSON Schema。"""

    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AgentStep(BaseModel):
    """一次循环里的一个动作，构成完整 trace。用来算成功率、步数分布。"""

    index: int
    kind: Literal["llm", "tool_call", "tool_result", "parse_error", "loop_guard", "final", "memory"]
    content: str = ""
    tool_name: str = ""
    arguments: Dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    elapsed_ms: float = 0.0


class AgentUsage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model_calls: int = 0


class AgentRun(BaseModel):
    """一次 Agent 执行的完整结果：给用户的回复 + 全过程 trace + 可统计的指标。"""

    reply: str
    steps: List[AgentStep] = Field(default_factory=list)
    tool_calls: int = 0
    tool_errors: int = 0
    parse_errors: int = 0
    model_calls: int = 0
    stopped: StopReason = "final"
    elapsed_ms: float = 0.0
    usage: AgentUsage = Field(default_factory=AgentUsage)

    @property
    def tool_success_rate(self) -> Optional[float]:
        """工具调用成功率：手写 loop 最核心的一个可交付数字。"""
        if self.tool_calls == 0:
            return None
        return round((self.tool_calls - self.tool_errors) / self.tool_calls, 4)

    def render_md(self) -> str:
        """把 trace 渲染成人能读的样子，调试与写进报告都方便。"""
        lines = [
            f"### 回复\n{self.reply}\n",
            f"- 模型调用 {self.model_calls} 次 / 工具调用 {self.tool_calls} 次"
            f" / 工具失败 {self.tool_errors} 次 / 解析失败 {self.parse_errors} 次",
            f"- 停止原因：{self.stopped}，总耗时 {self.elapsed_ms:.0f} ms",
        ]
        sr = self.tool_success_rate
        if sr is not None:
            lines.append(f"- 工具成功率：{sr:.2%}")
        lines.append("\n### Trace")
        for st in self.steps:
            head = f"{st.index}. [{st.kind}]"
            if st.tool_name:
                head += f" {st.tool_name}"
            head += f" ({st.elapsed_ms:.0f} ms)"
            body = st.content.replace("\n", " ")[:160]
            lines.append(f"{head}\n   {body}")
        return "\n".join(lines)
