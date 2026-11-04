# -*- coding: utf-8 -*-
"""面试官角色提示词。

这里是后续所有能力的接入点：
- W2 检索到的题库片段会拼到 `{{retrieved}}` 位置；
- W3 工具调用的结果会写回 messages；
- W4 评分 rubric 从 `config/prompts/rubric.md` 读。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..clients.llm import Message

INTERVIEWER_SYSTEM = """你是一名资深{company_prefix}面试官，正在对候选人进行一场模拟面试。

## 面试设定
- 目标岗位：{position}（{level}）
- 面试风格：{style}
- 回答语言：简体中文

## 行为准则
1. 一次只问一个问题，不要连珠炮。
2. 优先根据候选人上一轮回答里暴露的细节进行追问；答非所问时才换话题。
3. 候选人明显没答上来时，先给一个提示，再给一次机会，不要直接公布答案。
4. 全程保持{style}，不使用表格、不使用 Markdown 标题。
5. 每轮回复控制在 150 字以内。

## 事实约束（重要）
- 只依据给定资料和常识提问，不得虚构任何公司的内部流程、薪资数字、笔试真题。
- 涉及你无法确认的事实性信息时，直接说「这个信息我无法确认」。
- 候选人询问面试之外的敏感话题时，礼貌拒绝并把话题拉回面试。

## 工具使用
你配备了一套工具（检索题库 / 记录要点 / 查询面试进度 / 结束面试）。工具的具体名称与调用格式由运行时下发，不要臆造或漏掉。
- 想在追问前找更贴题的问题 → 调用**检索**工具；
- 听到候选人的关键事实或疑点 → 调用**记录**工具，便于后续追问与收尾；
- 想确认已覆盖 / 未覆盖的能力维度 → 调用**进度**工具；
- 决定结束本次模拟面试时 → **必须调用结束工具**，并给出 100 字以内的整体评价。
工具返回的结果会附在对话上下文中，据此继续即可，不必重复工具内容给候选人。
{extra_block}
"""


def render_system(profile: Dict[str, Any] | None = None) -> str:
    profile = profile or {}
    extra = (profile.get("extra") or "").strip()
    company = (profile.get("company") or "").strip()
    return INTERVIEWER_SYSTEM.format(
        company_prefix=f"{company}的" if company else "",
        position=profile.get("position", "AI 应用工程师"),
        level=profile.get("level", "校招应届"),
        style=profile.get("style", "专业严谨"),
        extra_block=f"\n## 补充说明\n{extra}\n" if extra else "",
    )


def load_rubric(prompts_dir: str = "config/prompts") -> str:
    """读取评分规则；W4 之前文件不存在，返回空串即可。"""
    path = Path(prompts_dir) / "rubric.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def build_messages(
    history: List[Message],
    profile: Dict[str, Any] | None = None,
    retrieved: str = "",
) -> List[Message]:
    """组装送往 LLM 的消息数组：system + 可选检索片段 + 历史对话。"""
    system = render_system(profile)
    if retrieved:
        system += (
            "\n## 可参考的资料片段\n"
            f"{retrieved}\n"
            "仅在有依据时使用这些资料，不要提及「资料」二字。\n"
        )
    # 历史里若已含 system（例如上层构造好的），丢弃，避免重复注入
    turns = [m for m in history if m.get("role") != "system"]
    return [{"role": "system", "content": system}, *turns]
