# -*- coding: utf-8 -*-
"""真实模型冒烟：验证 native function calling 是否能让 GLM-4-Flash 真正调用工具。

跑法：
    ZHIPU_API_KEY=xxx python scripts/smoke_native.py

只做最小验证（3 轮候选人对答），确认「工具被调用」这一核心结论，不作为性能评测。
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.loop import AgentLoop
from src.agent.state import SessionStore
from src.agent.tools import build_default_registry, load_question_bank
from src.clients.llm import OpenAICompatClient

USER_TURNS = [
    "我是王柘童，深大 BME 硕士，做过 MALDI 质谱判别项目和 AI 模拟面试官毕设。",
    "我对 RAG 和工具调用比较熟，毕设里手写了一个 ReAct 循环来调用检索和笔记工具。",
    "好的，那我们继续，你随便挑个项目深挖。",
]


async def main() -> None:
    key = os.environ.get("ZHIPU_API_KEY")
    if not key:
        raise SystemExit("请通过环境变量 ZHIPU_API_KEY 传入智谱 Key")
    client = OpenAICompatClient(
        model="glm-4-flash",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key=key,
        provider="zhipu",
        temperature=0.7,
        max_tokens=1024,
    )
    bank = load_question_bank("config/question_bank.json")
    reg = build_default_registry(bank)
    store = SessionStore()
    loop = AgentLoop(client, reg, store, bank, max_steps=3, protocol="native")

    messages: list[dict] = []
    total_tool_calls = 0
    for i, u in enumerate(USER_TURNS):
        messages.append({"role": "user", "content": u})
        run = await loop.run(
            messages,
            session_id="smoke-native",
            profile={"position": "AI 应用工程师", "level": "校招应届", "style": "专业严谨"},
        )
        total_tool_calls += run.tool_calls
        print(f"--- 第 {i + 1} 轮 ---")
        print("回复:", run.reply[:140].replace("\n", " "))
        print(f"工具调用: {run.tool_calls} 次 | 停止原因: {run.stopped} | 步数: {len(run.steps)}")
        for st in run.steps:
            if st.tool_name:
                print(f"   · 工具: {st.tool_name} (ok={st.ok})")
        messages.append({"role": "assistant", "content": run.reply})

    print("\n==== 汇总 ====")
    print(f"3 轮累计工具调用: {total_tool_calls} 次")
    print("结论:", "✅ native 工具调用已生效" if total_tool_calls > 0 else "❌ 仍为 0 调用，需强化 prompt 或回退 handwritten")


if __name__ == "__main__":
    asyncio.run(main())
