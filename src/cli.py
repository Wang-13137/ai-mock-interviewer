# -*- coding: utf-8 -*-
"""命令行入口：`aimi serve`（起服务）与 `aimi chat`（终端里直接对话）。"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Optional

from .app import build_client, create_app
from .core.config import load_settings
from .prompts.interviewer import build_messages


async def _chat_loop(settings) -> None:
    client = build_client(settings)
    print(f"[provider={client.provider_name} model={client.model} ready={client.ready}]")
    print("输入内容后回车即投递；输入 exit / 直接回车退出。\n")

    history = []
    profile = {"position": "AI 应用工程师", "level": "校招应届", "style": "专业严谨"}
    try:
        while True:
            user = input("你 > ").strip()
            if not user or user.lower() in {"exit", "quit"}:
                break
            history.append({"role": "user", "content": user})

            print("面试官 > ", end="", flush=True)
            parts = []
            async for delta in client.astream(
                build_messages(history, profile)
            ):
                parts.append(delta)
                print(delta, end="", flush=True)
            print("\n")
            reply = "".join(parts)
            history.append({"role": "assistant", "content": reply})
    finally:
        await client.aclose()


async def _agent_loop(settings) -> None:
    """带工具的对话：把每一步工具调用打印出来，肉眼就能验收 loop 是否收敛。"""
    from src.agent.loop import AgentLoop
    from src.agent.state import SessionStore
    from src.agent.tools import build_default_registry, load_question_bank

    client = build_client(settings)
    store = SessionStore()
    bank = load_question_bank(settings.agent.question_bank)
    registry = None
    if settings.agent.enabled:
        registry = build_default_registry(
            bank,
            strategy=settings.rag.strategy,
            embedder_name=settings.rag.embedder,
            model_name=settings.rag.model_name,
            min_score=settings.rag.min_score,
            reranker_name=settings.rag.rerank,
            reranker_model=settings.rag.reranker_model,
        )
    loop = AgentLoop(client, registry, store, bank, max_steps=settings.agent.max_steps)

    print(f"[provider={client.provider_name} model={client.model} ready={client.ready}]")
    print(f"[tools={','.join(registry.names()) if registry else '无'} max_steps={settings.agent.max_steps}]")
    print("输入内容后回车即投递；输入 exit / 直接回车退出。\n")

    history: list = []
    profile = {"position": "AI 应用工程师", "level": "校招应届", "style": "专业严谨"}
    session_id = "cli-session"
    try:
        while True:
            user = input("你 > ").strip()
            if not user or user.lower() in {"exit", "quit"}:
                break
            history.append({"role": "user", "content": user})

            run = await loop.run(history, profile, session_id=session_id)
            for st in run.steps:
                if st.kind in ("tool_call", "tool_result", "parse_error", "loop_guard"):
                    mark = "OK " if st.ok else "ERR"
                    detail = st.tool_name or st.content[:40]
                    print(f"   · [{mark}] {st.kind:<12} {detail}")
            print(f"\n面试官 > {run.reply}")
            print(f"   · {run.model_calls} 次模型调用 / {run.tool_calls} 次工具 / stopped={run.stopped}\n")
            history.append({"role": "assistant", "content": run.reply})
    finally:
        await client.aclose()


def cmd_agent(args: argparse.Namespace) -> None:
    settings = load_settings(**_overrides(args))
    asyncio.run(_agent_loop(settings))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    settings = load_settings(**_overrides(args))
    app = create_app(settings)
    uvicorn.run(
        app,
        host=args.host or settings.server.host,
        port=args.port or settings.server.port,
        reload=args.reload,
        log_level="debug" if settings.env == "dev" else "info",
    )


def cmd_chat(args: argparse.Namespace) -> None:
    settings = load_settings(**_overrides(args))
    asyncio.run(_chat_loop(settings))


def cmd_eval(args: argparse.Namespace) -> None:
    """跑一批评测：生成 -> 裁判打分 -> 与人工标注对齐。"""
    from src.eval.dataset import load_dataset
    from src.eval.runner import run_eval, write_report

    overrides: dict = {}
    if getattr(args, "generator", None):
        overrides["llm"] = {"provider": args.generator}
    if getattr(args, "judge", None):
        overrides["judge"] = {"provider": args.judge}
    settings = load_settings(**overrides)

    dataset = load_dataset(args.dataset)
    report = asyncio.run(run_eval(settings, dataset, limit=args.limit, concurrency=args.concurrency))
    md, js = write_report(report)
    print("\n" + report.render_md())
    print(f"\n报告：{md}\n      {js}")


def _overrides(args: argparse.Namespace) -> dict:
    out = {}
    if getattr(args, "provider", None):
        out["llm"] = {"provider": args.provider}
    return out


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="aimi", description="AI 模拟面试官 服务/命令行工具")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="启动 FastAPI 服务")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.add_argument("--provider", default=None, help="覆盖 llm.provider，如 zhipu / deepseek")
    p_serve.set_defaults(func=cmd_serve)

    p_chat = sub.add_parser("chat", help="终端内直接和面试官对话（不带工具）")
    p_chat.add_argument("--provider", default=None)
    p_chat.set_defaults(func=cmd_chat)

    p_agent = sub.add_parser("agent", help="终端对话 + 工具调用，并打印每步 trace")
    p_agent.add_argument("--provider", default=None)
    p_agent.set_defaults(func=cmd_agent)

    p_eval = sub.add_parser("eval", help="跑评测集：生成 → 裁判打分 → 一致性统计")
    p_eval.add_argument("--dataset", default="config/eval_set.example.json")
    p_eval.add_argument("--generator", default=None, help="生成模型 provider，如 zhipu")
    p_eval.add_argument("--judge", default=None, help="裁判模型 provider，**应与 generator 不同**")
    p_eval.add_argument("--limit", type=int, default=None)
    p_eval.add_argument("--concurrency", type=int, default=4)
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
