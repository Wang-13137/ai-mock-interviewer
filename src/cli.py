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

    p_chat = sub.add_parser("chat", help="终端内直接和面试官对话")
    p_chat.add_argument("--provider", default=None)
    p_chat.set_defaults(func=cmd_chat)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
