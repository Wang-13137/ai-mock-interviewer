# -*- coding: utf-8 -*-
"""命令行入口：`aimi serve`（起服务）与 `aimi chat`（终端里直接对话）。

另含两个平台的运维入口：
    aimi data  ls|sync|lineage     数据中台：数据集登记、质量、血缘
    aimi evals run|ls|gate         评测中台：跑评测、看历史、做回归门禁
"""

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
    from src.agent.memory import MemoryManager, MemoryPolicy

    memory = None
    if settings.memory.enabled:
        memory = MemoryManager(
            client=client,
            policy=MemoryPolicy(
                max_tokens=settings.memory.max_tokens,
                keep_recent=settings.memory.keep_recent,
                summary_max_chars=settings.memory.summary_max_chars,
            ),
        )
    loop = AgentLoop(client, registry, store, bank, max_steps=settings.agent.max_steps, memory=memory)

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


# --------------------------------------------------------------------------- #
# 数据中台 / 评测中台
# --------------------------------------------------------------------------- #
def cmd_data(args: argparse.Namespace) -> None:
    from src.datahub import DataHub

    hub = DataHub(load_settings())

    if args.action == "sync":
        targets = [args.dataset] if args.dataset else [s.id for s in hub.list_datasets()]
        for dataset_id in targets:
            r = hub.sync(dataset_id, force=args.force)
            mark = {"registered": "✅", "unchanged": "＝", "blocked": "⛔", "missing": "❓"}.get(r.status, "?")
            print(f"{mark} {r.dataset_id:<16} {r.status:<11} rows={r.rows:<4} hash={r.content_hash or '-':<18} {r.message}")
            if r.quality:
                for e in r.quality.errors:
                    print(f"     [error] {e.code}: {e.message}")
        return

    if args.action == "lineage":
        graph = hub.lineage()
        if not graph.edges:
            print("（无引用关系）")
        for e in graph.edges:
            print(f"{e.from_dataset} --{e.field}--> {e.to_dataset}")
        return

    # ls
    print(f"{'数据集':<16} {'版本':>4} {'行数':>5}  {'质量':<8} {'hash':<18} 被引用")
    for rec in hub.list_datasets():
        latest = rec.latest
        flag = "（未同步）" if latest is None else ("（dirty）" if rec.dirty else "")
        print(
            f"{rec.spec.id:<16} {rec.versions:>4} {latest.rows if latest else 0:>5}  "
            f"{(latest.quality_status if latest else '-'):<8} "
            f"{(latest.content_hash if latest else '-'):<18} {','.join(rec.referenced_by) or '-'}{flag}"
        )


def cmd_evals(args: argparse.Namespace) -> None:
    import json as _json

    from src.datahub import DataHub
    from src.evalhub import EvalHub, EvalRunRequest, EvalTask
    from src.evalhub.gate import render_gate

    settings = load_settings(**({"llm": {"provider": args.provider}} if getattr(args, "provider", None) else {}))
    datahub = DataHub(settings)
    datahub.sync_all()
    hub = EvalHub(settings, datahub)

    if args.action == "ls":
        runs = hub.list_runs(task=args.task, limit=args.limit)
        if not runs:
            print("暂无 run")
        for rec in runs:
            print(f"  {rec.summary_line()}  dataset={rec.dataset_id}@{rec.dataset_hash[:8]}")
        return

    if args.action == "gate":
        gate = hub.gate_run(args.run_id, baseline=args.baseline, threshold=args.threshold)
        print(render_gate(gate))
        return

    # run
    params = _json.loads(args.params) if args.params else {}
    req = EvalRunRequest(
        task=EvalTask(args.task),
        dataset_id=args.dataset or "",
        params=params,
        baseline_run=args.baseline,
    )
    record = asyncio.run(hub.run(req))
    print(f"\nrun {record.id}  status={record.status.value}  trusted={record.trusted}")
    if record.error:
        print(f"  ✗ {record.error}")
    for k, v in sorted(record.metrics.items()):
        print(f"  {k:<22} {v}")
    for w in record.warnings:
        print(f"  ⚠️  {w}")
    if record.gate:
        print()
        print(render_gate(record.gate))
    if not record.trusted:
        print("\n注意：本 run 不可外引，也不得作为回归基线。")


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

    # ---- 数据中台 ----
    p_data = sub.add_parser("data", help="数据中台：同步 / 列表 / 血缘")
    data_sub = p_data.add_subparsers(dest="action", required=True)
    d_sync = data_sub.add_parser("sync", help="同步数据集（内容未变则为 unchanged）")
    d_sync.add_argument("--dataset", default=None, help="只同步指定数据集")
    d_sync.add_argument("--force", action="store_true", help="忽略 hash 幂等，强制重新校验")
    d_sync.set_defaults(func=cmd_data)
    d_ls = data_sub.add_parser("ls", help="列出已登记数据集与质量状态")
    d_ls.set_defaults(func=cmd_data)
    d_lineage = data_sub.add_parser("lineage", help="查看数据集引用关系")
    d_lineage.set_defaults(func=cmd_data)

    # ---- 评测中台 ----
    p_evals = sub.add_parser("evals", help="评测中台：跑评测 / 历史 / 回归门禁")
    e_sub = p_evals.add_subparsers(dest="action", required=True)
    e_run = e_sub.add_parser("run", help="执行一次评测并落库")
    e_run.add_argument("--task", default="retrieval", choices=["retrieval", "generation", "agent"])
    e_run.add_argument("--dataset", default=None, help="数据中台里的 dataset id")
    e_run.add_argument("--params", default=None, help='任务参数 JSON，如 {"top_k": 5}')
    e_run.add_argument("--baseline", default=None, help="基线 run id，或 latest")
    e_run.add_argument("--provider", default=None)
    e_run.set_defaults(func=cmd_evals)
    e_ls = e_sub.add_parser("ls", help="历史 run 列表")
    e_ls.add_argument("--task", default=None)
    e_ls.add_argument("--limit", type=int, default=10)
    e_ls.set_defaults(func=cmd_evals)
    e_gate = e_sub.add_parser("gate", help="把已有 run 与基线比较")
    e_gate.add_argument("run_id")
    e_gate.add_argument("--baseline", default="latest")
    e_gate.add_argument("--threshold", type=float, default=None)
    e_gate.set_defaults(func=cmd_evals)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
