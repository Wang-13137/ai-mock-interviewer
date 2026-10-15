# -*- coding: utf-8 -*-
"""W1 Step1：非流式 vs 流式的端到端时延对比基准。

产出：scripts/../reports/bench_<tag>_<ts>.json 与同名 .md —— 那张表就是简历第一条
「首字延迟 / P95」两组数字的**唯一来源**。没跑过就没有数字，别手写。

用法
----
1) 先起服务（推荐打真实 HTTP，端到端才有意义）：
     python -m src.cli serve
   export AIMI_LLM__PROVIDER=zhipu && export ZHIPU_API_KEY=...

2) 跑基准：
     python scripts/bench.py --repeat 30 --tag glm4flash

3) 只想验证脚本本身（离线替身、不需要服务进程）：
     python scripts/bench.py --inline --repeat 5

注意
----
* --inline 模式走进程内 ASGI，**不含网络与 uvicorn 栈开销**，且替身不会真流式，
  该模式下的 TTFT 只能用来验证脚本，不能上简历。
*  health.ready=false 时报告顶部会打 MOCK 警示：替身的数字一律不得写进任何材料。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from src.metrics import percentile, summarize  # noqa: E402  复用同一套统计口径

DEFAULT_QUESTIONS = ROOT / "config" / "bench_questions.json"
DEFAULT_REPORT_DIR = ROOT / "reports"


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class CallStat:
    mode: str            # plain | stream
    qid: str
    ok: bool
    ttft_ms: Optional[float] = None     # 客户端视角首字延迟
    total_ms: float = 0.0               # 客户端视角端到端
    chars: int = 0
    server_ttft_ms: Optional[float] = None   # 服务端自己埋的 TTFT
    server_total_ms: Optional[float] = None
    error: str = ""

    @property
    def as_row(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# 单次调用
# --------------------------------------------------------------------------- #
async def call_plain(client: httpx.AsyncClient, payload: Dict[str, Any], qid: str) -> CallStat:
    t0 = time.perf_counter()
    try:
        resp = await client.post("/v1/chat", json=payload, timeout=90.0)
    except httpx.HTTPError as exc:
        return CallStat("plain", qid, False, error=f"{type(exc).__name__}: {exc}")

    total_ms = (time.perf_counter() - t0) * 1000
    if resp.status_code != 200:
        return CallStat("plain", qid, False, total_ms=total_ms, error=f"HTTP {resp.status_code}: {resp.text[:120]}")

    body = resp.json()
    return CallStat(
        mode="plain",
        qid=qid,
        ok=True,
        ttft_ms=None,               # 非流式在客户端看得到「首字」，它就是全部
        total_ms=round(total_ms, 2),
        chars=len(body.get("reply", "")),
        server_total_ms=body.get("latency_ms"),
    )


async def call_stream(client: httpx.AsyncClient, payload: Dict[str, Any], qid: str) -> CallStat:
    t0 = time.perf_counter()
    ttft: Optional[float] = None
    server_ttft: Optional[float] = None
    server_total: Optional[float] = None
    chars = 0

    try:
        async with client.stream("POST", "/v1/chat/stream", json=payload, timeout=90.0) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", errors="ignore")
                return CallStat("stream", qid, False, error=f"HTTP {resp.status_code}: {body[:120]}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[len("data:"):].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                kind = event.get("type")
                if kind == "delta" and ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                    chars += len(event.get("delta", ""))
                elif kind == "delta":
                    chars += len(event.get("delta", ""))
                elif kind == "done":
                    server_ttft = event.get("ttft_ms")
                    server_total = event.get("total_ms")
                elif kind == "error":
                    return CallStat("stream", qid, False, error=str(event.get("error"))[:160])
    except httpx.HTTPError as exc:
        return CallStat("stream", qid, False, error=f"{type(exc).__name__}: {exc}")

    return CallStat(
        mode="stream",
        qid=qid,
        ok=True,
        ttft_ms=round(ttft, 2) if ttft is not None else None,
        total_ms=round((time.perf_counter() - t0) * 1000, 2),
        chars=chars,
        server_ttft_ms=server_ttft,
        server_total_ms=server_total,
    )


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def summarize_mode(stats: List[CallStat]) -> Dict[str, Any]:
    rows = [s.as_row for s in stats if s.ok]
    ttft = [r["ttft_ms"] for r in rows if r["ttft_ms"] is not None]
    e2e = [r["total_ms"] for r in rows]
    srv = [r["server_ttft_ms"] for r in rows if r["server_ttft_ms"] is not None]
    return {
        "n": len(stats),
        "success": len(rows),
        "failed": len(stats) - len(rows),
        "client_ttft": {
            "mean": round(statistics.fmean(ttft), 2) if ttft else None,
            "p50": round(percentile(ttft, 50), 2) if ttft else None,
            "p95": round(percentile(ttft, 95), 2) if ttft else None,
        },
        "client_e2e": {
            "mean": round(statistics.fmean(e2e), 2) if e2e else None,
            "p50": round(percentile(e2e, 50), 2) if e2e else None,
            "p95": round(percentile(e2e, 95), 2) if e2e else None,
        },
        "server_ttft_p50": round(percentile(srv, 50), 2) if srv else None,
        "chars_mean": round(statistics.fmean([r["chars"] for r in rows]), 1) if rows else None,
    }


def render_md(
    tag: str,
    provider: str,
    model: str,
    ready: bool,
    modes: Dict[str, Dict[str, Any]],
    meta: Dict[str, Any],
) -> str:
    def cell(d: Optional[float]) -> str:
        return "—" if d is None else f"{d:.1f}"

    lines = [
        f"# 时延基准报告 `{tag}`",
        "",
        f"- 时间：{meta['ts']}",
        f"- provider / model：`{provider}` / `{model}`",
        f"- 服务 `{meta['target']}`，问题集 {meta['questions']} 条 × 重复 {meta['repeat']} 次，并发 {meta['concurrency']}",
        "",
    ]
    if not ready:
        lines += [
            "> [!WARNING]",
            "> 本次跑在**离线替身 EchoClient** 上（health.ready=false）。",
            "> 这些数字只证明链路可用，**不得写进简历、面经或任何对外材料**。",
            "> 填好真实 Key 后重跑一次，才有可引用的数字。",
            "",
        ]

    lines += [
        "| 链路 | 样本 | 客户端 TTFT P50 | TTFT P95 | 端到端 P50 | 端到端 P95 | 平均字数 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for mode, name in (("plain", "非流式 `/v1/chat`"), ("stream", "流式 `/v1/chat/stream`")):
        s = modes.get(mode)
        if not s:
            continue
        lines.append(
            f"| {name} | {s['success']}/{s['n']} | {cell(s['client_ttft']['p50'])} | "
            f"{cell(s['client_ttft']['p95'])} | {cell(s['client_e2e']['p50'])} | "
            f"{cell(s['client_e2e']['p95'])} | {cell(s['chars_mean'])} |"
        )

    lines += ["", "## 结论口径（写简历前先读）", ""]
    if ready:
        st, pl = modes.get("stream"), modes.get("plain")
        if st and pl and st["client_ttft"]["p50"] and pl["client_e2e"]["p50"]:
            drop = round((1 - st["client_ttft"]["p50"] / pl["client_e2e"]["p50"]) * 100, 1)
            lines.append(
                f"- 用户从「发问到看见第一个字」的等待，由非流式的 {pl['client_e2e']['p50']:.1f} ms "
                f"降到流式后的 {st['client_ttft']['p50']:.1f} ms（降幅 {drop}%）"
            )
    else:
        lines.append("- 替身模式：无可对比结论。")
    lines += [
        "- TTFT 与端到端是两件事：TTFT 管「体感」，端到端管「总成本」，两条都要留。",
        "- 样本量建议 ≥ 30 次再看 P95；只跑几次的分位数没有意义。",
        "",
        "> 红线：引用任何一个数字前，确认它来自本次报告，且 `ready=true`。",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
async def build_client(args: argparse.Namespace):
    """返回 (client, target描述, provider, model, ready)。"""
    if args.inline:
        # 进程内直连：不触发 uvicorn/lifespan，手动装配依赖（tests 里已覆盖 lifespan 本身）
        from src.agent.state import SessionStore
        from src.agent.tools import build_default_registry, load_question_bank
        from src.app import build_client as build_llm_client, create_app
        from src.core.config import load_settings
        from src.metrics import JsonlSink

        settings = load_settings()
        app = create_app(settings)
        settings = settings.model_copy(update={"metrics": settings.metrics.model_copy(update={"enabled": False})})
        app.state.settings = settings
        app.state.client = build_llm_client(settings)
        app.state.sink = JsonlSink(settings.metrics.path, enabled=False)
        # lifespan 不会触发（ASGITransport 不调 lifespan），手工补齐 Agent 依赖
        app.state.store = SessionStore()
        app.state.bank = load_question_bank(settings.agent.question_bank)
        app.state.registry = build_default_registry(
            app.state.bank,
            strategy=settings.rag.strategy,
            embedder_name=settings.rag.embedder,
            model_name=settings.rag.model_name,
            min_score=settings.rag.min_score,
            reranker_name=settings.rag.rerank,
            reranker_model=settings.rag.reranker_model,
        )
        app.state.retriever = app.state.registry.retriever

        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://inline")
        health = (await client.get("/health")).json()
        return client, "inline ASGI（进程内，无网络栈）", health["provider"], health["model"], health["ready"]

    base = args.base_url.rstrip("/")
    client = httpx.AsyncClient(base_url=base)
    try:
        resp = await client.get("/health", timeout=5.0)
        resp.raise_for_status()
        health = resp.json()
    except Exception as exc:
        await client.aclose()
        raise SystemExit(
            f"连不上服务 {base}：{exc}\n"
            "先在另一个终端执行 `python -m src.cli serve`，或加 --inline 用进程内模式。"
        )
    return client, base, health["provider"], health["model"], health["ready"]


async def run(args: argparse.Namespace) -> Path:
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    client, target, provider, model, ready = await build_client(args)

    modes = ["plain", "stream"] if args.mode == "both" else [args.mode]
    payload_of = lambda q: {
        "session_id": f"bench-{q['id']}",
        "messages": [{"role": "user", "content": q["text"]}],
        "profile": {"position": "AI 应用工程师", "level": "校招应届", "style": "专业严谨"},
    }

    sem = asyncio.Semaphore(args.concurrency)
    stats: Dict[str, List[CallStat]] = {m: [] for m in modes}

    async def one(mode: str, q: Dict[str, Any], warm: bool) -> None:
        async with sem:
            fn = call_plain if mode == "plain" else call_stream
            st = await fn(client, payload_of(q), q["id"])
        if not warm:
            stats[mode].append(st)
        print(f"  [{mode:6s}] {q['id']} {'warmup' if warm else ''} "
              f"ttft={st.ttft_ms} total={st.total_ms} chars={st.chars} {st.error}", flush=True)

    try:
        if not args.no_warmup:  # 丢弃冷启动：首次调用含模型/连接预热
            print("warmup…", flush=True)
            await asyncio.gather(*[one(m, questions[0], True) for m in modes])

        print(f"benchmarking {len(questions)} questions × {args.repeat} repeats…", flush=True)
        for rep in range(args.repeat):
            tasks = [one(m, q, False) for m in modes for q in questions]
            await asyncio.gather(*tasks)
            print(f"  repeat {rep + 1}/{args.repeat} done", flush=True)
    finally:
        await client.aclose()

    summary = {m: summarize_mode(stats[m]) for m in modes}
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    reports = DEFAULT_REPORT_DIR
    reports.mkdir(parents=True, exist_ok=True)
    meta = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "target": target,
        "provider": provider,
        "model": model,
        "ready": ready,
        "questions": len(questions),
        "repeat": args.repeat,
        "concurrency": args.concurrency,
        "inline": bool(args.inline),
    }

    stem = reports / f"bench_{args.tag}_{ts}"
    (stem.with_suffix(".json")).write_text(
        json.dumps({"meta": meta, "summary": summary,
                    "raw": {m: [s.as_row for s in stats[m]] for m in modes}},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path = stem.with_suffix(".md")
    md_path.write_text(render_md(args.tag, provider, model, ready, summary, meta), encoding="utf-8")

    print("\n" + render_md(args.tag, provider, model, ready, summary, meta))
    print(f"\n报告已写入：{stem.with_suffix('.json')}")
    print(f"          {md_path}")
    _ = summarize  # 保持与 src.metrics 的导入关系显式可见
    return md_path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="非流式 vs 流式端到端时延对比基准")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--inline", action="store_true", help="进程内直连 app，无需起服务")
    p.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--mode", choices=["both", "plain", "stream"], default="both")
    p.add_argument("--tag", default="run")
    p.add_argument("--no-warmup", action="store_true")
    args = p.parse_args(argv)
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
