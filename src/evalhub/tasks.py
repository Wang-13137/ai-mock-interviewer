# -*- coding: utf-8 -*-
"""三类评测任务的实现。

设计要点：**每个任务自己声明 `trusted`，而不是由中台统一猜。**
因为「这次数字可不可信」的知识是领域相关的：
    · 检索任务只用 BM25 时不涉及任何模型，数字天然真实；
    · 检索任务走 vector/hybrid 但 embedding 退化成哈希替身时，数字不可引用；
    · 生成任务必须生成与裁判都是真实模型，且最好不是同一家（自评偏差）。
把这些判断塞进中台只会造成一堆 if；让任务自己回答，中台只负责执行与记录。

延迟导入的原因：
    `build_client` 定义在 src/app.py，而 app.py 会挂载 evalhub 的路由。
    顶层 import 会形成 app → evalhub.api → tasks → app 的循环。
    函数内导入把这个环打开，代价是每次都走一次 sys.modules 查找（可忽略）。
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..core.config import Settings, is_ready
from .models import CaseRow, TaskInfo

DEFAULT_PROFILE = {"position": "AI 应用工程师", "level": "校招应届", "style": "专业严谨"}


@dataclass
class TaskContext:
    """任务执行环境：只暴露「配置 + 数据」，不暴露存储，任务不写库。"""

    settings: Settings
    datahub: Any  # DataHub，用 Any 避免与 datahub 包互相 import

    def path_of(self, dataset_id: str) -> Path:
        return self.datahub.path_of(dataset_id)

    def load(self, dataset_id: str) -> Any:
        return self.datahub.load_json(dataset_id)


@dataclass
class TaskResult:
    metrics: Dict[str, float] = field(default_factory=dict)
    cases: List[CaseRow] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    trusted: bool = False
    fingerprint_detail: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


# --------------------------------------------------------------------------- #
# 1. 检索任务：Recall@k / MRR@10 / Hit@1（零 LLM，离线可跑）
# --------------------------------------------------------------------------- #
async def run_retrieval(ctx: TaskContext, dataset_id: str, params: Dict[str, Any]) -> TaskResult:
    from ..rag.evaluate import evaluate, load_cases
    from ..rag.corpus import load_bank
    from ..rag.retriever import Retriever

    cfg = ctx.settings.rag
    ks = tuple(params.get("ks") or (1, 3, 5))
    top_k = int(params.get("top_k") or max(ks))

    cases = load_cases(ctx.path_of(dataset_id))
    bank = load_bank(ctx.path_of("question_bank"))
    if not cases:
        return TaskResult(error="评测集为空，无法计算检索指标")

    def _build_and_run() -> Dict[str, Any]:
        retriever = Retriever.from_bank(
            bank,
            strategy=cfg.strategy,
            embedder_name=cfg.embedder,
            model_name=cfg.model_name,
            min_score=cfg.min_score,
            reranker_name=cfg.rerank,
            reranker_model=cfg.reranker_model,
        )
        return evaluate(retriever, cases, ks)

    # BM25 建索引与逐条检索是纯 CPU 活，丢到线程里避免阻塞事件循环
    report = await asyncio.to_thread(_build_and_run)

    metrics: Dict[str, float] = {f"recall@{k}": float(report[f"recall@{k}"]) for k in ks}
    metrics["mrr@10"] = float(report["mrr@10"])
    metrics["hit@1"] = float(report["hit@1"])
    for kind, block in (report.get("by_kind") or {}).items():
        metrics[f"{kind}_mrr@10"] = float(block["mrr@10"])
        metrics[f"{kind}_recall@3"] = float(block.get("recall@3", 0.0))

    case_rows = [
        CaseRow(
            case_id=str(i),
            metrics={"recall@3": float(c["recall"].get("3", c["recall"].get(3, 0.0))), "rr": float(c["rr"])},
            detail={"query": c["query"], "kind": c["kind"], "relevant": c["relevant"], "retrieved": c["retrieved"]},
        )
        for i, c in enumerate(report["cases"])
    ]

    strategy = report["strategy"]
    semantic = bool(report["semantic"])
    warnings: List[str] = []
    if strategy == "bm25":
        trusted = True  # 纯词法检索，不涉及任何模型，数字是真实测量
    elif semantic:
        trusted = True
    else:
        trusted = False
        warnings.append(
            f"检索策略 {strategy} 的 embedding 退化成了 HashingEmbedder 替身（semantic=false），"
            "本次指标仅用于链路验证，不得对外引用、不得作为回归基线。"
        )

    return TaskResult(
        metrics=metrics,
        cases=case_rows,
        warnings=warnings,
        trusted=trusted,
        fingerprint_detail={
            "strategy": strategy,
            "embedder": report["embedder"],
            "semantic_embedding": semantic,
            "min_score": cfg.min_score,
            "rerank": cfg.rerank,
            "top_k": top_k,
            "documents": report["documents"],
            "queries": report["queries"],
        },
    )


# --------------------------------------------------------------------------- #
# 2. 生成任务：LLM-as-judge 五维评分 + 与人工标注的一致性
# --------------------------------------------------------------------------- #
async def run_generation(ctx: TaskContext, dataset_id: str, params: Dict[str, Any]) -> TaskResult:
    from ..eval.dataset import load_dataset
    from ..eval.runner import run_eval

    dataset = load_dataset(ctx.path_of(dataset_id))
    report = await run_eval(
        ctx.settings,
        dataset,
        limit=params.get("limit"),
        concurrency=int(params.get("concurrency") or 4),
    )

    metrics: Dict[str, float] = {"parse_rate": float(report.parse_rate)}
    means = report.mean_score_by_dimension()
    if means:
        metrics["mean_score"] = round(sum(means.values()) / len(means), 3)
        for dim, val in means.items():
            metrics[f"dim::{dim}"] = float(val)
    agree = report.agreement()
    for key in ("acc", "acc_within_1", "kappa", "qwk"):
        if agree.get(key) is not None:
            metrics[key] = float(agree[key])

    case_rows = [
        CaseRow(
            case_id=o.case_id,
            metrics={
                k: float(v)
                for k, v in (("auto_score", o.judge.score), ("human_score", o.human_score))
                if v is not None
            },
            detail={
                "dimension": o.dimension,
                "parse_ok": o.judge.parse_ok,
                "reply_chars": len(o.reply),
                "reason": o.judge.reason[:200],
            },
        )
        for o in report.outcomes
    ]

    # 生成与裁判都必须是真实模型；任一跑在替身上，分数就没有意义
    gen_ready = is_ready(ctx.settings.llm)
    judge_ready = is_ready(ctx.settings.judge)
    warnings = list(report.warnings)
    if not gen_ready:
        warnings.append("生成模型跑在离线替身 EchoClient 上，reply 是模板文本，评分无意义。")
    if not judge_ready:
        warnings.append("裁判跑在离线替身 EchoClient 上，分数仅用于验证链路。")

    return TaskResult(
        metrics=metrics,
        cases=case_rows,
        warnings=warnings,
        trusted=bool(gen_ready and judge_ready),
        fingerprint_detail={
            "generator": report.meta.get("generator"),
            "judge": report.meta.get("judge"),
            "rubric_version": report.meta.get("rubric_version"),
            "dataset_size": report.meta.get("size"),
        },
    )


# --------------------------------------------------------------------------- #
# 3. Agent 任务：工具调用率 / 完成率 / 平均步数
# --------------------------------------------------------------------------- #
async def run_agent(ctx: TaskContext, dataset_id: str, params: Dict[str, Any]) -> TaskResult:
    from ..agent.loop import AgentLoop
    from ..agent.memory import MemoryManager, MemoryPolicy
    from ..agent.state import SessionStore
    from ..agent.tools import build_default_registry, load_question_bank
    from ..app import build_client  # 延迟导入，避免循环

    limit = int(params.get("limit") or 3)
    prompts = ctx.load(dataset_id)
    if isinstance(prompts, dict):
        prompts = prompts.get("items") or []
    prompts = prompts[:limit]
    if not prompts:
        return TaskResult(error="bench 脚本为空，无法跑 Agent 任务")

    cfg = ctx.settings
    client = build_client(cfg, role="chat")
    bank = load_question_bank(str(ctx.path_of("question_bank")))
    registry = build_default_registry(
        bank,
        strategy=cfg.rag.strategy,
        embedder_name=cfg.rag.embedder,
        model_name=cfg.rag.model_name,
        min_score=cfg.rag.min_score,
        reranker_name=cfg.rag.rerank,
        reranker_model=cfg.rag.reranker_model,
    )
    store = SessionStore()
    memory = None
    if cfg.memory.enabled:
        memory = MemoryManager(
            policy=MemoryPolicy(
                max_tokens=cfg.memory.max_tokens,
                keep_recent=cfg.memory.keep_recent,
                summary_max_chars=cfg.memory.summary_max_chars,
            )
        )
    loop = AgentLoop(
        client=client,
        registry=registry,
        store=store,
        bank=bank,
        max_steps=cfg.agent.max_steps,
        memory=memory,
        protocol=cfg.agent.protocol,
    )

    rows: List[CaseRow] = []
    tool_calls_total = 0
    steps_total = 0
    completed = 0
    prompts_with_tools = 0
    try:
        for i, item in enumerate(prompts):
            text = item.get("text") if isinstance(item, dict) else str(item)
            cid = str(item.get("id") if isinstance(item, dict) else i)
            t0 = time.perf_counter()
            run = await loop.run(
                [{"role": "user", "content": text or ""}],
                DEFAULT_PROFILE,
                session_id=f"evalhub-{cid}",
            )
            elapsed = round((time.perf_counter() - t0) * 1000, 1)
            tool_calls_total += run.tool_calls
            steps_total += len(run.steps)
            if run.stopped == "final":
                completed += 1
            if run.tool_calls > 0:
                prompts_with_tools += 1
            rows.append(
                CaseRow(
                    case_id=cid,
                    metrics={
                        "tool_calls": float(run.tool_calls),
                        "steps": float(len(run.steps)),
                        "model_calls": float(run.model_calls),
                        "latency_ms": float(elapsed),
                    },
                    detail={
                        "prompt": (text or "")[:120],
                        "stopped": run.stopped,
                        "tool_errors": run.tool_errors,
                        "parse_errors": run.parse_errors,
                        "tools_used": [s.tool_name for s in run.steps if s.kind == "tool_call"],
                    },
                )
            )
    except Exception as exc:  # 单轮失败不该让整批结果消失
        return TaskResult(cases=rows, error=f"Agent 批次中断：{exc}")
    finally:
        await client.aclose()

    n = len(rows) or 1
    metrics = {
        "completion_rate": round(completed / n, 4),
        "tool_call_rate": round(prompts_with_tools / n, 4),
        "avg_tool_calls": round(tool_calls_total / n, 3),
        "avg_steps": round(steps_total / n, 3),
    }
    ready = bool(client.ready)
    warnings: List[str] = []
    if not ready:
        warnings.append(
            "Agent 跑在离线替身上：模型不会发起工具调用，tool_call_rate=0 是替身行为而非真实能力，"
            "本 run 不得作为回归基线。"
        )
    return TaskResult(
        metrics=metrics,
        cases=rows,
        warnings=warnings,
        trusted=ready,
        fingerprint_detail={
            "provider": client.provider_name,
            "model": client.model,
            "protocol": cfg.agent.protocol,
            "max_steps": cfg.agent.max_steps,
            "strategy": cfg.rag.strategy,
            "prompts": len(prompts),
        },
    )


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #
TaskRunner = Callable[[TaskContext, str, Dict[str, Any]], Awaitable[TaskResult]]


@dataclass(frozen=True)
class TaskDef:
    info: TaskInfo
    runner: TaskRunner


TASKS: Dict[str, TaskDef] = {
    "retrieval": TaskDef(
        info=TaskInfo(
            name="retrieval",
            title="检索质量：Recall@k / MRR@10 / Hit@1",
            default_dataset="retrieval_eval",
            requires_llm=False,
            metrics=["recall@1", "recall@3", "recall@5", "mrr@10", "hit@1"],
            description="零 LLM、离线可跑；BM25 数字天然可信，向量策略需 semantic=true 才可信。",
        ),
        runner=run_retrieval,
    ),
    "generation": TaskDef(
        info=TaskInfo(
            name="generation",
            title="生成质量：LLM-as-judge 五维 + 与人工标注一致性",
            default_dataset="eval_set",
            requires_llm=True,
            metrics=["parse_rate", "mean_score", "kappa", "qwk"],
            description="生成与裁判必须是真实模型；同 provider 会产生自评偏差（报告里会警告）。",
        ),
        runner=run_generation,
    ),
    "agent": TaskDef(
        info=TaskInfo(
            name="agent",
            title="Agent 能力：工具调用率 / 完成率 / 平均步数",
            default_dataset="bench_questions",
            requires_llm=True,
            metrics=["completion_rate", "tool_call_rate", "avg_tool_calls", "avg_steps"],
            description="替身模型不会发起工具调用，此时 tool_call_rate=0 不代表真实能力。",
        ),
        runner=run_agent,
    ),
}


def list_tasks() -> List[TaskInfo]:
    return [t.info for t in TASKS.values()]


def median_or_zero(values: List[float]) -> float:
    return float(statistics.median(values)) if values else 0.0
