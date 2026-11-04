# -*- coding: utf-8 -*-
"""中台冒烟：数据资产同步 -> 跑检索评测 -> 与基线比较。

用途有两个：
    1. 验证中台端到端可用（不依赖任何 API Key，BM25 全离线）；
    2. 面试现场可以当场跑给面试官看「评测是怎么被管起来的」。

跑法：
    python scripts/smoke_platform.py
    python scripts/smoke_platform.py --task generation   # 需要真实 Key 才有意义
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_settings  # noqa: E402
from src.datahub import DataHub  # noqa: E402
from src.evalhub import EvalHub, EvalRunRequest, EvalTask  # noqa: E402
from src.evalhub.bootstrap import mean_of, reconstruct  # noqa: E402
from src.evalhub.gate import compare, render_gate  # noqa: E402
from src.evalhub.models import CaseRow, EvalRunRecord  # noqa: E402


def show_datasets(hub: DataHub) -> None:
    print("\n== 数据中台：数据资产 ==")
    for rec in hub.list_datasets():
        latest = rec.latest
        flag = "⚠️ 未同步" if latest is None else ("🔸 dirty" if rec.dirty else "✅")
        print(
            f"  {rec.spec.id:<16} {flag:<10} versions={rec.versions} "
            f"rows={latest.rows if latest else 0:<4} hash={(latest.content_hash if latest else '-'):<18} "
            f"quality={(latest.quality_status if latest else '-'):<8} 被引用={rec.referenced_by or '无'}"
        )
        if rec.quality and (rec.quality.errors or rec.quality.warnings):
            for e in rec.quality.errors:
                print(f"      [error]   {e.code}: {e.message}")
            for w in rec.quality.warnings[:3]:
                print(f"      [warning] {w.code}: {w.message}")

    print("\n== 数据中台：血缘 ==")
    graph = hub.lineage()
    if not graph.edges:
        print("  （无引用关系）")
    for edge in graph.edges:
        print(f"  {edge.from_dataset} --{edge.field}--> {edge.to_dataset}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="数据中台 / 评测中台冒烟")
    parser.add_argument("--task", default="retrieval", choices=["retrieval", "generation", "agent"])
    parser.add_argument("--baseline", default="latest", help="基线 run id，或 latest / none")
    args = parser.parse_args()

    settings = load_settings()
    datahub = DataHub(settings)
    for result in datahub.sync_all():
        print(f"  sync {result.dataset_id:<16} -> {result.status:<11} {result.message}")
    show_datasets(datahub)

    hub = EvalHub(settings, datahub)
    print(f"\n== 评测中台：可用任务 ==")
    for info in hub.tasks():
        llm = "需要真实模型" if info.requires_llm else "零 LLM 可离线"
        print(f"  {info.name:<12} {info.title}（{llm}）")

    print(f"\n== 第 1 次 run（无基线） ==")
    first = await hub.run(EvalRunRequest(task=EvalTask(args.task)))
    print(f"  {first.summary_line()}")
    print(f"  metrics: {first.metrics}")
    if first.warnings:
        for w in first.warnings:
            print(f"  ⚠️  {w}")

    baseline = None if args.baseline == "none" else (args.baseline if args.baseline != "latest" else first.id)
    print(f"\n== 第 2 次 run（baseline={baseline}） ==")
    second = await hub.run(EvalRunRequest(task=EvalTask(args.task), baseline_run=baseline))
    print(f"  {second.summary_line()}")
    if second.gate:
        print()
        print(render_gate(second.gate))

    print(f"\n历史 run：{len(hub.list_runs(limit=20))} 条")
    for rec in hub.list_runs(limit=5):
        print(f"  - {rec.summary_line()}")

    # ---- 第 3 段：注入式劣化对照 ----
    # 门禁最大的风险是"从没报过警"，于是没人知道它到底会不会报警。
    # 这一段故意把逐条明细劣化 30%，看门禁是否真的判 regressed 并给出 CI。
    # **注入数据不可外引**，它只用来验证判定逻辑，不是模型效果。
    if second.gate and args.task == "retrieval":
        print("\n== 第 3 段：注入劣化 30%，验证门禁真的能抓到退步 ==")
        print("  （注入式对照：数字不可外引，仅用于证明门禁会报警）")
        base_cases = hub.get_cases(second.id)
        degraded = []
        for i, row in enumerate(base_cases):
            m = dict(row.metrics)
            if i % 10 < 3:  # 每 10 条打坏 3 条
                m["recall@3"] = 0.0
                m["rr"] = 0.0
            degraded.append(CaseRow(case_id=row.case_id, metrics=m, detail=dict(row.detail)))

        metrics = {}
        for name, value in second.metrics.items():
            series = reconstruct(name, degraded)
            metrics[name] = round(mean_of(series), 6) if series else value

        simulated = EvalRunRecord(
            id=f"{second.id}-sim",
            task=second.task,
            metrics=metrics,
            trusted=True,
            config_fingerprint=second.config_fingerprint,
            dataset_hash=second.dataset_hash,
            code_version="simulated",
        )
        gate = compare(
            second,
            simulated,
            abs_threshold=settings.evalhub.gate_abs_threshold,
            pct_threshold=settings.evalhub.gate_threshold,
            base_cases=base_cases,
            cur_cases=degraded,
            iters=settings.evalhub.bootstrap_iters,
        )
        print()
        print(render_gate(gate))
        print(f"\n  结论：门禁判定 = {gate.verdict.value}（期望 regressed）")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
