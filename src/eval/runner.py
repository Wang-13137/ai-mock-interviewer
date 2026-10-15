# -*- coding: utf-8 -*-
"""跑一批评测：生成回复 -> 另一个模型打分 -> 与人工标注对齐。"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..app import build_client
from ..clients.llm import BaseLLMClient
from ..core.config import Settings
from ..prompts.interviewer import build_messages
from .dataset import EvalCase, EvalDataset
from .judge import JudgeResult, judge_one
from .stats import agreement_report, interpret

log = logging.getLogger("aimi.eval")

DEFAULT_PROFILE = {"position": "AI 应用工程师", "level": "校招应届", "style": "专业严谨"}


@dataclass
class EvalOutcome:
    case_id: str
    dimension: str
    reply: str
    judge: JudgeResult
    human_score: Optional[int] = None


@dataclass
class EvalReport:
    outcomes: List[EvalOutcome] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---------------- 统计 ---------------- #
    @property
    def scored(self) -> List[EvalOutcome]:
        return [o for o in self.outcomes if o.judge.score is not None]

    @property
    def parse_rate(self) -> float:
        """裁判输出可解析率：低于 95% 就说明 rubric 或 prompt 要改。"""
        if not self.outcomes:
            return 0.0
        return round(sum(1 for o in self.outcomes if o.judge.parse_ok) / len(self.outcomes), 4)

    def mean_score_by_dimension(self) -> Dict[str, float]:
        buckets: Dict[str, List[int]] = {}
        for o in self.scored:
            buckets.setdefault(o.dimension, []).append(o.judge.score)  # type: ignore[arg-type]
        return {k: round(sum(v) / len(v), 3) for k, v in sorted(buckets.items())}

    def agreement(self) -> Dict[str, Optional[float]]:
        human = [o.human_score for o in self.scored if o.human_score is not None]
        auto = [o.judge.score for o in self.scored if o.human_score is not None]
        if not human:
            return {"n": 0, "acc": None, "acc_within_1": None, "kappa": None, "qwk": None, "mean_diff": None}
        report = agreement_report(human, auto)
        report["acc_within_1"] = report.pop("acc_within_1", None)
        return report

    # ---------------- 输出 ---------------- #
    def to_rows(self) -> List[Dict[str, Any]]:
        return [
            {
                "case_id": o.case_id,
                "dimension": o.dimension,
                "human_score": o.human_score,
                "auto_score": o.judge.score,
                "reason": o.judge.reason,
                "parse_ok": o.judge.parse_ok,
                "error": o.judge.error,
                "reply_chars": len(o.reply),
            }
            for o in self.outcomes
        ]

    def render_md(self) -> str:
        agree = self.agreement()
        lines = [
            "# 评测报告",
            "",
            f"- 数据集：{self.meta.get('dataset')}（{self.meta.get('size')} 条，rubric {self.meta.get('rubric_version')}）",
            f"- 生成模型：`{self.meta.get('generator')}` ｜ 裁判模型：`{self.meta.get('judge')}`",
            f"- 裁判输出可解析率：**{self.parse_rate:.1%}**（<95% 说明 rubric/prompt 需要改）",
            f"- 样本 n={self.meta.get('n')}",
            "",
        ]
        for w in self.warnings:
            lines.append(f"> [!WARNING] {w}")
        if self.warnings:
            lines.append("")

        means = self.mean_score_by_dimension()
        if means:
            lines += ["| 维度 | 平均分 | 条数 |", "| --- | --- | --- |"]
            counts: Dict[str, int] = {}
            for o in self.scored:
                counts[o.dimension] = counts.get(o.dimension, 0) + 1
            for dim, mean in means.items():
                lines.append(f"| {dim} | {mean} | {counts[dim]} |")
            lines.append("")

        if agree.get("n"):
            lines += [
                "## 与人工标注的一致性",
                "",
                "| 指标 | 值 | 解读 |",
                "| --- | --- | --- |",
                f"| 严格一致率 acc | {agree['acc']} | — |",
                f"| 差一档以内 acc±1 | {agree['acc_within_1']} | — |",
                f"| Cohen's kappa | {agree['kappa']} | {interpret(agree['kappa'])} |",
                f"| 二次加权 kappa (QWK) | {agree['qwk']} | {interpret(agree['qwk'])} |",
                f"| 自动-人工 均值差 | {agree['mean_diff']} | 正值=自动评分偏高 |",
                "",
                "> 评分是有序等级，所以以 **QWK** 为主指标；kappa 偏低时应先改 rubric 而不是改标注。",
                "",
            ]
        else:
            lines += ["尚未做人工标注（case 里 `human_score` 全为空），抽 ≥30 条标完再谈一致性。", ""]
        return "\n".join(lines)


async def run_eval(
    settings: Settings,
    dataset: EvalDataset,
    *,
    limit: Optional[int] = None,
    profile: Optional[Dict[str, Any]] = None,
    concurrency: int = 4,
) -> EvalReport:
    generator: BaseLLMClient = build_client(settings, role="chat")
    judge_client: BaseLLMClient = build_client(settings, role="judge")
    profile = profile or DEFAULT_PROFILE

    report = EvalReport(meta={
        "dataset": dataset.name,
        "size": dataset.size,
        "rubric_version": dataset.rubric_version,
        "generator": f"{generator.provider_name}/{generator.model}",
        "judge": f"{judge_client.provider_name}/{judge_client.model}",
        "ts": datetime.now().isoformat(timespec="seconds"),
    })

    if generator.provider_name == judge_client.provider_name and generator.provider_name != "mock":
        report.warnings.append(
            f"裁判与生成使用了同一个 provider（{generator.provider_name}），存在**自评偏差**；"
            "建议换成不同厂商/不同模型（例如生成用 GLM、裁判用 DeepSeek）。"
        )
    if not judge_client.ready:
        report.warnings.append(
            "裁判跑在离线替身 EchoClient 上，分数无意义——这只是链路验证，不得写进任何材料。"
        )

    cases: List[EvalCase] = dataset.cases[:limit] if limit else dataset.cases
    sem = asyncio.Semaphore(concurrency)

    async def one(case: EvalCase) -> EvalOutcome:
        hist, query = case.history_and_query
        messages = build_messages(hist + [{"role": "user", "content": query}], profile)
        async with sem:
            try:
                res = await generator.achat(messages)
                reply = res.content
            except Exception as exc:
                reply = ""
                log.error("生成失败 %s: %s", case.id, exc)
            outcome = EvalOutcome(
                case_id=case.id,
                dimension=case.dimension.value,
                reply=reply,
                judge=JudgeResult(case_id=case.id, dimension=case.dimension.value),
                human_score=case.human_score,
            )
            outcome.judge = await judge_one(judge_client, case, reply)
            log.info("[%s] %s score=%s %s", case.id, case.dimension.value,
                     outcome.judge.score, outcome.judge.reason[:30])
            return outcome

    report.outcomes = list(await asyncio.gather(*[one(c) for c in cases]))
    report.meta["n"] = len(report.outcomes)
    await asyncio.gather(generator.aclose(), judge_client.aclose())
    return report


def write_report(report: EvalReport, out_dir: str | Path = "reports") -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = out / f"eval_{ts}"
    md = stem.with_suffix(".md")
    js = stem.with_suffix(".json")
    md.write_text(report.render_md(), encoding="utf-8")
    js.write_text(json.dumps({"meta": report.meta, "warnings": report.warnings,
                              "rows": report.to_rows()}, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    return md, js
