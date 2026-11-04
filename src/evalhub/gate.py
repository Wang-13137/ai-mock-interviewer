# -*- coding: utf-8 -*-
"""回归门禁：拿本次 run 和基线 run 比，判断有没有退步。

为什么「评测」必须配「门禁」才有意义：
    跑分本身只是观测。真正能防住退步的是**比较**——改完检索参数、换了 chunk 策略、
    升级了 prompt 之后，得有个东西自动告诉你「MRR 掉了 3 个点」。
    没有门禁的评测体系，最后都会退化成"跑一次、看一眼、觉得还行"。

三道可比性检查（不过就直接 INCOMPARABLE，不硬比）：
    1. task 必须相同——检索指标和生成分数没有可比性；
    2. config_fingerprint 必须相同——换了模型/策略/阈值，涨跌都不归因于代码改动；
    3. dataset_hash 必须相同——**数据换了、指标当然会变**，这是最常见的假警报来源。
    外加一条：双方都必须 trusted。拿替身当基线，等于用假尺子量真东西。

阈值语义：
    `delta_pct` 是相对跌幅（(cur-base)/base）。绝对差值再配合阈值一起用，
    因为 baseline 很小（例如 0.05）时，相对跌幅会放大到失真。
"""

from __future__ import annotations

from typing import Dict, List

from .models import EvalRunRecord, GateResult, MetricDelta, Verdict

# 数值越低越好的指标：成本/性能类。它们退步的表现是**变大**。
LOWER_IS_BETTER_PREFIXES = ("latency", "avg_steps", "steps", "model_calls", "duration")

# 不参与门禁比较的指标：它们的涨跌没有「好坏」含义
IGNORED_METRICS = {"queries", "documents", "dataset_size", "prompts"}


def higher_is_better(metric: str) -> bool:
    return not metric.startswith(LOWER_IS_BETTER_PREFIXES)


def compare(
    baseline: EvalRunRecord,
    current: EvalRunRecord,
    threshold: float = 0.02,
) -> GateResult:
    """比较两个 run，产出 pass / regressed / incomparable。"""
    reasons: List[str] = []

    if baseline.task != current.task:
        return GateResult(
            verdict=Verdict.INCOMPARABLE,
            baseline_run=baseline.id,
            threshold=threshold,
            reasons=[f"任务不同：baseline={baseline.task} current={current.task}"],
        )

    if not baseline.trusted or not current.trusted:
        which = []
        if not baseline.trusted:
            which.append(f"基线 {baseline.id}")
        if not current.trusted:
            which.append(f"本次 {current.id}")
        return GateResult(
            verdict=Verdict.INCOMPARABLE,
            baseline_run=baseline.id,
            threshold=threshold,
            reasons=[f"{'、'.join(which)} 由替身产出（trusted=false），不能用作回归比较"],
        )

    if baseline.dataset_hash != current.dataset_hash:
        return GateResult(
            verdict=Verdict.INCOMPARABLE,
            baseline_run=baseline.id,
            threshold=threshold,
            reasons=[
                "评测数据版本不同，指标差异无法归因于代码改动",
                f"baseline hash={baseline.dataset_hash or '(空)'} current hash={current.dataset_hash or '(空)'}",
            ],
        )

    if baseline.config_fingerprint != current.config_fingerprint:
        return GateResult(
            verdict=Verdict.INCOMPARABLE,
            baseline_run=baseline.id,
            threshold=threshold,
            reasons=["配置指纹不同（换了模型/策略/参数），涨跌不可归因于代码改动"],
        )

    deltas: List[MetricDelta] = []
    for metric in sorted(set(baseline.metrics) & set(current.metrics)):
        if metric in IGNORED_METRICS:
            continue
        base = float(baseline.metrics[metric])
        cur = float(current.metrics[metric])
        delta = round(cur - base, 6)
        delta_pct = round(delta / abs(base), 6) if base else 0.0
        hib = higher_is_better(metric)

        # 退步判定：方向相反 + 幅度超阈值（绝对值与相对值任一超阈即算）
        worse = (delta < 0) if hib else (delta > 0)
        magnitude_ok = abs(delta) >= threshold or abs(delta_pct) >= threshold
        deltas.append(
            MetricDelta(
                metric=metric,
                baseline=base,
                current=cur,
                delta=delta,
                delta_pct=delta_pct,
                higher_is_better=hib,
                regressed=bool(worse and magnitude_ok),
            )
        )

    regressed = [d for d in deltas if d.regressed]
    if regressed:
        for d in regressed:
            reasons.append(
                f"{d.metric} 从 {d.baseline} 变为 {d.current}（{d.delta_pct:+.1%}），超出阈值 {threshold:.1%}"
            )
        return GateResult(
            verdict=Verdict.REGRESSED,
            baseline_run=baseline.id,
            threshold=threshold,
            deltas=deltas,
            reasons=reasons,
        )

    if not deltas:
        reasons.append("两个 run 没有共同的数值指标，无法比较")
        return GateResult(
            verdict=Verdict.INCOMPARABLE,
            baseline_run=baseline.id,
            threshold=threshold,
            reasons=reasons,
        )

    return GateResult(verdict=Verdict.PASS, baseline_run=baseline.id, threshold=threshold, deltas=deltas, reasons=reasons)


def render_gate(gate: GateResult) -> str:
    """把门禁结论渲染成一行行给人看的表格。"""
    lines = [f"门禁结论：{gate.verdict.value}（基线 {gate.baseline_run or '—'}，阈值 {gate.threshold:.1%}）"]
    for reason in gate.reasons:
        lines.append(f"  · {reason}")
    if gate.deltas:
        lines.append("")
        lines.append(f"  {'指标':<20} {'基线':>8} {'本次':>8} {'变化':>9}  方向")
        for d in gate.deltas:
            flag = "↓退步" if d.regressed else ""
            arrow = "越高越好" if d.higher_is_better else "越低越好"
            lines.append(
                f"  {d.metric:<20} {d.baseline:>8.4f} {d.current:>8.4f} {d.delta_pct:>8.1%}  {arrow} {flag}"
            )
    return "\n".join(lines)
