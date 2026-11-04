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

退步判定（v0.8 起，三个条件同时满足才算）：
    1. 方向错了；
    2. 幅度够大：**绝对差 ≥ abs_threshold 且 相对差 ≥ pct_threshold**；
    3. 统计上分得开：Δ 的 95% CI 不跨 0（有逐条明细时做配对 bootstrap）。

    为什么第 2 条从「或」改成「且」：旧实现里一个 threshold 兼任两种单位，
    baseline=0.04 时绝对差 0.004 只有 1 条 case 翻转，相对却是 10%，
    于是被判成退步——**小 baseline 的相对放大恰恰是噪声最响的地方**。
    拆成两个阈值并要求同时满足，方向才对。

    为什么第 3 条必须有：60 条样本的 Recall 标准误约 0.05，0.02 的波动本来就在
    噪声范围内。没有显著性检验的门禁只有两种结局——太松抓不住，太紧天天误报，
    误报几次之后团队就会把它关掉，这比没有门禁更糟（它给了虚假的安全感）。

    **判 pass 时也必须在 reasons 里交代**"哪个指标跌了、为什么判成噪声"，
    否则使用者无法区分"真的没退步"和"门禁没看见"。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .bootstrap import DEFAULT_ITERS, DEFAULT_SEED, bootstrap_metric
from .models import CaseRow, EvalRunRecord, GateResult, MetricDelta, Verdict

# 数值越低越好的指标：成本/性能类。它们退步的表现是**变大**。
LOWER_IS_BETTER_PREFIXES = ("latency", "avg_steps", "steps", "model_calls", "duration")

# 不参与门禁比较的指标：它们的涨跌没有「好坏」含义
IGNORED_METRICS = {"queries", "documents", "dataset_size", "prompts"}

# 无逐条明细时无法做显著性检验，此时退回点估计——但要明说，不能假装验过
_NO_DETAIL_NOTE = "未提供逐条明细，仅按点估计判定（无显著性检验）"


def higher_is_better(metric: str) -> bool:
    return not metric.startswith(LOWER_IS_BETTER_PREFIXES)


def _incomparable(baseline: EvalRunRecord, reasons: List[str], *, abs_threshold: float, pct: float) -> GateResult:
    return GateResult(
        verdict=Verdict.INCOMPARABLE,
        baseline_run=baseline.id,
        threshold=pct,
        abs_threshold=abs_threshold,
        reasons=reasons,
    )


def compare(
    baseline: EvalRunRecord,
    current: EvalRunRecord,
    threshold: Optional[float] = None,
    *,
    abs_threshold: float = 0.02,
    pct_threshold: Optional[float] = None,
    base_cases: Optional[Sequence[CaseRow]] = None,
    cur_cases: Optional[Sequence[CaseRow]] = None,
    iters: int = DEFAULT_ITERS,
    seed: int = DEFAULT_SEED,
) -> GateResult:
    """比较两个 run，产出 pass / regressed / incomparable。

    `threshold` 是历史参数，等同 `pct_threshold`（相对阈值），保留以避免旧调用方静默改变语义。
    """
    pct = pct_threshold if pct_threshold is not None else (threshold if threshold is not None else 0.02)

    if baseline.task != current.task:
        return _incomparable(
            baseline,
            [f"任务不同：baseline={baseline.task} current={current.task}"],
            abs_threshold=abs_threshold,
            pct=pct,
        )

    if not baseline.trusted or not current.trusted:
        which = []
        if not baseline.trusted:
            which.append(f"基线 {baseline.id}")
        if not current.trusted:
            which.append(f"本次 {current.id}")
        return _incomparable(
            baseline,
            [f"{'、'.join(which)} 由替身产出（trusted=false），不能用作回归比较"],
            abs_threshold=abs_threshold,
            pct=pct,
        )

    if baseline.dataset_hash != current.dataset_hash:
        return _incomparable(
            baseline,
            [
                "评测数据版本不同，指标差异无法归因于代码改动",
                f"baseline hash={baseline.dataset_hash or '(空)'} current hash={current.dataset_hash or '(空)'}",
            ],
            abs_threshold=abs_threshold,
            pct=pct,
        )

    if baseline.config_fingerprint != current.config_fingerprint:
        return _incomparable(
            baseline,
            ["配置指纹不同（换了模型/策略/参数），涨跌不可归因于代码改动"],
            abs_threshold=abs_threshold,
            pct=pct,
        )

    deltas: List[MetricDelta] = []
    noisy: List[str] = []
    unverified: List[str] = []

    for metric in sorted(set(baseline.metrics) & set(current.metrics)):
        if metric in IGNORED_METRICS:
            continue
        base_v = float(baseline.metrics[metric])
        cur_v = float(current.metrics[metric])
        delta = round(cur_v - base_v, 6)
        delta_pct = round(delta / abs(base_v), 6) if base_v else 0.0
        hib = higher_is_better(metric)

        # ---- 第 3 条：显著性 ----
        ci = None
        if base_cases is not None and cur_cases is not None:
            ci, why_not = bootstrap_metric(
                metric, base_cases, cur_cases, base_v, cur_v, iters=iters, seed=seed
            )
            if ci is None and why_not:
                unverified.append(why_not)

        if ci is not None:
            significant: Optional[bool] = (ci.high < 0.0) if hib else (ci.low > 0.0)
        else:
            significant = None  # 无法检验，不假装

        # ---- 第 1、2 条：方向 + 幅度（绝对与相对都要超）----
        worse = (delta < 0) if hib else (delta > 0)
        magnitude_ok = abs(delta) >= abs_threshold and abs(delta_pct) >= pct
        regressed = bool(worse and magnitude_ok and (significant is not False))

        deltas.append(
            MetricDelta(
                metric=metric,
                baseline=base_v,
                current=cur_v,
                delta=delta,
                delta_pct=delta_pct,
                higher_is_better=hib,
                regressed=regressed,
                ci_low=ci.low if ci else None,
                ci_high=ci.high if ci else None,
                sample=ci.n if ci else 0,
                method="bootstrap" if ci else "point",
                significant=significant,
            )
        )

        # 跌了但判为噪声：必须留痕，否则使用者分不清"没退步"和"没看见"
        if worse and magnitude_ok and significant is False:
            noisy.append(
                f"{metric} 点估计 {delta_pct:+.1%}，但 95% CI {ci.span() if ci else '—'} 跨 0"
                f"（n={ci.n if ci else 0}），统计上无法与噪声区分，判为未退步"
            )

    regressed_deltas = [d for d in deltas if d.regressed]
    reasons: List[str] = []
    if regressed_deltas:
        for d in regressed_deltas:
            ci_note = f"，95% CI {d.render_ci()} 整体偏离 0" if d.ci_low is not None else "（点估计判定）"
            reasons.append(
                f"{d.metric} 从 {d.baseline} 变为 {d.current}（{d.delta_pct:+.1%}）{ci_note}，"
                f"超出阈值（绝对 {abs_threshold:.3f} / 相对 {pct:.1%}）"
            )
        reasons.extend(noisy)
        if not any(d.method == "bootstrap" for d in deltas):
            reasons.append(_NO_DETAIL_NOTE)
        return GateResult(
            verdict=Verdict.REGRESSED,
            baseline_run=baseline.id,
            threshold=pct,
            abs_threshold=abs_threshold,
            deltas=deltas,
            reasons=reasons,
        )

    if not deltas:
        return _incomparable(
            baseline,
            ["两个 run 没有共同的数值指标，无法比较"],
            abs_threshold=abs_threshold,
            pct=pct,
        )

    ci_used = any(d.method == "bootstrap" for d in deltas)
    reasons.append(
        f"全部 {len(deltas)} 个指标均未同时满足「方向变差 + 绝对差≥{abs_threshold:.3f} + "
        f"相对差≥{pct:.1%}" + (" + CI 显著」" if ci_used else "」")
    )
    reasons.extend(noisy)
    if unverified:
        reasons.append("以下指标未做显著性检验：" + "；".join(unverified[:3]) + ("…" if len(unverified) > 3 else ""))
    elif not ci_used:
        reasons.append(_NO_DETAIL_NOTE)

    return GateResult(
        verdict=Verdict.PASS,
        baseline_run=baseline.id,
        threshold=pct,
        abs_threshold=abs_threshold,
        deltas=deltas,
        reasons=reasons,
    )


def render_gate(gate: GateResult) -> str:
    """把门禁结论渲染成一行行给人看的表格。"""
    lines = [
        f"门禁结论：{gate.verdict.value}"
        f"（基线 {gate.baseline_run or '—'}，"
        f"阈值 绝对 {gate.abs_threshold:.3f} / 相对 {gate.threshold:.1%}）"
    ]
    for reason in gate.reasons:
        lines.append(f"  · {reason}")
    if gate.deltas:
        lines.append("")
        lines.append(f"  {'指标':<20} {'基线':>8} {'本次':>8} {'变化':>9}  {'95% CI':<22} 方向")
        for d in gate.deltas:
            flag = "↓退步" if d.regressed else ""
            arrow = "越高越好" if d.higher_is_better else "越低越好"
            lines.append(
                f"  {d.metric:<20} {d.baseline:>8.4f} {d.current:>8.4f} {d.delta_pct:>8.1%}  "
                f"{d.render_ci():<22} {arrow} {flag}"
            )
    return "\n".join(lines)
