# -*- coding: utf-8 -*-
"""配对 bootstrap：把「指标掉了 2 个点」升级成「这 2 个点是不是噪声」。

为什么门禁必须做这一步：
    60 条 query 的评测集，Recall 的标准误约 sqrt(0.8*0.2/60) ≈ 0.05。
    也就是说 **一次 run 的 0.02 波动完全在噪声范围内**。
    只拿点估计比阈值，结果只有两种：要么阈值设得松到抓不住真退步，
    要么设得紧到天天误报——误报几次之后，团队就会把门禁关掉，
    这比没有门禁更糟（因为它给了一种"我们有回归保护"的错觉）。

    所以判"退步"要同时满足三件事：
        1. 方向错了（越高越好的指标变低）；
        2. 幅度够大（绝对差 AND 相对差都超阈值）；
        3. **统计上分得开**（Δ 的 95% 置信区间不跨 0）。
    第 3 条只能靠重采样得到，因为指标分布未知、样本量小、不满足正态假设。

为什么用配对（paired）而不是两独立样本：
    两次 run 跑的是**同一批 case**，逐条配对后做差，
    把"这条 query 本身就难"的方差消掉了，检出力显著更高。

为什么要有还原校验（reconcile）：
    逐条明细里的字段名和汇总指标名并不总是一一对应（`mrr@10` ← `rr`，
    `hit@1` ← `rr>=1`）。映射一旦写错，CI 会算得很漂亮但完全是错的。
    所以先验证「明细重建出来的均值 ≈ 落库的汇总指标」，对不上就**放弃检验**，
    退回点估计并明确写进 reasons。宁可说"这次没法判显著"，也不给假结论。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .models import CaseRow

DEFAULT_ITERS = 2000
DEFAULT_SEED = 42
DEFAULT_ALPHA = 0.05
# 少于这个样本量，bootstrap 的区间没有意义（重采样只会在原样本里打转）
MIN_CASES = 8
# 明细重建均值与汇总指标的最大允许偏差；超过即认为映射不成立
RECONCILE_TOL = 0.02


@dataclass(frozen=True)
class BootstrapResult:
    """一个指标的配对 bootstrap 结果。"""

    metric: str
    n: int                 # 参与重采样的配对样本数
    delta: float           # 重采样 Δ 的均值（≈ 点估计差）
    low: float             # 95% CI 下界
    high: float            # 95% CI 上界
    iters: int
    seed: int

    @property
    def significant(self) -> bool:
        """区间是否整体偏离 0。"""
        return self.low > 0.0 or self.high < 0.0

    def span(self) -> str:
        return f"[{self.low:+.4f}, {self.high:+.4f}]"


# --------------------------------------------------------------------------- #
# 明细 → 指标重建
# --------------------------------------------------------------------------- #
def _split_kind(metric: str) -> Tuple[Optional[str], str]:
    """`semantic_mrr@10` → ("semantic", "mrr@10")。分组指标只在该子集上重采样。"""
    for kind in ("literal", "semantic"):
        prefix = f"{kind}_"
        if metric.startswith(prefix):
            return kind, metric[len(prefix):]
    return None, metric


def _extract(metric: str, row: CaseRow) -> Optional[float]:
    """从一条明细里取出该指标的值。取不到就返回 None（不猜、不补 0）。"""
    m = row.metrics or {}
    d = row.detail or {}

    if metric in m:
        return float(m[metric])

    if metric in ("mrr@10", "mrr"):
        # 单条 query 的 reciprocal rank 取均值就是 MRR
        if "rr" in m:
            return float(m["rr"])
        return None

    if metric == "hit@1":
        # top1 命中 ⇔ rr == 1
        if "rr" in m:
            return 1.0 if float(m["rr"]) >= 1.0 else 0.0
        return None

    if metric.startswith("recall@"):
        try:
            k = int(metric.split("@", 1)[1])
        except (IndexError, ValueError):
            return None
        rel, got = d.get("relevant"), d.get("retrieved")
        if not rel or got is None:
            return None
        rels = {str(x) for x in rel}
        if not rels:
            return None
        topk = [str(x) for x in list(got)[:k]]
        return len(rels & set(topk)) / len(rels)

    # 生成任务
    if metric == "mean_score":
        return float(m["auto_score"]) if "auto_score" in m else None
    if metric == "parse_rate":
        v = d.get("parse_ok")
        return None if v is None else (1.0 if v else 0.0)

    # Agent 任务
    if metric == "completion_rate":
        v = d.get("stopped")
        return None if v is None else (1.0 if v == "final" else 0.0)
    if metric == "tool_call_rate":
        v = m.get("tool_calls")
        return None if v is None else (1.0 if float(v) > 0 else 0.0)
    if metric in ("avg_tool_calls", "avg_steps"):
        key = "tool_calls" if metric == "avg_tool_calls" else "steps"
        return float(m[key]) if key in m else None

    return None


def reconstruct(metric: str, rows: Sequence[CaseRow]) -> Optional[Dict[str, float]]:
    """把逐条明细还原成 `{case_id: value}`。任一 case 取不到就整体放弃（None）。"""
    kind, base_metric = _split_kind(metric)
    out: Dict[str, float] = {}
    for row in rows:
        if kind is not None and str((row.detail or {}).get("kind")) != kind:
            continue
        value = _extract(base_metric, row)
        if value is None:
            return None
        out[str(row.case_id)] = value
    return out or None


def mean_of(series: Dict[str, float]) -> float:
    return sum(series.values()) / len(series) if series else 0.0


# --------------------------------------------------------------------------- #
# 配对 bootstrap
# --------------------------------------------------------------------------- #
def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return float(sorted_values[idx])


def paired_bootstrap(
    base: Dict[str, float],
    cur: Dict[str, float],
    *,
    metric: str = "",
    iters: int = DEFAULT_ITERS,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> Optional[BootstrapResult]:
    """对同一批 case 的两次结果做配对重采样，返回 Δ 的 percentile CI。

    返回 None 表示样本不足或两组无法配对——调用方应退回点估计判定。
    """
    common = sorted(set(base) & set(cur))
    if iters <= 0:  # 配置成 0 = 显式关闭显著性检验，退回点估计
        return None
    if len(common) < MIN_CASES:
        return None
    # 配对率太低说明两次 run 的 case 集合不同源，比出来的差异无法归因
    if len(common) < 0.8 * max(len(base), len(cur)):
        return None

    pairs = [(base[cid], cur[cid]) for cid in common]
    n = len(pairs)
    rng = random.Random(seed)
    deltas: List[float] = []
    for _ in range(iters):
        total = 0.0
        for _ in range(n):
            b, c = pairs[rng.randrange(n)]
            total += c - b
        deltas.append(total / n)
    deltas.sort()

    return BootstrapResult(
        metric=metric,
        n=n,
        delta=round(sum(deltas) / len(deltas), 6),
        low=round(_quantile(deltas, alpha / 2), 6),
        high=round(_quantile(deltas, 1 - alpha / 2), 6),
        iters=iters,
        seed=seed,
    )


def bootstrap_metric(
    metric: str,
    base_rows: Sequence[CaseRow],
    cur_rows: Sequence[CaseRow],
    base_value: float,
    cur_value: float,
    *,
    iters: int = DEFAULT_ITERS,
    seed: int = DEFAULT_SEED,
    tol: float = RECONCILE_TOL,
) -> Tuple[Optional[BootstrapResult], str]:
    """一步到位：重建 + 校验 + 重采样。第二个返回值是"为什么没做"的原因（成功时为空）。"""
    base_series = reconstruct(metric, base_rows)
    cur_series = reconstruct(metric, cur_rows)
    if not base_series or not cur_series:
        return None, f"{metric} 无法从逐条明细重建（该指标不是逐条均值/比例），跳过显著性检验"

    for label, series, reported in (
        ("基线", base_series, base_value),
        ("本次", cur_series, cur_value),
    ):
        if abs(mean_of(series) - reported) > tol:
            return (
                None,
                f"{metric} 的明细重建均值 {mean_of(series):.4f} 与落库指标 {reported:.4f} 不符"
                f"（{label}），明细与汇总不同源，拒绝据此计算置信区间",
            )

    ci = paired_bootstrap(base_series, cur_series, metric=metric, iters=iters, seed=seed)
    if ci is None:
        return None, f"{metric} 可配对样本不足（基线 {len(base_series)} / 本次 {len(cur_series)}），不足以做显著性判断"
    return ci, ""
