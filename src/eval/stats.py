# -*- coding: utf-8 -*-
"""一致性统计：把「自动评分 vs 人工评分」的差距算清楚。

面试时能讲出 Cohen's kappa 与二次加权 kappa 的区别，比空说「做了人工校准」有力得多：
- 评分是有序等级（1<2<3<4<5），相邻差一档和差四档不是同等严重 → 用 **QWK**；
- 想看「完全一致」的严格程度 → 用普通 Cohen's kappa / 准确率。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

Score = int


def _pairs(a: Sequence[Score], b: Sequence[Score]) -> List[Tuple[Score, Score]]:
    if len(a) != len(b):
        raise ValueError(f"两组评分长度不一致：{len(a)} vs {len(b)}")
    return [(x, y) for x, y in zip(a, b) if x is not None and y is not None]


def accuracy_within(a: Sequence[Score], b: Sequence[Score], tol: int = 0) -> float:
    """|x-y| <= tol 的比例。tol=0 即严格一致率，tol=1 是「差一档以内」。"""
    pairs = _pairs(a, b)
    if not pairs:
        return 0.0
    hit = sum(1 for x, y in pairs if abs(x - y) <= tol)
    return round(hit / len(pairs), 4)


def cohens_kappa(a: Sequence[Score], b: Sequence[Score]) -> Optional[float]:
    """普通 Cohen's kappa（把评分当名义变量，任何不一致都等权）。

    返回 None 表示分母退化（例如两组都是常数），此时该指标不可解释。
    """
    pairs = _pairs(a, b)
    if not pairs:
        return None
    n = len(pairs)
    po = sum(1 for x, y in pairs if x == y) / n

    labels = sorted({x for pair in pairs for x in pair})
    fa = {k: sum(1 for x, _ in pairs if x == k) / n for k in labels}
    fb = {k: sum(1 for _, y in pairs if y == k) / n for k in labels}
    pe = sum(fa[k] * fb[k] for k in labels)

    if abs(1 - pe) < 1e-12:
        return None
    return round((po - pe) / (1 - pe), 4)


def quadratic_weighted_kappa(
    a: Sequence[Score], b: Sequence[Score], min_score: int = 1, max_score: int = 5
) -> Optional[float]:
    """二次加权 kappa：不一致的惩罚随差距平方增长（1 vs 5 远比 1 vs 2 严重）。"""
    pairs = _pairs(a, b)
    if not pairs:
        return None
    n = len(pairs)
    lo, hi = min_score, max_score
    span = max(1, hi - lo)

    wa = [0.0] * (span + 1)
    wb = [0.0] * (span + 1)
    for x, y in pairs:
        wa[x - lo] += 1
        wb[y - lo] += 1
    observed = 0.0
    for x, y in pairs:
        observed += ((x - y) ** 2) / (span ** 2)

    total_pairs = n * n
    expected = 0.0
    for i in range(span + 1):
        for j in range(span + 1):
            expected += ((i - j) ** 2) / (span ** 2) * (wa[i] * wb[j]) / total_pairs

    if expected == 0:
        return None
    return round(1 - observed / expected, 4)


def agreement_report(human: Sequence[Score], auto: Sequence[Score]) -> Dict[str, Optional[float]]:
    """一次算齐：样本数、严格一致率、差一档一致率、kappa、QWK、均值差。"""
    pairs = _pairs(human, auto)
    if not pairs:
        return {"n": 0, "acc": None, "acc_within_1": None, "kappa": None,
                "qwk": None, "mean_diff": None}
    h = [x for x, _ in pairs]
    m = [y for _, y in pairs]
    return {
        "n": len(pairs),
        "acc": accuracy_within(h, m, tol=0),
        "acc_within_1": accuracy_within(h, m, tol=1),
        "kappa": cohens_kappa(h, m),
        "qwk": quadratic_weighted_kappa(h, m),
        "mean_diff": round(sum(m) / len(m) - sum(h) / len(h), 4),
    }


def interpret(kappa: Optional[float]) -> str:
    """Landis & Koch 的常用解释档位，写报告时直接用（别自己发明形容词）。"""
    if kappa is None:
        return "不可计算"
    if kappa < 0:
        return "差于随机"
    if kappa < 0.21:
        return "轻微一致 (slight)"
    if kappa < 0.41:
        return "一般一致 (fair)"
    if kappa < 0.61:
        return "中等一致 (moderate)"
    if kappa < 0.81:
        return "高度一致 (substantial)"
    return "几乎完全一致 (almost perfect)"
