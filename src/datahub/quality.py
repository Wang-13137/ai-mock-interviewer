# -*- coding: utf-8 -*-
"""数据质量门禁：登记数据集时必须过这一关。

设计原则：**校验规则要能抓到真实发生过的错，而不是凑数。**

这套规则来自本项目实际踩过的坑，每条都对应一个具体后果：
    · 评测集指向不存在的题号 → 检索指标会悄悄变低（召回永远拿不到那篇），但不报错；
      2026-10 扩题库（30→80 条）时就差点出现这个问题，靠人肉比对才发现。
    · 同一条 query 标了两次 → 该 query 在均值里被双倍计权，指标被少数样本绑架。
    · 用 tags 自动生成 relevant → 答案泄露进标注，Recall 虚高（retrieval_eval.json 的 note 里记着这条教训）。
    · 对话用例最后一句不是 user → 生成侧拿不到"当前提问"，评测结果无意义。
    · 某评测维度 0 条 → 那一维的均分是空的，但报告里看起来像"全维度都测了"。

分级：
    error   数据不可用，拒绝入库（引用断裂、必填缺失、主键重复）
    warning 可以入库，但必须能解释（近似重复、维度缺口、人工标注未覆盖）
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .models import DatasetKind, QualityIssue, QualityReport

# 与 src/eval/dataset.py 的 Dimension 保持一致（这里用字符串避免循环导入）
EVAL_DIMENSIONS = ("追问合理性", "上下文一致性", "事实可靠性", "面试风格符合度", "应拒答场景")
RETRIEVAL_KINDS = ("literal", "semantic")

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[\s，,。.、；;：:！!？?（）()【】\[\]「」\"'`]+")


def normalize(text: str) -> str:
    """归一化：去空白与常见标点，用于「近似重复」判定。

    不做同义词/向量相似度——那会把判断变成不可解释的黑盒；
    质检结论要能被人工复核，所以用最朴素的可复现规则。
    """
    return _PUNCT.sub("", _WS.sub("", str(text or ""))).lower()


# --------------------------------------------------------------------------- #
# 各 kind 的校验器：返回 (errors, warnings, stats)
# --------------------------------------------------------------------------- #
def check_question_bank(data: Any, ctx: Dict[str, Any]) -> Tuple[List[QualityIssue], List[QualityIssue], Dict[str, Any]]:
    errors: List[QualityIssue] = []
    warnings: List[QualityIssue] = []
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        errors.append(QualityIssue(level="error", code="empty_dataset", message="题库为空或结构不是列表"))
        return errors, warnings, {"rows": 0}

    ids: List[str] = []
    dims: Counter = Counter()
    diffs: Counter = Counter()
    seen_norm: Dict[str, str] = {}
    q_lens: List[int] = []

    for i, item in enumerate(items):
        where = f"items[{i}]"
        if not isinstance(item, dict):
            errors.append(QualityIssue(level="error", code="bad_item_type", message="条目不是对象", where=where))
            continue
        qid = str(item.get("id") or "").strip()
        question = str(item.get("question") or "").strip()
        if not qid:
            errors.append(QualityIssue(level="error", code="missing_id", message="缺少 id", where=where))
        else:
            ids.append(qid)
        if not question:
            errors.append(QualityIssue(level="error", code="empty_question", message="题干为空", where=qid or where))
        else:
            q_lens.append(len(question))
            key = normalize(question)
            if key in seen_norm and key:
                warnings.append(
                    QualityIssue(
                        level="warning",
                        code="duplicate_question",
                        message=f"题干与 {seen_norm[key]} 近似重复（归一化后相同）",
                        where=qid or where,
                    )
                )
            elif key:
                seen_norm[key] = qid or where
        dim = str(item.get("dimension") or "").strip()
        if not dim:
            warnings.append(QualityIssue(level="warning", code="missing_dimension", message="缺少 dimension，检索语料里会归到「未分类」", where=qid or where))
        else:
            dims[dim] += 1
        diff = item.get("difficulty")
        if not isinstance(diff, int) or not (1 <= diff <= 5):
            warnings.append(QualityIssue(level="warning", code="bad_difficulty", message=f"difficulty={diff!r} 不在 1..5，检索语料会按 2 处理", where=qid or where))
        else:
            diffs[diff] += 1
        if not item.get("tags"):
            warnings.append(QualityIssue(level="warning", code="missing_tags", message="tags 为空，BM25 字段加权会退化为纯题干", where=qid or where))

    dup_ids = [i for i, c in Counter(ids).items() if c > 1]
    if dup_ids:
        errors.append(
            QualityIssue(
                level="error",
                code="duplicate_id",
                message=f"{len(dup_ids)} 个 id 重复：{dup_ids[:5]}",
                where="items",
            )
        )

    stats = {
        "rows": len(items),
        "unique_ids": len(set(ids)),
        "dimensions": dict(dims),
        "difficulty": {str(k): v for k, v in sorted(diffs.items())},
        "question_len_avg": round(sum(q_lens) / len(q_lens), 1) if q_lens else 0,
        "question_len_min": min(q_lens) if q_lens else 0,
        "question_len_max": max(q_lens) if q_lens else 0,
    }
    return errors, warnings, stats


def check_retrieval_eval(
    data: Any, ctx: Dict[str, Any]
) -> Tuple[List[QualityIssue], List[QualityIssue], Dict[str, Any]]:
    errors: List[QualityIssue] = []
    warnings: List[QualityIssue] = []
    cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(cases, list) or not cases:
        errors.append(QualityIssue(level="error", code="empty_dataset", message="检索评测集为空"))
        return errors, warnings, {"rows": 0}

    ref_ids: Optional[set] = ctx.get("ref_ids")  # 依赖数据集（题库）的合法 id 集合
    covered: set = set()
    kinds: Counter = Counter()
    rel_counts: List[int] = []
    seen_query: Dict[str, int] = {}
    dangling: List[str] = []

    for i, case in enumerate(cases):
        where = f"cases[{i}]"
        if not isinstance(case, dict):
            errors.append(QualityIssue(level="error", code="bad_case_type", message="用例不是对象", where=where))
            continue
        query = str(case.get("query") or "").strip()
        if not query:
            errors.append(QualityIssue(level="error", code="empty_query", message="query 为空", where=where))
            continue
        key = normalize(query)
        if key in seen_query:
            errors.append(
                QualityIssue(
                    level="error",
                    code="duplicate_query",
                    message=f"与 cases[{seen_query[key]}] 的 query 重复——同一条会被双倍计权，均值失真",
                    where=where,
                )
            )
        else:
            seen_query[key] = i

        relevant = case.get("relevant")
        if not isinstance(relevant, list) or not relevant:
            errors.append(QualityIssue(level="error", code="empty_relevant", message="relevant 为空，该 query 无法计算召回", where=where))
        else:
            rel_counts.append(len(relevant))
            covered.update(str(r) for r in relevant)
            if ref_ids is not None:
                for rid in relevant:
                    if str(rid) not in ref_ids:
                        dangling.append(f"{where}:{rid}")
        kind = case.get("kind", "literal") or "literal"
        if kind not in RETRIEVAL_KINDS:
            errors.append(QualityIssue(level="error", code="bad_kind", message=f"kind={kind!r} 不在 {RETRIEVAL_KINDS}", where=where))
        else:
            kinds[kind] += 1

    # 引用完整性：这是中台存在的最硬理由
    if dangling:
        errors.append(
            QualityIssue(
                level="error",
                code="dangling_reference",
                message=(
                    f"{len(dangling)} 处 relevant 指向题库里不存在的 id（前 5 个：{dangling[:5]}）。"
                    "这类断裂不会让评测报错，只会让指标静默变低——必须先修数据。"
                ),
                where="cases[].relevant",
            )
        )

    if not kinds.get("semantic"):
        warnings.append(
            QualityIssue(level="warning", code="no_semantic_subset", message="没有 semantic 子集，无法证明向量检索比 BM25 强")
        )
    if ref_ids:
        coverage = len(covered & ref_ids) / max(len(ref_ids), 1)
        if coverage < 0.5:
            warnings.append(
                QualityIssue(
                    level="warning",
                    code="low_corpus_coverage",
                    message=f"只覆盖了 {coverage:.0%} 的题库条目，作为全库检索评测代表性不足",
                )
            )

    stats = {
        "rows": len(cases),
        "kinds": dict(kinds),
        "avg_relevant": round(sum(rel_counts) / len(rel_counts), 2) if rel_counts else 0,
        "corpus_ids": len(ref_ids) if ref_ids else None,
        "corpus_coverage": round(len(covered & ref_ids) / len(ref_ids), 3) if ref_ids else None,
        "dangling_refs": len(dangling),
    }
    return errors, warnings, stats


def check_eval_set(
    data: Any, ctx: Dict[str, Any]
) -> Tuple[List[QualityIssue], List[QualityIssue], Dict[str, Any]]:
    errors: List[QualityIssue] = []
    warnings: List[QualityIssue] = []
    cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(cases, list) or not cases:
        errors.append(QualityIssue(level="error", code="empty_dataset", message="对话评测集为空"))
        return errors, warnings, {"rows": 0}

    ids: List[str] = []
    dims: Counter = Counter()
    refusals = 0
    scored = 0
    turn_counts: List[int] = []

    for i, case in enumerate(cases):
        where = f"cases[{i}]"
        if not isinstance(case, dict):
            errors.append(QualityIssue(level="error", code="bad_case_type", message="用例不是对象", where=where))
            continue
        cid = str(case.get("id") or "").strip()
        if not cid:
            errors.append(QualityIssue(level="error", code="missing_id", message="缺少 id", where=where))
        else:
            ids.append(cid)
        dim = case.get("dimension")
        if dim not in EVAL_DIMENSIONS:
            errors.append(
                QualityIssue(level="error", code="bad_dimension", message=f"dimension={dim!r} 不在五维内", where=cid or where)
            )
        else:
            dims[str(dim)] += 1
        dialogue = case.get("dialogue")
        if not isinstance(dialogue, list) or not dialogue:
            errors.append(QualityIssue(level="error", code="empty_dialogue", message="dialogue 为空", where=cid or where))
        else:
            turn_counts.append(len(dialogue))
            last_role = (dialogue[-1] or {}).get("role") if isinstance(dialogue[-1], dict) else None
            if last_role != "user":
                errors.append(
                    QualityIssue(
                        level="error",
                        code="last_turn_not_user",
                        message="dialogue 最后一轮不是 user——生成侧拿不到当前提问",
                        where=cid or where,
                    )
                )
        if case.get("expects_refusal"):
            refusals += 1
            if not str(case.get("reference") or "").strip():
                warnings.append(
                    QualityIssue(level="warning", code="refusal_without_reference", message="expects_refusal=True 但没有 reference，拒答判分无依据", where=cid or where)
                )
        if case.get("human_score") is not None:
            scored += 1

    dup_ids = [i for i, c in Counter(ids).items() if c > 1]
    if dup_ids:
        errors.append(QualityIssue(level="error", code="duplicate_id", message=f"id 重复：{dup_ids[:5]}", where="cases"))

    missing_dims = [d for d in EVAL_DIMENSIONS if not dims.get(d)]
    if missing_dims:
        warnings.append(
            QualityIssue(
                level="warning",
                code="dimension_gap",
                message=f"{len(missing_dims)} 个维度没有用例：{missing_dims}——报告里那一维会是空的，别当成「已覆盖」",
            )
        )
    if scored < len(cases):
        warnings.append(
            QualityIssue(
                level="warning",
                code="human_labels_incomplete",
                message=f"人工标注覆盖 {scored}/{len(cases)} 条；n<30 时 kappa/QWK 不可解读（kappa paradox）",
            )
        )

    stats = {
        "rows": len(cases),
        "dimensions": dict(dims),
        "refusal_cases": refusals,
        "human_scored": scored,
        "avg_turns": round(sum(turn_counts) / len(turn_counts), 2) if turn_counts else 0,
        "rubric_version": (data.get("rubric_version") if isinstance(data, dict) else None),
    }
    return errors, warnings, stats


def check_bench(
    data: Any, ctx: Dict[str, Any]
) -> Tuple[List[QualityIssue], List[QualityIssue], Dict[str, Any]]:
    errors: List[QualityIssue] = []
    warnings: List[QualityIssue] = []
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        errors.append(QualityIssue(level="error", code="empty_dataset", message="bench 题目为空"))
        return errors, warnings, {"rows": 0}

    ids: List[str] = []
    for i, item in enumerate(items):
        where = f"items[{i}]"
        if not isinstance(item, dict):
            errors.append(QualityIssue(level="error", code="bad_item_type", message="条目不是对象", where=where))
            continue
        iid = str(item.get("id") or "").strip()
        text = str(item.get("text") or "").strip()
        if not iid:
            errors.append(QualityIssue(level="error", code="missing_id", message="缺少 id", where=where))
        else:
            ids.append(iid)
        if not text:
            errors.append(QualityIssue(level="error", code="empty_text", message="text 为空", where=iid or where))
    dup = [i for i, c in Counter(ids).items() if c > 1]
    if dup:
        errors.append(QualityIssue(level="error", code="duplicate_id", message=f"id 重复：{dup[:5]}", where="items"))
    return errors, warnings, {"rows": len(items), "unique_ids": len(set(ids))}


CHECKERS = {
    DatasetKind.QUESTION_BANK: check_question_bank,
    DatasetKind.RETRIEVAL_EVAL: check_retrieval_eval,
    DatasetKind.EVAL_SET: check_eval_set,
    DatasetKind.BENCH: check_bench,
}


def check(
    dataset_id: str,
    kind: DatasetKind,
    content_hash: str,
    data: Any,
    ctx: Optional[Dict[str, Any]] = None,
) -> QualityReport:
    """按 kind 分发校验，产出可直接存档的报告。"""
    checker = CHECKERS.get(kind)
    if checker is None:  # pragma: no cover - DatasetKind 是枚举，理论上到不了
        errors = [QualityIssue(level="error", code="unknown_kind", message=f"未登记的 kind: {kind}")]
        return QualityReport(dataset_id=dataset_id, content_hash=content_hash, errors=errors)

    errors, warnings, stats = checker(data, ctx or {})
    return QualityReport(
        dataset_id=dataset_id,
        content_hash=content_hash,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


def extract_ids(kind: DatasetKind, data: Any) -> List[str]:
    """取出可用于引用完整性的主键集合（目前只有题库被引用）。"""
    if kind is not DatasetKind.QUESTION_BANK:
        return []
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    return [str(i.get("id")) for i in items if isinstance(i, dict) and i.get("id")]
