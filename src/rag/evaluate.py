# -*- coding: utf-8 -*-
"""检索质量评测：Recall@k / MRR@10 / Hit@1。

这套指标的意义在于**同一份标注上的相对比较**，绝对值不可用——见「语料规模」警告。

为什么不用「准确率」：检索是排序问题，用户只看前几条。
Recall@3 下降 20% 可能完全不影响体感，而 MRR 掉一点就说明「第一条总不是最贴的」。
所以三个指标必须同时报，不能只挑好看的那个——这跟 W4 评测子集
（acc / acc±1 / kappa / QWK 同时报）是同一个原则。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .retriever import Retriever

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    relevant: List[str]
    note: str = ""
    # "literal"  = 字面改写，关键词方法也能命中
    # "semantic" = 语义改写，与原文几乎不共享词面，**只有语义检索能救**
    kind: str = "literal"


@dataclass(frozen=True)
class CaseResult:
    query: str
    relevant: List[str]
    retrieved: List[str]
    recall: Dict[int, float]
    reciprocal_rank: float
    hit_at_1: float
    kind: str = "literal"


def load_cases(path: str | Path = "config/retrieval_eval.json") -> List[RetrievalCase]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    data = json.loads(p.read_text(encoding="utf-8"))
    return [
        RetrievalCase(
            query=c["query"],
            relevant=list(c["relevant"]),
            note=c.get("note", ""),
            kind=c.get("kind", "literal") or "literal",
        )
        for c in data.get("cases", [])
    ]


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(set(list(retrieved)[:k]) & rel) / len(rel)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    rel = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in rel:
            return 1.0 / rank
    return 0.0


def run_case(
    retriever: Retriever, case: RetrievalCase, ks: Sequence[int] = (1, 3, 5), top_k: int = 5
) -> CaseResult:
    hits = retriever.search(case.query, top_k=top_k)
    ids = [h.chunk.id for h in hits]
    return CaseResult(
        query=case.query,
        relevant=case.relevant,
        retrieved=ids,
        recall={k: recall_at_k(ids, case.relevant, k) for k in ks},
        reciprocal_rank=reciprocal_rank(ids, case.relevant),
        hit_at_1=1.0 if ids and ids[0] in set(case.relevant) else 0.0,
        kind=getattr(case, "kind", "literal") or "literal",
    )


def evaluate(
    retriever: Retriever, cases: Sequence[RetrievalCase], ks: Sequence[int] = (1, 3, 5)
) -> Dict[str, Any]:
    results = [run_case(retriever, c, ks) for c in cases]
    n = len(results) or 1
    meta = retriever.describe()
    summary: Dict[str, Any] = {
        "queries": len(results),
        "documents": len(retriever.corpus),
        "strategy": retriever.strategy.value,
        "embedder": meta["embedder"],
        "semantic": meta["semantic_embedding"],
    }
    for k in ks:
        summary[f"recall@{k}"] = round(sum(r.recall[k] for r in results) / n, 4)
    summary["mrr@10"] = round(sum(r.reciprocal_rank for r in results) / n, 4)
    summary["hit@1"] = round(sum(r.hit_at_1 for r in results) / n, 4)

    # 按 query 难度分组：字面对齐组 vs 语义改写组
    summary["by_kind"] = {}
    for kind in ("literal", "semantic"):
        sub = [r for r in results if r.kind == kind]
        if not sub:
            continue
        summary["by_kind"][kind] = {
            "queries": len(sub),
            **{f"recall@{k}": round(sum(x.recall[k] for x in sub) / len(sub), 4) for k in ks},
            "mrr@10": round(sum(x.reciprocal_rank for x in sub) / len(sub), 4),
            "hit@1": round(sum(x.hit_at_1 for x in sub) / len(sub), 4),
        }

    summary["cases"] = [
        {
            "query": r.query,
            "relevant": r.relevant,
            "retrieved": r.retrieved,
            "recall": r.recall,
            "rr": round(r.reciprocal_rank, 3),
            "kind": r.kind,
        }
        for r in results
    ]
    return summary


def discriminative_power(reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """评测集是否有区分度。

    如果所有策略的 MRR 都接近满分，说明评测集**测不出东西**——用它选策略等于掷骰子。
    这和 W4 踩到的「kappa paradox」是同一类教训：**指标看起来完美，往往是因为它什么都测不到**。
    """
    mrrs = [r["mrr@10"] for r in reports]
    spread = round(max(mrrs) - min(mrrs), 4) if mrrs else 0.0
    semantic = None
    for r in reports:
        k = r.get("by_kind", {}).get("semantic")
        if k:
            semantic = k["mrr@10"]
            break
    return {
        "mrr_spread": spread,
        "saturated": max(mrrs) >= 0.999 if mrrs else False,
        "semantic_subset_mrr": semantic,
        "useless": spread < 0.02 and (max(mrrs) >= 0.95 if mrrs else False),
    }


def render_report(
    reports: Sequence[Dict[str, Any]], ks: Sequence[int] = (1, 3, 5), note: str = ""
) -> str:
    lines: List[str] = ["# 检索策略对比报告", ""]
    if note:
        lines += [f"> {note}", ""]

    header = "| 策略 | " + " | ".join(f"Recall@{k}" for k in ks) + " | MRR@10 | Hit@1 | 语义嵌入 |"
    lines += [header, "|" + "---|" * (len(ks) + 4)]
    for r in reports:
        rows = " | ".join(f"{r[f'recall@{k}']:.3f}" for k in ks)
        lines.append(
            f"| {r['strategy']} | {rows} | {r['mrr@10']:.3f} | {r['hit@1']:.3f} | {r['embedder'] or '—'} |"
        )
    lines += ["", f"评测集：{reports[0]['queries'] if reports else 0} 条 query × {reports[0]['documents'] if reports else 0} 条文档。", ""]

    # 分组对比：看得出来「字面题全对、语义题全崩」这种结构性差异
    kinds = [k for k in ("literal", "semantic") if any(k in r.get("by_kind", {}) for r in reports)]
    if kinds:
        lines += ["## 按 query 类型拆分", ""]
        lines += ["| 策略 | " + " | ".join(f"{k} MRR@10" for k in kinds) + " |", "|---|---|"]
        for r in reports:
            cells = []
            for k in kinds:
                sub = r.get("by_kind", {}).get(k)
                cells.append(f"{sub['mrr@10']:.3f} (n={sub['queries']})" if sub else "—")
            lines.append(f"| {r['strategy']} | " + " | ".join(cells) + " |")
        lines.append("")
        lines += [
            "`literal` = 字面改写（`...` 之外的同词替换），关键词方法通常能命中；",
            "`semantic` = 语义改写，与原题几乎不共享词面——**这一组的差距才是换 Vector/Hybrid 的理由**。",
            "",
        ]

    lines += ["## 逐条明细（第一个策略）", ""]
    lines += ["| query | 类型 | 相关 | Recall@3 | RR |", "|---|---|---|---|---|"]
    for c in reports[0]["cases"]:
        lines.append(
            f"| {c['query'][:24]} | {c['kind']} | {'、'.join(c['relevant'])} | "
            f"{c['recall'][3]:.2f} | {c['rr']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)
