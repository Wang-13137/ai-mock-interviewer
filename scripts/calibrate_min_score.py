# -*- coding: utf-8 -*-
"""为稠密检索标定 min_score（相似度下限）。

为什么需要这个脚本：不同 embedding 模型的余弦分布差异极大，
HashingEmbedder 时代定下的 0.20 阈值换到 bge-small-zh 上完全不适用。
正确做法是**在标注评测集上扫阈值**，选指标最优且「相关/不相关分数间隔」最大的值。

用法（必须先装 sentence-transformers）：

    python scripts/calibrate_min_score.py
    python scripts/calibrate_min_score.py --candidates 0.30,0.35,0.40,0.45,0.50

输出：
- 每个候选阈值的 Recall@3 / MRR@10 / Hit@1（vector 策略）
- 相关命中 vs 不相关命中的分数分布（用于人工确认间隔）
- 推荐阈值，以及需要同步修改的位置（RagConfig.min_score / retriever.DEFAULT_MIN_SCORE）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.corpus import build_corpus, load_bank  # noqa: E402
from src.rag.evaluate import evaluate, load_cases  # noqa: E402
from src.rag.retriever import Retriever  # noqa: E402
from src.rag.vector import build_embedder, has_sentence_transformers  # noqa: E402

KS = (1, 3, 5)


def main() -> int:
    ap = argparse.ArgumentParser(description="标定稠密检索 min_score")
    ap.add_argument("--bank", default="config/question_bank.json")
    ap.add_argument("--cases", default="config/retrieval_eval.json")
    ap.add_argument(
        "--candidates",
        default="0.30,0.35,0.40,0.45,0.50,0.55,0.60",
        help="逗号分隔的候选阈值",
    )
    args = ap.parse_args()

    if not has_sentence_transformers():
        print("未安装 sentence-transformers，无法标定真实阈值。", file=sys.stderr)
        return 2

    bank = load_bank(args.bank)
    cases = load_cases(args.cases)
    corpus = build_corpus(bank)
    emb = build_embedder("sentence-transformers")
    print(f"embedder={emb.name} dim={emb.dim} | 语料 {len(corpus)} | 评测 {len(cases)} 条\n")

    rows = []
    for t in [float(x) for x in args.candidates.split(",")]:
        r = Retriever(corpus, strategy="vector", embedder=emb, min_score=t)
        rep = evaluate(r, cases, KS)
        rows.append((t, rep))
        sem = rep.get("by_kind", {}).get("semantic", {}).get("mrr@10", 0.0)
        print(
            f"  min_score={t:.2f}  Recall@3={rep['recall@3']:.3f} "
            f"MRR@10={rep['mrr@10']:.3f} Hit@1={rep['hit@1']:.3f} "
            f"（semantic 子集 MRR={sem:.3f}）"
        )

    # 分数间隔：每条 case 里相关命中的最低分 vs 不相关命中的最高分
    rel_scores, irrel_scores = [], []
    probe = Retriever(corpus, strategy="vector", embedder=emb, min_score=0.0)
    for c in cases:
        rel = set(c.relevant)
        for h in probe.search(c.query, top_k=10):
            (rel_scores if h.chunk.id in rel else irrel_scores).append(h.score)
    print(
        f"\n分数分布：相关命中 [{min(rel_scores):.3f}, {max(rel_scores):.3f}] "
        f"中位 {sorted(rel_scores)[len(rel_scores)//2]:.3f} | "
        f"不相关命中 [{min(irrel_scores):.3f}, {max(irrel_scores):.3f}] "
        f"中位 {sorted(irrel_scores)[len(irrel_scores)//2]:.3f}"
    )

    best = max(rows, key=lambda x: (x[1]["recall@3"], x[1]["mrr@10"]))
    print(f"\n推荐 min_score = {best[0]:.2f}")
    print("需同步修改：src/core/config.py RagConfig.min_score 与 src/rag/retriever.py DEFAULT_MIN_SCORE")
    print("⚠️ 本结论只在当前评测集与 bge-small-zh-v1.5 上成立；扩语料或换模型后必须重跑本脚本。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
