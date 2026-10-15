# -*- coding: utf-8 -*-
"""统一检索入口。

三条策略在同一份语料上对比：

| strategy | 组成 | 什么时候用 |
|---|---|---|
| `bm25` | 纯关键词 | **默认**。零依赖、可解释，小语料下的强基线 |
| `vector` | 稠密向量余弦 | 装了 sentence-transformers 之后；擅长同义替换、字面不匹配 |
| `hybrid` | BM25 + 向量，RRF 融合 | 两者都要。实践中收益最大，但要付双倍建索引成本 |

`boost_dimensions` 是面试场景特有的一步：把**还没考察到的维度**加权，
让 Agent 主动往覆盖面走，而不是沿着当前话题越钻越深。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from .bm25 import BM25Index, Hit
from .corpus import Chunk, Corpus, build_corpus
from .fusion import reciprocal_rank_fusion
from .rerank import NoopReranker, Reranker
from .tokenize import tokenize
from .vector import Embedder, VectorIndex, build_embedder, has_sentence_transformers

log = logging.getLogger("aimi.rag.retriever")

BOOST_FACTOR = 1.3

# 稠密检索的相似度下限。用 bge-small-zh-v1.5 在 80 语料 × 60 条评测集上标定（2026-10-16 二轮）：
# 平台区 0.30–0.40，取上沿 0.40；0.45 起缓慢下滑，0.60 崩。
# **换模型、扩语料或扩评测集后必须重跑 scripts/calibrate_min_score.py 重新标定**——
# 不同模型/数据分布下照搬阈值会直接掐掉一半召回。
DEFAULT_MIN_SCORE = 0.40


class Strategy(str, Enum):
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class Retrieved:
    chunk: Chunk
    score: float
    strategy: str


class Retriever:
    def __init__(
        self,
        corpus: Corpus,
        strategy: Strategy | str = Strategy.BM25,
        embedder: Optional[Embedder] = None,
        min_score: Optional[float] = None,
        reranker: Optional[Reranker] = None,
    ) -> None:
        if not len(corpus):
            raise ValueError("语料为空，无法建立检索器")
        self.corpus = corpus
        self.strategy = Strategy(strategy)
        self.bm25 = BM25Index(corpus)
        self.vector: Optional[VectorIndex] = None
        # cross-encoder 只对粗召回的候选池（top_k*3）精排，开销可控
        self.reranker: Reranker = reranker or NoopReranker()

        # 稠密检索必须配相似度阈值（理由见 VectorIndex.search）。
        # 0.20 是先用替身 embedding 定下的起点，**换真 embedding 后必须用评测集重新标定**。
        if min_score is None:
            min_score = 0.0 if self.strategy == Strategy.BM25 else DEFAULT_MIN_SCORE
        self.min_score = min_score

        if self.strategy in (Strategy.VECTOR, Strategy.HYBRID):
            emb = embedder or build_embedder()
            self.vector = VectorIndex(corpus, emb, min_score=self.min_score)
            self.embedder: Optional[Embedder] = emb
        else:
            self.embedder = None

    # ---------------- 工厂 ---------------- #
    @classmethod
    def from_bank(
        cls,
        bank: Dict[str, Any],
        strategy: Strategy | str = Strategy.BM25,
        embedder_name: str = "auto",
        model_name: str = "BAAI/bge-small-zh-v1.5",
        min_score: Optional[float] = None,
        reranker_name: str = "none",
        reranker_model: str = "BAAI/bge-reranker-base",
    ) -> "Retriever":
        from .rerank import build_reranker  # 延迟导入，避免循环

        corpus = build_corpus(bank)
        emb = (
            build_embedder(embedder_name, model_name)
            if Strategy(strategy) != Strategy.BM25
            else None
        )
        reranker = build_reranker(reranker_name, reranker_model)
        return cls(corpus, strategy=strategy, embedder=emb, min_score=min_score, reranker=reranker)

    # ---------------- 检索 ---------------- #
    def search(
        self,
        query: str,
        top_k: int = 3,
        boost_dimensions: Sequence[str] = (),
    ) -> List[Retrieved]:
        hits = self._rank(query, top_k * 3, boost_dimensions)
        if self.reranker.name != "noop":
            hits = self.reranker.rerank(query, self.corpus, hits)
        hits = hits[: max(1, top_k)]
        return [
            Retrieved(chunk=self.corpus.by_id[h.chunk_id], score=h.score, strategy=self.strategy.value)
            for h in hits
            if h.chunk_id in self.corpus.by_id
        ]

    def _rank(
        self, query: str, top_k: int, boost_dimensions: Sequence[str]
    ) -> List[Hit]:
        if self.strategy == Strategy.BM25:
            return self.bm25.search(
                self.corpus, tokenize(query), top_k, boost_dimensions, BOOST_FACTOR
            )

        if self.strategy == Strategy.VECTOR and self.vector is not None:
            return self.vector.search(self.corpus, query, top_k, boost_dimensions, BOOST_FACTOR)

        if self.strategy == Strategy.HYBRID and self.vector is not None:
            sparse = self.bm25.search(
                self.corpus, tokenize(query), top_k, boost_dimensions, BOOST_FACTOR
            )
            dense = self.vector.search(self.corpus, query, top_k, boost_dimensions, BOOST_FACTOR)
            return reciprocal_rank_fusion([sparse, dense])[:top_k]

        raise RuntimeError(f"检索器 {self.strategy} 未正确初始化")

    # ---------------- 元信息 ---------------- #
    def describe(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "documents": len(self.corpus),
            "semantic_embedding": bool(self.vector and getattr(self.embedder, "semantic", False)),
            "embedder": getattr(self.embedder, "name", None),
            "reranker": self.reranker.name,
            "semantic_reranker": bool(getattr(self.reranker, "semantic", False)),
        }


def available_backends() -> Dict[str, bool]:
    return {"bm25": True, "sentence-transformers": has_sentence_transformers()}
