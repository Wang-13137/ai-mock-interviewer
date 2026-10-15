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
from .tokenize import tokenize
from .vector import Embedder, VectorIndex, build_embedder, has_sentence_transformers

log = logging.getLogger("aimi.rag.retriever")

BOOST_FACTOR = 1.3

# 稠密检索的相似度下限。**这个值是用 HashingEmbedder 定的起点，
# 换成真实语义模型（bge-small-zh）后必须用评测集重新标定**——
# 不同模型的余弦分布差异极大，照搬阈值会直接掐掉一半召回。
DEFAULT_MIN_SCORE = 0.20


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
    ) -> None:
        if not len(corpus):
            raise ValueError("语料为空，无法建立检索器")
        self.corpus = corpus
        self.strategy = Strategy(strategy)
        self.bm25 = BM25Index(corpus)
        self.vector: Optional[VectorIndex] = None

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
    ) -> "Retriever":
        corpus = build_corpus(bank)
        emb = build_embedder(embedder_name) if Strategy(strategy) != Strategy.BM25 else None
        return cls(corpus, strategy=strategy, embedder=emb)

    # ---------------- 检索 ---------------- #
    def search(
        self,
        query: str,
        top_k: int = 3,
        boost_dimensions: Sequence[str] = (),
    ) -> List[Retrieved]:
        hits = self._rank(query, top_k * 3, boost_dimensions)
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
        }


def available_backends() -> Dict[str, bool]:
    return {"bm25": True, "sentence-transformers": has_sentence_transformers()}
