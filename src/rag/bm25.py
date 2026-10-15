# -*- coding: utf-8 -*-
"""Okapi BM25，零依赖实现。

一个刻意的工程选择：IDF 用 **非负变体**

    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

而不是教科书里的 `ln((N - df + 0.5) / (df + 0.5))`。
原因：在只有几十条文档的**小语料**里，当一个词出现在过半文档时（比如「项目」），
教科书 IDF 会变成负数，于是「包含这个词」反而扣分，
结果是**越通用的词越能压制文档得分**，排序在语义上说不通。
Lucene 与多数生产 BM25 实现也都这么处理。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .corpus import Chunk, Corpus


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float

    def __repr__(self) -> str:  # pragma: no cover - 调试便利
        return f"Hit({self.chunk_id}, {self.score:.4f})"


class BM25Index:
    def __init__(self, corpus: Corpus, k1: float = 1.5, b: float = 0.75) -> None:
        if not len(corpus):
            raise ValueError("语料为空，无法建立索引")
        self.k1 = k1
        self.b = b
        self.doc_len: Dict[str, int] = {}
        self.tf: Dict[str, Counter] = {}
        total = 0
        for c in corpus.chunks:
            self.doc_len[c.id] = len(c.tokens)
            self.tf[c.id] = Counter(c.tokens)
            total += len(c.tokens)
        self.avgdl = total / len(corpus.chunks)
        self.n = len(corpus.chunks)

        df: Counter = Counter()
        for c in corpus.chunks:
            df.update(set(c.tokens))
        self.idf: Dict[str, float] = {
            t: math.log(1 + (self.n - d + 0.5) / (d + 0.5)) for t, d in df.items()
        }

    def score(self, chunk: Chunk, query_tokens: Sequence[str]) -> float:
        """标准 Okapi BM25：`idf * (f*(k1+1)) / (f + k1*(1-b+b*dl/avgdl))`。"""
        tf = self.tf.get(chunk.id)
        if tf is None:
            return 0.0
        dl = self.doc_len.get(chunk.id, 0)
        norm = self.k1 * (1 - self.b + self.b * (dl / self.avgdl if self.avgdl else 1.0))
        total = 0.0
        for term in set(query_tokens):
            f = tf.get(term, 0)
            idf = self.idf.get(term)
            if not f or idf is None:
                continue
            total += idf * (f * (self.k1 + 1)) / (f + norm)
        return total

    def search(
        self,
        corpus: Corpus,
        query_tokens: Sequence[str],
        top_k: int = 3,
        boost_dimensions: Sequence[str] = (),
        boost_factor: float = 1.3,
    ) -> List[Hit]:
        hits: List[Hit] = []
        boosts = set(boost_dimensions or ())
        for chunk in corpus.chunks:
            s = self.score(chunk, query_tokens)
            if s <= 0:
                continue
            if chunk.dimension in boosts:
                s *= boost_factor
            hits.append(Hit(chunk.id, s))
        hits.sort(key=lambda h: (-h.score, h.chunk_id))
        return hits[: max(1, top_k)]
