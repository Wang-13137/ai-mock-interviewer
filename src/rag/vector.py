# -*- coding: utf-8 -*-
"""向量检索后端。

⚠️ **红线（务必读懂再往下看）**

`HashingEmbedder` 是**管道替身，不是语义 embedding**。
它做的是 hashing trick：把词袋投影到固定维度并做符号哈希。语义相近但字面不同的词
（比如「跑批」和「批处理」）在它眼里毫无关系。

它的存在只有一个目的：让「嵌入 → 归一化 → 余弦 top-k → 融合」这条链路在**没有
torch / transformers / faiss 的机器上**也能跑起来、也能被测试覆盖。

因此：**用 HashingEmbedder 跑出来的召回率一律不得出现在简历、面经或 README 的效果表**。
真实语义检索请装 `sentence-transformers` 后用 `SentenceTransformerEmbedder`
（计划模型：BAAI/bge-small-zh-v1.5，约 90MB）。两者的接口完全一致，切换只改配置。
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import math
from typing import Dict, List, Optional, Protocol, Sequence, runtime_checkable

from .bm25 import Hit
from .corpus import Chunk, Corpus
from .tokenize import tokenize

log = logging.getLogger("aimi.rag.vector")


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int
    semantic: bool  # False 表示这是替身，不得用于效果报告

    def embed(self, texts: Sequence[str]) -> List[List[float]]: ...


class HashingEmbedder:
    """零依赖的词袋 hashing 嵌入。**替身**，见模块 docstring 的红线。"""

    name = "hashing-trick"
    semantic = False

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        toks = tokenize(text)
        for _rep in range(1):  # 保留结构：未来可加 bigram 二次哈希增强
            for t in toks:
                h = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16)
                idx = h % self.dim
                vec[idx] += 1.0 if (h >> 31) & 1 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm:
            vec = [v / norm for v in vec]
        return vec

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]


class SentenceTransformerEmbedder:
    """真实语义嵌入。依赖 `sentence-transformers`，缺失时给出可执行的安装提示。"""

    semantic = True

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        if importlib.util.find_spec("sentence_transformers") is None:
            raise RuntimeError(
                "未安装 sentence-transformers。请先执行：\n"
                "  pip install sentence-transformers\n"
                "首次运行会自动下载模型权重（bge-small-zh 约 90MB）。"
            )
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        out = self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        return [list(map(float, row)) for row in out]


def has_sentence_transformers() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


class VectorIndex:
    """暴力余弦检索。

    30 条语料下暴力搜索比引入 FAISS 更快也更简单。
    **这是刻意的取舍，不是偷懒**：语料上万条后再换 ANN 索引，接口不需要变。
    """

    def __init__(
        self, corpus: Corpus, embedder: Embedder, min_score: float = 0.0
    ) -> None:
        if not len(corpus):
            raise ValueError("语料为空，无法建立索引")
        self.embedder = embedder
        self.min_score = min_score
        texts = [" ".join([c.question, c.dimension, " ".join(c.tags)]) for c in corpus.chunks]
        self.vectors: Dict[str, List[float]] = dict(
            zip([c.id for c in corpus.chunks], embedder.embed(texts))
        )

    def search(
        self,
        corpus: Corpus,
        query: str,
        top_k: int = 3,
        boost_dimensions: Sequence[str] = (),
        boost_factor: float = 1.3,
        min_score: Optional[float] = None,
    ) -> List[Hit]:
        score_threshold = self.min_score if min_score is None else min_score
        q = self.embedder.embed([query])[0]
        boosts = set(boost_dimensions or ())
        hits: List[Hit] = []
        for chunk in corpus.chunks:
            v = self.vectors.get(chunk.id)
            if v is None:
                continue
            s = sum(a * b for a, b in zip(q, v))
            # 余弦相似度几乎不会正好等于 0：再不相干的文档也有零点几的分数。
            # 没有这个阈值，「无命中」这个概念在稠密检索里根本不存在，
            # Agent 会被一堆垃圾召回带偏。（BM25 天然有 >0 语义，不需要这一步。）
            if s < score_threshold:
                continue
            if chunk.dimension in boosts:
                s *= boost_factor
            hits.append(Hit(chunk.id, s))
        hits.sort(key=lambda h: (-h.score, h.chunk_id))
        return hits[: max(1, top_k)]


def build_embedder(name: str = "auto", model: str = "BAAI/bge-small-zh-v1.5") -> Embedder:
    """按名字构建 embedder。`auto`：有 sentence-transformers 就用，否则退化为替身并记录警告。"""
    if name == "hashing":
        return HashingEmbedder()
    if name in ("auto", "sentence-transformers", "bge"):
        if has_sentence_transformers():
            return SentenceTransformerEmbedder(model)
        if name != "auto":
            raise RuntimeError("要求使用 sentence-transformers，但本机未安装")
        log.warning("未检测到 sentence-transformers，退回 HashingEmbedder（替身，不得用于效果报告）")
        return HashingEmbedder()
    raise ValueError(f"未知 embedder: {name}")
