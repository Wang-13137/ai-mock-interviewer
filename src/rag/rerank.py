# -*- coding: utf-8 -*-
"""cross-encoder 重排。

为什么检索后面还要一级重排：bi-encoder（bge-small-zh）把 query 和文档**分别**编码成向量，
点乘打分——快，但 query 与文档之间没有 token 级交互。cross-encoder 把 (query, 文档) **拼在一起**
过一遍 Transformer，直接输出相关性分数——慢一个数量级，但准。

工程上的标准做法是两段式：bi-encoder/BM25 先粗召回 top-N（N≈10–20），
cross-encoder 只对这 N 条精排。只对 top-N 打分是关键——全语料过 cross-encoder 的开销
没人付得起，这也是「召回管全、重排管准」的分工。

⚠️ 与 embedder 同样的红线：`CrossEncoderReranker` 需要 sentence-transformers 与本地权重。
测试里用 `NoopReranker` / 假实现替身，**替身跑出的指标不得对外引用**。
"""

from __future__ import annotations

import importlib.util
import logging
from typing import List, Protocol, Sequence, runtime_checkable

from .bm25 import Hit
from .corpus import Corpus
from .vector import _resolve_model_ref, has_sentence_transformers

log = logging.getLogger("aimi.rag.rerank")


@runtime_checkable
class Reranker(Protocol):
    name: str
    semantic: bool  # False = 替身

    def rerank(self, query: str, corpus: Corpus, hits: Sequence[Hit]) -> List[Hit]:
        """对粗召回的 hits 重打分并返回新顺序。分数域与粗排不同（logits），仅供排序。"""
        ...


class NoopReranker:
    """直通：什么都不做。用于「关重排」与单测基线。"""

    name = "noop"
    semantic = False

    def rerank(self, query: str, corpus: Corpus, hits: Sequence[Hit]) -> List[Hit]:
        return list(hits)


class CrossEncoderReranker:
    """BAAI/bge-reranker-base（XLM-R base，1.1GB）。缺失依赖时给出安装提示。"""

    semantic = True

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        if not has_sentence_transformers():
            raise RuntimeError(
                "未安装 sentence-transformers。请先执行：\n"
                "  pip install sentence-transformers\n"
                "并把模型权重放到 models/bge-reranker-base/（本机走 ModelScope 下载）。"
            )
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        path = _resolve_model_ref(model_name)
        self.name = model_name
        self._model = CrossEncoder(path)

    def rerank(self, query: str, corpus: Corpus, hits: Sequence[Hit]) -> List[Hit]:
        if not hits:
            return []
        pairs = [
            (query, " ".join([corpus.by_id[h.chunk_id].question,
                              corpus.by_id[h.chunk_id].dimension,
                              " ".join(corpus.by_id[h.chunk_id].tags)]))
            for h in hits
            if h.chunk_id in corpus.by_id
        ]
        kept = [h for h in hits if h.chunk_id in corpus.by_id]
        scores = self._model.predict(pairs)
        out = [Hit(h.chunk_id, float(s)) for h, s in zip(kept, scores)]
        out.sort(key=lambda h: (-h.score, h.chunk_id))
        return out


def build_reranker(name: str = "none", model: str = "BAAI/bge-reranker-base") -> Reranker:
    """`none`：不重排；`cross-encoder`/`bge`：真实模型，缺库时报错（不静默退化——
    重排是加分项不是必需品，调用方应显式选择，退化了还以为自己开着反而更糟）。"""
    if name in ("none", "off", ""):
        return NoopReranker()
    if name in ("cross-encoder", "bge"):
        if importlib.util.find_spec("sentence_transformers") is None:
            raise RuntimeError("要求 cross-encoder 重排，但本机未安装 sentence-transformers")
        return CrossEncoderReranker(model)
    raise ValueError(f"未知 reranker: {name}")
