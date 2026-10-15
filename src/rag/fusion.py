# -*- coding: utf-8 -*-
"""倒数排名融合（Reciprocal Rank Fusion）。

RRF 的好处是**不需要归一化各路的分数**——BM25 的分数在 0~十几之间浮动，
余弦相似度在 -1~1 之间，直接相加毫无意义。只比较名次就避开了这个坑：

    score(d) = Σ 1 / (k + rank_i(d))

常数 k 的经验值取 60（原始论文），作用是压低头部文档的绝对优势，让尾部也有机会。
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from .bm25 import Hit

DEFAULT_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Hit]], k: int = DEFAULT_K
) -> List[Hit]:
    fused: Dict[str, float] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
    out = [Hit(cid, score) for cid, score in fused.items()]
    out.sort(key=lambda h: (-h.score, h.chunk_id))
    return out
