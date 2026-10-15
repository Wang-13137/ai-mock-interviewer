# -*- coding: utf-8 -*-
"""语料构造：把题库条目变成可检索的文档。

字段加权用最朴素的办法实现——**重复词项放大 TF**，而不是给 BM25 打补丁：

    tags 词 x3   dimension 词 x2   question 词 x1

好处是打分函数保持标准 BM25，不需要额外维护一套加权分支；
坏处是文档长度会被拉长（BM25 的长度归一化会部分抵消这个增益），
这一点可以从 ablation 里看出来，正好是能讲的东西。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .tokenize import tokenize

REPO_ROOT = Path(__file__).resolve().parents[2]

TAG_WEIGHT = 3
DIM_WEIGHT = 2
QUESTION_WEIGHT = 1


@dataclass(frozen=True)
class Chunk:
    """一条可检索单元。"""

    id: str
    question: str
    dimension: str
    difficulty: int
    tags: List[str] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)


class Corpus:
    def __init__(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        self.by_id: Dict[str, Chunk] = {c.id: c for c in chunks}

    def __len__(self) -> int:
        return len(self.chunks)

    def empty_message(self) -> str:
        return "题库为空（未加载配置文件）。"


def _weighted_tokens(item: Dict[str, Any]) -> List[str]:
    tags = " ".join(item.get("tags", []) or [])
    dimension = item.get("dimension", "") or ""
    question = item.get("question", "") or ""

    tokens: List[str] = []
    tokens += tokenize(tags) * TAG_WEIGHT
    tokens += tokenize(dimension) * DIM_WEIGHT
    tokens += tokenize(question) * QUESTION_WEIGHT
    return tokens


def build_corpus(bank: Dict[str, Any]) -> Corpus:
    """从题库 JSON（`{"items": [...]}`）构建语料。"""
    chunks: List[Chunk] = []
    for item in bank.get("items", []) or []:
        if not item.get("id") or not item.get("question"):
            continue
        chunks.append(
            Chunk(
                id=str(item["id"]),
                question=str(item["question"]),
                dimension=str(item.get("dimension", "未分类") or "未分类"),
                difficulty=int(item.get("difficulty", 2) or 2),
                tags=list(item.get("tags", []) or []),
                tokens=_weighted_tokens(item),
            )
        )
    return Corpus(chunks)


def load_bank(path: str | Path = "config/question_bank.json") -> Dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        return {"version": "empty", "items": []}
    return json.loads(p.read_text(encoding="utf-8"))
