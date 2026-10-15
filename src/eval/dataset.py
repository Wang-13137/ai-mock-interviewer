# -*- coding: utf-8 -*-
"""评测数据集的数据契约。

W4 的产出物：`config/eval_set.json`（目标 100 条）。
与 `medical-rag` 的评测集是同一套方法论，但**用例必须完全不同**——
边界是：复用方法论与代码，不复用数据。
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class Turn(BaseModel):
    """一轮对话。评测用例里 dialogue 至少包含一个 user turn。"""

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)


class Dimension(str, Enum):
    """五个评测维度（写进 rubric 给裁判模型用）。"""

    FOLLOWUP = "追问合理性"
    CONSISTENCY = "上下文一致性"
    FACTUALITY = "事实可靠性"
    STYLE = "面试风格符合度"
    REFUSAL = "应拒答场景"


class EvalCase(BaseModel):
    id: str
    dimension: Dimension
    dialogue: List[Turn] = Field(..., min_length=1)
    reference: str = ""            # 期望要点 / 拒答时的期望行为
    expects_refusal: bool = False  # True = 模型必须拒绝回答
    notes: str = ""
    human_score: Optional[int] = Field(default=None, ge=1, le=5)  # 人工标注，用于算一致率

    @model_validator(mode="after")
    def last_turn_must_be_user(self) -> "EvalCase":
        if self.dialogue[-1].role != "user":
            raise ValueError(f"{self.id}: dialogue 最后一轮必须是 user")
        return self

    @property
    def history_and_query(self) -> tuple[List[Dict[str, str]], str]:
        """拆成（历史上下文, 当前提问）。"""
        *hist, last = self.dialogue
        return [{"role": t.role, "content": t.content} for t in hist], last.content


class EvalDataset(BaseModel):
    name: str = "mock-interview-eval"
    version: str = "0.1"
    rubric_version: str = "v1"
    cases: List[EvalCase]

    def by_dimension(self, dim: Dimension) -> List[EvalCase]:
        return [c for c in self.cases if c.dimension == dim]

    @property
    def size(self) -> int:
        return len(self.cases)


def load_dataset(path: str | Path) -> EvalDataset:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalDataset(**data)


def dump_sample(dest: str | Path, cases: List[EvalCase], **meta) -> Path:
    """把用例写回 JSON（标注完 KG5、补充用例时用）。"""
    ds = EvalDataset(cases=cases, **meta)
    p = Path(dest)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(ds.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")
    return p
