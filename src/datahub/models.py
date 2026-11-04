# -*- coding: utf-8 -*-
"""数据中台的数据契约。

为什么需要一层「中台」，而不是继续直接读 JSON 文件：
    项目里已经有 4 份 JSON 数据（题库 80 条、检索评测 60 条、对话评测 10 条、bench 题 12 条），
    它们之间存在**引用关系**（检索评测的 relevant 指向题库 id），也存在**质量约束**
    （用例最后一句必须是 user、标注不能泄露答案）。散着读文件时，这些约束全靠人记；
    一旦题库改了 id 而评测集没跟着改，评测会静默跑出错误的数字——**不报错，但结论是假的**。

中台把这类静默错误变成显式契约：
    1. 每份数据登记成 dataset，带 content_hash（内容指纹）——数据一改，指纹就变，评测可追溯；
    2. 登记时过质量门禁，error 级别的问题直接拒绝入库；
    3. 跨数据集做引用完整性检查（评测集指向的题目必须真实存在）；
    4. 评测中台引用数据集时绑定的是 hash 快照，而不是路径——路径会被悄悄改，hash 不会。

这里只放 Pydantic 模型，不碰 IO，便于测试与替换存储。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DatasetKind(str, Enum):
    """数据集种类：决定用哪套质量校验器。"""

    QUESTION_BANK = "question_bank"
    RETRIEVAL_EVAL = "retrieval_eval"
    EVAL_SET = "eval_set"
    BENCH = "bench"


class DatasetSpec(BaseModel):
    """一份数据集的登记信息（静态部分）。

    `path` 相对仓库根目录；`depends_on` 声明它引用了哪些数据集——这是血缘的输入。
    """

    id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z0-9_\-]+$")
    kind: DatasetKind
    path: str
    title: str = ""
    owner: str = "wzt"
    tags: List[str] = Field(default_factory=list)
    description: str = ""
    depends_on: List[str] = Field(default_factory=list)


class QualityIssue(BaseModel):
    """一条质量问题。

    error   = 数据不可用，拒绝入库（例如评测集指向不存在的题号）
    warning = 可以入库，但会记在报告里，面试/发布前必须能解释
    """

    level: Literal["error", "warning"]
    code: str
    message: str
    where: str = ""


class QualityReport(BaseModel):
    """一次质量检查的结果。按 content_hash 存档，所以「同一份数据只算一次」。"""

    dataset_id: str
    content_hash: str
    checked_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    errors: List[QualityIssue] = Field(default_factory=list)
    warnings: List[QualityIssue] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.errors

    @property
    def status(self) -> str:
        if self.errors:
            return "blocked"
        return "warning" if self.warnings else "ok"

    def summary(self) -> str:
        return f"{self.status}（error={len(self.errors)} warning={len(self.warnings)}）"


class DatasetVersion(BaseModel):
    """数据集的一个版本快照。content_hash 是主键的一部分，保证幂等。"""

    dataset_id: str
    content_hash: str
    rows: int = 0
    size_bytes: int = 0
    quality_status: str = "ok"
    registered_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class DatasetRecord(BaseModel):
    """对外返回的聚合视图：登记信息 + 最新版本 + 质量摘要。"""

    spec: DatasetSpec
    latest: Optional[DatasetVersion] = None
    versions: int = 0
    quality: Optional[QualityReport] = None
    # 被谁引用（反向血缘）：例如 question_bank 被 retrieval_eval 引用
    referenced_by: List[str] = Field(default_factory=list)
    missing: bool = False  # 文件不存在
    # 磁盘内容 ≠ 已登记的最新版本：说明"改了但没同步"，此时引用它跑出的评测不可追溯
    dirty: bool = False


class LineageEdge(BaseModel):
    """血缘边：from 引用 to。"""

    from_dataset: str
    to_dataset: str
    field: str
    count: int = 0


class LineageGraph(BaseModel):
    nodes: List[str] = Field(default_factory=list)
    edges: List[LineageEdge] = Field(default_factory=list)


class SyncResult(BaseModel):
    """一次同步的结果：登记了哪些、跳过了哪些（hash 未变）、哪些被门禁拦下。"""

    dataset_id: str
    content_hash: str = ""
    status: Literal["registered", "unchanged", "blocked", "missing"] = "registered"
    rows: int = 0
    quality: Optional[QualityReport] = None
    message: str = ""
