# -*- coding: utf-8 -*-
"""面试会话状态。

面试 Agent 和普通聊天机器人的本质区别就在这里：**它是有状态的**。
已经问过哪些维度、候选人答到了什么点、还剩几轮收尾——这些都是工具要读写的对象。

当前是进程内内存版（够单实例开发和评测用）。W5 换成 Redis / SQLite 时，
只要保持 `SessionStore` 的接口不变，工具层一行都不用改。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# 一场模拟面试要覆盖的能力维度，题库与进度工具共用
DIMENSIONS: List[str] = [
    "项目深挖",
    "LLM 工程能力",
    "数据与统计",
    "工程基础",
    "行为与动机",
]


@dataclass
class SessionState:
    session_id: str
    asked_dimensions: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    turns: int = 0
    finished: bool = False
    summary: str = ""

    @property
    def remaining_dimensions(self) -> List[str]:
        return [d for d in DIMENSIONS if d not in self.asked_dimensions]

    @property
    def coverage(self) -> float:
        """维度覆盖率 —— 面试 Agent 的关键指标之一。"""
        if not DIMENSIONS:
            return 0.0
        return round(len(self.asked_dimensions) / len(DIMENSIONS), 4)


class SessionStore:
    """带 asyncio 锁的会话表。锁在这里是必要的：FastAPI 并发下多个请求可能同 session。"""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions: Dict[str, SessionState] = {}

    async def get(self, session_id: str) -> SessionState:
        async with self._lock:
            st = self._sessions.get(session_id)
            if st is None:
                st = SessionState(session_id=session_id)
                self._sessions[session_id] = st
            return st

    def peek(self, session_id: str) -> Optional[SessionState]:
        """非加锁读取，仅测试与只读接口使用。"""
        return self._sessions.get(session_id)

    async def reset(self, session_id: Optional[str] = None) -> int:
        async with self._lock:
            if session_id:
                existed = self._sessions.pop(session_id, None)
                return 1 if existed else 0
            n = len(self._sessions)
            self._sessions.clear()
            return n

    def size(self) -> int:
        return len(self._sessions)
