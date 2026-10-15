# -*- coding: utf-8 -*-
"""时延与用量埋点。

W1 Step1 的「改造前 / 改造后」对比表就是从这个 JSONL 里算出来的，
每条记录都必须能在简历里追到一个真实跑过的场景。
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .clients.llm import LLMUsage


class LatencyMeter:
    """记录 TTFT（首字延迟）与端到端耗时。线程/协程内单实例使用。"""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self._first_token: Optional[float] = None

    @property
    def ttft_ms(self) -> Optional[float]:
        if self._first_token is None:
            return None
        return round((self._first_token - self.t0) * 1000, 2)

    @property
    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self.t0) * 1000, 2)

    @property
    def saw_first_token(self) -> bool:
        return self._first_token is not None

    def mark_first_token(self) -> None:
        if self._first_token is None:
            self._first_token = time.perf_counter()

    def report(self, usage: Optional[LLMUsage] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "ttft_ms": self.ttft_ms,
            "total_ms": self.elapsed_ms,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
        }
        if usage is not None and usage.total_tokens is not None:
            out["total_tokens"] = usage.total_tokens
        return out


class JsonlSink:
    """把指标追加写成本地 JSONL；关掉 enabled 即为 no-op（测试默认关）。"""

    def __init__(self, path: str | Path = "logs/metrics.jsonl", enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled

    def write(self, record: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows


def percentile(values: Sequence[float], p: float) -> float:
    """线性插值分位数，p 取 0–100。W1 出 P50/P95 用。"""
    data = sorted(v for v in values if v is not None)
    if not data:
        return 0.0
    if len(data) == 1:
        return float(data[0])
    k = (len(data) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(data) - 1)
    if lo == hi:
        return float(data[lo])
    return float(data[lo] + (data[hi] - data[lo]) * (k - lo))


def summarize(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, float]:
    """从指标行里汇总某个数值字段的均值 / P50 / P95。"""
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    if not vals:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "n": len(vals),
        "mean": round(sum(vals) / len(vals), 2),
        "p50": round(percentile(vals, 50), 2),
        "p95": round(percentile(vals, 95), 2),
    }
