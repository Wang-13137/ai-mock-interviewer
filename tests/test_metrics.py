# -*- coding: utf-8 -*-
"""埋点工具测试：W1 出 P50/P95 表之前先保证这里对。"""

import json

from src.clients.llm import LLMUsage
from src.metrics import JsonlSink, LatencyMeter, percentile, summarize


def test_percentile_basic():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 100) == 5
    assert percentile([], 95) == 0.0
    assert percentile([7.0], 95) == 7.0


def test_percentile_interpolation():
    values = list(range(1, 101))
    assert 94.0 < percentile(values, 95) <= 96.0


def test_latency_meter_ttft():
    m = LatencyMeter()
    assert m.ttft_ms is None
    report = m.report(None)
    assert report["prompt_tokens"] is None
    assert report["total_ms"] >= 0


def test_latency_meter_with_usage():
    m = LatencyMeter()
    m.mark_first_token()
    report = m.report(LLMUsage(prompt_tokens=10, completion_tokens=4))
    assert report["ttft_ms"] is not None
    assert report["total_tokens"] == 14


def test_jsonl_sink_roundtrip(tmp_path):
    path = tmp_path / "logs" / "metrics.jsonl"
    sink = JsonlSink(path, enabled=True)
    sink.write({"ttft_ms": 1.0, "total_ms": 2.0})
    sink.write({"ttft_ms": 3.0, "total_ms": 4.0})
    rows = sink.read_all()
    assert len(rows) == 2
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["ttft_ms"] == 1.0


def test_sink_disabled_is_noop(tmp_path):
    path = tmp_path / "nope.jsonl"
    JsonlSink(path, enabled=False).write({"a": 1})
    assert not path.exists()


def test_summarize_over_rows():
    rows = [{"total_ms": 100.0}, {"total_ms": 200.0}, {"total_ms": 300.0}]
    stat = summarize(rows, "total_ms")
    assert stat["n"] == 3
    assert stat["mean"] == 200.0
    assert stat["p50"] == 200.0
    assert stat["p95"] >= 280.0
    assert summarize([], "total_ms")["n"] == 0
