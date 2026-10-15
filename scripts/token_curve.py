# -*- coding: utf-8 -*-
"""多轮会话的 Token 增长曲线对照（W5 的可交付数字）。

对照两臂，同一份脚本化话术：
  full    —— 不做任何压缩，历史原文全塞进上下文（线性增长）
  summary —— 开启滚动摘要（token 预算 + 保留最近 N 轮原文）

真实模型的 prompt_tokens 来自 API usage；估算值用 memory.estimate_tokens 同源口径，
两条一起记，便于事后核对估算器是否靠谱。

用法：
    AIMI_LLM__PROVIDER=zhipu AIMI_LLM__API_KEY=<key> \
      python scripts/token_curve.py --turns 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # 直接用 python scripts/xxx.py 启动时保证 src 可导入
    sys.path.insert(0, str(ROOT))

CANDIDATE_SCRIPT = [
    "我在深圳湾实验室做 MALDI-TOF 质谱的代谢组学分析，负责谱图采集、预处理和后续的机器学习建模。",
    "我们在中文临床指南问答系统里自建了 RAG：16 份指南切分后用 bge-small-zh 建索引，再对比了 BM25 混合检索和向量召回。",
    "评测集是我自建的 100 题，其中 15 题是应拒答题；关键串逐条和原文对齐，用 McNemar 检验做过显著性判断。",
    "幻觉控制主要靠受限提示词强制引用标注，引用精确率从 0.010 提到了 0.651。",
    "模型选型的结论是混合检索加 cross-encoder 重排最好，关键串召回率从 0.794 提到 0.912。",
    "毕业后我希望做 AI 应用工程师，偏向 LLM 应用和检索增强生成方向。",
    "向量库我用的是 FAISS，索引类型选 IVF Flat，主要是为了平衡召回和内存占用。",
    "多轮对话的上下文管理我做了滑动窗口加摘要压缩，超预算就把最老的内容压成摘要。",
    "我觉得最难的是评测：怎么让自动打分和人工判断对齐，我用了另一个模型当裁判再人工抽检。",
    "这个项目里我踩过最大的坑是负结果：深度学习没有超过传统方法，但把原因查清楚了。",
]


def _build_loop(settings, with_memory: bool):
    from src.agent.loop import AgentLoop
    from src.agent.memory import MemoryManager, MemoryPolicy
    from src.agent.tools import build_default_registry, load_question_bank
    from src.app import build_client

    client = build_client(settings)
    bank = load_question_bank(ROOT / "config" / "question_bank.json")
    reg = build_default_registry(
        bank,
        strategy=settings.rag.strategy,
        embedder_name=settings.rag.embedder,
        model_name=settings.rag.model_name,
        min_score=settings.rag.min_score,
    )
    policy = MemoryPolicy(
        enabled=with_memory,
        max_tokens=settings.memory.max_tokens,
        keep_recent=settings.memory.keep_recent,
    )
    loop = AgentLoop(client, reg, bank=bank, max_steps=settings.agent.max_steps,
                     memory=MemoryManager(client, policy))
    return loop, client


async def run_arm(settings, turns: int, with_memory: bool, profile: Dict[str, str]) -> List[Dict]:
    from src.agent.memory import history_tokens

    loop, client = _build_loop(settings, with_memory)
    history: List[Dict[str, str]] = []
    rows: List[Dict] = []
    try:
        for t in range(1, turns + 1):
            user = CANDIDATE_SCRIPT[(t - 1) % len(CANDIDATE_SCRIPT)]
            history.append({"role": "user", "content": user})
            run = await loop.run(history, profile, session_id=f"token-curve-{'sum' if with_memory else 'full'}")
            history.append({"role": "assistant", "content": run.reply})
            rows.append({
                "turn": t,
                "messages": len(history),
                "prompt_tokens_api": run.usage.prompt_tokens,
                "est_tokens": history_tokens(history),
                "reply_chars": len(run.reply),
                "elapsed_ms": round(run.elapsed_ms, 1),
            })
            print(f"  [{'summary' if with_memory else 'full'}] turn {t:>2}: "
                  f"prompt={run.usage.prompt_tokens or '?':>5} est={rows[-1]['est_tokens']:>5} "
                  f"msgs={len(history)} {run.elapsed_ms:.0f}ms")
    finally:
        await client.aclose()
    return rows


def _pick(rows: List[Dict], turn: int) -> Optional[Dict]:
    for r in rows:
        if r["turn"] == turn:
            return r
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=30)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    from src.core.config import load_settings

    settings = load_settings()
    profile = {"position": "AI 应用工程师", "level": "校招应届", "style": "专业严谨"}

    print(f"provider={settings.llm.provider} model={settings.llm.model} "
          f"turns={args.turns} budget={settings.memory.max_tokens}")

    full = asyncio.run(run_arm(settings, args.turns, False, profile))
    summary = asyncio.run(run_arm(settings, args.turns, True, profile))

    marks = [5, 10, 20, args.turns]
    lines = []
    print("\n| 轮次 | 全历史 prompt_tokens | 滚动摘要 prompt_tokens | 压缩比 |")
    print("|---|---|---|---|")
    for m in marks:
        a, b = _pick(full, m), _pick(summary, m)
        if not a or not b:
            continue
        pa, pb = a["prompt_tokens_api"] or 0, b["prompt_tokens_api"] or 0
        ratio = f"{(1 - pb / pa) * 100:.1f}%" if pa else "—"
        print(f"| {m} | {pa} | {pb} | {ratio} |")
        lines.append({"turn": m, "full": pa, "summary": pb, "saved": ratio})
    if full and summary:
        growth_full = (full[-1]["prompt_tokens_api"] or 0) / max(1, full[0]["prompt_tokens_api"] or 1)
        growth_sum = (summary[-1]["prompt_tokens_api"] or 0) / max(1, summary[0]["prompt_tokens_api"] or 1)
        print(f"\n首轮→末轮增长倍数：全历史 {growth_full:.1f}×，滚动摘要 {growth_sum:.1f}×")
        lines.append({"growth_full": round(growth_full, 2), "growth_summary": round(growth_sum, 2)})

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    tag = args.tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "provider": f"{settings.llm.provider}/{settings.llm.model}",
        "turns": args.turns,
        "memory_policy": {
            "max_tokens": settings.memory.max_tokens,
            "keep_recent": settings.memory.keep_recent,
        },
        "full_history": full,
        "rolling_summary": summary,
        "marks": lines,
    }
    out = out_dir / f"token-curve_{tag}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n原始数据：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
