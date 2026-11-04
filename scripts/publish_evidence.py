"""把 reports/ 里的检索评测产物脱敏后发布到 docs/evidence/。

动机：README / 博客里引用的 Recall@3 0.825、MRR@10 0.829 等数字，
原始报告因含本机绝对路径与本机用户名而被 .gitignore 排除，公开读者无法验证。
本脚本只做「路径脱敏 + 复制」，不改动任何数值、不重新计算指标。

用法：
    python scripts/publish_evidence.py

输入（不进公开仓库）：
    reports/retrieval-bge-60q_20261016-004711.{json,md}      最终 60 条评测集，无重排
    reports/retrieval-rerank-60q_20261016-010128.{json,md}   最终 60 条评测集，有重排
    reports/retrieval-bge-80_20261016-004201.{json,md}       早期 30 条评测集，无重排
    reports/retrieval-rerank-80_20261016-004236.{json,md}    早期 30 条评测集，有重排

输出（提交进公开仓库）：
    docs/evidence/retrieval-{bge,rerank}-{60q,80}.{json,md}

四份都公开，是为了让 README 里两组数字（最终 60 条表的 0.825/0.829、
早期 30 条集上重排 MRR 0.883→0.828）都能逐格追溯到原始逐条明细。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = ROOT / "docs" / "evidence"

SOURCES = {
    "retrieval-bge-60q": "retrieval-bge-60q_20261016-004711",
    "retrieval-rerank-60q": "retrieval-rerank-60q_20261016-010128",
    "retrieval-bge-80": "retrieval-bge-80_20261016-004201",
    "retrieval-rerank-80": "retrieval-rerank-80_20261016-004236",
}

# 本机绝对路径（Windows / POSIX 都覆盖），只保留模型名
PATH_RE = re.compile(
    r"[A-Za-z]:[\\/][^\s\"'|,]*?[\\/](?:bge-small-zh-v1\.5)|"
    r"/(?:Users|home)/[^\s\"'|,]*?/models/[^\s\"'|,]*",
)

DISCLOSURE = "bge-small-zh-v1.5 (local weights, not shipped with this repo)"


def scrub(value):
    """递归脱敏：任何含本机模型路径的字符串替换为模型名 + 说明。"""
    if isinstance(value, str):
        if PATH_RE.search(value) or ("models" in value and "bge-small" in value):
            return DISCLOSURE
        return value
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    return value


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for target, stem in SOURCES.items():
        src_json = REPORTS / f"{stem}.json"
        src_md = REPORTS / f"{stem}.md"
        if not src_json.exists() or not src_md.exists():
            print(f"[SKIP] 缺原始报告：{stem}")
            continue

        data = json.loads(src_json.read_text(encoding="utf-8"))
        cleaned = scrub(data)
        (OUT / f"{target}.json").write_text(
            json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        md = src_md.read_text(encoding="utf-8")
        md = PATH_RE.sub(DISCLOSURE, md)
        md = re.sub(
            r"C:\\Users\\[^\s|]*?bge-small-zh-v1\.5",
            DISCLOSURE,
            md,
        )
        header = (
            f"> 来源：本机 `reports/{stem}`，2026-10-16 实跑；"
            "本文件仅对模型权重的本机绝对路径做脱敏，数值与逐条明细未改动。\n"
            "> 可引用范围见 [`docs/RESULTS.md`](../RESULTS.md)。\n\n"
        )
        (OUT / f"{target}.md").write_text(header + md, encoding="utf-8")

        # 硬性校验：产物里不得残留本机路径
        for f in (OUT / f"{target}.json", OUT / f"{target}.md"):
            text = f.read_text(encoding="utf-8")
            assert "C:\\Users" not in text and "/Users/" not in text, f"残留本机路径：{f}"
        print(f"[OK] {target}.json / {target}.md（已脱敏，数值未改）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
