# -*- coding: utf-8 -*-
"""溯源信息：这个数字是**哪份代码、在什么环境**下跑出来的。

一个 run 记录如果只有指标和配置指纹，能回答"结果是否可比"，
但回答不了"退步是哪次提交引入的"。这两件事需要不同的字段：

    config_fingerprint  参与可比性判定。**刻意不含代码版本**——
                        如果代码一改就不可比，门禁就永远抓不到代码改动导致的退步，
                        而这正是门禁存在的意义。
    code_version        只记录，不参与判定。用于事后归因：把两个 run 的 sha 一对比，
                        就知道该看哪几条 commit。

`dirty` 后缀很重要：工作区有未提交改动时，指标对应的代码**不存在于任何提交里**，
这种 run 可以用来调试，但不该被当成基线——数字无法复现。
"""

from __future__ import annotations

import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def code_version(root: str | None = None) -> str:
    """git short sha，工作区有改动时加 `-dirty`。取不到就返回空串（不猜）。"""
    cwd = root or str(REPO_ROOT)
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        if not sha:
            return ""
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd, capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.SubprocessError):
        return ""


@lru_cache(maxsize=1)
def runtime_snapshot() -> Dict[str, str]:
    """运行环境快照。只记影响数值的几项，不做 pip freeze（那会让指纹噪声化）。"""
    snap: Dict[str, str] = {"python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"}
    try:
        from importlib.metadata import PackageNotFoundError, version

        for pkg in ("fastapi", "pydantic", "httpx"):
            try:
                snap[pkg] = version(pkg)
            except PackageNotFoundError:
                snap[pkg] = "?"
    except Exception:  # pragma: no cover - importlib.metadata 在 3.8+ 一定存在
        pass
    return snap
