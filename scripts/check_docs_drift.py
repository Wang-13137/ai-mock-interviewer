# -*- coding: utf-8 -*-
"""文档漂移检查：让「真值表」不会自己过期。

为什么需要这个脚本：
    `docs/RESULTS.md` 自称「本项目唯一权威的数字口径」，但中台加完 40 个用例后，
    实际已是 179 passed，README 更新了、真值表却还写着 139。
    **一份自称权威、本身却是唯一错的文件，比没有这份文件更糟。**

    人一定会忘。所以纪律要写成机器能把关的形式——这正是这个项目一贯的做法
    （`trusted` 字段、质量门禁、回归门禁，都是同一个思路）。

检查两件事：
    1. 所有文档里的「<数字> passed」必须等于 `pytest --collect-only` 的真实用例数；
    2. `pyproject.toml` 的 version 必须等于 `src/version.py` 的 `__version__`
       （发版时最容易漏同步的一对）。

豁免：行内出现 `drift-ignore` 的行跳过（用于引用历史数字的叙述）。
用法：`python scripts/check_docs_drift.py`，退出码 0 = 一致。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
PASSED = re.compile(r"(\d+)\s*passed")
IGNORE = "drift-ignore"


def actual_test_count() -> int:
    """跑一次 collect-only 取真实用例数。只看 stdout 末端的汇总行。"""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    hits = re.findall(r"(\d+)\s+tests?\s+collected", out)
    if not hits:
        raise RuntimeError(f"无法从 pytest 输出解析用例数：\n{out[-800:]}")
    return int(hits[-1])


def scanned_files() -> List[Path]:
    """README + docs 顶层 md。docs/evidence/ 是历史报告，不参与——它记录的是当时的事实。"""
    return [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def version_drift() -> List[str]:
    problems: List[str] = []
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^\s*version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    pkg_version = m.group(1) if m else ""
    init = (ROOT / "src" / "version.py").read_text(encoding="utf-8")
    m2 = re.search(r'__version__\s*=\s*"([^"]+)"', init)
    mod_version = m2.group(1) if m2 else ""
    if not pkg_version or not mod_version:
        problems.append("无法解析版本号：pyproject.toml 或 src/version.py")
    elif pkg_version != mod_version:
        problems.append(f"版本不一致：pyproject.toml={pkg_version} src/version.py={mod_version}")
    return problems


def main() -> int:
    real = actual_test_count()
    problems: List[str] = []
    checked = 0

    for path in scanned_files():
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if IGNORE in line:
                continue
            for num in PASSED.findall(line):
                checked += 1
                if int(num) != real:
                    problems.append(
                        f"{path.relative_to(ROOT)}:{lineno}  写着 {num} passed，实际 {real} passed"
                    )

    problems.extend(version_drift())

    if problems:
        print(f"[FAIL] 文档漂移检查未通过（真实用例数 {real}）：")
        for p in problems:
            print(f"  - {p}")
        print("\n修法：把文档里的数字改成真实值，或在本行加 drift-ignore 标记（仅限引用历史数字时）。")
        return 1

    print(f"[OK] 文档一致：{checked} 处用例数断言均等于 {real}；版本号两处同步。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
