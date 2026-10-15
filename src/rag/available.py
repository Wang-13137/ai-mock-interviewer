# -*- coding: utf-8 -*-
"""后端可用性探测（单独放一个模块，方便脚本与测试复用）。"""

from __future__ import annotations

from typing import Dict

from .retriever import available_backends as _ab

__all__ = ["available_backends"]


def available_backends() -> Dict[str, bool]:
    return _ab()
