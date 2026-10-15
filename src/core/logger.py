# -*- coding: utf-8 -*-
"""统一日志入口。服务/CLI 都从这里初始化，保证格式一致。"""

from __future__ import annotations

import logging
import sys


def setup_logging(env: str = "dev") -> None:
    level = logging.DEBUG if env == "dev" else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)-10s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
