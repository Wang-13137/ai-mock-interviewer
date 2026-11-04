# -*- coding: utf-8 -*-
"""单一版本号来源。

`pyproject.toml` 里的 version 是打包元数据（静态，需与这里同步）；
运行时 `/health`、OpenAPI、`Settings.version` 全部引用本文件，
避免再出现「tag 是 v0.6、/health 却报 0.1.0」的版本漂移。

发版流程：改这里 → 同步 pyproject.toml → 打 git tag（两者保持一致）。
"""
from __future__ import annotations

__version__ = "0.8.0"
