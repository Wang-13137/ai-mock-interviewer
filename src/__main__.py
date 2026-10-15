# -*- coding: utf-8 -*-
"""src/__main__.py —— 允许 `python -m src serve` / `python -m src chat`。"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
