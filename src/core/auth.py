# -*- coding: utf-8 -*-
"""写接口鉴权：一个 API Key 就够，不做 RBAC。

判断依据很简单——**这个服务的真实风险不是"越权读数据"，而是"任何人调一次
POST /v1/eval/runs 就能烧掉额度、占满 CPU"**。所以保护面只需要覆盖写操作，
读接口保持公开（否则本地调试和演示都要先配 key，得不偿失）。

空 key = 不校验，这是本地开发与测试的默认值。**服务一旦对外可达就必须填**
（`AIMI_AUTH__API_KEY=...`），这是部署清单里的一条硬性检查。

用 `hmac.compare_digest` 而不是 `==`：字符串比较会在第一个不同字节就返回，
理论上可被用来逐字节爆破 key。这类细节不影响功能，但它是"知道自己在写服务端"
和"只是把接口跑通"的分界。
"""

from __future__ import annotations

import hmac

from fastapi import HTTPException, Request, status


def require_api_key(request: Request) -> None:
    """FastAPI 依赖：校验 X-API-Key。配置为空时放行。"""
    settings = getattr(request.app.state, "settings", None)
    auth = getattr(settings, "auth", None)
    expected = (getattr(auth, "api_key", "") or "").strip()
    if not expected:
        return

    provided = (request.headers.get("X-API-Key") or "").strip()
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="写接口需要 X-API-Key（服务端配置项 auth.api_key，环境变量 AIMI_AUTH__API_KEY）",
        )
