"""
速率限制模块
============

基于 slowapi + limits 库，提供 per-user (authenticated) 或 per-IP (unauthenticated) 限流。

- 认证用户：按 user_id 限流，60 req/min
- 未认证用户（login/register）：按 IP 限流
- Crisis 接口（/safety/*, /health）：不限流 — 使用唯一 per-request key
- 默认存储：内存（单进程），生产通过 RATE_LIMIT_STORAGE_URI 切换 Redis

环境变量：
  RATE_LIMIT_DEFAULT     — 默认限额（默认 "60/minute"）
  RATE_LIMIT_STORAGE_URI — Redis 地址（为空时使用内存存储）
"""

from __future__ import annotations

import os
import uuid
from typing import Iterable, Optional

from fastapi import Request
from fastapi.routing import _IncludedRouter
from starlette.routing import BaseRoute, Match
from starlette.types import Scope
from slowapi import Limiter
from slowapi.util import get_remote_address


# ── 补丁：slowapi 的 _find_route_handler 不认识 FastAPI 的 _IncludedRouter ──
# app.include_router() 会把路由包在 _IncludedRouter 里，后者没有 .endpoint 属性。
# 这里递归展开 _IncludedRouter → 找到真正的 Route.endpoint。
def _patched_find_route_handler(
    routes: Iterable[BaseRoute], scope: Scope
) -> Optional:
    """递归搜索路由，兼容 FastAPI 的 _IncludedRouter 包装。

    FastAPI 的 app.include_router() 会把路由包在 _IncludedRouter 里，
    后者没有 .endpoint 属性，需要递归进入 original_router.routes 查找。
    original_router.routes 里的 Route 持有完整路径（已含 prefix），
    所以递归时复用原始 scope 即可。
    """
    for route in routes:
        match, _ = route.matches(scope)
        if match != Match.FULL:
            continue
        if hasattr(route, "endpoint"):
            return route.endpoint
        # FastAPI _IncludedRouter: 进入子路由递归
        if isinstance(route, _IncludedRouter):
            handler = _patched_find_route_handler(
                route.original_router.routes, scope
            )
            if handler is not None:
                return handler
    return None


import slowapi.middleware
slowapi.middleware._find_route_handler = _patched_find_route_handler

# 豁免路径前缀 — 不做限流
_EXEMPT_PREFIXES = (
    "/api/v1/multimodal/safety/",
    "/api/v1/modules/safety/",   # safety/check
)
_EXEMPT_EXACT = (
    "/api/v1/health",
    "/ping",
)


def _get_rate_limit_key(request: Request) -> str:
    """认证用户按 user_id、否则按 IP。Crisis / health 豁免。"""
    path = request.url.path

    # Crisis + 健康检查：唯一 key → 永不限流
    if path in _EXEMPT_EXACT or any(path.startswith(p) for p in _EXEMPT_PREFIXES):
        return f"exempt_{uuid.uuid4()}"

    # 认证用户按 user_id
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        try:
            token = auth.split(" ", 1)[1]
            from modules.token_service import TokenService
            payload = TokenService.verify_token(token)
            return payload["user_id"]
        except Exception:
            pass

    return get_remote_address(request)


# 默认限额：可通过环境变量覆盖（测试用时设 "5/minute" 加速验证）
_default_limit = os.environ.get("RATE_LIMIT_DEFAULT", "60/minute")

# 存储后端：Redis URL or None（内存）
_storage_uri = os.environ.get("RATE_LIMIT_STORAGE_URI", "")
if not _storage_uri:
    _storage_uri = os.environ.get("REDIS_URL", "")

limiter = Limiter(
    key_func=_get_rate_limit_key,
    storage_uri=_storage_uri or None,  # None → in-memory
    default_limits=[_default_limit],
    headers_enabled=True,
)
