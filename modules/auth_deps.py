"""
Auth Dependencies — FastAPI Depends 注入
========================================

从 Authorization: Bearer <token> header 提取并验证 JWT，
返回 user_id。
"""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

from modules.token_service import TokenService

_http_bearer = HTTPBearer()


def get_current_user_id(
    credential: Optional[HTTPBearer] = Depends(_http_bearer),
) -> str:
    """FastAPI Depends：从 Bearer header 提取当前用户 ID。"""
    if not credential or not credential.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 Authorization header",
        )
    token = credential.credentials
    try:
        payload = TokenService.verify_token(token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        )
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="请使用 access_token（非 refresh_token）",
        )

    user_id = payload["user_id"]

    # 检查用户状态：已注销的用户 token 立即失效
    from schemas.database import db_manager
    from schemas.database_v2 import User
    with db_manager.get_session_direct() as s:
        user = s.query(User).filter(User.user_id == user_id).first()
        if user is None or user.status == "deleted":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="账号已注销或不存在",
            )

    return user_id
