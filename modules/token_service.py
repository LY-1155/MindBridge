"""
Token Service — JWT 签发与验证
==============================

access_token: 30 分钟，用于业务请求
refresh_token: 30 天，用于换取新 access_token

密钥: JWT_SECRET 环境变量
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import jwt

_ALGORITHM = "HS256"


def _secret() -> str:
    raw = os.environ.get("JWT_SECRET", "dev-jwt-secret-change-in-production")
    return raw


class TokenService:
    """JWT Token 相关静态方法。"""

    @staticmethod
    def issue_access_token(user_id: str, expires_in_seconds: int = 3600) -> str:
        """签发 access_token（默认 1 小时）。"""
        now = datetime.now(timezone.utc)
        payload = {
            "user_id": user_id,
            "type": "access",
            "iat": now,
            "exp": now + timedelta(seconds=expires_in_seconds),
        }
        return jwt.encode(payload, _secret(), algorithm=_ALGORITHM)

    @staticmethod
    def issue_refresh_token(user_id: str) -> str:
        """签发 refresh_token（30 天）。"""
        now = datetime.now(timezone.utc)
        payload = {
            "user_id": user_id,
            "type": "refresh",
            "iat": now,
            "exp": now + timedelta(days=30),
        }
        return jwt.encode(payload, _secret(), algorithm=_ALGORITHM)

    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """验证 token 并返回 payload。不合法则 raise ValueError。"""
        try:
            payload = jwt.decode(
                token, _secret(), algorithms=[_ALGORITHM],
                options={"require": ["exp", "user_id", "type"]},
            )
        except jwt.ExpiredSignatureError:
            raise ValueError("token 已过期")
        except jwt.InvalidTokenError as exc:
            raise ValueError(f"token 无效: {exc}")
        return payload
