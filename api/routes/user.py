"""
User API 端点
===========

GET /api/v1/user/export — 全量导出用户数据（需密码验证）
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query, status

from modules.auth_deps import get_current_user_id
from modules.encryption import safe_decrypt_field

router = APIRouter(prefix="/api/v1/user", tags=["user"])


def _serialize_datetime(val):
    """将 datetime 对象序列化为 ISO 字符串。"""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.isoformat()
    return str(val)


def _row_to_dict(row, columns: List[str]) -> Dict[str, Any]:
    """将 ORM 对象按指定列名转为 dict。"""
    result = {}
    for col in columns:
        val = getattr(row, col, None)
        result[col] = _serialize_datetime(val)
    return result


def _collect_user_data(user_id: str) -> Dict[str, Any]:
    """收集用户全部数据，加密字段在导出时解密。"""
    from schemas.database import db_manager
    from schemas.database_v2 import (
        User, SessionV2, MessageV2, EmotionRecordV2,
        SafetyFlag, ScaleScreening,
    )

    with db_manager.get_session_direct() as s:
        # 1. 用户档案（不含凭证）
        user = s.query(User).filter(User.user_id == user_id).first()
        if user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")
        profile = {
            "user_id": user.user_id,
            "display_name": user.display_name,
            "status": user.status,
            "created_at": _serialize_datetime(user.created_at),
            "updated_at": _serialize_datetime(user.updated_at),
            "consent_at": _serialize_datetime(user.consent_at),
            "consent_version": user.consent_version,
        }

        # 2. 会话
        sessions_rows = s.query(SessionV2).filter(SessionV2.user_id == user_id).all()
        sessions = []
        for row in sessions_rows:
            sessions.append(_row_to_dict(row, [
                "id", "session_id", "user_id", "message_count",
                "key_topics", "scale_history", "created_at", "last_active",
            ]))

        # 3. 消息 — content 解密
        session_ids = [s["session_id"] for s in sessions]
        messages_rows = []
        if session_ids:
            messages_rows = s.query(MessageV2).filter(
                MessageV2.session_id.in_(session_ids)
            ).all()
        messages = []
        for row in messages_rows:
            d = _row_to_dict(row, ["id", "session_id", "role", "created_at"])
            d["content"] = safe_decrypt_field(row.content) or ""
            messages.append(d)

        # 4. 情绪记录 — context 解密
        emotion_rows = s.query(EmotionRecordV2).filter(
            EmotionRecordV2.user_id == user_id
        ).all()
        emotion_records = []
        for row in emotion_rows:
            d = _row_to_dict(row, [
                "id", "session_id", "user_id", "primary_emotion",
                "intensity", "risk", "triggers", "intent",
                "modality_notes", "created_at",
            ])
            d["context"] = safe_decrypt_field(row.context)
            emotion_records.append(d)

        # 5. 安全标记 — matched_terms 解密
        safety_flags_rows = s.query(SafetyFlag).filter(
            SafetyFlag.user_id == user_id
        ).all()
        safety_flags = []
        for row in safety_flags_rows:
            d = _row_to_dict(row, [
                "id", "session_id", "user_id", "level", "blocked",
                "reviewed", "reviewed_by", "reviewed_at", "created_at",
            ])
            d["matched_terms"] = safe_decrypt_field(row.matched_terms)
            safety_flags.append(d)

        # 6. 量表筛查
        screenings_rows = s.query(ScaleScreening).filter(
            ScaleScreening.user_id == user_id
        ).all()
        scale_screenings = []
        for row in screenings_rows:
            scale_screenings.append(_row_to_dict(row, [
                "id", "session_id", "user_id", "scale_type", "state",
                "responses", "scores", "total_score", "interpretation",
                "created_at", "completed_at",
            ]))

    return {
        "profile": profile,
        "sessions": sessions,
        "messages": messages,
        "emotion_records": emotion_records,
        "safety_flags": safety_flags,
        "scale_screenings": scale_screenings,
    }


@router.get("/export", status_code=status.HTTP_200_OK)
def export_user_data(
    password: str = Query(..., description="当前密码，用于确认身份"),
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """导出用户全量数据（需密码验证）。

    返回结构化 JSON，包含：
    - profile：用户档案
    - sessions：所有会话
    - messages：所有消息（已解密）
    - emotion_records：所有情绪记录（已解密）
    - safety_flags：所有安全标记（已解密）
    - scale_screenings：所有量表筛查结果

    不导出凭证密文。
    """
    from modules.auth_service import AuthService
    from schemas.database import db_manager
    from schemas.database_v2 import Credential

    # 验证密码
    with db_manager.get_session_direct() as s:
        cred = s.query(Credential).filter(
            Credential.user_id == user_id,
            Credential.type == "password",
        ).first()
        if not cred or not cred.secret:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="账号未设置密码",
            )
        if not AuthService.verify_password(password, cred.secret):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="密码错误",
            )

    return _collect_user_data(user_id)
