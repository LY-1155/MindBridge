"""
User Service — 用户生命周期管理
===============================

用户创建、查询、状态变更、软删除/恢复。
只依赖 database_v2 ORM，不涉及认证逻辑（认证在 auth_service.py）。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from schemas.database import db_manager
from schemas.database_v2 import User

_VALID_STATUSES = {"active", "disabled", "deleted"}
GRACE_PERIOD_DAYS = 30


class UserService:
    """用户管理的静态方法集合。"""

    @staticmethod
    def create_user(display_name: Optional[str] = None) -> str:
        """创建新用户，返回 user_id。"""
        user = User(
            display_name=display_name,
            status="active",
        )
        with db_manager.get_session_direct() as s:
            s.add(user)
            s.commit()
            return user.user_id

    @staticmethod
    def get_user(user_id: str) -> Optional[Dict[str, Any]]:
        """按 user_id 查询用户。"""
        with db_manager.get_session_direct() as s:
            user = s.query(User).filter(User.user_id == user_id).first()
            if user is None:
                return None
            return _user_to_dict(user)

    @staticmethod
    def set_status(user_id: str, status: str) -> bool:
        """更新用户状态。"""
        if status not in _VALID_STATUSES:
            raise ValueError(f"无效状态: {status}，合法值: {_VALID_STATUSES}")
        with db_manager.get_session_direct() as s:
            user = s.query(User).filter(User.user_id == user_id).first()
            if user is None:
                return False
            user.status = status
            user.updated_at = datetime.now(timezone.utc)
            s.commit()
            return True

    @staticmethod
    def soft_delete(user_id: str) -> bool:
        """软删除：标记为 deleted 并记录时间。"""
        with db_manager.get_session_direct() as s:
            user = s.query(User).filter(User.user_id == user_id).first()
            if user is None:
                return False
            user.status = "deleted"
            user.deleted_at = datetime.now(timezone.utc)
            user.updated_at = datetime.now(timezone.utc)
            s.commit()
            return True

    @staticmethod
    def is_within_grace_period(user_id: str) -> Tuple[bool, Optional[str]]:
        """检查用户是否在 30 天后悔期内。返回 (in_grace_period, error_msg)。"""
        with db_manager.get_session_direct() as s:
            user = s.query(User).filter(User.user_id == user_id).first()
            if user is None:
                return False, "用户不存在"
            if user.status != "deleted":
                return False, "用户未处于注销状态"
            if user.deleted_at is None:
                return False, "缺少注销时间记录"
            # MySQL 可能返回 naive datetime，统一转 UTC 再比较
            deleted = user.deleted_at
            if deleted.tzinfo is None:
                deleted = deleted.replace(tzinfo=timezone.utc)
            deadline = deleted + timedelta(days=GRACE_PERIOD_DAYS)
            if datetime.now(timezone.utc) > deadline:
                return False, f"已超过 {GRACE_PERIOD_DAYS} 天后悔期，无法恢复"
            return True, None

    @staticmethod
    def restore(user_id: str) -> bool:
        """恢复软删除的用户（仅在 30 天后悔期内）。"""
        in_grace, _ = UserService.is_within_grace_period(user_id)
        if not in_grace:
            return False
        with db_manager.get_session_direct() as s:
            user = s.query(User).filter(User.user_id == user_id).first()
            if user is None:
                return False
            user.status = "active"
            user.deleted_at = None
            user.updated_at = datetime.now(timezone.utc)
            s.commit()
            return True

    @staticmethod
    def purge_expired_accounts() -> int:
        """物理删除超过 30 天后悔期的已注销用户及其关联数据。返回删除数量。"""
        cutoff = datetime.now(timezone.utc) - timedelta(days=GRACE_PERIOD_DAYS)
        with db_manager.get_session_direct() as s:
            all_deleted = s.query(User).filter(User.status == "deleted").all()
            expired = []
            for user in all_deleted:
                if user.deleted_at is None:
                    continue
                deleted = user.deleted_at
                if deleted.tzinfo is None:
                    deleted = deleted.replace(tzinfo=timezone.utc)
                if deleted < cutoff:
                    expired.append(user)
            count = len(expired)
            for user in expired:
                s.delete(user)
            s.commit()
            return count

    @staticmethod
    def record_consent(user_id: str, version: str = "1.0") -> bool:
        """记录用户知情同意。"""
        with db_manager.get_session_direct() as s:
            user = s.query(User).filter(User.user_id == user_id).first()
            if user is None:
                return False
            user.consent_at = datetime.now(timezone.utc)
            user.consent_version = version
            user.updated_at = datetime.now(timezone.utc)
            s.commit()
            return True

    @staticmethod
    def has_consented(user_id: str) -> bool:
        """检查用户是否已签署知情同意。"""
        with db_manager.get_session_direct() as s:
            user = s.query(User).filter(User.user_id == user_id).first()
            if user is None:
                return False
            return user.consent_at is not None

    @staticmethod
    def list_users(include_deleted: bool = False) -> List[Dict[str, Any]]:
        """列出用户。"""
        with db_manager.get_session_direct() as s:
            q = s.query(User)
            if not include_deleted:
                q = q.filter(User.status != "deleted")
            return [_user_to_dict(u) for u in q.all()]


def _user_to_dict(user: User) -> Dict[str, Any]:
    return {
        "user_id": user.user_id,
        "display_name": user.display_name,
        "status": user.status,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "updated_at": user.updated_at.isoformat() if user.updated_at else None,
        "deleted_at": user.deleted_at.isoformat() if user.deleted_at else None,
        "consent_at": user.consent_at.isoformat() if user.consent_at else None,
        "consent_version": user.consent_version,
    }
