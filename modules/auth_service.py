"""
Auth Service — 认证逻辑
=======================

密码哈希（bcrypt）、credential 绑定、注册/登录/改密。
JWT token 签发在 api/routes/auth.py 中调用本模块。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import bcrypt

from schemas.database import db_manager
from schemas.database_v2 import Credential, User


class AuthService:
    """认证相关的静态方法集合。"""

    # ---- 密码哈希 ----

    @staticmethod
    def hash_password(password: str) -> str:
        """对明文密码做 bcrypt 哈希。"""
        return bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt(),
        ).decode("utf-8")

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """验证密码是否匹配。"""
        return bcrypt.checkpw(
            password.encode("utf-8"),
            hashed.encode("utf-8"),
        )

    # ---- Credential 管理 ----

    @staticmethod
    def bind_credential(
        user_id: str,
        cred_type: str,
        identifier: str,
        password: Optional[str] = None,
    ) -> bool:
        """为 user 绑定一个登录凭证。"""
        with db_manager.get_session_direct() as s:
            # 检查唯一约束
            existing = s.query(Credential).filter(
                Credential.type == cred_type,
                Credential.identifier == identifier,
            ).first()
            if existing:
                raise ValueError(f"凭证已存在: {cred_type}:{identifier}")

            secret = AuthService.hash_password(password) if password else None
            cred = Credential(
                user_id=user_id,
                type=cred_type,
                identifier=identifier,
                secret=secret,
            )
            s.add(cred)
            s.commit()
            return True

    @staticmethod
    def _get_password_credential(username: str):
        """获取账密类型的 credential。"""
        with db_manager.get_session_direct() as s:
            return s.query(Credential).filter(
                Credential.type == "password",
                Credential.identifier == username,
            ).first()

    # ---- 注册 / 登录 ----

    @staticmethod
    def register(
        username: str,
        password: str,
        display_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """注册新用户：创建 user + 绑定 password credential。"""
        # 检查用户名唯一性
        with db_manager.get_session_direct() as s:
            existing = s.query(Credential).filter(
                Credential.type == "password",
                Credential.identifier == username,
            ).first()
            if existing:
                raise ValueError(f"用户名已存在: {username}")

            # 创建 user
            user = User(
                display_name=display_name,
                status="active",
            )
            s.add(user)
            s.flush()  # 获得 user.user_id

            # 绑定 credential
            secret = AuthService.hash_password(password)
            cred = Credential(
                user_id=user.user_id,
                type="password",
                identifier=username,
                secret=secret,
            )
            s.add(cred)
            s.commit()
            return {"user_id": user.user_id, "username": username}

    @staticmethod
    def login(username: str, password: str) -> Dict[str, Any]:
        """验证账密，返回 user_id。"""
        with db_manager.get_session_direct() as s:
            cred = s.query(Credential).filter(
                Credential.type == "password",
                Credential.identifier == username,
            ).first()

            if not cred:
                raise ValueError("用户不存在")

            # 检查 user 状态
            user = s.query(User).filter(User.user_id == cred.user_id).first()
            if user and user.status == "deleted":
                raise ValueError("用户已注销")

            # 验证密码
            if not cred.secret:
                raise ValueError("密码错误")
            if not AuthService.verify_password(password, cred.secret):
                raise ValueError("密码错误")

            return {"user_id": cred.user_id, "username": username}

    # ---- 改密 ----

    @staticmethod
    def change_password(user_id: str, old_password: str, new_password: str) -> bool:
        """修改密码：验证旧密码后更新。"""
        with db_manager.get_session_direct() as s:
            cred = s.query(Credential).filter(
                Credential.user_id == user_id,
                Credential.type == "password",
            ).first()

            if not cred:
                raise ValueError("用户不存在或未绑定账密")

            if not cred.secret or not AuthService.verify_password(old_password, cred.secret):
                raise ValueError("原密码错误")

            cred.secret = AuthService.hash_password(new_password)
            s.commit()
            return True
