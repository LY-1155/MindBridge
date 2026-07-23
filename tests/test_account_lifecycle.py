"""
Gap #13: 账号软删除 + 30 天后悔期

验证行为：
  1. soft_delete 标记 status=deleted + 记录 deleted_at
  2. 注销后 token 立即失效（middleware 检查 user status）
  3. 30 天内可恢复账号
  4. 超过 30 天无法恢复
  5. purge_expired_accounts 物理删除过期账号
  6. DELETE /auth/account 端点（需密码验证）
  7. POST /auth/account/restore 端点
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
from fastapi.testclient import TestClient


def _ensure_user(user_id, password="testpass123"):
    """确保测试用户存在，返回 (user_id, access_token)。"""
    from modules.auth_service import AuthService
    from modules.token_service import TokenService
    from modules.user_service import UserService
    from schemas.database import db_manager
    from schemas.database_v2 import User, Credential

    user = UserService.get_user(user_id)
    if user:
        # 确保 active
        if user["status"] == "deleted":
            UserService.restore(user_id)
    else:
        with db_manager.get_session_direct() as s:
            u = User(user_id=user_id, display_name="lifecycle-test", status="active")
            s.add(u)
            s.commit()

    # 确保 credential 存在
    with db_manager.get_session_direct() as s:
        existing_cred = s.query(Credential).filter(
            Credential.user_id == user_id,
            Credential.type == "password",
        ).first()
        if not existing_cred:
            hashed = AuthService.hash_password(password)
            cred = Credential(
                user_id=user_id,
                type="password",
                identifier=f"lifecycle_{user_id}",
                secret=hashed,
            )
            s.add(cred)
            s.commit()

    token = TokenService.issue_access_token(user_id)
    return token


# ============================================================
# 1. UserService soft_delete / restore / purge
# ============================================================

class TestSoftDelete:
    """soft_delete 正确标记用户状态"""

    _test_uid = "test-soft-delete-001"

    @pytest.fixture(autouse=True)
    def _setup(self):
        from modules.user_service import UserService
        from schemas.database import db_manager
        from schemas.database_v2 import User

        user = UserService.get_user(self._test_uid)
        if user is None:
            with db_manager.get_session_direct() as s:
                u = User(user_id=self._test_uid, display_name="lifecycle-test", status="active")
                s.add(u)
                s.commit()
        elif user["status"] == "deleted":
            UserService.restore(self._test_uid)
        yield

    def test_soft_delete_sets_status_and_timestamp(self):
        from modules.user_service import UserService
        ok = UserService.soft_delete(self._test_uid)
        assert ok is True

        user = UserService.get_user(self._test_uid)
        assert user["status"] == "deleted"
        assert user["deleted_at"] is not None

    def test_restore_within_grace_period_succeeds(self):
        from modules.user_service import UserService
        UserService.soft_delete(self._test_uid)
        ok = UserService.restore(self._test_uid)
        assert ok is True

        user = UserService.get_user(self._test_uid)
        assert user["status"] == "active"
        assert user["deleted_at"] is None

    def test_restore_outside_grace_period_fails(self):
        from modules.user_service import UserService, GRACE_PERIOD_DAYS
        from schemas.database import db_manager
        from schemas.database_v2 import User

        UserService.soft_delete(self._test_uid)
        # 手动将 deleted_at 推到 31 天前
        with db_manager.get_session_direct() as s:
            u = s.query(User).filter(User.user_id == self._test_uid).first()
            u.deleted_at = datetime.now(timezone.utc) - timedelta(days=GRACE_PERIOD_DAYS + 1)
            s.commit()

        ok = UserService.restore(self._test_uid)
        assert ok is False

    def test_is_within_grace_period_returns_correct_values(self):
        from modules.user_service import UserService, GRACE_PERIOD_DAYS
        from schemas.database import db_manager
        from schemas.database_v2 import User

        UserService.soft_delete(self._test_uid)
        # 刚删除 — 在后悔期内
        in_grace, error = UserService.is_within_grace_period(self._test_uid)
        assert in_grace is True
        assert error is None

        # 过期
        with db_manager.get_session_direct() as s:
            u = s.query(User).filter(User.user_id == self._test_uid).first()
            u.deleted_at = datetime.now(timezone.utc) - timedelta(days=GRACE_PERIOD_DAYS + 1)
            s.commit()

        in_grace, error = UserService.is_within_grace_period(self._test_uid)
        assert in_grace is False
        assert "后悔期" in error

    def test_purge_expired_accounts_deletes_old_accounts(self):
        from modules.user_service import UserService, GRACE_PERIOD_DAYS
        from schemas.database import db_manager
        from schemas.database_v2 import User

        UserService.soft_delete(self._test_uid)
        # 过期
        with db_manager.get_session_direct() as s:
            u = s.query(User).filter(User.user_id == self._test_uid).first()
            u.deleted_at = datetime.now(timezone.utc) - timedelta(days=GRACE_PERIOD_DAYS + 1)
            s.commit()

        purged = UserService.purge_expired_accounts()
        assert purged >= 1

        # 用户应不存在
        from schemas.database import db_manager
        with db_manager.get_session_direct() as s:
            u = s.query(User).filter(User.user_id == self._test_uid).first()
            assert u is None


# ============================================================
# 2. Token 吊销
# ============================================================

class TestTokenRevocation:
    """注销后 token 立即失效"""

    _test_uid = "test-token-revoke-002"

    @pytest.fixture
    def client(self):
        token = _ensure_user(self._test_uid)
        from api.main import app
        from modules.auth_deps import get_current_user_id

        # 使用真实 token（不 override auth）
        # 但 DELETE account 端点需要 override 因为我们用 token 验证
        # 先保存 token，测试时手动带 header

        # Clear old overrides
        app.dependency_overrides.clear()
        return TestClient(app), token

    def test_active_user_token_works(self, client):
        tc, token = client
        resp = tc.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_deleted_user_token_rejected_at_middleware(self, client):
        from modules.user_service import UserService

        tc, token = client
        UserService.soft_delete(self._test_uid)

        resp = tc.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        # token 应被 middleware 拒绝
        assert resp.status_code == 401

    def test_deleted_user_cannot_refresh(self, client):
        from modules.user_service import UserService
        from modules.token_service import TokenService

        tc, _ = client
        refresh_token = TokenService.issue_refresh_token(self._test_uid)
        UserService.soft_delete(self._test_uid)

        resp = tc.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        assert resp.status_code == 401


# ============================================================
# 3. DELETE /auth/account 端点
# ============================================================

class TestDeleteAccountEndpoint:
    """DELETE /auth/account 完整流程"""

    _test_uid = "test-delete-endpoint-003"

    @pytest.fixture
    def client(self):
        _ensure_user(self._test_uid)
        from modules.auth_deps import get_current_user_id
        from api.main import app

        async def _fake_user():
            return self._test_uid

        app.dependency_overrides[get_current_user_id] = _fake_user
        yield TestClient(app)
        app.dependency_overrides.clear()

    def test_delete_account_requires_password(self, client):
        resp = client.request(
            "DELETE", "/api/v1/auth/account",
            json={"password": "wrongpass"},
        )
        assert resp.status_code == 400

    def test_delete_account_with_correct_password(self, client):
        resp = client.request(
            "DELETE", "/api/v1/auth/account",
            json={"password": "testpass123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"

    def test_restore_after_delete_succeeds(self, client):
        # 先删除
        client.request("DELETE", "/api/v1/auth/account", json={"password": "testpass123"})
        # 再恢复
        resp = client.post("/api/v1/auth/account/restore")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "active"

    def test_restore_active_user_returns_error(self, client):
        resp = client.post("/api/v1/auth/account/restore")
        assert resp.status_code == 400  # 未处于删除状态
