"""
TDD #4: Session 归属校验测试

验证用户只能访问/删除自己的 session，不能越权。
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(_ROOT))

import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture(autouse=True)
def _ensure_jwt_secret():
    if not os.environ.get("JWT_SECRET"):
        os.environ["JWT_SECRET"] = "test-jwt-secret-key-for-testing-only"


class TestSessionOwnership:
    """Session 归属校验：不同用户不能跨权访问。"""

    @pytest.fixture(scope="class", autouse=True)
    def _migrate(self):
        from alembic.config import Config
        from alembic import command
        ini_path = os.path.join(_ROOT, "alembic.ini")
        if os.path.exists(ini_path):
            cfg = Config(ini_path)
            try:
                command.downgrade(cfg, "base")
            except Exception:
                pass
            command.upgrade(cfg, "head")
        yield

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def _register_and_login(self, client, username: str) -> str:
        """注册 + 登录，返回 access_token。"""
        client.post("/api/v1/auth/register", json={
            "username": username, "password": "test123456",
        })
        resp = client.post("/api/v1/auth/login", json={
            "username": username, "password": "test123456",
        })
        assert resp.status_code == 200, resp.text
        return resp.json()["access_token"]

    def test_same_user_can_access_own_session(self, client):
        """用户 A 创建 session → 用户 A 访问 → 200。"""
        import uuid
        uname = f"owner_{uuid.uuid4().hex[:6]}"
        token_a = self._register_and_login(client, uname)

        # 用户 A 发消息创建 session
        resp = client.post("/api/v1/chat", json={
            "message": "你好",
        }, headers={"Authorization": f"Bearer {token_a}"})
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]

        # 用户 A 查询自己的 session
        resp = client.get(f"/api/v1/session/{session_id}",
                          headers={"Authorization": f"Bearer {token_a}"})
        assert resp.status_code == 200

    def test_different_user_cannot_access_others_session(self, client):
        """用户 A 创建 session → 用户 B 访问 → 403。"""
        import uuid
        uname_a = f"alice_{uuid.uuid4().hex[:4]}"
        uname_b = f"bob_{uuid.uuid4().hex[:4]}"
        token_a = self._register_and_login(client, uname_a)
        token_b = self._register_and_login(client, uname_b)

        # 用户 A 创建 session
        resp = client.post("/api/v1/chat", json={
            "message": "用户A的私密对话",
        }, headers={"Authorization": f"Bearer {token_a}"})
        assert resp.status_code == 200
        session_a = resp.json()["session_id"]

        # 用户 B 尝试访问 A 的 session
        resp = client.get(f"/api/v1/session/{session_a}",
                          headers={"Authorization": f"Bearer {token_b}"})
        assert resp.status_code == 403

    def test_different_user_cannot_delete_others_session(self, client):
        """用户 A 创建 session → 用户 B 删除 → 403。"""
        import uuid
        uname_a = f"alice2_{uuid.uuid4().hex[:4]}"
        uname_b = f"bob2_{uuid.uuid4().hex[:4]}"
        token_a = self._register_and_login(client, uname_a)
        token_b = self._register_and_login(client, uname_b)

        # 用户 A 创建 session
        resp = client.post("/api/v1/chat", json={
            "message": "A的会话",
        }, headers={"Authorization": f"Bearer {token_a}"})
        assert resp.status_code == 200
        session_a = resp.json()["session_id"]

        # 用户 B 尝试删除 A 的 session
        resp = client.delete(f"/api/v1/session/{session_a}",
                             headers={"Authorization": f"Bearer {token_b}"})
        assert resp.status_code == 403

        # 确认 A 的 session 还在
        resp = client.get(f"/api/v1/session/{session_a}",
                          headers={"Authorization": f"Bearer {token_a}"})
        assert resp.status_code == 200
