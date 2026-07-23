"""
TDD #5: 管线/聊天/模块 端点接入认证测试

验证所有业务端点拒绝未认证请求，放行已认证请求。
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


class TestHealthRemainsPublic:
    """公开端点不受认证影响。"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_ping_returns_200(self, client):
        resp = client.get("/ping")
        assert resp.status_code == 200


class TestChatEndpoints:
    """聊天端点接入认证。"""

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

    @pytest.fixture
    def access_token(self, client):
        from modules.auth_service import AuthService
        import uuid
        uname = f"authtest_{uuid.uuid4().hex[:8]}"
        AuthService.register(uname, "testpw123")
        resp = client.post("/api/v1/auth/login", json={
            "username": uname, "password": "testpw123",
        })
        assert resp.status_code == 200, resp.text
        return resp.json()["access_token"]

    def test_chat_returns_401_without_token(self, client):
        resp = client.post("/api/v1/chat", json={
            "message": "hello", "session_id": "nonexistent",
        })
        assert resp.status_code == 401

    def test_list_sessions_returns_401_without_token(self, client):
        resp = client.get("/api/v1/sessions")
        assert resp.status_code == 401

    def test_chat_returns_200_with_valid_token(self, client, access_token):
        resp = client.post("/api/v1/chat", json={
            "message": "hello",
        }, headers={"Authorization": f"Bearer {access_token}"})
        assert resp.status_code == 200


class TestPipelineEndpoints:
    """管线端点接入认证。"""

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

    @pytest.fixture
    def access_token(self, client):
        from modules.auth_service import AuthService
        import uuid
        uname = f"pipelinetest_{uuid.uuid4().hex[:8]}"
        AuthService.register(uname, "testpw123")
        resp = client.post("/api/v1/auth/login", json={
            "username": uname, "password": "testpw123",
        })
        return resp.json()["access_token"]

    def test_pipeline_run_returns_401_without_token(self, client):
        resp = client.post("/api/v1/pipeline/run", json={
            "text": "test", "session_id": "nonexistent",
        })
        assert resp.status_code == 401


class TestParallelModuleEndpoints:
    """独立模块端点接入认证。"""

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

    def test_safety_check_returns_401_without_token(self, client):
        resp = client.post("/api/v1/modules/safety/check", json={
            "text": "test",
        })
        assert resp.status_code == 401

    def test_emotion_analyze_returns_401_without_token(self, client):
        resp = client.post("/api/v1/modules/emotion/analyze", json={
            "text": "test",
        })
        assert resp.status_code == 401
