"""
TDD #3: JWT Token + Auth 中间件测试

验证 token 签发、验证、过期检测、中间件拦截。
"""
from __future__ import annotations

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(_ROOT))

import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture(autouse=True)
def _ensure_jwt_secret():
    if not os.environ.get("JWT_SECRET"):
        os.environ["JWT_SECRET"] = "test-jwt-secret-key-for-testing-only"


class TestTokenService:
    """JWT token 签发与验证。"""

    def test_issue_access_token(self, _ensure_jwt_secret):
        from modules.token_service import TokenService
        token = TokenService.issue_access_token("u_test_123")
        assert token is not None
        assert len(token) > 20

    def test_issue_refresh_token(self, _ensure_jwt_secret):
        from modules.token_service import TokenService
        token = TokenService.issue_refresh_token("u_test_123")
        assert token is not None

    def test_verify_valid_token(self, _ensure_jwt_secret):
        from modules.token_service import TokenService
        token = TokenService.issue_access_token("u_abc")
        payload = TokenService.verify_token(token)
        assert payload["user_id"] == "u_abc"
        assert payload["type"] == "access"

    def test_verify_refresh_token(self, _ensure_jwt_secret):
        from modules.token_service import TokenService
        token = TokenService.issue_refresh_token("u_xyz")
        payload = TokenService.verify_token(token)
        assert payload["user_id"] == "u_xyz"
        assert payload["type"] == "refresh"

    def test_verify_expired_token_fails(self, _ensure_jwt_secret):
        from modules.token_service import TokenService
        token = TokenService.issue_access_token("u_expired", expires_in_seconds=-1)
        with pytest.raises(ValueError, match="过期"):
            TokenService.verify_token(token)

    def test_verify_tampered_token_fails(self, _ensure_jwt_secret):
        from modules.token_service import TokenService
        token = TokenService.issue_access_token("u_tamper")
        tampered = token[:-5] + "XXXXX"
        with pytest.raises(ValueError):
            TokenService.verify_token(tampered)


class TestAuthMiddleware:
    """FastAPI Depends 注入：从 Bearer header 提取 user_id。"""

    def test_valid_bearer_returns_user_id(self, _ensure_jwt_secret):
        from modules.token_service import TokenService
        from modules.auth_deps import get_current_user_id
        token = TokenService.issue_access_token("u_mid_001")
        # 直接调 dependency 函数模拟中间件行为
        result = get_current_user_id(f"Bearer {token}")
        assert result == "u_mid_001"

    def test_missing_header_raises_401(self, _ensure_jwt_secret):
        from modules.auth_deps import get_current_user_id
        import fastapi
        with pytest.raises(fastapi.HTTPException) as exc:
            get_current_user_id(None)
        assert exc.value.status_code == 401

    def test_malformed_header_raises_401(self, _ensure_jwt_secret):
        from modules.auth_deps import get_current_user_id
        import fastapi
        with pytest.raises(fastapi.HTTPException) as exc:
            get_current_user_id("NotBearer xyz")
        assert exc.value.status_code == 401

    def test_invalid_token_raises_401(self, _ensure_jwt_secret):
        from modules.auth_deps import get_current_user_id
        import fastapi
        with pytest.raises(fastapi.HTTPException) as exc:
            get_current_user_id("Bearer invalid.token.here")
        assert exc.value.status_code == 401


class TestProtectedEndpoint:
    """端到端：注册 → 登录 → 带 token 调受保护端点。"""

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
        if os.path.exists(ini_path):
            cfg = Config(ini_path)
            command.downgrade(cfg, "base")

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_unauthenticated_request_returns_401(self, client):
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code == 401

    def test_authenticated_request_returns_200(self, client):
        # 注册
        resp = client.post("/api/v1/auth/register", json={
            "username": "jwt_test_user", "password": "jwt_test_pw",
        })
        assert resp.status_code == 200, resp.text

        # 登录获取 token
        resp = client.post("/api/v1/auth/login", json={
            "username": "jwt_test_user", "password": "jwt_test_pw",
        })
        assert resp.status_code == 200, resp.text
        tokens = resp.json()
        access_token = tokens["access_token"]

        # 带 token 访问 /me
        resp = client.get("/api/v1/auth/me", headers={
            "Authorization": f"Bearer {access_token}",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] is not None

    def test_refresh_token_gets_new_access(self, client):
        # 注册+登录
        client.post("/api/v1/auth/register", json={
            "username": "refresh_test", "password": "pw1234",
        })
        resp = client.post("/api/v1/auth/login", json={
            "username": "refresh_test", "password": "pw1234",
        })
        tokens = resp.json()

        # 用 refresh token 换新 access
        resp = client.post("/api/v1/auth/refresh", json={
            "refresh_token": tokens["refresh_token"],
        })
        assert resp.status_code == 200
        new_tokens = resp.json()
        assert "access_token" in new_tokens
