"""
Gap #17: 速率限制 — slowapi + user_id key

验证行为：
  1. 默认限制触发 429
  2. login/register 受限流
  3. crisis 端点（safety/image 等）不受限
  4. /health 不受限
  5. 响应头含 X-RateLimit-* 字段

注意：conftest.py 先于本模块导入 api.main → modules.rate_limit，
因此无法通过 os.environ 覆盖；此处直接操作 limiter._default_limits。
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
from fastapi.testclient import TestClient
from slowapi.wrappers import LimitGroup

_TEST_LIMIT = "5/minute"


def _configure_limiter_for_test():
    """临时切换全局 limiter 到 5/min 限额；返回恢复函数。

    conftest.py 的 autouse fixture 已负责每测试前重置 storage，
    这里只需更改 _default_limits。
    """
    from modules.rate_limit import limiter

    saved_defaults = limiter._default_limits
    limiter._default_limits = [
        LimitGroup(_TEST_LIMIT, limiter._key_func, None, False, None, None, None, 1, False)
    ]

    def restore():
        limiter._default_limits = saved_defaults

    return restore


def _ensure_user(user_id="test-limiter-001", password="testpass123"):
    """确保测试用户存在，返回 (access_token, user_id)。"""
    from modules.auth_service import AuthService
    from modules.token_service import TokenService
    from modules.user_service import UserService
    from schemas.database import db_manager
    from schemas.database_v2 import User, Credential

    user = UserService.get_user(user_id)
    if user:
        if user["status"] == "deleted":
            UserService.restore(user_id)
    else:
        with db_manager.get_session_direct() as s:
            u = User(user_id=user_id, display_name="rate-limit-test", status="active")
            s.add(u)
            s.commit()

    with db_manager.get_session_direct() as s:
        cred = s.query(Credential).filter(
            Credential.user_id == user_id, Credential.type == "password"
        ).first()
        if not cred:
            hashed = AuthService.hash_password(password)
            cred = Credential(
                user_id=user_id, type="password",
                identifier=f"ratelimit_{user_id}", secret=hashed,
            )
            s.add(cred)
            s.commit()

    return TokenService.issue_access_token(user_id), user_id


class TestRateLimitDefault:
    """默认限额：5/min"""

    @pytest.fixture
    def client(self):
        restore_cb = _configure_limiter_for_test()
        token, uid = _ensure_user()
        from modules.auth_deps import get_current_user_id
        from api.main import app

        async def _fake_user():
            return uid

        app.dependency_overrides[get_current_user_id] = _fake_user

        yield TestClient(app)

        app.dependency_overrides.clear()
        restore_cb()

    def test_blocks_after_exceeding_limit(self, client):
        """请求超过 5 次后应返回 429。"""
        for i in range(5):
            resp = client.get(
                "/api/v1/auth/me",
                headers={"Authorization": f"Bearer test-token-override"},
            )
            assert resp.status_code != 429, f"Unexpected 429 on request {i+1}"

        # 第 6 次应被限流
        resp = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer test-token-override"},
        )
        assert resp.status_code == 429, f"Expected 429, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "请求过于频繁" in data.get("error", "")

    def test_rate_limit_header_present(self, client):
        """响应头应包含 X-RateLimit-* 字段。"""
        resp = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer test-token-override"},
        )
        assert "x-ratelimit-limit" in resp.headers or "X-RateLimit-Limit" in resp.headers


class TestLoginRateLimit:
    """登录接口受限流"""

    def test_login_has_its_own_limit_header(self):
        """登录接口有限流响应头。"""
        restore_cb = _configure_limiter_for_test()
        from api.main import app

        try:
            with TestClient(app) as client:
                resp = client.post(
                    "/api/v1/auth/login",
                    json={"username": "no-such-user", "password": "x"},
                )
                has_limit_header = any(
                    key.lower().startswith("x-ratelimit") for key in resp.headers
                )
                assert has_limit_header, "Login endpoint should have rate limit headers"
        finally:
            restore_cb()


class TestCrisisExemption:
    """Crisis 端点不受限流（即使默认 5/min）"""

    @pytest.fixture
    def client(self):
        restore_cb = _configure_limiter_for_test()
        token, uid = _ensure_user("test-crisis-limiter-001")
        from modules.auth_deps import get_current_user_id
        from api.main import app

        async def _fake_user():
            return uid

        app.dependency_overrides[get_current_user_id] = _fake_user
        yield TestClient(app)
        app.dependency_overrides.clear()
        restore_cb()

    def test_safety_image_not_rate_limited(self, client):
        """safety/image 端点多次请求不会触发 429（crisis 豁免）。"""
        for i in range(10):
            resp = client.post("/api/v1/multimodal/safety/image")
            assert resp.status_code != 429, f"Safety image endpoint got 429 on request {i+1}"

    def test_safety_check_not_rate_limited(self, client):
        """modules/safety/check 端点不受限。"""
        for i in range(10):
            resp = client.post(
                "/api/v1/modules/safety/check",
                json={"text": "test message", "session_id": "x"},
            )
            assert resp.status_code != 429, f"Safety check got 429 on request {i+1}"

    def test_emergency_push_not_rate_limited(self, client):
        """emergency-push 端点不受限。"""
        for i in range(10):
            resp = client.post(
                "/api/v1/multimodal/safety/emergency-push",
                json={"text": "test", "level": "1", "session_id": "x"},
            )
            assert resp.status_code != 429, f"Emergency push got 429 on request {i+1}"


class TestHealthExemption:
    """健康检查不受限流"""

    def test_health_not_rate_limited(self):
        restore_cb = _configure_limiter_for_test()
        from api.main import app
        try:
            with TestClient(app) as client:
                for i in range(10):
                    resp = client.get("/api/v1/health")
                    assert resp.status_code == 200
                    assert resp.status_code != 429
        finally:
            restore_cb()
