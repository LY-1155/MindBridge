"""
Gap #12: 知情同意 API — 记录和查询同意状态

验证行为：
  1. GET /consent-status 返回 consented 状态
  2. POST /consent 记录同意
  3. 记录后 consented 变为 true
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest


class TestConsentAPI:
    """知情同意端点"""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from modules.auth_deps import get_current_user_id
        from api.main import app
        from schemas.database import db_manager
        from schemas.database_v2 import User

        # 确保测试用户存在（直接 upsert）
        _test_uid = "test-user-gap12-consent"
        with db_manager.get_session_direct() as s:
            u = s.query(User).filter(User.user_id == _test_uid).first()
            if u is None:
                u = User(user_id=_test_uid, display_name="gap12-test", status="active")
                s.add(u)
                s.commit()

        async def _fake_user():
            return _test_uid

        app.dependency_overrides[get_current_user_id] = _fake_user
        yield
        app.dependency_overrides.clear()

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)

    def test_consent_status_returns_valid_json(self, client):
        """consent-status 返回有效 JSON"""
        resp = client.get("/api/v1/auth/consent-status")
        assert resp.status_code == 200
        data = resp.json()
        assert "consented" in data
        assert "consent_at" in data or "consent_version" in data

    def test_consent_record_returns_ok(self, client):
        """POST /consent 返回 status ok"""
        resp = client.post(
            "/api/v1/auth/consent",
            json={"version": "1.0"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "ok"
        assert data.get("version") == "1.0"

    def test_consent_persisted_after_record(self, client):
        """记录同意后 consent-status 返回 consented=true"""
        # 先记录
        client.post("/api/v1/auth/consent", json={"version": "1.0"})
        # 再查询
        resp = client.get("/api/v1/auth/consent-status")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("consented") is True
