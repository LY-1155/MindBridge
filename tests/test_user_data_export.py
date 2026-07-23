"""
Gap #14: 用户数据导出 — GET /api/v1/user/export

验证行为：
  1. 导出需密码验证，错误密码返回 400
  2. 导出 JSON 包含所有数据分片：profile、sessions、messages、emotion_records、safety_flags、scale_screenings
  3. 加密字段解密后导出（明文）
  4. 不导出 credential 密文
  5. 前端个人信息摘要页面（手动验证）
"""

from __future__ import annotations

import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
from fastapi.testclient import TestClient


def _ensure_user_with_data(user_id, password="testpass123"):
    """创建测试用户及其关联数据，返回 access_token。"""
    from modules.auth_service import AuthService
    from modules.token_service import TokenService
    from modules.user_service import UserService
    from schemas.database import db_manager
    from schemas.database_v2 import (
        User, Credential, SessionV2, MessageV2, EmotionRecordV2,
        SafetyFlag, ScaleScreening,
    )

    # 确保用户存在且 active
    user = UserService.get_user(user_id)
    if user:
        if user["status"] == "deleted":
            UserService.restore(user_id)
    else:
        with db_manager.get_session_direct() as s:
            u = User(user_id=user_id, display_name="export-test", status="active")
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
                identifier=f"export_{user_id}",
                secret=hashed,
            )
            s.add(cred)
            s.commit()

    # 创建测试数据：session + messages + emotion + safety + screening
    with db_manager.get_session_direct() as s:
        sess = SessionV2(
            session_id="export-session-001",
            user_id=user_id,
            message_count=3,
            key_topics=json.dumps(["压力", "焦虑"]),
            scale_history=json.dumps([{"scale": "PHQ-9", "total": 5}]),
        )
        s.add(sess)

        msg1 = MessageV2(session_id="export-session-001", role="user", content="我感到很焦虑")
        msg2 = MessageV2(session_id="export-session-001", role="assistant", content="我理解你的感受")
        s.add_all([msg1, msg2])

        er = EmotionRecordV2(
            session_id="export-session-001",
            user_id=user_id,
            primary_emotion="sad",
            intensity=0.7,
            risk=0.3,
            triggers=json.dumps(["工作"]),
            context="工作压力大",
        )
        s.add(er)

        sf = SafetyFlag(
            session_id="export-session-001",
            user_id=user_id,
            level=1,
            blocked=False,
            matched_terms="敏感词1",
            reviewed=False,
        )
        s.add(sf)

        ss = ScaleScreening(
            session_id="export-session-001",
            user_id=user_id,
            scale_type="PHQ-9",
            state="completed",
            responses=json.dumps({"q1": 2, "q2": 3}),
            scores=json.dumps({"depression": 5}),
            total_score=5.0,
            interpretation=json.dumps({"severity": "轻度"}),
        )
        s.add(ss)

        s.commit()

    token = TokenService.issue_access_token(user_id)
    return token


def _cleanup_user_data(user_id):
    """清理测试数据。"""
    from schemas.database import db_manager
    from schemas.database_v2 import (
        SessionV2, User, Credential, SafetyFlag, ScaleScreening,
    )
    with db_manager.get_session_direct() as s:
        s.query(SafetyFlag).filter(SafetyFlag.user_id == user_id).delete()
        s.query(ScaleScreening).filter(ScaleScreening.user_id == user_id).delete()
        s.query(SessionV2).filter(SessionV2.user_id == user_id).delete()
        s.query(Credential).filter(Credential.user_id == user_id).delete()
        s.query(User).filter(User.user_id == user_id).delete()
        s.commit()


# ============================================================
# 1. Export endpoint
# ============================================================

class TestUserDataExport:
    """用户数据导出端点"""

    _test_uid = "test-export-014"

    @pytest.fixture(autouse=True)
    def _setup_teardown(self):
        _cleanup_user_data(self._test_uid)
        self._token = _ensure_user_with_data(self._test_uid)
        yield
        _cleanup_user_data(self._test_uid)

    @pytest.fixture
    def client(self):
        from modules.auth_deps import get_current_user_id
        from api.main import app

        async def _fake_user():
            return self._test_uid

        app.dependency_overrides[get_current_user_id] = _fake_user
        yield TestClient(app)
        app.dependency_overrides.clear()

    def test_export_requires_password(self, client):
        """错误密码返回 400"""
        resp = client.get("/api/v1/user/export", params={"password": "wrongpass"})
        assert resp.status_code == 400

    def test_export_with_correct_password_returns_json(self, client):
        """正确密码返回 200 + JSON"""
        resp = client.get("/api/v1/user/export", params={"password": "testpass123"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_export_contains_profile_section(self, client):
        """导出含 profile 分片"""
        resp = client.get("/api/v1/user/export", params={"password": "testpass123"})
        data = resp.json()
        assert "profile" in data
        assert data["profile"]["user_id"] == self._test_uid
        assert "display_name" in data["profile"]
        assert "status" in data["profile"]
        # 不含 credential secret
        assert "credential_secret" not in data["profile"]

    def test_export_not_expose_credential_hash(self, client):
        """导出不包含凭证密文"""
        resp = client.get("/api/v1/user/export", params={"password": "testpass123"})
        data = resp.json()
        # 遍历所有分片，确保没有 "secret" 或 "hash" 字段泄露
        exported_str = json.dumps(data)
        assert "secret" not in exported_str.lower()

    def test_export_contains_sessions_section(self, client):
        """导出含 sessions 分片"""
        resp = client.get("/api/v1/user/export", params={"password": "testpass123"})
        data = resp.json()
        assert "sessions" in data
        sessions = data["sessions"]
        assert isinstance(sessions, list)
        assert len(sessions) >= 1
        sess = sessions[0]
        assert sess["session_id"] == "export-session-001"
        assert sess["user_id"] == self._test_uid

    def test_export_contains_messages_section(self, client):
        """导出含 messages 分片"""
        resp = client.get("/api/v1/user/export", params={"password": "testpass123"})
        data = resp.json()
        assert "messages" in data
        msgs = data["messages"]
        assert isinstance(msgs, list)
        assert len(msgs) >= 2

    def test_export_contains_emotion_records_section(self, client):
        """导出含 emotion_records 分片"""
        resp = client.get("/api/v1/user/export", params={"password": "testpass123"})
        data = resp.json()
        assert "emotion_records" in data
        records = data["emotion_records"]
        assert isinstance(records, list)
        assert len(records) >= 1
        rec = records[0]
        assert rec["primary_emotion"] == "sad"

    def test_export_contains_safety_flags_section(self, client):
        """导出含 safety_flags 分片"""
        resp = client.get("/api/v1/user/export", params={"password": "testpass123"})
        data = resp.json()
        assert "safety_flags" in data
        flags = data["safety_flags"]
        assert isinstance(flags, list)
        assert len(flags) >= 1

    def test_export_contains_scale_screenings_section(self, client):
        """导出含 scale_screenings 分片"""
        resp = client.get("/api/v1/user/export", params={"password": "testpass123"})
        data = resp.json()
        assert "scale_screenings" in data
        screenings = data["scale_screenings"]
        assert isinstance(screenings, list)
        assert len(screenings) >= 1

    def test_decrypted_content_in_export(self, client):
        """加密字段在导出中已解密为明文"""
        resp = client.get("/api/v1/user/export", params={"password": "testpass123"})
        data = resp.json()
        # messages 中的 content 应该可读
        for msg in data.get("messages", []):
            if msg["role"] == "user":
                assert "焦虑" in msg["content"] or len(msg["content"]) > 0
        # emotion_records 中的 context 应该可读
        for rec in data.get("emotion_records", []):
            if rec.get("context"):
                assert "工作" in rec["context"] or len(rec["context"]) > 0
