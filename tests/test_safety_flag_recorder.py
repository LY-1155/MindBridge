"""
TDD #6: 安全标记累积升级 + 人审接口测试

验证 SafetyFlagRecorder 的持久化、滑动窗口累积、自动升级逻辑。
"""
from __future__ import annotations

import os
import sys
import uuid

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(_ROOT))

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(autouse=True)
def _ensure_jwt_secret():
    if not os.environ.get("JWT_SECRET"):
        os.environ["JWT_SECRET"] = "test-jwt-secret-key-for-testing-only"
    if not os.environ.get("JWT_ALGORITHM"):
        os.environ["JWT_ALGORITHM"] = "HS256"


# ── DB migration fixture ────────────────────────────────────────

@pytest.fixture(scope="class", autouse=True)
def _migrate():
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


# ── Helper ─────────────────────────────────────────────────────

def _make_user() -> str:
    """创建一个真实用户（绕过 safety_flags FK 约束），返回 user_id。"""
    from modules.user_service import UserService
    return UserService.create_user(display_name="test_reviewer")


# ── Unit: SafetyFlagRecorder 累积逻辑 ──────────────────────────

class TestSafetyFlagRecorder:
    """直接测试 SafetyFlagRecorder 的持久化 + 累积逻辑。"""

    @pytest.fixture
    def user_id(self):
        return _make_user()

    def test_record_persists_flag(self, user_id):
        from modules.safety.flag_recorder import SafetyFlagRecorder

        recorder = SafetyFlagRecorder()
        uid = user_id
        sid = f"s_{uuid.uuid4().hex[:8]}"

        flag = recorder.record(
            user_id=uid,
            session_id=sid,
            level=1,
            blocked=False,
            matched_terms=["自杀"],
        )
        assert flag.id is not None
        assert flag.user_id == uid
        assert flag.level == 1
        assert flag.blocked is False

    def test_evaluate_level_zero_passes_through(self, user_id):
        from modules.safety.flag_recorder import SafetyFlagRecorder

        recorder = SafetyFlagRecorder()
        uid = user_id
        sid = f"s_{uuid.uuid4().hex[:8]}"

        result = recorder.evaluate(
            user_id=uid, session_id=sid,
            level=0, blocked=False, matched_terms=[],
        )
        assert result["recorded"] is True
        assert result["original_level"] == 0
        assert result["final_level"] == 0
        assert result["escalated"] is False
        assert result["final_blocked"] is False

    def test_evaluate_level_two_passes_through_directly(self, user_id):
        from modules.safety.flag_recorder import SafetyFlagRecorder

        recorder = SafetyFlagRecorder()
        uid = user_id
        sid = f"s_{uuid.uuid4().hex[:8]}"

        result = recorder.evaluate(
            user_id=uid, session_id=sid,
            level=2, blocked=True, matched_terms=["暴力威胁"],
        )
        assert result["recorded"] is True
        assert result["original_level"] == 2
        assert result["final_level"] == 2
        assert result["escalated"] is False
        assert result["final_blocked"] is True

    def test_single_level_one_no_escalation(self, user_id):
        from modules.safety.flag_recorder import SafetyFlagRecorder

        recorder = SafetyFlagRecorder()
        uid = user_id
        sid = f"s_{uuid.uuid4().hex[:8]}"

        result = recorder.evaluate(
            user_id=uid, session_id=sid,
            level=1, blocked=False, matched_terms=["焦虑"],
        )
        assert result["recorded"] is True
        assert result["original_level"] == 1
        assert result["final_level"] == 1
        assert result["escalated"] is False
        assert result["final_blocked"] is False
        assert result["recent_warnings"] >= 1

    def test_count_recent_warnings_excludes_old_entries(self, user_id):
        from modules.safety.flag_recorder import SafetyFlagRecorder
        from schemas.database_v2 import SafetyFlag
        from schemas.database import db_manager
        from datetime import datetime, timedelta

        recorder = SafetyFlagRecorder(window_minutes=5, threshold=3)
        uid = user_id
        sid = f"s_{uuid.uuid4().hex[:8]}"

        # 插入一条窗口外的旧标记（直接通过 DB 操作，绕过 record 的 NOW()）
        with db_manager.get_session_direct() as s:
            old = SafetyFlag(
                user_id=uid,
                session_id=sid,
                level=1,
                blocked=False,
                matched_terms='["旧标记"]',
            )
            old.created_at = datetime.utcnow() - timedelta(minutes=10)
            s.add(old)
            s.commit()

        # 窗口内只有 0 条 → 不会升级
        assert recorder.count_recent_warnings(uid) == 0

    def test_three_warnings_in_window_escalates(self, user_id):
        from modules.safety.flag_recorder import SafetyFlagRecorder

        recorder = SafetyFlagRecorder(window_minutes=60, threshold=3)
        uid = user_id

        # 前两次 level=1 → 不升级
        for i in range(2):
            result = recorder.evaluate(
                user_id=uid, session_id=f"s_{i}",
                level=1, blocked=False,
                matched_terms=[f"警告{i}"],
            )
            assert result["escalated"] is False, f"第{i}次不应升级"

        # 第三次 level=1 → 触发升级
        result = recorder.evaluate(
            user_id=uid, session_id=f"s_3",
            level=1, blocked=False,
            matched_terms=["警告3"],
        )
        assert result["escalated"] is True
        assert result["final_level"] == 2
        # 收窄后：累积升级是软升级（不设 blocked），交 LLM 评估器/router 二次裁决
        assert result["final_blocked"] is False
        assert result["recent_warnings"] >= 3

    def test_should_escalate_returns_bool(self, user_id):
        from modules.safety.flag_recorder import SafetyFlagRecorder

        recorder = SafetyFlagRecorder(window_minutes=60, threshold=3)
        uid = user_id

        # 没有标记 → False
        assert recorder.should_escalate(uid, "s_x") is False

        # 插入 2 条 → False
        recorder.record(user_id=uid, session_id="s_a", level=1, blocked=False, matched_terms=["a"])
        recorder.record(user_id=uid, session_id="s_b", level=1, blocked=False, matched_terms=["b"])
        assert recorder.should_escalate(uid, "s_c") is False

        # 插入第 3 条 → True
        recorder.record(user_id=uid, session_id="s_c", level=1, blocked=False, matched_terms=["c"])
        assert recorder.should_escalate(uid, "s_d") is True


# ── Unit: 人审接口 ─────────────────────────────────────────────

class TestHumanReview:
    """人审：list_pending_review + mark_reviewed。"""

    @pytest.fixture
    def user_id(self):
        return _make_user()

    def test_list_pending_returns_unreviewed_flags(self, user_id):
        from modules.safety.flag_recorder import SafetyFlagRecorder

        recorder = SafetyFlagRecorder()
        uid = user_id

        recorder.record(user_id=uid, session_id="s_hr_1", level=1, blocked=False, matched_terms=["敏感词A"])
        recorder.record(user_id=uid, session_id="s_hr_2", level=2, blocked=True, matched_terms=["敏感词B"])

        flags = recorder.list_pending_review(user_id=uid)
        assert len(flags) >= 2
        for f in flags:
            assert f["reviewed"] is False
            assert f["user_id"] == uid

    def test_mark_reviewed_sets_fields(self, user_id):
        from modules.safety.flag_recorder import SafetyFlagRecorder

        recorder = SafetyFlagRecorder()
        uid = user_id

        flag = recorder.record(user_id=uid, session_id="s_r", level=1, blocked=False, matched_terms=["test"])
        ok = recorder.mark_reviewed(flag.id, reviewed_by="tester")
        assert ok is True

        # 确认不再出现在待审列表
        pending = recorder.list_pending_review(user_id=uid)
        pending_ids = [f["id"] for f in pending]
        assert flag.id not in pending_ids


# ── Integration: 人审 API 端点 ─────────────────────────────────

class TestHumanReviewAPI:
    """通过 HTTP API 测试人审端点。"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def _login(self, client):
        uname = f"reviewer_{uuid.uuid4().hex[:6]}"
        client.post("/api/v1/auth/register", json={
            "username": uname, "password": "test123456",
        })
        resp = client.post("/api/v1/auth/login", json={
            "username": uname, "password": "test123456",
        })
        return resp.json()["access_token"]

    def test_list_pending_flags_api_returns_list(self, client):
        """GET /api/v1/multimodal/safety/flags/pending 返回待人审标记列表。"""
        token = self._login(client)

        resp = client.get("/api/v1/multimodal/safety/flags/pending",
                          headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_list_pending_flags_filter_by_user(self, client):
        """按 user_id 过滤待人审标记。"""
        token = self._login(client)

        resp = client.get("/api/v1/multimodal/safety/flags/pending?user_id_filter=nonexistent_user_xyz",
                          headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_mark_reviewed_api_returns_success(self, client):
        """POST /api/v1/multimodal/safety/flags/{id}/review 标记已审。"""
        token = self._login(client)

        # 对一个不存在的 id 调用 → 返回 success=False
        resp = client.post("/api/v1/multimodal/safety/flags/99999/review",
                           json={"reviewed_by": "tester"},
                           headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["success"] is False
