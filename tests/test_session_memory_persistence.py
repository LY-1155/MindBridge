"""
会话蒸馏状态持久化回归测试（P0 修复）
=====================================

背景：SessionMetadata 的蒸馏临床状态（phase / 工作假设 / 家庭成员 /
已探测维度 / SCID 跨轮累积 / SCID 访谈状态机 / 危机状态机 / scale_state）
此前从不落库——MySQL sessions 表无对应列，Redis/MySQL 存储层只读写
key_topics/scale_state 等少数字段。USE_DATABASE=true 下每次请求新建实例
从存储重载时，这些状态全部归默认值：医生模式 phase 重置、SCID 累积清零、
危机状态机回到 NONE（安全风险）。

修复后：整份状态以 state_json（MySQL，AES 加密）+ Redis state 字段
（与 MySQL 同一序列化器）落库。本测试验证蒸馏状态能跨实例
（模拟"新请求新实例"）完整恢复。

需要真实 MySQL（prism_db_test；conftest 自动建库 + alembic upgrade head）。
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

from config.settings import settings

_HYPOTHESIS = "孩子的不上学可能承担了转移父母冲突的功能"
_SCID_STATE = {"module": "MDD", "status": "active", "step": "gate"}
_SAFETY_STATE = {"status": "PROBING", "probe_count": 1, "denial_mark": False}
_SCALE_STATE = {"scale_name": "PHQ-9", "status": "in_progress", "step": 3}
_USER_ID = "persist-user"


def _ensure_user_exists(user_id: str) -> None:
    """确保测试用户在 users 表存在。

    sessions.user_id 是 NOT NULL 外键 → users.user_id（database_v2.py）。
    生产里 session 必然属于已注册用户，所以测试复刻真实流程：先建 user 再建 session。
    否则 INSERT 触发 FK IntegrityError，被 create_session 的 except 吞掉 → 假阴性。
    """
    from schemas.database import db_manager
    from schemas.database_v2 import User

    with db_manager.get_session_direct() as s:
        if s.query(User).filter(User.user_id == user_id).first() is None:
            s.add(User(user_id=user_id, display_name="persist-test", status="active"))
            s.commit()


@pytest.fixture
def use_database(monkeypatch):
    """强制 USE_DATABASE=true，走 Redis/MySQL 持久化路径（测试库 prism_db_test）。"""
    monkeypatch.setattr(settings, "USE_DATABASE", True)


@pytest.fixture
def require_mysql():
    """MySQL 不可用时跳过。

    与 conftest 建库逻辑一致。若无此护栏，MySQL 挂了会静默走
    _memory_sessions 内存兜底导致"假绿"，掩盖持久化的真实状态。
    """
    try:
        import pymysql
        conn = pymysql.connect(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            connect_timeout=3,
        )
        conn.close()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"MySQL 不可用，跳过 DB 持久化测试: {e}")


@pytest.fixture
def persist_user(require_mysql):
    """确保测试用户存在后返回 user_id（create_session 前置条件）。"""
    _ensure_user_exists(_USER_ID)
    return _USER_ID


def _set_distilled_state(session) -> None:
    """写入全部蒸馏临床字段（每个 update 都触发 _save_to_database 持久化）。"""
    session.update_phase("interpret")
    session.update_hypothesis(_HYPOTHESIS)
    session.add_family_member("妈妈", "焦虑型")
    session.add_probed_dimension("睡眠")
    # scale_state 无 update 方法，量表单测直接赋值；后续 update 触发持久化
    session.metadata.scale_state = _SCALE_STATE
    session.update_scid_flags({"MDD": {"criteria_met": ["sleep", "anhedonia"], "count": 2}})
    session.update_scid_interview_state(_SCID_STATE)
    session.update_safety_state(_SAFETY_STATE)


def _assert_distilled_state(session) -> None:
    """断言蒸馏状态完整恢复。"""
    m = session.metadata
    assert m.phase == "interpret"
    assert m.working_hypothesis == _HYPOTHESIS
    assert m.family_members and m.family_members[0]["role"] == "妈妈"
    assert m.family_members[0]["label"] == "焦虑型"
    assert "睡眠" in m.probed_dimensions
    assert m.scale_state == _SCALE_STATE
    assert m.scid_flags["MDD"]["count"] == 2
    assert sorted(m.scid_flags["MDD"]["criteria_met"]) == ["anhedonia", "sleep"]
    assert m.scid_interview_state == _SCID_STATE
    assert m.safety_state == _SAFETY_STATE


@pytest.mark.integration
class TestDistilledStatePersistence:
    """蒸馏状态跨实例恢复（DB 版回归测试）。"""

    def test_full_roundtrip_via_storage(self, use_database, persist_user):
        """新会话写入全部蒸馏字段 → 新实例从存储重载 → 全部恢复。"""
        from core.memory.db_storage import DatabaseStorage
        from core.memory.session_memory import SessionManager

        session_id = SessionManager.create_session(user_id=persist_user)
        try:
            # 确认真的走了 DB 路径（而非 _memory_sessions 内存兜底）
            assert DatabaseStorage.session_exists(session_id) is True

            session = SessionManager.get_session(session_id, user_id=persist_user)
            _set_distilled_state(session)
            session.add_user_message("最近睡不好，总是半夜醒")
            session.add_ai_message("听起来睡眠困扰比较明显")

            # 模拟"下一次请求"：DB 模式下 get_session 必然创建新实例并重载
            session2 = SessionManager.get_session(session_id, user_id=persist_user)
            assert session2 is not session
            _assert_distilled_state(session2)
            assert session2.metadata.message_count == 2
        finally:
            SessionManager.remove_session(session_id)

    def test_mysql_only_roundtrip(self, use_database, persist_user, monkeypatch):
        """Redis 不可用时强制走 MySQL 路径，验证耐久持久化通道。"""
        import redis
        import core.memory.redis_storage as redis_storage
        from core.memory.db_storage import DatabaseStorage
        from core.memory.session_memory import SessionManager

        def _no_redis(*args, **kwargs):
            raise redis.RedisError("redis disabled for test")

        monkeypatch.setattr(redis_storage, "_get_redis", _no_redis)

        session_id = SessionManager.create_session(user_id=persist_user)
        try:
            assert DatabaseStorage.session_exists(session_id) is True

            session = SessionManager.get_session(session_id, user_id=persist_user)
            _set_distilled_state(session)

            # 新实例：Redis 路径必然失败 → 回退 MySQL 重载
            session2 = SessionManager.get_session(session_id, user_id=persist_user)
            assert session2 is not session
            _assert_distilled_state(session2)
        finally:
            SessionManager.remove_session(session_id)
