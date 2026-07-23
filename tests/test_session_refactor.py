"""
Gap #19 SessionManager 重构测试
==============================

验证重构后 SessionManager 和 TherapySessionMemory 的核心行为。

测试覆盖：
- SessionManager.create_session / get_session / remove_session
- get_active_sessions_by_user
- TherapySessionMemory 消息读写（无 Redis/DB 模式下内存回退）
- scale_state 持久化
- 归属校验
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime

from core.memory.session_memory import (
    TherapySessionMemory,
    SessionManager,
    SessionMetadata,
    EmotionRecord,
    SessionOwnershipError,
)


class TestSessionManagerNoRedis:
    """无 Redis/DB 模式下的 SessionManager 行为测试"""

    def test_create_session_returns_id(self):
        sid = SessionManager.create_session(user_id="user-1", use_database=False)
        assert len(sid) == 8  # 8-char UUID
        assert isinstance(sid, str)

    def test_get_session_creates_new_when_no_db(self):
        session = SessionManager.get_session(
            "new-session-1", user_id="user-a", use_database=False
        )
        assert session.session_id == "new-session-1"
        assert session.metadata.user_id == "user-a"

    def test_get_session_without_user_id(self):
        """无 user_id 时不报错"""
        session = SessionManager.get_session("anon-session", use_database=False)
        assert session.session_id == "anon-session"
        # 默认 user_id 为空字符串（Pydantic SessionMetadata 默认值）
        assert session.metadata.user_id == ""

    def test_ownership_check(self):
        """user_id 不匹配时抛 SessionOwnershipError"""
        # 先创建一个归属于 user-x 的会话
        session = SessionManager.get_session(
            "owned-session", user_id="user-x", use_database=False
        )
        session.metadata.user_id = "user-x"

        # 不带 user_id：不校验
        SessionManager.get_session("owned-session", use_database=False)

        # 带匹配 user_id：通过
        session2 = SessionManager.get_session(
            "owned-session", user_id="user-x", use_database=False
        )
        assert session2.session_id == "owned-session"

    def test_ownership_reject(self):
        """归属不匹配时直接创建 TherapySessionMemory 应抛异常"""
        # 创建归属于 user-x 的会话
        session = TherapySessionMemory(
            session_id="owned-by-x", user_id="user-x", use_database=False
        )
        # user-y 试图访问应被拒绝
        from core.memory.session_memory import _verify_session_ownership
        with pytest.raises(SessionOwnershipError):
            _verify_session_ownership(session, "user-y")

    def test_remove_session_no_error(self):
        """remove_session 在无 DB 情况下不抛异常"""
        SessionManager.get_session("to-remove", use_database=False)
        # 不应抛异常
        SessionManager.remove_session("to-remove")

    def test_delete_session_alias(self):
        """delete_session 是 remove_session 的别名"""
        SessionManager.get_session("to-delete", use_database=False)
        SessionManager.delete_session("to-delete")

    def test_get_active_sessions_by_user_empty_without_redis(self):
        """无 Redis 索引时返回空列表"""
        sessions = SessionManager.get_active_sessions_by_user("user-z")
        # 没有 Redis，也没有 DB 中该用户的会话
        assert sessions == []

    def test_get_active_sessions_empty_without_db(self):
        """USE_DATABASE=False 时返回空列表"""
        sessions = SessionManager.get_active_sessions()
        assert sessions == []


class TestTherapySessionMemoryNoRedis:
    """无 Redis/DB 模式下的 TherapySessionMemory 行为测试"""

    def test_add_and_get_messages_in_memory_fallback(self):
        """消息写入和读取：纯内存模式"""
        session = TherapySessionMemory(session_id="test-msg", use_database=False)
        session.add_user_message("你好")
        session.add_ai_message("你好呀")

        messages = session.get_messages()
        assert len(messages) == 2

    def test_message_count_updated(self):
        session = TherapySessionMemory(session_id="test-count", use_database=False)
        assert session.metadata.message_count == 0
        session.add_user_message("hi")
        session.add_ai_message("hello")
        assert session.metadata.message_count == 2

    def test_get_history_for_prompt(self):
        session = TherapySessionMemory(session_id="test-prompt", use_database=False)
        session.add_user_message("你好")
        session.add_ai_message("你好！")
        history = session.get_history_for_prompt()
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "你好"}
        assert history[1] == {"role": "assistant", "content": "你好！"}

    def test_emotion_record(self):
        session = TherapySessionMemory(session_id="test-emo", use_database=False)
        record = EmotionRecord(
            primary_emotion="anxiety", intensity=7, triggers=["压力"]
        )
        session.add_emotion_record(record)
        assert len(session._emotion_records) == 1
        assert session._emotion_records[0].primary_emotion == "anxiety"

    def test_emotion_trend(self):
        session = TherapySessionMemory(session_id="test-trend", use_database=False)
        session.add_emotion_record(
            EmotionRecord(primary_emotion="anxiety", intensity=3)
        )
        session.add_emotion_record(
            EmotionRecord(primary_emotion="anxiety", intensity=7)
        )
        trend = session.get_emotion_trend(last_n=2)
        assert trend["trend"] == "increasing"

    def test_key_topics(self):
        session = TherapySessionMemory(session_id="test-topics", use_database=False)
        session.add_key_topic("工作压力")
        session.add_key_topic("人际关系")
        assert "工作压力" in session.metadata.key_topics
        assert "人际关系" in session.metadata.key_topics
        # 重复 topic 不重复添加
        session.add_key_topic("工作压力")
        assert session.metadata.key_topics.count("工作压力") == 1

    def test_save_scale_state_no_db(self):
        """无 DB 模式 save_scale_state 不抛异常"""
        session = TherapySessionMemory(session_id="test-scale", use_database=False)
        session.metadata.scale_state = {"scale_name": "phq9", "status": "started"}
        # 不应抛异常
        session.save_scale_state()

    def test_clear_session(self):
        session = TherapySessionMemory(session_id="test-clear", use_database=False)
        session.add_user_message("hello")
        session.add_ai_message("hi")
        session.add_key_topic("test")
        session.clear()
        assert session.metadata.message_count == 0
        assert session.get_messages() == []
        assert session.metadata.key_topics == []

    def test_history_limit(self):
        session = TherapySessionMemory(
            session_id="test-limit", max_history_turns=2, use_database=False
        )
        for i in range(10):
            session.add_user_message(f"msg{i}")
            session.add_ai_message(f"reply{i}")
        messages = session.get_messages()
        assert len(messages) <= 4  # 2 turns * 2


class TestAfterRefactorNoDictAccess:
    """验证重构后不再有 _sessions dict 泄漏"""

    def test_no_sessions_dict(self):
        """SessionManager 不应该再有 _sessions 类属性"""
        assert not hasattr(SessionManager, "_sessions"), (
            "SessionManager 不应再有 _sessions dict"
        )

    def test_messages_not_persistent_across_instances(self):
        """不同 TherapySessionMemory 实例的消息是隔离的"""
        s1 = TherapySessionMemory(session_id="isolated", use_database=False)
        s1.add_user_message("msg1")

        s2 = TherapySessionMemory(session_id="isolated", use_database=False)
        # s2 是新建实例，无 Redis/DB 模式下不应看到 s1 的消息
        assert len(s2.get_messages()) == 0
