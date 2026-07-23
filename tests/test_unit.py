import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory.session_memory import TherapySessionMemory, SessionManager, EmotionRecord


class TestSessionMemory:
    def test_session_creation(self):
        session = TherapySessionMemory(session_id="test-1", use_database=False)
        assert session.session_id == "test-1"
        assert session.metadata.message_count == 0

    def test_add_messages(self):
        session = TherapySessionMemory(session_id="test-2", use_database=False)
        session.add_user_message("你好")
        session.add_ai_message("你好！有什么可以帮助你的？")
        assert session.metadata.message_count == 2
        assert len(session.get_messages()) == 2

    def test_emotion_record(self):
        session = TherapySessionMemory(session_id="test-3", use_database=False)
        record = EmotionRecord(
            primary_emotion="anxiety",
            intensity=7,
            triggers=["工作压力"]
        )
        session.add_emotion_record(record)
        assert len(session._emotion_records) == 1
        trend = session.get_emotion_trend()
        assert trend["average_intensity"] == 7

    def test_history_limit(self):
        session = TherapySessionMemory(
            session_id="test-4", max_history_turns=2, use_database=False,
        )
        for i in range(10):
            session.add_user_message(f"消息{i}")
            session.add_ai_message(f"回复{i}")
        messages = session.get_messages()
        assert len(messages) <= 4


class TestSessionManager:
    def test_get_session(self):
        session = SessionManager.get_session(
            "manager-test-1", use_database=False,
        )
        assert session.session_id == "manager-test-1"

    def test_remove_session(self):
        SessionManager.get_session("manager-test-2", use_database=False)
        # remove_session 不应抛异常
        SessionManager.remove_session("manager-test-2")
        # 无 Redis/DB 模式下 remove 后 get_session 仍可创建新实例
        session = SessionManager.get_session("manager-test-2", use_database=False)
        assert session.session_id == "manager-test-2"


