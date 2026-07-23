"""
MySQL存储功能测试脚本
===================

测试会话数据是否能正确保存到MySQL数据库。
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.memory.session_memory import SessionManager, TherapySessionMemory, EmotionRecord
from core.memory.db_storage import DatabaseStorage
from config.settings import settings


def test_database_storage():
    """测试数据库存储功能"""
    print("=" * 50)
    print("MySQL存储功能测试")
    print("=" * 50)
    
    print(f"\n配置信息：")
    print(f"  USE_DATABASE: {settings.USE_DATABASE}")
    print(f"  DATABASE_URL: {settings.DATABASE_URL}")
    
    # 测试1：创建会话
    print("\n【测试1】创建新会话...")
    session_id = SessionManager.create_session()
    print(f"  ✓ 会话ID: {session_id}")
    
    # 测试2：获取会话并添加消息
    print("\n【测试2】添加消息...")
    session = SessionManager.get_session(session_id)
    session.add_user_message("你好，我最近感觉很焦虑")
    session.add_ai_message("你好，我理解你的感受。能告诉我更多关于你焦虑的情况吗？")
    session.add_user_message("主要是工作压力太大了")
    print(f"  ✓ 添加了3条消息")
    
    # 测试3：添加情绪记录
    print("\n【测试3】添加情绪记录...")
    emotion = EmotionRecord(
        primary_emotion="anxiety",
        intensity=7,
        triggers=["工作压力"],
        context="用户表达工作压力大"
    )
    session.add_emotion_record(emotion)
    print(f"  ✓ 情绪记录: {emotion.primary_emotion}, 强度: {emotion.intensity}")
    
    # 测试4：添加关键话题
    print("\n【测试4】添加关键话题...")
    session.add_key_topic("工作压力")
    print(f"  ✓ 关键话题: {session.metadata.key_topics}")

    # 测试5：从数据库验证
    print("\n【测试5】从数据库验证数据...")
    
    # 查询会话
    session_data = DatabaseStorage.load_session(session_id)
    if session_data:
        print(f"  ✓ 会话数据: messages={session_data['message_count']}")
    else:
        print("  ✗ 未找到会话数据")
        return False
    
    # 查询消息
    messages = DatabaseStorage.get_messages(session_id)
    print(f"  ✓ 消息数量: {len(messages)}")
    for i, msg in enumerate(messages):
        role = "用户" if msg.type == "human" else "AI"
        content = msg.content[:30] + "..." if len(msg.content) > 30 else msg.content
        print(f"    [{role}]: {content}")
    
    # 查询情绪记录
    emotions = DatabaseStorage.get_emotion_records(session_id)
    print(f"  ✓ 情绪记录数: {len(emotions)}")
    for e in emotions:
        print(f"    - {e['primary_emotion']}: 强度{e['intensity']}")
    
    # 测试6：列出所有会话
    print("\n【测试6】列出所有会话...")
    all_sessions = DatabaseStorage.list_sessions()
    print(f"  ✓ 总会话数: {len(all_sessions)}")
    print(f"  ✓ 会话列表: {all_sessions}")
    
    print("\n" + "=" * 50)
    print("测试完成！数据已保存到MySQL数据库")
    print("你可以使用以下SQL查看数据：")
    print(f"  SELECT * FROM sessions WHERE session_id = '{session_id}';")
    print(f"  SELECT * FROM messages WHERE session_id = '{session_id}';")
    print(f"  SELECT * FROM emotion_records WHERE session_id = '{session_id}';")
    print("=" * 50)
    
    return True


if __name__ == "__main__":
    test_database_storage()
