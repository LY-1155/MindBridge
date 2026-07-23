"""
详细API测试脚本
==============

测试各个API接口的详细信息。
"""

import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import httpx

BASE_URL = "http://localhost:8000"


def test_chat_and_session():
    """测试对话和会话管理"""
    print("=" * 50)
    print("详细API测试")
    print("=" * 50)
    
    # 1. 发送消息
    print("\n【步骤1】发送消息...")
    payload = {
        "message": "你好，我最近感觉很焦虑"
    }
    
    response = httpx.post(f"{BASE_URL}/api/v1/chat", json=payload, timeout=30)
    print(f"状态码: {response.status_code}")
    
    if response.status_code != 200:
        print(f"错误: {response.text}")
        return
    
    data = response.json()
    session_id = data["session_id"]
    print(f"会话ID: {session_id}")
    print(f"AI回复: {data['response']}")
    
    # 2. 直接查询数据库验证
    print("\n【步骤2】查询数据库验证...")
    from core.memory.db_storage import DatabaseStorage
    
    session_data = DatabaseStorage.load_session(session_id)
    if session_data:
        print(f"数据库中的会话: {session_data}")
    else:
        print("数据库中未找到会话")
    
    messages = DatabaseStorage.get_messages(session_id)
    print(f"数据库中的消息数: {len(messages)}")
    
    # 3. 获取会话信息API
    print("\n【步骤3】获取会话信息API...")
    response = httpx.get(f"{BASE_URL}/api/v1/session/{session_id}", timeout=10)
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.text}")
    
    # 4. 列出所有会话
    print("\n【步骤4】列出所有会话...")
    response = httpx.get(f"{BASE_URL}/api/v1/sessions", timeout=10)
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")


if __name__ == "__main__":
    test_chat_and_session()
