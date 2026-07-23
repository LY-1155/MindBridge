"""
API接口测试脚本
==============

测试心理咨询AI的API接口，验证输入输出和数据库存储。
"""

import httpx
import json
import time

BASE_URL = "http://localhost:8000"


def test_health():
    """测试健康检查接口"""
    print("\n【测试1】健康检查")
    print("-" * 40)
    
    response = httpx.get(f"{BASE_URL}/ping")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    return response.status_code == 200


def test_chat():
    """测试对话接口"""
    print("\n【测试2】对话测试")
    print("-" * 40)
    
    # 第一轮对话
    print("\n第一轮对话:")
    payload = {
        "message": "你好，我最近感觉很焦虑，工作压力很大",
        "enable_thought_chain": True,
        "enable_emotion_analysis": True
    }
    
    response = httpx.post(f"{BASE_URL}/api/v1/chat", json=payload, timeout=30)
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"会话ID: {data.get('session_id')}")
        print(f"AI回复: {data.get('response')}")
        print(f"情绪分析: {data.get('emotion_analysis')}")
        session_id = data.get('session_id')
    else:
        print(f"错误: {response.text}")
        return None
    
    # 第二轮对话（使用相同session_id）
    print("\n第二轮对话（继续会话）:")
    payload2 = {
        "message": "主要是项目进度太紧，每天都要加班到很晚",
        "session_id": session_id
    }
    
    response2 = httpx.post(f"{BASE_URL}/api/v1/chat", json=payload2, timeout=30)
    if response2.status_code == 200:
        data2 = response2.json()
        print(f"会话ID: {data2.get('session_id')} (相同)")
        print(f"AI回复: {data2.get('response')}")
    else:
        print(f"错误: {response2.text}")
    
    return session_id


def test_sessions(session_id: str):
    """测试会话管理接口"""
    print("\n【测试3】会话管理")
    print("-" * 40)
    
    # 获取会话信息
    print(f"\n获取会话信息 (session_id={session_id}):")
    response = httpx.get(f"{BASE_URL}/api/v1/sessions/{session_id}")
    print(f"状态码: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"消息数量: {data.get('message_count')}")
        print(f"关键话题: {data.get('key_topics')}")
    
    # 获取所有会话
    print("\n获取所有会话:")
    response = httpx.get(f"{BASE_URL}/api/v1/sessions")
    if response.status_code == 200:
        sessions = response.json().get('sessions', [])
        print(f"会话总数: {len(sessions)}")
        for s in sessions[:5]:
            print(f"  - {s}")


def main():
    print("=" * 50)
    print("心理咨询AI - API接口测试")
    print("=" * 50)
    
    try:
        # 测试健康检查
        if not test_health():
            print("服务不可用，请检查是否启动")
            return
        
        # 测试对话
        session_id = test_chat()
        
        if session_id:
            # 测试会话管理
            test_sessions(session_id)
        
        print("\n" + "=" * 50)
        print("测试完成！")
        print("\n验证数据库存储:")
        print("  SELECT * FROM sessions;")
        print("  SELECT * FROM messages;")
        print("  SELECT * FROM emotion_records;")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("请确保API服务已启动: python -m uvicorn api.main:app --port 8000")


if __name__ == "__main__":
    main()
