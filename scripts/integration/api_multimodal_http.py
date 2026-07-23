"""
API测试脚本
==========

测试多模态API接口。
"""

import asyncio
import base64
import os

import requests

# 默认与本仓库 uvicorn 端口一致；旧版脚本曾写 8001，可通过环境变量覆盖。
BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


def test_health():
    """测试健康检查"""
    print("\n=== 测试健康检查 ===")
    
    response = requests.get(f"{BASE_URL}/ping")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    
    return response.status_code == 200


def test_tts():
    """测试语音合成API"""
    print("\n=== 测试TTS API ===")
    
    response = requests.post(
        f"{BASE_URL}/api/v1/multimodal/text-to-speech/base64",
        json={
            "text": "你好，这是一个测试。"
        }
    )
    
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"音频格式: {data['format']}")
        print(f"声音: {data['voice']}")
        print(f"Base64长度: {len(data['audio_base64'])}")
        return True
    else:
        print(f"错误: {response.text}")
        return False


def test_voices():
    """测试获取声音列表"""
    print("\n=== 测试获取声音列表 ===")
    
    response = requests.get(
        f"{BASE_URL}/api/v1/multimodal/voices",
        params={"language": "zh"}
    )
    
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"可用声音数量: {len(data['voices'])}")
        for v in data['voices'][:5]:
            print(f"  - {v['short_name']}: {v['gender']}, {v['locale']}")
        return True
    else:
        print(f"错误: {response.text}")
        return False


def test_chat():
    """测试聊天API"""
    print("\n=== 测试聊天API ===")
    
    response = requests.post(
        f"{BASE_URL}/api/v1/chat",
        json={
            "message": "你好",
            "enable_thought_chain": True,
            "enable_emotion_analysis": True
        }
    )
    
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"会话ID: {data['session_id']}")
        print(f"回复: {data['response'][:100]}...")
        return True
    else:
        print(f"错误: {response.text}")
        return False


def test_multimodal_chat():
    """测试多模态聊天API"""
    print("\n=== 测试多模态聊天API ===")
    
    response = requests.post(
        f"{BASE_URL}/api/v1/multimodal/chat",
        json={
            "text": "我今天心情不太好",
            "enable_tts": True
        }
    )
    
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"会话ID: {data['session_id']}")
        print(f"回复: {data['response'][:100]}...")
        if data.get('audio_base64'):
            print(f"语音回复长度: {len(data['audio_base64'])}")
        return True
    else:
        print(f"错误: {response.text}")
        return False


def main():
    print("=" * 50)
    print("多模态API测试")
    print("=" * 50)
    
    results = []
    
    try:
        results.append(("健康检查", test_health()))
        results.append(("TTS", test_tts()))
        results.append(("声音列表", test_voices()))
        results.append(("聊天", test_chat()))
        results.append(("多模态聊天", test_multimodal_chat()))
    except Exception as e:
        print(f"\n测试出错: {e}")
    
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n总计: {passed}/{total} 通过")


if __name__ == "__main__":
    main()
