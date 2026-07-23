"""
简化API测试脚本
"""

import httpx
import asyncio

BASE_URL = "http://localhost:8000"


async def test_api():
    print("=" * 60)
    print(" 心理咨询AI API测试")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        
        # 测试1: 健康检查
        print("\n[测试1] 健康检查 GET /ping")
        try:
            response = await client.get(f"{BASE_URL}/ping")
            print(f"  状态码: {response.status_code}")
            print(f"  响应: {response.json()}")
        except Exception as e:
            print(f"  错误: {e}")
        
        # 测试2: 根路由
        print("\n[测试2] 根路由 GET /")
        try:
            response = await client.get(f"{BASE_URL}/")
            print(f"  状态码: {response.status_code}")
            print(f"  响应: {response.json()}")
        except Exception as e:
            print(f"  错误: {e}")
        
        # 测试3: 发送消息
        print("\n[测试3] 发送消息 POST /api/v1/chat")
        try:
            payload = {"message": "我最近感觉很焦虑"}
            response = await client.post(
                f"{BASE_URL}/api/v1/chat",
                json=payload
            )
            print(f"  状态码: {response.status_code}")
            data = response.json()
            if response.status_code == 200:
                print(f"  会话ID: {data.get('session_id')}")
                print(f"  AI回应: {data.get('response', '')[:100]}...")
                print(f"  安全警报: {data.get('safety_alert')}")
            else:
                print(f"  错误: {data}")
        except Exception as e:
            print(f"  错误: {e}")
        
    print("\n" + "=" * 60)
    print(" API测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_api())
