"""
测试通义千问模型
==============

验证API密钥是否配置正确，模型是否可用。
"""

import asyncio
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.llm.base import get_llm_adapter
from config.settings import settings


async def test_qwen():
    print("=" * 50)
    print("通义千问模型测试")
    print("=" * 50)

    print(f"\n配置信息：")
    print(f"  API_KEY: {settings.OPENAI_API_KEY[:20]}...")
    print(f"  API_BASE: {settings.OPENAI_API_BASE}")
    print(f"  MODEL_NAME: {settings.MODEL_NAME}")

    print("\n正在初始化模型适配器...")
    try:
        llm_adapter = get_llm_adapter("qwen")
        print(f"✓ 适配器创建成功: {type(llm_adapter).__name__}")
        
        print("\n正在测试对话...")
        response = await llm_adapter.chat_with_system(
            user_input="你好，我最近感觉很焦虑",
            system_prompt="你是一个专业的心理咨询师，请用温暖、理解的语气回复。"
        )
        
        print(f"\n模型回复：")
        print(f"  {response}")
        
        print("\n" + "=" * 50)
        print("测试成功！模型可以正常使用")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        print("\n可能的原因：")
        print("  1. API密钥不正确")
        print("  2. 网络连接问题")
        print("  3. API服务暂时不可用")


if __name__ == "__main__":
    asyncio.run(test_qwen())
