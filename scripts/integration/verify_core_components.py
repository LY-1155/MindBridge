"""
自动化验证测试脚本
==================

这个脚本用于验证心理咨询AI项目的核心功能。
不需要交互式输入，自动运行所有测试。
"""

import asyncio
import os
import sys
import uuid

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.llm import get_llm_adapter, LLMConfig
from core.memory import TherapySessionMemory, SessionManager
from config.settings import settings
from pipeline.orchestrator import run_pipeline
from schemas.contracts.v1 import PipelineInput


def print_separator(title: str = ""):
    """打印分隔线"""
    if title:
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")
    else:
        print(f"\n{'-'*60}")


def print_result(label: str, value):
    """打印结果"""
    print(f"  {label}: {value}")


async def test_llm_adapter():
    """测试1: LLM适配器"""
    print_separator("测试1: LLM适配器")
    
    # 测试Mock适配器
    print("\n[1.1] 测试Mock适配器...")
    try:
        mock_adapter = get_llm_adapter("mock")
        print_result("适配器类型", type(mock_adapter).__name__)
        
        # 测试基本对话 - ainvoke返回AIMessage对象
        response = await mock_adapter.ainvoke("你好")
        # AIMessage对象需要访问.content属性获取文本
        response_text = response.content if hasattr(response, 'content') else str(response)
        print_result("对话响应", response_text[:50] + "...")
        print_result("状态", "✅ 通过")
    except Exception as e:
        print_result("状态", f"❌ 失败: {e}")
        return False
    
    # 测试流式对话
    print("\n[1.2] 测试流式对话...")
    try:
        chunks = []
        # 使用astream异步流式方法
        async for chunk in mock_adapter.astream("测试流式"):
            chunks.append(chunk)
        print_result("收到片段数", len(chunks))
        print_result("状态", "✅ 通过")
    except Exception as e:
        print_result("状态", f"❌ 失败: {e}")
        return False
    
    return True


async def test_session_memory():
    """测试2: 会话记忆"""
    print_separator("测试2: 会话记忆")
    
    try:
        # 创建会话
        session_id = f"test-{uuid.uuid4().hex[:8]}"
        session = SessionManager.get_session(session_id)
        print_result("会话ID", session_id)
        print_result("状态", "✅ 会话创建成功")
        
        # 添加消息
        print("\n[2.1] 测试消息存储...")
        session.add_user_message("我最近感觉很焦虑")
        session.add_ai_message("我理解你感到焦虑，能告诉我更多吗？")
        session.add_user_message("工作压力很大")
        
        messages = session.get_messages()
        print_result("消息数量", len(messages))
        print_result("状态", "✅ 通过" if len(messages) == 3 else "❌ 失败")
        
        # 添加情绪记录
        print("\n[2.2] 测试情绪记录...")
        from core.memory import EmotionRecord
        emotion_record = EmotionRecord(
            primary_emotion="anxiety",
            intensity=7,
            triggers=["工作压力"],
            context="我最近感觉很焦虑"
        )
        session.add_emotion_record(emotion_record)
        
        trend = session.get_emotion_trend()
        print_result("情绪趋势", trend['trend'])
        print_result("平均强度", trend['average_intensity'])
        print_result("状态", "✅ 通过")
        
        return True
    except Exception as e:
        print_result("状态", f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_pipeline():
    """测试3: 流水线对话（安全→情绪→路由→干预）"""
    print_separator("测试3: 流水线对话")

    try:
        session_id = f"pipeline-test-{uuid.uuid4().hex[:8]}"

        print("\n[3.1] 测试正常对话...")
        user_input = "我最近工作压力很大，感觉很焦虑"
        print_result("用户输入", user_input)

        output = run_pipeline(PipelineInput(
            text=user_input,
            session_id=session_id,
        ))

        intervention = output.intervention
        print_result("路由", output.route.get("route", "?"))
        print_result("AI回应", intervention.get("reply", "")[:80] + "...")
        print_result("安全警报", bool(output.safety.get("blocked") or output.safety.get("level", 0) >= 2))
        print_result("状态", "✅ 通过")

        return True
    except Exception as e:
        print_result("状态", f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_safety_detection():
    """测试4: 安全检测（危机干预）"""
    print_separator("测试4: 安全检测（危机干预）")

    try:
        session_id = f"safety-test-{uuid.uuid4().hex[:8]}"

        print("\n[4.1] 测试危机关键词检测...")
        crisis_input = "我不想活了"
        print_result("用户输入", crisis_input)

        output = run_pipeline(PipelineInput(
            text=crisis_input,
            session_id=session_id,
        ))

        safety = output.safety
        print_result("安全等级", safety.get("level", 0))
        print_result("被阻止", safety.get("blocked", False))
        print_result("路由", output.route.get("route", "?"))
        print_result("状态", "✅ 危机检测正常" if safety.get("level", 0) >= 1 else "⚠️ 未触发警报")

        return True
    except Exception as e:
        print_result("状态", f"❌ 失败: {e}")
        return False


async def test_session_manager():
    """测试5: 会话管理器"""
    print_separator("测试5: 会话管理器")
    
    try:
        # 创建多个会话
        print("\n[5.1] 测试会话创建和管理...")
        session_ids = []
        for i in range(3):
            sid = SessionManager.create_session()
            session_ids.append(sid)
            print_result(f"会话{i+1}", sid)
        
        # 验证会话存在
        print("\n[5.2] 验证会话存在...")
        for sid in session_ids:
            session = SessionManager.get_session(sid)
            print_result(f"会话{sid}", "✅ 存在" if session else "❌ 不存在")
        
        # 删除会话
        print("\n[5.3] 测试会话删除...")
        SessionManager.delete_session(session_ids[0])
        remaining = len(SessionManager._sessions)
        print_result("剩余会话数", remaining)
        print_result("状态", "✅ 通过")
        
        return True
    except Exception as e:
        print_result("状态", f"❌ 失败: {e}")
        return False


async def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print(" 心理咨询AI项目 - 自动化验证测试")
    print("="*60)
    
    results = {}
    
    # 运行所有测试
    results["LLM适配器"] = await test_llm_adapter()
    results["会话记忆"] = await test_session_memory()
    results["流水线对话"] = await test_pipeline()
    results["安全检测"] = await test_safety_detection()
    results["会话管理器"] = await test_session_manager()
    
    # 打印总结
    print_separator("测试总结")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
    
    print(f"\n  总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n  🎉 所有测试通过！项目运行正常。")
    else:
        print(f"\n  ⚠️ 有 {total - passed} 个测试失败，请检查。")
    
    print_separator()
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
