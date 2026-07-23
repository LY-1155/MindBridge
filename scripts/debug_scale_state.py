"""验证 scale_state 在 Redis/MySQL 中的读写是否正常"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory.session_memory import SessionManager, TherapySessionMemory

# 模拟量表流程
sid = "debug-scale-001"
print(f"1. 创建 session: {sid}")

session = TherapySessionMemory(session_id=sid, use_database=True, user_id="test-user")
print(f"   scale_state 初始值: {session.metadata.scale_state}")

# 模拟 orch.start() 设置 scale_state
test_state = {"status": "in_progress", "scale_name": "PHQ-9", "current_item_index": 3, "scores": [1, 2, 2]}
session.metadata.scale_state = test_state
print(f"2. 设置 scale_state: {test_state}")

# 保存
try:
    session.save_scale_state()
    print("3. save_scale_state() 成功")
except Exception as e:
    print(f"3. save_scale_state() 失败: {e}")

# 重新加载（模拟下一轮请求）
print("4. 重新加载 session...")
session2 = TherapySessionMemory(session_id=sid, use_database=True, user_id="test-user")
loaded = session2.metadata.scale_state
print(f"   scale_state 加载值: {loaded}")

if loaded == test_state:
    print("\n✅ scale_state 读写正常 — 量表状态可以跨请求保留")
elif loaded is None:
    print("\n❌ scale_state 加载为 None — 这就是量表反复重启的原因！")
    print("   检查 Redis (localhost:6379) 是否正常连接")
else:
    print(f"\n⚠️ scale_state 不匹配: {loaded}")
