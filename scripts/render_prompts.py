"""渲染三条路由的完整 prompt 用于人工审阅。

运行方式: python scripts/render_prompts.py
目的: 展示模型实际收到的完整指令文本，人工判断指令质量
"""

from __future__ import annotations
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.intervention.generator import (
    COMFORT_SYSTEM_PROMPT,
    KNOWLEDGE_SYSTEM_PROMPT,
    GENERAL_SYSTEM_PROMPT,
)
from modules.prompt_guard import INSTRUCTION_HIERARCHY_SUFFIX, wrap_user_text


# ── 模拟测试场景 ────────────────────────────────────────────
TEST_CASES = [
    {
        "route": "COMFORT",
        "prompt": COMFORT_SYSTEM_PROMPT,
        "values": {
            "primary_emotion": "anxiety",
            "intensity": "0.75",
            "risk": "0.2",
            "conversation_history": (
                "用户：最近总是睡不着，脑子里停不下来\n"
                "助手：听起来你脑子里有很多事情在转，让你没法放松下来。愿意多说一点吗？\n"
                "用户：就是担心工作上的事，还有家里的事"
            ),
        },
        "user_input": "我也不知道怎么办了，感觉做什么都没用",
    },
    {
        "route": "KNOWLEDGE",
        "prompt": KNOWLEDGE_SYSTEM_PROMPT,
        "values": {
            "primary_emotion": "anxiety",
            "intensity": "0.65",
            "risk": "0.4",
            "conversation_history": (
                "用户：我最近总是忍不住往最坏的方向想\n"
                "助手：那种脑子不停往最坏方向跑的感觉，一定很消耗你。你最容易在什么事情上有这种感觉？"
            ),
            "retrieved_knowledge": (
                "【认知行为疗法 - 灾难化思维干预】\n"
                "灾难化思维是焦虑障碍中的常见认知扭曲，表现为自动将情境预期为最坏结果。"
                "CBT 干预方法包括：1) 识别自动思维 2) 检查证据 3) 生成替代性解释。"
                "简单的自助方法：记录'我担心的/实际发生的'对照表，"
                "逐步积累'预测偏差'的觉察。"
            ),
        },
        "user_input": "我控制不住，每次一有什么事就开始想最坏的结果",
    },
    {
        "route": "GENERAL",
        "prompt": GENERAL_SYSTEM_PROMPT,
        "values": {
            "primary_emotion": "neutral",
            "intensity": "0.1",
            "risk": "0.0",
            "conversation_history": "用户：你好\n助手：你好呀，今天有什么我可以帮你的吗？",
        },
        "user_input": "哈哈其实我也没什么事，就是最近有点睡不着",
    },
]


def main():
    for case in TEST_CASES:
        route = case["route"]
        print("=" * 70)
        print(f"  路由: {route}")
        print("=" * 70)

        # 1. 渲染系统提示词
        system_text = case["prompt"].format(**case["values"])
        # 2. 追加安全后缀
        system_text += INSTRUCTION_HIERARCHY_SUFFIX
        # 3. 包裹用户输入
        wrapped_input = wrap_user_text(case["user_input"])

        # ── 输出 ──
        print("\n【最终发给 LLM 的完整消息】\n")
        print("--- SystemMessage ---")
        print(system_text)
        print("\n--- HumanMessage ---")
        print(wrapped_input)
        print("\n")

    print("=" * 70)
    print("人工审阅检查清单：")
    print("=" * 70)
    print("  [ ] COMFORT: 是否明确要求镜映用户原词？")
    print("  [ ] COMFORT: 是否有 ≤3 句话的长度约束？")
    print("  [ ] COMFORT: 示例是否展示了深共情 vs 浅共情的区别？")
    print("  [ ] KNOWLEDGE: 是否有检索内容可信度三级判断？")
    print("  [ ] KNOWLEDGE: 是否要求先共情再讲知识？")
    print("  [ ] KNOWLEDGE: 示例是否覆盖了'信息不足时诚实说不知道'？")
    print("  [ ] GENERAL: 是否有情绪信号识别？")
    print("  [ ] GENERAL: 是否禁止强行切入咨询师模式？")
    print("  [ ] ALL: 安全后缀是否正确追加？用户输入是否被 <user_message> 包裹？")


if __name__ == "__main__":
    main()
