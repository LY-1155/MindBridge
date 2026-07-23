"""用真实 LLM 测试三条路由的提示词效果。

运行方式: python scripts/test_prompts_live.py
输出: scripts/live_test_results.txt
"""

from __future__ import annotations
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import SystemMessage, HumanMessage

from modules.intervention.generator import (
    COMFORT_SYSTEM_PROMPT,
    KNOWLEDGE_SYSTEM_PROMPT,
    GENERAL_SYSTEM_PROMPT,
)
from modules.prompt_guard import INSTRUCTION_HIERARCHY_SUFFIX, wrap_user_text
from core.llm.base import OpenAICompatibleAdapter, LLMConfig


# ── 测试用例 ──────────────────────────────────────────────────
# 模拟不同路由的典型场景，覆盖医生的两个关注点：
#   1. 共情深度（镜像原词、捕捉矛盾、命名未言明）
#   2. 回复简洁度（不以一次性解答为目标，推动持续对话）

TEST_CASES = [
    # ── COMFORT 路由 ──
    {
        "label": "COMFORT - 工作压力 + 无力感",
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
        "label": "COMFORT - 人际冲突 + 委屈",
        "prompt": COMFORT_SYSTEM_PROMPT,
        "values": {
            "primary_emotion": "sadness",
            "intensity": "0.70",
            "risk": "0.15",
            "conversation_history": "用户：今天跟我妈吵架了",
        },
        "user_input": "她说我一点都不关心她，但我真的已经很尽力了，好累",
    },
    {
        "label": "COMFORT - 表面说没事但情绪高",
        "prompt": COMFORT_SYSTEM_PROMPT,
        "values": {
            "primary_emotion": "sadness",
            "intensity": "0.65",
            "risk": "0.1",
            "conversation_history": "（无历史对话，这是第一轮）",
        },
        "user_input": "其实也不是什么大事，就是最近总是莫名想哭",
    },

    # ── KNOWLEDGE 路由 ──
    {
        "label": "KNOWLEDGE - 灾难化思维 + RAG 匹配",
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
                "简单的自助方法：记录'我担心的/实际发生的'对照表。"
            ),
        },
        "user_input": "我控制不住，每次一有什么事就开始想最坏的结果，已经影响工作了",
    },
    {
        "label": "KNOWLEDGE - 药物问题 + RAG 不匹配",
        "prompt": KNOWLEDGE_SYSTEM_PROMPT,
        "values": {
            "primary_emotion": "anxiety",
            "intensity": "0.55",
            "risk": "0.35",
            "conversation_history": "用户：我想问一下关于吃药的事",
            "retrieved_knowledge": (
                "【一般性心理咨询科普】\n"
                "心理咨询与药物治疗的区别：心理咨询通过谈话帮助来访者探索内心，"
                "药物治疗通过调整神经递质来缓解症状。两者常结合使用效果更佳。"
                "（注：本文不包含任何具体药物的剂量、副作用或用药建议）"
            ),
        },
        "user_input": "我这种情况需要吃舍曲林吗？副作用大不大？",
    },

    # ── GENERAL 路由 ──
    {
        "label": "GENERAL - 试探性求助信号",
        "prompt": GENERAL_SYSTEM_PROMPT,
        "values": {
            "primary_emotion": "neutral",
            "intensity": "0.2",
            "risk": "0.0",
            "conversation_history": "用户：你好\n助手：你好呀，今天有什么我可以帮你的吗？",
        },
        "user_input": "哈哈其实也没什么，就是最近有点睡不着",
    },
]


def main():
    # 初始化 LLM（关闭 streaming，同步调用需要）
    # max_tokens 设高 —— qwen3.7-max 是推理模型，reasoning tokens 占了配额
    config = LLMConfig(temperature=0.7, max_tokens=4096, streaming=False)
    adapter = OpenAICompatibleAdapter(config=config)
    llm = adapter.llm  # ChatOpenAI 实例

    output_path = os.path.join(os.path.dirname(__file__), "live_test_results.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  提示词改动效果验证 — 真实 LLM 输出 (model: qwen3.7-max)\n")
        f.write("=" * 70 + "\n")
        f.write("审阅维度：\n")
        f.write("  1. 是否用了用户的词而非替换成术语？（深共情）\n")
        f.write("  2. 回复是否简短，以提问结尾推动继续对话？（短回复）\n")
        f.write("  3. 是否避免了空洞的'我理解你'？（具体 > 声称）\n")
        f.write("  4. 是否在没被求助时给了建议？（不该做的）\n\n")

        for i, case in enumerate(TEST_CASES, 1):
            # 构建消息
            system_text = case["prompt"].format(**case["values"])
            system_text += INSTRUCTION_HIERARCHY_SUFFIX
            wrapped_input = wrap_user_text(case["user_input"])

            messages = [
                SystemMessage(content=system_text),
                HumanMessage(content=wrapped_input),
            ]

            # 调用 LLM
            print(f"[{i}/{len(TEST_CASES)}] {case['label']} ... ", end="", flush=True)
            try:
                response = llm.invoke(messages)
                reasoning = response.response_metadata.get('token_usage', {}).get('completion_tokens_details', {}).get('reasoning_tokens', 0)
                reply = response.content or ""
                print(f"OK (reply={len(reply)} chars, reasoning={reasoning} tokens)", flush=True)
            except Exception as e:
                reply = f"[ERROR] {e}"
                import traceback
                traceback.print_exc()
                print(f"FAILED: {e}")

            # 写入结果
            f.write(f"--- 用例 {i}: {case['label']} ---\n\n")
            f.write(f"【用户输入】\n{case['user_input']}\n\n")
            f.write(f"【AI 回复】\n{reply}\n\n")
            f.write(f"【审阅笔记】\n")
            f.write(f"  字数: {len(reply)} 字符\n")
            f.write(f"  是否以提问结尾: {'是' if '？' in reply or '?' in reply else '否'}\n")
            # 检查是否有空洞共情语
            shallow = ["我理解你", "我理解", "我听到了", "这很难", "你要坚强"]
            found_shallow = [s for s in shallow if s in reply]
            if found_shallow:
                f.write(f"  ⚠️ 空洞共情语: {found_shallow}\n")
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("  审阅完成\n")
        f.write("=" * 70 + "\n")

    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
