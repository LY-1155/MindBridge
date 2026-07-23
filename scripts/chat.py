"""命令行交互聊天 — 直接用新提示词与 LLM 对话。

运行方式: python scripts/chat.py
退出: 输入 /quit 或 /q

这个脚本绕过 API pipeline，直接使用 modules/intervention/generator.py 的
三个路由提示词，让你能快速验证提示词改动后的实际对话效果。

路由逻辑（简化版，与 router_rules.json 对齐）：
  - risk < 0.2  → GENERAL
  - risk < 0.5  → COMFORT
  - risk >= 0.5 → KNOWLEDGE (有知识库检索) / COMFORT (无检索内容)
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

# 根据风险等级选择路由
ROUTE_THRESHOLDS = {
    "general": 0.2,
    "comfort": 0.5,
}

# 固定情绪上下文（模拟管线产出；真实应用中由 emotion 模块提供）
MOCK_EMOTION = {
    "general": {"primary_emotion": "neutral", "intensity": 0.1, "risk": 0.0},
    "comfort": {"primary_emotion": "anxiety", "intensity": 0.6, "risk": 0.3},
    "knowledge": {"primary_emotion": "anxiety", "intensity": 0.55, "risk": 0.45},
}


def detect_route(user_input: str) -> tuple[str, dict]:
    """简化的路由检测，基于关键词 + 默认评分"""
    crisis_words = ["自杀", "不想活", "死", "结束生命", "自残", "伤害自己"]
    knowledge_hints = ["怎么办", "为什么", "是什么", "怎么治疗", "药", "症状",
                       "诊断", "是不是", "正常吗", "如何改善", "怎样才能"]

    if any(w in user_input for w in crisis_words):
        return "comfort", MOCK_EMOTION["comfort"]  # 不触发真实危机干预

    if any(w in user_input for w in knowledge_hints):
        return "knowledge", MOCK_EMOTION["knowledge"]

    # 默认走 comfort
    if len(user_input) > 15:
        return "comfort", MOCK_EMOTION["comfort"]
    return "general", MOCK_EMOTION["general"]


def main():
    print("=" * 60)
    print("  PRISM 心理咨询AI — 命令行聊天测试")
    print("  使用新提示词 (v3) | model: qwen3.7-max")
    print("=" * 60)
    print("  输入 /q 或 /quit 退出")
    print("  输入 /route 查看当前路由")
    print("=" * 60)

    # 初始化 LLM
    config = LLMConfig(temperature=0.7, max_tokens=4096, streaming=True)
    adapter = OpenAICompatibleAdapter(config=config)
    llm = adapter.llm

    history: list[str] = []

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/q", "/quit", "/exit"):
            print("再见！")
            break
        if user_input.lower() == "/route":
            route, emotion = detect_route(user_input)
            print(f"[当前路由: {route.upper()} | 情绪: {emotion['primary_emotion']} "
                  f"| 强度: {emotion['intensity']} | 风险: {emotion['risk']}]")
            continue

        # ── 路由 + 构建 prompt ──
        route, emotion = detect_route(user_input)
        history_str = "\n".join(history[-6:]) if history else "（无历史对话，这是第一轮）"

        if route == "knowledge":
            prompt = KNOWLEDGE_SYSTEM_PROMPT
            extra = {"retrieved_knowledge": "（知识库未连接，请基于你的专业知识回答，但要标注你不确定的地方）"}
        else:
            prompt = COMFORT_SYSTEM_PROMPT if route == "comfort" else GENERAL_SYSTEM_PROMPT
            extra = {}

        values = {
            "primary_emotion": emotion["primary_emotion"],
            "intensity": str(emotion["intensity"]),
            "risk": str(emotion["risk"]),
            "conversation_history": history_str,
            **extra,
        }

        system_text = prompt.format(**values) + INSTRUCTION_HIERARCHY_SUFFIX
        wrapped_input = wrap_user_text(user_input)

        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=wrapped_input),
        ]

        # ── 流式输出 ──
        print(f"\nAI ({route}): ", end="", flush=True)
        full_reply: list[str] = []
        try:
            for chunk in llm.stream(messages):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if content:
                    print(content, end="", flush=True)
                    full_reply.append(content)
        except Exception as e:
            print(f"\n[ERROR] {e}")
            continue

        reply = "".join(full_reply)
        print()

        # ── 更新历史 ──
        history.append(f"用户：{user_input}")
        history.append(f"助手：{reply}")
        if len(history) > 20:
            history = history[-20:]


if __name__ == "__main__":
    main()
