import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import re


def format_thought_chain_output(thought_chain: Dict[str, Any]) -> str:
    output = []
    output.append("=" * 50)
    output.append("思维链分析")
    output.append("=" * 50)

    sections = {
        "emotion_recognition": "情绪识别",
        "emotion_intensity": "情绪强度",
        "user_needs": "用户需求",
        "therapy_approach": "治疗方法",
        "reasoning_process": "推理过程",
        "response_strategy": "回应策略",
        "empathy_expression": "共情表达",
        "safety_check": "安全检查"
    }

    for key, label in sections.items():
        value = thought_chain.get(key, "N/A")
        if isinstance(value, list):
            value = "\n  - ".join([""] + value)
        output.append(f"\n【{label}】")
        output.append(f"  {value}")

    return "\n".join(output)


def format_emotion_analysis(emotion_data: Dict[str, Any]) -> str:
    output = []
    output.append("\n" + "-" * 30)
    output.append("情绪分析结果")
    output.append("-" * 30)

    emotion_names = {
        "anxiety": "焦虑",
        "depression": "抑郁",
        "anger": "愤怒",
        "sadness": "悲伤",
        "loneliness": "孤独",
        "stress": "压力",
        "confusion": "困惑",
        "fear": "恐惧",
        "grief": "哀伤",
        "neutral": "平静"
    }

    primary = emotion_data.get("primary_emotion", "unknown")
    output.append(f"主要情绪: {emotion_names.get(primary, primary)}")
    output.append(f"情绪强度: {emotion_data.get('intensity', 0)}/10")

    cues = emotion_data.get("emotion_cues", [])
    if cues:
        output.append(f"情绪线索: {', '.join(cues)}")

    needs = emotion_data.get("underlying_needs", [])
    if needs:
        output.append(f"潜在需求: {', '.join(needs)}")

    distortions = emotion_data.get("cognitive_distortions", [])
    if distortions:
        output.append(f"认知扭曲: {', '.join(distortions)}")

    safety = emotion_data.get("safety_concerns", "none")
    output.append(f"安全风险: {safety}")

    return "\n".join(output)


def print_session_summary(session_data: Dict[str, Any]) -> None:
    print("\n" + "=" * 50)
    print("会话摘要")
    print("=" * 50)
    print(f"会话ID: {session_data.get('session_id', 'N/A')}")
    print(f"消息数量: {session_data.get('message_count', 0)}")
    print(f"最后活跃: {session_data.get('last_active', 'N/A')}")

    topics = session_data.get("key_topics", [])
    if topics:
        print(f"关键话题: {', '.join(topics)}")


def validate_session_id(session_id: str) -> bool:
    pattern = r'^[a-zA-Z0-9_-]+$'
    return bool(re.match(pattern, session_id)) and len(session_id) <= 64


def truncate_text(text: str, max_length: int = 100) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
