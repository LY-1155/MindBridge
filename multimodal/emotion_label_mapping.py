"""
Ekman 7 类 → 契约 8 类情绪标签映射。

视觉模型（HSEmotion）输出 Ekman 基本表情，管线契约使用心理评估标签。
直接对应保留全置信度，跨域映射施加信心惩罚，未映射的标签由文本/音频补充。
"""
from __future__ import annotations

from typing import Tuple

# Ekman → 契约映射表：(目标标签, 置信度系数)
_MAPPING: dict[str, Tuple[str, float]] = {
    "angry": ("anger", 1.0),
    "fear": ("fear", 1.0),
    "happy": ("happiness", 1.0),
    "sad": ("sadness", 1.0),
    "neutral": ("neutral", 1.0),
    "disgust": ("stress", 0.6),
    "surprise": ("anxiety", 0.4),
}


def map_ekman_to_contract(ekman_label: str, confidence: float) -> Tuple[str, float]:
    """将 Ekman 标签映射为契约标签，并施加置信度惩罚。

    Args:
        ekman_label: HSEmotion 输出的 Ekman 标签（小写）。
        confidence: 原始置信度，0~1。

    Returns:
        (contract_label, adjusted_confidence) 元组。
    """
    entry = _MAPPING.get(ekman_label.lower())
    if entry is None:
        return ("neutral", confidence)
    target_label, coefficient = entry
    return (target_label, confidence * coefficient)
