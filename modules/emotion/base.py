"""文本情绪分类引擎协议与共享类型。

将 _text_emotion 从 EmotionService 中解耦为可插拔策略，
支持关键词匹配、ONNX 推理等不同实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Protocol, Tuple

# 8 种支持情绪（与 contracts.SUPPORTED_EMOTIONS 对齐，不含 distress）
_SUPPORTED_EMOTIONS = (
    "neutral", "anxiety", "sadness", "anger",
    "fear", "stress", "happiness", "confusion",
)

IntentLabel = Literal["information", "emotion_expression", "casual_chat", "unknown"]


@dataclass
class TextEmotionResult:
    """文本情绪分类统一输出。"""

    primary_emotion: str
    confidence: float
    all_emotions: Dict[str, float] = field(default_factory=dict)
    intent: IntentLabel = "unknown"
    hit_count: int = 0               # 关键词命中数（用于强度计算）
    model_name: str = "text_keywords"


class TextEmotionEngine(Protocol):
    """文本情绪分类引擎协议。

    实现类只需提供 predict() 方法和 model_name / is_ready 属性。
    """

    def predict(self, text: str) -> TextEmotionResult: ...

    @property
    def model_name(self) -> str: ...

    @property
    def is_ready(self) -> bool: ...
