"""关键词匹配文本情绪引擎。

从 EmotionService._text_emotion 提取为独立引擎，
实现 TextEmotionEngine 协议，始终可用、零外部依赖。
"""

from __future__ import annotations

import re
from typing import Dict

from modules.emotion.base import TextEmotionEngine, TextEmotionResult, _SUPPORTED_EMOTIONS

# -- 关键词分组（情绪分类用） --
_KEYWORD_GROUPS: Dict[str, list] = {
    "anxiety": [r"焦虑", r"紧张", r"担心", r"害怕", r"\banxious\b", r"\bnervous\b", r"\bworried\b"],
    "sadness": [r"难过", r"伤心", r"低落", r"委屈", r"难受", r"痛苦", r"绝望", r"\bsad\b", r"\bdown\b", r"\bdepressed\b"],
    "anger": [r"生气", r"愤怒", r"烦死", r"气死", r"\bangry\b", r"\bmad\b", r"\bfurious\b"],
    "fear": [r"恐惧", r"害怕", r"不敢", r"\bafraid\b", r"\bscared\b", r"\bfear\b"],
    "stress": [r"压力", r"崩溃", r"扛不住", r"\bstress\b", r"\boverwhelmed\b"],
    "happiness": [r"开心", r"高兴", r"轻松", r"\bhappy\b", r"\brelieved\b", r"\bglad\b"],
    "confusion": [r"迷茫", r"困惑", r"不知道", r"\bconfused\b", r"\blost\b"],
}

# -- 信息型问句模式（用于 intent=information 检测） --
_INFORMATION_PATTERNS = [
    r"什么是", r"是什么", r"什么意思", r"是什么意思",
    r"怎么", r"如何", r"为什么", r"为啥",
    r"怎样", r"怎么样", r"哪里", r"什么时候", r"哪个",
    r"能不能", r"可以.*吗", r"有没有",
]
_INFORMATION_RE = re.compile("|".join(_INFORMATION_PATTERNS))

# -- 强度计算关键词（加重级症状词） --
_INTENSITY_KEYWORDS = [
    "焦虑", "紧张", "害怕", "难过", "伤心", "生气", "愤怒", "恐惧", "崩溃",
    "低落", "没兴趣", "吃不下", "睡不着", "受不了", "绝望", "撑不住",
    "严重", "非常", "特别", "极了", "死", "完全", "一直",
]

class KeywordEmotionEngine:
    """基于正则关键词匹配的文本情绪分类引擎。"""

    def __init__(self) -> None:
        self._keyword_groups: Dict[str, list] = _KEYWORD_GROUPS
        self._supported = _SUPPORTED_EMOTIONS
        self._intensity_keywords = _INTENSITY_KEYWORDS

    def _detect_intent(self, text: str, has_emotion: bool) -> str:
        """检测用户输入意图。

        仅在无情绪信号时判断是否为信息型提问；
        有情绪信号时保持 unknown，由 risk 驱动路由。
        """
        if has_emotion:
            return "unknown"
        if _INFORMATION_RE.search(text):
            return "information"
        return "unknown"

    def predict(self, text: str) -> TextEmotionResult:
        lowered = text.lower()
        scores = {emotion: 0 for emotion in self._supported}
        for emotion, patterns in self._keyword_groups.items():
            for pattern in patterns:
                if re.search(pattern, lowered):
                    scores[emotion] += 1

        # 标点加成
        if "!" in text:
            scores["anger"] += 0.5
            scores["happiness"] += 0.25
        if "..." in text or "？" in text or "?" in text:
            scores["anxiety"] += 0.25
            scores["confusion"] += 0.25

        primary = max(scores.items(), key=lambda item: item[1])[0]
        total_hits = sum(scores.values())
        intent = self._detect_intent(text, total_hits > 0)

        if total_hits == 0:
            return TextEmotionResult(
                primary_emotion="neutral",
                confidence=1.0,
                all_emotions={e: 0.0 for e in self._supported},
                intent=intent,
                hit_count=0,
                model_name="text_keywords",
            )

        # 归一化分数 → all_emotions
        max_score = max(scores.values())
        all_emotions = {
            e: round(scores[e] / max_score, 4) if max_score > 0 else 0.0
            for e in self._supported
        }

        confidence = min(total_hits / 4.0, 1.0)
        result = TextEmotionResult(
            primary_emotion=primary,
            confidence=round(confidence, 4),
            all_emotions=all_emotions,
            intent=intent,
            hit_count=round(total_hits),
            model_name="text_keywords",
        )
        return result

    @property
    def model_name(self) -> str:
        return "text_keywords"

    @property
    def is_ready(self) -> bool:
        return True
