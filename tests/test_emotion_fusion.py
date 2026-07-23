"""情绪融合层行为测试：1~N 路融合、降级、summary 动态生成。"""
from __future__ import annotations

import pytest

from multimodal.emotion_fusion import (
    EmotionSignal,
    FusedEmotionResult,
    build_signal,
    fuse_emotions,
)


def make_signal(source: str, primary: str, confidence: float = 0.7) -> EmotionSignal:
    return EmotionSignal(
        source=source,
        primary_emotion=primary,
        confidence=confidence,
        all_emotions={primary: confidence},
    )


class TestFuseEmotions:
    """核心融合逻辑。"""

    def test_single_signal_returns_same_emotion(self):
        signal = make_signal("audio", "anxiety", 0.8)
        result = fuse_emotions(signal)
        assert result is not None
        assert result.primary_emotion == "anxiety"
        assert result.signal_count == 1
        assert result.mixed_signals is False

    def test_two_agreeing_signals(self):
        a = make_signal("audio", "sadness", 0.8)
        b = make_signal("text", "sadness", 0.4)
        result = fuse_emotions(a, b)
        assert result is not None
        assert result.primary_emotion == "sadness"
        assert result.signal_count == 2
        assert result.mixed_signals is False

    def test_two_disagreeing_signals_picks_strongest(self):
        a = make_signal("audio", "anger", 0.9)
        b = make_signal("text", "sadness", 0.3)
        result = fuse_emotions(a, b)
        assert result is not None
        # 高置信度音频信号应该主导
        assert result.primary_emotion == "anger"
        assert result.mixed_signals is True

    def test_all_none_signals_returns_none(self):
        result = fuse_emotions(None, None, None)
        assert result is None

    def test_skips_none_in_args(self):
        a = make_signal("audio", "fear", 0.7)
        result = fuse_emotions(None, a, None)
        assert result is not None
        assert result.primary_emotion == "fear"
        assert result.signal_count == 1

    def test_three_signals_works(self):
        a = make_signal("audio", "stress", 0.7)
        b = make_signal("text", "anxiety", 0.4)
        c = make_signal("visual", "stress", 0.6)
        result = fuse_emotions(a, b, c)
        assert result is not None
        assert result.signal_count == 3

    def test_summary_dynamically_mentions_sources(self):
        a = make_signal("audio", "sadness", 0.8)
        b = make_signal("text", "sadness", 0.4)
        result = fuse_emotions(a, b)
        assert result is not None
        # summary 应该用 source 字段，不硬编码 "Audio and visual"
        assert "audio" in result.summary and "text" in result.summary

    def test_summary_does_not_mention_visual_when_not_present(self):
        a = make_signal("audio", "anger", 0.9)
        b = make_signal("text", "sadness", 0.3)
        result = fuse_emotions(a, b)
        assert result is not None
        # 没有视觉信号时，summary 不应该提到 "visual"
        assert "visual" not in result.summary.lower()


class TestBuildSignal:
    """辅助函数 build_signal。"""

    def test_valid_payload(self):
        sig = build_signal("audio", {"primary_emotion": "anxiety", "confidence": 0.8})
        assert sig is not None
        assert sig.source == "audio"
        assert sig.primary_emotion == "anxiety"
        assert sig.confidence == 0.8

    def test_empty_payload_returns_none(self):
        assert build_signal("audio", None) is None
        assert build_signal("audio", {}) is None

    def test_missing_primary_returns_none(self):
        assert build_signal("audio", {"confidence": 0.5}) is None
