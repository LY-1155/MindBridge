"""
情绪标签映射行为测试：Ekman 7 类 → 契约 8 类 + 信心惩罚。
"""
from __future__ import annotations

import pytest

from multimodal.emotion_label_mapping import map_ekman_to_contract


class TestDirectMapping:
    """直接对应：angry→anger, fear→fear, happy→happiness, sad→sadness, neutral→neutral，全置信。"""

    def test_angry_maps_to_anger_full_confidence(self):
        label, confidence = map_ekman_to_contract("angry", 0.9)
        assert label == "anger"
        assert confidence == 0.9

    def test_fear_maps_to_fear_full_confidence(self):
        label, confidence = map_ekman_to_contract("fear", 0.8)
        assert label == "fear"
        assert confidence == 0.8

    def test_happy_maps_to_happiness_full_confidence(self):
        label, confidence = map_ekman_to_contract("happy", 0.95)
        assert label == "happiness"
        assert confidence == 0.95

    def test_sad_maps_to_sadness_full_confidence(self):
        label, confidence = map_ekman_to_contract("sad", 0.7)
        assert label == "sadness"
        assert confidence == 0.7

    def test_neutral_maps_to_neutral_full_confidence(self):
        label, confidence = map_ekman_to_contract("neutral", 0.6)
        assert label == "neutral"
        assert confidence == 0.6


class TestCrossDomainMapping:
    """跨域映射：disgust→stress ×0.6, surprise→anxiety ×0.4，置信度打折。"""

    def test_disgust_maps_to_stress_with_penalty(self):
        label, confidence = map_ekman_to_contract("disgust", 0.9)
        assert label == "stress"
        assert confidence == pytest.approx(0.9 * 0.6)

    def test_surprise_maps_to_anxiety_with_penalty(self):
        label, confidence = map_ekman_to_contract("surprise", 0.8)
        assert label == "anxiety"
        assert confidence == pytest.approx(0.8 * 0.4)


class TestUnknownLabel:
    """Ekman 未知标签 → neutral 降级。"""

    def test_unknown_label_falls_back_to_neutral(self):
        label, confidence = map_ekman_to_contract("unknown_emotion", 0.5)
        assert label == "neutral"
        assert confidence == 0.5

    def test_empty_string_falls_back_to_neutral(self):
        label, confidence = map_ekman_to_contract("", 0.5)
        assert label == "neutral"
        assert confidence == 0.5


class TestEdgeCases:
    """边界值：confidence 为 0、1、None。"""

    def test_zero_confidence_preserved_for_direct(self):
        label, confidence = map_ekman_to_contract("angry", 0.0)
        assert label == "anger"
        assert confidence == 0.0

    def test_zero_confidence_preserved_for_cross_domain(self):
        label, confidence = map_ekman_to_contract("disgust", 0.0)
        assert label == "stress"
        assert confidence == 0.0

    def test_max_confidence_for_cross_domain(self):
        label, confidence = map_ekman_to_contract("surprise", 1.0)
        assert label == "anxiety"
        assert confidence == pytest.approx(0.4)
