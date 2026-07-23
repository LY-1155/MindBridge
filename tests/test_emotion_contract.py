"""
情绪分析模块契约测试：版本号与标签约束。
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.contracts.v1 import CONTRACT_VERSION, EmotionTags


def test_contract_version_is_1_3():
    assert CONTRACT_VERSION == "1.3", f"期望 1.3，实际 {CONTRACT_VERSION}"


@pytest.mark.parametrize("emotion", [
    "neutral", "anxiety", "sadness", "anger",
    "fear", "stress", "happiness", "confusion",
    "distress",
])
def test_valid_primary_emotions(emotion):
    tags = EmotionTags(primary_emotion=emotion, intensity=0.5, risk=0.3)
    assert tags.primary_emotion == emotion


def test_invalid_primary_emotion_fails():
    with pytest.raises(ValidationError):
        EmotionTags(primary_emotion="invalid_emotion", intensity=0.5, risk=0.3)
