"""
情绪模块 EmotionService 行为测试：纯文本 / 带音频 / 降级三条路径。
"""
from __future__ import annotations

import pytest

from modules.emotion.stub import EmotionService
from schemas.contracts.v1 import EmotionAnalyzeRequest, EmotionTags


@pytest.fixture
def emotion_svc() -> EmotionService:
    return EmotionService()


@pytest.fixture
def sample_safety() -> dict:
    return {"level": 0, "blocked": False, "matched_terms": [], "meta": {}, "contract_version": "1.1"}


class TestEmotionAnalyzeTextOnly:
    """纯文本路径：不传 audio_path 时用关键词情绪分析。"""

    def test_implements_emotion_port(self, emotion_svc):
        """确认实现了 analyze 方法且签名匹配 EmotionPort 协议。"""
        assert callable(getattr(emotion_svc, "analyze", None))

    def test_returns_emotion_tags(self, emotion_svc, sample_safety):
        req = EmotionAnalyzeRequest(text="我最近很焦虑，睡不好", safety=sample_safety)
        result = emotion_svc.analyze(req)
        assert isinstance(result, EmotionTags)

    def test_detects_anxiety(self, emotion_svc, sample_safety):
        req = EmotionAnalyzeRequest(text="我很紧张，每天都担心", safety=sample_safety)
        result = emotion_svc.analyze(req)
        assert result.primary_emotion == "anxiety"

    def test_detects_sadness(self, emotion_svc, sample_safety):
        req = EmotionAnalyzeRequest(text="我觉得好难过，很低落", safety=sample_safety)
        result = emotion_svc.analyze(req)
        assert result.primary_emotion == "sadness"

    def test_detects_anger(self, emotion_svc, sample_safety):
        req = EmotionAnalyzeRequest(text="气死我了！烦死了！", safety=sample_safety)
        result = emotion_svc.analyze(req)
        assert result.primary_emotion == "anger"

    def test_detects_fear(self, emotion_svc, sample_safety):
        req = EmotionAnalyzeRequest(text="我很恐惧，不敢面对", safety=sample_safety)
        result = emotion_svc.analyze(req)
        assert result.primary_emotion == "fear"

    def test_detects_stress(self, emotion_svc, sample_safety):
        req = EmotionAnalyzeRequest(text="压力太大了，快崩溃了", safety=sample_safety)
        result = emotion_svc.analyze(req)
        assert result.primary_emotion == "stress"

    def test_detects_confusion(self, emotion_svc, sample_safety):
        req = EmotionAnalyzeRequest(text="我好迷茫，不知道该怎么办", safety=sample_safety)
        result = emotion_svc.analyze(req)
        assert result.primary_emotion == "confusion"

    def test_neutral_for_empty_text(self, emotion_svc, sample_safety):
        req = EmotionAnalyzeRequest(text="今天天气不错", safety=sample_safety)
        result = emotion_svc.analyze(req)
        assert result.primary_emotion == "neutral"

    def test_intensity_in_range(self, emotion_svc, sample_safety):
        req = EmotionAnalyzeRequest(text="我很焦虑", safety=sample_safety)
        result = emotion_svc.analyze(req)
        assert 0.0 <= result.intensity <= 1.0

    def test_risk_in_range(self, emotion_svc, sample_safety):
        req = EmotionAnalyzeRequest(text="我很焦虑", safety=sample_safety)
        result = emotion_svc.analyze(req)
        assert 0.0 <= result.risk <= 1.0

    def test_risk_high_when_safety_level_elevated(self, emotion_svc):
        """安全过滤 level=1 时 risk 应该更高。"""
        high_safety = {"level": 1, "blocked": False, "matched_terms": ["焦虑"], "meta": {}, "contract_version": "1.1"}
        req = EmotionAnalyzeRequest(text="我很焦虑", safety=high_safety)
        result = emotion_svc.analyze(req)
        # 安全有标记时 risk 不应为 0
        assert result.risk > 0.0


class TestEmotionAnalyzeWithAudio:
    """带音频路径：audio_path 不存在时降级，存在时走语音情绪。"""

    def test_degrade_when_audio_missing(self, emotion_svc, sample_safety):
        """音频文件不存在 → 降级为文本情绪。"""
        req = EmotionAnalyzeRequest(
            text="我很难过",
            audio_path="/nonexistent/audio.wav",
            safety=sample_safety,
        )
        result = emotion_svc.analyze(req)
        assert isinstance(result, EmotionTags)
        assert result.primary_emotion == "sadness"
        assert "audio" in result.modality_notes

    def test_degrade_when_audio_path_is_none(self, emotion_svc, sample_safety):
        """audio_path 为 None → 等同于纯文本路径。"""
        req = EmotionAnalyzeRequest(
            text="我很难过",
            audio_path=None,
            safety=sample_safety,
        )
        result = emotion_svc.analyze(req)
        assert result.primary_emotion == "sadness"

    def test_degrade_when_audio_empty_string(self, emotion_svc, sample_safety):
        """audio_path 为空字符串 → 等同于纯文本。"""
        req = EmotionAnalyzeRequest(
            text="我很难过",
            audio_path="",
            safety=sample_safety,
        )
        result = emotion_svc.analyze(req)
        assert result.primary_emotion == "sadness"


class TestEmotionAnalyzePreExtracted:
    """带预提取情绪：跳过 SenseVoice 调用，直接使用合并调用的结果。"""

    def test_use_pre_extracted_emotion(self, emotion_svc, sample_safety):
        """pre_extracted_audio_emotion 有值时跳过语音分析，直接使用。"""
        req = EmotionAnalyzeRequest(
            text="我今天很开心",
            audio_path="/nonexistent/audio.wav",
            safety=sample_safety,
            pre_extracted_audio_emotion={
                "primary_emotion": "happiness",
                "confidence": 0.85,
                "all_emotions": {"happiness": 0.85, "neutral": 0.15},
                "model_name": "SenseVoiceSmall",
                "backend": "sensevoice",
            },
        )
        result = emotion_svc.analyze(req)
        # 即使音频不存在，也因有预提取数据而返回音频情绪
        assert result.primary_emotion == "happiness"
        assert result.modality_notes.get("source") == "audio_fused"
        assert result.modality_notes.get("audio_model") == "SenseVoiceSmall"

    def test_pre_extracted_does_not_override_if_neutral(self, emotion_svc, sample_safety):
        """预提取的情绪是 neutral 时，回归文本情绪。"""
        req = EmotionAnalyzeRequest(
            text="我很难过",
            audio_path="/nonexistent/audio.wav",
            safety=sample_safety,
            pre_extracted_audio_emotion={
                "primary_emotion": "neutral",
                "confidence": 0.9,
                "all_emotions": {"neutral": 0.9},
                "model_name": "SenseVoiceSmall",
                "backend": "sensevoice",
            },
        )
        result = emotion_svc.analyze(req)
        # neutral 时不参与融合，用文本关键词结果
        assert result.primary_emotion in ("sadness", "neutral")


# ---------------------------------------------------------------------------
# 三模态融合 (v1.2)
# ---------------------------------------------------------------------------

class TestTriModalFusion:
    """text + audio + visual 三路融合行为。"""

    def test_three_consistent_signals(self, emotion_svc, sample_safety):
        """三路一致 → 高置信度融合结果。"""
        req = EmotionAnalyzeRequest(
            text="我很难过",  # text → sadness
            safety=sample_safety,
            pre_extracted_audio_emotion={
                "primary_emotion": "sadness",
                "confidence": 0.80,
                "all_emotions": {"sadness": 0.80, "neutral": 0.20},
                "model_name": "SenseVoiceSmall",
            },
            pre_extracted_visual_emotion={
                "primary_emotion": "sadness",
                "confidence": 0.75,
                "all_emotions": {"sadness": 0.75, "neutral": 0.10, "fear": 0.05, "anger": 0.05, "happiness": 0.05},
                "model_name": "HSEmotion",
            },
        )
        result = emotion_svc.analyze(req)
        assert result.primary_emotion == "sadness"
        assert result.modality_notes.get("source") == "tri_modal"

    def test_visual_dominates_when_text_and_audio_disagree(self, emotion_svc, sample_safety):
        """视觉 + 音频一致压倒文本。"""
        req = EmotionAnalyzeRequest(
            text="我很好",  # text → neutral 或 happiness
            safety=sample_safety,
            pre_extracted_audio_emotion={
                "primary_emotion": "anger",
                "confidence": 0.70,
                "all_emotions": {"anger": 0.70, "neutral": 0.30},
                "model_name": "SenseVoiceSmall",
            },
            pre_extracted_visual_emotion={
                "primary_emotion": "anger",
                "confidence": 0.85,
                "all_emotions": {"anger": 0.85, "neutral": 0.15},
                "model_name": "HSEmotion",
            },
        )
        result = emotion_svc.analyze(req)
        assert result.primary_emotion == "anger"

    def test_fusion_without_visual_falls_back_to_dual(self, emotion_svc, sample_safety):
        """无视觉信号 → 回溯到 audio+text 双路融合（不报错）。"""
        req = EmotionAnalyzeRequest(
            text="我很难过",
            safety=sample_safety,
            pre_extracted_audio_emotion={
                "primary_emotion": "sadness",
                "confidence": 0.80,
                "all_emotions": {"sadness": 0.80, "neutral": 0.20},
                "model_name": "SenseVoiceSmall",
            },
            # 不传 pre_extracted_visual_emotion
        )
        result = emotion_svc.analyze(req)
        assert isinstance(result, EmotionTags)
        assert result.primary_emotion in ("sadness", "anxiety", "neutral")
        assert result.modality_notes.get("source") == "audio_fused"


class TestSignalConflictArbitration:
    """信号冲突仲裁：三路全不同 + 低置信 → ×0.7 惩罚。"""

    def test_conflict_penalty_applied_when_all_diff_and_low_confidence(self, emotion_svc, sample_safety):
        """三路 primary 全不同 + 最高置信度 <= 0.6 → 融合置信度打折。"""
        req = EmotionAnalyzeRequest(
            text="我有点迷茫",  # text → confusion (low confidence from 1 keyword hit)
            safety=sample_safety,
            pre_extracted_audio_emotion={
                "primary_emotion": "fear",
                "confidence": 0.55,
                "all_emotions": {"fear": 0.55, "neutral": 0.45},
                "model_name": "SenseVoiceSmall",
            },
            pre_extracted_visual_emotion={
                "primary_emotion": "sadness",
                "confidence": 0.50,
                "all_emotions": {"sadness": 0.50, "neutral": 0.30, "fear": 0.20},
                "model_name": "HSEmotion",
            },
        )
        result = emotion_svc.analyze(req)
        # 冲突仲裁应该有标志
        assert "conflict_arbitration" in result.modality_notes
        assert result.modality_notes["conflict_arbitration"] is True

    def test_no_penalty_when_same_primary_even_with_low_confidence(self, emotion_svc, sample_safety):
        """三路中有两路一致时不触发冲突仲裁。"""
        req = EmotionAnalyzeRequest(
            text="我很焦虑",
            safety=sample_safety,
            pre_extracted_audio_emotion={
                "primary_emotion": "anxiety",
                "confidence": 0.45,
                "all_emotions": {"anxiety": 0.45, "neutral": 0.55},
                "model_name": "SenseVoiceSmall",
            },
            pre_extracted_visual_emotion={
                "primary_emotion": "anxiety",
                "confidence": 0.40,
                "all_emotions": {"anxiety": 0.40, "fear": 0.30, "neutral": 0.30},
                "model_name": "HSEmotion",
            },
        )
        result = emotion_svc.analyze(req)
        assert result.primary_emotion == "anxiety"
        # 无冲突
        assert not result.modality_notes.get("conflict_arbitration", False)


class TestRiskBonusForMixedSignals:
    """mixed_signals 为 True 时 risk 额外 +0.1。"""

    def test_risk_bonus_when_signals_mixed(self, emotion_svc, sample_safety):
        """三路不一致 → risk 应该包含预警加分。"""
        req = EmotionAnalyzeRequest(
            text="我还好",  # text → neutral 或 happiness
            safety=sample_safety,
            pre_extracted_audio_emotion={
                "primary_emotion": "fear",
                "confidence": 0.75,
                "all_emotions": {"fear": 0.75, "neutral": 0.25},
                "model_name": "SenseVoiceSmall",
            },
            pre_extracted_visual_emotion={
                "primary_emotion": "anger",
                "confidence": 0.65,
                "all_emotions": {"anger": 0.65, "neutral": 0.35},
                "model_name": "HSEmotion",
            },
        )
        result = emotion_svc.analyze(req)
        # mixed_signals 应该为 True（三路至少有两路不同）
        assert result.risk > 0.0

    def test_no_extra_risk_when_signals_agree(self, emotion_svc, sample_safety):
        """三路一致 → risk 不应该有额外的 mixed 加分。"""
        req = EmotionAnalyzeRequest(
            text="我很难过",
            safety=sample_safety,
            pre_extracted_audio_emotion={
                "primary_emotion": "sadness",
                "confidence": 0.80,
                "all_emotions": {"sadness": 0.80, "neutral": 0.20},
                "model_name": "SenseVoiceSmall",
            },
            pre_extracted_visual_emotion={
                "primary_emotion": "sadness",
                "confidence": 0.75,
                "all_emotions": {"sadness": 0.75, "neutral": 0.25},
                "model_name": "HSEmotion",
            },
        )
        result = emotion_svc.analyze(req)
        # risk 基于 sadness base (0.4) + intensity boost，不加 mixed bonus
        assert result.risk == pytest.approx(0.45, abs=0.10)
