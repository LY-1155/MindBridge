"""
视频管线端到端测试：run_video_pipeline → 四阶段全跑通。
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from pipeline.orchestrator import run_video_pipeline
from schemas.contracts.v1 import PipelineOutput


def _fake_preprocess_result():
    """构造一个 VideoPreprocessor 的输出。"""
    from multimodal.video_preprocessor import PreprocessResult
    return PreprocessResult(
        text="我今天很焦虑",
        audio_emotion={
            "primary_emotion": "anxiety",
            "confidence": 0.82,
            "all_emotions": {"anxiety": 0.82, "neutral": 0.18},
            "model_name": "SenseVoiceSmall",
            "backend": "sensevoice",
        },
        visual_emotion={
            "primary_emotion": "sadness",
            "confidence": 0.75,
            "all_emotions": {"sadness": 0.75, "neutral": 0.15, "fear": 0.10},
            "model_name": "HSEmotion",
            "face_detection_rate": 0.8,
            "valid_frames": 4,
            "total_frames": 5,
        },
        warnings=[],
    )


class TestRunVideoPipeline:
    """端到端：视频 → 四阶段 → PipelineOutput。"""

    def test_video_pipeline_returns_pipeline_output(self, tmp_path):
        """视频管线应该产出有效的 PipelineOutput。"""
        video_path = str(tmp_path / "test.mp4")
        with open(video_path, "wb") as f:
            f.write(b"\x00" * 1024)

        with patch(
            "multimodal.video_preprocessor.VideoPreprocessor.process",
            return_value=_fake_preprocess_result(),
        ):
            result = run_video_pipeline(video_path=video_path)

        assert isinstance(result, PipelineOutput)
        assert result.contract_version == "1.2"
        assert "level" in result.safety
        assert result.emotion["primary_emotion"] in (
            "neutral", "anxiety", "sadness", "anger",
            "fear", "stress", "happiness", "confusion",
        )
        assert result.route["route"] in ("general", "comfort", "knowledge", "crisis")
        assert "reply" in result.intervention
        assert result.stopped_after_safety is False

    def test_video_pipeline_with_safety_shortcut(self, tmp_path):
        """紧急文本触发安全短路时，跳过 emotion/route，直接 crisis 干预。"""
        video_path = str(tmp_path / "test.mp4")
        with open(video_path, "wb") as f:
            f.write(b"\x00" * 1024)

        with patch(
            "multimodal.video_preprocessor.VideoPreprocessor.process",
            return_value=_fake_preprocess_result(),
        ):
            result = run_video_pipeline(
                video_path=video_path,
                safety_text="我要自杀",
            )

        assert result.stopped_after_safety is True
        assert result.route["route"] == "crisis"
        assert result.emotion["primary_emotion"] == "distress"
