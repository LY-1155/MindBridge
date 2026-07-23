"""
VideoPreprocessor 行为测试：视频解构 → 三路信号产出。
依赖注入允许 mock 模型和外部调用（ffmpeg/SenseVoice/MediaPipe/HSEmotion）。
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import Mock, patch

import numpy as np
import pytest

from multimodal.video_preprocessor import PreprocessResult, VideoPreprocessor


# ---------------------------------------------------------------------------
# Helpers — synthetic test video
# ---------------------------------------------------------------------------

def _create_synthetic_video(path: str, num_frames: int = 100, width: int = 640, height: int = 480, fps: int = 25):
    """创建一个合成测试视频（4 秒，100 帧），抽帧后至少有 4 帧可检测到"人脸"。"""
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if fourcc == -1:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (i * 25 % 255, 100, 150)
        writer.write(frame)
    writer.release()


# ---------------------------------------------------------------------------
# Mock helpers — simulate model outputs at system boundaries
# ---------------------------------------------------------------------------

def _fake_asr_fn(audio_path: str) -> str:
    return "我今天很焦虑，睡不着觉"


def _fake_audio_emotion_fn(audio_path: str) -> dict:
    return {
        "primary_emotion": "anxiety",
        "confidence": 0.82,
        "all_emotions": {"anxiety": 0.82, "neutral": 0.18},
        "model_name": "SenseVoiceSmall",
        "backend": "sensevoice",
    }


def _fake_face_detector(frame: np.ndarray) -> list:
    """返回一个虚构的人脸边界框。"""
    return [(100, 80, 200, 240)]


def _fake_emotion_model(face_img) -> dict:
    """模拟 HSEmotion 返回 Ekman 7 类分布。"""
    return {
        "primary_emotion": "sad",
        "confidence": 0.75,
        "all_emotions": {
            "sad": 0.75, "neutral": 0.10, "fear": 0.05,
            "angry": 0.03, "happy": 0.02, "disgust": 0.03, "surprise": 0.02,
        },
    }


# ---------------------------------------------------------------------------
# Tracer bullet test
# ---------------------------------------------------------------------------

class TestVideoPreprocessorEndToEnd:
    """端到端：视频文件 → text + audio_emotion + visual_emotion。"""

    @pytest.fixture
    def tmp_video(self, tmp_path) -> str:
        path = str(tmp_path / "test.mp4")
        _create_synthetic_video(path, num_frames=15, fps=25)
        return path

    @pytest.fixture
    def tmp_wav(self, tmp_path) -> str:
        """一个 1 秒的 16kHz 单声道 WAV（静音），用于模拟音频输入。"""
        import wave
        path = str(tmp_path / "test.wav")
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000)
        return path

    @pytest.fixture
    def preprocessor(self):
        return VideoPreprocessor(
            asr_fn=_fake_asr_fn,
            audio_emotion_fn=_fake_audio_emotion_fn,
            face_detector=_fake_face_detector,
            emotion_model=_fake_emotion_model,
        )

    def test_process_video_produces_all_three_signals(self, preprocessor, tmp_video, tmp_wav):
        """Tracer bullet：视频 + 独立音频 → text、audio_emotion、visual_emotion。"""
        result = preprocessor.process(tmp_video, audio_path=tmp_wav)

        assert isinstance(result, PreprocessResult)
        assert result.text == "我今天很焦虑，睡不着觉"
        assert result.audio_emotion is not None
        assert result.audio_emotion["primary_emotion"] == "anxiety"
        assert result.visual_emotion is not None
        assert "primary_emotion" in result.visual_emotion
        assert "confidence" in result.visual_emotion
        assert "all_emotions" in result.visual_emotion

    def test_process_video_without_audio_degraded(self, preprocessor, tmp_video):
        """无音频时 text 为空但不崩溃，视觉信号正常。"""
        result = preprocessor.process(tmp_video)

        assert isinstance(result, PreprocessResult)
        assert result.visual_emotion is not None
        # 无音频无 ASR，text 为空但 audio_emotion 为 None
        assert result.audio_emotion is None


class TestVideoPreprocessorDegradation:
    """降级路径：无面视频、抽帧失败等。"""

    @pytest.fixture
    def tmp_video(self, tmp_path) -> str:
        path = str(tmp_path / "test.mp4")
        _create_synthetic_video(path, num_frames=100, fps=25)
        return path

    @pytest.fixture
    def tmp_wav(self, tmp_path) -> str:
        import wave
        path = str(tmp_path / "test.wav")
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000)
        return path

    def test_no_faces_degradation(self, tmp_video, tmp_wav):
        """视频全程无面 → visual_emotion=None + 降级 warning。"""
        preprocessor = VideoPreprocessor(
            asr_fn=_fake_asr_fn,
            audio_emotion_fn=_fake_audio_emotion_fn,
            face_detector=lambda frame: [],  # 永远检测不到脸
            emotion_model=_fake_emotion_model,
        )
        result = preprocessor.process(tmp_video, audio_path=tmp_wav)

        assert result.visual_emotion is None
        # 必须有降级 warning
        degraded = [w for w in result.warnings if "degraded" in w.lower()]
        assert len(degraded) > 0
        # 音频和文本仍然正常
        assert result.text is not None
        assert result.audio_emotion is not None
