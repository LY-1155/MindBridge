"""ASRAdapter 行为测试：音频 bytes → 文本 + 情绪，含退化路径。"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest

from pipeline.asr_adapter import ASRAdapter


# ── stub converter (skip ffmpeg) ─────────────────────────────


def _stub_converter(src: str, dst: str) -> bool:
    """模拟 ffmpeg 转码：直接把 src 内容复制到 dst（测试用）。"""
    import shutil
    shutil.copyfile(src, dst)
    return True


# ── test doubles ──────────────────────────────────────────────


@dataclass
class FakeSpeechRecognizer:
    """模拟 SpeechRecognizer，可配置原始输出。"""

    raw_output: str = "<|zh|>我今天很焦虑<|sad|>"
    fail: bool = False

    def transcribe(self, audio_path, language=None, task="transcribe"):
        if self.fail:
            raise RuntimeError("SenseVoice crashed")
        from multimodal.asr import TranscriptionResult

        return TranscriptionResult(
            text="我今天很焦虑",
            language="zh",
            segments=[],
            duration=1.5,
            asr_backend="sensevoice",
        )

    @property
    def sensevoice_model(self):
        raw = self.raw_output

        class FakeSVModel:
            def generate(self, input, use_itn=True):
                return [{"text": raw}]

        return FakeSVModel()


class FakeSpeechRecognizerNoEmotion:
    """模拟 SenseVoice 输出无情绪标签。"""

    raw_output = "<|zh|>今天天气不错"

    def transcribe(self, audio_path, language=None, task="transcribe"):
        from multimodal.asr import TranscriptionResult

        return TranscriptionResult(
            text="今天天气不错",
            language="zh",
            segments=[],
            duration=1.0,
            asr_backend="sensevoice",
        )

    @property
    def sensevoice_model(self):
        raw = self.raw_output

        class FakeSVModel:
            def generate(self, input, use_itn=True):
                return [{"text": raw}]

        return FakeSVModel()


# ── tests ─────────────────────────────────────────────────────


class TestASRAdapterHappyPath:
    """正常流程：有文本 + 有情绪。"""

    def test_returns_transcribed_text(self):
        adapter = ASRAdapter(converter=_stub_converter)
        text, emotion = adapter.transcribe(
            b"fake_audio", recognizer=FakeSpeechRecognizer()
        )
        assert text == "我今天很焦虑"

    def test_returns_audio_emotion_from_sensevoice_tags(self):
        adapter = ASRAdapter(converter=_stub_converter)
        _, emotion = adapter.transcribe(
            b"fake_audio", recognizer=FakeSpeechRecognizer()
        )
        assert emotion is not None
        assert emotion["primary_emotion"] == "sadness"
        assert emotion["backend"] == "sensevoice"

    def test_accepts_arbitrary_audio_bytes(self):
        adapter = ASRAdapter(converter=_stub_converter)
        # 应该自动创建临时文件、写入 bytes、清理
        text, emotion = adapter.transcribe(
            bytes(range(256)), recognizer=FakeSpeechRecognizer()
        )
        assert text == "我今天很焦虑"

    def test_bytes_input_does_not_require_file_path(self):
        """适配器接收 raw bytes，不要求调用方提供路径。"""
        adapter = ASRAdapter(converter=_stub_converter)
        text, emotion = adapter.transcribe(
            b"\x00\x01\x02", recognizer=FakeSpeechRecognizer()
        )
        assert isinstance(text, str)
        assert len(text) > 0


class TestASRAdapterDegradation:
    """退化路径。"""

    def test_no_emotion_tag_returns_none_emotion(self):
        adapter = ASRAdapter(converter=_stub_converter)
        text, emotion = adapter.transcribe(
            b"neutral audio", recognizer=FakeSpeechRecognizerNoEmotion()
        )
        assert text == "今天天气不错"
        assert emotion is None  # 无情绪标签 → None

    def test_sensevoice_unavailable_falls_back_to_whisper(self):
        """SenseVoice 失败 → 回退到 Whisper（只有文本，无情绪）。"""
        adapter = ASRAdapter(converter=_stub_converter)

        # 模拟 SenseVoice 模型加载失败
        class SenseVoiceFailRecognizer:
            raw_output = ""

            def transcribe(self, audio_path, language=None, task="transcribe"):
                from multimodal.asr import TranscriptionResult

                return TranscriptionResult(
                    text="fallback text from whisper",
                    language="zh",
                    segments=[],
                    duration=1.0,
                    asr_backend="whisper",  # 已回退
                )

            @property
            def sensevoice_model(self):
                raise ImportError("funasr not installed")

        text, emotion = adapter.transcribe(
            b"test", recognizer=SenseVoiceFailRecognizer()
        )
        assert text == "fallback text from whisper"
        assert emotion is None

    def test_sensevoice_generate_fails_gracefully(self):
        """SenseVoice 加载成功但推理失败。"""
        adapter = ASRAdapter(converter=_stub_converter)

        class GenerateFailRecognizer:
            raw_output = ""

            def transcribe(self, audio_path, language=None, task="transcribe"):
                raise RuntimeError("GPU OOM during inference")

            @property
            def sensevoice_model(self):
                class FailModel:
                    def generate(self, input, use_itn=True):
                        raise RuntimeError("CUDA out of memory")

                return FailModel()

        text, emotion = adapter.transcribe(
            b"test", recognizer=GenerateFailRecognizer()
        )
        # 应该返回空回退，不崩溃
        assert isinstance(text, str)
        assert emotion is None
