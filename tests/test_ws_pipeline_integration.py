"""WebSocket 管道集成测试：真实 mock 服务 + 流式干预。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List

import pytest

from config.settings import Settings, settings as default_settings
from modules.factory import PipelineServices, build_pipeline_services
from pipeline.ws_pipeline import Callbacks, WebSocketPipeline


# ── test doubles ──────────────────────────────────────────────


@dataclass
class SpyCallbacks(Callbacks):
    statuses: List[str] = field(default_factory=list)
    text_deltas: List[str] = field(default_factory=list)
    audio_chunks: List[bytes] = field(default_factory=list)
    emotion_results: List[Dict[str, Any]] = field(default_factory=list)
    done_session_ids: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    async def on_status(self, phase: str) -> None:
        self.statuses.append(phase)

    async def on_text_delta(self, content: str) -> None:
        self.text_deltas.append(content)

    async def on_audio_delta(self, chunk: bytes) -> None:
        self.audio_chunks.append(chunk)

    async def on_emotion_result(self, data: Dict[str, Any]) -> None:
        self.emotion_results.append(data)

    async def on_done(self, session_id: str) -> None:
        self.done_session_ids.append(session_id)

    async def on_error(self, message: str) -> None:
        self.errors.append(message)


class IntegratedASR:
    """真实 ASRAdapter 接口的测试双。"""

    def transcribe(self, audio_bytes: bytes, recognizer=None):
        return (
            "我最近总是失眠，很焦虑",
            {
                "primary_emotion": "anxiety",
                "confidence": 0.72,
                "all_emotions": {"anxiety": 0.72},
                "model_name": "SenseVoiceSmall",
                "backend": "sensevoice",
            },
        )


class RealTTS:
    """TTS 引擎的测试双（SentenceTTS 兼容接口）。"""

    def __init__(self):
        self.synthesized: List[str] = []

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.synthesized.append(text)
        # 产出简化的音频 chunk
        yield text.encode()[:16]
        yield text.encode()[16:] if len(text.encode()) > 16 else b""


# ── tests ─────────────────────────────────────────────────────


@pytest.fixture
def mock_services() -> PipelineServices:
    """使用项目 mock 服务。MockSafety 永远是 safe，MockEmotion 固定返回值，
    MockRouter 固定 general，MockIntervention 固定回复。"""
    cfg = Settings(
        MOCK_SAFETY=True,
        MOCK_EMOTION=True,
        MOCK_ROUTER=True,
        MOCK_INTERVENTION=True,
    )
    return build_pipeline_services(cfg)


class TestWebSocketPipelineWithMockServices:
    """使用项目 Mock 服务的集成测试。"""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_mock_services(self, mock_services):
        spy = SpyCallbacks()
        tts = RealTTS()

        pipeline = WebSocketPipeline(
            asr=IntegratedASR(),
            tts=tts,
            services=mock_services,
        )
        await pipeline.run(b"fake_audio", spy)

        # 应该完整跑完
        assert spy.statuses == ["asr", "safety", "emotion", "generating"]
        assert len(spy.done_session_ids) == 1
        assert len(spy.errors) == 0

    @pytest.mark.asyncio
    async def test_text_deltas_from_streaming_intervention(self, mock_services):
        spy = SpyCallbacks()
        tts = RealTTS()

        pipeline = WebSocketPipeline(
            asr=IntegratedASR(),
            tts=tts,
            services=mock_services,
        )
        await pipeline.run(b"fake", spy)

        # Mock 干预会返回固定文本
        assert len(spy.text_deltas) > 0

    @pytest.mark.asyncio
    async def test_tts_called_with_intervention_text(self, mock_services):
        tts = RealTTS()

        pipeline = WebSocketPipeline(
            asr=IntegratedASR(),
            tts=tts,
            services=mock_services,
        )
        await pipeline.run(b"fake", SpyCallbacks())

        # TTS 引擎被调用了（至少一句完整的文本）
        assert len(tts.synthesized) > 0

    @pytest.mark.asyncio
    async def test_pipeline_uses_asr_emotion_in_pipeline(self, mock_services):
        spy = SpyCallbacks()

        pipeline = WebSocketPipeline(
            asr=IntegratedASR(),
            tts=RealTTS(),
            services=mock_services,
        )
        await pipeline.run(b"fake", spy)

        # emotion_result 来自 ASR 提取的情绪
        assert len(spy.emotion_results) == 1
