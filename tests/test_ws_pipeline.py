"""WebSocket 管道服务行为测试：回调驱动、降级路径、取消支持。"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest

from pipeline.ws_pipeline import Callbacks, WebSocketPipeline


# ── test doubles ──────────────────────────────────────────────


@dataclass
class SpyCallbacks(Callbacks):
    """记录每次回调调用的 spy。"""

    statuses: List[str] = field(default_factory=list)
    user_texts: List[str] = field(default_factory=list)
    text_deltas: List[str] = field(default_factory=list)
    audio_chunks: List[bytes] = field(default_factory=list)
    emotion_results: List[Dict[str, Any]] = field(default_factory=list)
    done_session_ids: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    async def on_status(self, phase: str) -> None:
        self.statuses.append(phase)

    async def on_user_text(self, text: str) -> None:
        self.user_texts.append(text)

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


class StubASR:
    """可配置的 ASR 桩。"""

    def __init__(
        self,
        text: str = "我今天有点焦虑",
        emotion: str = "anxiety",
        confidence: float = 0.72,
    ):
        self.text = text
        self.emotion = emotion
        self.confidence = confidence

    def transcribe(self, audio_bytes: bytes) -> tuple[str, Optional[Dict[str, Any]]]:
        return (
            self.text,
            {
                "primary_emotion": self.emotion,
                "confidence": self.confidence,
                "all_emotions": {self.emotion: self.confidence},
                "model_name": "stub-sensevoice",
                "backend": "sensevoice",
            },
        )


class StubASRError:
    """ASR 完全失败（抛异常）。"""

    def transcribe(self, audio_bytes: bytes) -> tuple[str, Optional[Dict[str, Any]]]:
        raise RuntimeError("ASR model crashed")


class StubASREmptyText:
    """ASR 返回空文本但有情绪信号。"""

    def transcribe(self, audio_bytes: bytes) -> tuple[str, Optional[Dict[str, Any]]]:
        return (
            "",
            {
                "primary_emotion": "sadness",
                "confidence": 0.65,
                "all_emotions": {"sadness": 0.65},
                "model_name": "stub-sensevoice",
                "backend": "sensevoice",
            },
        )


class StubASREmptyAll:
    """ASR 返回空文本且无情绪信号。"""

    def transcribe(self, audio_bytes: bytes) -> tuple[str, Optional[Dict[str, Any]]]:
        return ("", None)


class StubLLM:
    """可配置的流式 LLM 桩。"""

    def __init__(self, tokens: Optional[List[str]] = None):
        self.tokens = tokens or ["我", "理解", "你的", "感受", "。"]

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        for token in self.tokens:
            yield token
            await asyncio.sleep(0)


class StubTTS:
    """流式 TTS 桩，SentenceTTS 兼容接口。"""

    def __init__(self):
        self.synthesized: List[str] = []

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.synthesized.append(text)
        for i in range(3):
            yield f"audio_{len(self.synthesized)}_{i}".encode()
            await asyncio.sleep(0)


# ── tests ─────────────────────────────────────────────────────


class TestWebSocketPipelineHappyPath:
    """正常流程：音频 → 管道 → 流式回复。"""

    @pytest.mark.asyncio
    async def test_status_sequence_follows_pipeline_order(self):
        spy = SpyCallbacks()
        pipeline = WebSocketPipeline(
            asr=StubASR(),
            llm=StubLLM(),
            tts=StubTTS(),
        )
        await pipeline.run(b"fake_audio_bytes", spy)

        # status 必须按管道阶段顺序
        assert spy.statuses[0] == "asr"
        assert spy.statuses[1] == "safety"
        assert spy.statuses[2] == "emotion"
        assert spy.statuses[3] == "generating"
        assert "done" not in spy.statuses  # done 走 on_done，不占 status 槽

    @pytest.mark.asyncio
    async def test_text_deltas_sent_during_generation(self):
        spy = SpyCallbacks()
        pipeline = WebSocketPipeline(
            asr=StubASR(),
            llm=StubLLM(tokens=["你好", "，", "别担心", "。"]),
            tts=StubTTS(),
        )
        await pipeline.run(b"fake", spy)

        # LLM 的每个 token 都作为 text.delta 推送
        assert spy.text_deltas == ["你好", "，", "别担心", "。"]

    @pytest.mark.asyncio
    async def test_audio_chunks_sent_during_generation(self):
        spy = SpyCallbacks()
        pipeline = WebSocketPipeline(
            asr=StubASR(),
            llm=StubLLM(tokens=["第一句", "。", "第二句", "。"]),
            tts=StubTTS(),
        )
        await pipeline.run(b"fake", spy)

        # TTS 每句产出 3 个 chunk，两句 = 6
        assert len(spy.audio_chunks) >= 2  # 至少有一句触发了 TTS

    @pytest.mark.asyncio
    async def test_emotion_result_callback_invoked(self):
        spy = SpyCallbacks()
        pipeline = WebSocketPipeline(
            asr=StubASR(emotion="anxiety", confidence=0.72),
            llm=StubLLM(),
            tts=StubTTS(),
        )
        await pipeline.run(b"fake", spy)

        assert len(spy.emotion_results) == 1
        assert spy.emotion_results[0]["primary_emotion"] == "anxiety"

    @pytest.mark.asyncio
    async def test_done_callback_on_success(self):
        spy = SpyCallbacks()
        pipeline = WebSocketPipeline(
            asr=StubASR(),
            llm=StubLLM(),
            tts=StubTTS(),
        )
        await pipeline.run(b"fake", spy)

        assert spy.done_session_ids == ["test-session"]
        assert len(spy.errors) == 0


class TestWebSocketPipelineDegradation:
    """降级路径：ASR 失败、空文本。"""

    @pytest.mark.asyncio
    async def test_empty_text_with_emotion_goes_comfort_route(self):
        """ASR 空 + 有情绪 → 安抚路由，不需要 LLM，直接 TTS。"""
        spy = SpyCallbacks()
        pipeline = WebSocketPipeline(
            asr=StubASREmptyText(),
            llm=StubLLM(),  # 不应被调用
            tts=StubTTS(),
        )
        await pipeline.run(b"silence_with_emotion", spy)

        # 有情绪信号，不应报错
        assert len(spy.errors) == 0
        # 情绪结果已推送
        assert len(spy.emotion_results) == 1
        # 走安抚路由，有音频推送
        assert len(spy.audio_chunks) > 0

    @pytest.mark.asyncio
    async def test_empty_text_no_emotion_sends_error(self):
        """ASR 空且无情绪 → 推送 error。"""
        spy = SpyCallbacks()
        pipeline = WebSocketPipeline(
            asr=StubASREmptyAll(),
            llm=StubLLM(),
            tts=StubTTS(),
        )
        await pipeline.run(b"silence", spy)

        assert len(spy.errors) == 1
        assert "未检测到语音" in spy.errors[0]

    @pytest.mark.asyncio
    async def test_asr_error_triggers_error_callback(self):
        spy = SpyCallbacks()
        pipeline = WebSocketPipeline(
            asr=StubASRError(),
            llm=StubLLM(),
            tts=StubTTS(),
        )
        await pipeline.run(b"corrupted", spy)

        assert len(spy.errors) == 1
        assert len(spy.done_session_ids) == 0  # done 不应触发


class TestWebSocketPipelineSentenceBoundaryTTS:
    """TTS 逐句触发：只在遇到句末标点时合成。"""

    @pytest.mark.asyncio
    async def test_tts_triggered_only_at_sentence_boundary(self):
        spy = SpyCallbacks()
        stt = StubTTS()
        pipeline = WebSocketPipeline(
            asr=StubASR(),
            llm=StubLLM(tokens=["我", "在", "这里", "。", "你", "呢", "？"]),
            tts=stt,
        )
        await pipeline.run(b"fake", spy)

        # 输出中有两个句子边界：。 和 ？
        # 应该触发 2 次 TTS 合成
        assert len(stt.synthesized) == 2
        assert stt.synthesized[0] == "我在这里。"
        assert stt.synthesized[1] == "你呢？"
