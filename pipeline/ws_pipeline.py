"""
WebSocket 实时语音对话的管道服务。

以回调驱动的方式运行管线：音频 bytes → ASR → Safety → Emotion → Router →
LLM 流式生成 → 逐句 TTS → 推送状态/文本/音频至前端。

可注入 PipelineServices 走完整管道，也可注入 stub ASR/LLM/TTS 做单元测试。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

from pipeline.sentence_tts import SentenceTTS
from schemas.contracts import InterventionRequest, PipelineInput

logger = logging.getLogger(__name__)

# 降级安抚话术（ASR 空但有情绪信号时，直接 TTS 推送，不走 LLM）
_COMFORT_FALLBACK = "我在听，慢慢说，不着急。"


class Callbacks(ABC):
    """WebSocket 管道向客户端推送消息的回调接口。"""

    @abstractmethod
    async def on_status(self, phase: str) -> None:
        """管道阶段变更：asr / safety / emotion / generating。"""
        ...

    @abstractmethod
    async def on_user_text(self, text: str) -> None:
        """ASR 识别出的用户文本。"""
        ...

    @abstractmethod
    async def on_text_delta(self, content: str) -> None:
        """LLM 流式文本增量（字幕）。"""
        ...

    @abstractmethod
    async def on_audio_delta(self, chunk: bytes) -> None:
        """TTS 流式音频 chunk。"""
        ...

    @abstractmethod
    async def on_emotion_result(self, data: Dict[str, Any]) -> None:
        """情绪分析结果。"""
        ...

    @abstractmethod
    async def on_done(self, session_id: str) -> None:
        """本轮处理完成。"""
        ...

    @abstractmethod
    async def on_error(self, message: str) -> None:
        """处理过程中出错。"""
        ...


class WebSocketPipeline:
    """WebSocket 版管线。

    两种使用模式：

    1. **测试模式**：注入 ``asr`` / ``llm`` / ``tts`` 桩，内部走简化逻辑
    2. **生产模式**：注入 ``services`` (PipelineServices) + ``asr`` (真实 ASRAdapter) + ``tts`` (真实 TTS 引擎)，
       走完整 Safety → Emotion → Router → 流式 Intervention → 逐句 TTS
    """

    def __init__(
        self,
        *,
        asr: Any = None,
        llm: Any = None,
        tts: Any = None,
        services: Any = None,
    ):
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._services = services

    @property
    def _has_services(self) -> bool:
        return self._services is not None

    async def run(
        self,
        audio_bytes: bytes,
        callbacks: Callbacks,
        *,
        session_id: str = "test-session",
        user_id: Optional[str] = None,
    ) -> None:
        """执行完整管道，通过 callbacks 推送所有中间和最终结果。"""
        try:
            # ── 1. ASR ──────────────────────────────────
            await callbacks.on_status("asr")
            try:
                text, audio_emotion = self._asr.transcribe(audio_bytes)
            except Exception as exc:
                logger.warning("ASR failed: %s", exc)
                await callbacks.on_error(str(exc))
                return

            # ── 降级：ASR 文本为空 ─────────────────────
            if not text or not text.strip():
                await self._handle_empty_text(audio_emotion, callbacks, session_id)
                return

            # 推送用户文本到前端
            await callbacks.on_user_text(text)

            if self._has_services:
                await self._run_with_services(
                    text, audio_emotion, callbacks, session_id, user_id
                )
            else:
                await self._run_stubbed(text, audio_emotion, callbacks, session_id)

        except Exception as exc:
            logger.exception("WebSocket pipeline error")
            await callbacks.on_error(str(exc))

    # ── production path ────────────────────────────────────────

    async def _run_with_services(
        self,
        text: str,
        audio_emotion: Optional[Dict[str, Any]],
        callbacks: Callbacks,
        session_id: str,
        user_id: Optional[str],
    ) -> None:
        """使用真实 PipelineServices 的完整管线。"""
        # 构建 PipelineInput
        inp = PipelineInput(
            text=text,
            user_id=user_id,
            pre_extracted_audio_emotion=audio_emotion,
            session_id=session_id,
        )

        # Safety → Emotion → Router（复用现有管线）
        await callbacks.on_status("safety")
        from pipeline.orchestrator import run_pipeline_pre

        pre_result = run_pipeline_pre(inp, services=self._services)

        safety_dict = pre_result["safety"]
        emotion_dict = pre_result["emotion"]
        route_dict = pre_result["route"]
        stopped = pre_result["stopped_after_safety"]

        # 推送情绪结果（融合后的）
        await callbacks.on_status("emotion")
        await callbacks.on_emotion_result(emotion_dict)

        # 流式干预 + 逐句 TTS
        await callbacks.on_status("generating")
        intervention_req = InterventionRequest(
            contract_version=inp.contract_version,
            user_text=text,
            route=route_dict,
            emotion=emotion_dict,
            safety=safety_dict,
            session_id=session_id,
        )
        tts = SentenceTTS(self._tts)
        async for token in self._services.intervention.astream_intervene(
            intervention_req
        ):
            await callbacks.on_text_delta(token)
            async for chunk in tts.feed_token(token):
                await callbacks.on_audio_delta(chunk)

        # 尾部残余
        async for chunk in tts.flush():
            await callbacks.on_audio_delta(chunk)

        await callbacks.on_done(session_id)

    # ── test / stub path (no PipelineServices) ────────────────

    async def _run_stubbed(
        self,
        text: str,
        audio_emotion: Optional[Dict[str, Any]],
        callbacks: Callbacks,
        session_id: str,
    ) -> None:
        """无 PipelineServices 时的简化路径（单元测试用）。"""
        await callbacks.on_status("safety")
        await callbacks.on_status("emotion")
        emotion_data = self._build_emotion_result(text, audio_emotion)
        await callbacks.on_emotion_result(emotion_data)

        await callbacks.on_status("generating")
        await self._stream_stubbed_reply(text, emotion_data, callbacks)
        await callbacks.on_done(session_id)

    async def _handle_empty_text(
        self,
        audio_emotion: Optional[Dict[str, Any]],
        callbacks: Callbacks,
        session_id: str,
    ) -> None:
        """ASR 文本为空时的降级处理。"""
        if audio_emotion and audio_emotion.get("primary_emotion"):
            logger.info("Empty ASR text but emotion detected (%s), using comfort fallback",
                        audio_emotion.get("primary_emotion"))
            await callbacks.on_status("emotion")
            await callbacks.on_emotion_result(audio_emotion)
            await callbacks.on_status("generating")
            await callbacks.on_text_delta(_COMFORT_FALLBACK)
            async for chunk in self._tts.synthesize(_COMFORT_FALLBACK):
                await callbacks.on_audio_delta(chunk)
            await callbacks.on_done(session_id)
        else:
            logger.warning("Empty ASR text and no audio emotion — possible silence or low volume")
            await callbacks.on_error("未检测到语音，请重试")

    def _build_emotion_result(
        self,
        text: str,
        audio_emotion: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if audio_emotion:
            return dict(audio_emotion)
        return {
            "primary_emotion": "neutral",
            "confidence": 0.5,
            "all_emotions": {"neutral": 0.5},
            "model_name": "fallback",
            "backend": "heuristic",
        }

    async def _stream_stubbed_reply(
        self,
        user_text: str,
        emotion_data: Dict[str, Any],
        callbacks: Callbacks,
    ) -> None:
        """stub 模式：LLM 桩 → 逐句 TTS。"""
        prompt = (
            f"用户说：{user_text}\n"
            f"用户情绪：{emotion_data.get('primary_emotion', 'neutral')}\n"
            f"请给出共情回复："
        )
        token_stream = self._llm.astream(prompt)
        tts = SentenceTTS(self._tts)

        async for token in token_stream:
            await callbacks.on_text_delta(token)
            async for chunk in tts.feed_token(token):
                await callbacks.on_audio_delta(chunk)

        async for chunk in tts.flush():
            await callbacks.on_audio_delta(chunk)
