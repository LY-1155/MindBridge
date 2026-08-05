"""
WebSocket 实时语音对话端点。

路径：``/ws/voice``
协议：JSON 控制帧 + 二进制音频帧，首个字节区分。
认证：URL 参数 ``token``（JWT access token）。
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, status
from fastapi.exceptions import HTTPException

from pipeline.audio_buffer import AudioBuffer

logger = logging.getLogger(__name__)

router = APIRouter()


async def _authenticate_ws(ws: WebSocket, token: Optional[str]) -> str:
    """从 URL 参数解析 JWT token，返回 user_id。"""
    if not token:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="Missing token")
        raise WebSocketDisconnect(code=status.WS_1008_POLICY_VIOLATION)

    from modules.token_service import TokenService

    try:
        payload = TokenService.verify_token(token)
        if payload.get("type") != "access":
            await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="Not an access token")
            raise WebSocketDisconnect(code=status.WS_1008_POLICY_VIOLATION)
        user_id = payload.get("user_id")
        if not user_id:
            raise ValueError("Token missing user_id")
        return user_id
    except Exception as exc:
        logger.warning("WebSocket auth failed: %s", exc)
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token")
        raise WebSocketDisconnect(code=status.WS_1008_POLICY_VIOLATION)


@router.websocket("/ws/voice")
async def voice_ws_endpoint(
    ws: WebSocket,
    token: Optional[str] = Query(default=None),
    session_id: Optional[str] = Query(default=None),
):
    """实时语音对话 WebSocket。

    前端：
    1. 建立连接：``new WebSocket("ws://host/ws/voice?token=xxx&session_id=yyy")``
    2. 按住录音时发送二进制 audio chunk
    3. 松手时发送 JSON ``{"type":"audio.end"}``
    4. 监听 ``status`` / ``text.delta`` / ``audio.delta`` / ``emotion.result`` / ``done`` / ``error``
    """
    await ws.accept()

    try:
        user_id = await _authenticate_ws(ws, token)
        logger.info("WebSocket voice connected: user=%s session=%s", user_id, session_id)
    except WebSocketDisconnect:
        return

    # 解析或创建 session_id
    if not session_id:
        try:
            from core.memory.session_memory import SessionManager
            session_id = SessionManager.create_session(user_id=user_id)
        except Exception as exc:
            logger.warning("SessionManager unavailable, using fallback id: %s", exc)
            import uuid
            session_id = f"ws_{uuid.uuid4().hex[:8]}"

    buffer = AudioBuffer()

    # 发送连接确认
    await ws.send_json({"type": "connected", "session_id": session_id})

    try:
        while True:
            data = await ws.receive()

            if "bytes" in data:
                # 二进制 → 音频 chunk
                buffer.feed(data["bytes"])

            elif "text" in data:
                msg = json.loads(data["text"])
                msg_type = msg.get("type", "")

                if msg_type == "audio.end":
                    # 用户松手 → 跑管道
                    audio_bytes = buffer.end()
                    if not audio_bytes:
                        await ws.send_json({"type": "error", "message": "未收到音频数据"})
                        continue

                    await _run_pipeline_and_stream(
                        ws=ws,
                        audio_bytes=audio_bytes,
                        session_id=session_id,
                        user_id=user_id,
                    )

                elif msg_type == "cancel":
                    # 用户取消
                    buffer.cancel()
                    await ws.send_json({"type": "cancelled"})

    except WebSocketDisconnect:
        logger.info("WebSocket voice disconnected: user=%s session=%s", user_id, session_id)
    except Exception as exc:
        logger.exception("WebSocket voice error: user=%s", user_id)
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


async def _run_pipeline_and_stream(
    ws: WebSocket,
    audio_bytes: bytes,
    session_id: str,
    user_id: str,
) -> None:
    """运行管道并通过 WebSocket 逐帧推送结果。"""
    from config.settings import settings
    from modules.factory import PipelineServices, build_pipeline_services
    from pipeline.asr_adapter import ASRAdapter
    from pipeline.ws_pipeline import Callbacks, WebSocketPipeline

    # 构建服务
    svc: PipelineServices = build_pipeline_services(settings)
    asr = ASRAdapter()

    # TTS: 优先 CosyVoice（自然情感语音），不可用则回退 Windows 本地
    try:
        from pipeline.cosyvoice_tts import CosyVoiceTTS
        tts = CosyVoiceTTS(voice="longxiaochun", rate=1.0)
        logger.info("TTS: DashScope CosyVoice")
    except Exception as exc:
        logger.warning("CosyVoice init failed (%s), falling back to Windows local TTS", exc)
        from pipeline.windows_tts import WindowsLocalTTS
        tts = WindowsLocalTTS()
        logger.info("TTS: Windows SAPI5 (offline)")

    # 回调适配器：WebSocket 推送
    class WSCallbacks(Callbacks):
        async def on_status(self, phase: str) -> None:
            await ws.send_json({"type": "status", "phase": phase})

        async def on_user_text(self, text: str) -> None:
            await ws.send_json({"type": "user_text", "content": text})

        async def on_text_delta(self, content: str) -> None:
            await ws.send_json({"type": "text.delta", "content": content})

        async def on_audio_delta(self, chunk: bytes) -> None:
            await ws.send_bytes(chunk)

        async def on_emotion_result(self, data: dict) -> None:
            await ws.send_json({"type": "emotion.result", **data})

        async def on_done(self, sid: str) -> None:
            await ws.send_json({"type": "done", "session_id": sid})

        async def on_error(self, message: str) -> None:
            await ws.send_json({"type": "error", "message": message})

    pipeline = WebSocketPipeline(
        asr=asr,
        tts=tts,
        services=svc,
    )
    await pipeline.run(
        audio_bytes=audio_bytes,
        callbacks=WSCallbacks(),
        session_id=session_id,
        user_id=user_id,
    )
