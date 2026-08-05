"""
DashScope CosyVoice TTS 适配器。

使用阿里云 DashScope 的 CosyVoice 模型，生成自然、有情感的语音，
效果接近豆包。与项目 LLM 共用同一个 API Key。

API 文档：https://help.aliyun.com/document_detail/2815022.html

音色试听：https://help.aliyun.com/document_detail/2815040.html
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from config.settings import settings

logger = logging.getLogger(__name__)

# 可用音色：longxiaochun（女声·温柔）、longxiaoxia（女声·活泼）、longxiaoming（男声）
# 更多音色见：https://help.aliyun.com/document_detail/2815040.html
_COSYVOICE_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer"


class CosyVoiceTTS:
    """DashScope CosyVoice TTS，SentenceTTS 兼容接口。

    Parameters
    ----------
    voice : str
        音色名。默认 "longxiaochun"（温柔女声，适合心理咨询场景）。
    rate : float
        语速倍率，0.5–2.0，默认 1.0。
    pitch : float
        音高倍率，0.5–2.0，默认 1.0。
    volume : int
        音量，1–100，默认 50。
    format : str
        输出格式 mp3 / wav / pcm，默认 "mp3"。
    """

    def __init__(
        self,
        voice: str = "longxiaochun",
        rate: float = 1.0,
        pitch: float = 1.0,
        volume: int = 50,
        fmt: str = "mp3",
    ) -> None:
        self._voice = voice
        self._rate = rate
        self._pitch = pitch
        self._volume = volume
        self._fmt = fmt
        self._api_key = settings.OPENAI_API_KEY
        if not self._api_key or self._api_key == "sk-placeholder":
            logger.warning("OPENAI_API_KEY not configured, CosyVoice TTS will not work")

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """合成文本为音频，yield MP3/WAV bytes。"""
        if not text or not text.strip():
            return

        import json
        import httpx

        payload = {
            "model": "cosyvoice-v1",
            "input": {
                "text": text,
                "voice": self._voice,
                "format": self._fmt,
                "volume": self._volume,
                "rate": self._rate,
                "pitch": self._pitch,
            },
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    _COSYVOICE_ENDPOINT,
                    headers=headers,
                    json=payload,
                )
                if response.status_code != 200:
                    logger.warning("CosyVoice TTS error %d: %s",
                                   response.status_code, response.text[:300])
                    return

                result = response.json()

                output = result.get("output", {})
                # speech-synthesizer 端点: output.audio_url 直接是音频下载链接
                audio_url = output.get("audio_url", "")
                # 也兼容嵌套格式: output.audio.url / output.audio.data
                audio = output.get("audio", {})
                if not audio_url:
                    audio_url = audio.get("url", "")
                audio_data = audio.get("data", "")

                if audio_data:
                    import base64
                    yield base64.b64decode(audio_data)
                elif audio_url:
                    audio_resp = await client.get(audio_url)
                    audio_resp.raise_for_status()
                    yield audio_resp.content
                else:
                    logger.warning("CosyVoice no audio in response: %s",
                                   json.dumps(result, ensure_ascii=False)[:500])
        except httpx.HTTPError as exc:
            logger.warning("CosyVoice TTS request failed: %s", exc)
        except Exception as exc:
            logger.warning("CosyVoice TTS error: %s", exc)
