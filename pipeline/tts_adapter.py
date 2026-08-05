"""
Edge TTS 流式合成适配器。

实现 SentenceTTS 所需的 ``async synthesize(text) -> AsyncIterator[bytes]`` 接口，
封装 edge-tts 的流式 communicate()。
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class EdgeTTSAdapter:
    """Edge TTS 引擎适配器。

    Parameters
    ----------
    voice : str
        TTS 语音名，默认 "zh-CN-XiaoxiaoNeural"（微软 Xiaoxiao 女声）。
    rate : str
        语速偏移，如 "+10%" 或 "-5%"。
    pitch : str
        音高偏移，如 "+0Hz"。
    """

    def __init__(
        self,
        voice: str = "zh-CN-XiaoxiaoNeural",
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> None:
        self._voice = voice
        self._rate = rate
        self._pitch = pitch

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """将文本合成为 MP3 音频流。

        Yields
        ------
        bytes
            每个 chunk 是一段 MP3 帧数据，前端可连续拼接播放。
        """
        if not text or not text.strip():
            return

        try:
            import edge_tts

            communicate = edge_tts.Communicate(
                text=text,
                voice=self._voice,
                rate=self._rate,
                pitch=self._pitch,
            )
            async for chunk in communicate.stream():
                if chunk["type"] == "audio" and chunk["data"]:
                    yield chunk["data"]
        except Exception as exc:
            logger.warning("Edge TTS synthesis failed: %s", exc)
