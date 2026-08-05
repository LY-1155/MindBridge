"""
Windows SAPI5 本地 TTS 适配器。

使用 Windows 内置语音引擎（pyttsx3 → SAPI5），无需联网，不依赖 Edge TTS。
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class WindowsLocalTTS:
    """Windows SAPI5 离线 TTS，SentenceTTS 兼容接口。

    Parameters
    ----------
    rate : int
        语速，默认 180（words per minute）。
    voice_id : int
        SAPI5 语音索引，0=中文女声（默认）。
    """

    def __init__(self, rate: int = 180, voice_id: int = 0) -> None:
        self._rate = rate
        self._voice_id = voice_id
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            import pyttsx3

            self._engine = pyttsx3.init()
            voices = self._engine.getProperty("voices")
            # 优先选中文语音
            if voices:
                zh_voices = [v for v in voices if "chinese" in v.name.lower() or "中文" in v.name]
                target = zh_voices[0] if zh_voices else voices[min(self._voice_id, len(voices) - 1)]
                self._engine.setProperty("voice", target.id)
                logger.info("Windows TTS voice: %s", target.name)
            self._engine.setProperty("rate", self._rate)
        return self._engine

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """合成文本为 WAV bytes。"""
        if not text or not text.strip():
            return

        engine = self._get_engine()
        loop = asyncio.get_event_loop()

        # pyttsx3 是同步库，跑在 executor 中
        def _do():
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            try:
                engine.save_to_file(text, tmp.name)
                engine.runAndWait()
                with open(tmp.name, "rb") as f:
                    data = f.read()
                return data
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass

        data = await loop.run_in_executor(None, _do)
        if data:
            yield data
