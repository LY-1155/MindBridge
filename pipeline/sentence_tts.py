"""
逐句 TTS 合成器。

接收 LLM token 流 → 按句末标点积累 → 完整的句子送 TTS 引擎合成 → 产出音频 chunk 流。
句末标点包括：。！？!?\n…~
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, List

logger = logging.getLogger(__name__)

_SENTENCE_BOUNDARY = frozenset({"。", "！", "？", "!", "?", "\n", "…", "~", "～"})


class SentenceTTS:
    """按语义边界积累 token 并触发 TTS 合成。

    Parameters
    ----------
    engine
        TTS 引擎，需实现 ``async synthesize(text: str) -> AsyncIterator[bytes]``。
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._buffer: List[str] = []

    async def feed_token(self, token: str) -> AsyncIterator[bytes]:
        """喂入一个 token，若触发句边界则产出音频 chunk。

        调用方应 ``async for chunk in tts.feed_token(token):`` 消费音频。
        """
        self._buffer.append(token)

        if self._is_boundary(token):
            sentence = "".join(self._buffer)
            self._buffer.clear()
            async for chunk in self._engine.synthesize(sentence):
                yield chunk

    async def flush(self) -> AsyncIterator[bytes]:
        """冲刷尾部残余文本（无句末标点结尾时调用）。"""
        if self._buffer:
            sentence = "".join(self._buffer)
            self._buffer.clear()
            async for chunk in self._engine.synthesize(sentence):
                yield chunk

    def _is_boundary(self, token: str) -> bool:
        """token 是独立的句末标点，或以句末标点结尾。"""
        if token in _SENTENCE_BOUNDARY:
            return True
        return len(token) >= 2 and token[-1] in _SENTENCE_BOUNDARY
