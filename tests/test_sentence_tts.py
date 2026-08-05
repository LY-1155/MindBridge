"""SentenceTTS 行为测试：token 积累、句边界检测、Edge TTS 调用。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, List

import pytest

from pipeline.sentence_tts import SentenceTTS


# ── test double ──────────────────────────────────────────────


class SpyTTSEngine:
    """记录每次 TTS 调用的 spy 引擎。"""

    def __init__(self):
        self.calls: List[str] = []
        self._call_counter = 0

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.calls.append(text)
        self._call_counter += 1
        # 每次调用输出 2 个 chunk
        yield f"chunk_{self._call_counter}_a".encode()
        yield f"chunk_{self._call_counter}_b".encode()


class NullTTSEngine:
    """不产音频的引擎（文本端末无边界时不触发 TTS）。"""

    def __init__(self):
        self.calls: List[str] = []

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.calls.append(text)
        # 不产出任何 chunk
        if False:
            yield b""


# ── tests ─────────────────────────────────────────────────────


class TestSentenceTTSBoundaryDetection:
    """句边界检测 + TTS 触发。"""

    @pytest.mark.asyncio
    async def test_triggers_tts_on_period(self):
        engine = SpyTTSEngine()
        ttss = SentenceTTS(engine)
        tokens = ["你好", "。"]
        chunks = []
        for t in tokens:
            async for c in ttss.feed_token(t):
                chunks.append(c)

        assert len(engine.calls) == 1
        assert engine.calls[0] == "你好。"
        assert len(chunks) == 2  # spy 每句 2 chunk

    @pytest.mark.asyncio
    async def test_triggers_tts_on_question_mark(self):
        engine = SpyTTSEngine()
        ttss = SentenceTTS(engine)
        tokens = ["没事吧", "？"]
        async for _ in ttss.feed_token(tokens[0]):
            pass
        chunks = [c async for c in ttss.feed_token(tokens[1])]

        assert len(engine.calls) == 1
        assert engine.calls[0] == "没事吧？"
        assert len(chunks) == 2

    @pytest.mark.asyncio
    async def test_triggers_tts_on_exclamation(self):
        engine = SpyTTSEngine()
        ttss = SentenceTTS(engine)
        for t in ["太好了", "！"]:
            async for _ in ttss.feed_token(t):
                pass

        assert engine.calls[0] == "太好了！"

    @pytest.mark.asyncio
    async def test_triggers_tts_on_newline(self):
        engine = SpyTTSEngine()
        ttss = SentenceTTS(engine)
        for t in ["嗯", "\n"]:
            async for _ in ttss.feed_token(t):
                pass

        assert engine.calls[0] == "嗯\n"

    @pytest.mark.asyncio
    async def test_multi_sentence_text(self):
        engine = SpyTTSEngine()
        ttss = SentenceTTS(engine)
        tokens = ["你好", "。", "我很好", "。", "谢谢", "。"]
        for t in tokens:
            async for _ in ttss.feed_token(t):
                pass

        assert len(engine.calls) == 3
        assert engine.calls == ["你好。", "我很好。", "谢谢。"]

    @pytest.mark.asyncio
    async def test_boundary_at_end_of_token(self):
        """token 本身以句末标点结尾（如 "感受。"），不是独立标点 token。"""
        engine = SpyTTSEngine()
        ttss = SentenceTTS(engine)
        tokens = ["我感到很", "难过的。"]
        for t in tokens:
            async for _ in ttss.feed_token(t):
                pass

        assert len(engine.calls) == 1
        assert engine.calls[0] == "我感到很难过的。"

    @pytest.mark.asyncio
    async def test_no_tts_triggered_without_boundary(self):
        """没有句末标点时 TTS 不触发。"""
        engine = NullTTSEngine()
        ttss = SentenceTTS(engine)
        tokens = ["你好", "世界"]
        for t in tokens:
            async for _ in ttss.feed_token(t):
                pass

        assert len(engine.calls) == 0  # 还没到边界


class TestSentenceTTSFlush:
    """尾部残余文本冲刷。"""

    @pytest.mark.asyncio
    async def test_flush_sends_remaining_text(self):
        engine = SpyTTSEngine()
        ttss = SentenceTTS(engine)
        # 喂一个没句末标点的 token
        async for _ in ttss.feed_token("没有标点结尾"):
            pass

        # flush 应触发最后一次 TTS
        chunks = [c async for c in ttss.flush()]
        assert len(engine.calls) == 1
        assert engine.calls[0] == "没有标点结尾"
        assert len(chunks) == 2

    @pytest.mark.asyncio
    async def test_flush_with_empty_buffer_noop(self):
        engine = SpyTTSEngine()
        ttss = SentenceTTS(engine)
        # 喂完整句子后 flush
        async for _ in ttss.feed_token("完整句子。"):
            pass
        async for _ in ttss.flush():
            pass

        # 边界已触发一次，flush 不重复
        assert len(engine.calls) == 1

    @pytest.mark.asyncio
    async def test_flush_after_multi_boundary_noop(self):
        engine = SpyTTSEngine()
        ttss = SentenceTTS(engine)
        for t in ["A", "。", "B", "。"]:
            async for _ in ttss.feed_token(t):
                pass
        async for _ in ttss.flush():
            pass

        assert len(engine.calls) == 2  # flush 不额外触发
