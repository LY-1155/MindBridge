"""AudioBuffer 行为测试：音频 chunk 积累、终止、取消。"""
from __future__ import annotations

import pytest

from pipeline.audio_buffer import AudioBuffer


class TestAudioBufferFeedEnd:
    """feed + end 主流程。"""

    def test_end_returns_empty_when_no_feed(self):
        buf = AudioBuffer()
        assert buf.end() == b""

    def test_single_chunk_returns_same_bytes(self):
        buf = AudioBuffer()
        buf.feed(b"\x00\x01\x02")
        assert buf.end() == b"\x00\x01\x02"

    def test_multiple_chunks_concatenated_in_order(self):
        buf = AudioBuffer()
        buf.feed(b"hello ")
        buf.feed(b"world")
        assert buf.end() == b"hello world"

    def test_end_resets_buffer(self):
        buf = AudioBuffer()
        buf.feed(b"first")
        assert buf.end() == b"first"
        # 新一轮，buffer 已清空
        assert buf.end() == b""

    def test_feed_after_end_starts_new_cycle(self):
        buf = AudioBuffer()
        buf.feed(b"round1")
        buf.end()
        buf.feed(b"round2")
        assert buf.end() == b"round2"


class TestAudioBufferCancel:
    """cancel 清空缓冲区。"""

    def test_cancel_clears_buffered_data(self):
        buf = AudioBuffer()
        buf.feed(b"sensitive content")
        buf.cancel()
        assert buf.end() == b""

    def test_cancel_then_feed_new_works(self):
        buf = AudioBuffer()
        buf.feed(b"cancelled")
        buf.cancel()
        buf.feed(b"new recording")
        assert buf.end() == b"new recording"

    def test_cancel_without_feed_is_noop(self):
        buf = AudioBuffer()
        buf.cancel()
        assert buf.end() == b""


class TestAudioBufferConcurrency:
    """并发安全。"""

    def test_feed_is_thread_safe(self):
        import threading

        buf = AudioBuffer()
        chunks = [b"a", b"b", b"c", b"d"]

        def feed_all():
            for c in chunks:
                buf.feed(c)

        t1 = threading.Thread(target=feed_all)
        t2 = threading.Thread(target=feed_all)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        result = buf.end()
        # 两线程各写 4 个 chunk = 8 个 chunk，每 chunk 1 byte = 8 bytes
        assert len(result) == 8
        assert result.count(b"a"[0]) == 2
        assert result.count(b"b"[0]) == 2
        assert result.count(b"c"[0]) == 2
        assert result.count(b"d"[0]) == 2
