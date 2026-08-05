"""
WebSocket 实时语音对话的音频缓冲区。

积累前端推送的 Opus/WebM 音频 chunk，支持：
- ``feed(chunk)`` 追加数据
- ``end()`` 返回完整音频并重置缓冲区
- ``cancel()`` 丢弃缓冲区内容
"""

from __future__ import annotations

import threading


class AudioBuffer:
    """线程安全的音频 chunk 缓冲区。"""

    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self._lock = threading.Lock()

    def feed(self, chunk: bytes) -> None:
        """追加一个音频 chunk。"""
        with self._lock:
            self._chunks.append(chunk)

    def end(self) -> bytes:
        """返回全部已缓冲的音频数据，并清空缓冲区。"""
        with self._lock:
            data = b"".join(self._chunks)
            self._chunks.clear()
            return data

    def cancel(self) -> None:
        """丢弃缓冲区中的所有数据。"""
        with self._lock:
            self._chunks.clear()
