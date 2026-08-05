"""
WebSocket 管线的 ASR 适配器。

接收 raw audio bytes → ffmpeg 转 WAV → SenseVoice 一次推理产出文本 + 语调情绪 →
自动回退 Whisper（文本 only，无情绪）→ 返回 (text, emotion_dict)。
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class FFmpegNotFoundError(RuntimeError):
    """ffmpeg 未安装或不在 PATH 中，无法转换音频格式。"""


class ASRAdapter:
    """音频 bytes ↔ 文本 + 情绪 的适配层。

    封装了 WebM→WAV 转码、临时文件管理、SenseVoice 原始输出解析、
    以及 SenseVoice 不可用时间退到 Faster-Whisper 的退化路径。

    可通过 ``recognizer`` 和 ``converter`` 参数注入依赖，方便测试替换。
    """

    def __init__(self, converter=None):
        self._converter = converter or _convert_to_wav

    def transcribe(
        self,
        audio_bytes: bytes,
        recognizer: Any = None,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """将音频 bytes 转为 (文本, 可选情绪)。

        Parameters
        ----------
        audio_bytes : bytes
            原始音频数据（浏览器 MediaRecorder 产 Opus/WebM）。
        recognizer : SpeechRecognizer, optional
            注入的识别器，不传则取全局单例。

        Returns
        -------
        (text, emotion_dict_or_none)
        """
        recognizer = recognizer or _get_recognizer()

        # 1. 写原始音频 bytes
        tmp_raw = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            tmp_raw.write(audio_bytes)
            tmp_raw.close()
            tmp_wav.close()

            # 2. ffmpeg: WebM/Opus → 16kHz mono WAV (SenseVoice 需要)
            if not self._converter(tmp_raw.name, tmp_wav.name):
                logger.warning("ffmpeg conversion failed for %d bytes of audio", len(audio_bytes))
                return "", None

            # 3. 一次 SenseVoice 推理，同时拿原始输出（含情绪标签）和文本
            raw_output = _try_sensevoice_raw(tmp_wav.name, recognizer)
            emotion = _extract_emotion_from_raw(raw_output)

            # 4. 文本：走 SpeechRecognizer 管道（SenseVoice → 空则自动回退 Whisper）
            try:
                result = recognizer.transcribe(tmp_wav.name)
                text = result.text
                logger.info(
                    "ASR result: backend=%s text_len=%d emotion=%s",
                    result.asr_backend,
                    len(text),
                    emotion.get("primary_emotion") if emotion else None,
                )
            except Exception as exc:
                logger.warning("ASR transcribe failed: %s", exc)
                text = ""

            return text, emotion

        finally:
            _safe_unlink(tmp_raw.name)
            _safe_unlink(tmp_wav.name)


def _get_recognizer():
    """获取全局单例 SpeechRecognizer。"""
    from multimodal.asr import get_speech_recognizer

    return get_speech_recognizer()


def _convert_to_wav(src: str, dst: str) -> bool:
    """ffmpeg: 任意音频格式 → 16kHz mono WAV。

    Returns True on success.
    Raises FFmpegNotFoundError if ffmpeg is not installed.
    Returns False if conversion itself fails (corrupted audio, etc.).
    """
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                dst,
            ],
            capture_output=True,
            timeout=30,
            check=True,
        )
        return os.path.exists(dst) and os.path.getsize(dst) > 0
    except FileNotFoundError:
        raise FFmpegNotFoundError(
            "ffmpeg 未安装或不在系统 PATH 中。请运行 winget install ffmpeg 安装后重启终端。"
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffmpeg conversion failed: %s", exc)
        return False


def _try_sensevoice_raw(audio_path: str, recognizer: Any) -> str:
    """尝试获取 SenseVoice 原始输出文本（含 <|sad|> 等情绪标签）。"""
    try:
        model = recognizer.sensevoice_model
        out = model.generate(input=audio_path, use_itn=True)
        return str(out[0] if isinstance(out, list) and out else out)
    except Exception as exc:
        logger.info("SenseVoice unavailable, audio emotion skipped: %s", exc)
        return ""


def _extract_emotion_from_raw(raw_output: str) -> Optional[Dict[str, Any]]:
    """从 SenseVoice 原始输出提取情绪标签。"""
    if not raw_output:
        return None
    from multimodal.audio_emotion import AudioEmotionRecognizer

    recognizer = AudioEmotionRecognizer(backend="sensevoice")
    emotion_tag = recognizer._extract_tagged_emotion(raw_output)
    if emotion_tag:
        confidence = 0.72 if emotion_tag != "neutral" else 0.6
        return {
            "primary_emotion": emotion_tag,
            "confidence": confidence,
            "all_emotions": recognizer._distribution_from_primary(
                emotion_tag, confidence
            ),
            "model_name": "SenseVoiceSmall",
            "backend": "sensevoice",
        }
    return None


def _safe_unlink(path: str) -> None:
    """清理临时文件，不抛异常。"""
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass
