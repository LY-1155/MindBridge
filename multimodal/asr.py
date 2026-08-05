"""
语音识别模块（ASR）
=================

听写层可配置为：

- **sensevoice**（默认，偏中文口语）：FunASR `iic/SenseVoiceSmall`，需安装 `funasr`、`modelscope`。
  未安装或推理失败时自动回退到 Faster-Whisper。
- **whisper**：Faster-Whisper（CTranslate2）。

使用方式：
    result = get_speech_recognizer().transcribe("audio.wav", language="zh")
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TAG_STRIP_RE = re.compile(r"<\|[^>|]+\|>")


@dataclass
class TranscriptionResult:
    text: str
    language: str
    segments: List[Dict[str, Any]]
    duration: float
    asr_backend: str = "whisper"


def _audio_duration_seconds(path: str) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            proc = subprocess.run(
                [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", path],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode == 0 and proc.stdout:
                fmt = json.loads(proc.stdout).get("format") or {}
                d = float(fmt.get("duration") or 0)
                if d > 0:
                    return d
        except Exception:
            pass
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0


def _clean_sensevoice_text(raw: str) -> str:
    if not raw:
        return ""
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        return str(rich_transcription_postprocess(raw)).strip()
    except Exception:
        return _TAG_STRIP_RE.sub("", raw).strip()


def _language_from_sensevoice_raw(raw: str, hint: Optional[str]) -> str:
    if hint:
        return hint
    low = raw.lower()
    if "<|en|>" in low:
        return "en"
    if "<|yue|>" in low or "<|zh|>" in low or "<|ja|>" in low:
        return "zh"
    return "zh"


def _parse_sensevoice_generate(out: Any) -> str:
    if isinstance(out, list) and out:
        first = out[0]
        if isinstance(first, dict) and "text" in first:
            return str(first["text"])
        return str(first)
    if isinstance(out, dict) and "text" in out:
        return str(out["text"])
    return str(out)


class SpeechRecognizer:
    MODEL_SIZES = ["tiny", "base", "small", "medium", "large"]
    DEFAULT_MODEL = "base"

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        model_path: Optional[str] = None,
        backend: Optional[str] = None,
        sensevoice_model_id: Optional[str] = None,
    ):
        try:
            from config.settings import settings
        except ImportError:
            settings = None  # type: ignore

        self.backend = (backend or (settings.ASR_BACKEND if settings else "whisper")).lower()
        if self.backend not in ("whisper", "sensevoice"):
            logger.warning("Unknown ASR_BACKEND '%s', using whisper", self.backend)
            self.backend = "whisper"

        self.sensevoice_model_id = sensevoice_model_id or (
            settings.SENSEVOICE_ASR_MODEL if settings else "iic/SenseVoiceSmall"
        )
        self.model_size = model_size or (settings.WHISPER_MODEL_SIZE if settings else self.DEFAULT_MODEL)
        self.device = device or (settings.WHISPER_DEVICE if settings else "auto")
        mp = model_path if model_path is not None else (settings.WHISPER_MODEL_PATH if settings else None)
        self.model_path = mp or None

        self._model = None
        self._sensevoice_model = None
        self._sensevoice_device = (
            settings.SENSEVOICE_DEVICE if settings else "cuda:0"
        )

        logger.info(
            "ASR init: backend=%s sensevoice_model=%s whisper=%s",
            self.backend,
            self.sensevoice_model_id,
            self.model_size,
        )

    @property
    def model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            logger.info("Loading Whisper model: %s", self.model_size)
            model_source = self.model_path if self.model_path else self.model_size

            # 本地路径：标准化并验证目录存在，否则回退到模型名自动下载
            if isinstance(model_source, str) and not os.path.isabs(model_source):
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                model_source = os.path.normpath(os.path.join(project_root, model_source))

            if isinstance(model_source, str) and os.path.isdir(model_source):
                logger.info("Whisper local path: %s", model_source)
            elif isinstance(model_source, str) and model_source != self.model_size:
                logger.warning(
                    "Whisper model path %s not found, falling back to %s",
                    model_source, self.model_size,
                )
                model_source = self.model_size

            self._model = WhisperModel(
                model_source,
                device=self.device,
                compute_type="int8",
            )
            logger.info("Whisper model ready")
        return self._model

    @property
    def sensevoice_model(self):
        if self._sensevoice_model is None:
            from funasr import AutoModel

            logger.info("Loading SenseVoice ASR: %s", self.sensevoice_model_id)
            self._sensevoice_model = AutoModel(
                model=self.sensevoice_model_id,
                trust_remote_code=True,
                device=self._sensevoice_device,
            )
        return self._sensevoice_model

    def _transcribe_whisper(
        self,
        audio_path: str,
        language: Optional[str],
        task: str,
    ) -> TranscriptionResult:
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            task=task,
            beam_size=5,
            vad_filter=True,
        )
        text_parts: List[str] = []
        segment_list: List[Dict[str, Any]] = []
        for segment in segments:
            text_parts.append(segment.text)
            segment_list.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "confidence": getattr(segment, "avg_logprob", 0),
                }
            )
        return TranscriptionResult(
            text=" ".join(text_parts).strip(),
            language=info.language,
            segments=segment_list,
            duration=float(info.duration),
            asr_backend="whisper",
        )

    def _transcribe_sensevoice(
        self,
        audio_path: str,
        language: Optional[str],
        task: str,
    ) -> TranscriptionResult:
        if task != "transcribe":
            logger.info("SenseVoice ASR ignores non-transcribe task=%s", task)
        try:
            from funasr import AutoModel  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(f"funasr not installed: {exc}") from exc

        raw_text = _parse_sensevoice_generate(
            self.sensevoice_model.generate(input=audio_path, use_itn=True)
        )
        clean = _clean_sensevoice_text(raw_text)
        lang = _language_from_sensevoice_raw(raw_text, language)
        duration = _audio_duration_seconds(audio_path)
        return TranscriptionResult(
            text=clean,
            language=lang,
            segments=[
                {
                    "start": 0.0,
                    "end": duration,
                    "text": clean,
                    "confidence": 0.0,
                }
            ]
            if clean
            else [],
            duration=duration,
            asr_backend="sensevoice",
        )

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> TranscriptionResult:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        if self.backend == "sensevoice":
            try:
                logger.info("Transcribe (SenseVoice): %s", audio_path)
                result = self._transcribe_sensevoice(audio_path, language, task)
                if result.text.strip():
                    return result
                logger.warning(
                    "SenseVoice returned empty text (duration=%.1fs), falling back to Whisper",
                    result.duration,
                )
            except Exception as exc:
                logger.warning("SenseVoice ASR failed, falling back to Whisper: %s", exc)

        logger.info("Transcribe (Whisper): %s", audio_path)
        return self._transcribe_whisper(audio_path, language, task)

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        suffix: str = ".wav",
    ) -> TranscriptionResult:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        try:
            return self.transcribe(temp_path, language=language)
        finally:
            os.unlink(temp_path)

    def detect_language(self, audio_path: str) -> str:
        return self.transcribe(audio_path, language=None).language


_recognizer_instance: Optional[SpeechRecognizer] = None
_recognizer_key: Optional[tuple] = None


def get_speech_recognizer(
    model_size: Optional[str] = None,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    backend: Optional[str] = None,
    sensevoice_model_id: Optional[str] = None,
) -> SpeechRecognizer:
    global _recognizer_instance, _recognizer_key

    from config.settings import settings

    key = (
        backend or settings.ASR_BACKEND,
        sensevoice_model_id or settings.SENSEVOICE_ASR_MODEL,
        model_size or settings.WHISPER_MODEL_SIZE,
        model_path if model_path is not None else (settings.WHISPER_MODEL_PATH or ""),
        device or settings.WHISPER_DEVICE,
    )
    if _recognizer_instance is None or _recognizer_key != key:
        _recognizer_instance = SpeechRecognizer(
            model_size=model_size,
            model_path=model_path,
            device=device,
            backend=backend or settings.ASR_BACKEND,
            sensevoice_model_id=sensevoice_model_id,
        )
        _recognizer_key = key

    return _recognizer_instance
