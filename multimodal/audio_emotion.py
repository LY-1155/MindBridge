"""
Audio emotion recognition helpers.

The project prefers a specialist speech-emotion model when available
(`SenseVoice` first, then other future backends), but keeps a lightweight
fallback so the API still returns structured emotion metadata in local/dev
environments without extra model dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging
import os
import re

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class AudioEmotionResult:
    primary_emotion: str
    confidence: float
    all_emotions: Dict[str, float]
    model_name: str
    backend: str
    transcript: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class AudioEmotionRecognizer:
    """Recognizer with optional specialist backend and heuristic fallback."""

    SUPPORTED_BACKENDS = ("sensevoice", "emotion2vec", "heuristic")
    EMOTIONS = ("neutral", "anxiety", "sadness", "anger", "fear", "stress", "happiness", "confusion")

    def __init__(self, backend: Optional[str] = None):
        self.backend = (backend or settings.AUDIO_EMOTION_BACKEND).lower()
        if self.backend not in self.SUPPORTED_BACKENDS:
            logger.warning("Unknown audio emotion backend '%s', falling back to heuristic", self.backend)
            self.backend = "heuristic"

    def recognize(
        self,
        *,
        audio_path: Optional[str] = None,
        transcript: Optional[str] = None,
    ) -> AudioEmotionResult:
        if self.backend == "sensevoice":
            result = self._recognize_with_sensevoice(audio_path=audio_path, transcript=transcript)
            if result:
                return result
        elif self.backend == "emotion2vec":
            result = self._recognize_with_emotion2vec(audio_path=audio_path, transcript=transcript)
            if result:
                return result

        return self._recognize_with_heuristic(transcript=transcript)

    def _recognize_with_sensevoice(
        self,
        *,
        audio_path: Optional[str],
        transcript: Optional[str],
    ) -> Optional[AudioEmotionResult]:
        try:
            from funasr import AutoModel  # type: ignore
        except Exception as exc:
            logger.info("SenseVoice backend unavailable: %s", exc)
            return None

        if not audio_path or not os.path.exists(audio_path):
            return None

        try:
            model = AutoModel(
                model="iic/SenseVoiceSmall",
                device=settings.SENSEVOICE_DEVICE,
            )
            result = model.generate(input=audio_path)
            text = str(result[0] if isinstance(result, list) and result else result)
            emotion = self._extract_tagged_emotion(text) or self._heuristic_primary_emotion(transcript or text)
            confidence = 0.72 if emotion != "neutral" else 0.6
            all_emotions = self._distribution_from_primary(emotion, confidence)
            return AudioEmotionResult(
                primary_emotion=emotion,
                confidence=confidence,
                all_emotions=all_emotions,
                model_name="SenseVoiceSmall",
                backend="sensevoice",
                transcript=transcript,
            )
        except Exception as exc:
            logger.warning("SenseVoice inference failed, falling back to heuristic: %s", exc)
            return None

    def _recognize_with_emotion2vec(
        self,
        *,
        audio_path: Optional[str],
        transcript: Optional[str],
    ) -> Optional[AudioEmotionResult]:
        # Placeholder for future specialized integration.
        # Returning None keeps the runtime dependency optional.
        return None

    def _recognize_with_heuristic(self, *, transcript: Optional[str]) -> AudioEmotionResult:
        emotion = self._heuristic_primary_emotion(transcript or "")
        confidence = 0.42 if transcript else 0.25
        warnings = []
        if not transcript:
            warnings.append("No transcript available; audio emotion used heuristic fallback.")
        else:
            warnings.append("Specialist audio emotion backend unavailable; used transcript-aware heuristic fallback.")
        return AudioEmotionResult(
            primary_emotion=emotion,
            confidence=confidence,
            all_emotions=self._distribution_from_primary(emotion, confidence),
            model_name="heuristic-transcript-fallback",
            backend="heuristic",
            transcript=transcript,
            warnings=warnings,
        )

    def _heuristic_primary_emotion(self, transcript: str) -> str:
        text = transcript.lower()
        keyword_groups = {
            "anxiety": [r"焦虑", r"紧张", r"担心", r"害怕", r"\banxious\b", r"\bnervous\b", r"\bworried\b"],
            "sadness": [r"难过", r"伤心", r"低落", r"委屈", r"\bsad\b", r"\bdown\b", r"\bdepressed\b"],
            "anger": [r"生气", r"愤怒", r"烦死", r"气死", r"\bangry\b", r"\bmad\b", r"\bfurious\b"],
            "fear": [r"恐惧", r"害怕", r"不敢", r"\bafraid\b", r"\bscared\b", r"\bfear\b"],
            "stress": [r"压力", r"崩溃", r"扛不住", r"\bstress\b", r"\boverwhelmed\b"],
            "happiness": [r"开心", r"高兴", r"轻松", r"\bhappy\b", r"\brelieved\b", r"\bglad\b"],
            "confusion": [r"迷茫", r"困惑", r"不知道", r"\bconfused\b", r"\blost\b"],
        }

        scores = {emotion: 0 for emotion in self.EMOTIONS}
        for emotion, patterns in keyword_groups.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    scores[emotion] += 1

        if "!" in transcript:
            scores["anger"] += 0.5
            scores["happiness"] += 0.25
        if "..." in transcript or "？" in transcript or "?" in transcript:
            scores["anxiety"] += 0.25
            scores["confusion"] += 0.25

        primary = max(scores.items(), key=lambda item: item[1])[0]
        return primary if scores[primary] > 0 else "neutral"

    def _distribution_from_primary(self, primary: str, confidence: float) -> Dict[str, float]:
        base = {emotion: 0.0 for emotion in self.EMOTIONS}
        residual = max(0.0, 1.0 - confidence)
        spread = residual / max(1, len(self.EMOTIONS) - 1)
        for emotion in base:
            base[emotion] = confidence if emotion == primary else spread
        return base

    def _extract_tagged_emotion(self, text: str) -> Optional[str]:
        lowered = text.lower()
        mapping = {
            "<|angry|>": "anger",
            "<|sad|>": "sadness",
            "<|fearful|>": "fear",
            "<|happy|>": "happiness",
            "<|neutral|>": "neutral",
            "<|surprised|>": "confusion",
        }
        for token, emotion in mapping.items():
            if token in lowered:
                return emotion
        return None


_recognizers: Dict[str, AudioEmotionRecognizer] = {}


def get_audio_emotion_recognizer(backend: Optional[str] = None) -> AudioEmotionRecognizer:
    selected = (backend or settings.AUDIO_EMOTION_BACKEND).lower()
    recognizer = _recognizers.get(selected)
    if recognizer is None:
        recognizer = AudioEmotionRecognizer(selected)
        _recognizers[selected] = recognizer
    return recognizer
