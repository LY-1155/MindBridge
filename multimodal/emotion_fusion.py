"""
Utilities for fusing audio and visual emotion signals into a single summary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EmotionSignal:
    source: str
    primary_emotion: str
    confidence: float
    all_emotions: Dict[str, float] = field(default_factory=dict)
    model_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedEmotionResult:
    primary_emotion: str
    confidence: float
    mixed_signals: bool
    sources: List[str]
    signal_count: int
    summary: str
    all_emotions: Dict[str, float] = field(default_factory=dict)
    dominant_source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_signal(
    source: str,
    payload: Optional[Dict[str, Any]],
    *,
    model_name: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[EmotionSignal]:
    if not payload:
        return None
    primary = payload.get("primary_emotion")
    if not primary:
        return None
    return EmotionSignal(
        source=source,
        primary_emotion=primary,
        confidence=float(payload.get("confidence") or 0.0),
        all_emotions=dict(payload.get("all_emotions") or {}),
        model_name=model_name or str(payload.get("model_name") or ""),
        metadata=metadata or {},
    )


def fuse_emotions(*signals: Optional[EmotionSignal]) -> Optional[FusedEmotionResult]:
    valid = [signal for signal in signals if signal is not None]
    if not valid:
        return None

    totals: Dict[str, float] = {}
    weights = 0.0
    dominant = max(valid, key=lambda signal: signal.confidence)

    for signal in valid:
        weight = max(signal.confidence, 0.05)
        weights += weight
        if signal.all_emotions:
            for emotion, score in signal.all_emotions.items():
                totals[emotion] = totals.get(emotion, 0.0) + score * weight
        else:
            totals[signal.primary_emotion] = totals.get(signal.primary_emotion, 0.0) + weight

    if weights > 0:
        totals = {emotion: score / weights for emotion, score in totals.items()}

    primary, confidence = max(totals.items(), key=lambda item: item[1])
    distinct_primary = {signal.primary_emotion for signal in valid}
    mixed = len(distinct_primary) > 1
    sources = [signal.source for signal in valid]

    sources_text = "、".join(sources)
    if mixed:
        summary = (
            f"信号 {sources_text} 不一致，{dominant.source} 主导，综合倾向 {primary}。"
        )
    else:
        summary = f"信号 {sources_text} 一致，综合倾向 {primary}。"

    return FusedEmotionResult(
        primary_emotion=primary,
        confidence=confidence,
        mixed_signals=mixed,
        sources=sources,
        signal_count=len(valid),
        summary=summary,
        all_emotions=totals,
        dominant_source=dominant.source,
    )
