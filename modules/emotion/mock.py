"""情感分析 · Mock"""

from __future__ import annotations

from schemas.contracts import EmotionAnalyzeRequest, EmotionTags


class MockEmotionService:
    def analyze(self, req: EmotionAnalyzeRequest) -> EmotionTags:
        return EmotionTags(
            primary_emotion="neutral",
            intensity=0.5,
            risk=0.1,
            modality_notes={"implementation": "mock"},
        )
