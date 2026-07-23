"""情感分析 · EmotionService 实现。

- 有 video：视觉 + 音频 + 文本三模态融合
- 有 audio_path：语音情绪 + 文本辅助 → 融合
- 无 audio_path：文本情绪分析（可插拔引擎：关键词 / ONNX）
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from schemas.contracts import EmotionAnalyzeRequest, EmotionTags
from multimodal.audio_emotion import AudioEmotionResult, get_audio_emotion_recognizer
from multimodal.emotion_fusion import EmotionSignal, FusedEmotionResult, build_signal, fuse_emotions

from modules.emotion.base import TextEmotionEngine, TextEmotionResult
from modules.emotion.keyword_engine import _INTENSITY_KEYWORDS, KeywordEmotionEngine

_CONFLICT_PENALTY = 0.7
_CONFLICT_MAX_CONFIDENCE = 0.6
_MIXED_SIGNAL_RISK_BONUS = 0.1

# ── 慢性化/绝望感关键词（用于风险加成）──────────────────────────
_CHRONICITY_KEYWORDS = [
    "好不了", "没救了", "没希望", "绝望", "撑不下去", "活不下去",
    "不想活", "死了算了", "结束吧", "解脱",
    "一两年", "好几年", "一年多", "两年多", "几年了", "好几个月",
    "越来越", "更严重", "加重", "恶化", "变严重",
    "什么都不想干", "什么都不想做", "动不了", "没劲",
    "好难受", "好痛苦", "受不了", "熬不下去",
]
_CHRONICITY_RISK_BONUS = 0.22  # 足以将中度慢性案例推入 comfort 区间


class EmotionService:
    """情绪分析服务。实现 EmotionPort 协议。

    文本情绪分类由可插拔的 TextEmotionEngine 完成，
    默认使用 KeywordEmotionEngine（零依赖）。
    """

    def __init__(self, text_engine: Optional[TextEmotionEngine] = None,
                 risk_config: Optional[dict] = None) -> None:
        self._text_engine: TextEmotionEngine = text_engine or KeywordEmotionEngine()
        self._last_text_result: Optional[TextEmotionResult] = None
        # risk formula config, defaults to hardcoded values if not provided
        cfg = risk_config or {}
        self._risk_emotion_base: dict = cfg.get("emotion_base", {})
        self._risk_intensity_weight: float = float(cfg.get("intensity_weight", 0.25))
        self._risk_safety_weight: float = float(cfg.get("safety_weight", 0.20))

    def analyze(self, req: EmotionAnalyzeRequest) -> EmotionTags:
        text_result = self._text_engine.predict(req.text)
        self._last_text_result = text_result

        text_primary = text_result.primary_emotion
        kw_count = text_result.hit_count

        audio_result: Optional[AudioEmotionResult] = None
        modality_notes: Dict[str, Any] = {
            "source": "text_keywords",
            "text_engine": self._text_engine.model_name,
        }
        audio_duration_s: Optional[float] = None

        # 尝试语音情绪：优先用预提取数据，否则调 AudioEmotionRecognizer
        if req.pre_extracted_audio_emotion:
            pre = req.pre_extracted_audio_emotion
            audio_result = AudioEmotionResult(
                primary_emotion=pre.get("primary_emotion", "neutral"),
                confidence=float(pre.get("confidence", 0.0)),
                all_emotions=pre.get("all_emotions", {}),
                model_name=pre.get("model_name", "pre_extracted"),
                backend=pre.get("backend", "pre_extracted"),
                transcript=req.text,
            )
        elif req.audio_path and os.path.exists(req.audio_path):
            try:
                recognizer = get_audio_emotion_recognizer()
                audio_result = recognizer.recognize(
                    audio_path=req.audio_path,
                    transcript=req.text,
                )
            except Exception:
                audio_result = None

        # 计算强度（使用引擎结果优化 ONNX 场景）
        if text_result.hit_count > 0 or not text_result.all_emotions:
            intensity = self._compute_intensity(req.text, kw_count, audio_duration_s)
        else:
            intensity = self._compute_intensity_from_model(text_result, req.text)

        # 构建各路信号
        # 文本信号（必选）：使用引擎的真实 confidence 和 all_emotions
        text_signal = build_signal(
            source="text",
            payload={
                "primary_emotion": text_primary,
                "confidence": text_result.confidence,
                "all_emotions": text_result.all_emotions,
            },
            model_name=self._text_engine.model_name,
        )

        # 音频信号
        audio_signal: Optional[EmotionSignal] = None
        if audio_result and audio_result.primary_emotion != "neutral":
            audio_signal = build_signal(
                source="audio",
                payload={
                    "primary_emotion": audio_result.primary_emotion,
                    "confidence": audio_result.confidence,
                    "all_emotions": audio_result.all_emotions,
                    "model_name": audio_result.model_name,
                },
                model_name=audio_result.model_name,
            )

        # 视觉信号（v1.2 新增）
        visual_signal: Optional[EmotionSignal] = None
        if req.pre_extracted_visual_emotion:
            vis = req.pre_extracted_visual_emotion
            visual_signal = build_signal(
                source="visual",
                payload={
                    "primary_emotion": vis.get("primary_emotion", "neutral"),
                    "confidence": float(vis.get("confidence", 0.0)),
                    "all_emotions": vis.get("all_emotions", {}),
                    "model_name": vis.get("model_name", "HSEmotion"),
                },
                model_name=vis.get("model_name", "HSEmotion"),
            )

        # 融合：按可用信号数量决定来源标记
        fusion_signals = [s for s in [audio_signal, text_signal, visual_signal] if s is not None]
        is_tri = visual_signal is not None and audio_signal is not None

        fused = fuse_emotions(*fusion_signals)

        if fused:
            # ---- 信号冲突仲裁 ----
            all_primaries = {s.primary_emotion for s in fusion_signals}
            max_raw_conf = max(s.confidence for s in fusion_signals)
            if len(all_primaries) == len(fusion_signals) and max_raw_conf <= _CONFLICT_MAX_CONFIDENCE:
                fused.confidence *= _CONFLICT_PENALTY
                modality_notes["conflict_arbitration"] = True

            primary = fused.primary_emotion
            modality_notes["fusion_summary"] = fused.summary
            if is_tri:
                modality_notes["source"] = "tri_modal"
            else:
                modality_notes["source"] = "audio_fused"
            if audio_result:
                modality_notes["audio_model"] = audio_result.model_name
                modality_notes["audio_backend"] = audio_result.backend
                if audio_result.warnings:
                    modality_notes["audio_warnings"] = audio_result.warnings
            if visual_signal:
                modality_notes["visual_model"] = vis.get("model_name", "HSEmotion")

            modality_notes["text_emotion"] = text_primary
        else:
            primary = text_primary
            if req.audio_path:
                modality_notes["audio"] = "degraded_no_valid_signal"

        # 音频降级标记（audio_path 存在但不可用）
        if req.audio_path and audio_signal is None:
            modality_notes["audio"] = "degraded_no_valid_signal"

        # ---- risk 计算（含 intensity 加权 + 慢性化加成 + mixed_signals 预警加分） ----
        risk = self._compute_risk(primary, intensity, req.safety)
        chronicity = self._compute_chronicity_bonus(req.text)
        if chronicity > 0:
            risk = round(min(risk + chronicity, 1.0), 2)
            modality_notes["chronicity_bonus"] = chronicity
        if fused and fused.mixed_signals:
            risk = round(min(risk + _MIXED_SIGNAL_RISK_BONUS, 1.0), 2)
            modality_notes["mixed_signals_risk_bonus"] = True

        return EmotionTags(
            primary_emotion=primary,
            intensity=intensity,
            risk=risk,
            modality_notes=modality_notes,
            intent=text_result.intent,
        )

    def _compute_intensity(self, text: str, _kw_count: int, audio_duration_s: Optional[float]) -> float:
        kw_hits = sum(text.count(kw) for kw in _INTENSITY_KEYWORDS)
        # 持续时间词额外加分（暗示慢性/严重）
        for dur in ["两周", "几个月", "一直", "每天", "总是", "长期", "很久"]:
            if dur in text:
                kw_hits += 1
        kw_factor = min(kw_hits / 4.0, 1.0)

        if audio_duration_s and audio_duration_s > 0:
            cps = len(text) / audio_duration_s
            speed_factor = min(cps / 7.0, 1.0)
        else:
            speed_factor = 0.5

        intensity = speed_factor * 0.3 + kw_factor * 0.7
        return round(min(intensity, 1.0), 2)

    def _compute_intensity_from_model(self, text_result: TextEmotionResult, text: str) -> float:
        """ONNX 模式下：从模型概率计算强度，辅以关键词强度词。"""
        negative_probs = [
            text_result.all_emotions.get(e, 0.0)
            for e in ("anxiety", "sadness", "anger", "fear", "stress", "confusion")
        ]
        model_factor = min(max(negative_probs), 1.0) if negative_probs else 0.5

        # 辅以关键词强度词
        kw_hits = sum(text.count(kw) for kw in _INTENSITY_KEYWORDS)
        for dur in ["两周", "几个月", "一直", "每天", "总是", "长期", "很久"]:
            if dur in text:
                kw_hits += 1
        kw_factor = min(kw_hits / 4.0, 1.0)

        # 模型权重 0.6，关键词权重 0.4
        intensity = model_factor * 0.6 + kw_factor * 0.4
        return round(min(intensity, 1.0), 2)

    def _compute_chronicity_bonus(self, text: str) -> float:
        """检测慢性化/绝望感信号，返回风险加成（0.0 ~ _CHRONICITY_RISK_BONUS）。

        设计意图：关键词情绪引擎对"好不了""一两年""越来越严重"等慢性
        绝望表达的强度感知不足，导致长期痛苦用户被卡在 knowledge 路由里
        反复听科普而不是获得安抚。此加成直接作用于 risk 计算，确保慢性
        中重度案例能突破 comfort 阈值（0.5）。
        """
        hits = 0
        lowered = text.lower()
        for kw in _CHRONICITY_KEYWORDS:
            if kw in lowered:
                hits += 1
        if hits == 0:
            return 0.0
        # 1-2 个命中给一半加成，3+ 个命中给满加成
        factor = min(hits / 3.0, 1.0)
        return round(_CHRONICITY_RISK_BONUS * factor, 3)

    def _compute_risk(self, primary_emotion: str, intensity: float, safety: dict) -> float:
        base = self._risk_emotion_base.get(primary_emotion, 0.0)
        intensity_boost = intensity * self._risk_intensity_weight
        safety_level = int(safety.get("level", 0))
        safety_bonus = self._risk_safety_weight * safety_level
        return round(min(base + intensity_boost + safety_bonus, 1.0), 2)
