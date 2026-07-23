"""智能路由 · 规则引擎实现"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from schemas.contracts import RouteDecision, RouteRequest


BAND_MAP = {"general": "none", "knowledge": "low", "comfort": "mid", "crisis": "high"}


def _load_rules(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class RouterService:
    def __init__(self, rules_path: str) -> None:
        self._rules: Dict[str, Any] = _load_rules(rules_path)

    def route(self, req: RouteRequest) -> RouteDecision:
        emotion = req.emotion or {}
        safety = req.safety or {}
        risk = float(emotion.get("risk", 0.0))
        safety_level = int(safety.get("level", 0))
        primary_emotion = str(emotion.get("primary_emotion", "neutral"))
        intensity = float(emotion.get("intensity", 0.5))
        modality_notes = emotion.get("modality_notes", {}) or {}
        mixed_signals = bool(modality_notes.get("mixed_signals", False))
        intent = str(emotion.get("intent", "unknown") or "unknown")

        t = self._rules["thresholds"]
        crisis_t = float(t["crisis_risk"])
        comfort_t = float(t["comfort_risk"])
        knowledge_t = float(t.get("knowledge_risk", 0.15))

        # 第一步：risk → 段位
        risk_route = self._risk_to_route(risk, crisis_t, comfort_t, knowledge_t)

        # 第二步：意图覆盖 — 纯信息提问从 general 提升到知识科普
        intent_override = False
        route = risk_route
        if risk_route == "general" and intent == "information":
            route = "knowledge"
            intent_override = True

        # 第三步：安全等级修正
        escalated = False
        if self._should_escalate_safety(safety_level):
            escalated = (route != self._escalate_band(route))
            route = self._escalate_band(route)

        # 第四步：情绪偏向（不跨段，只影响 meta 和 confidence）
        bias_info = self._compute_emotion_bias(primary_emotion, intensity, route)

        # 第五步：置信度
        confidence = self._compute_confidence(
            risk=risk,
            route=route,
            crisis_t=crisis_t,
            comfort_t=comfort_t,
            knowledge_t=knowledge_t,
            risk_route=risk_route,
            escalated=escalated,
            bias_info=bias_info,
            mixed_signals=mixed_signals,
        )

        # reason 格式："{风险段}:{修正标签}"
        risk_label = BAND_MAP[route] if BAND_MAP[route] != "none" else "general"
        reason_parts = [f"{risk_label}_risk"]
        if intent_override:
            reason_parts.append("intent_override")
        if escalated:
            reason_parts.append("safety_escalate")
        if bias_info["bias_tag"]:
            reason_parts.append(bias_info["bias_tag"])

        # LLM 降级（预留）
        llm_fallback = False
        llm_agree = None
        fallback_cfg = self._rules.get("llm_fallback", {})
        if fallback_cfg.get("enabled"):
            if confidence < float(fallback_cfg.get("trigger_confidence_below", 0.7)):
                llm_fallback = True
            elif mixed_signals and fallback_cfg.get("trigger_on_mixed_signals"):
                llm_fallback = True

        return RouteDecision(
            route=route,
            reason=":".join(reason_parts),
            confidence=round(confidence, 4),
            meta={
                "risk_level": BAND_MAP[route],
                "safety_escalated": escalated,
                "intent_override": intent_override,
                "intent": intent,
                "emotion_bias": bias_info["label"],
                "emotion_bias_applied": bias_info["applied"],
                "boundary_distance": bias_info["boundary_dist"],
                "mixed_signals": mixed_signals,
                "llm_fallback_triggered": llm_fallback,
                "llm_agreement": llm_agree,
            },
        )

    # ==================================================================
    # 私有步骤
    # ==================================================================

    def _should_escalate_safety(self, safety_level: int) -> bool:
        esc = self._rules.get("safety_escalation", {})
        return bool(esc.get("level_1_escalate")) and safety_level >= 2

    def _compute_emotion_bias(self, primary_emotion: str, intensity: float, route: str) -> dict:
        """计算情绪偏向信息，返回 label, tag, applied, boundary_dist 等。"""
        bias_cfg = self._rules.get("emotion_bias", {}).get(primary_emotion, {})
        prefer = bias_cfg.get("prefer")

        # anger 高强度反转
        if primary_emotion == "anger":
            rev = self._rules.get("anger_high_intensity_reverse", {})
            if intensity > float(rev.get("threshold", 0.7)):
                prefer = rev.get("prefer", "comfort")

        if prefer is None:
            return {
                "label": None,
                "bias_tag": None,
                "applied": None,
                "boundary_dist": 0.0,
            }

        bias_applied = (prefer == route)
        bias_tag = f"{primary_emotion}_bias" if not bias_applied else f"{primary_emotion}_bias(matched)"

        # 如果是 anger 反转
        if primary_emotion == "anger" and intensity > 0.7 and prefer == "comfort":
            bias_label = f"anger_high_intensity→comfort"
            if prefer != route:
                bias_tag = "anger_high_intensity_bias_locked"
            else:
                bias_tag = "anger_high_intensity_bias(matched)"
        else:
            bias_label = f"{primary_emotion}→{prefer}" if prefer else None

        return {
            "label": bias_label,
            "bias_tag": bias_tag,
            "applied": bias_applied,
            "boundary_dist": 0.0,
        }

    def _compute_confidence(
        self,
        risk: float,
        route: str,
        crisis_t: float,
        comfort_t: float,
        knowledge_t: float,
        risk_route: str,
        escalated: bool,
        bias_info: dict,
        mixed_signals: bool,
    ) -> float:
        conf_cfg = self._rules.get("confidence", {})
        confidence = 1.0

        # 边界距离扣分（到最近边界的距离）
        candidates = []
        if route == "crisis" or risk < crisis_t:
            candidates.append(abs(risk - crisis_t))
        if route in ("crisis", "comfort") or risk < comfort_t:
            candidates.append(abs(risk - comfort_t))
        if route in ("crisis", "comfort", "knowledge") or risk < knowledge_t:
            candidates.append(abs(risk - knowledge_t))
        dist = min(candidates) if candidates else 1.0
        near_d = float(conf_cfg.get("boundary_near_distance", 0.05))
        close_d = float(conf_cfg.get("boundary_close_distance", 0.10))
        if dist < near_d:
            confidence -= float(conf_cfg.get("boundary_near_penalty", 0.15))
        elif dist < close_d:
            confidence -= float(conf_cfg.get("boundary_close_penalty", 0.08))

        # 安全升段扣分
        if escalated:
            confidence -= float(conf_cfg.get("safety_escalate_penalty", 0.05))

        # 情绪偏向冲突扣分（只在与段位不一致时扣）
        if bias_info["applied"] is False and bias_info["label"] is not None:
            confidence -= float(conf_cfg.get("emotion_bias_conflict_penalty", 0.05))

        # 混合信号扣分
        if mixed_signals:
            confidence -= float(conf_cfg.get("mixed_signals_penalty", 0.10))

        # 截断
        c_min = float(conf_cfg.get("min", 0.5))
        c_max = float(conf_cfg.get("max", 1.0))
        confidence = max(c_min, min(c_max, confidence))

        # 更新 boundary_dist 到 bias_info
        bias_info["boundary_dist"] = round(dist, 4)

        return confidence

    # ==================================================================
    # 静态工具
    # ==================================================================

    @staticmethod
    def _risk_to_route(risk: float, crisis: float, comfort: float, knowledge: float) -> str:
        if risk >= crisis:
            return "crisis"
        if risk >= comfort:
            return "comfort"
        if risk >= knowledge:
            return "knowledge"
        return "general"

    @staticmethod
    def _escalate_band(route: str) -> str:
        if route == "general":
            return "knowledge"
        if route == "knowledge":
            return "comfort"
        if route == "comfort":
            return "crisis"
        return route
