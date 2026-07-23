"""
Router 模块 · TDD 测试套件
==========================

从公共接口验证路由行为，不依赖实现细节。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from schemas.contracts import RouteDecision, RouteRequest
from modules.router.router_service import RouterService


_RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "router_rules.json"


@pytest.fixture(scope="module")
def svc():
    """模块级 fixture，复用 RouterService 实例。"""
    return RouterService(str(_RULES_PATH))


def _make_req(
    risk: float = 0.5,
    primary_emotion: str = "neutral",
    intensity: float = 0.5,
    safety_level: int = 0,
    safety_blocked: bool = False,
    matched_terms: list | None = None,
    mixed_signals: bool = False,
    intent: str = "unknown",
) -> RouteRequest:
    """构造标准 RouteRequest，减少测试样板代码。"""
    return RouteRequest(
        emotion={
            "primary_emotion": primary_emotion,
            "intensity": intensity,
            "risk": risk,
            "intent": intent,
            "modality_notes": {
                "mixed_signals": mixed_signals,
            },
        },
        safety={
            "level": safety_level,
            "blocked": safety_blocked,
            "matched_terms": matched_terms or [],
        },
    )


# ---------------------------------------------------------------------------
# 规则配置加载
# ---------------------------------------------------------------------------

class TestRulesLoading:
    """验证 router_rules.json 可被正常加载、结构完整。"""

    def test_rules_file_exists(self):
        assert _RULES_PATH.exists(), f"配置文件缺失: {_RULES_PATH}"

    def test_rules_json_valid(self):
        rules = json.loads(_RULES_PATH.read_text(encoding="utf-8"))
        assert "thresholds" in rules
        assert "emotion_bias" in rules
        assert "confidence" in rules
        assert "llm_fallback" in rules
        assert "safety_escalation" in rules

    def test_thresholds_reasonable(self):
        rules = json.loads(_RULES_PATH.read_text(encoding="utf-8"))
        t = rules["thresholds"]
        assert 0.0 < t["knowledge_risk"] < t["comfort_risk"] < t["crisis_risk"] <= 1.0

    def test_all_eight_emotions_defined(self):
        rules = json.loads(_RULES_PATH.read_text(encoding="utf-8"))
        expected = {"neutral", "anxiety", "sadness", "anger", "fear", "stress", "happiness", "confusion"}
        assert set(rules["emotion_bias"].keys()) == expected


# ---------------------------------------------------------------------------
# Risk 四段路由
# ---------------------------------------------------------------------------

class TestRiskBandRouting:
    """risk 值应正确映射到四个路由段位。"""

    def test_very_low_risk_routes_to_general(self, svc):
        assert svc.route(_make_req(risk=0.05)).route == "general"

    def test_low_risk_routes_to_knowledge(self, svc):
        assert svc.route(_make_req(risk=0.2)).route == "knowledge"

    def test_mid_risk_routes_to_comfort(self, svc):
        assert svc.route(_make_req(risk=0.5)).route == "comfort"

    def test_high_risk_routes_to_crisis(self, svc):
        assert svc.route(_make_req(risk=0.8)).route == "crisis"


# ---------------------------------------------------------------------------
# 阈值边界（>=）
# ---------------------------------------------------------------------------

class TestThresholdBoundary:
    """临界值使用 >= 语义，归高风险侧。"""

    def test_risk_0_7_is_crisis(self, svc):
        assert svc.route(_make_req(risk=0.7)).route == "crisis"

    def test_risk_0_4_is_knowledge(self, svc):
        assert svc.route(_make_req(risk=0.4)).route == "knowledge"

    def test_risk_0_15_is_knowledge(self, svc):
        assert svc.route(_make_req(risk=0.15)).route == "knowledge"

    def test_risk_0_05_is_general(self, svc):
        assert svc.route(_make_req(risk=0.05)).route == "general"


# ---------------------------------------------------------------------------
# Safety 升段
# ---------------------------------------------------------------------------

class TestSafetyEscalation:
    """safety.level>=2 应将路由升一段。"""

    def test_level_2_escalates_general_to_knowledge(self, svc):
        result = svc.route(_make_req(risk=0.05, safety_level=2))
        assert result.route == "knowledge"

    def test_level_2_escalates_knowledge_to_comfort(self, svc):
        result = svc.route(_make_req(risk=0.2, safety_level=2))
        assert result.route == "comfort"

    def test_level_2_escalates_comfort_to_crisis(self, svc):
        result = svc.route(_make_req(risk=0.5, safety_level=2))
        assert result.route == "crisis"

    def test_level_2_on_crisis_stays_crisis(self, svc):
        result = svc.route(_make_req(risk=0.8, safety_level=2))
        assert result.route == "crisis"


# ---------------------------------------------------------------------------
# 情绪偏向 — meta 反映偏向
# ---------------------------------------------------------------------------

class TestEmotionBias:
    """情绪类型影响 meta 和 reason，但不跨段改变 route。"""

    def test_anxiety_bias_in_comfort_band(self, svc):
        result = svc.route(_make_req(risk=0.5, primary_emotion="anxiety"))
        assert result.route == "comfort"
        assert result.meta.get("emotion_bias_applied") is True
        assert "anxiety" in result.meta.get("emotion_bias", "")

    def test_confusion_bias_in_comfort_band(self, svc):
        result = svc.route(_make_req(risk=0.5, primary_emotion="confusion"))
        assert result.route == "comfort"
        assert result.meta.get("emotion_bias_applied") is False

    def test_neutral_has_no_bias(self, svc):
        result = svc.route(_make_req(risk=0.5, primary_emotion="neutral"))
        assert result.meta.get("emotion_bias") is None


# ---------------------------------------------------------------------------
# Anger 高强度反转
# ---------------------------------------------------------------------------

class TestAngerHighIntensityReversal:
    """anger 在 intensity > 0.7 时偏向反转为 comfort。"""

    def test_anger_high_intensity_reverses_to_comfort(self, svc):
        result = svc.route(_make_req(risk=0.5, primary_emotion="anger", intensity=0.8))
        assert "anger_high_intensity" in str(result.meta)
        assert result.meta.get("emotion_bias") is not None


# ---------------------------------------------------------------------------
# 跨段禁止
# ---------------------------------------------------------------------------

class TestCrossBandLock:
    """情绪偏向不能跨段改变 route 值。"""

    def test_anger_in_knowledge_band_stays_knowledge(self, svc):
        result = svc.route(_make_req(risk=0.2, primary_emotion="anger"))
        assert result.route == "knowledge"
        assert result.meta.get("emotion_bias_applied") is True

    def test_sadness_in_comfort_band_stays_comfort(self, svc):
        result = svc.route(_make_req(risk=0.5, primary_emotion="sadness"))
        assert result.route == "comfort"
        assert result.meta.get("emotion_bias_applied") is True


# ---------------------------------------------------------------------------
# 意图覆盖 — information intent 提升 general → knowledge
# ---------------------------------------------------------------------------

class TestIntentOverride:
    """纯信息提问（intent=information）应从 general 提升到 knowledge。"""

    def test_information_intent_promotes_general_to_knowledge(self, svc):
        result = svc.route(_make_req(risk=0.02, intent="information"))
        assert result.route == "knowledge"
        assert result.meta.get("intent_override") is True
        assert "intent_override" in result.reason

    def test_information_stays_knowledge(self, svc):
        """已在 knowledge 段的不需要 override，但也标记 intent 信息。"""
        result = svc.route(_make_req(risk=0.2, intent="information"))
        assert result.route == "knowledge"
        assert result.meta.get("intent_override") is False

    def test_information_does_not_override_comfort(self, svc):
        """有情绪时，即使 intent=information 也不降段 — risk 优先。"""
        result = svc.route(_make_req(risk=0.6, intent="information"))
        assert result.route == "comfort"
        assert result.meta.get("intent_override") is False

    def test_unknown_intent_no_override(self, svc):
        """intent=unknown 不触发覆盖，保持 risk 路由结果。"""
        result = svc.route(_make_req(risk=0.02, intent="unknown"))
        assert result.route == "general"
        assert result.meta.get("intent_override") is False

    def test_intent_override_then_safety_escalation(self, svc):
        """intent 覆盖后仍可被 safety 升段。"""
        result = svc.route(_make_req(risk=0.02, intent="information", safety_level=2))
        assert result.route == "comfort"  # general→knowledge→comfort
        assert result.meta.get("intent_override") is True
        assert result.meta.get("safety_escalated") is True


# ---------------------------------------------------------------------------
# 置信度计算
# ---------------------------------------------------------------------------

class TestConfidence:
    """验证各种扣分因子对 confidence 的影响。"""

    def test_boundary_near_reduces_confidence(self, svc):
        result = svc.route(_make_req(risk=0.65))
        assert result.confidence < 0.9

    def test_safety_escalate_reduces_confidence(self, svc):
        result = svc.route(_make_req(risk=0.2, safety_level=2))
        assert result.confidence < 1.0, f"safety 升段应扣分，实际 {result.confidence}"

    def test_mixed_signals_reduces_confidence(self, svc):
        result = svc.route(_make_req(risk=0.5, mixed_signals=True))
        assert result.confidence < 0.95, f"混合信号应扣分，实际 {result.confidence}"

    def test_confidence_bounded_at_min(self, svc):
        result = svc.route(_make_req(risk=0.68, safety_level=1, mixed_signals=True, primary_emotion="anger"))
        assert result.confidence >= 0.5


# ---------------------------------------------------------------------------
# 情绪与段位同向 → 不扣分
# ---------------------------------------------------------------------------

class TestBiasSameDirection:
    """情绪偏向与段位方向一致时，confidence 不应被情绪修正扣分。"""

    def test_anxiety_in_comfort_no_extra_penalty(self, svc):
        result = svc.route(_make_req(risk=0.6, primary_emotion="anxiety"))
        assert result.confidence > 0.9, f"期望高置信度，实际 {result.confidence}"

    def test_confusion_in_knowledge_no_extra_penalty(self, svc):
        result = svc.route(_make_req(risk=0.2, primary_emotion="confusion"))
        assert result.confidence > 0.9, f"期望高置信度，实际 {result.confidence}"


# ---------------------------------------------------------------------------
# reason / meta 格式
# ---------------------------------------------------------------------------

class TestReasonFormat:
    """路由结果的结构完整性。"""

    def test_meta_has_risk_level(self, svc):
        result = svc.route(_make_req(risk=0.5))
        assert result.meta.get("risk_level") in ("none", "low", "mid", "high")

    def test_meta_contains_required_fields(self, svc):
        result = svc.route(_make_req(risk=0.5))
        meta = result.meta
        for key in ("risk_level", "safety_escalated", "intent_override",
                     "intent", "emotion_bias", "emotion_bias_applied",
                     "boundary_distance", "mixed_signals",
                     "llm_fallback_triggered", "llm_agreement"):
            assert key in meta, f"meta 缺少字段: {key}"
