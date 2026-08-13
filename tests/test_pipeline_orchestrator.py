import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Settings
from modules.factory import build_pipeline_services
from pipeline.orchestrator import EMERGENCY_SAFETY_LEVEL, run_pipeline
from schemas.contracts import PipelineInput, SafetyCheckResult


def test_run_pipeline_all_mock_json_roundtrip():
    inp = PipelineInput(text="你好", session_id="s1")
    out = run_pipeline(inp)
    assert out.contract_version == inp.contract_version
    assert isinstance(out.safety, dict)
    assert isinstance(out.emotion, dict)
    assert isinstance(out.route, dict)
    assert isinstance(out.intervention, dict)
    SafetyCheckResult.model_validate(out.safety)


def test_pipeline_stub_router_risk_band():
    """关掉路由 Mock 时使用 Stub 规则。"""
    cfg = Settings(
        MOCK_SAFETY=True,
        MOCK_EMOTION=False,
        MOCK_ROUTER=False,
        MOCK_INTERVENTION=True,
        DOCTOR_MODE=False,  # 聚焦路由测试，跳过语义安全评估器（避免真实 LLM 调用）
    )
    svc = build_pipeline_services(cfg)

    class FixedEmotion:
        def analyze(self, req):
            from schemas.contracts import EmotionTags

            return EmotionTags(primary_emotion="sadness", intensity=0.8, risk=0.5)

    svc.emotion = FixedEmotion()
    out = run_pipeline(PipelineInput(text="失眠"), services=svc)
    # risk=0.5 恰好落在 comfort 段（router_rules.json: comfort_risk=0.5，sadness 有段内偏向）
    assert out.route.get("route") == "comfort"


def test_safety_shortcut_skips_emotion_router_calls():
    class BlockingSafety:
        def check(self, req):
            return SafetyCheckResult(level=EMERGENCY_SAFETY_LEVEL, blocked=True, matched_terms=["x"])

    cfg = Settings(
        MOCK_SAFETY=False,
        MOCK_EMOTION=True,
        MOCK_ROUTER=True,
        MOCK_INTERVENTION=True,
    )
    svc = build_pipeline_services(cfg)
    svc.safety = BlockingSafety()

    class ShouldNotRun:
        def analyze(self, req):
            raise AssertionError("emotion should not run when safety short-circuits")

        def route(self, req):
            raise AssertionError("router should not run when safety short-circuits")

    svc.emotion = ShouldNotRun()
    svc.router = ShouldNotRun()

    out = run_pipeline(PipelineInput(text="紧急测试"), services=svc)
    assert out.stopped_after_safety is True
    assert out.route.get("route") == "crisis"
