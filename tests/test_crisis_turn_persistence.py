"""回归测试：危机路由消息落库（A3 修复）

修复前：crisis 路由直接 return/yield CrisisHandler 结果，从不 _save_turn——
用户高风险原话和 AI 危机话术在 messages 表无留痕，仅靠 safety_flags 间接记录。
修复后：_save_crisis_turn 落库用户原话 + 危机话术，且不初始化重量级 generator。
"""

import types

import pytest

from schemas.contracts import InterventionResult
from modules.intervention.crisis_handler import CrisisHandler
from modules.intervention.service import InterventionService


class _FakePush:
    def __init__(self, crisis_type="suicide"):
        self._type = crisis_type
        self.last_call = None

    def trigger(self, session_id, matched_terms, user_text, crisis_type=None):
        self.last_call = dict(session_id=session_id, matched_terms=matched_terms, user_text=user_text)
        from modules.safety.emergency_push import EmergencyPushResult
        return EmergencyPushResult(
            triggered=True,
            session_id=session_id,
            crisis_type=crisis_type or self._type,
            matched_terms=matched_terms,
            user_text=user_text,
            template="【紧急心理危机干预】请立即拨打 400-161-9995",
            template_title="测试危机模板",
            rescue_api_called=True,
            rescue_api_result={"status": "ok"},
            timestamp="2025-01-01T00:00:00",
        )


def _make_crisis_req():
    from schemas.contracts import InterventionRequest
    return InterventionRequest(
        user_text="我不想活了",
        route={"route": "crisis", "reason": "高危关键词匹配", "confidence": 0.95},
        emotion={"primary_emotion": "distress", "intensity": 0.9, "risk": 0.95},
        safety={"level": 2, "blocked": False, "matched_terms": ["自杀", "不想活"]},
        session_id="test-crisis-persist-001",
        user_id="test-user-001",
    )


class TestCrisisTurnPersistence:
    def test_crisis_route_saves_user_and_ai_messages(self):
        """intervene() 的 crisis 分支应调用 _save_crisis_turn，落库用户原话 + 危机话术"""
        calls = []

        class FakeSession:
            def __init__(self):
                # intervene() 在路由分发前会读 scale_state（service.py:313）
                self.metadata = types.SimpleNamespace(scale_state=None)

            def add_user_message(self, content):
                calls.append(("user", content))

            def add_ai_message(self, content):
                calls.append(("ai", content))

        svc = InterventionService(crisis_handler=CrisisHandler(push_service=_FakePush()))
        svc._get_session = lambda sid: FakeSession()
        # 跳过医生评估管线（与 A3 无关，避免测试绑定 assessor/SCID 的 session 字段）
        svc._run_doctor_assessment = lambda req, session, route: ("crisis", None, None)

        req = _make_crisis_req()
        result = svc.intervene(req)

        assert calls, "危机路径应落库消息"
        assert calls[0] == ("user", "我不想活了")
        assert calls[1] == ("ai", result.reply)
        assert result.reply  # 正常返回危机话术

    def test_crisis_save_when_session_none_is_silent(self):
        """会话不存在时 _save_crisis_turn 应静默跳过，不抛异常、不阻塞危机响应"""
        svc = InterventionService(crisis_handler=CrisisHandler(push_service=_FakePush()))
        svc._get_session = lambda sid: None

        result = svc.intervene(_make_crisis_req())

        assert isinstance(result, InterventionResult)
        assert result.emergency_triggered is True
