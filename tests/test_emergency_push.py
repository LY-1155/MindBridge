"""
TDD #5: Emergency Push 真实通路测试

验证 dry-run / 生产模式切换、cooling 机制、HTTP 错误处理。
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(_ROOT))

import pytest
from modules.safety.emergency_push import (
    EmergencyPushService, EmergencyPushResult, get_emergency_push_service,
)


class TestDryRunMode:
    """dry-run 模式：不产生真实 HTTP 调用。"""

    def test_trigger_suicide_returns_template(self):
        svc = EmergencyPushService(cooldown_seconds=10, enabled=False)
        result = svc.trigger(
            session_id="s_dry_001",
            matched_terms=["自杀", "不想活"],
            user_text="我真的不想活了",
        )
        assert result.triggered is True
        assert result.crisis_type == "suicide"
        assert result.rescue_api_called is True
        assert result.rescue_api_result["status"] == "dry_run_success"
        assert "生命" in result.template_title

    def test_trigger_violence_classifies_correctly(self):
        svc = EmergencyPushService(cooldown_seconds=10, enabled=False)
        result = svc.trigger(
            session_id="s_dry_002",
            matched_terms=["暴力", "杀人"],
            user_text="我想报复",
        )
        assert result.triggered is True
        assert result.crisis_type == "violence"

    def test_self_harm_classifies_correctly(self):
        svc = EmergencyPushService(cooldown_seconds=10, enabled=False)
        result = svc.trigger(
            session_id="s_dry_003",
            matched_terms=["自残", "割腕"],
            user_text="我想伤害自己",
        )
        assert result.crisis_type == "self_harm"

    def test_unknown_falls_back_to_crisis(self):
        svc = EmergencyPushService(cooldown_seconds=10, enabled=False)
        result = svc.trigger(
            session_id="s_dry_004",
            matched_terms=["紧急情况"],
            user_text="有紧急情况",
        )
        assert result.crisis_type == "crisis"


class TestCooldown:
    """冷却期内不重复推送。"""

    def test_second_trigger_in_cooldown_blocked(self):
        svc = EmergencyPushService(cooldown_seconds=60, enabled=False)
        r1 = svc.trigger(session_id="s_cool_001", matched_terms=["自杀"], user_text="x")
        assert r1.triggered is True

        r2 = svc.trigger(session_id="s_cool_001", matched_terms=["自杀"], user_text="y")
        assert r2.triggered is False
        assert "冷却期" in r2.reason

    def test_different_sessions_independent(self):
        svc = EmergencyPushService(cooldown_seconds=60, enabled=False)
        r1 = svc.trigger(session_id="s_indep_a", matched_terms=["自杀"], user_text="a")
        r2 = svc.trigger(session_id="s_indep_b", matched_terms=["自杀"], user_text="b")
        assert r1.triggered is True
        assert r2.triggered is True

    def test_reset_cooldown_allows_retrigger(self):
        svc = EmergencyPushService(cooldown_seconds=60, enabled=False)
        svc.trigger(session_id="s_reset_001", matched_terms=["自杀"], user_text="x")
        svc.reset_cooldown("s_reset_001")
        r2 = svc.trigger(session_id="s_reset_001", matched_terms=["自杀"], user_text="y")
        assert r2.triggered is True


class TestProductionMode:
    """生产模式：真实 HTTP 调用（错误处理）。"""

    def test_bad_url_handles_error_gracefully(self):
        svc = EmergencyPushService(
            cooldown_seconds=10,
            enabled=True,
            rescue_api_url="http://127.0.0.1:19999/not-exist",
            rescue_api_key="test-key",
        )
        result = svc.trigger(
            session_id="s_prod_001",
            matched_terms=["自杀"],
            user_text="test",
        )
        assert result.triggered is True
        assert result.rescue_api_called is True
        # HTTP call should fail → status=error
        assert result.rescue_api_result["status"] == "error"


class TestResultSerialization:
    """to_dict 序列化。"""

    def test_to_dict_includes_all_keys(self):
        svc = EmergencyPushService(cooldown_seconds=10, enabled=False)
        result = svc.trigger(
            session_id="s_dict_001",
            matched_terms=["轻生"],
            user_text="我想轻生",
        )
        d = result.to_dict()
        for key in ("triggered", "session_id", "crisis_type", "matched_terms",
                     "template", "template_title", "rescue_api_called",
                     "rescue_api_result", "reason", "timestamp"):
            assert key in d, f"missing key: {key}"
