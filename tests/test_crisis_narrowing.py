"""危机收窄（v3.1）行为测试：意念→软警告 / 计划·即刻·主动意图→硬拦

覆盖：
- sensitive_words.json 等级分流（SensitivityFilter 直接 check）
- StubSafetyService 复核门控（L2 意念不复核、P0 跳过复核、L1 非 P0 才复核）
- 非 DOCTOR_MODE 下"现在就去死"仍硬拦短路、"不想活了"不再硬拦
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(_ROOT))

import pytest

from modules.safety.keyword_filter import SensitivityFilter
from modules.safety.stub import StubSafetyService
from schemas.contracts import SafetyCheckRequest


# ── 1. 敏感词等级分流 ───────────────────────────────────────────

class TestSensitiveWordTiering:
    @pytest.fixture(scope="class")
    def sf(self):
        return SensitivityFilter()

    def test_ideation_soft_warning(self, sf):
        """纯意念/情绪表达 → 永不硬拦；命中关键词者记 level 2 软警告。"""
        # 命中关键词的意念表达 → (False, 2)
        for text in ["不想活了", "活着没意思", "好想解脱", "我想死"]:
            blocked, level, _ = sf.check(text, user_id=None)
            assert (blocked, level) == (False, 2), f"{text} 应软警告 (False,2)，实得 ({blocked},{level})"
        # 口语化变体/情绪句未命中任何关键词 → (False, 0)，也不硬拦（语义交给 judge 裁决）
        for text in ["活着太痛苦", "还不如不活了"]:
            blocked, level, _ = sf.check(text, user_id=None)
            assert blocked is False, f"{text} 不应硬拦，实得 blocked={blocked} level={level}"

    def test_method_noun_soft_warning(self, sf):
        """方法名词（无计划即刻）→ 不硬拦。"""
        for text in ["我有时候想跳楼", "看到割腕的新闻", "听说过天台"]:
            blocked, level, _ = sf.check(text, user_id=None)
            assert blocked is False, f"{text} 不应硬拦"

    def test_plan_imminent_hard_block(self, sf):
        """计划/即刻/准备/主动第一人称意图 → (True, 1)。

        user_id=None 绕过 5 分钟防重复冷却（同用户连续硬拦只推一次，
        测试复用同一实例会误触冷却）。
        """
        hard = [
            "我现在就去死", "我现在就要去死", "马上就去死",
            "我已经决定结束自己", "遗书写好了", "正在割腕",
            "买了安眠药", "我要自杀", "我想自杀",
        ]
        for text in hard:
            blocked, level, _ = sf.check(text, user_id=None)
            assert (blocked, level) == (True, 1), f"{text} 应硬拦 (True,1)，实得 ({blocked},{level})"

    def test_negation_downgrade(self, sf):
        """否定语境（没有/不会）→ 硬拦词降级软警告。"""
        for text in ["我还没有决定结束自己", "我不会现在就去死"]:
            blocked, level, _ = sf.check(text, user_id=None)
            assert (blocked, level) == (False, 2), f"{text} 应降级 (False,2)"


# ── 2. StubSafetyService 复核门控 ─────────────────────────────

class TestStubSafetyGate:
    def _check(self, text, monkeypatch):
        svc = StubSafetyService()
        called = []
        import modules.safety.stub as stub_mod

        def fake_verify(t, kws):
            called.append((t, kws))
            return (False, "mock: 非真实危机")

        monkeypatch.setattr(stub_mod, "_verify_with_llm", fake_verify)
        result = svc.check(SafetyCheckRequest(
            contract_version="1.2", text=text, session_id="s-gate",
        ))
        return result, called

    def test_l2_ideation_not_verified_level_kept(self, monkeypatch):
        """L2 意念 → contract level=1（不降 0、keywords 保留），复核不被调用。"""
        result, called = self._check("不想活了", monkeypatch)
        assert result.level == 1
        assert result.blocked is False
        assert any("不想活" in term for term in (result.matched_terms or []))
        assert called == []

    def test_p0_skips_verify_hard_blocked(self, monkeypatch):
        """P0 即刻/主动意图 → 跳过复核，直接 blocked。"""
        result, called = self._check("我现在就去死", monkeypatch)
        assert result.level == 2
        assert result.blocked is True
        assert called == []

    def test_l1_non_p0_verify_downgrade(self, monkeypatch):
        """L1 非 P0（如"遗书"）→ 复核判误报 → 降级 level 0。"""
        result, called = self._check("我只是梦到遗书", monkeypatch)
        assert result.level == 0
        assert result.blocked is False
        assert len(called) == 1


# ── 3. 非 DOCTOR_MODE 快捷路径 ────────────────────────────────

class TestNonDoctorModeShortcut:
    def _build(self):
        from config.settings import Settings
        from modules.factory import build_pipeline_services
        from pipeline.orchestrator import run_pipeline
        from schemas.contracts import PipelineInput

        cfg = Settings(
            MOCK_SAFETY=False,
            MOCK_EMOTION=True,
            MOCK_ROUTER=True,
            MOCK_INTERVENTION=True,
            DOCTOR_MODE=False,
        )
        svc = build_pipeline_services(cfg)
        return run_pipeline, svc, PipelineInput

    def test_imminent_hard_shortcut(self):
        run_pipeline, svc, PipelineInput = self._build()
        out = run_pipeline(PipelineInput(text="我现在就去死", session_id="s-nodoc-1"), services=svc)
        assert out.stopped_after_safety is True
        assert out.route.get("route") == "crisis"

    def test_ideation_no_shortcut(self):
        run_pipeline, svc, PipelineInput = self._build()
        out = run_pipeline(PipelineInput(text="不想活了", session_id="s-nodoc-2"), services=svc)
        assert out.stopped_after_safety is False
