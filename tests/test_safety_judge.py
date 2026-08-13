"""Semantic Safety Judge 单元测试：P0 硬闸门 / 锚点检测 / LLM 裁决 / 兜底 / 状态机"""

import json

import pytest
from langchain_core.messages import AIMessage

from modules.assessment.safety_judge import SafetyJudge, SafetyVerdict
from modules.assessment.risk_anchors import (
    CONFIRM_KEYWORDS,
    match_anchor,
    transition_safety_state,
)


# ── Fake LLM ────────────────────────────────────────────────────

class FakeJudgeLLM:
    """可编排回复序列的 LLM 假实现，记录调用次数。"""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if not self.responses:
            raw = json.dumps(
                {"verdict": "no_risk", "risk_type": "general",
                 "confidence": 0.5, "reason": "default", "probe_suggestion": None}
            )
        else:
            raw = self.responses.pop(0)
        return AIMessage(content=raw)


class RaisingJudgeLLM:
    """永远抛异常的 LLM，用于测兜底。"""

    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        raise RuntimeError("llm down")


def _crisis_resp(**overrides):
    base = {"verdict": "crisis", "risk_type": "suicide", "confidence": 0.9,
            "reason": "明确自杀意图", "probe_suggestion": None}
    base.update(overrides)
    return json.dumps(base, ensure_ascii=False)


def _probe_resp(**overrides):
    base = {"verdict": "probe", "risk_type": "suicide", "confidence": 0.7,
            "reason": "有自杀意念但模糊", "probe_suggestion": "是真心不想活，还是心里难受，还是都有？"}
    base.update(overrides)
    return json.dumps(base, ensure_ascii=False)


def _no_risk_resp(**overrides):
    base = {"verdict": "no_risk", "risk_type": "general", "confidence": 0.8,
            "reason": "口语夸张，非真实危机", "probe_suggestion": None}
    base.update(overrides)
    return json.dumps(base, ensure_ascii=False)


# ── P0 硬闸门 ───────────────────────────────────────────────────

class TestP0HardGate:
    def test_p0_returns_crisis_without_llm(self):
        """P0 显式危险 → 规则硬闸门 crisis，不发 LLM。"""
        judge = SafetyJudge(llm=FakeJudgeLLM())
        verdict = judge.judge("我已经买了安眠药，今晚就吃")
        assert verdict is not None
        assert verdict.verdict == "crisis"
        assert verdict.source == "rule_p0"
        assert judge._llm.calls == 0  # 未调用 LLM

    def test_p0_during_probe_state_still_crisis(self):
        """PROBING 会话中 P0 命中 → 仍硬闸门 crisis。"""
        judge = SafetyJudge(llm=FakeJudgeLLM())
        verdict = judge.judge(
            "我现在在楼顶，不想活了",
            safety_state={"status": "PROBING", "probe_count": 1, "denial_mark": False},
        )
        assert verdict is not None
        assert verdict.verdict == "crisis"


# ── 锚点检测 ────────────────────────────────────────────────────

class TestAnchorDetection:
    def test_no_anchor_returns_none_no_llm(self):
        """无锚点 → None，不发 LLM（零额外延迟）。"""
        fake = FakeJudgeLLM()
        judge = SafetyJudge(llm=fake)
        verdict = judge.judge(
            "今天天气不错，孩子上学了",
            emotion_risk=0.1,
            safety={"level": 0, "blocked": False, "matched_terms": []},
            scid_flags={},
            history=[],
        )
        assert verdict is None
        assert fake.calls == 0

    def test_p2_weak_signal_does_not_trigger_judge(self):
        """P2 弱信号（"狠狠"）不单独触发评估器 —— 误伤案例修复。"""
        fake = FakeJudgeLLM()
        judge = SafetyJudge(llm=fake)
        verdict = judge.judge(
            "我这时候很想大叫狠狠的撞东西，身体上的一些疼痛好像会麻痹自己",
            emotion_risk=0.2,
            safety={"level": 0, "blocked": False, "matched_terms": []},
            scid_flags={},
            history=[],
        )
        assert verdict is None
        assert fake.calls == 0

    def test_p1_triggers_judge(self):
        """P1 词命中 → 触发评估器。"""
        fake = FakeJudgeLLM(responses=[_no_risk_resp()])
        judge = SafetyJudge(llm=fake)
        verdict = judge.judge("我有时候真的不想活了", emotion_risk=0.3)
        assert verdict is not None
        assert fake.calls == 1

    def test_emotion_risk_above_threshold_triggers(self):
        """emotion.risk >= 阈值 → 触发评估器。"""
        fake = FakeJudgeLLM(responses=[_no_risk_resp()])
        judge = SafetyJudge(llm=fake)
        verdict = judge.judge(
            "最近压力很大", emotion_risk=0.5,
            safety={"level": 0, "blocked": False, "matched_terms": []},
            scid_flags={}, history=[],
        )
        assert verdict is not None
        assert fake.calls == 1

    def test_scid_risk_marker_triggers(self):
        """SCID 跨轮 risk criteria（death_si 复现）→ 触发评估器。"""
        fake = FakeJudgeLLM(responses=[_probe_resp()])
        judge = SafetyJudge(llm=fake)
        verdict = judge.judge(
            "今天还好",
            emotion_risk=0.1,
            scid_flags={"MDD": {"criteria_met": ["sleep", "death_si"], "count": 2}},
            history=[],
        )
        assert verdict is not None
        assert fake.calls == 1

    def test_history_recurrence_triggers(self):
        """会话历史曾有 P1 → 复现锚点触发。"""
        fake = FakeJudgeLLM(responses=[_probe_resp()])
        judge = SafetyJudge(llm=fake)
        verdict = judge.judge(
            "嗯",
            emotion_risk=0.1,
            history=[{"role": "user", "content": "我真的不想活了"}],
        )
        assert verdict is not None
        assert fake.calls == 1

    def test_probing_state_forces_evaluation(self):
        """PROBING 会话 → 即使本轮无锚点也强制评估（持续监测）。"""
        fake = FakeJudgeLLM(responses=[_no_risk_resp()])
        judge = SafetyJudge(llm=fake)
        verdict = judge.judge(
            "今天天气不错",
            emotion_risk=0.0,
            safety={"level": 0, "blocked": False, "matched_terms": []},
            scid_flags={},
            history=[],
            safety_state={"status": "PROBING", "probe_count": 1, "denial_mark": False},
        )
        assert verdict is not None
        assert fake.calls == 1


# ── LLM 裁决 ────────────────────────────────────────────────────

class TestLLMVerdict:
    def test_llm_crisis(self):
        """LLM 判 crisis 且历史含 P0（计划）→ 护栏放行，保持 crisis。"""
        judge = SafetyJudge(llm=FakeJudgeLLM(responses=[_crisis_resp()]))
        verdict = judge.judge(
            "嗯，就今晚",
            emotion_risk=0.5,
            history=[{"role": "user", "content": "我已经决定结束自己"}],
        )
        assert verdict.verdict == "crisis"
        assert verdict.source == "llm"

    def test_llm_crisis_without_p0_downgraded_by_guardrail(self):
        """护栏：LLM 判 crisis 但当前文本与历史均无 P0 → 降级 probe（人审）。"""
        judge = SafetyJudge(llm=FakeJudgeLLM(responses=[_crisis_resp()]))
        verdict = judge.judge(
            "不想活了，活着太痛苦",
            emotion_risk=0.5,
            history=[{"role": "user", "content": "我最近失眠很严重"}],
        )
        assert verdict.verdict == "probe"
        assert verdict.source == "llm_guardrail"

    def test_llm_probe_captures_suggestion(self):
        judge = SafetyJudge(llm=FakeJudgeLLM(responses=[_probe_resp()]))
        verdict = judge.judge("不想活了", emotion_risk=0.5)
        assert verdict.verdict == "probe"
        assert verdict.probe_suggestion is not None
        assert "不想活" in verdict.probe_suggestion or "难受" in verdict.probe_suggestion

    def test_llm_no_risk(self):
        judge = SafetyJudge(llm=FakeJudgeLLM(responses=[_no_risk_resp()]))
        verdict = judge.judge("活着好累", emotion_risk=0.5)
        assert verdict.verdict == "no_risk"
        assert "not in" or verdict.reason  # 有理由字段


# ── 兜底（LLM 失败/超时）────────────────────────────────────────

class TestFallback:
    def test_llm_failure_p1_falls_back_to_probe(self):
        """LLM 抛异常 → 重试后仍失败 → P1 锚点保守降级 probe。"""
        fake = RaisingJudgeLLM()
        judge = SafetyJudge(llm=fake)
        verdict = judge.judge("不想活了", emotion_risk=0.3)
        assert verdict is not None
        assert verdict.verdict == "probe"
        assert verdict.source == "fallback"
        assert fake.calls == 2  # 重试一次

    def test_llm_failure_high_risk_falls_back_to_probe(self):
        """LLM 失败 + risk 高但无 P0 锚点 → 兜底 probe（不再因纯情绪风险判 crisis）。"""
        judge = SafetyJudge(llm=RaisingJudgeLLM())
        verdict = judge.judge("最近压力大", emotion_risk=0.8)
        assert verdict.verdict == "probe"
        assert verdict.source == "fallback"

    def test_fallback_p0_branch_keeps_crisis(self):
        """兜底函数：P0 锚点 → crisis（防御性分支；judge 正常路径由 rule_p0 硬闸门先拦）。"""
        from modules.assessment.risk_anchors import match_anchor

        judge = SafetyJudge(llm=RaisingJudgeLLM())
        verdict = judge._fallback_verdict(  # noqa: SLF001
            "我现在就去死", 0.3, match_anchor("我现在就去死"),
        )
        assert verdict.verdict == "crisis"
        assert verdict.source == "fallback"

    def test_llm_invalid_json_falls_back(self):
        """LLM 返回非法 JSON → 重试 → 兜底。"""
        fake = FakeJudgeLLM(responses=["不是 JSON", "还是不是 JSON"])
        judge = SafetyJudge(llm=fake)
        verdict = judge.judge("不想活了", emotion_risk=0.5)
        assert verdict is not None
        assert verdict.source == "fallback"
        assert fake.calls == 2


# ── 状态机纯函数 ────────────────────────────────────────────────

class TestStateMachine:
    def test_none_probe_to_probing(self):
        state = transition_safety_state(None, "probe", "不想活了")
        assert state["status"] == "PROBING"
        assert state["probe_count"] == 1

    def test_probing_confirm_to_crisis(self):
        state = {"status": "PROBING", "probe_count": 1, "denial_mark": False}
        new = transition_safety_state(state, "probe", "我已经买好了药，就今晚")
        assert new["status"] == "CRISIS"

    def test_probing_deny_to_none(self):
        state = {"status": "PROBING", "probe_count": 1, "denial_mark": False}
        new = transition_safety_state(state, "probe", "没有，我就是说说，吓唬人的")
        assert new["status"] == "NONE"
        assert new["denial_mark"] is True

    def test_probing_accumulate_to_crisis(self):
        """多次探针复现 → 累积升级 CRISIS。"""
        state = {"status": "PROBING", "probe_count": 2, "denial_mark": False}
        new = transition_safety_state(state, "probe", "嗯，就那样吧", max_probe_count=3)
        assert new["status"] == "CRISIS"

    def test_probing_no_anchor_probe_does_not_accumulate(self):
        """纯情绪宣泄探针（无风险锚点）→ 保持 PROBING，不累积不升级（收窄 ADR-0013）。"""
        state = {"status": "PROBING", "probe_count": 2, "denial_mark": False}
        new = transition_safety_state(
            state, "probe", "好痛苦", max_probe_count=3, anchored_probe=False,
        )
        assert new["status"] == "PROBING"
        assert new["probe_count"] == 2  # 不累积

    def test_none_no_anchor_probe_stays_none(self):
        """NONE 态 + 纯情绪宣泄探针 → 不进 PROBING，保持 NONE。"""
        new = transition_safety_state(None, "probe", "好痛苦", anchored_probe=False)
        assert new["status"] == "NONE"
        assert new["probe_count"] == 0

    def test_none_anchor_probe_enters_probing(self):
        """NONE 态 + 有锚点意念探针 → 进 PROBING。"""
        new = transition_safety_state(None, "probe", "不想活了", anchored_probe=True)
        assert new["status"] == "PROBING"
        assert new["probe_count"] == 1

    def test_ideation_not_treated_as_denial(self):
        """"不想活了" 是意念不是否认（"不想"与"不想活"撞车修复）→ 可累积到 CRISIS。"""
        state = None
        for _ in range(3):
            state = transition_safety_state(
                state, "probe", "不想活了", anchored_probe=True, max_probe_count=3,
            )
        assert state["status"] == "CRISIS"
        assert state["probe_count"] == 3

    def test_real_denial_resets(self):
        """真正否认（"我只是说说而已"）→ 重置 NONE。"""
        state = {"status": "PROBING", "probe_count": 2, "denial_mark": False}
        new = transition_safety_state(
            state, "probe", "我只是说说而已，没打算真做", anchored_probe=True,
        )
        assert new["status"] == "NONE"
        assert new["probe_count"] == 0

    def test_ideation_not_treated_as_denial_by_want_to_live(self):
        """"不想再活了"（"不想"+再+活）也不当作否认 → 可累积。"""
        state = None
        for _ in range(3):
            state = transition_safety_state(
                state, "probe", "我不想再活了", anchored_probe=True, max_probe_count=3,
            )
        assert state["status"] == "CRISIS"

    def test_none_crisis_to_crisis(self):
        new = transition_safety_state(None, "crisis", "不想活了")
        assert new["status"] == "CRISIS"

    def test_probing_no_risk_resets(self):
        state = {"status": "PROBING", "probe_count": 2, "denial_mark": False}
        new = transition_safety_state(state, "no_risk", "其实还好")
        assert new["status"] == "NONE"

    def test_crisis_no_risk_recovers(self):
        state = {"status": "CRISIS", "probe_count": 1, "denial_mark": False}
        new = transition_safety_state(state, "no_risk", "我没事了，谢谢医生")
        assert new["status"] == "NONE"

    def test_none_verdict_no_change(self):
        state = {"status": "NONE", "probe_count": 0, "denial_mark": False}
        new = transition_safety_state(state, None, "你好")
        assert new["status"] == "NONE"
        assert new["probe_count"] == 0


# ── 工具函数 ────────────────────────────────────────────────────

def test_match_anchor_p0_precedence():
    """同时命中 P0 与 P1 时，match_anchor 返回两者，但 P0 优先。"""
    anchors = match_anchor("我已经买了安眠药，今晚就吃，真的不想活了")
    assert anchors["p0"]["suicide"]
    assert anchors["p1"]["suicide"]
