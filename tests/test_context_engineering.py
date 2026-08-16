"""包 B（上下文工程）回归测试：token 预算自适应截断 + 摘要注入 + LLM 滚动摘要 + 危机零延迟铁律。

测试目标（与包 A「每修必测」惯例一致）：
- token 预算动态选「摘要 + 最近 N 轮原文」（替代旧固定 10 轮）
- get_context_summary() + rolling_summary 注入 prompt
- 每 SUMMARY_EVERY_N_TURNS 轮触发 LLM 滚动摘要（仅普通对话路径）
- 危机路径零延迟：不初始化 generator（LLM/检索器）、不触发滚动摘要
"""

import pytest
from config.settings import settings
from schemas.contracts import InterventionRequest


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------

class FakeLLM:
    """Fake LLM 适配器：返回预设回复，记录最近一次调用参数。"""

    def __init__(self, response="我在听，你慢慢说。"):
        self.response = response
        self.last_messages = None

    def invoke(self, messages):
        if hasattr(messages, "to_messages"):
            self.last_messages = messages.to_messages()
        else:
            self.last_messages = messages
        from langchain_core.messages import AIMessage
        return AIMessage(content=self.response)


class FakePushService:
    """Fake EmergencyPushService，返回可控的危机话术。"""

    def __init__(self):
        self._template = (
            "【紧急心理危机干预】\n\n"
            "请立即拨打：\n"
            "   - 全国24小时心理危机干预热线：400-161-9995\n"
            "   - 急救电话：120\n\n"
            "你并不孤单。"
        )

    def trigger(self, session_id, matched_terms, user_text, crisis_type=None):
        from modules.safety.emergency_push import EmergencyPushResult
        return EmergencyPushResult(
            triggered=True,
            session_id=session_id,
            crisis_type=crisis_type or "suicide",
            matched_terms=matched_terms,
            user_text=user_text,
            template=self._template,
            template_title="测试危机模板",
            rescue_api_called=True,
            rescue_api_result={"status": "ok"},
            timestamp="2025-01-01T00:00:00",
        )


def _make_crisis_req(**overrides) -> InterventionRequest:
    """构建 crisis 路由的请求。"""
    defaults = dict(
        user_text="我不想活了",
        route={"route": "crisis", "reason": "高危关键词匹配", "confidence": 0.95},
        emotion={"primary_emotion": "distress", "intensity": 0.9, "risk": 0.95},
        safety={"level": 2, "blocked": False, "matched_terms": ["自杀", "不想活"]},
        session_id="test-ctx-crisis-001",
    )
    defaults.update(overrides)
    return InterventionRequest(**defaults)


def _get_session(sid, **kwargs):
    """创建/获取一个纯内存真实会话（use_database=false，走进程缓存）。"""
    from core.memory.session_memory import SessionManager
    SessionManager.remove_session(sid)
    kwargs.setdefault("use_database", False)
    return SessionManager.get_session(sid, **kwargs)


@pytest.fixture
def memory_session_cleanup():
    """收集本测试创建的 session id，测试结束后清理进程内存缓存。"""
    created = []

    def _track(sid):
        created.append(sid)

    yield _track

    from core.memory.session_memory import SessionManager
    for sid in created:
        SessionManager.remove_session(sid)


# ---------------------------------------------------------------------------
# 1. token 预算自适应选轮
# ---------------------------------------------------------------------------

class TestTokenBudgetAdaptiveTruncation:
    """固定 10 轮 → 按 token 预算动态选最近 N 轮原文。"""

    def test_smaller_budget_keeps_fewer_turns(self, memory_session_cleanup):
        from modules.intervention.generator import InterventionReplyGenerator

        sid = "test-budget-adaptive"
        memory_session_cleanup(sid)
        session = _get_session(sid)
        long_line = "我最近总是失眠，凌晨三四点就醒，醒了再也睡不着，白天完全没有精神，工作也做不下去，" * 2
        for _ in range(6):
            session.add_user_message(long_line)
            session.add_ai_message(long_line)

        gen = InterventionReplyGenerator(llm=FakeLLM())
        orig = settings.HISTORY_TOKEN_BUDGET
        try:
            settings.HISTORY_TOKEN_BUDGET = 400
            small = gen._format_history(sid)
            settings.HISTORY_TOKEN_BUDGET = 20000
            large = gen._format_history(sid)
        finally:
            settings.HISTORY_TOKEN_BUDGET = orig

        small_turns = small.count("用户：")
        large_turns = large.count("用户：")
        assert small_turns >= 1                    # 预算再紧也至少保留一轮
        assert large_turns > small_turns           # 预算大 → 保留更多轮
        assert "最近对话" in small

    def test_summary_consumes_budget_before_recent_turns(self, memory_session_cleanup):
        """预算先覆盖摘要前缀，余量才给最近原文。"""
        from modules.intervention.generator import InterventionReplyGenerator

        sid = "test-budget-summary"
        memory_session_cleanup(sid)
        session = _get_session(sid)
        long_line = "工作压力特别大，回家还要照顾孩子，感觉自己被掏空了，不知道还能撑多久，" * 2
        for _ in range(6):
            session.add_user_message(long_line)
            session.add_ai_message(long_line)
        # 注入一条较大的滚动摘要，压缩最近原文的可用预算
        session.set_rolling_summary("用户长期失眠、工作压力大" * 30, last_turn=12)

        gen = InterventionReplyGenerator(llm=FakeLLM())
        orig_budget, orig_turns = settings.HISTORY_TOKEN_BUDGET, settings.SUMMARY_EVERY_N_TURNS
        try:
            settings.HISTORY_TOKEN_BUDGET = 600
            settings.SUMMARY_EVERY_N_TURNS = 100  # 本轮不触发摘要生成
            with_summary = gen._format_history(sid)
            # 摘要字段被注入
            assert "早期对话摘要" in with_summary
        finally:
            settings.HISTORY_TOKEN_BUDGET = orig_budget
            settings.SUMMARY_EVERY_N_TURNS = orig_turns


# ---------------------------------------------------------------------------
# 2. 摘要注入 prompt（get_context_summary + rolling_summary）
# ---------------------------------------------------------------------------

class TestSummaryInjection:
    """get_context_summary() 与 LLM 滚动摘要都注入 prompt 的对话历史块。"""

    def test_context_and_rolling_summary_injected(self, memory_session_cleanup):
        from modules.intervention.generator import InterventionReplyGenerator

        sid = "test-summary-inject"
        memory_session_cleanup(sid)
        session = _get_session(sid)
        session.add_key_topic("失眠")
        session.add_key_topic("工作压力")
        session.set_rolling_summary("用户长期被失眠困扰，认为与工作压力直接相关。", last_turn=2)
        session.add_user_message("最近总是睡不好")
        session.add_ai_message("睡眠问题确实磨人，能说说是从什么时候开始的吗？")

        text = InterventionReplyGenerator(llm=FakeLLM())._format_history(sid)

        # 接线 get_context_summary()：事实摘要进 prompt
        assert "讨论的主要话题" in text
        assert "失眠" in text and "工作压力" in text
        assert "对话轮数" in text
        # LLM 滚动摘要注入
        assert "早期对话摘要" in text
        assert "用户长期被失眠困扰" in text
        # 最近原文仍保留
        assert "用户：最近总是睡不好" in text
        assert "助手：睡眠问题确实磨人" in text

    def test_no_session_or_empty_history_graceful(self, memory_session_cleanup):
        """session 为空 / session_id 为 None → 返回第一轮占位，不报错。"""
        from modules.intervention.generator import InterventionReplyGenerator

        gen = InterventionReplyGenerator(llm=FakeLLM())
        assert "第一轮" in gen._format_history(None)

        sid = "test-empty-hist"
        memory_session_cleanup(sid)
        _get_session(sid)
        assert "第一轮" in gen._format_history(sid)

    def test_count_tokens_smoke(self):
        from modules.intervention.generator import InterventionReplyGenerator

        gen = InterventionReplyGenerator(llm=FakeLLM())
        assert gen._count_tokens("") == 0
        assert gen._count_tokens("最近总是睡不好觉") > 0


# ---------------------------------------------------------------------------
# 3. LLM 滚动摘要：每 N 轮触发 + 滚动合并
# ---------------------------------------------------------------------------

class TestRollingSummary:
    """每 SUMMARY_EVERY_N_TURNS 轮把旧对话压成滚动摘要（对标 MemGPT working/summary）。"""

    def test_triggered_at_interval_and_rolls_forward(self, memory_session_cleanup):
        from modules.intervention.generator import InterventionReplyGenerator

        sid = "test-roll-trigger"
        memory_session_cleanup(sid)
        session = _get_session(sid)
        summary_llm = FakeLLM(response="用户长期失眠，与工作压力相关。")
        gen = InterventionReplyGenerator(llm=FakeLLM(), summary_llm=summary_llm)

        orig = settings.SUMMARY_EVERY_N_TURNS
        try:
            settings.SUMMARY_EVERY_N_TURNS = 2
            # 第 1 轮：未到间隔 → 不触发
            gen._save_turn(sid, "最近睡不着", "嗯，持续多久了？")
            assert session.metadata.rolling_summary is None
            # 第 2 轮：到间隔 → 触发
            gen._save_turn(sid, "快一周了", "那确实挺折磨的。")
            assert session.metadata.rolling_summary == "用户长期失眠，与工作压力相关。"
            assert session.metadata.rolling_summary_turn == 2
            # 第 3 轮：非间隔 → 不触发，摘要保持
            gen._save_turn(sid, "白天也没精神", "白天也受影响吗？")
            assert session.metadata.rolling_summary == "用户长期失眠，与工作压力相关。"
            # 第 4 轮：再触发，且把已有摘要并入输入（滚动合并）
            gen._save_turn(sid, "是，很影响", "理解了，这确实不容易。")
            assert session.metadata.rolling_summary_turn == 4
            assert "已有摘要" in summary_llm.last_messages[-1].content
            assert "用户长期失眠" in summary_llm.last_messages[-1].content
        finally:
            settings.SUMMARY_EVERY_N_TURNS = orig

    def test_summary_failure_does_not_break_turn_save(self, memory_session_cleanup):
        """摘要 LLM 抛异常 → 本轮回复仍落库，摘要保持为空。"""
        from modules.intervention.generator import InterventionReplyGenerator

        sid = "test-roll-fail"
        memory_session_cleanup(sid)
        session = _get_session(sid)

        class ExplodingSummaryLLM:
            def invoke(self, messages):
                raise RuntimeError("summary boom")

        gen = InterventionReplyGenerator(
            llm=FakeLLM(), summary_llm=ExplodingSummaryLLM(),
        )
        orig = settings.SUMMARY_EVERY_N_TURNS
        try:
            settings.SUMMARY_EVERY_N_TURNS = 1  # 每轮都尝试摘要
            gen._save_turn(sid, "最近睡不好", "嗯，持续多久了？")
        finally:
            settings.SUMMARY_EVERY_N_TURNS = orig

        assert len(session.get_history_for_prompt()) == 2  # 回复正常落库
        assert session.metadata.rolling_summary is None    # 摘要失败不阻断


# ---------------------------------------------------------------------------
# 4. 危机路径零延迟铁律
# ---------------------------------------------------------------------------

class TestCrisisPathZeroLatency:
    """铁律：crisis 路径零延迟——不初始化 generator（LLM/检索器），不触发滚动摘要。"""

    def test_crisis_never_initializes_generator(self):
        from modules.intervention.crisis_handler import CrisisHandler
        from modules.intervention.service import InterventionService

        handler = CrisisHandler(push_service=FakePushService())
        svc = InterventionService(crisis_handler=handler)
        result = svc.intervene(_make_crisis_req())
        assert result.emergency_triggered is True
        # 从未初始化 generator（= 未加载 LLM / 检索器 / BGE 重排）
        assert svc._generator is None

    def test_crisis_turn_does_not_roll_summary(self, memory_session_cleanup):
        from core.memory.session_memory import SessionManager
        from modules.intervention.crisis_handler import CrisisHandler
        from modules.intervention.service import InterventionService

        sid = "test-crisis-nosummary"
        memory_session_cleanup(sid)
        session = _get_session(sid)
        # 预置对话到超过摘要触发阈值（message_count 足够大）
        for _ in range(5):
            session.add_user_message("我很难受")
            session.add_ai_message("我在听，能多说一点吗")
        assert session.metadata.rolling_summary is None

        handler = CrisisHandler(push_service=FakePushService())
        svc = InterventionService(crisis_handler=handler)
        result = svc.intervene(_make_crisis_req(session_id=sid))
        assert result.emergency_triggered is True

        # 危机路径只落消息，绝不在摘要字段上做任何事（零 LLM 调用）
        reloaded = SessionManager.get_session(sid, use_database=False)
        assert reloaded.metadata.rolling_summary is None
        assert svc._generator is None
