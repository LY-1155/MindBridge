"""干预模块 TDD 测试：InterventionService + CrisisHandler + InterventionReplyGenerator"""

import pytest
from schemas.contracts import InterventionRequest, InterventionResult


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------

class FakePushService:
    """Fake EmergencyPushService，返回可控的危机话术"""

    def __init__(self, crisis_type="suicide"):
        self._crisis_type = crisis_type
        self.last_call = None
        self._template = (
            "【紧急心理危机干预】\n\n"
            "您好，我们非常重视您当前的状态。\n"
            "请立即拨打：\n"
            "   - 全国24小时心理危机干预热线：400-161-9995\n"
            "   - 急救电话：120\n\n"
            "你并不孤单。"
        )

    def trigger(self, session_id, matched_terms, user_text, crisis_type=None):
        self.last_call = dict(
            session_id=session_id,
            matched_terms=matched_terms,
            user_text=user_text,
            crisis_type=crisis_type,
        )
        from modules.safety.emergency_push import EmergencyPushResult

        ct = crisis_type or self._crisis_type
        return EmergencyPushResult(
            triggered=True,
            session_id=session_id,
            crisis_type=ct,
            matched_terms=matched_terms,
            user_text=user_text,
            template=self._template,
            template_title="测试危机模板",
            rescue_api_called=True,
            rescue_api_result={"status": "ok"},
            timestamp="2025-01-01T00:00:00",
        )


def _make_crisis_req(**overrides) -> InterventionRequest:
    """构建 crisis 路由的请求"""
    defaults = dict(
        user_text="我不想活了",
        route={"route": "crisis", "reason": "高危关键词匹配", "confidence": 0.95},
        emotion={"primary_emotion": "distress", "intensity": 0.9, "risk": 0.95},
        safety={"level": 2, "blocked": False, "matched_terms": ["自杀", "不想活"]},
        session_id="test-crisis-001",
    )
    defaults.update(overrides)
    return InterventionRequest(**defaults)


def _make_comfort_req(**overrides) -> InterventionRequest:
    """构建 comfort 路由的请求"""
    defaults = dict(
        user_text="最近工作压力很大，睡不好觉",
        route={"route": "comfort", "reason": "低风险共情", "confidence": 0.9},
        emotion={"primary_emotion": "stress", "intensity": 0.6, "risk": 0.2},
        safety={"level": 0, "blocked": False, "matched_terms": []},
        session_id="test-comfort-001",
    )
    defaults.update(overrides)
    return InterventionRequest(**defaults)


def _make_knowledge_req(**overrides) -> InterventionRequest:
    """构建 knowledge 路由的请求"""
    defaults = dict(
        user_text="我最近总是很焦虑，有什么方法可以缓解吗",
        route={"route": "knowledge", "reason": "中风险认知干预", "confidence": 0.8},
        emotion={"primary_emotion": "anxiety", "intensity": 0.7, "risk": 0.45},
        safety={"level": 0, "blocked": False, "matched_terms": []},
        session_id="test-knowledge-001",
    )
    defaults.update(overrides)
    return InterventionRequest(**defaults)


class FakeLLM:
    """Fake LLM 适配器，返回预设回复，记录调用参数"""

    def __init__(self, response="我理解你的感受，这确实不容易。"):
        self.response = response
        self.last_messages = None

    def invoke(self, messages):
        # LCEL ChatPromptTemplate produces ChatPromptValue; convert to list for test assertions
        if hasattr(messages, 'to_messages'):
            self.last_messages = messages.to_messages()
        else:
            self.last_messages = messages
        from langchain_core.messages import AIMessage
        return AIMessage(content=self.response)


# ---------------------------------------------------------------------------
# 第 1 轮：Crisis 路由 → 确定性话术模板
# ---------------------------------------------------------------------------

class TestCrisisRouteDeterministicTemplate:
    """Crisis 路由：不走 LLM，走 EmergencyPushService 确定性模板"""

    def test_crisis_returns_template_from_push_service(self):
        """crisis 路由时，reply 包含危机热线模板，emergency_triggered=True"""
        from modules.intervention.crisis_handler import CrisisHandler
        from modules.intervention.service import InterventionService

        fake_push = FakePushService(crisis_type="suicide")
        handler = CrisisHandler(push_service=fake_push)
        svc = InterventionService(crisis_handler=handler)

        req = _make_crisis_req()
        result = svc.intervene(req)

        # 核心断言
        assert isinstance(result, InterventionResult)
        assert result.emergency_triggered is True
        assert "400-161-9995" in result.reply
        assert result.empathy == ""  # v1 不做结构化拆解，留空字符串
        assert result.chain_of_thought is None

        # action_items 由代码硬填，不靠 LLM
        assert any("400-161-9995" in item for item in result.action_items)

        # meta 记录实现路径
        assert result.meta["implementation"] == "crisis_template"
        assert result.meta["crisis_type"] == "suicide"

    def test_crisis_passes_safety_matched_terms_to_push(self):
        """危机路由应将 safety.matched_terms 传给 EmergencyPushService"""
        from modules.intervention.crisis_handler import CrisisHandler
        from modules.intervention.service import InterventionService

        fake_push = FakePushService()
        handler = CrisisHandler(push_service=fake_push)
        svc = InterventionService(crisis_handler=handler)

        req = _make_crisis_req(
            safety={"level": 2, "blocked": False, "matched_terms": ["割腕", "自残"]}
        )
        svc.intervene(req)

        assert fake_push.last_call is not None
        assert "割腕" in fake_push.last_call["matched_terms"]
        assert "自残" in fake_push.last_call["matched_terms"]
        assert fake_push.last_call["session_id"] == "test-crisis-001"
        assert fake_push.last_call["user_text"] == "我不想活了"

    def test_different_crisis_types_map_correctly(self):
        """不同危机类型应输出对应模板"""
        from modules.intervention.crisis_handler import CrisisHandler
        from modules.intervention.service import InterventionService

        for ct in ["suicide", "violence", "self_harm", "crisis"]:
            fake_push = FakePushService(crisis_type=ct)
            handler = CrisisHandler(push_service=fake_push)
            svc = InterventionService(crisis_handler=handler)

            result = svc.intervene(_make_crisis_req())
            assert result.emergency_triggered is True
            assert result.meta["crisis_type"] == ct
            assert len(result.reply) > 50  # 模板非空


# ---------------------------------------------------------------------------
# 第 3 轮：Comfort 路由 → LLM 共情生成
# ---------------------------------------------------------------------------

class TestComfortRouteLLMGeneration:
    """Comfort 路由：走 LLM 生成共情回复，不执行自身情绪/安全分析"""

    def test_comfort_calls_llm_with_emotion_context(self):
        """comfort 路由调用 LLM，prompt 含管线产出的情绪标签"""
        from modules.intervention.generator import InterventionReplyGenerator
        from modules.intervention.service import InterventionService

        fake_llm = FakeLLM(response="我听到你的压力了。工作上的事情确实会让人睡不好。")
        generator = InterventionReplyGenerator(llm=fake_llm)
        svc = InterventionService(generator=generator)

        req = _make_comfort_req()
        result = svc.intervene(req)

        assert isinstance(result, InterventionResult)
        assert "压力" in result.reply
        assert result.emergency_triggered is False
        assert result.empathy == ""
        assert result.meta["implementation"] == "llm_comfort"

        # LLM 被正确调用
        assert fake_llm.last_messages is not None

        # prompt 应包含情绪上下文和用户原文
        system_content = str(fake_llm.last_messages[0].content)
        assert "stress" in system_content.lower() or "共情" in system_content
        human_content = str(fake_llm.last_messages[-1].content)
        assert req.user_text in human_content

    def test_comfort_does_not_call_crisis_handler(self):
        """comfort 路由不走 CrisisHandler"""
        from modules.intervention.generator import InterventionReplyGenerator
        from modules.intervention.service import InterventionService

        fake_llm = FakeLLM()
        generator = InterventionReplyGenerator(llm=fake_llm)
        svc = InterventionService(generator=generator)

        result = svc.intervene(_make_comfort_req())

        assert result.emergency_triggered is False
        assert "crisis" not in result.meta.get("implementation", "")

    def test_comfort_route_does_not_run_own_emotion_analysis(self):
        """comfort 路由不重新分析情绪，管线结论为唯一输入"""
        from modules.intervention.generator import InterventionReplyGenerator
        from modules.intervention.service import InterventionService

        fake_llm = FakeLLM(response="我理解。")
        generator = InterventionReplyGenerator(llm=fake_llm)
        svc = InterventionService(generator=generator)

        req = _make_comfort_req(
            emotion={"primary_emotion": "anxiety", "intensity": 0.7, "risk": 0.3}
        )
        result = svc.intervene(req)

        assert result.reply == "我理解。"
        # 没有调用额外的情绪分析 API —— 唯一的 LLM 调用就是回复生成
        assert "anxiety" in str(fake_llm.last_messages[0].content).lower()


# ---------------------------------------------------------------------------
# 第 4 轮：Knowledge 路由 → RAG + LLM
# ---------------------------------------------------------------------------

class FakeRetriever:
    """Fake RAG 检索器，返回预设知识片段"""

    def __init__(self, documents=None):
        self.documents = documents or ["深呼吸和正念练习可以缓解焦虑"]
        self.last_query = None

    def retrieve(self, query: str, top_k: int = 3):
        self.last_query = query
        return self.documents[:top_k]


class TestKnowledgeRouteRAGLLM:
    """Knowledge 路由：先 RAG 检索，结果注入 prompt，再交给 LLM 生成"""

    def test_knowledge_calls_rag_before_llm(self):
        """knowledge 路由时，RAG 在 LLM 之前被调用"""
        from modules.intervention.generator import InterventionReplyGenerator
        from modules.intervention.service import InterventionService

        retriever = FakeRetriever(["认知行为疗法(CBT)是处理焦虑的有效方法"])
        fake_llm = FakeLLM(response="CBT可以帮助你识别和调整负面思维模式。")
        generator = InterventionReplyGenerator(llm=fake_llm, retriever=retriever)
        svc = InterventionService(generator=generator)

        req = _make_knowledge_req()
        result = svc.intervene(req)

        # RAG 被调用
        assert retriever.last_query is not None
        assert "焦虑" in retriever.last_query

        # LLM 生成的回复含知识内容
        assert "CBT" in result.reply
        assert result.emergency_triggered is False
        assert result.meta["implementation"] == "llm_knowledge"

    def test_knowledge_injects_rag_results_into_prompt(self):
        """RAG 检索结果被注入到 LLM prompt 中"""
        from modules.intervention.generator import InterventionReplyGenerator
        from modules.intervention.service import InterventionService

        retriever = FakeRetriever(["正念冥想：每天10分钟专注呼吸可降低焦虑水平"])
        fake_llm = FakeLLM(response="正念冥想是很好的方法。")
        generator = InterventionReplyGenerator(llm=fake_llm, retriever=retriever)
        svc = InterventionService(generator=generator)

        req = _make_knowledge_req()
        svc.intervene(req)

        # LLM prompt 应包含检索到的知识
        system_content = str(fake_llm.last_messages[0].content)
        assert "正念冥想" in system_content
        assert "10分钟" in system_content

    def test_knowledge_sets_suggestion_with_source(self):
        """knowledge 路由的 suggestion 填知识来源"""
        from modules.intervention.generator import InterventionReplyGenerator
        from modules.intervention.service import InterventionService

        retriever = FakeRetriever(["渐进式肌肉放松法：从脚趾到头部逐步放松"])
        fake_llm = FakeLLM(response="你可以试试渐进式肌肉放松。")
        generator = InterventionReplyGenerator(llm=fake_llm, retriever=retriever)
        svc = InterventionService(generator=generator)

        result = svc.intervene(_make_knowledge_req())
        assert result.suggestion != ""
        assert "知识来源" in result.suggestion


# ---------------------------------------------------------------------------
# 第 5 轮：低置信度回退
# ---------------------------------------------------------------------------

class TestLowConfidenceFallback:
    """confidence < 0.5 时，knowledge → comfort 回退；crisis 不降级"""

    def test_knowledge_low_confidence_falls_back_to_comfort(self):
        """knowledge 路由 + confidence < 0.5 → 退回 comfort 风格，不走 RAG"""
        from modules.intervention.generator import InterventionReplyGenerator
        from modules.intervention.service import InterventionService

        retriever = FakeRetriever(["CBT相关知识"])
        fake_llm = FakeLLM(response="我理解你的困惑。让我们先聊聊你现在的感受。")
        generator = InterventionReplyGenerator(llm=fake_llm, retriever=retriever)
        svc = InterventionService(generator=generator)

        req = _make_knowledge_req(
            route={"route": "knowledge", "reason": "中风险但信号模糊", "confidence": 0.35}
        )
        result = svc.intervene(req)

        # 回退到 comfort：提示词里不应该有知识库检索内容
        assert "comfort" in result.meta["implementation"]
        assert retriever.last_query is None  # RAG 未被调用
        assert "知识来源" not in result.suggestion

    def test_crisis_does_not_fallback_on_low_confidence(self):
        """crisis 路由即使 confidence 低也照发危机模板，安全第一"""
        from modules.intervention.crisis_handler import CrisisHandler
        from modules.intervention.service import InterventionService

        fake_push = FakePushService(crisis_type="crisis")
        handler = CrisisHandler(push_service=fake_push)
        svc = InterventionService(crisis_handler=handler)

        req = _make_crisis_req(
            route={"route": "crisis", "reason": "边界模糊", "confidence": 0.3}
        )
        result = svc.intervene(req)

        assert result.emergency_triggered is True
        assert "400-161-9995" in result.reply
        assert result.meta["implementation"] == "crisis_template"


# ---------------------------------------------------------------------------
# 第 6 轮：会话记忆注入 — 历史读取 + 回复写回
# ---------------------------------------------------------------------------

class FakeSession:
    """Fake TherapySessionMemory：可预设历史，追踪写入的消息"""

    def __init__(self, history=None):
        self.history = list(history) if history else []
        self.added_messages = []

    def get_history_for_prompt(self):
        return list(self.history)

    def add_user_message(self, content):
        self.added_messages.append(("user", content))

    def add_ai_message(self, content):
        self.added_messages.append(("ai", content))


class FakeSessionStore:
    """Fake SessionManager 接口：通过 session_id 返回预设的 FakeSession"""

    def __init__(self, sessions=None):
        self._sessions = sessions or {}

    def get_session(self, session_id):
        return self._sessions.get(session_id)


class TestSessionHistoryInjection:
    """验证会话历史被注入到 LLM prompt 中"""

    def test_comfort_injects_history_into_prompt(self):
        """comfort 路由：prompt 包含对话历史内容"""
        from modules.intervention.generator import InterventionReplyGenerator
        from modules.intervention.service import InterventionService

        session = FakeSession(history=[
            {"role": "user", "content": "我最近总是失眠"},
            {"role": "assistant", "content": "失眠确实让人痛苦，能和我说说是什么时候开始的吗？"},
        ])
        store = FakeSessionStore(sessions={"test-comfort-001": session})
        fake_llm = FakeLLM(response="结合你之前的失眠情况，工作压力可能是加重因素。")
        generator = InterventionReplyGenerator(llm=fake_llm, session_store=store)
        svc = InterventionService(generator=generator)

        req = _make_comfort_req(session_id="test-comfort-001")
        svc.intervene(req)

        system_content = str(fake_llm.last_messages[0].content)
        assert "失眠" in system_content
        assert "什么时候开始的" in system_content

    def test_comfort_empty_history_does_not_crash(self):
        """无历史时 prompt 正常生成，不报错"""
        from modules.intervention.generator import InterventionReplyGenerator
        from modules.intervention.service import InterventionService

        session = FakeSession(history=[])
        store = FakeSessionStore(sessions={"test-comfort-002": session})
        fake_llm = FakeLLM(response="我能理解你的感受。")
        generator = InterventionReplyGenerator(llm=fake_llm, session_store=store)
        svc = InterventionService(generator=generator)

        req = _make_comfort_req(session_id="test-comfort-002")
        result = svc.intervene(req)

        assert "我能理解你的感受" in result.reply
        system_content = str(fake_llm.last_messages[0].content)
        assert "第一轮" in system_content

    def test_no_session_id_does_not_crash(self):
        """session_id 为 None 时静默跳过，不影响主流程"""
        from modules.intervention.generator import InterventionReplyGenerator
        from modules.intervention.service import InterventionService

        fake_llm = FakeLLM(response="没关系的，我在这里。")
        generator = InterventionReplyGenerator(llm=fake_llm)
        svc = InterventionService(generator=generator)

        req = _make_comfort_req(session_id=None)
        result = svc.intervene(req)

        assert "没关系的" in result.reply

    def test_knowledge_injects_history_into_prompt(self):
        """knowledge 路由：prompt 也包含对话历史"""
        from modules.intervention.generator import InterventionReplyGenerator
        from modules.intervention.service import InterventionService

        session = FakeSession(history=[
            {"role": "user", "content": "我焦虑的时候会手抖"},
            {"role": "assistant", "content": "这是焦虑的常见躯体反应，你注意到了这一点很好。"},
        ])
        store = FakeSessionStore(sessions={"test-knowledge-002": session})
        retriever = FakeRetriever(["深呼吸练习可以缓解焦虑躯体化症状"])
        fake_llm = FakeLLM(response="基于你之前的描述，深呼吸可以帮你稳定身体反应。")
        generator = InterventionReplyGenerator(
            llm=fake_llm, retriever=retriever, session_store=store,
        )
        svc = InterventionService(generator=generator)

        req = _make_knowledge_req(session_id="test-knowledge-002")
        svc.intervene(req)

        system_content = str(fake_llm.last_messages[0].content)
        assert "手抖" in system_content
        assert "躯体反应" in system_content


class TestSessionWriteBack:
    """验证生成的回复被写回到会话存储"""

    def test_comfort_writes_user_and_ai_back(self):
        """comfort 路由：生成回复后，用户消息和 AI 回复都写入 session"""
        from modules.intervention.generator import InterventionReplyGenerator
        from modules.intervention.service import InterventionService

        session = FakeSession()
        store = FakeSessionStore(sessions={"test-writeback-001": session})
        fake_llm = FakeLLM(response="工作压力大的时候确实会影响睡眠，可以试试睡前放松练习。")
        generator = InterventionReplyGenerator(llm=fake_llm, session_store=store)
        svc = InterventionService(generator=generator)

        req = _make_comfort_req(session_id="test-writeback-001")
        result = svc.intervene(req)

        # 验证写回
        assert len(session.added_messages) == 2
        assert session.added_messages[0] == ("user", req.user_text)
        assert session.added_messages[1] == ("ai", result.reply)

    def test_knowledge_writes_user_and_ai_back(self):
        """knowledge 路由同样写回用户消息和 AI 回复"""
        from modules.intervention.generator import InterventionReplyGenerator
        from modules.intervention.service import InterventionService

        session = FakeSession()
        store = FakeSessionStore(sessions={"test-writeback-002": session})
        retriever = FakeRetriever(["正念冥想可以缓解焦虑"])
        fake_llm = FakeLLM(response="你可以试试正念冥想来缓解焦虑。")
        generator = InterventionReplyGenerator(
            llm=fake_llm, retriever=retriever, session_store=store,
        )
        svc = InterventionService(generator=generator)

        req = _make_knowledge_req(session_id="test-writeback-002")
        result = svc.intervene(req)

        assert len(session.added_messages) == 2
        assert session.added_messages[0] == ("user", req.user_text)
        assert session.added_messages[1] == ("ai", result.reply)

    def test_no_session_id_skips_writeback(self):
        """session_id 为 None 时跳过写回，不报错"""
        from modules.intervention.generator import InterventionReplyGenerator
        from modules.intervention.service import InterventionService

        fake_llm = FakeLLM(response="我理解。")
        generator = InterventionReplyGenerator(llm=fake_llm)
        svc = InterventionService(generator=generator)

        req = _make_comfort_req(session_id=None)
        result = svc.intervene(req)

        assert result.reply == "我理解。"
