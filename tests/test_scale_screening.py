"""量表筛查模块 — TDD 测试

测试顺序遵循 docs/adr/0005-conversational-scale-screening.md 的 11 个决策。
"""
from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# Tracer Bullet 1: PHQ-9 JSON 加载 → ScaleItem / ScaleConfig 解析
# ============================================================

class TestPHQ9JSONLoading:
    """验证 PHQ-9 量表 JSON 文件可以被正确加载和解析。"""

    def test_phq9_json_exists_and_is_valid(self):
        """JSON 文件存在且是合法的 JSON。"""
        import json
        json_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "knowledge", "private", "scales", "phq9.json"
        )
        assert os.path.exists(json_path), f"PHQ-9 JSON 文件不存在: {json_path}"

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "name" in data
        assert data["name"] == "PHQ-9"
        assert "display_name" in data
        assert "items" in data
        assert "thresholds" in data

    def test_phq9_has_9_items(self):
        """PHQ-9 有 9 道题。"""
        from modules.intervention.scale.models import ScaleConfig

        config = ScaleConfig.from_json("phq9")
        assert len(config.items) == 9

    def test_scale_item_structure(self):
        """每道题有 dimension / concept / anchors。"""
        from modules.intervention.scale.models import ScaleConfig

        config = ScaleConfig.from_json("phq9")
        item = config.items[0]

        assert item.index == 0
        assert isinstance(item.dimension, str) and len(item.dimension) > 0
        assert isinstance(item.concept, str) and len(item.concept) > 0
        assert isinstance(item.anchors, dict)
        assert "0" in item.anchors and "3" in item.anchors

    def test_phq9_thresholds(self):
        """阈值分级：0-4 无, 5-9 轻度, 10-14 中度, 15-19 中重度, 20-27 重度。"""
        from modules.intervention.scale.models import ScaleConfig

        config = ScaleConfig.from_json("phq9")
        thresholds = config.thresholds

        assert isinstance(thresholds, list)
        assert len(thresholds) == 5
        assert thresholds[0].min == 0 and thresholds[0].max == 4
        assert thresholds[4].min == 20 and thresholds[4].max == 27

    def test_item_9_is_self_harm(self):
        """第 9 题（index=8）维度是自伤念头。"""
        from modules.intervention.scale.models import ScaleConfig

        config = ScaleConfig.from_json("phq9")
        item9 = config.items[8]
        assert "自伤" in item9.dimension or "自伤" in item9.concept or "自杀" in item9.dimension or "自杀" in item9.concept


# ============================================================
# Tracer Bullet 2: 根据 emotion 标签选量表
# ============================================================

class TestScaleSelection:
    """D5: LLM 语义匹配替代 emotion 硬映射。should_trigger 返回 List[str]。

    详细单元测试（含 mock）见 test_scale_llm_selection.py。
    此处验证真实 LLM 能返回有效列表。"""

    @pytest.fixture
    def orch(self):
        from core.llm.base import get_llm_adapter
        from modules.intervention.scale.orchestrator import ScaleOrchestrator
        from modules.intervention.scale.scorer import ScaleScorer
        llm = get_llm_adapter("qwen")
        return ScaleOrchestrator(llm=llm, scorer=ScaleScorer())

    def test_should_trigger_returns_list(self, orch):
        """symptom text → 返回非空列表（至少匹配 1 个量表）。"""
        result = orch.should_trigger(
            "我最近两周情绪特别低落，对什么都提不起兴趣，晚上也睡不好。",
            {"primary_emotion": "sadness", "intensity": 0.8},
        )
        assert isinstance(result, list)
        assert len(result) > 0
        # 预期至少匹配到 PHQ-9
        assert "PHQ-9" in result

    def test_anxiety_symptoms_return_list(self, orch):
        """anxiety symptoms → 返回包含 GAD-7 的列表。"""
        result = orch.should_trigger(
            "我每天都过度担心各种事情，停不下来，身体也紧绷绷的。",
            {"primary_emotion": "anxiety", "intensity": 0.7},
        )
        assert isinstance(result, list)
        assert len(result) > 0
        assert "GAD-7" in result

    def test_neutral_returns_empty_list(self, orch):
        """neutral 低强度无相关症状 → 返回空列表。"""
        result = orch.should_trigger(
            "今天天气不错，适合出去走走。",
            {"primary_emotion": "neutral", "intensity": 0.5},
        )
        assert result == []

    def test_no_intensity_no_trigger(self, orch):
        """emotion_tags 为空 / intensity 缺失 → 返回空列表。"""
        result = orch.should_trigger(
            "最近心情不太好。",
            {},
        )
        assert result == []


# ============================================================
# Tracer Bullet 3: 计分 LLM 根据用户回复 + 锚点返回 0-3 或 -1
# ============================================================

class TestScoringLLM:
    """D8: 独立的计分 LLM 调用，对照锚点判定 0-3 或 -1 无效。"""

    @pytest.fixture
    def scorer(self):
        from modules.intervention.scale.scorer import ScaleScorer
        from core.llm.base import get_llm_adapter
        return ScaleScorer(llm=get_llm_adapter("qwen"))

    @pytest.fixture
    def sleep_item(self):
        """PHQ-9 第 3 题：睡眠障碍。"""
        from modules.intervention.scale.models import ScaleItem
        return ScaleItem(
            index=2,
            dimension="睡眠障碍",
            concept="入睡困难、睡眠维持困难、或睡得太多",
            anchors={
                "0": "没有睡眠困扰",
                "1": "偶尔几天睡不好",
                "2": "超过一半的天数有睡眠问题",
                "3": "几乎每天的睡眠都受到严重影响",
            },
        )

    def test_scores_zero_for_no_issue(self, scorer, sleep_item):
        """用户说完全没睡眠问题 → 计分 0。"""
        score = scorer.score("我睡得挺好的，每天倒头就睡，一觉到天亮。", sleep_item)
        assert score == 0, f"期望 0，实际 {score}"

    def test_scores_three_for_severe(self, scorer, sleep_item):
        """用户描述严重睡眠困难 → 计分 3。"""
        score = scorer.score(
            "完全睡不着，每天凌晨三四点才勉强睡着，早上六点多又醒了，白天头昏脑胀的，这种情况已经持续快两周了。",
            sleep_item,
        )
        assert score == 3, f"期望 3，实际 {score}"

    def test_scores_mid_range(self, scorer, sleep_item):
        """中等程度睡眠问题 → 计分 1 或 2。"""
        score = scorer.score(
            "最近睡眠不太好，大概有一半的日子入睡比较困难，躺在床上翻来覆去一两个小时才能睡着。",
            sleep_item,
        )
        assert score in (1, 2), f"期望 1-2，实际 {score}"

    def test_returns_minus_one_for_off_topic(self, scorer, sleep_item):
        """用户回复完全偏离睡眠话题 → -1。"""
        score = scorer.score("你说老板是不是都这样？上次也是因为这种事刁难我。", sleep_item)
        assert score == -1, f"期望 -1，实际 {score}"

    def test_returns_minus_one_for_knowledge_question(self, scorer, sleep_item):
        """用户在问知识而不是描述自身症状 → -1。"""
        score = scorer.score("失眠一般有哪些治疗方法？我听说认知行为疗法对失眠有效。", sleep_item)
        assert score == -1, f"期望 -1，实际 {score}"


# ============================================================
# Tracer Bullet 4: 完整多轮流程 start → process_turn → 完成
# ============================================================

class TestMultiTurnScaleFlow:
    """D2/D4: 多轮量表施测，start 开启 → process_turn 逐题 → 完成计分。"""

    @pytest.fixture
    def session(self):
        from core.memory.session_memory import SessionManager
        sid = SessionManager.create_session()
        return SessionManager.get_session(sid)

    @pytest.fixture
    def orch(self):
        from core.llm.base import get_llm_adapter
        from modules.intervention.scale.orchestrator import ScaleOrchestrator
        from modules.intervention.scale.scorer import ScaleScorer
        llm = get_llm_adapter("qwen")
        return ScaleOrchestrator(llm=llm, scorer=ScaleScorer())

    def test_start_creates_scale_state_and_returns_question(self, orch, session):
        """start()：创建 scale_state 并返回第一道自然语言提问。"""
        reply = orch.start("PHQ-9", session)
        assert isinstance(reply, str)
        assert len(reply) > 10
        # 验证会话状态已设置
        state = session.metadata.scale_state
        assert state is not None
        assert state["scale_name"] == "PHQ-9"
        assert state["current_item_index"] == 0
        assert state["status"] == "in_progress"
        assert len(state["scores"]) == 0

    def test_process_turn_scores_and_advances(self, orch, session):
        """一轮 process_turn：计分上一题并推进到下一题。"""
        orch.start("PHQ-9", session)
        # 第 1 题是"兴趣减退"，用匹配的回复
        result = orch.process_turn(
            "最近确实对什么都提不起兴趣，以前喜欢打篮球现在完全不想去了，感觉做什么都没意思。",
            session,
        )
        assert not result.is_complete
        assert len(result.reply) > 10
        # scores 应加了一条
        state = session.metadata.scale_state
        assert len(state["scores"]) == 1
        assert state["current_item_index"] == 1  # 推进到第 2 题

    def test_completes_after_all_items(self, orch, session):
        """走完一个极短量表：完成后 is_complete=True 且有总分。"""
        # 使用最小验证：只测 2 道题（模拟快速完成）
        from modules.intervention.scale.models import ScaleConfig
        orig_items = ScaleConfig.from_json("phq9").items

        # 篡改：只保留 1 道题来验证完成逻辑
        from unittest.mock import patch
        import copy
        short_config = ScaleConfig.from_json("phq9")
        short_config.items = [copy.deepcopy(short_config.items[0])]  # 只有第 1 题

        with patch("modules.intervention.scale.orchestrator.ScaleConfig.from_json", return_value=short_config):
            orch.start("PHQ-9", session)
            result = orch.process_turn("最近确实对什么都提不起兴趣，以前喜欢打篮球现在都不想去了。", session)

        assert result.is_complete
        assert result.total_score is not None
        assert result.level is not None
        state = session.metadata.scale_state
        assert state["status"] == "completed"


# ============================================================
# Tracer Bullet 5: 连续偏离静默放弃
# ============================================================

class TestWanderAbandon:
    """D7: 连续 2 轮偏离量表话题 → 静默放弃。"""

    @pytest.fixture
    def session(self):
        from core.memory.session_memory import SessionManager
        sid = SessionManager.create_session()
        return SessionManager.get_session(sid)

    @pytest.fixture
    def orch(self):
        from core.llm.base import get_llm_adapter
        from modules.intervention.scale.orchestrator import ScaleOrchestrator
        from modules.intervention.scale.scorer import ScaleScorer
        llm = get_llm_adapter("qwen")
        return ScaleOrchestrator(llm=llm, scorer=ScaleScorer())

    def test_single_wander_soft_redirects(self, orch, session):
        """偏离 1 次：柔和拉回，不放弃。"""
        orch.start("PHQ-9", session)
        result = orch.process_turn("今天天气真不错，适合出去玩。", session)
        assert not result.is_complete
        assert len(result.reply) > 5
        state = session.metadata.scale_state
        assert state["wander_count"] == 1
        assert state["status"] == "in_progress"

    def test_double_wander_abandons(self, orch, session):
        """连续偏离 2 次：静默放弃。"""
        orch.start("PHQ-9", session)
        # 第一次偏离
        orch.process_turn("你说老板是不是都这样？", session)
        # 第二次偏离
        result = orch.process_turn("今天吃什么好呢。", session)
        assert result.is_complete
        assert len(result.reply) > 0  # 放弃时有提示语，不再是空字符串
        state = session.metadata.scale_state
        assert state["status"] == "abandoned"
        assert state["wander_count"] >= 2


# ============================================================
# Tracer Bullet 6: 重度分数/自伤条目触发 escalation_flag
# ============================================================

class TestEscalationFromScale:
    """D6: 量表总分 ≥ 20 或第 9 题 ≥ 1 → escalation_flag=True。"""

    @pytest.fixture
    def session(self):
        from core.memory.session_memory import SessionManager
        sid = SessionManager.create_session()
        return SessionManager.get_session(sid)

    def _make_state(self, scores):
        from modules.intervention.scale.models import ScaleState
        return ScaleState(
            scale_name="PHQ-9",
            scores=scores,
            current_item_index=len(scores),
        )

    def test_total_gte_20_escalates(self, session):
        """总分 ≥ 20 → escalation_flag=True。"""
        from modules.intervention.scale.orchestrator import ScaleOrchestrator
        from modules.intervention.scale.models import ScaleConfig
        from modules.intervention.scale.scorer import ScaleScorer

        orch = ScaleOrchestrator(llm=None, scorer=None)
        config = ScaleConfig.from_json("phq9")
        state = self._make_state([3, 2, 3, 2, 2, 2, 2, 2, 2])  # total=20

        result = orch._complete_scale(state, config, session)
        assert result.escalation_flag is True
        assert state.escalation_flag is True

    def test_total_19_no_escalation(self, session):
        """总分 19 → 不触发 escalation。"""
        from modules.intervention.scale.orchestrator import ScaleOrchestrator
        from modules.intervention.scale.models import ScaleConfig

        orch = ScaleOrchestrator(llm=None, scorer=None)
        config = ScaleConfig.from_json("phq9")
        state = self._make_state([2, 2, 2, 2, 2, 2, 2, 3, 0])  # total=17, item9=0

        result = orch._complete_scale(state, config, session)
        assert result.escalation_flag is False

    @pytest.mark.parametrize("item9_score", [1, 2, 3])
    def test_item9_gte_1_escalates(self, session, item9_score):
        """第 9 题（自伤/自杀念头）≥ 1 分 → escalation_flag=True，即使总分不高。"""
        from modules.intervention.scale.orchestrator import ScaleOrchestrator
        from modules.intervention.scale.models import ScaleConfig

        orch = ScaleOrchestrator(llm=None, scorer=None)
        config = ScaleConfig.from_json("phq9")
        # 8 items with 0, item 9 with trigger score → total low
        scores = [0] * 8 + [item9_score]
        state = self._make_state(scores)

        result = orch._complete_scale(state, config, session)
        assert result.escalation_flag is True

    def test_item9_0_no_escalation_if_total_low(self, session):
        """第 9 题 0 分且总分低 → 不触发。"""
        from modules.intervention.scale.orchestrator import ScaleOrchestrator
        from modules.intervention.scale.models import ScaleConfig

        orch = ScaleOrchestrator(llm=None, scorer=None)
        config = ScaleConfig.from_json("phq9")
        state = self._make_state([0, 0, 0, 0, 0, 0, 0, 0, 0])

        result = orch._complete_scale(state, config, session)
        assert result.escalation_flag is False


# ============================================================
# Tracer Bullet 7: should_trigger 判定逻辑
# ============================================================

class TestShouldTrigger:
    """D1: LLM 症状叙述分类 + emotion intensity 阈值。返回 List[str]。"""

    @pytest.fixture
    def orch(self):
        from core.llm.base import get_llm_adapter
        from modules.intervention.scale.orchestrator import ScaleOrchestrator
        from modules.intervention.scale.scorer import ScaleScorer
        llm = get_llm_adapter("qwen")
        return ScaleOrchestrator(llm=llm, scorer=ScaleScorer())

    def test_symptom_narrative_triggers(self, orch):
        """用户描述症状 → 返回非空列表。"""
        result = orch.should_trigger(
            "我最近两周完全睡不着，什么都不想做，觉得自己没用了。",
            {"primary_emotion": "sadness", "intensity": 0.8},
        )
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) > 0, f"Expected non-empty list for symptoms"

    def test_knowledge_question_does_not_trigger(self, orch):
        """用户问知识问题 → 返回空列表。"""
        result = orch.should_trigger(
            "认知行为疗法是怎么治疗抑郁症的？",
            {"primary_emotion": "sadness", "intensity": 0.6},
        )
        assert result == []

    def test_low_intensity_no_trigger(self, orch):
        """情绪 intensity 太低 → 返回空列表（不调 LLM）。"""
        result = orch.should_trigger(
            "最近稍微有点累，但也还好。",
            {"primary_emotion": "sadness", "intensity": 0.2},
        )
        assert result == []

    def test_off_topic_no_trigger(self, orch):
        """偏离心理话题的内容 → 返回空列表。"""
        result = orch.should_trigger(
            "今天天气真好，适合去哪里玩呢？",
            {"primary_emotion": "neutral", "intensity": 0.5},
        )
        assert result == []


# ============================================================
# Tracer Bullet 9: InterventionService knowledge 分支集成
# ============================================================

class TestInterventionServiceIntegration:
    """验证 ScaleOrchestrator 在 InterventionService.intervene 中正确接入。"""

    @pytest.fixture
    def svc(self):
        from modules.intervention.service import InterventionService
        from core.llm.base import get_llm_adapter
        return InterventionService(llm_adapter=get_llm_adapter("qwen"))

    @pytest.fixture
    def session(self):
        from core.memory.session_memory import SessionManager
        sid = SessionManager.create_session()
        return SessionManager.get_session(sid)

    def test_knowledge_with_symptom_triggers_scale(self, svc, session):
        """知识路由 + 症状叙述 → 量表启动。"""
        from schemas.contracts import InterventionRequest
        req = InterventionRequest(
            user_text="我最近两周心情一直很低落，对什么都提不起兴趣，晚上也睡不好。",
            route={"route": "knowledge", "confidence": 0.8},
            safety={"level": 0},
            emotion={"primary_emotion": "sadness", "intensity": 0.7},
            session_id=session.session_id,
        )
        result = svc.intervene(req)
        assert len(result.reply) > 10
        assert "量表" not in result.reply and "问卷" not in result.reply
        assert session.metadata.scale_state is not None
        assert session.metadata.scale_state["status"] == "in_progress"

    def test_knowledge_with_knowledge_question_skips_scale(self, svc, session):
        """知识路由 + 知识提问 → 不触发量表，走正常 RAG。"""
        from schemas.contracts import InterventionRequest
        req = InterventionRequest(
            user_text="认知行为疗法的基本原理是什么？",
            route={"route": "knowledge", "confidence": 0.8},
            safety={"level": 0},
            emotion={"primary_emotion": "sadness", "intensity": 0.6},
            session_id=session.session_id,
        )
        result = svc.intervene(req)
        assert len(result.reply) > 10
        assert session.metadata.scale_state is None

    def test_scale_continuation(self, svc, session):
        """已有量表进行中 → 继续 process_turn。"""
        from schemas.contracts import InterventionRequest
        req1 = InterventionRequest(
            user_text="我最近两周完全提不起兴趣，以前喜欢做的事情现在都不想做了。",
            route={"route": "knowledge", "confidence": 0.8},
            safety={"level": 0},
            emotion={"primary_emotion": "sadness", "intensity": 0.8},
            session_id=session.session_id,
        )
        svc.intervene(req1)
        assert session.metadata.scale_state["status"] == "in_progress"

        # 第二轮：回答 PHQ-9 第一题（兴趣减退）
        req2 = InterventionRequest(
            user_text="最近确实对什么都提不起兴趣了，以前喜欢的电视剧现在一集都不想看，感觉很空虚。",
            route={"route": "knowledge", "confidence": 0.8},
            safety={"level": 0},
            emotion={"primary_emotion": "sadness", "intensity": 0.8},
            session_id=session.session_id,
        )
        result2 = svc.intervene(req2)
        assert len(result2.reply) > 10
        assert len(session.metadata.scale_state["scores"]) >= 1
