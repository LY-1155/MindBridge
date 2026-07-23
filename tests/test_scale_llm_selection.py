"""LLM 语义量表选择 + 串行执行 — TDD 测试

测试修改后的 should_trigger (List[str]) 和 service.py 串行逻辑。
遵循 docs/adr/0005-conversational-scale-screening.md D5/D10 新决策。
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock


# ============================================================
# B1: should_trigger 返回列表 — 单量表匹配
# ============================================================

class TestShouldTriggerReturnsList:
    """D5: should_trigger 现在返回 List[str] 而非 bool。"""

    @pytest.fixture
    def mock_llm(self):
        """返回单量表 ISI 的 mock LLM。"""
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content='{"trigger": true, "scales": ["ISI"], "reason": "用户描述严重失眠症状"}'
        )
        return llm

    @pytest.fixture
    def orch(self, mock_llm):
        from modules.intervention.scale.orchestrator import ScaleOrchestrator
        from modules.intervention.scale.scorer import ScaleScorer
        return ScaleOrchestrator(llm=mock_llm, scorer=ScaleScorer())

    def test_should_trigger_returns_list(self, orch):
        """单量表匹配 → 返回包含量表名的列表。"""
        result = orch.should_trigger(
            "我每天晚上都睡不着，凌晨三四点才迷迷糊糊睡着",
            {"primary_emotion": "sadness", "intensity": 0.8},
        )
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert result == ["ISI"]

    def test_returns_empty_list_when_no_match(self, orch):
        """无匹配 → 返回空列表。"""
        orch._llm.invoke.return_value = MagicMock(
            content='{"trigger": false, "scales": [], "reason": "用户未描述任何可匹配的症状"}'
        )
        result = orch.should_trigger(
            "今天天气真好，适合出去玩。",
            {"primary_emotion": "neutral", "intensity": 0.5},
        )
        assert result == []

    def test_rejects_low_intensity_regardless_of_llm(self, orch):
        """intensity < 0.4 → 即使 LLM 返回量表也不触发。"""
        orch._llm.invoke.return_value = MagicMock(
            content='{"trigger": true, "scales": ["PHQ-9"], "reason": "用户情绪低落"}'
        )
        result = orch.should_trigger(
            "最近心情不太好。",
            {"primary_emotion": "sadness", "intensity": 0.2},
        )
        assert result == []


# ============================================================
# B2: 共病多量表匹配
# ============================================================

class TestComorbidityMultiScale:
    """D5: 共病场景返回多量表列表，按 relevance 排序。"""

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content=(
                '{"trigger": true, "scales": ["PHQ-9", "GAD-7"], '
                '"reason": "用户描述抑郁+焦虑共病: 情绪低落、兴趣丧失, 同时过度担忧、紧张不安"}'
            )
        )
        return llm

    @pytest.fixture
    def orch(self, mock_llm):
        from modules.intervention.scale.orchestrator import ScaleOrchestrator
        from modules.intervention.scale.scorer import ScaleScorer
        return ScaleOrchestrator(llm=mock_llm, scorer=ScaleScorer())

    def test_comorbidity_returns_multiple_scales(self, orch):
        """共病 → 返回多个量表名列表。"""
        result = orch.should_trigger(
            "我情绪特别低落，什么都不想做，同时每天担心各种事情停不下来。",
            {"primary_emotion": "sadness", "intensity": 0.7},
        )
        assert len(result) >= 2
        assert "PHQ-9" in result
        assert "GAD-7" in result

    def test_scales_ordered_by_relevance(self, orch):
        """返回列表保持 LLM 输出的 relevance 排序。"""
        result = orch.should_trigger(
            "心情很差，睡眠也差。",
            {"primary_emotion": "sadness", "intensity": 0.6},
        )
        # 第一项应该是最匹配的
        assert result[0] == "PHQ-9"


# ============================================================
# B0: scale catalog 构建
# ============================================================

class TestScaleCatalog:
    """验证量表目录构建：从 11 个 JSON 文件中提取 name + description。"""

    def test_catalog_contains_all_scales(self):
        """catalog 应包含所有 11 个量表。"""
        from modules.intervention.scale.orchestrator import _build_scale_catalog
        catalog = _build_scale_catalog()
        names = [s["name"] for s in catalog]
        assert len(names) == 11
        for expected in ["PHQ-9", "GAD-7", "ISI", "LSAS", "PCL-5", "ASRS",
                         "AUDIT", "MDQ", "OCI-R", "PHQ-15", "SCOFF"]:
            assert expected in names, f"Missing {expected}"

    def test_catalog_entry_has_required_fields(self):
        """每个条目有 name + display_name + description。"""
        from modules.intervention.scale.orchestrator import _build_scale_catalog
        catalog = _build_scale_catalog()
        for entry in catalog:
            assert "name" in entry
            assert "display_name" in entry
            assert "description" in entry
            assert len(entry["description"]) > 10
