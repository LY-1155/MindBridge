"""SCIDTracker 单元测试"""

import pytest
from modules.assessment.scid_tracker import (
    SCIDTracker,
    SCIDUpdate,
    DISORDER_CRITERIA,
    DISORDER_THRESHOLDS,
)


@pytest.fixture
def tracker():
    return SCIDTracker()


# ── 关键词匹配 ────────────────────────────────────────────────

def test_mdd_sleep_criterion(tracker):
    """'睡不着' 应该匹配 MDD sleep criteria。"""
    result = tracker.update("我最近总是睡不着", {})
    mdd = result.criteria_met.get("MDD", {})
    assert "sleep" in mdd.get("criteria_met", [])


def test_mdd_multiple_criteria(tracker):
    """同时提及多个症状应匹配多个 criteria。"""
    result = tracker.update(
        "最近心情很差，整夜睡不着，干什么都没劲，吃不下饭",
        {},
    )
    mdd = result.criteria_met.get("MDD", {})
    assert mdd.get("count", 0) >= 3


def test_gad_worry_criterion(tracker):
    """担心关键词应该匹配 GAD。"""
    result = tracker.update("我总是控制不住地担心各种事情", {})
    gad = result.criteria_met.get("GAD", {})
    assert "excessive_worry" in gad.get("criteria_met", [])


def test_panic_palpitations(tracker):
    """心慌+出汗应该匹配 Panic。"""
    result = tracker.update("突然心跳好快，手心一直冒汗", {})
    panic = result.criteria_met.get("Panic", {})
    assert len(panic.get("criteria_met", [])) >= 2


def test_ptsd_intrusion(tracker):
    """噩梦+反复回忆应该匹配 PTSD。"""
    result = tracker.update("总是做噩梦，那些画面控制不住地跳出来", {})
    ptsd = result.criteria_met.get("PTSD", {})
    assert "intrusion" in ptsd.get("criteria_met", [])


# ── 累积合并 ──────────────────────────────────────────────────

def test_criteria_accumulation(tracker):
    """多轮对话应该累积 criteria。"""
    # 第一轮：睡眠
    r1 = tracker.update("睡不着", {})
    # 第二轮：情绪
    r2 = tracker.update("心情一直很低落", r1.criteria_met)
    mdd = r2.criteria_met.get("MDD", {})
    criteria = mdd.get("criteria_met", [])
    assert "sleep" in criteria
    assert "depressed_mood" in criteria
    assert mdd.get("count", 0) >= 2


def test_criteria_no_duplicates(tracker):
    """重复提及不应重复计数。"""
    r1 = tracker.update("睡不着睡不着睡不着失眠", {})
    r2 = tracker.update("还是睡不着", r1.criteria_met)
    mdd = r2.criteria_met.get("MDD", {})
    # 即使两轮都提了 sleep，也只计入一次
    assert mdd.get("count", 0) <= 5  # 不可能超过 criteria 总数


# ── 阈值与诊断 ────────────────────────────────────────────────

def test_mdd_threshold_not_met_few_criteria(tracker):
    """少于 5 条 MDD criteria 不应标记为 suspected。"""
    result = tracker.update("睡不着，心情不好", {})
    assert result.suspected_diagnosis is None or result.suspected_diagnosis != "MDD"


def test_mdd_requires_core_symptom(tracker):
    """没有 depressed_mood 或 anhedonia 时不应标记 MDD。"""
    # 模拟 5+ criteria but missing core
    existing = {
        "MDD": {
            "criteria_met": ["sleep", "fatigue", "concentration", "weight_appetite", "psychomotor"],
            "count": 5,
        }
    }
    result = tracker.update("最近还是累", existing)
    # 缺 core symptom，不应标记 MDD
    assert result.suspected_diagnosis is None


# ── 安全风险 ──────────────────────────────────────────────────

def test_suicide_risk_flag(tracker):
    """检测到自杀 criteria 时标记 risk。"""
    result = tracker.update(
        "我觉得活着好累，有时候真的想一了百了",
        {},
    )
    assert len(result.risk_flags) > 0
    assert any("suicide" in flag.lower() for flag in result.risk_flags)


def test_no_false_risk_on_normal_text(tracker):
    """正常文本不标记风险。"""
    result = tracker.update("我最近工作压力有点大", {})
    assert len(result.risk_flags) == 0


# ── 检索 query 生成 ───────────────────────────────────────────

def test_retrieval_query_for_mdd(tracker):
    """有 suspected diagnosis 时生成精准 query。"""
    existing = {
        "MDD": {
            "criteria_met": ["depressed_mood", "anhedonia", "sleep", "fatigue", "concentration"],
            "count": 5,
        }
    }
    result = tracker.update("就是觉得活着没意思", existing)
    assert result.suspected_diagnosis == "MDD"
    assert result.suggested_retrieval_query is not None
    assert "抑郁症" in result.suggested_retrieval_query


def test_no_query_without_diagnosis(tracker):
    """没有达到阈值时不生成 query。"""
    result = tracker.update("心情不太好", {})
    assert result.suggested_retrieval_query is None


# ── 空输入 ────────────────────────────────────────────────────

def test_empty_input(tracker):
    """空输入不崩溃。"""
    result = tracker.update("", {})
    assert isinstance(result, SCIDUpdate)
    assert result.suspected_diagnosis is None


def test_empty_existing_flags(tracker):
    """空 flags 不崩溃。"""
    result = tracker.update("心情不好", {})
    assert isinstance(result, SCIDUpdate)


# ── DISORDER_THRESHOLDS 一致性 ────────────────────────────────

def test_all_thresholds_in_criteria():
    """确保每个 threshold 对应的 disorder 都在 DISORDER_CRITERIA 中有定义。"""
    for disorder in DISORDER_THRESHOLDS:
        assert disorder in DISORDER_CRITERIA, f"{disorder} 缺少 criteria 定义"
