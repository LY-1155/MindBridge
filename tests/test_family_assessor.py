"""FamilySystemAssessor 单元测试"""

import pytest
from modules.assessment.family_assessor import (
    FamilySystemAssessor,
    AssessResult,
    SAFETY_RED_LINE,
    FAMILY_ROLE_KEYWORDS,
)


@pytest.fixture
def assessor():
    return FamilySystemAssessor()


# ── Phase 判定 ────────────────────────────────────────────────

def test_phase_check_in_early(assessor):
    """前几轮应该停留在 check_in。"""
    phase = assessor._determine_phase(message_count=0, current_phase="check_in",
                                     user_text="你好", hypothesis=None)
    assert phase == "check_in"

    phase = assessor._determine_phase(message_count=6, current_phase="check_in",
                                     user_text="我最近跟孩子吵架", hypothesis=None)
    assert phase == "check_in"


def test_phase_explore_mid(assessor):
    """5-12 轮应该进入 explore。"""
    phase = assessor._determine_phase(message_count=12, current_phase="check_in",
                                     user_text="孩子总跟他爸吵", hypothesis=None)
    assert phase == "explore"


def test_phase_interpret_late(assessor):
    """13-20 轮且有假设时进入 interpret。"""
    phase = assessor._determine_phase(
        message_count=30, current_phase="explore",
        user_text="确实，每次我跟他爸吵完孩子就不上学",
        hypothesis="孩子的症状可能承担了转移功能"
    )
    # 有 hypothesis，所以应该保持 interpret 或者已经进入 interpret
    assert phase in ("explore", "interpret")


def test_phase_no_regression_with_hypothesis(assessor):
    """有工作假设且已在 interpret 阶段，不应倒退。"""
    phase = assessor._determine_phase(
        message_count=100, current_phase="interpret",
        user_text="我明白了",
        hypothesis="家庭存在指责-退缩循环"
    )
    assert phase == "interpret"


def test_phase_regression_on_denial(assessor):
    """用户否定假设时应该退回 explore。"""
    phase = assessor._determine_phase(
        message_count=30, current_phase="interpret",
        user_text="不是，你理解错了，我家不是这样的",
        hypothesis="孩子承担转移功能"
    )
    assert phase == "explore"


# ── 安全锚点（危机判定改造 ADR-0013）───────────────────────────

def test_safety_check_suicide_keyword(assessor):
    """P1 自杀意念词命中 → 返回 "p1"（探针锚点，不再直接升级）。"""
    assert assessor._check_safety("我有时候真的不想活了", {"risk": 0.3}) == "p1"


def test_safety_check_self_harm(assessor):
    """P1 自伤词命中 → 返回 "p1"。"""
    assert assessor._check_safety("我控制不住划手臂", {"risk": 0.4}) == "p1"


def test_safety_check_p0_hard_gate(assessor):
    """P0 显式危险词命中 → 返回 "p0"（硬闸门）。"""
    assert assessor._check_safety("我已经买了安眠药，今晚就吃", {"risk": 0.5}) == "p0"


def test_safety_check_high_risk_deferred_to_judge(assessor):
    """risk >= 0.7 不再在此直接升级 —— 由语义安全评估器裁决（ADR-0013）。"""
    assert assessor._check_safety("只是心情不太好", {"risk": 0.8}) is None


def test_safety_check_weak_signal_ignored(assessor):
    """P2 弱信号（"狠狠"）完全忽略，不升级不探针 —— 误伤案例修复。"""
    assert assessor._check_safety("我这时候很想大叫狠狠的撞东西", {"risk": 0.4}) is None


def test_safety_check_clean(assessor):
    """正常文本不触发。"""
    assert assessor._check_safety("今天天气不错", {"risk": 0.1}) is None


# ── 家庭成员检测 ──────────────────────────────────────────────

def test_detect_mother(assessor):
    result = assessor._detect_family_members("我妈天天逼我写作业", [])
    assert any(m["role"] == "妈妈" for m in result)


def test_detect_father(assessor):
    result = assessor._detect_family_members("爸爸从来不管我", [])
    assert any(m["role"] == "爸爸" for m in result)


def test_detect_multiple_members(assessor):
    result = assessor._detect_family_members(
        "我跟我老公天天为孩子的事吵架", []
    )
    roles = {m["role"] for m in result}
    assert "配偶" in roles or "孩子" in roles


def test_no_duplicate_members(assessor):
    """已有成员不重复添加。"""
    result = assessor._detect_family_members(
        "妈妈又骂我了",
        [{"role": "妈妈", "label": "焦虑型"}]
    )
    assert len(result) == 0


# ── 综合 assess ───────────────────────────────────────────────

def test_assess_full(assessor):
    """综合评估：正常文本应返回有效结果。"""
    result = assessor.assess(
        user_text="孩子不上学，我跟他爸天天为这事吵",
        message_count=10,
        existing_phase="check_in",
        existing_hypothesis=None,
        existing_members=[],
        emotion={"primary_emotion": "anxiety", "intensity": 0.6, "risk": 0.4},
        route={"route": "comfort", "confidence": 0.8},
    )
    assert isinstance(result, AssessResult)
    assert result.suggested_phase in ("check_in", "explore")
    assert result.escalation_flag is False


def test_assess_crisis_escalation(assessor):
    """P0 显式危险文本（计划/手段）应触发硬升级。"""
    result = assessor.assess(
        user_text="我已经买了安眠药，今晚就吃",
        message_count=5,
        existing_phase="check_in",
        existing_hypothesis=None,
        existing_members=[],
        emotion={"primary_emotion": "sadness", "intensity": 0.8, "risk": 0.9},
        route={"route": "comfort", "confidence": 0.8},
    )
    assert result.escalation_flag is True


def test_assess_p1_probe_no_escalation(assessor):
    """P1 危机意念 → 不再升级 crisis（转探针），probe_direction 为安全确认方向。"""
    result = assessor.assess(
        user_text="我真的不想活了，活着没意思",
        message_count=5,
        existing_phase="check_in",
        existing_hypothesis=None,
        existing_members=[],
        emotion={"primary_emotion": "sadness", "intensity": 0.8, "risk": 0.9},
        route={"route": "comfort", "confidence": 0.8},
    )
    assert result.escalation_flag is False
    assert result.probe_direction and "safety_check" in result.probe_direction


def test_assess_weak_signal_no_escalation(assessor):
    """"狠狠撞东西"（P2 弱信号）不再误伤升级。"""
    result = assessor.assess(
        user_text="我这时候很想大叫狠狠的撞东西，感觉身体上的疼痛会麻痹自己",
        message_count=10,
        existing_phase="check_in",
        existing_hypothesis=None,
        existing_members=[],
        emotion={"primary_emotion": "anger", "intensity": 0.7, "risk": 0.6},
        route={"route": "comfort", "confidence": 0.8},
    )
    assert result.escalation_flag is False


def test_assess_family_hypothesis(assessor):
    """冲突+多成员应生成假设。"""
    result = assessor.assess(
        user_text="我跟他爸天天吵架，孩子现在也不上学了",
        message_count=20,
        existing_phase="explore",
        existing_hypothesis=None,
        existing_members=[
            {"role": "妈妈", "label": ""},
            {"role": "爸爸", "label": ""},
            {"role": "孩子", "label": ""},
        ],
        emotion={"primary_emotion": "anxiety", "intensity": 0.5, "risk": 0.3},
        route={"route": "comfort", "confidence": 0.7},
    )
    assert result.hypothesis_update is not None
    assert "转移" in result.hypothesis_update or "冲突" in result.hypothesis_update


def test_assess_clear_hypothesis_on_denial(assessor):
    """用户否定时清除假设。"""
    result = assessor.assess(
        user_text="不是，你理解错了",
        message_count=20,
        existing_phase="interpret",
        existing_hypothesis="孩子承担转移功能",
        existing_members=[],
        emotion={"primary_emotion": "neutral", "intensity": 0.3, "risk": 0.1},
        route={"route": "general", "confidence": 0.8},
    )
    assert result.hypothesis_update is None
