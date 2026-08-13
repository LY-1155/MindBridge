"""主动式 SCID 访谈引擎单元测试"""

import pytest

from modules.assessment.scid_interview import (
    SCIDInterviewEngine,
    classify_reply,
    should_start_interview,
    MDD_STEPS,
)
from core.memory.session_memory import SessionMetadata


@pytest.fixture
def engine():
    return SCIDInterviewEngine()


# ── 启动条件 ────────────────────────────────────────────────────

def test_no_trigger_no_start(engine):
    """无抑郁触发词 + 无被动证据 → 不启动访谈。"""
    state, directive = engine.step_turn(None, "今天天气真不错", [])
    assert state is None
    assert directive == ""


def test_trigger_starts_gate(engine):
    """『心情不好』触发启动，启动轮先问 gate，不把触发文本当答案。"""
    state, directive = engine.step_turn(None, "我最近心情不好，睡不着", ["sleep"])
    assert state is not None
    assert state["status"] == "active"
    assert state["step"] == "gate"
    assert state["waiting"] is True
    assert "入门筛查" in directive
    assert "过去这两周" in directive


def test_passive_core_criterion_starts(engine):
    """被动 tracker 已有 depressed_mood → 也启动。"""
    state, _ = engine.step_turn(None, "随便聊聊", ["depressed_mood"])
    assert state is not None


def test_somatic_presentation_starts_on_two_passive(engine):
    """躯体化主诉（失眠+乏力、没说情绪）→ 被动≥2 条也启动，gate 先确认情绪。"""
    text = "我最近一直头痛，但去医院查了没问题。然后我失眠很严重，或者睡不醒，整天没力气。"
    state, directive = engine.step_turn(None, text, ["sleep", "fatigue"])
    assert state is not None
    assert state["step"] == "gate"
    assert "入门筛查" in directive
    # 单条被动（如只有失眠）不启动，避免过度触发
    state2, _ = engine.step_turn(None, "我有点失眠", ["sleep"])
    assert state2 is None


def test_should_start_tolerates_tracker_dict_shape():
    """兼容被动 tracker 的 {criteria_met: [...]} 字典结构。"""
    from modules.assessment.scid_interview import should_start_interview

    as_dict = {"MDD": {"criteria_met": ["sleep", "fatigue"], "count": 2}}
    assert should_start_interview("随便聊聊", as_dict, None) is True


def test_intensifier_variant_triggers(engine):
    """『心情很不好』这类加程度副词的变体也要能触发。"""
    state, _ = engine.step_turn(None, "我最近心情很不好，什么都不想做", [])
    assert state is not None
    assert state["step"] == "gate"


def test_done_state_not_restarted(engine):
    """访谈结束后不再重启。"""
    state, _ = engine.step_turn(None, "我最近心情不好", [])
    state["status"] = "done"
    state, directive = engine.step_turn(state, "我最近心情还是不好", ["depressed_mood"])
    assert directive == ""


# ── gate 分支 ───────────────────────────────────────────────────

def test_gate_confirm_advances_to_a3(engine):
    """gate 确认 → 记录 A1/A2，推进到 A3（食欲）。"""
    state, _ = engine.step_turn(None, "我最近心情不好", [])
    state, directive = engine.step_turn(state, "对，几乎每天都这样，一个多月了", [])
    assert state["step"] == "A3_appetite"
    assert "A1" in state["criteria_confirmed"]
    assert "A2" in state["criteria_confirmed"]
    assert "A3 食欲/体重变化" in directive


def test_gate_short_duration_is_deny(engine):
    """『有，但就这两天』→ 视为否认（时长不足 2 周）。"""
    state, _ = engine.step_turn(None, "我最近心情不好", [])
    state, directive = engine.step_turn(state, "有，但就这两天", [])
    assert state["status"] == "done"
    assert "入门问题未得到确认" in directive


def test_gate_deny_skips_module(engine):
    """gate 直接否认 → 跳过抑郁模块，结论为不确认。"""
    state, _ = engine.step_turn(None, "我最近心情不好", [])
    state, directive = engine.step_turn(state, "没有，就是偶尔心情不好", [])
    assert state["status"] == "done"
    assert "skip_gate" == state["conclusion"]


# ── 症状条目推进 ────────────────────────────────────────────────

def test_criterion_confirm_advances(engine):
    """A3 确认（吃不下）→ 推进到 A4（睡眠）。"""
    state, _ = engine.step_turn(None, "我最近心情不好", [])
    state, _ = engine.step_turn(state, "对，每天都很低落", [])
    state, directive = engine.step_turn(state, "胃口很差，吃不下", [])
    assert "A3" in state["criteria_confirmed"]
    assert state["step"] == "A4_sleep"
    assert "A4 睡眠紊乱" in directive


def test_criterion_deny_advances(engine):
    """A4 否认（睡得好）→ 推进到 A5。"""
    state, _ = engine.step_turn(None, "我最近心情不好", [])
    state, _ = engine.step_turn(state, "对，每天都很低落", [])
    state, _ = engine.step_turn(state, "吃不下", [])
    state, directive = engine.step_turn(state, "睡眠挺好的，没变化", [])
    assert "A4" in state["criteria_denied"]
    assert state["step"] == "A5_psychomotor"


def test_passive_autoconfirm_skips(engine):
    """被动 tracker 已匹配的条目（体重/睡眠）自动确认并跳过，不重复提问。"""
    passive = ["weight_appetite", "sleep"]
    state, _ = engine.step_turn(None, "我最近心情不好", passive)
    state, directive = engine.step_turn(state, "对，每天都很低落", passive)
    assert state["step"] == "A5_psychomotor"
    assert "A3" in state["criteria_confirmed"]
    assert "A4" in state["criteria_confirmed"]
    assert "A5 精神运动迟滞/激越" in directive


def test_unclear_reask_then_skip(engine):
    """含糊回答先温和再问，超过上限后按未确认处理。"""
    state, _ = engine.step_turn(None, "我最近心情不好", [])
    state, directive = engine.step_turn(state, "我也不太确定，好像有时候有点", [])
    assert state["reask_count"] == 1
    assert "再确认一次" in directive
    state, directive = engine.step_turn(state, "可能吧，我也说不清", [])
    assert state["status"] == "done"
    assert "入门问题未得到确认" in directive


# ── 结论 ────────────────────────────────────────────────────────

def _drive_full_mdd(engine, bipolar_answer="没有"):
    """走完整流程到结论：gate 确认 → A3..A9 全部确认 → 双相筛查 → 功能损害。"""
    confirms = {
        "A3_appetite": "吃不下",
        "A4_sleep": "睡不着",
        "A5_psychomotor": "很迟钝，动不了",
        "A6_fatigue": "很累",
        "A7_worthlessness": "觉得自己很没用",
        "A8_concentration": "集中不了",
        "A9_death": "有时候觉得活着没意思",
    }
    state, _ = engine.step_turn(None, "我最近心情不好", [])
    state, _ = engine.step_turn(state, "对，几乎每天都这样，一个多月了", [])
    for _ in range(20):  # 安全上限，防死循环
        step_id = state.get("step")
        if state.get("status") != "active":
            break
        if step_id in confirms:
            state, _ = engine.step_turn(state, confirms[step_id], [])
        elif step_id == "bipolar_screen":
            state, _ = engine.step_turn(state, bipolar_answer, [])
        elif step_id == "impairment":
            state, _ = engine.step_turn(state, "有影响，上班都没心思", [])
        else:
            raise AssertionError(f"unexpected step {step_id}")
    return state


def test_full_mdd_positive_conclusion(engine):
    """9 条全确认 → 结论为阳性阈值，且不下诊断。"""
    state = _drive_full_mdd(engine)
    assert state["status"] == "done"
    assert len(state["criteria_confirmed"]) == 9
    directive = engine._build_directive(state)  # noqa: SLF001
    assert "已达到结构化筛查的阳性阈值" in directive
    assert "不要直接对用户下诊断" in directive


def test_bipolar_positive_overrides_conclusion(engine):
    """双相筛查阳性 → 结论优先提示双相谱系。"""
    state = _drive_full_mdd(engine, bipolar_answer="有过，连续几天特别兴奋，睡得很少")
    directive = engine._build_directive(state)  # noqa: SLF001
    assert "双相谱系" in directive


# ── 分类器 ──────────────────────────────────────────────────────

def test_classify_negation_order():
    """否认短语优先于确认短语：『有，但就这两天』→ deny。"""
    step = MDD_STEPS["gate"]
    assert classify_reply("有，但就这两天", step) == "deny"
    assert classify_reply("几乎每天都这样", step) == "confirm"


def test_classify_criterion_specific():
    """步骤专属短语优先于通用否定：『没兴趣』→ confirm（anhedonia 类）。"""
    step = MDD_STEPS["A4_sleep"]
    assert classify_reply("没有，我睡得好", step) == "deny"
    step2 = MDD_STEPS["A9_death"]
    assert classify_reply("偶尔会想死", step2) == "confirm"


def test_classify_intensifier_jiushi_not_confirm():
    """『就是很累』是程度填充，不构成确认；『没觉得』是否认。"""
    step = MDD_STEPS["A5_psychomotor"]
    # 没有『没觉得』时，『就是很累』不应命中任何确认 → unclear/deny
    assert classify_reply("倒没觉得，就是很累", step) == "deny"
    # 单独『就是很累』（无否认词）→ 不该被判 confirm
    assert classify_reply("就是很累", MDD_STEPS["A5_psychomotor"]) != "confirm"


# ── generator 集成（纯格式化，不触达 LLM/向量库）─────────────────

def test_format_scid_directive_empty_and_full():
    """指令为空 → 不改变 prompt；非空 → 包成指令块并带安全约束。"""
    from modules.intervention.generator import InterventionReplyGenerator

    assert InterventionReplyGenerator._format_scid_directive("") == ""
    out = InterventionReplyGenerator._format_scid_directive("· 模块：重性抑郁")
    assert "## SCID-5 结构化访谈指令" in out
    assert "· 模块：重性抑郁" in out
    assert "绝不暴露『我在按手册访谈』" in out
    assert "一次只问一个问题" in out


# ── session 元数据 ──────────────────────────────────────────────

def test_session_metadata_field_exists():
    """SessionMetadata 提供 scid_interview_state 字段。"""
    meta = SessionMetadata(session_id="t1")
    assert meta.scid_interview_state is None
    meta.scid_interview_state = {"module": "MDD", "status": "active"}
    assert meta.scid_interview_state["status"] == "active"
