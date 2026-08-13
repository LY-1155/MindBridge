"""SCID-5 主动式结构化访谈引擎（医生模式，仅抑郁模块起步）

与 scid_tracker.py 的关系：
- scid_tracker 是「被动」的：从用户自由文本里捞关键词，静默更新 session.scid_flags。
- 本引擎是「主动」的：按 SCID-5-CV 抑郁模块的流程（gate 入门 → A3~A9 核心症状 →
  双相鉴别筛查 → 功能损害 → 结论）逐步驱动对话，决定「下一句该问什么」。

设计要点：
1. 纯规则 + 无 LLM 调用 —— 状态机可单测、可复现、零成本。
2. 引擎只产出一个「指令文本」（directive），由 generator 注入 system prompt，
   用周医生的语气自然问出；引擎本身从不直接生成面向用户的句子。
3. 与被动 tracker 协同：某条核心症状已被被动 tracker 在对话历史里匹配到
   （如用户说过「睡不着」→ sleep），进入该步骤时自动确认并跳过，避免重复提问。
4. 安全边界：引擎只是辅助评估，从不输出诊断；结论一律落在「建议专业评估」。
   自杀相关条目（A9）确认后，仍由既有的 safety_judge / risk_flags 链路接管安全。

先只覆盖重性抑郁（MDD）一个模块作原型；后续可扩展其他模块。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── 核心症状 ID → 中文标签 ─────────────────────────────────────────
CRITERION_LABELS = {
    "A1": "情绪低落",
    "A2": "兴趣/愉快感丧失",
    "A3": "食欲/体重变化",
    "A4": "睡眠紊乱",
    "A5": "精神运动迟滞/激越",
    "A6": "疲劳/精力不足",
    "A7": "无价值感/内疚",
    "A8": "注意力集中困难",
    "A9": "死亡/自杀意念",
}

# 流程步骤 → 被动 tracker 的 criteria 名（用于自动确认，避免重复提问）
CRITERION_TO_PASSIVE = {
    "A3": "weight_appetite",
    "A4": "sleep",
    "A5": "psychomotor",
    "A6": "fatigue",
    "A7": "worthlessness_guilt",
    "A8": "concentration",
    "A9": "death_si",
}

# 同一回复最多重问几次，之后按「未确认」跳过该步骤
MAX_REASK = 2


# ── MDD 访谈流程（步骤数据，问题话术为自然口语化改写）────────────────

def _step(
    id: str, title: str, question: str,
    confirm: List[str], deny: List[str],
    criterion: Optional[str] = None,
    probe: str = "",
    next: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "id": id, "title": title, "question": question,
        "confirm_phrases": confirm, "deny_phrases": deny,
        "criterion": criterion, "probe": probe, "next": next,
    }


MDD_STEPS: Dict[str, Dict[str, Any]] = {
    "gate": _step(
        id="gate",
        title="入门筛查（gate）",
        question=(
            "过去这两周里，是不是几乎每天、大部分时间都觉得情绪低落、沮丧或绝望？"
            "或者，以前喜欢做的事——比如打游戏、刷剧、跟朋友出去玩——现在几乎提不起兴趣了？"
        ),
        probe="需要确认两点：是否『几乎每天』，以及是否『持续至少两周』。",
        confirm=[
            "几乎每天", "每天都", "每天", "一直", "总是", "天天", "经常",
            "两个星期", "两周", "半个月", "很久", "两个多月", "两个月",
            "低落", "沮丧", "绝望", "开心不起来", "没意思", "没兴趣",
            "提不起兴趣", "高兴不起来",
        ],
        deny=[
            "没有", "没有过", "不会", "不至于", "还好", "还行",
            "就几天", "就这两天", "这几天", "最近才", "刚开始", "没到那个程度",
        ],
        next="A3_appetite",
    ),
    "A3_appetite": _step(
        id="A3_appetite",
        criterion="A3",
        title="A3 食欲/体重变化",
        question="这段时间的胃口跟以前比有变化吗？比如吃不下、没胃口、体重明显掉了，或者反过来吃得特别多？",
        probe="刻意节食/减肥导致的体重变化不算。",
        confirm=[
            "吃不下", "没胃口", "不想吃", "瘦了", "体重下降", "体重掉了",
            "吃得少", "吃很多", "暴饮暴食", "食欲", "胖了", "体重增加",
        ],
        deny=["没变", "没变化", "没什么变化", "跟以前一样", "正常", "还好", "饭量正常"],
        next="A4_sleep",
    ),
    "A4_sleep": _step(
        id="A4_sleep",
        criterion="A4",
        title="A4 睡眠紊乱",
        question="睡眠怎么样？是睡不着、半夜容易醒、早上醒得太早，还是反而睡得特别多、白天也一直想睡？",
        probe="入睡困难更指向焦虑；早醒（凌晨三四点醒后无法再睡）更指向抑郁。",
        confirm=[
            "睡不着", "入睡困难", "半夜醒", "早醒", "失眠", "睡不好",
            "睡得浅", "睡得少", "嗜睡", "睡太多", "难入睡", "整夜醒",
        ],
        deny=["睡得好", "没变化", "还好", "正常", "睡得挺好", "没影响"],
        next="A5_psychomotor",
    ),
    "A5_psychomotor": _step(
        id="A5_psychomotor",
        criterion="A5",
        title="A5 精神运动迟滞/激越",
        question="最近有没有人跟你说你动作、说话变慢了？或者反过来，坐不住、停不下来？",
        probe="不要直接问『你有没有精神运动迟滞』，用户不会这样描述自己；问的是他人的观察。",
        confirm=[
            "坐不住", "停不下来", "变慢", "迟钝", "反应慢", "动不了",
            "说话变慢", "很慢", "焦躁", "踱步",
        ],
        deny=["没有", "还好", "正常", "不会", "没人说"],
        next="A6_fatigue",
    ),
    "A6_fatigue": _step(
        id="A6_fatigue",
        criterion="A6",
        title="A6 疲劳/精力不足",
        question="最近有没有觉得特别容易累、没什么精力，哪怕没干什么体力活也觉得没劲？",
        probe="是精神上的疲惫，不是干了很多活之后的累。",
        confirm=[
            "累", "疲劳", "没精力", "疲倦", "乏力", "浑身没劲",
            "不想动", "没力气", "很懒", "提不起劲",
        ],
        deny=["不累", "还好", "有精力", "正常", "没有", "精神很好"],
        next="A7_worthlessness",
    ),
    "A7_worthlessness": _step(
        id="A7_worthlessness",
        criterion="A7",
        title="A7 无价值感/内疚",
        question="有没有经常觉得自己很没用、是个失败者、或者拖累了别人？会不会为一些其实不是你的责任的事反复自责？",
        probe="这是区分抑郁和单纯悲伤的关键点：抑郁的自责常超出实际责任范围。",
        confirm=[
            "没用", "废物", "拖累", "对不起", "内疚", "自责",
            "都是我的错", "不值得", "很糟糕", "失败者", "我不好",
        ],
        deny=["没觉得", "不会", "还好", "没有", "不自责"],
        next="A8_concentration",
    ),
    "A8_concentration": _step(
        id="A8_concentration",
        criterion="A8",
        title="A8 注意力集中困难",
        question="最近看东西、听别人说话能集中注意力吗？做决定——比如今天吃什么、穿什么——会不会犹豫很久？",
        probe="抑郁的注意力问题是『脑子转不动』；焦虑是『脑子太满』。",
        confirm=[
            "集中不了", "记不住", "忘事", "走神", "注意力", "想不清楚",
            "犹豫", "很难做决定", "脑子不转", "看不进去",
        ],
        deny=["还好", "正常", "能集中", "没有", "没问题"],
        next="A9_death",
    ),
    "A9_death": _step(
        id="A9_death",
        criterion="A9",
        title="A9 死亡/自杀意念",
        question="有没有觉得活着没意思、不想活下去，或者想过一些关于死亡的念头？",
        probe="如果用户肯定，先评估具体程度（有没有计划/企图），并确保走安全链路。",
        confirm=[
            "不想活", "想死", "自杀", "活不下去", "结束自己", "轻生",
            "活着没意思", "没意义", "解脱", "一了百了",
        ],
        deny=["没有", "没想过", "没有过", "不会", "还好"],
        next="bipolar_screen",
    ),
    "bipolar_screen": _step(
        id="bipolar_screen",
        title="双相鉴别筛查",
        question=(
            "我想再问一个可能听起来有点不相关的问题——过去有没有过一段时间，"
            "不是几个小时，而是连续几天甚至更久，你感觉特别兴奋、精力特别旺盛、"
            "睡得很少也不觉得困、或者说话做事比平时明显快很多？"
        ),
        probe="若有躁狂/轻躁狂发作史 → 需转双相评估，即使当前处于抑郁相也不能按单纯抑郁处理。",
        confirm=[
            "有", "是", "对", "过", "确实", "连续几天", "那段时间",
            "精力特别旺盛", "睡得少", "不觉得困", "话特别多", "花钱特别多",
        ],
        deny=["没有", "没有过", "从来没有", "没", "不会"],
        next="impairment",
    ),
    "impairment": _step(
        id="impairment",
        title="B 功能损害",
        question="这些情况有没有影响到你上班、上学、跟人相处，或者平时该做的事？",
        probe="",
        confirm=[
            "影响", "做不了", "上不了", "没法", "请假", "辞职", "没心思",
            "耽误", "躲着人", "没去",
        ],
        deny=["没影响", "还能", "正常", "还好", "不影响", "扛得住"],
        next="conclusion",
    ),
    "conclusion": _step(
        id="conclusion",
        title="评估结论",
        question="",
        confirm=[], deny=[], next=None,
    ),
}


# ── 通用肯定/否定词（步骤级短语没命中时兜底）────────────────────────
# 注意：『就是』不做通用确认——『就是很累』里的『就是』是程度填充词，
# 不是确认；靠步骤级短语兜（如 A6 的『累』）。
_GENERIC_CONFIRM = ["是", "对", "有", "会", "嗯", "是的", "对啊", "有啊", "确实", "差不多"]
_GENERIC_DENY = [
    "没有", "没有过", "从来没有", "不会", "不是", "还好", "还行", "不至于",
    "没觉得", "不觉得", "没发现", "应该没有", "没有吧",
]


def classify_reply(user_text: str, step: Dict[str, Any]) -> str:
    """把用户对当前问题的回答分类为 confirm / deny / unclear。

    优先级：步骤专属的确认/否认短语 → 通用肯定/否定词 → unclear。
    注意：否认短语先于确认短语检查（如 gate 步骤里『就这两天』优先于『低落』），
    避免「有，但就这两天」被误判为确认。
    """
    text = (user_text or "").strip()
    # 单个字的词（如『有』『没』『是』）只在回答是裸肯定/否定（≤2 字）时才参与匹配，
    # 否则『有时候有点』会命中『有』、『睡得很少』会命中『没』、『就是很累』会命中『是』。
    short_reply = len(text) <= 2

    for p in step.get("deny_phrases", []):
        if p and (len(p) > 1 or short_reply) and p in text:
            return "deny"
    for p in step.get("confirm_phrases", []):
        if p and (len(p) > 1 or short_reply) and p in text:
            return "confirm"
    for p in _GENERIC_DENY:
        if p in text:
            return "deny"
    for p in _GENERIC_CONFIRM:
        if (len(p) > 1 or short_reply) and p in text:
            return "confirm"
    return "unclear"


def should_start_interview(
    user_text: str,
    passive_mdd: List[str],
    existing_state: Optional[Dict[str, Any]],
) -> bool:
    """判断本轮是否要启动（或继续）抑郁访谈。"""
    if existing_state is not None:
        # 已结束的访谈不重启
        return existing_state.get("status") == "active"
    # 兼容多种传参：可能是 list，也可能是 {criteria_met: [...]} 或 {MDD: {criteria_met: [...]}}
    if isinstance(passive_mdd, dict):
        passive_mdd = passive_mdd.get("criteria_met", []) or (
            passive_mdd.get("MDD") or {}
        ).get("criteria_met", [])
    gate_trigger = [
        "心情不好", "心情低落", "情绪低落", "心情差", "低落", "沮丧", "绝望",
        "没意思", "开心不起来", "高兴不起来", "没兴趣", "提不起兴趣", "没劲",
        "什么都不想做", "很难过", "郁闷", "很丧", "丧", "不开心", "不想活",
        "心情很不好", "心情特别不好", "心情一直不好", "心情不太好", "很不好受",
    ]
    for kw in gate_trigger:
        if kw in (user_text or ""):
            return True
    if "depressed_mood" in passive_mdd or "anhedonia" in passive_mdd:
        return True
    # 躯体化表现是 MDD 最常见的东方主诉（头痛查不出毛病+失眠+乏力，情绪不一定说出口）。
    # 被动 tracker 已累积 ≥2 条核心症状就值得启动访谈 —— gate 第一问本来就会确认情绪，
    # 若实为否认则干净跳过，不会误诊。
    if len(passive_mdd) >= 2:
        return True
    return False


def _new_state() -> Dict[str, Any]:
    return {
        "module": "MDD",
        "status": "active",
        "step": "gate",
        "waiting": True,
        "criteria_confirmed": [],
        "criteria_denied": [],
        "reask_count": 0,
        "bipolar_positive": False,
        "conclusion": None,
    }


class SCIDInterviewEngine:
    """按 SCID-5 抑郁模块流程推进对话的状态机。

    用法（每轮一次）：
        engine = SCIDInterviewEngine()
        new_state, directive = engine.step_turn(state, user_text, passive_mdd)
    其中 passive_mdd 为被动 tracker 累积的 MDD criteria_met 列表。
    """

    def step_turn(
        self,
        state: Optional[Dict[str, Any]],
        user_text: str,
        passive_mdd: Optional[List[str]] = None,
    ) -> tuple[Optional[Dict[str, Any]], str]:
        passive_mdd = passive_mdd or []

        # 1) 未开始
        if state is None:
            if not should_start_interview(user_text, passive_mdd, None):
                return None, ""
            state = _new_state()
            # 启动轮不把触发文本当答案，直接先问 gate
            return state, self._build_directive(state)

        # 2) 已结束
        if state.get("status") != "active":
            return state, ""

        step = MDD_STEPS[state["step"]]

        # 3) 等待回答中 → 分类并推进
        if state.get("waiting"):
            verdict = classify_reply(user_text, step)
            if verdict == "unclear":
                state["reask_count"] = state.get("reask_count", 0) + 1
                if state["reask_count"] >= MAX_REASK:
                    state = self._mark_step(state, step, "unconfirmed")
                    if state.get("status") == "active":
                        state = self._advance(state, passive_mdd)
                else:
                    # 留在原步骤，指令 = 温和再问一次
                    return state, self._build_reask_directive(state, step)
            else:
                state = self._mark_step(state, step, verdict)
                if state.get("status") == "active":
                    state = self._advance(state, passive_mdd)

        return state, self._build_directive(state)

    # ── 内部 ──────────────────────────────────────────────────

    def _mark_step(
        self, state: Dict[str, Any], step: Dict[str, Any], verdict: str
    ) -> Dict[str, Any]:
        """把当前步骤的判定写入状态。"""
        criterion = step.get("criterion")
        if criterion:
            if verdict == "confirm" and criterion not in state["criteria_confirmed"]:
                state["criteria_confirmed"].append(criterion)
            elif verdict in ("deny", "unconfirmed") and criterion not in state["criteria_denied"]:
                state["criteria_denied"].append(criterion)

        if step["id"] == "gate":
            if verdict == "confirm":
                # gate 确认 ⇒ 核心 A1（情绪低落）+ A2（兴趣丧失）视为满足
                for c in ("A1", "A2"):
                    if c not in state["criteria_confirmed"]:
                        state["criteria_confirmed"].append(c)
            else:
                # gate 未确认 ⇒ 跳过抑郁模块
                state["status"] = "done"
                state["step"] = "conclusion"
                state["conclusion"] = "skip_gate"

        if step["id"] == "bipolar_screen" and verdict == "confirm":
            state["bipolar_positive"] = True

        return state

    def _advance(
        self, state: Dict[str, Any], passive_mdd: List[str]
    ) -> Dict[str, Any]:
        """从当前步骤走到下一个待评估的步骤。

        规则：已被确认过的条目跳过；被动 tracker 已在历史里匹配到的条目自动确认并跳过。
        """
        next_id = MDD_STEPS[state["step"]].get("next")
        while True:
            # 终态步骤（conclusion 是唯一 next=None 的步骤）→ 直接结束访谈
            if next_id is None or next_id == "conclusion":
                state["status"] = "done"
                state["step"] = "conclusion"
                return state
            step = MDD_STEPS[next_id]
            crit = step.get("criterion")
            if crit and crit in state["criteria_confirmed"]:
                next_id = step.get("next")
                continue
            # 被动 tracker 已在对话历史匹配到的判据 → 自动确认并跳过
            passive_name = CRITERION_TO_PASSIVE.get(crit) if crit else None
            if passive_name and passive_name in passive_mdd:
                if crit not in state["criteria_confirmed"]:
                    state["criteria_confirmed"].append(crit)
                next_id = step.get("next")
                continue
            state["step"] = next_id
            state["waiting"] = True
            state["reask_count"] = 0
            return state

    # ── 指令生成 ──────────────────────────────────────────────

    def _progress_lines(self, state: Dict[str, Any]) -> List[str]:
        lines = []
        if state["criteria_confirmed"]:
            labels = [CRITERION_LABELS[c] for c in state["criteria_confirmed"] if c in CRITERION_LABELS]
            lines.append("· 已确认：" + "、".join(labels))
        if state["criteria_denied"]:
            labels = [CRITERION_LABELS[c] for c in state["criteria_denied"] if c in CRITERION_LABELS]
            lines.append("· 已排除：" + "、".join(labels))
        return lines

    def _build_directive(self, state: Dict[str, Any]) -> str:
        if state.get("status") != "active":
            return self._build_conclusion_directive(state)
        step = MDD_STEPS[state["step"]]
        lines = [
            "· 模块：重性抑郁（SCID-5-CV 抑郁模块，结构化评估进行中）",
        ]
        lines.extend(self._progress_lines(state))
        lines.append(f"· 本步评估：{step['title']}")
        lines.append(f"· 请用你自己的话、自然地问出：「{step['question']}」")
        if step.get("probe"):
            lines.append(f"· 追问要点：{step['probe']}")
        return "\n".join(lines)

    def _build_reask_directive(self, state: Dict[str, Any], step: Dict[str, Any]) -> str:
        return (
            f"· 模块：重性抑郁（结构化评估进行中）。用户刚才对「{step['title']}」的回答不太明确，"
            "请用更温和、更口语的方式再确认一次，不要重复你上一轮的原话："
            f"「{step['question']}」"
        )

    def _build_conclusion_directive(self, state: Dict[str, Any]) -> str:
        if state.get("conclusion") == "skip_gate":
            return (
                "· 重性抑郁筛查的入门问题未得到确认（情绪低落/兴趣丧失未持续『几乎每天』且『至少两周』）。"
                "不要强行追问。做法：继续共情陪伴；留意是否存在更长期的慢性低落（或为持续性抑郁）"
                "或与生活事件直接相关的适应反应。"
            )
        if state.get("bipolar_positive"):
            return (
                "· 用户报告了可能的躁狂/轻躁狂发作史。这是双相谱系的强烈信号——即使当前处于抑郁相，"
                "也不能按单纯抑郁处理。做法：用共情承接当下的痛苦，温和建议寻求精神科专业评估"
                "（双相需要专门的评估与用药方案，绝不要自行给出处理建议）。"
            )
        confirmed = state["criteria_confirmed"]
        n = len(confirmed)
        safety_note = ""
        if "A9" in confirmed:
            safety_note = " 已确认存在死亡/自杀意念——继续关注安全，必要时走危机链路。"
        if n >= 5:
            return (
                f"· 抑郁评估：已确认 {n} 条核心症状（含 A1 情绪低落 / A2 兴趣丧失）。"
                f"已达到结构化筛查的阳性阈值。{safety_note}"
                "⚠️ 绝对不要直接对用户下诊断。做法：温和共情地总结你听到的困扰，"
                "并建议寻求专业心理咨询或精神科评估（自然地说，如『这些情况已经持续一段时间、"
                "也影响到了生活，找专业的人聊一聊会更有帮助』），随后转回陪伴式对话。"
            )
        if n >= 3:
            return (
                f"· 抑郁评估：已确认 {n} 条核心症状，尚未达到阳性阈值（需≥5条）。{safety_note}"
                "做法：继续共情陪伴，留意症状是否持续或加重，不强行推进评估。"
            )
        return (
            f"· 抑郁评估：目前确认的症状较少（{n} 条），暂不符合抑郁筛查阳性。"
            "做法：继续共情陪伴，不强行推进评估。"
        )
