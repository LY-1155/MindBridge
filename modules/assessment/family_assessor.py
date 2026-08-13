"""FamilySystemAssessor：家庭系统评估器

读取 session state + 当前用户输入 → 产出对话引导建议。
纯规则驱动，不依赖 LLM。由 InterventionService 在生成回复前调用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# ── 家庭成员关键词 ─────────────────────────────────────────────
FAMILY_ROLE_KEYWORDS: Dict[str, List[str]] = {
    "妈妈": ["我妈", "妈妈", "母亲", "她妈", "婆婆", "丈母娘"],
    "爸爸": ["我爸", "爸爸", "父亲", "她爸", "公公", "老丈人"],
    "孩子": ["孩子", "女儿", "儿子", "小孩", "娃娃", "宝宝", "我家那个"],
    "配偶": ["老公", "老婆", "丈夫", "妻子", "爱人", "对象", "我先生", "我太太"],
    "兄弟姐妹": ["哥哥", "姐姐", "弟弟", "妹妹", "大哥", "大姐", "老二", "老三"],
    "祖辈": ["爷爷", "奶奶", "外公", "外婆", "姥姥", "姥爷"],
}

# ── 关系冲突关键词 ─────────────────────────────────────────────
CONFLICT_KEYWORDS = [
    "吵架", "打架", "冷战", "不说话", "闹翻", "吵", "打", "不理",
    "冲突", "矛盾", "吵得", "闹得", "不愉快", "争执", "争吵",
]

# ── 安全锚点（危机判定改造 ADR-0013）───────────────────────────
# 风险词命中 ≠ 危机触发。分级词表统一来自 modules.assessment.risk_anchors：
#   P0 显式危险 → 硬闸门升级 crisis；P1 意念/冲动 → 转探针；P2 弱信号（"狠狠"等）→ 忽略。
from modules.assessment.risk_anchors import RISK_ANCHOR_P0, RISK_ANCHOR_P1

# 兼容别名：旧扁平红线表 = P0 + P1 合并（P2 弱信号已剔除，"狠狠"不再命中）。
# 新代码请用 risk_anchors 的 P0/P1/P2 分级表。
SAFETY_RED_LINE: Dict[str, List[str]] = {
    k: list(dict.fromkeys(RISK_ANCHOR_P0.get(k, []) + RISK_ANCHOR_P1.get(k, [])))
    for k in set(RISK_ANCHOR_P0) | set(RISK_ANCHOR_P1)
}

# P0 硬闸门 / P1 探针的规则兜底探针方向（LLM 生成探针失败时的三选一框架）
_P0_PROBE_DIRECTION = (
    "safety_check: 使用三选一框架直接问："
    "'是真的是不想活，还是吓唬他们，还是都有？'"
)
_P1_PROBE_DIRECTION = (
    "safety_check: 用户有潜在风险信号，用自然的方式温和确认："
    "'是真心不想活，还是心里难受，还是都有？'"
)

# ── Phase 判定阈值 ─────────────────────────────────────────────
PHASE_TURN_THRESHOLDS = {
    "check_in": 4,    # 1-4 轮：建立信任、了解基本情况
    "explore": 12,    # 5-12 轮：深入探索家庭互动模式
    "interpret": 20,  # 13-20 轮：形成假设、解释模式
    "intervene": 999, # 21+ 轮：干预/巩固
}


@dataclass
class AssessResult:
    """评估结果 — 注入 prompt 的上下文"""
    suggested_phase: str = "check_in"
    probe_direction: Optional[str] = None
    # 示例："问问妈妈怎么看爸爸的管教方式"
    hypothesis_update: Optional[str] = None
    # 新假设文本；None 表示无需更新
    escalation_flag: bool = False
    # True 表示需要触发危机路由
    new_family_members: List[Dict[str, str]] = field(default_factory=list)


class FamilySystemAssessor:
    """家庭系统评估器 — 规则驱动"""

    def __init__(self):
        self._known_members: Dict[str, List[str]] = FAMILY_ROLE_KEYWORDS

    # ── Public API ──────────────────────────────────────────────

    def assess(
        self,
        user_text: str,
        message_count: int,
        existing_phase: str,
        existing_hypothesis: Optional[str],
        existing_members: List[Dict[str, Any]],
        emotion: Dict[str, Any],
        route: Dict[str, Any],
    ) -> AssessResult:
        """主入口：综合评估并返回引导建议。

        Args:
            user_text: 用户当前输入
            message_count: 当前会话消息数（从 0 开始）
            existing_phase: 当前阶段
            existing_hypothesis: 当前工作假设
            existing_members: 已识别的家庭成员
            emotion: EmotionTags dict
            route: RouteDecision dict
        """
        result = AssessResult()

        # 1. 安全锚点 — 最高优先级（P0 硬闸门；P1 转探针，不再直接升级）
        safety_tier = self._check_safety(user_text, emotion)
        if safety_tier == "p0":
            result.escalation_flag = True
            result.probe_direction = _P0_PROBE_DIRECTION
            return result

        # 2. 家庭成员识别
        new_members = self._detect_family_members(user_text, existing_members)
        result.new_family_members = new_members

        # 3. Phase 判定
        suggested = self._determine_phase(
            message_count, existing_phase,
            user_text, existing_hypothesis
        )
        result.suggested_phase = suggested

        # 4. 探针方向生成
        result.probe_direction = self._generate_probe(
            suggested, existing_members, existing_hypothesis,
            user_text, existing_phase
        )

        # 5. 工作假设更新
        result.hypothesis_update = self._update_hypothesis(
            suggested, user_text, existing_hypothesis,
            existing_members
        )

        # 6. P1 安全探针覆盖家庭探针（本轮以安全确认为优先）
        if safety_tier == "p1":
            result.probe_direction = _P1_PROBE_DIRECTION

        return result

    # ── Internal: 安全锚点 ─────────────────────────────────────

    def _check_safety(
        self, user_text: str, emotion: Dict[str, Any]
    ) -> Optional[str]:
        """检查安全信号，返回锚点级别：p0（硬闸门）/ p1（探针）/ None。

        危机判定改造（ADR-0013）：emotion.risk 高值不再在此直接升级——
        语义安全评估器已在图上读取情绪风险 + 会话上下文做最终裁决。
        这里只做词表级别的 P0 硬闸门与 P1 探针标记。
        """
        from modules.assessment.risk_anchors import match_anchor

        anchors = match_anchor(user_text)
        if anchors["p0"]:
            return "p0"
        if anchors["p1"]:
            return "p1"
        return None

    # ── Internal: 家庭成员识别 ──────────────────────────────────

    def _detect_family_members(
        self, user_text: str, existing: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """从用户文本中识别新的家庭成员。"""
        existing_roles = {m.get("role", "") for m in existing}
        new: List[Dict[str, str]] = []
        for role, keywords in self._known_members.items():
            if role in existing_roles:
                continue
            for kw in keywords:
                if kw in user_text:
                    new.append({"role": role, "label": ""})
                    break
        return new

    # ── Internal: Phase 判定 ────────────────────────────────────

    def _determine_phase(
        self,
        message_count: int,
        current_phase: str,
        user_text: str,
        hypothesis: Optional[str],
    ) -> str:
        """基于消息轮数 + 工作假设状态判定阶段。"""
        turns = max(1, message_count // 2)

        # 如果用户明确否定假设 → 退回 explore（必须在"不倒退"检查之前）
        denial_kw = ["不是", "你理解错了", "不对", "不是这样的", "你错了"]
        if hypothesis and current_phase == "interpret":
            if any(kw in user_text for kw in denial_kw):
                return "explore"

        # 如果有假设且进入了解释阶段, 不倒退
        if hypothesis and current_phase in ("interpret", "intervene"):
            return current_phase

        # 按轮数推进
        if turns <= PHASE_TURN_THRESHOLDS["check_in"]:
            return "check_in"
        elif turns <= PHASE_TURN_THRESHOLDS["explore"]:
            return "explore"
        elif turns <= PHASE_TURN_THRESHOLDS["interpret"]:
            return "interpret"
        else:
            return "intervene"

    # ── Internal: 探针方向 ──────────────────────────────────────

    def _generate_probe(
        self,
        phase: str,
        members: List[Dict[str, Any]],
        hypothesis: Optional[str],
        user_text: str,
        old_phase: str,
    ) -> Optional[str]:
        """生成本轮探针方向。"""

        # phase 刚推进 → 给出过渡引导
        if phase != old_phase and phase == "explore":
            return ("phase_transition: 从建立信任过渡到家庭探索。"
                    "如果对话中有多位家庭成员，开始用循环提问：'爸爸觉得妈妈的处理方式怎么样？'")

        if phase == "check_in":
            return "rapport_building: 建立信任，了解谁最先发现问题、谁最着急。不要急于探测症状。"

        if phase == "explore":
            return self._explore_probe(members, user_text)

        if phase == "interpret":
            return ("formulate_hypothesis: 基于已收集的家庭互动证据，"
                    "用'听起来你们家似乎……'的句式做一个试探性重构。"
                    "不提诊断标签，用日常语言描述模式。")

        if phase == "intervene":
            return ("consolidate: 强化已识别的模式，布置家庭作业或行为实验。"
                    "如'这周你们试试，下次孩子发脾气的时候，爸爸先不说话，"
                    "让妈妈一个人处理——看看有什么不同。'")

        return None

    def _explore_probe(
        self, members: List[Dict[str, Any]], user_text: str
    ) -> Optional[str]:
        """探索阶段的探针方向选择。"""
        roles = {m.get("role", "") for m in members}

        # 有冲突信号 → 循环提问
        if any(kw in user_text for kw in CONFLICT_KEYWORDS):
            return ("circular_questioning: 检测到家庭冲突信号。"
                    "用循环提问探关系：'所以爸爸觉得是妈妈太焦虑了——"
                    "孩子怎么看？他同意爸爸的判断吗？'")

        # 有家庭成员但没探过关系 → 探关系
        if len(roles) >= 2:
            return ("relationship_inquiry: 已识别多位家庭成员。"
                    "探索互动模式：'你们三个在一起的时候，"
                    "谁最容易先着急？谁通常是最先让步的那个？'")

        # 只有一个人 → 询问家庭背景
        return ("family_background: 逐步了家庭结构。"
                "'家里还有谁跟你一样着急这件事吗？'")

    # ── Internal: 工作假设 ──────────────────────────────────────

    def _update_hypothesis(
        self,
        phase: str,
        user_text: str,
        existing: Optional[str],
        members: List[Dict[str, Any]],
    ) -> Optional[str]:
        """更新工作假设。仅在 interpret 阶段或有强证据时更新。"""

        # 用户明确否定 → 清除假设
        denial_kw = ["不是", "你理解错了", "不对", "不是这样的"]
        if existing and any(kw in user_text for kw in denial_kw):
            return None

        # 如果有冲突+孩子症状 → 可能形成家庭功能假设
        if phase in ("explore", "interpret") and not existing:
            if any(kw in user_text for kw in CONFLICT_KEYWORDS):
                roles = {m.get("role", "") for m in members}
                if "孩子" in roles and len(roles) >= 2:
                    return "孩子的症状可能承担了转移家庭冲突的功能——当孩子的「问题」成为焦点，父母就不用面对彼此之间的矛盾。"

        # 如果有冲突+没有孩子被提及 → 可能是夫妻关系假设
        if phase in ("explore", "interpret") and not existing:
            conflict_count = sum(
                1 for kw in CONFLICT_KEYWORDS if kw in user_text
            )
            if conflict_count >= 2:
                return "夫妻之间的沟通模式可能存在「指责-退缩」循环，一方的焦虑表达被另一方体验为攻击。"

        # 保持现有假设
        return existing
