"""风险锚点词表 + 会话级危机状态机

危机判定改造的核心数据层（统一评估层词表来源）。

核心原则：**风险词命中 ≠ 危机触发**。
- P0（显式危险）→ 硬闸门，直接 crisis，不发 LLM
- P1（危机意念/冲动）→ 触发语义安全评估器 → probe
- P2（弱信号）→ 不单独触发评估器（如"狠狠""撞东西"，纯子串误伤源）

`transition_safety_state` 是纯函数，供 InterventionService 做会话级状态迁移，可单测。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# ── P0：显式危险 / 计划 / 手段 / 即刻行动 ────────────────────
# 命中 → 无条件 crisis（规则硬闸门，评估器不得降级，ADR-0013 约束 2）。
# 刻意保守：只收录真正无歧义的"计划/手段/正在发生"表达。
RISK_ANCHOR_P0: Dict[str, List[str]] = {
    "suicide": [
        "已经买了安眠药", "买了安眠药", "安眠药已经买",
        "今晚就跳", "马上跳楼", "现在要跳楼", "正在楼顶", "现在在楼顶",
        "现在就死", "现在就要死", "马上就要死", "活不过今晚", "等不到明天",
        "现在就去死", "现在就要去死", "马上就去死", "今天就去死",
        "这就去死", "这就要去死", "我现在就去死", "我现在就要去死",
        "我马上就去死", "我马上就死",
        "我要自杀", "我想自杀", "我决定自杀", "我准备自杀", "我打算自杀",
        "遗书写好了", "遗书已经写好", "遗书已经写了",
        "刀已经准备好", "药已经拿到", "绳子已经准备好", "东西都准备好了",
        "已经想好怎么死", "已经决定结束自己", "决定结束自己", "准备结束自己",
    ],
    "self_harm": [
        "正在割腕", "割腕了", "已经在割", "正在用刀割",
        "正在自残", "马上要自残", "已经在自残", "正在划手",
    ],
    "violence": [
        "我现在要杀了他", "马上要杀", "已经在动手打人", "正在打他",
        "现在就动手打", "我要弄死他",
    ],
}

# ── P1：危机意念 / 冲动（模糊，需探针确认"念头 vs 计划"）──────
# 命中 → 触发评估器；LLM 裁决 probe → 医生注入安全探针。
RISK_ANCHOR_P1: Dict[str, List[str]] = {
    "suicide": [
        "不想活", "想死", "死了算了", "自杀", "结束自己",
        "活着没意思", "活不下去了", "解脱", "一了百了", "想不开",
        "轻生", "跳楼", "割腕", "安眠药", "遗书", "不想活了",
        "楼顶", "寻死", "想了结", "了结自己",
    ],
    "self_harm": [
        "自残", "划手", "割手", "伤害自己", "自伤", "扣自己",
        "掐自己", "拧自己", "撞墙", "用刀", "伤害身体", "打自己",
        "想划", "想割", "想自残",
    ],
    "violence": [
        "打人", "踹", "动手", "家暴", "暴力", "打他", "打她",
        "揍", "往死里打", "想打人", "想动手", "要打他", "要打人",
    ],
}

# ── P2：弱信号（口语夸张/情绪宣泄，不单独触发评估器）──────────
RISK_ANCHOR_P2: Dict[str, List[str]] = {
    "violence": ["狠狠", "撞东西", "用力砸", "摔东西", "砸东西"],
    "general": ["活得好累", "累死了", "气死了", "烦死了", "没意思"],
}

# ── 探针后的确认词（在 PROBING 状态，用户确认了计划/手段）─────
CONFIRM_KEYWORDS: List[str] = [
    "已经买", "买好了", "准备好了", "都准备好了", "已经准备",
    "就今晚", "就这两天", "已经想好", "想好了", "有刀", "有药",
    "割过", "跳过", "吃过", "从楼上", "等不了了", "不想再忍了",
]

# ── 探针后的否认词（用户否认危机念头）─────────────────────────
DENY_KEYWORDS: List[str] = [
    "不是", "没有", "不会", "不想", "只是说说", "发泄", "开玩笑",
    "没事", "还好", "没想过", "不会做", "吓唬", "不会的", "别担心",
]

# "不想" 与意念词"不想活"撞车：状态机把"不想活了"误判成否认，导致
# 最常见的意念表达永远不累积（安全网失效）。"不想"后跟"活"是"想结束生命"，
# 属意念而非否认；只有"不想"后跟其他（"不想那样做""不想死"）才算否认。
_DENY_NOT_IDEATION = re.compile(r"不想(?:再)?活")


def _is_denial(user_text: str) -> bool:
    for kw in DENY_KEYWORDS:
        if kw == "不想" and _DENY_NOT_IDEATION.search(user_text):
            continue  # "不想活了/不想再活了" → 意念，不是否认
        if kw in user_text:
            return True
    return False


def match_anchor(text: str) -> Dict[str, Dict[str, List[str]]]:
    """匹配文本命中 P0/P1/P2 风险锚点。

    Returns:
        {"p0": {risk_type: [matched_terms]}, "p1": {...}, "p2": {...}}
    """
    result: Dict[str, Dict[str, List[str]]] = {"p0": {}, "p1": {}, "p2": {}}
    tiers = {"p0": RISK_ANCHOR_P0, "p1": RISK_ANCHOR_P1, "p2": RISK_ANCHOR_P2}
    for tier_key, table in tiers.items():
        for risk_type, terms in table.items():
            matched = [kw for kw in terms if kw in text]
            if matched:
                result[tier_key][risk_type] = matched
    return result


def anchor_strength(anchors: Dict[str, Dict[str, List[str]]]) -> Optional[str]:
    """返回最高命中层级：p0 / p1 / p2 / None。"""
    for tier in ("p0", "p1", "p2"):
        if anchors[tier]:
            return tier
    return None


def flatten_anchors(anchors: Dict[str, Dict[str, List[str]]]) -> List[str]:
    """把锚点展平为审计列表，如 ["p1:suicide:不想活"]。"""
    out: List[str] = []
    for tier, risk_map in anchors.items():
        for risk_type, terms in risk_map.items():
            for t in terms:
                out.append(f"{tier}:{risk_type}:{t}")
    return out


def first_risk_type(anchors: Dict[str, Dict[str, List[str]]]) -> Optional[str]:
    """取最高层级的第一个风险类型（suicide/self_harm/violence）。"""
    for tier in ("p0", "p1", "p2"):
        if anchors[tier]:
            return next(iter(anchors[tier]))
    return None


def transition_safety_state(
    state: Optional[Dict[str, Any]],
    verdict: Optional[str],
    user_text: str,
    *,
    max_probe_count: int = 3,
    anchored_probe: bool = True,
) -> Dict[str, Any]:
    """会话级危机状态机迁移（纯函数，无副作用）。

    Args:
        state: 当前 safety_state dict（None 视为 NONE）
        verdict: 本轮评估器裁决：crisis / probe / no_risk / None（跳过）
        user_text: 本轮用户输入（用于确认/否认检测）
        max_probe_count: probe 累积升级阈值
        anchored_probe: probe 裁决是否由风险锚点驱动（matched_anchors 非空）。
            False（纯情绪宣泄触发、无任何意念/计划词）→ 不累积、不升级，
            仅保留看护态——"好痛苦"反复出现不触发危机（收窄原则，ADR-0013）。

    Returns:
        新的 state dict：{"status": "NONE"|"PROBING"|"CRISIS",
                          "probe_count": int, "denial_mark": bool}

    状态迁移：
        NONE  ─(P0 / crisis 裁决)────────────────→ CRISIS
        NONE  ─(有锚点 probe 裁决)──────────────→ PROBING
        NONE  ─(无锚点 probe / 情绪宣泄)─────────→ 保持 NONE（不升级）
        PROBING ─(用户确认计划/手段)─────────────→ CRISIS
        PROBING ─(用户否认/仅念头)───────────────→ NONE（denial_mark 保留）
        PROBING ─(多次有锚点 probe 复现)─────────→ CRISIS（累积升级）
        PROBING ─(无锚点 probe)─────────────────→ 保持 PROBING（不累积）
    """
    cur = state or {}
    status = cur.get("status", "NONE")
    probe_count = int(cur.get("probe_count", 0))
    denial_mark = bool(cur.get("denial_mark", False))

    if verdict == "crisis":
        status = "CRISIS"
    elif verdict == "probe":
        if status == "CRISIS":
            pass  # 已是危机，保持
        elif status == "PROBING":
            if any(kw in user_text for kw in CONFIRM_KEYWORDS):
                status = "CRISIS"
            elif _is_denial(user_text):
                status = "NONE"
                denial_mark = True
                probe_count = 0
            elif anchored_probe:
                probe_count += 1
                if probe_count >= max_probe_count:
                    status = "CRISIS"
            # 无锚点 probe（情绪宣泄）→ 保持 PROBING 看护态，不累积
        elif anchored_probe:  # NONE + 有锚点意念
            status = "PROBING"
            probe_count = 1
        # NONE + 无锚点 probe（情绪宣泄）→ 保持 NONE，不进 PROBING
    elif verdict == "no_risk":
        # 裁决无风险 → 重置为 NONE（允许从 PROBING / CRISIS 恢复）
        status = "NONE"
        denial_mark = True
        probe_count = 0
    # verdict is None（评估器跳过）→ 状态不变

    return {
        "status": status,
        "probe_count": probe_count,
        "denial_mark": denial_mark,
    }
