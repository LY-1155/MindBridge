"""SCIDTracker：静默 SCID-5 诊断标准追踪器

对照 DSM-5 标准，基于规则（关键词+模式）匹配临床 criteria。
完全静默 — 不产生对话输出，仅更新 session.scid_flags。

危机判定改造（ADR-0013）：risk_flags 语义变为**语义安全评估器的锚点输入**，
不再直接升级路由。跨轮累积判断（如 death_si 复现）由评估器读取 session.scid_flags
在图上完成（见 modules/assessment/safety_judge.py 的 _scid_has_risk）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# ── DSM-5 Disorder ←→ Criteria Keywords ──────────────────────
# Phase 1 覆盖：MDD, GAD, Panic, PTSD, Suicide Risk
# 每个 criteria 对应一组中文关键词

DISORDER_CRITERIA: Dict[str, Dict[str, List[str]]] = {
    "MDD": {
        "depressed_mood": [
            "心情不好", "心情低落", "情绪低落", "不开心", "郁闷", "绝望", "空虚",
            "难过", "悲伤", "高兴不起来", "闷闷不乐", "很丧", "丧", "低落",
        ],
        "anhedonia": [
            "没兴趣", "没劲", "什么都不想做", "以前喜欢的", "打不起精神",
            "没意思", "无聊", "提不起兴趣", "不好玩了",
        ],
        "weight_appetite": [
            "吃不下", "没胃口", "食欲", "瘦了", "体重下降", "不想吃饭",
            "吃得很少", "暴饮暴食", "吃很多", "体重增加",
        ],
        "sleep": [
            "睡不着", "入睡困难", "半夜醒", "早醒", "失眠", "睡不好",
            "睡得浅", "老醒", "睡太多", "嗜睡", "躺着不想起",
        ],
        "psychomotor": [
            "坐不住", "停不下来", "很慢", "迟钝", "反应慢", "动不了",
        ],
        "fatigue": [
            "累", "疲劳", "没精力", "疲倦", "乏力", "浑身没劲",
            "不想动", "没力气",
        ],
        "worthlessness_guilt": [
            "没用", "废物", "拖累", "对不起", "我不好", "不值得",
            "内疚", "自责", "都是我的错", "对不起家人",
        ],
        "concentration": [
            "集中不了", "记不住", "忘事", "走神", "听不进去",
            "注意力", "脑子不转", "想不清楚",
        ],
        "death_si": [
            "不想活", "想死", "死了算了", "自杀", "结束自己",
            "活着没意思", "活不下去了", "解脱", "一了百了", "想不开", "轻生",
        ],
    },
    "GAD": {
        "excessive_worry": [
            "担心", "控制不住地想", "停不下来想", "总在想", "胡思乱想",
            "想太多", "过度思考", "胡思乱想",
        ],
        "restlessness": [
            "坐立不安", "静不下来", "焦躁", "急", "安不下心",
        ],
        "fatigue": [
            "累", "疲劳", "精力差", "没精神",
        ],
        "concentration": [
            "集中不了", "大脑空白", "记不住", "走神",
        ],
        "irritability": [
            "烦躁", "易怒", "发脾气", "暴躁", "不耐烦", "很烦",
        ],
        "muscle_tension": [
            "肌肉紧张", "肩膀酸", "脖子硬", "僵硬", "紧绷",
        ],
        "sleep": [
            "睡不着", "入睡难", "睡不好", "睡眠浅", "早醒",
        ],
    },
    "Panic": {
        "palpitations": [
            "心跳快", "心跳好快", "心慌", "心悸", "心跳加速", "心脏要跳出来", "心跳", "心率",
        ],
        "sweating": ["出汗", "冒汗", "冷汗"],
        "trembling": ["发抖", "手抖", "颤抖"],
        "shortness_breath": [
            "喘不上气", "呼吸困难", "胸闷", "窒息感", "透不过气",
        ],
        "choking": ["窒息", "喉咙堵", "噎住"],
        "chest_pain": ["胸痛", "胸口不舒服", "心脏疼"],
        "nausea": ["恶心", "肚子不舒服", "想吐"],
        "dizziness": ["头晕", "眩晕", "站不稳", "要晕倒"],
        "chills_heat": ["冷", "热", "发冷", "发热"],
        "paresthesia": ["发麻", "麻木", "针扎感"],
        "derealization": ["不真实", "梦一样", "恍惚", "在梦里"],
        "fear_losing_control": ["失控", "要疯了", "控制不住自己"],
        "fear_dying": ["要死了", "活不成了", "濒死感"],
    },
    "PTSD": {
        "intrusion": [
            "噩梦", "闪回", "跳出来", "浮现", "反复想", "控制不住地回忆",
        ],
        "avoidance": [
            "逃避", "避开", "不去想", "不敢去", "绕开", "躲着",
        ],
        "negative_cognition": [
            "世界不安全", "不能相信", "都是我的错", "我应该", "变了个人",
        ],
        "hyperarousal": [
            "容易被吓到", "警觉", "紧张", "睡不好", "易怒", "突然",
        ],
    },
}

# ── Disorder threshold：至少满足 N 条 criteria 才标记为 suspected ──
DISORDER_THRESHOLDS = {
    "MDD": 5,    # DSM-5: ≥5 of 9 (含 depressed_mood 或 anhedonia)
    "GAD": 3,    # DSM-5: ≥3 of 6
    "Panic": 4,  # DSM-5: ≥4 of 13
    "PTSD": 4,   # DSM-5: 每个 cluster ≥1
}

# ── 安全风险 disorders ─────────────────────────────────────────
RISK_DISORDERS = {
    "MDD": ["death_si"],  # criteria name → 触发时标记 risk
    "Panic": ["fear_dying", "fear_losing_control"],
}


@dataclass
class SCIDUpdate:
    """SCID 追踪结果 — 不包含任何面向用户的文本"""
    suspected_diagnosis: Optional[str] = None
    # 当前最可能的诊断方向（内部用）
    criteria_met: Dict[str, List[str]] = field(default_factory=dict)
    # {"MDD": ["sleep", "anhedonia", ...], ...}
    risk_flags: List[str] = field(default_factory=list)
    # ["MDD_suicide_risk", ...] → InterventionService 据此升级
    suggested_retrieval_query: Optional[str] = None
    # 为 RAG 检索生成的精准 query


class SCIDTracker:
    """静默 SCID-5 追踪器。"""

    def update(
        self,
        user_text: str,
        existing_flags: Dict[str, Any],
    ) -> SCIDUpdate:
        """基于用户本轮文本 + 历史 flags 做增量匹配。

        Args:
            user_text: 当前用户输入
            existing_flags: session.metadata.scid_flags（历史累积）
        """
        result = SCIDUpdate()
        all_criteria: Dict[str, List[str]] = {}

        # 1. 关键词匹配 — 逐 disorder 检查
        for disorder, criteria_map in DISORDER_CRITERIA.items():
            matched: List[str] = []
            for criterion_name, keywords in criteria_map.items():
                for kw in keywords:
                    if kw in user_text:
                        matched.append(criterion_name)
                        break
            if matched:
                all_criteria[disorder] = matched

        # 合并已有 flags
        merged = dict(existing_flags)
        for disorder, new_criteria in all_criteria.items():
            if disorder in merged:
                existing_criteria = set(
                    merged[disorder].get("criteria_met", [])
                )
                combined = sorted(existing_criteria | set(new_criteria))
                merged[disorder] = {
                    "criteria_met": combined,
                    "count": len(combined),
                }
            else:
                merged[disorder] = {
                    "criteria_met": new_criteria,
                    "count": len(new_criteria),
                }

        result.criteria_met = merged

        # 2. 阈值检查 → suspected diagnosis
        best_disorder = None
        best_count = 0
        for disorder, threshold in DISORDER_THRESHOLDS.items():
            data = merged.get(disorder, {})
            count = data.get("count", 0)
            if count >= threshold and count > best_count:
                # MDD 特殊规则：必须有 depressed_mood 或 anhedonia
                if disorder == "MDD":
                    criteria_list = data.get("criteria_met", [])
                    if "depressed_mood" not in criteria_list and "anhedonia" not in criteria_list:
                        continue
                best_disorder = disorder
                best_count = count

        result.suspected_diagnosis = best_disorder

        # 3. 安全风险标记
        risk_flags: List[str] = []
        for disorder, risk_criteria_list in RISK_DISORDERS.items():
            data = merged.get(disorder, {})
            criteria_list = data.get("criteria_met", [])
            for rc in risk_criteria_list:
                if rc in criteria_list:
                    risk_flags.append(f"{disorder}_suicide_risk")
        result.risk_flags = risk_flags

        # 4. 生成检索查询
        result.suggested_retrieval_query = self._build_retrieval_query(
            best_disorder, merged, user_text
        )

        logger.debug(
            "SCID: suspected=%s criteria=%s risk=%s",
            result.suspected_diagnosis,
            {k: v.get("count", 0) for k, v in merged.items()},
            result.risk_flags,
        )

        return result

    def _build_retrieval_query(
        self,
        suspected: Optional[str],
        criteria: Dict[str, Any],
        user_text: str,
    ) -> Optional[str]:
        """基于 SCID 追踪结果构建精准检索 query。"""
        if not suspected:
            return None

        disorder_cn = {
            "MDD": "抑郁症", "GAD": "广泛性焦虑", "Panic": "惊恐障碍", "PTSD": "创伤后应激障碍",
        }
        stage_keywords = {
            "MDD": "家庭治疗 CBT 行为激活 青少年",
            "GAD": "认知行为治疗 放松训练 暴露治疗",
            "Panic": "惊恐控制治疗 内感受暴露 认知重构",
            "PTSD": "创伤知情护理 稳定化技术 安全岛",
        }

        cn = disorder_cn.get(suspected, suspected)
        stage = stage_keywords.get(suspected, "")
        return f"{cn} {stage} 循证心理治疗 中文 临床指南"
