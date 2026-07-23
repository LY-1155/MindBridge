"""量表结果反哺知识检索：将已完成量表的评估结果注入 RAG 查询，使检索到
的心理学知识与用户的具体状况（障碍类型、严重程度、突出症状维度）精准匹配。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from modules.intervention.scale.models import ScaleConfig, ScaleState

logger = logging.getLogger(__name__)

# ── 量表 → 障碍领域 / 知识检索关键词 ──────────────────────────
SCALE_TO_DISORDER: Dict[str, Dict[str, str]] = {
    "PHQ-9": {
        "area": "抑郁症",
        "keywords": "抑郁 情绪低落 兴趣减退 睡眠障碍 自我否定 精力不足 食欲改变 注意力困难 自杀念头",
    },
    "GAD-7": {
        "area": "广泛性焦虑障碍",
        "keywords": "焦虑 过度担忧 紧张不安 难以放松 坐立不安 易怒 恐惧感 无法停止担忧",
    },
    "ASRS": {
        "area": "成人注意缺陷多动障碍",
        "keywords": "注意力不集中 多动冲动 组织困难 任务收尾困难 健忘 ADHD 执行功能",
    },
    "AUDIT": {
        "area": "酒精使用障碍",
        "keywords": "饮酒 酒精依赖 酒精滥用 成瘾 戒断 酗酒",
    },
    "ISI": {
        "area": "失眠障碍",
        "keywords": "失眠 入睡困难 睡眠维持困难 早醒 睡眠质量 日间功能 疲劳",
    },
    "LSAS": {
        "area": "社交焦虑障碍",
        "keywords": "社交焦虑 社交恐惧 回避行为 表现焦虑 社交互动恐惧 人际敏感",
    },
    "MDQ": {
        "area": "双相情感障碍",
        "keywords": "心境障碍 躁狂 轻躁狂 情绪波动 精力过剩 易激惹 睡眠需求减少",
    },
    "OCI-R": {
        "area": "强迫障碍",
        "keywords": "强迫思维 强迫行为 检查 清洗 囤积 排序 中和 OCD",
    },
    "PCL-5": {
        "area": "创伤后应激障碍",
        "keywords": "PTSD 创伤 闯入回忆 噩梦 回避 警觉性 闪回 情绪麻木",
    },
    "PHQ-15": {
        "area": "躯体症状障碍",
        "keywords": "躯体症状 身体疼痛 胃肠不适 心悸 头晕 疲乏 健康焦虑",
    },
    "SCOFF": {
        "area": "进食障碍",
        "keywords": "进食障碍 厌食 暴食 催吐 体重控制 体象障碍 饮食行为",
    },
}

# ── 严重程度 → 治疗阶段 / 干预方向关键词 ───────────────────────
LEVEL_TO_CONTEXT: Dict[str, str] = {
    "无": "心理健康维护 预防 生活方式",
    "轻度": "早期干预 自助策略 生活方式调整 心理教育",
    "中度": "循证心理治疗 认知行为疗法 CBT 正念 行为激活 人际治疗",
    "中重度": "综合治疗 强化心理治疗 药物辅助 定期监测 社会支持",
    "重度": "专业精神科治疗 药物治疗 心理治疗 危机干预 支持系统",
}

# ── 可作为退避覆盖所有未匹配 level 的兜底 ───────────────────────
_DEFAULT_LEVEL_CONTEXT = "心理健康 心理治疗 自我关怀"


def get_high_concern_dimensions(
    scale_state: ScaleState,
    config: ScaleConfig,
    top_n: int = 3,
) -> List[str]:
    """从量表评分中提取分值最高的维度。

    对 sum/count_yes/asrs 量表返回 scores >= 2 的 item dimension，
    对 LSAS 由于双轴计分，取值取两维均值。按得分降序取前 top_n 个。
    """
    if config.scoring_mode == "lsas":
        # scores 交替存储 fear / avoidance，需要按 item 对求均值
        items = config.items
        item_scores = []
        for i in range(len(items)):
            fear_idx = i * 2
            avoid_idx = i * 2 + 1
            fear = scale_state.scores[fear_idx] if fear_idx < len(scale_state.scores) else 0
            avoid = scale_state.scores[avoid_idx] if avoid_idx < len(scale_state.scores) else 0
            avg = (fear + avoid) / 2
            item_scores.append((i, avg))
        # 按均值降序，取 >= 1.5 的
        item_scores.sort(key=lambda x: x[1], reverse=True)
        dims = []
        for idx, score in item_scores:
            if score >= 1.5 and idx < len(items):
                item = items[idx]
                dims.append(item.dimension or item.situation or item.concept)
        return dims[:top_n]

    # 普通量表：单维 score
    pairs = list(enumerate(scale_state.scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    dims = []
    for idx, score in pairs:
        if score >= 2 and idx < len(config.items):
            dims.append(config.items[idx].dimension or config.items[idx].concept)
    return dims[:top_n]


def build_scale_profile(
    scale_state: ScaleState,
    config: Optional[ScaleConfig] = None,
) -> Optional[Dict]:
    """从已完成量表的 ScaleState 构建结构化的用户评估画像。

    Returns None 如果量表未完成或配置加载失败。
    """
    if scale_state.status != "completed" or scale_state.scale_name is None:
        return None

    try:
        cfg = config or ScaleConfig.from_json(scale_state.scale_name)
    except Exception:
        logger.warning("加载量表配置失败: %s", scale_state.scale_name, exc_info=True)
        return None

    disorder = SCALE_TO_DISORDER.get(scale_state.scale_name, {})
    high_dims = get_high_concern_dimensions(scale_state, cfg)

    return {
        "scale_name": scale_state.scale_name,
        "display_name": cfg.display_name,
        "total_score": scale_state.total_score,
        "level": scale_state.level or "未知",
        "high_dimensions": high_dims,
        "disorder_area": disorder.get("area", ""),
        "disorder_keywords": disorder.get("keywords", ""),
        "escalation_flag": scale_state.escalation_flag,
    }


def enrich_query_with_scale(
    user_text: str,
    session,
) -> str:
    """将量表评估结果注入用户查询，提升知识检索精准度。

    从 session.metadata.scale_state 读取最近完成的量表，构建结构化背景段
    附加到用户原始 query 之后。无已完成量表时原样返回 user_text。
    """
    try:
        raw = getattr(session.metadata, "scale_state", None)
        if not raw or raw.get("status") != "completed":
            return user_text

        scale_state = ScaleState(**raw)
        profile = build_scale_profile(scale_state)
        if profile is None:
            return user_text

        # 严重程度对应的干预方向
        # 按 key 长度降序匹配，避免 "中度" 在 "中重度" 前抢先命中
        level = profile["level"]
        level_context = _DEFAULT_LEVEL_CONTEXT
        for key in sorted(LEVEL_TO_CONTEXT, key=len, reverse=True):
            if key in level:
                level_context = LEVEL_TO_CONTEXT[key]
                break

        high_dims_text = "、".join(profile["high_dimensions"]) if profile["high_dimensions"] else "无突出维度"

        context = (
            f"\n\n[用户评估背景] "
            f"已完成{profile['display_name']}（{profile['disorder_area']}），"
            f"评估程度：{profile['level']}。"
            f"突出症状维度：{high_dims_text}。"
            f"检索倾向：{level_context}；{profile['disorder_keywords']}。"
        )

        enriched = user_text + context
        logger.info(
            "量表反哺：scale=%s level=%s high_dims=%s",
            profile["scale_name"],
            profile["level"],
            high_dims_text,
        )
        return enriched

    except Exception:
        logger.warning("量表反哺失败，降级为原始查询", exc_info=True)
        return user_text
