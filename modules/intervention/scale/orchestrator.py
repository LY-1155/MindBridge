"""ScaleOrchestrator:量表生命周期管理

触发 → 施测 → 计分 → 完成/放弃。LLM 和 Scorer 由外部注入。
"""
from __future__ import annotations

import json
import logging
import os
from typing import List, Dict

from modules.intervention.scale.models import (
    ScaleConfig,
    ScaleState,
    ScaleTurnResult,
)
from modules.intervention.scale.scorer import ScaleScorer
from modules.prompt_guard import wrap_user_text, INSTRUCTION_HIERARCHY_SUFFIX

logger = logging.getLogger(__name__)

# D1: intensity 低于此值不触发量表
TRIGGER_INTENSITY_THRESHOLD = 0.4

_SCALES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "data", "knowledge", "private", "scales",
)

# D5: LLM 语义量表选择 prompt
SCALE_SELECTION_PROMPT = """你是心理评估专家。根据用户发言判断是否需要量表筛查,并推荐最匹配的量表。

## 可用量表
{scale_catalog}

## 判断规则
- 用户描述自身心理/身体症状、情绪困扰 → 匹配相关量表
- 用户在问知识问题、讨论概念 → 不触发量表
- 找不到任何匹配 → trigger: false
- 多量表匹配时按 relevance 排序

## 用户发言
{user_text}

## 输出格式（严格 JSON,不要其他文字）
{{"trigger": true/false, "scales": ["量表名"], "reason": "简短判断依据"}}"""

# Prompt:将量表维度转化为自然对话提问
QUESTION_PROMPT = """你是一位温暖的心理咨询师，正在和用户进行自然对话。

## 对话历史（最近几轮对话）
{conversation_history}

## 你想了解的维度
- 症状维度：{dimension}
- 症状范围：{concept}

## 对话上下文
{context}

## 要求
用自然、共情的方式引出一个关于上述维度的询问。不要念题、不要说量表名字、不要让用户觉得在填问卷。
只需输出一句话的询问，不要加引号。

⚠️ **禁止重复提问**：上面对话历史里如果你或用户已经聊过某个话题，绝对不要换说法再问一遍。用户刚回答过的就当已知，顺着往前聊。"""


def _build_scale_catalog() -> List[Dict]:
    """从 JSON 文件中提取所有量表的 name + display_name + description。"""
    catalog = []
    for filename in sorted(os.listdir(_SCALES_DIR)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(_SCALES_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            catalog.append({
                "name": data["name"],
                "display_name": data.get("display_name", ""),
                "description": data.get("description", ""),
            })
        except Exception:
            logger.warning("Failed to load scale catalog entry: %s", filename, exc_info=True)
    return catalog


class ScaleOrchestrator:
    """量表生命周期:start → process_turn × N → complete/abandon"""

    def __init__(self, llm, scorer: ScaleScorer):
        self._llm = llm
        self._scorer = scorer

    # ── public API ──────────────────────────────────────────

    def should_trigger(self, user_text: str, emotion_tags: Dict) -> List[str]:
        """D5: LLM 语义匹配量表。返回匹配的量表名列表（按 relevance 排序）。

        1. intensity 门槛检查（< 0.4 不触发）
        2. LLM 阅读用户发言,与全部 11 个量表 description 语义匹配
        3. 返回匹配的量表名列表,无匹配返回空列表
        """
        # 1. intensity 门槛
        intensity = float((emotion_tags or {}).get("intensity", 0))
        if intensity < TRIGGER_INTENSITY_THRESHOLD:
            return []

        # 2. LLM 语义量表选择
        catalog = _build_scale_catalog()
        if not catalog:
            return []

        catalog_text = "\n".join(
            f"{s['name']}（{s['display_name']}）:{s['description']}"
            for s in catalog
        )
        wrapped = wrap_user_text(user_text)
        prompt = SCALE_SELECTION_PROMPT.format(
            scale_catalog=catalog_text,
            user_text=wrapped,
        ) + INSTRUCTION_HIERARCHY_SUFFIX

        from langchain_core.messages import HumanMessage
        response = self._llm.invoke([HumanMessage(content=prompt)])
        text = (response.content if hasattr(response, "content") else str(response)).strip()

        # 3. 解析 JSON
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("量表选择 LLM 返回非 JSON: %.100s", text)
            return []

        if not result.get("trigger"):
            return []

        scales = result.get("scales", [])
        if not isinstance(scales, list):
            return []

        # 验证量表名有效性
        valid_names = {s["name"] for s in catalog}
        valid_scales = [s for s in scales if s in valid_names]
        if valid_scales:
            logger.info(
                "量表选择:scales=%s reason=%s",
                valid_scales,
                result.get("reason", ""),
            )
            logger.info("[SCALE:EVENT] SELECTED scales=%s reason=%s",
                         valid_scales, result.get("reason", ""))
        else:
            logger.info("[SCALE:EVENT] SELECTED scales=[] (no match or not triggered)")
        return valid_scales

    def start(self, scale_name: str, session) -> str:
        """初始化量表状态,返回第一道自然语言提问。"""
        config = ScaleConfig.from_json(scale_name)

        state = ScaleState(scale_name=scale_name)
        if config.scoring_mode == "lsas":
            state.current_dimension = "fear"
        session.metadata.scale_state = state.model_dump()
        session.save_scale_state()

        msg = "[SCALE:EVENT] START scale=%s mode=%s items=%d" % (
            scale_name, config.scoring_mode, len(config.items))
        logger.info(msg)

        return self._generate_question(config.items[0], "（对话刚开始）", config, dim_key=state.current_dimension, session=session)

    def process_turn(self, user_reply: str, session) -> ScaleTurnResult:
        """处理用户本轮回复:计分 → 推进/完成。"""
        raw = session.metadata.scale_state
        if raw is None:
            return ScaleTurnResult(
                reply="（量表状态异常）", is_complete=True, escalation_flag=False
            )
        state = ScaleState(**raw)
        config = ScaleConfig.from_json(state.scale_name)
        current_item = config.items[state.current_item_index]

        # 1. 计分（LSAS 双轴走专用路径）
        if config.scoring_mode == "lsas" and state.current_dimension:
            dims = config.dual_dimensions or {}
            dim_cfg = dims.get(state.current_dimension, {})
            score = self._scorer.score_dual(user_reply, current_item, state.current_dimension, dim_cfg)
        else:
            score = self._scorer.score(user_reply, current_item)

        # 2. 偏离检测
        if score == -1:
            state.wander_count += 1
            logger.info("[SCALE:EVENT] WANDER scale=%s wander=%d/2 item=%d",
                         state.scale_name, state.wander_count, state.current_item_index)
            if state.wander_count >= 2:
                state.status = "abandoned"
                session.metadata.scale_state = state.model_dump()
                session.save_scale_state()
                logger.info("[SCALE:EVENT] ABANDON scale=%s reason=wander_x2 item=%d scores=%s",
                             state.scale_name, state.current_item_index, state.scores)
                abandon_msg = "没关系,我们换个话题聊聊吧。如果你之后想继续了解自己的状态,随时可以告诉我。"
                return ScaleTurnResult(
                    reply=abandon_msg, is_complete=True, escalation_flag=False
                )
            next_msg = self._generate_soft_redirect(current_item, user_reply, config)
            session.metadata.scale_state = state.model_dump()
            session.save_scale_state()
            return ScaleTurnResult(
                reply=next_msg, is_complete=False, escalation_flag=False
            )

        # 3. 有效计分
        state.wander_count = 0
        state.scores.append(score)
        logger.info("[SCALE:EVENT] Q_%d scale=%s score=%d dim=%s",
                     state.current_item_index, state.scale_name, score,
                     state.current_dimension or "-")

        # 4. 推进:LSAS 双轴切换 vs 普通下一题
        if config.scoring_mode == "lsas" and state.current_dimension == "fear":
            # 刚评完 fear → 切到 avoidance,同一 item
            state.current_dimension = "avoidance"
            next_msg = self._generate_question(current_item, user_reply, config, dim_key=state.current_dimension, session=session)
            session.metadata.scale_state = state.model_dump()
            session.save_scale_state()
            return ScaleTurnResult(
                reply=next_msg, is_complete=False, escalation_flag=False
            )
        elif config.scoring_mode == "lsas" and state.current_dimension == "avoidance":
            # 刚评完 avoidance → 下一题
            state.current_dimension = "fear"
            state.current_item_index += 1
        else:
            # 普通量表:直接下一题
            state.current_item_index += 1

        # 5. LSAS 完成判定:48 个分数（24 题 × 2 维）
        if config.scoring_mode == "lsas":
            total_items = len(config.items)
            expected_scores = total_items * 2
            if len(state.scores) >= expected_scores:
                return self._complete_scale(state, config, session)
        elif state.current_item_index >= len(config.items):
            return self._complete_scale(state, config, session)

        # 6. 继续下一题
        next_item = config.items[state.current_item_index]
        next_msg = self._generate_question(next_item, user_reply, config, dim_key=state.current_dimension, session=session)
        session.metadata.scale_state = state.model_dump()
        session.save_scale_state()
        return ScaleTurnResult(
            reply=next_msg, is_complete=False, escalation_flag=False
        )

    # ── internal ────────────────────────────────────────────

    def _generate_question(self, item, context_summary: str, config: ScaleConfig = None,
                           dim_key: str = None, session=None) -> str:
        """用 LLM 生成一道自然的对话式询问。LSAS 模式通过 dim_key 指定维度。"""
        from langchain_core.messages import SystemMessage, HumanMessage

        dimension = item.dimension or item.situation or ""
        concept = item.concept

        # 提取对话历史（用于反重复）
        conv_history = "（无历史对话）"
        if session is not None:
            try:
                history = session.get_history_for_prompt()
                if history:
                    recent = history[-6:]  # 最近 3 轮
                    lines = [f"{'用户' if m['role'] == 'user' else '助手'}：{m['content']}" for m in recent]
                    conv_history = "\n".join(lines)
            except Exception:
                pass

        # LSAS:附加维度标签到概念描述中
        if config and config.scoring_mode == "lsas" and dim_key and config.dual_dimensions:
            dim_cfg = config.dual_dimensions.get(dim_key, {})
            dim_label = dim_cfg.get("label", dim_key)
            concept = f"在「{item.situation or item.concept}」这个情境下,{dim_label}的程度"

        prompt = QUESTION_PROMPT.format(
            dimension=dimension,
            concept=concept,
            context=context_summary,
            conversation_history=conv_history,
        )
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="请生成询问。"),
        ]
        response = self._llm.invoke(messages)
        return (response.content if hasattr(response, "content") else str(response)).strip()

    def _generate_soft_redirect(self, item, user_reply: str, config: ScaleConfig = None) -> str:
        """用户偏离时:先回应再柔和拉回。"""
        from langchain_core.messages import SystemMessage, HumanMessage

        dim = item.dimension or item.situation or ""
        wrapped = wrap_user_text(user_reply)
        prompt = f"""你是一位温暖的心理咨询师。

用户刚说了:"{wrapped}"

你需要先简短地共情回应,然后柔和地拉回到这个话题:
- 症状维度:{dim}
- 症状范围:{item.concept}

要求:先共情（1-2 句）,再自然过渡到询问。不要道歉、不要说"我们回到问卷"。""" + INSTRUCTION_HIERARCHY_SUFFIX
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="请回复。"),
        ]
        response = self._llm.invoke(messages)
        return (response.content if hasattr(response, "content") else str(response)).strip()

    def _complete_scale(self, state: ScaleState, config: ScaleConfig, session) -> ScaleTurnResult:
        """完成量表: 按 scoring_mode 计总分, 判定分级和升段."""
        mode = config.scoring_mode
        scores = state.scores

        # ── 计总分 ──
        if mode == "asrs":
            total = self._count_asrs(scores, config)
        elif mode == "mdq":
            # Q1 题数（items 0-12）的 yes 计数
            q1_count = sum(scores[:13])
            q2 = scores[13] if len(scores) > 13 else 0
            q3 = scores[14] if len(scores) > 14 else 0
            rule = config.positive_rule or {}
            is_positive = (
                q1_count >= rule.get("q1_yes_min", 7)
                and (not rule.get("q2_required", True) or q2 == 1)
                and q3 >= rule.get("q3_min_level", 2)
            )
            total = q1_count if is_positive else q1_count  # show Q1 count as total
        else:
            # sum / count_yes / lsas:全部求和
            total = sum(scores)

        # ── 判定分级 ──
        if mode == "mdq":
            rule = config.positive_rule or {}
            q2 = scores[13] if len(scores) > 13 else 0
            q3 = scores[14] if len(scores) > 14 else 0
            is_positive = (
                total >= rule.get("q1_yes_min", 7)
                and (not rule.get("q2_required", True) or q2 == 1)
                and q3 >= rule.get("q3_min_level", 2)
            )
            level = config.thresholds[1].level if is_positive else config.thresholds[0].level
        else:
            level = self._classify(total, config)

        # ── 升段判定 ──
        escalation = False
        for item_idx in config.escalation.item_triggers:
            if item_idx < len(scores) and scores[item_idx] >= config.escalation.item_threshold:
                escalation = True
                break
        if total >= config.escalation.total_score_threshold:
            escalation = True

        state.status = "completed"
        state.total_score = total
        state.level = level
        state.escalation_flag = escalation
        session.metadata.scale_state = state.model_dump()
        session.save_scale_state()

        msg = "[SCALE:EVENT] DONE scale=%s total=%d level=%s escalation=%s" % (
            state.scale_name, total, level, escalation)
        logger.info("%s scores=%s", msg, scores)

        summary = self._generate_completion_summary(config, total, level, escalation)
        return ScaleTurnResult(
            reply=summary,
            is_complete=True,
            total_score=total,
            level=level,
            escalation_flag=escalation,
        )

    def _count_asrs(self, scores: list, config: ScaleConfig) -> int:
        """ASRS 非对称计分:每项对照 item_thresholds 判断达标,数达标题数。"""
        thresholds = config.item_thresholds or {}
        count = 0
        for i, s in enumerate(scores):
            th = thresholds.get(str(i), 2)
            if s >= th:
                count += 1
        return count

    def _classify(self, total: int, config: ScaleConfig) -> str:
        for t in config.thresholds:
            if t.min <= total <= t.max:
                return t.level
        return config.thresholds[-1].level

    def _generate_completion_summary(self, config: ScaleConfig, total: int, level: str, escalation: bool) -> str:
        """生成完成后的自然语言总结（不暴露原始分数）。"""
        if self._llm is None:
            return f"（评估完成:{level}）"
        from langchain_core.messages import SystemMessage, HumanMessage

        prompt = f"""你是一位温暖的心理咨询师。用户刚完成了关于情绪状态的自我评估。

评估维度和结果（仅供参考）:
- 量表:{config.display_name}
- 大致程度:{level}

请用自然、温和的方式给用户一个简短的反馈:
1. 肯定用户花时间了解自己的状态
2. 用自然语言描述当前的感受程度（不要报数字或分数）
3. 给一个温和的行动建议
4. {"如果这些感受让您难以承受,请随时拨打心理援助热线 400-161-9995,我们也可以帮您联系专业人士。" if escalation else ""}

不要提到"量表"、"得分"、"问卷"这些词。就像朋友间的关心。"""
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="请生成反馈。"),
        ]
        response = self._llm.invoke(messages)
        return (response.content if hasattr(response, "content") else str(response)).strip()
