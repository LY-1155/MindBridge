"""输入与安全过滤 · Stub（关键词 + LLM 双重验证）"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from schemas.contracts import SafetyCheckRequest, SafetyCheckResult
from modules.safety.keyword_filter import SensitivityFilter
from modules.safety.emergency_push import get_emergency_push_service
from modules.prompt_guard import wrap_user_text, INSTRUCTION_HIERARCHY_SUFFIX
from modules.assessment.risk_anchors import match_anchor

logger = logging.getLogger(__name__)

# 模块级单例，延迟初始化避免循环导入
_filter_instance: Optional[SensitivityFilter] = None
_llm_instance = None

# LLM 验证 prompt：判断关键词命中是否为真实危机
_SAFETY_LLM_PROMPT = """你是心理危机识别专家。用户的一句话触发了敏感词检测，请判断是否为真实的自杀/自伤危机。

## 触发词
{matched_terms}

## 用户完整输入
{user_text}

## 判断标准
- 真实危机：用户明确了具体的自杀/自伤**计划、手段、时间安排，或即刻行动**，
  如"我现在就要去死""我已经决定结束自己""遗书写好了""正在割腕"
- 意念/情绪表达：仅说"不想活了""活着没意思""不如不活了""活着太痛苦"等，
  无计划与即刻行动 → 不是真实危机（但仍需医生继续关心）
- 时间描述：凌晨"三四点"、日期、数量等正常表达是误报
- 日常用语："活着好累""累死了""气死了"等口语夸张表达不是危机

## 输出格式（严格 JSON，不要其他文字）
{{"is_real_crisis": true/false, "reason": "一句话判断依据"}}"""


def _get_llm():
    """延迟初始化 LLM，避免循环导入和启动时的网络依赖。"""
    global _llm_instance
    if _llm_instance is None:
        try:
            from core.llm.base import get_llm_adapter, LLMConfig
            from config.settings import settings

            # qwen3.x 默认开思考模式：二分类 JSON 无需推理，关闭它避免 max_tokens
            # 被思考占满导致 content 为空（实测开思考 10s+ / 关思考 0.6s）。
            config = LLMConfig(
                model_name=getattr(settings, 'SCORING_MODEL_NAME', settings.MODEL_NAME),
                temperature=0,
                max_tokens=512,
                model_kwargs={"extra_body": {"enable_thinking": False}},
            )
            _llm_instance = get_llm_adapter("openai_compatible", config=config)
            logger.info("StubSafetyService: LLM 验证器已加载 (model=%s)", config.model_name)
        except Exception as e:
            logger.warning("StubSafetyService: LLM 验证器加载失败，回退纯关键词: %s", e)
            _llm_instance = None
    return _llm_instance


def _parse_verify_json(raw: str) -> Optional[dict]:
    """鲁棒解析 LLM 验证输出：容忍 ```json 代码块、前后杂文本（对齐 safety_judge._parse_json_fallback）。"""
    if not raw:
        return None
    text = raw.strip()
    # 剥掉 markdown 代码围栏（```json ... ``` / ``` ... ```）
    text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()
    # 直接解析
    try:
        return json.loads(text)
    except Exception:
        pass
    # 提取第一个完整 JSON 对象
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # 兜底：只取布尔判断
    m = re.search(r'"is_real_crisis"\s*[:：]\s*(true|false)', text, re.IGNORECASE)
    if m:
        return {"is_real_crisis": m.group(1).lower() == "true", "reason": ""}
    return None


def _verify_with_llm(text: str, keywords: list) -> tuple[bool, str]:
    """用 LLM 验证关键词命中是否为真实危机。返回 (is_real_crisis, reason)。"""
    llm = _get_llm()
    if llm is None:
        return True, "LLM 不可用，保守按真实危机处理"

    prompt = _SAFETY_LLM_PROMPT.format(
        matched_terms=", ".join(keywords),
        user_text=wrap_user_text(text),
    ) + INSTRUCTION_HIERARCHY_SUFFIX

    try:
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content if hasattr(response, "content") else str(response)
        result = _parse_verify_json(raw)
        if result is None:
            logger.warning("LLM 安全验证输出无法解析，保守按真实危机处理: %s", str(raw)[:120])
            return True, "LLM 输出格式异常"
        return bool(result.get("is_real_crisis", True)), str(result.get("reason", ""))
    except Exception as e:
        logger.warning("LLM 安全验证失败，保守按真实危机处理: %s", e)
        return True, "LLM 调用异常"


def _get_filter() -> SensitivityFilter:
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = SensitivityFilter()
        logger.info("StubSafetyService: 敏感词过滤器已加载")
    return _filter_instance


class StubSafetyService:
    """
    安全过滤服务 — 真实实现

    使用 SensitivityFilter 进行敏感词检测，含：
    - 一级高危词（自杀/暴力）→ 紧急短路
    - 二级警告词（负面情绪）→ 记录标记
    - 谐音变体检测（紫砂→自杀）
    - 防重复触发冷却机制
    """

    def check(self, req: SafetyCheckRequest) -> SafetyCheckResult:
        sf = _get_filter()

        # 调用关键词过滤器。
        # 注意：不传 user_id，因为过滤器的冷却机制是针对"紧急推送通知"的防骚扰设计，
        # 而非内容过滤决策。高危内容无论触发多少次都应被标记。
        blocked_raw, level_raw, keywords = sf.check(
            text=req.text,
            user_id=None,
        )

        # 将 filter 内部 level 映射为契约 level：
        #   filter level=1 (高危拦截)    → contract level=2 (紧急短路)
        #   filter level=2 (警告记录)    → contract level=1 (记录标记)
        #   filter level=0 (通过)        → contract level=0
        if level_raw == 1:
            contract_level = 2
            contract_blocked = blocked_raw
        elif level_raw == 2:
            contract_level = 1
            contract_blocked = False
        else:
            contract_level = 0
            contract_blocked = False

        # ── LLM 语义验证：level_1 硬拦命中且非 P0 时，用 LLM 判断是否为真实危机 ──
        # 只复核 level_1（contract_level>=2）：level_2 是软警告，跳过复核可避免把
        # 意念词降级到 level 0 后 orchestrator 的累积判定（level>=1）永不触发，
        # flag_recorder 的累积兜底随之失效。
        # P0（计划/即刻行动）短语跳过复核：避免 LLM 误降级破坏"我现在就去死"
        # "我想自杀"的快速硬拦，也省一次 LLM 调用。
        llm_verified = False
        p0_hit = bool(match_anchor(req.text).get("p0"))
        if contract_level >= 2 and keywords and not p0_hit:
            is_real, reason = _verify_with_llm(req.text, keywords)
            llm_verified = True
            if not is_real:
                # LLM 判定为误报：降级
                logger.info(
                    "LLM 安全验证: 误报降级 — keywords=%s reason=%s",
                    keywords, reason,
                )
                contract_level = 0
                contract_blocked = False
                keywords = []  # 清除误报词
            else:
                logger.info(
                    "LLM 安全验证: 确认为真实危机 — keywords=%s reason=%s",
                    keywords, reason,
                )

        # 紧急推送：当 contract_level >= 2 时触发
        emergency_push_result = None
        if contract_level >= 2:
            eps = get_emergency_push_service()
            emergency_push_result = eps.trigger(
                session_id=req.session_id or "unknown",
                matched_terms=keywords,
                user_text=req.text,
            )

        return SafetyCheckResult(
            level=contract_level,
            blocked=contract_blocked,
            matched_terms=keywords,
            meta={
                "implementation": "keyword_filter" if not llm_verified else "keyword + llm_verify",
                "filter_level_raw": level_raw,
                "filter_blocked_raw": blocked_raw,
                "locale": req.locale,
                **(
                    {"emergency_push": emergency_push_result.to_dict()}
                    if emergency_push_result is not None
                    else {}
                ),
            },
        )
