"""输入与安全过滤 · Stub（关键词 + LLM 双重验证）"""

from __future__ import annotations

import json
import logging
from typing import Optional

from schemas.contracts import SafetyCheckRequest, SafetyCheckResult
from modules.safety.keyword_filter import SensitivityFilter
from modules.safety.emergency_push import get_emergency_push_service
from modules.prompt_guard import wrap_user_text, INSTRUCTION_HIERARCHY_SUFFIX

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
- 真实危机：用户明确表达了结束生命、伤害自己或他人的意图或计划
- 情绪表达：用户只是在描述痛苦、疲惫、失眠等，没有自我伤害的意图
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

            config = LLMConfig(
                model_name=getattr(settings, 'SCORING_MODEL_NAME', settings.MODEL_NAME),
                temperature=0,
                max_tokens=128,
            )
            _llm_instance = get_llm_adapter("openai_compatible", config=config)
            logger.info("StubSafetyService: LLM 验证器已加载 (model=%s)", config.model_name)
        except Exception as e:
            logger.warning("StubSafetyService: LLM 验证器加载失败，回退纯关键词: %s", e)
            _llm_instance = None
    return _llm_instance


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
        result = json.loads(raw)
        return result.get("is_real_crisis", True), result.get("reason", "")
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

        # ── LLM 语义验证：关键词命中后，用 LLM 判断是否为真实危机 ──
        llm_verified = False
        if contract_level >= 1 and keywords:
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
