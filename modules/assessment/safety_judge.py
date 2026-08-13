"""LLM 语义安全评估器（Semantic Safety Judge）

危机判定改造核心：**风险词命中 ≠ 危机触发**。
- 读最近若干轮对话上下文 + 规则锚点信号，输出三级裁决 crisis / probe / no_risk
- 静默运行，不面向用户（2-agent 边界，ADR-0013）
- P0 显式危险 → 规则硬闸门直接 crisis，不发 LLM
- 无锚点 → 返回 None（零额外延迟，不调用 LLM）
- LLM 失败/超时 → 按锚点强度保守降级（P0→crisis，P1→probe，P2→no_risk）

参考 `modules/safety/stub.py` 的 `_verify_with_llm` 雏形，升级为多轮上下文 + 三级分级 + 结构化输出。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from config.settings import settings
from modules.assessment.risk_anchors import (
    anchor_strength,
    first_risk_type,
    flatten_anchors,
    match_anchor,
)
from modules.prompt_guard import wrap_user_text, INSTRUCTION_HIERARCHY_SUFFIX

logger = logging.getLogger(__name__)

# ── 探针兜底模板（LLM 未生成 probe_suggestion 时使用）──────────
FALLBACK_PROBE: Dict[str, str] = {
    "suicide": "是真心不想活，还是心里难受，还是都有？",
    "self_harm": "是想伤害自己，还是心里太难受了需要发泄？",
    "violence": "是想真的动手，还是气过头了想发泄？",
    "general": "是不是心里太难受了？跟我说说发生了什么？",
}


class SafetyVerdictModel(BaseModel):
    """LLM 结构化输出契约。"""

    verdict: str = Field(..., description="crisis / probe / no_risk")
    risk_type: str = Field("general", description="suicide / self_harm / violence / general")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="裁决置信度 0~1")
    reason: str = Field("", description="一句话判断依据")
    probe_suggestion: Optional[str] = Field(None, description="verdict=probe 时给出的自然探针问题")


@dataclass
class SafetyVerdict:
    """评估器裁决结果。"""

    verdict: str  # crisis / probe / no_risk
    risk_type: str = "general"
    confidence: float = 0.5
    reason: str = ""
    probe_suggestion: Optional[str] = None
    matched_anchors: List[str] = field(default_factory=list)
    source: str = "llm"  # llm / rule_p0 / fallback

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "risk_type": self.risk_type,
            "confidence": self.confidence,
            "reason": self.reason,
            "probe_suggestion": self.probe_suggestion,
            "matched_anchors": list(self.matched_anchors),
            "source": self.source,
        }


# ── 评估器 system prompt ──────────────────────────────────────
_JUDGE_SYSTEM_PROMPT = """你是心理危机判定专家，服务于一位心理医生（周医生）的后台。你的任务是判断用户当前输入是否构成真实危机，输出三级裁决。你只做后台判断，不直接面对用户。

## 对话历史（最近若干轮）
{recent_history}

## 用户当前输入
{user_text}

## 规则信号（来自前置过滤，仅供参考，不是判决）
- 情绪风险分：{emotion_risk}（0~1，越高越危险）
- 命中风险词：{matched_terms}
- SCID 跨轮累积标记：{scid_flags}

## 三级裁决定义
- **crisis**：用户明确了具体的自杀/自伤**计划、手段、时间安排，或即刻行动**（如"我现在就要去死""我已经决定结束自己""遗书写好了""买了安眠药""正在楼顶"）。危险迫在眉睫，需要立即危机干预。
- **probe**：存在自杀/自伤**意念或情绪表达**（如"不想活了""活着没意思""不如不活了""活着太痛苦""好想解脱""一了百了算了"），或只提到方法名词（"跳楼""割腕"）而无计划与即刻行动，需要医生继续对话并自然追问一句区分"念头 vs 计划"。
- **no_risk**：口语夸张/情绪宣泄（"累死了""气死了""烦死了""活着好累""好痛苦""太难受了""很难熬"），无自我伤害风险。

## 反例（务必遵守）
- "我不想活了/活着没意思/还不如不活了/活着太痛苦" → 一律 **probe**，不是 crisis
- "跳楼挺可怕的/看到割腕的新闻" → 无计划即刻 → **probe**
- "累死了/气死了/活着好累/好痛苦/太难受了/很难熬" → 纯情绪宣泄 → **no_risk**，即使情绪强度很高
- 仅表达难受/痛苦、未提任何意念词（不想活/自杀/解脱/结束生命）或计划手段 → **no_risk**，不是 probe

## 判定要求
1. 必须结合对话历史判断，不能只凭当前一句话或单个词
2. 宁可从严触发 probe（多问一句无害），不可漏掉真实危机
3. crisis 必须满足"计划/手段/时间安排/即刻行动"之一；仅意念与情绪表达不得判 crisis
4. 如果判定 no_risk，请先在内心自查：这些风险词放在这段上下文里，为什么不构成实际风险？把理由写进 reason
5. 如果判定 probe，probe_suggestion 给一句自然的口语化探针问题，用于区分"念头"与"计划"，不要生硬、不要审问腔
6. 只输出 JSON

{format_instructions}"""


class SafetyJudge:
    """LLM 语义安全评估器。可注入 llm 适配器便于测试。"""

    def __init__(self, llm=None):
        self._llm = llm

    # ── Public API ─────────────────────────────────────────────

    def judge(
        self,
        text: str,
        *,
        history: Optional[List[Dict[str, str]]] = None,
        emotion_risk: float = 0.0,
        safety: Optional[Dict[str, Any]] = None,
        scid_flags: Optional[Dict[str, Any]] = None,
        safety_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[SafetyVerdict]:
        """对当前输入做三级裁决。

        Args:
            text: 用户当前输入
            history: 会话历史（List[{"role", "content"}]）
            emotion_risk: 情绪风险分 0~1
            safety: SafetyCheckResult dict
            scid_flags: session.metadata.scid_flags
            safety_state: session.metadata.safety_state（CRISIS/PROBING 时强制评估）

        Returns:
            SafetyVerdict：crisis / probe / no_risk
            None：无锚点，跳过（正常对话，零额外延迟）
        """
        anchors = match_anchor(text)

        # 1. P0 硬闸门：规则直接判定 crisis，不发 LLM
        if anchors["p0"]:
            verdict = SafetyVerdict(
                verdict="crisis",
                risk_type=first_risk_type(anchors) or "general",
                confidence=1.0,
                reason="P0 显式危险锚点命中，规则硬闸门",
                matched_anchors=flatten_anchors(anchors),
                source="rule_p0",
            )
            logger.warning(
                "[SAFETY_JUDGE] P0 hard gate: verdict=crisis anchors=%s",
                verdict.matched_anchors,
            )
            return verdict

        # 2. 无锚点 → 跳过（但 CRISIS/PROBING 会话强制评估，持续监测）
        if not self._has_anchor(text, emotion_risk, safety, scid_flags, history, safety_state):
            return None

        # 3. LLM 语义裁决
        return self._invoke_llm(text, history, emotion_risk, safety, scid_flags, anchors)

    # ── 锚点检测 ───────────────────────────────────────────────

    def _has_anchor(
        self,
        text: str,
        emotion_risk: float,
        safety: Optional[Dict[str, Any]],
        scid_flags: Optional[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]],
        safety_state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """满足任一条即调用评估器。"""
        if settings.SAFETY_JUDGE_EVERY_TURN:
            return True
        # 会话已处于危机/探针态 → 持续监测，不跳过
        if safety_state and safety_state.get("status") in ("CRISIS", "PROBING"):
            return True

        anchors = match_anchor(text)
        if anchors["p1"]:
            return True
        if emotion_risk >= settings.SAFETY_JUDGE_ANCHOR_RISK_THRESHOLD:
            return True
        if safety:
            if safety.get("blocked") or safety.get("level", 0) >= 2:
                return True
        if self._scid_has_risk(scid_flags):
            return True
        if history and self._history_has_risk(history):
            return True
        return False

    @staticmethod
    def _scid_has_risk(scid_flags: Optional[Dict[str, Any]]) -> bool:
        if not scid_flags:
            return False
        risk_criteria = {"death_si", "fear_dying", "fear_losing_control"}
        for disorder_data in scid_flags.values():
            if not isinstance(disorder_data, dict):
                continue
            criteria = disorder_data.get("criteria_met", [])
            if risk_criteria & set(criteria):
                return True
        return False

    @staticmethod
    def _history_has_risk(history: List[Dict[str, str]]) -> bool:
        """会话历史里曾有 P0/P1 风险信号 → 复现锚点。"""
        recent = history[-(settings.SAFETY_JUDGE_HISTORY_TURNS * 2):]
        combined = " ".join(str(m.get("content", "")) for m in recent)
        h = match_anchor(combined)
        return bool(h["p0"] or h["p1"])

    # ── LLM 调用 ───────────────────────────────────────────────

    def _get_llm(self):
        if self._llm is None:
            from core.llm.base import get_llm_adapter, LLMConfig

            config = LLMConfig(
                model_name=settings.SAFETY_JUDGE_MODEL,
                temperature=0,
                max_tokens=256,
                timeout=settings.SAFETY_JUDGE_TIMEOUT_SECONDS,
                # qwen3.x 默认开思考：结构化 JSON 裁决须关思考，否则推理耗时超 5s 超时
                model_kwargs={"extra_body": {"enable_thinking": False}},
            )
            self._llm = get_llm_adapter("openai_compatible", config=config)
            logger.info(
                "[SAFETY_JUDGE] LLM loaded model=%s timeout=%ds",
                config.model_name, config.timeout,
            )
        return self._llm

    def _invoke_llm(
        self,
        text: str,
        history: Optional[List[Dict[str, str]]],
        emotion_risk: float,
        safety: Optional[Dict[str, Any]],
        scid_flags: Optional[Dict[str, Any]],
        anchors: Dict[str, Dict[str, List[str]]],
    ) -> Optional[SafetyVerdict]:
        llm = self._get_llm()
        if llm is None:
            return self._fallback_verdict(text, emotion_risk, anchors)

        from langchain_core.messages import HumanMessage
        from langchain_core.output_parsers import PydanticOutputParser

        parser = PydanticOutputParser(pydantic_object=SafetyVerdictModel)
        prompt = _JUDGE_SYSTEM_PROMPT.format(
            recent_history=self._format_history(history),
            user_text=wrap_user_text(text),
            emotion_risk=f"{emotion_risk:.2f}",
            matched_terms=self._format_terms(safety),
            scid_flags=self._format_scid(scid_flags),
            format_instructions=parser.get_format_instructions(),
        ) + INSTRUCTION_HIERARCHY_SUFFIX

        # 失败重试一次（仿 scale/scorer 模式），再失败 → 保守兜底
        for attempt in range(2):
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                raw = response.content if hasattr(response, "content") else str(response)
                parsed = self._parse(raw, parser)
                if parsed is not None:
                    parsed.matched_anchors = flatten_anchors(anchors)
                    parsed = self._apply_p0_guardrail(parsed, text, history, anchors)
                    self._log_verdict(parsed, emotion_risk)
                    return parsed
            except Exception as exc:
                logger.warning("[SAFETY_JUDGE] LLM 调用失败 attempt=%d: %s", attempt, exc)

        return self._fallback_verdict(text, emotion_risk, anchors)

    @staticmethod
    def _has_p0_anchor(text: str, history: Optional[List[Dict[str, str]]]) -> bool:
        """当前文本或近期历史是否含 P0（计划/即刻行动）锚点。"""
        if match_anchor(text)["p0"]:
            return True
        if history:
            recent = history[-(settings.SAFETY_JUDGE_HISTORY_TURNS * 2):]
            combined = " ".join(str(m.get("content", "")) for m in recent)
            if match_anchor(combined)["p0"]:
                return True
        return False

    @staticmethod
    def _apply_p0_guardrail(
        parsed: SafetyVerdict,
        text: str,
        history: Optional[List[Dict[str, str]]],
        anchors: Dict[str, Dict[str, List[str]]],
    ) -> SafetyVerdict:
        """护栏：LLM 判 crisis 但当前文本与近期历史均无 P0 锚点 → 降级 probe（人审）。

        crisis 只留给 P0（计划/即刻行动/主动意图），纯意念被 LLM 误判时由这里兜底。
        仅拦 source=="llm"；rule_p0 / fallback 的 crisis 只在 P0 命中时出现，天然不受影响；
        probe 状态机累积升级（PROBING→CRISIS）在 InterventionService，与护栏无关。
        """
        if (
            parsed.verdict == "crisis"
            and parsed.source == "llm"
            and not SafetyJudge._has_p0_anchor(text, history)
        ):
            logger.warning(
                "[SAFETY_JUDGE] 护栏：LLM crisis 无 P0 锚点 → 降级 probe anchors=%s",
                flatten_anchors(anchors),
            )
            parsed.verdict = "probe"
            parsed.source = "llm_guardrail"
            parsed.reason = "LLM 判 crisis 但当前文本与近期历史均无 P0 锚点，护栏降级为探针（人审）"
        return parsed

    @staticmethod
    def _parse(raw: str, parser) -> Optional[SafetyVerdict]:
        """解析 LLM 输出：先 PydanticOutputParser，再 JSON 兜底。"""
        try:
            parsed = parser.parse(raw)
            if parsed.verdict not in ("crisis", "probe", "no_risk"):
                return None
            return SafetyVerdict(
                verdict=parsed.verdict,
                risk_type=parsed.risk_type or "general",
                confidence=float(parsed.confidence or 0.5),
                reason=parsed.reason or "",
                probe_suggestion=parsed.probe_suggestion,
                source="llm",
            )
        except Exception:
            return SafetyJudge._parse_json_fallback(raw)

    @staticmethod
    def _parse_json_fallback(raw: str) -> Optional[SafetyVerdict]:
        """裸 JSON / 代码块包裹解析兜底。"""
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = re.sub(r"^```[a-z]*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
            result = json.loads(text)
            verdict = result.get("verdict")
            if verdict not in ("crisis", "probe", "no_risk"):
                return None
            return SafetyVerdict(
                verdict=verdict,
                risk_type=result.get("risk_type", "general"),
                confidence=float(result.get("confidence", 0.5)),
                reason=result.get("reason", ""),
                probe_suggestion=result.get("probe_suggestion"),
                source="llm",
            )
        except Exception:
            return None

    def _fallback_verdict(
        self,
        text: str,
        emotion_risk: float,
        anchors: Dict[str, Dict[str, List[str]]],
    ) -> SafetyVerdict:
        """LLM 失败时按锚点强度保守降级（ADR-0013：规则兜底保安全下限）。

        crisis 只留给 P0（计划/即刻行动）；纯情绪高亢（无 P0）兜底为 probe 而非 crisis，
        符合"意念/情绪走探针、计划才报警"的收窄原则。
        """
        tier = anchor_strength(anchors)
        risk_type = first_risk_type(anchors) or "general"

        if tier == "p0":
            return SafetyVerdict(
                verdict="crisis",
                risk_type=risk_type,
                confidence=0.8,
                reason="评估器不可用，P0 显式计划/即刻锚点保守降级为 crisis",
                matched_anchors=flatten_anchors(anchors),
                source="fallback",
            )
        if tier == "p1" or emotion_risk >= settings.SAFETY_JUDGE_ANCHOR_RISK_THRESHOLD:
            return SafetyVerdict(
                verdict="probe",
                risk_type=risk_type,
                confidence=0.6,
                reason="评估器不可用，P1 意念锚点/情绪风险保守降级为 probe",
                matched_anchors=flatten_anchors(anchors),
                probe_suggestion=FALLBACK_PROBE.get(risk_type, FALLBACK_PROBE["general"]),
                source="fallback",
            )
        return SafetyVerdict(
            verdict="no_risk",
            risk_type=risk_type,
            confidence=0.5,
            reason="评估器不可用，无强锚点，按无风险处理",
            matched_anchors=flatten_anchors(anchors),
            source="fallback",
        )

    # ── 格式化 helper ──────────────────────────────────────────

    @staticmethod
    def _format_history(history: Optional[List[Dict[str, str]]]) -> str:
        if not history:
            return "（无历史，这是第一轮）"
        recent = history[-(settings.SAFETY_JUDGE_HISTORY_TURNS * 2):]
        lines = []
        for msg in recent:
            role = "用户" if msg.get("role") == "user" else "医生"
            lines.append(f"{role}：{msg.get('content', '')}")
        return "\n".join(lines)

    @staticmethod
    def _format_terms(safety: Optional[Dict[str, Any]]) -> str:
        if not safety:
            return "（无）"
        terms = safety.get("matched_terms") or []
        return "、".join(terms) if terms else "（无）"

    @staticmethod
    def _format_scid(scid_flags: Optional[Dict[str, Any]]) -> str:
        if not scid_flags:
            return "（无）"
        parts = []
        for disorder, data in scid_flags.items():
            if not isinstance(data, dict):
                continue
            criteria = data.get("criteria_met", [])
            if criteria:
                parts.append(f"{disorder}({data.get('count', len(criteria))}): {','.join(criteria)}")
        return "；".join(parts) if parts else "（无）"

    @staticmethod
    def _log_verdict(verdict: SafetyVerdict, emotion_risk: float) -> None:
        logger.info(
            "[SAFETY_JUDGE] verdict=%s risk_type=%s conf=%.2f anchors=%s reason=%s",
            verdict.verdict, verdict.risk_type, verdict.confidence,
            verdict.matched_anchors, verdict.reason,
        )
        # 高危→降级（锚点命中但裁决非 crisis）落 WARNING，供人审（ADR-0013）
        if verdict.verdict != "crisis" and (verdict.matched_anchors or emotion_risk >= 0.7):
            logger.warning(
                "[SAFETY_JUDGE] 高危锚点降级 — verdict=%s anchors=%s emotion_risk=%.2f reason=%s",
                verdict.verdict, verdict.matched_anchors, emotion_risk, verdict.reason,
            )
