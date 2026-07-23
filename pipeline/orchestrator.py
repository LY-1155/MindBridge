"""
四阶段流水线编排 — LangGraph StateGraph
========================================
Safety → Emotion → Router → Intervention，含安全阻断短路。
模块间仅传递 JSON / Pydantic dict，对外接口不变。
"""

from __future__ import annotations

import logging
from typing import Optional, TypedDict, Annotated

from langgraph.graph import StateGraph, END

from config.settings import Settings, settings as default_settings
from modules.factory import PipelineServices

logger = logging.getLogger(__name__)
from modules.runtime import get_pipeline_services
from modules.safety.flag_recorder import SafetyFlagRecorder
from schemas.contracts import (
    EmotionAnalyzeRequest,
    EmotionTags,
    InterventionRequest,
    InterventionResult,
    PipelineInput,
    PipelineOutput,
    RouteDecision,
    RouteRequest,
    SafetyCheckRequest,
    SafetyCheckResult,
)

EMERGENCY_SAFETY_LEVEL = 3


# ── Graph state ────────────────────────────────────────────────

class PipelineState(TypedDict):
    """LangGraph 管线状态。

    每个 node 返回部分字段，LangGraph 自动 merge。
    """
    # input
    contract_version: str
    text: str
    user_id: Optional[str]
    audio_path: Optional[str]
    pre_extracted_audio_emotion: Optional[dict]
    pre_extracted_visual_emotion: Optional[dict]
    session_id: Optional[str]
    # intermediates
    safety: Optional[dict]
    emotion: Optional[dict]
    route: Optional[dict]
    # output flags
    stopped_after_safety: bool
    # output
    intervention: Optional[dict]


# ── Graph construction ─────────────────────────────────────────

def _make_pipeline_graph(svc: PipelineServices) -> StateGraph:
    """构建管线状态图，通过闭包捕获服务实例。"""

    builder = StateGraph(PipelineState)

    # ── node: safety ──
    def _safety_node(state: PipelineState) -> dict:
        safety_req = SafetyCheckRequest(
            contract_version=state["contract_version"],
            text=state["text"],
            session_id=state["session_id"],
        )
        safety = svc.safety.check(safety_req)
        safety_dict = safety.model_dump()

        # 安全标记累积升级规则（仅当 user_id 存在时生效）
        user_id = state.get("user_id")
        session_id = state.get("session_id", "")
        if user_id and safety_dict.get("level", 0) >= 1:
            recorder = SafetyFlagRecorder()
            acc_result = recorder.evaluate(
                user_id=user_id,
                session_id=session_id or "",
                level=safety_dict["level"],
                blocked=safety_dict.get("blocked", False),
                matched_terms=safety_dict.get("matched_terms", []),
            )
            if acc_result.get("escalated"):
                safety_dict["level"] = acc_result["final_level"]
                safety_dict["blocked"] = acc_result["final_blocked"]
                safety_dict["meta"] = dict(
                    safety_dict.get("meta", {}),
                    flag_escalated=True,
                    recent_warnings=acc_result.get("recent_warnings", 0),
                    threshold=acc_result.get("threshold", 3),
                )

        logger.info("[PIPELINE:TRACE] SAFETY level=%d blocked=%s terms=%s",
                     safety_dict.get("level", 0),
                     safety_dict.get("blocked", False),
                     safety_dict.get("matched_terms", []))

        shortcut = bool(safety_dict.get("blocked") or safety_dict.get("level", 0) >= EMERGENCY_SAFETY_LEVEL)
        return {"safety": safety_dict, "stopped_after_safety": shortcut}

    # ── node: emotion (normal path) ──
    def _emotion_node(state: PipelineState) -> dict:
        emotion_req = EmotionAnalyzeRequest(
            contract_version=state["contract_version"],
            text=state["text"],
            audio_path=state.get("audio_path"),
            pre_extracted_audio_emotion=state.get("pre_extracted_audio_emotion"),
            pre_extracted_visual_emotion=state.get("pre_extracted_visual_emotion"),
            safety=state["safety"],
            session_id=state["session_id"],
        )
        emotion_tags = svc.emotion.analyze(emotion_req)
        emotion_dict = emotion_tags.model_dump()

        logger.info("[PIPELINE:TRACE] EMOTION primary=%s intensity=%.2f risk=%.2f",
            emotion_dict.get("primary_emotion", "?"),
            float(emotion_dict.get("intensity", 0)),
            float(emotion_dict.get("risk", 0)),
        )

        return {"emotion": emotion_dict}

    # ── node: router (normal path) ──
    def _router_node(state: PipelineState) -> dict:
        route_req = RouteRequest(
            contract_version=state["contract_version"],
            emotion=state["emotion"],
            safety=state["safety"],
        )
        route_decision = svc.router.route(route_req)
        route_dict = route_decision.model_dump()

        logger.info("[PIPELINE:TRACE] ROUTE → %s confidence=%.2f reason=%s",
            route_dict.get("route", "?"),
            float(route_dict.get("confidence", 0)),
            route_dict.get("reason", ""),
        )

        return {"route": route_dict}

    # ── node: crisis shortcut (注入占位 emotion + route) ──
    def _crisis_emotion_router_node(state: PipelineState) -> dict:
        emotion_tags = EmotionTags(
            primary_emotion="distress",
            intensity=1.0,
            risk=1.0,
            modality_notes={"shortcut": "safety_emergency"},
        )
        emotion_dict = emotion_tags.model_dump()
        route_decision = RouteDecision(
            route="crisis",
            reason="safety_blocked_or_emergency_level",
            confidence=1.0,
        )
        route_dict = route_decision.model_dump()
        return {"emotion": emotion_dict, "route": route_dict}

    # ── node: intervention (terminal) ──
    def _intervention_node(state: PipelineState) -> dict:
        intervention_req = InterventionRequest(
            contract_version=state["contract_version"],
            user_text=state["text"],
            route=state["route"],
            emotion=state["emotion"],
            safety=state["safety"],
            session_id=state["session_id"],
        )
        intervention = svc.intervention.intervene(intervention_req)
        intervention_dict = intervention.model_dump()

        logger.info("[PIPELINE:TRACE] INTERVENTION route=%s reply_len=%d",
            state["route"].get("route", "?"),
            len(intervention.reply),
        )

        return {"intervention": intervention_dict}

    # ── condition: safety → shortcut or normal ──
    def _should_shortcut(state: PipelineState) -> str:
        return "crisis" if state["stopped_after_safety"] else "normal"

    # ── assemble graph ──
    builder.add_node("safety", _safety_node)
    builder.add_node("emotion", _emotion_node)
    builder.add_node("router", _router_node)
    builder.add_node("crisis_emotion_router", _crisis_emotion_router_node)
    builder.add_node("intervention", _intervention_node)

    builder.set_entry_point("safety")
    builder.add_conditional_edges(
        "safety",
        _should_shortcut,
        {"crisis": "crisis_emotion_router", "normal": "emotion"},
    )
    builder.add_edge("emotion", "router")
    builder.add_edge("router", "intervention")
    builder.add_edge("crisis_emotion_router", "intervention")
    builder.add_edge("intervention", END)

    return builder.compile()


# ── Public API (signatures unchanged) ──────────────────────────

def run_pipeline(
    inp: PipelineInput,
    *,
    services: Optional[PipelineServices] = None,
    settings: Optional[Settings] = None,
) -> PipelineOutput:
    """四阶段管线（LangGraph 编排）。签名与返回值不变，调用方无需改动。"""
    cfg = settings if settings is not None else default_settings
    svc = services if services is not None else get_pipeline_services(cfg)

    graph = _make_pipeline_graph(svc)

    initial_state: PipelineState = {
        "contract_version": inp.contract_version,
        "text": inp.text,
        "user_id": inp.user_id,
        "audio_path": inp.audio_path,
        "pre_extracted_audio_emotion": inp.pre_extracted_audio_emotion,
        "pre_extracted_visual_emotion": None,
        "session_id": inp.session_id,
        "safety": None,
        "emotion": None,
        "route": None,
        "stopped_after_safety": False,
        "intervention": None,
    }

    final = graph.invoke(initial_state)

    return PipelineOutput(
        contract_version=inp.contract_version,
        safety=final["safety"],
        emotion=final["emotion"],
        route=final["route"],
        intervention=final["intervention"],
        stopped_after_safety=final["stopped_after_safety"],
    )


def run_pipeline_pre(
    inp: PipelineInput,
    *,
    services: Optional[PipelineServices] = None,
    settings: Optional[Settings] = None,
) -> dict:
    """运行管线前置三阶段（Safety → Emotion → Router），不执行 Intervention。
    返回 dict 包含 safety / emotion / route / stopped_after_safety / text / user_id / session_id，
    供流式端点生成 InterventionRequest 并自行处理流式生成。
    """
    cfg = settings if settings is not None else default_settings
    svc = services if services is not None else get_pipeline_services(cfg)

    # ── Safety ──
    safety_req = SafetyCheckRequest(
        contract_version=inp.contract_version,
        text=inp.text,
        session_id=inp.session_id,
    )
    safety = svc.safety.check(safety_req)
    safety_dict = safety.model_dump()

    # 安全标记累积升级规则（同 Graph node）
    user_id = inp.user_id
    session_id = inp.session_id or ""
    if user_id and safety_dict.get("level", 0) >= 1:
        recorder = SafetyFlagRecorder()
        acc_result = recorder.evaluate(
            user_id=user_id,
            session_id=session_id,
            level=safety_dict["level"],
            blocked=safety_dict.get("blocked", False),
            matched_terms=safety_dict.get("matched_terms", []),
        )
        if acc_result.get("escalated"):
            safety_dict["level"] = acc_result["final_level"]
            safety_dict["blocked"] = acc_result["final_blocked"]
            safety_dict["meta"] = dict(
                safety_dict.get("meta", {}),
                flag_escalated=True,
                recent_warnings=acc_result.get("recent_warnings", 0),
                threshold=acc_result.get("threshold", 3),
            )

    logger.info("[PIPELINE:TRACE] SAFETY level=%d blocked=%s terms=%s",
                 safety_dict.get("level", 0),
                 safety_dict.get("blocked", False),
                 safety_dict.get("matched_terms", []))

    stopped_after_safety = bool(safety_dict.get("blocked") or safety_dict.get("level", 0) >= EMERGENCY_SAFETY_LEVEL)

    # ── Emotion + Router ──
    if stopped_after_safety:
        # 安全短路：注入占位 emotion + crisis 路由
        emotion_dict = EmotionTags(
            primary_emotion="distress",
            intensity=1.0,
            risk=1.0,
            modality_notes={"shortcut": "safety_emergency"},
        ).model_dump()
        route_dict = RouteDecision(
            route="crisis",
            reason="safety_blocked_or_emergency_level",
            confidence=1.0,
        ).model_dump()
    else:
        emotion_req = EmotionAnalyzeRequest(
            contract_version=inp.contract_version,
            text=inp.text,
            audio_path=inp.audio_path,
            pre_extracted_audio_emotion=inp.pre_extracted_audio_emotion,
            pre_extracted_visual_emotion=None,
            safety=safety_dict,
            session_id=inp.session_id,
        )
        emotion_tags = svc.emotion.analyze(emotion_req)
        emotion_dict = emotion_tags.model_dump()

        logger.info("[PIPELINE:TRACE] EMOTION primary=%s intensity=%.2f risk=%.2f",
            emotion_dict.get("primary_emotion", "?"),
            float(emotion_dict.get("intensity", 0)),
            float(emotion_dict.get("risk", 0)),
        )

        route_req = RouteRequest(
            contract_version=inp.contract_version,
            emotion=emotion_dict,
            safety=safety_dict,
        )
        route_decision = svc.router.route(route_req)
        route_dict = route_decision.model_dump()

        logger.info("[PIPELINE:TRACE] ROUTE → %s confidence=%.2f reason=%s",
            route_dict.get("route", "?"),
            float(route_dict.get("confidence", 0)),
            route_dict.get("reason", ""),
        )

    return {
        "safety": safety_dict,
        "emotion": emotion_dict,
        "route": route_dict,
        "stopped_after_safety": stopped_after_safety,
        "text": inp.text,
        "user_id": user_id,
        "session_id": session_id,
    }


def validate_stages(
    safety: SafetyCheckResult,
    emotion: EmotionTags,
    route: RouteDecision,
    intervention: InterventionResult,
) -> None:
    """调试辅助：强校验各阶段模型。"""
    SafetyCheckResult.model_validate(safety.model_dump())
    EmotionTags.model_validate(emotion.model_dump())
    RouteDecision.model_validate(route.model_dump())
    InterventionResult.model_validate(intervention.model_dump())


def run_video_pipeline(
    *,
    video_path: str,
    audio_path: Optional[str] = None,
    safety_text: str = "",
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    services: Optional[PipelineServices] = None,
    settings: Optional[Settings] = None,
) -> PipelineOutput:
    """视频管线：VideoPreprocessor 前置 → 四阶段管线（LangGraph 编排）。"""
    from multimodal.video_preprocessor import VideoPreprocessor

    cfg = settings if settings is not None else default_settings
    svc = services if services is not None else get_pipeline_services(cfg)

    preprocessor = VideoPreprocessor()
    pre_result = preprocessor.process(video_path, audio_path=audio_path)

    safety_input = safety_text or pre_result.text or ""

    graph = _make_pipeline_graph(svc)

    initial_state: PipelineState = {
        "contract_version": "1.2",
        "text": safety_input,
        "user_id": user_id,
        "audio_path": audio_path or video_path,
        "pre_extracted_audio_emotion": pre_result.audio_emotion,
        "pre_extracted_visual_emotion": pre_result.visual_emotion,
        "session_id": session_id,
        "safety": None,
        "emotion": None,
        "route": None,
        "stopped_after_safety": False,
        "intervention": None,
    }

    final = graph.invoke(initial_state)

    return PipelineOutput(
        contract_version="1.2",
        safety=final["safety"],
        emotion=final["emotion"],
        route=final["route"],
        intervention=final["intervention"],
        stopped_after_safety=final["stopped_after_safety"],
    )
