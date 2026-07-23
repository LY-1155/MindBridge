from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from modules.auth_deps import get_current_user_id
from modules.ai_disclaimer import apply_disclaimer
from pydantic import BaseModel, Field

from config.settings import settings
from core.memory.session_memory import SessionManager, SessionOwnershipError
from pipeline.orchestrator import run_pipeline, run_pipeline_pre
from schemas.contracts.v1 import PipelineInput, InterventionRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["chat"])


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Session id")
    message: str = Field(min_length=1, max_length=2000)
    enable_thought_chain: bool = True
    enable_emotion_analysis: bool = True


class ChatResponse(BaseModel):
    session_id: str
    response: str
    thought_chain: Optional[Dict[str, Any]] = None
    emotion_analysis: Optional[Dict[str, Any]] = None
    suggested_techniques: List[str] = Field(default_factory=list)
    safety_alert: bool = False


class SessionInfo(BaseModel):
    session_id: str
    message_count: int
    created_at: str
    last_activity: str
    emotion_summary: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    version: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user_id: str = Depends(get_current_user_id)) -> ChatResponse:
    try:
        session_id = request.session_id or SessionManager.create_session(user_id=user_id)

        inp = PipelineInput(
            text=request.message,
            user_id=user_id,
            session_id=session_id,
        )
        output = run_pipeline(inp)

        # 持久化情绪记录到 emotion_records 表
        if output.emotion:
            try:
                from core.memory.db_storage import DatabaseStorage
                DatabaseStorage.add_emotion_record(
                    session_id=session_id,
                    primary_emotion=output.emotion.get("primary_emotion", "neutral"),
                    intensity=float(output.emotion.get("intensity", 0)),
                    risk=float(output.emotion.get("risk", 0)),
                    triggers=[],
                    user_id=user_id,
                )
            except Exception:
                pass  # 非关键路径，不阻塞 response

        intervention = output.intervention
        safety = output.safety

        return ChatResponse(
            session_id=session_id,
            response=apply_disclaimer(intervention.get("reply", "")),
            thought_chain=(
                {"chain_of_thought": intervention["chain_of_thought"]}
                if intervention.get("chain_of_thought") else None
            ),
            emotion_analysis=output.emotion,
            suggested_techniques=intervention.get("action_items", []),
            safety_alert=bool(
                safety.get("blocked") or safety.get("level", 0) >= 2
            ),
        )
    except Exception as exc:
        logger.exception("chat failed")
        raise HTTPException(status_code=500, detail=f"处理消息失败: {exc}")


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, user_id: str = Depends(get_current_user_id)):
    """流式聊天：Safety→Emotion→Router 同步完成，Intervention 通过 SSE 逐 token 推送"""
    try:
        session_id = request.session_id or SessionManager.create_session(user_id=user_id)

        inp = PipelineInput(
            text=request.message,
            user_id=user_id,
            session_id=session_id,
        )
        pre_state = run_pipeline_pre(inp)

        # 持久化情绪记录（同 /chat）
        if pre_state["emotion"]:
            try:
                from core.memory.db_storage import DatabaseStorage
                DatabaseStorage.add_emotion_record(
                    session_id=session_id,
                    primary_emotion=pre_state["emotion"].get("primary_emotion", "neutral"),
                    intensity=float(pre_state["emotion"].get("intensity", 0)),
                    risk=float(pre_state["emotion"].get("risk", 0)),
                    triggers=[],
                    user_id=user_id,
                )
            except Exception:
                pass

        safety = pre_state["safety"]
        safety_alert = bool(safety.get("blocked") or safety.get("level", 0) >= 2)

        intervention_req = InterventionRequest(
            user_text=request.message,
            route=pre_state["route"],
            emotion=pre_state["emotion"],
            safety=safety,
            session_id=session_id,
            user_id=user_id,
        )

        from modules.runtime import get_pipeline_services
        svc = get_pipeline_services()

        async def event_generator():
            # meta 事件：前置管线结果
            meta = {
                "type": "meta",
                "session_id": session_id,
                "safety_alert": safety_alert,
                "emotion_analysis": pre_state["emotion"],
                "route": pre_state["route"],
            }
            yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

            # status 事件：告知前端进入"思考/检索"阶段，避免长时间无反馈
            yield f"data: {json.dumps({'type': 'status', 'phase': 'thinking'}, ensure_ascii=False)}\n\n"

            try:
                async for token in svc.intervention.astream_intervene(intervention_req):
                    yield f"data: {json.dumps({'type': 'delta', 'token': token}, ensure_ascii=False)}\n\n"

                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
            except asyncio.CancelledError:
                pass  # 客户端断开，静默终止
            except Exception as exc:
                logger.exception("stream error")
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as exc:
        logger.exception("chat stream failed")
        raise HTTPException(status_code=500, detail=f"处理消息失败: {exc}")


@router.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str, user_id: str = Depends(get_current_user_id)) -> SessionInfo:
    try:
        session = SessionManager.get_session(session_id, user_id=user_id)
    except SessionOwnershipError:
        raise HTTPException(status_code=403, detail="无权访问此会话")
    try:
        metadata = session.metadata
        emotion_summary: Dict[str, int] = {}
        for record in session.emotion_records:
            emotion_summary[record.primary_emotion] = emotion_summary.get(record.primary_emotion, 0) + 1
        return SessionInfo(
            session_id=session_id,
            message_count=len(session.get_messages()),
            created_at=metadata.created_at.isoformat(),
            last_activity=metadata.last_active.isoformat(),
            emotion_summary=emotion_summary,
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在: {exc}")


@router.delete("/session/{session_id}")
async def delete_session(session_id: str, user_id: str = Depends(get_current_user_id)):
    # 通过 get_session 做归属校验
    try:
        SessionManager.get_session(session_id, user_id=user_id)
    except SessionOwnershipError:
        raise HTTPException(status_code=403, detail="无权删除此会话")

    SessionManager.delete_session(session_id)
    return {"message": f"会话 {session_id} 已删除"}


@router.get("/sessions")
async def list_sessions(limit: int = Query(default=10, ge=1, le=100), user_id: str = Depends(get_current_user_id)):
    # 优先 Redis 索引 → 回退 MySQL
    session_ids = SessionManager.get_active_sessions_by_user(user_id)
    if not session_ids:
        session_ids = SessionManager.get_active_sessions()
    return {"sessions": session_ids[:limit]}


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="healthy", version="1.0.0")
