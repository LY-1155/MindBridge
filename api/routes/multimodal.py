from __future__ import annotations

import base64
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from modules.auth_deps import get_current_user_id
from modules.file_upload_security import validate_upload
from modules.ai_disclaimer import apply_disclaimer
from fastapi.responses import Response
from pydantic import BaseModel, Field

from config.settings import settings
from core.memory.session_memory import SessionManager, SessionOwnershipError
from pipeline.orchestrator import run_pipeline
from schemas.contracts.v1 import PipelineInput
from multimodal.audio_emotion import get_audio_emotion_recognizer
from multimodal.emotion import get_emotion_recognizer
from multimodal.tts import TextToSpeech, get_tts

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/multimodal", tags=["multimodal"])

_MIME_TO_EXT = {
    "video/mp4": ".mp4",
    "video/x-msvideo": ".avi",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
    "video/webm": ".webm",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/flac": ".flac",
}


def _mime_to_ext(mime: str) -> str:
    """MIME 类型 → 文件后缀，未知返回空字符串。"""
    return _MIME_TO_EXT.get(mime, "")


class SpeechToTextResponse(BaseModel):
    text: str
    language: str
    duration: float
    confidence: float = 0.9


class TextToSpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    voice: Optional[str] = None
    rate: Optional[str] = "+0%"


class MultimodalChatRequest(BaseModel):
    session_id: Optional[str] = None
    text: Optional[str] = None
    audio_data: Optional[str] = None
    image_data: Optional[str] = None
    video_data: Optional[str] = None
    enable_tts: bool = True


class MultimodalChatResponse(BaseModel):
    session_id: str
    response: str
    audio_base64: Optional[str] = None
    emotion: Optional[Dict[str, Any]] = None
    visual_emotion: Optional[Dict[str, Any]] = None
    audio_emotion: Optional[Dict[str, Any]] = None
    fused_emotion: Optional[Dict[str, Any]] = None
    transcribed_text: Optional[str] = None


async def _detect_visual_emotion(image_bytes: bytes) -> Dict[str, Any]:
    recognizer = get_emotion_recognizer()
    result = recognizer.recognize_from_bytes(image_bytes)
    emotion_cn_map = {
        "angry": "愤怒",
        "disgust": "厌恶",
        "fear": "恐惧",
        "happy": "开心",
        "sad": "悲伤",
        "surprise": "惊讶",
        "neutral": "平静",
        "unknown": "未知",
    }
    return {
        "primary_emotion": result.primary_emotion,
        "confidence": result.confidence,
        "all_emotions": result.all_emotions,
        "face_detected": result.face_detected,
        "emotion_cn": emotion_cn_map.get(result.primary_emotion, result.primary_emotion),
        "model_name": settings.VISUAL_EMOTION_BACKEND,
    }


def _b64_decode(data: str) -> bytes:
    payload = data.split(",", 1)[1] if "," in data else data
    return base64.b64decode(payload)


def _build_emotion_context(
    *,
    fused_emotion: Optional[Dict[str, Any]],
    audio_emotion: Optional[Dict[str, Any]],
    visual_emotion: Optional[Dict[str, Any]],
) -> str:
    chunks: List[str] = []
    if fused_emotion:
        chunks.append(
            f"综合情绪={fused_emotion.get('primary_emotion')} confidence={float(fused_emotion.get('confidence') or 0):.2f} summary={fused_emotion.get('summary', '')}"
        )
    if audio_emotion:
        chunks.append(
            f"语音情绪={audio_emotion.get('primary_emotion')} confidence={float(audio_emotion.get('confidence') or 0):.2f}"
        )
    if visual_emotion:
        chunks.append(
            f"面部情绪={visual_emotion.get('primary_emotion')} confidence={float(visual_emotion.get('confidence') or 0):.2f}"
        )
    return "；".join(chunks)


@router.post("/speech-to-text", response_model=SpeechToTextResponse)
async def speech_to_text(
    audio: UploadFile = File(..., description="音频文件"),
    language: Optional[str] = Form(default=None, description="语言代码"),
    user_id: str = Depends(get_current_user_id),
):
    try:
        from multimodal.asr import get_speech_recognizer

        await validate_upload(audio, category="audio")
        audio_bytes = await audio.read()
        # 根据上传文件的文件名或 MIME 类型推断后缀，支持视频文件（mp4 等）
        suffix = ""
        if audio.filename:
            suffix = os.path.splitext(audio.filename)[1]
        if not suffix and audio.content_type:
            suffix = _mime_to_ext(audio.content_type)
        if not suffix:
            suffix = ".wav"
        recognizer = get_speech_recognizer()
        result = recognizer.transcribe_bytes(audio_bytes, language=language, suffix=suffix)
        return SpeechToTextResponse(
            text=result.text,
            language=result.language,
            duration=result.duration,
            confidence=0.9,
        )
    except Exception as exc:
        logger.error("speech_to_text failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"语音识别失败: {exc}")


@router.post("/text-to-speech")
async def text_to_speech(request: TextToSpeechRequest, user_id: str = Depends(get_current_user_id)):
    try:
        tts = get_tts(voice=request.voice)
        audio_bytes = await tts.synthesize_to_bytes(request.text, rate=request.rate)
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"},
        )
    except Exception as exc:
        logger.error("text_to_speech failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"语音合成失败: {exc}")


@router.post("/text-to-speech/base64")
async def text_to_speech_base64(request: TextToSpeechRequest, user_id: str = Depends(get_current_user_id)):
    try:
        tts = get_tts(voice=request.voice)
        base64_audio = await tts.synthesize_to_base64(request.text, rate=request.rate)
        return {"audio_base64": base64_audio, "format": "mp3", "voice": tts.voice}
    except Exception as exc:
        logger.error("text_to_speech_base64 failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"语音合成失败: {exc}")


@router.post("/chat", response_model=MultimodalChatResponse)
async def multimodal_chat(request: MultimodalChatRequest, user_id: str = Depends(get_current_user_id)):
    try:
        from multimodal.asr import get_speech_recognizer

        transcribed_text: Optional[str] = None
        audio_emotion_info: Optional[Dict[str, Any]] = None
        visual_emotion_info: Optional[Dict[str, Any]] = None
        fused_emotion_info: Optional[Dict[str, Any]] = None
        # 视频：提取音频并转录
        if request.video_data and transcribed_text is None:
            video_bytes = _b64_decode(request.video_data)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                tmp_video.write(video_bytes)
                tmp_video.close()
                temp_video_path = tmp_video.name
            temp_audio_path = None
            try:
                from modules.safety.multimodal.audio_processor import AudioProcessor
                ap = AudioProcessor(device=settings.SAFETY_DEVICE)
                temp_audio_path = ap.extract_audio(temp_video_path)
                if temp_audio_path and os.path.exists(temp_audio_path):
                    with open(temp_audio_path, 'rb') as f:
                        audio_bytes_read = f.read()
                    recognizer = get_speech_recognizer()
                    asr_result = recognizer.transcribe_bytes(audio_bytes_read)
                    transcribed_text = asr_result.text
                    try:
                        audio_emotion = get_audio_emotion_recognizer().recognize(
                            audio_path=temp_audio_path,
                            transcript=transcribed_text or "",
                        )
                        audio_emotion_info = {
                            "primary_emotion": audio_emotion.primary_emotion,
                            "confidence": audio_emotion.confidence,
                            "all_emotions": audio_emotion.all_emotions,
                            "model_name": audio_emotion.model_name,
                            "backend": audio_emotion.backend,
                            "warnings": audio_emotion.warnings,
                        }
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("视频音频提取/转录失败: %s", e)
            finally:
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                if temp_audio_path and os.path.exists(temp_audio_path):
                    try:
                        os.unlink(temp_audio_path)
                    except OSError:
                        pass

        if request.audio_data and transcribed_text is None:
            audio_bytes = _b64_decode(request.audio_data)
            recognizer = get_speech_recognizer()
            asr_result = recognizer.transcribe_bytes(audio_bytes)
            transcribed_text = asr_result.text
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                tmp_audio.write(audio_bytes)
                temp_audio_path = tmp_audio.name
            try:
                audio_emotion = get_audio_emotion_recognizer().recognize(
                    audio_path=temp_audio_path,
                    transcript=transcribed_text,
                )
                audio_emotion_info = {
                    "primary_emotion": audio_emotion.primary_emotion,
                    "confidence": audio_emotion.confidence,
                    "all_emotions": audio_emotion.all_emotions,
                    "model_name": audio_emotion.model_name,
                    "backend": audio_emotion.backend,
                    "warnings": audio_emotion.warnings,
                }
            finally:
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)

        if request.image_data:
            visual_emotion_info = await _detect_visual_emotion(_b64_decode(request.image_data))

        if settings.ENABLE_MULTIMODAL_EMOTION_FUSION and not fused_emotion_info:
            from multimodal.emotion_fusion import build_signal, fuse_emotions

            fused = fuse_emotions(
                build_signal("audio", audio_emotion_info),
                build_signal("visual", visual_emotion_info),
            )
            fused_emotion_info = fused.to_dict() if fused else None

        user_input = (request.text or transcribed_text or "").strip()
        if not user_input:
            raise HTTPException(status_code=400, detail="请提供文本、音频或视频输入")

        emotion_context = _build_emotion_context(
            fused_emotion=fused_emotion_info,
            audio_emotion=audio_emotion_info,
            visual_emotion=visual_emotion_info,
        )
        if emotion_context:
            user_input = f"[多模态情绪线索] {emotion_context}\n\n来访者表达: {user_input}"

        session_id = request.session_id or SessionManager.create_session(user_id=user_id)

        inp = PipelineInput(
            text=user_input,
            user_id=user_id,
            session_id=session_id,
        )
        output = run_pipeline(inp)
        reply = output.intervention.get("reply", "")

        # 持久化情绪记录到 emotion_records 表（对齐 text 路径 chat.py）
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

        audio_base64 = None
        if request.enable_tts:
            try:
                audio_base64 = await get_tts().synthesize_to_base64(reply)
            except Exception as exc:
                logger.warning("TTS synthesis failed for multimodal chat: %s", exc)

        return MultimodalChatResponse(
            session_id=session_id,
            response=apply_disclaimer(reply),
            audio_base64=audio_base64,
            emotion=fused_emotion_info or visual_emotion_info,
            visual_emotion=visual_emotion_info,
            audio_emotion=audio_emotion_info,
            fused_emotion=fused_emotion_info,
            transcribed_text=transcribed_text,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("multimodal_chat failed")
        raise HTTPException(status_code=500, detail=f"处理失败: {exc}")


@router.get("/voices")
async def list_voices(language: str = "zh", user_id: str = Depends(get_current_user_id)):
    try:
        voices = await TextToSpeech.list_voices(language)
        return {
            "voices": [
                {
                    "name": voice.name,
                    "short_name": voice.short_name,
                    "gender": voice.gender,
                    "locale": voice.locale,
                }
                for voice in voices
            ]
        }
    except Exception as exc:
        logger.error("list_voices failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"获取声音列表失败: {exc}")


# ==================== 多模态安全过滤 API ====================

class ImageSafetyResponse(BaseModel):
    level: int = 0
    blocked: bool = False
    violations: List[Dict[str, Any]] = Field(default_factory=list)
    nsfw_prob: float = 0.0
    normal_prob: float = 0.0


class AudioSafetyResponse(BaseModel):
    level: int = 0
    blocked: bool = False
    keywords: List[str] = Field(default_factory=list)
    transcript: str = ""


class VideoSafetyResponse(BaseModel):
    blocked: bool = False
    audio_level: int = 0
    audio_keywords: List[str] = Field(default_factory=list)
    audio_transcript: str = ""
    video_level: int = 0
    video_violations: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time: float = 0.0
    error: str = ""
    audio_error: str = ""
    video_error: str = ""


_video_safety_filter = None


def _get_video_safety_filter():
    global _video_safety_filter
    if _video_safety_filter is None:
        from modules.safety.multimodal.video_safety_filter import VideoSafetyFilter

        model_path = settings.SAFETY_MODEL_PATH or None
        nsfw_model_path = settings.SAFETY_NSFW_MODEL_PATH or None
        _video_safety_filter = VideoSafetyFilter(
            device=settings.SAFETY_DEVICE,
            save_evidence=True,
            local_model_path=model_path,
            nsfw_model_path=nsfw_model_path,
        )
    return _video_safety_filter


@router.post("/safety/image", response_model=ImageSafetyResponse)
async def safety_check_image(image: UploadFile = File(..., description="图片文件"), user_id: str = Depends(get_current_user_id)):
    try:
        import cv2, numpy as np
        from modules.safety.multimodal.frame_detector_local import FrameDetector

        await validate_upload(image, category="image")
        data = await image.read()
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="无法解码图片")

        model_path = settings.SAFETY_MODEL_PATH or None
        nsfw_model_path = settings.SAFETY_NSFW_MODEL_PATH or None
        detector = FrameDetector(
            device=settings.SAFETY_DEVICE,
            confidence_threshold=0.75,
            local_model_path=model_path,
            nsfw_model_path=nsfw_model_path,
        )
        detector._load_classifier()
        result = detector.detect_frame(frame)

        violations = result.get("violations", [])
        probs = result.get("all_probs", {})
        return ImageSafetyResponse(
            level=result.get("level", 0),
            blocked=result.get("level", 0) >= 1,
            violations=violations,
            nsfw_prob=probs.get("nsfw", 0.0),
            normal_prob=probs.get("normal", 0.0),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("safety_check_image failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"图片安全检测失败: {exc}")


@router.post("/safety/audio", response_model=AudioSafetyResponse)
async def safety_check_audio(audio: UploadFile = File(..., description="音频文件"), user_id: str = Depends(get_current_user_id)):
    try:
        from modules.safety.multimodal.audio_processor import AudioProcessor

        await validate_upload(audio, category="audio")
        suffix = os.path.splitext(audio.filename or "audio")[1] or ".wav"
        data = await audio.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            tmp.write(data)
            tmp.close()
            ap = AudioProcessor(
                device=settings.SAFETY_DEVICE,
                filter_config_path=None,
            )
            result = ap.check_audio(tmp.name)
            # check_audio 返回 Dict，level 是 filter 原始值（1=高危, 2=警告）
            raw_level = result.get("level", 0)
            if raw_level == 1:
                contract_level = 2
                blocked = True
            elif raw_level == 2:
                contract_level = 1
                blocked = False
            else:
                contract_level = 0
                blocked = False
            return AudioSafetyResponse(
                level=contract_level,
                blocked=blocked,
                keywords=result.get("keywords", []),
                transcript=result.get("transcript", ""),
            )
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("safety_check_audio failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"音频安全检测失败: {exc}")


@router.post("/safety/video", response_model=VideoSafetyResponse)
async def safety_check_video(video: UploadFile = File(..., description="视频文件"), user_id: str = Depends(get_current_user_id)):
    try:
        await validate_upload(video, category="video")
        suffix = os.path.splitext(video.filename or "video")[1] or ".mp4"
        data = await video.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            tmp.write(data)
            tmp.close()
            vsf = _get_video_safety_filter()
            result = vsf.check_video_file(os.path.abspath(tmp.name))
            return VideoSafetyResponse(
                blocked=result.get("blocked", False),
                audio_level=result.get("audio_result", {}).get("level", 0),
                audio_keywords=result.get("audio_result", {}).get("keywords", []),
                audio_transcript=result.get("audio_result", {}).get("transcript", "") or "",
                video_level=result.get("video_result", {}).get("level", 0),
                video_violations=result.get("video_result", {}).get("violations", []),
                processing_time=result.get("processing_time", 0.0),
                error=result.get("error") or "",
                audio_error=result.get("audio_result", {}).get("error") or "",
                video_error=result.get("video_result", {}).get("error") or "",
            )
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("safety_check_video failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"视频安全检测失败: {exc}")


# ==================== 紧急推送 API ====================

class EmergencyPushRequest(BaseModel):
    session_id: str = Field(..., description="会话标识")
    matched_terms: List[str] = Field(..., description="命中的敏感词列表")
    user_text: str = Field(default="", description="用户原始输入文本")
    crisis_type: Optional[str] = Field(default=None, description="危机类型（可选，不填则自动分类）")


class EmergencyPushResponse(BaseModel):
    triggered: bool
    session_id: str
    crisis_type: str
    matched_terms: List[str]
    template_title: str = ""
    template: str = ""
    rescue_api_called: bool = False
    rescue_api_result: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    timestamp: str = ""


class EmergencyPushHistoryResponse(BaseModel):
    history: Dict[str, Any]
    push_count: int


@router.post("/safety/emergency-push", response_model=EmergencyPushResponse)
async def trigger_emergency_push(body: EmergencyPushRequest, user_id: str = Depends(get_current_user_id)):
    """手动触发紧急推送（测试/管理用）。

    当高危关键词被检测到时，返回对应的危机话术模板，
    并模拟调用 120 救助 API。同一 session 在冷却期内不重复推送。
    """
    try:
        from modules.safety.emergency_push import get_emergency_push_service

        eps = get_emergency_push_service()
        result = eps.trigger(
            session_id=body.session_id,
            matched_terms=body.matched_terms,
            user_text=body.user_text,
            crisis_type=body.crisis_type,
        )
        return EmergencyPushResponse(**result.to_dict())
    except Exception as exc:
        logger.error("emergency_push failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"紧急推送失败: {exc}")


@router.get("/safety/emergency-push/history", response_model=EmergencyPushHistoryResponse)
async def get_emergency_push_history(session_id: Optional[str] = None, user_id: str = Depends(get_current_user_id)):
    """查询紧急推送触发历史。

    不传 session_id 则返回所有会话的触发时间；
    传入 session_id 则返回该会话的触发状态与冷却剩余时间。
    """
    try:
        from modules.safety.emergency_push import get_emergency_push_service

        eps = get_emergency_push_service()
        history = eps.get_history(session_id=session_id)
        return EmergencyPushHistoryResponse(
            history=history,
            push_count=len(eps.trigger_history),
        )
    except Exception as exc:
        logger.error("get_emergency_push_history failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"查询历史失败: {exc}")


@router.delete("/safety/emergency-push/reset")
async def reset_emergency_push_cooldown(session_id: str, user_id: str = Depends(get_current_user_id)):
    """重置某会话的紧急推送冷却期，使其可立即再次触发（管理操作用）。"""
    try:
        from modules.safety.emergency_push import get_emergency_push_service

        eps = get_emergency_push_service()
        ok = eps.reset_cooldown(session_id)
        return {
            "success": ok,
            "session_id": session_id,
            "message": "冷却期已重置" if ok else "该会话无冷却记录",
        }
    except Exception as exc:
        logger.error("reset_emergency_push_cooldown failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"重置冷却期失败: {exc}")


# ==================== 人审接口 ====================

class SafetyFlagItem(BaseModel):
    id: int
    user_id: str
    session_id: str
    level: int
    blocked: bool
    matched_terms: List[str]
    reviewed: bool
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    created_at: Optional[str] = None


class ReviewRequest(BaseModel):
    reviewed_by: str = Field(default="admin", description="审核人标识")


@router.get("/safety/flags/pending", response_model=List[SafetyFlagItem])
async def list_pending_safety_flags(
    user_id_filter: Optional[str] = Query(default=None, description="按用户 ID 过滤"),
    limit: int = Query(default=50, ge=1, le=200),
    user_id: str = Depends(get_current_user_id),
):
    """列出待人审的安全标记（管理员接口）。"""
    try:
        from modules.safety.flag_recorder import SafetyFlagRecorder

        recorder = SafetyFlagRecorder()
        flags = recorder.list_pending_review(user_id=user_id_filter, limit=limit)
        return [SafetyFlagItem(**f) for f in flags]
    except Exception as exc:
        logger.error("list_pending_safety_flags failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"查询失败: {exc}")


@router.post("/safety/flags/{flag_id}/review")
async def mark_safety_flag_reviewed(
    flag_id: int,
    body: ReviewRequest = ReviewRequest(),
    user_id: str = Depends(get_current_user_id),
):
    """将一条安全标记标记为"已人审"（管理员接口）。"""
    try:
        from modules.safety.flag_recorder import SafetyFlagRecorder

        recorder = SafetyFlagRecorder()
        ok = recorder.mark_reviewed(flag_id, reviewed_by=body.reviewed_by)
        return {"success": ok, "flag_id": flag_id}
    except Exception as exc:
        logger.error("mark_safety_flag_reviewed failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"审核失败: {exc}")
