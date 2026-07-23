"""
四模块并行开发 · 独立 HTTP 接口

各路径仅负责本模块契约；便于单人负责单模块时用 Swagger 联调。
全流程串联请使用 `/api/v1/pipeline/run`。
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from modules.auth_deps import get_current_user_id
from modules.file_upload_security import validate_upload

from config.settings import settings
from modules.runtime import get_pipeline_services
from schemas.contracts import (
    EmotionAnalyzeRequest,
    EmotionTags,
    InterventionRequest,
    InterventionResult,
    RouteDecision,
    RouteRequest,
    SafetyCheckRequest,
    SafetyCheckResult,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/modules", tags=["parallel-modules"])


@router.post("/safety/check", response_model=SafetyCheckResult)
def module_safety_check(body: SafetyCheckRequest, user_id: str = Depends(get_current_user_id)) -> SafetyCheckResult:
    return get_pipeline_services(settings).safety.check(body)


@router.post("/emotion/analyze", response_model=EmotionTags)
def module_emotion_analyze(body: EmotionAnalyzeRequest, user_id: str = Depends(get_current_user_id)) -> EmotionTags:
    return get_pipeline_services(settings).emotion.analyze(body)


@router.post("/emotion/analyze-audio", response_model=EmotionTags)
async def module_emotion_analyze_audio(
    audio: UploadFile = File(..., description="音频文件（wav/mp3/m4a 等）"),
    text: Optional[str] = Form(default="", description="可选的文本转录"),
    user_id: str = Depends(get_current_user_id),
):
    """情绪分析 · 音频文件上传端。

    与 /emotion/analyze（JSON）的区别：
    - 客户端直接上传音频文件，无需先把文件放到服务器
    - 后台自动将文件存为临时文件，传给 EmotionService.analyze()
    - 分析完毕后清理临时文件
    """
    await validate_upload(audio, category="audio")
    suffix = os.path.splitext(audio.filename or "audio")[1] or ".wav"
    data = await audio.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.close()

        sample_safety = {"level": 0, "blocked": False, "matched_terms": [], "meta": {}, "contract_version": "1.2"}
        req = EmotionAnalyzeRequest(
            text=text or "",
            audio_path=os.path.abspath(tmp.name),
            safety=sample_safety,
        )
        return get_pipeline_services(settings).emotion.analyze(req)
    except Exception as exc:
        logger.error("analyze_audio failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"音频情绪分析失败: {exc}")
    finally:
        if os.path.exists(tmp.name):
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


@router.post("/emotion/analyze-video", response_model=EmotionTags)
async def module_emotion_analyze_video(
    video: UploadFile = File(..., description="视频文件（mp4/avi/mov 等）"),
    audio: Optional[UploadFile] = File(default=None, description="可选独立音频文件"),
    text: Optional[str] = Form(default="", description="可选补充文本"),
    user_id: str = Depends(get_current_user_id),
):
    """情绪分析 · 视频文件上传端。

    与 /emotion/analyze（JSON）和 /emotion/analyze-audio 的区别：
    - 上传视频文件，自动完成 ASR + 音频情绪 + 视觉情绪三路信号提取
    - 三路信号送入 EmotionService 做多模态融合
    - 可选的独立音频文件（分离上传场景）
    - 分析完毕后清理临时文件
    """
    from multimodal.video_preprocessor import VideoPreprocessor

    # 保存视频临时文件
    await validate_upload(video, category="video")
    video_suffix = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    video_data = await video.read()
    video_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=video_suffix)
    video_path = os.path.abspath(video_tmp.name)

    # 可选：保存独立音频临时文件
    audio_path = None
    audio_tmp_path = None
    if audio:
        await validate_upload(audio, category="audio")
        audio_suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
        audio_data = await audio.read()
        audio_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=audio_suffix)
        audio_tmp_path = os.path.abspath(audio_tmp.name)
        audio_tmp.write(audio_data)
        audio_tmp.close()
        audio_path = audio_tmp_path

    try:
        video_tmp.write(video_data)
        video_tmp.close()

        # 视频预处理：提取三路信号
        preprocessor = VideoPreprocessor()
        pre_result = preprocessor.process(video_path, audio_path=audio_path)

        sample_safety = {
            "level": 0, "blocked": False, "matched_terms": [], "meta": {},
            "contract_version": "1.2",
        }
        req = EmotionAnalyzeRequest(
            text=text or pre_result.text,
            audio_path=audio_path or video_path,
            pre_extracted_audio_emotion=pre_result.audio_emotion,
            pre_extracted_visual_emotion=pre_result.visual_emotion,
            safety=sample_safety,
        )
        return get_pipeline_services(settings).emotion.analyze(req)
    except Exception as exc:
        logger.error("analyze_video failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"视频情绪分析失败: {exc}")
    finally:
        if os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except OSError:
                pass
        if audio_path and audio_tmp_path and os.path.exists(audio_tmp_path):
            try:
                os.unlink(audio_tmp_path)
            except OSError:
                pass


@router.post("/router/route", response_model=RouteDecision)
def module_router_route(body: RouteRequest, user_id: str = Depends(get_current_user_id)) -> RouteDecision:
    return get_pipeline_services(settings).router.route(body)


@router.post("/intervention/run", response_model=InterventionResult)
def module_intervention_run(body: InterventionRequest, user_id: str = Depends(get_current_user_id)) -> InterventionResult:
    # 干预闭环单独调试入口（不经过 pipeline）。装配见 modules.runtime → factory.get_intervention_service。
    return get_pipeline_services(settings).intervention.intervene(body)
