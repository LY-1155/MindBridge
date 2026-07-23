"""
四模块流水线 HTTP 入口（便于并行开发联调）
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from modules.auth_deps import get_current_user_id
from modules.file_upload_security import validate_upload
from modules.ai_disclaimer import apply_disclaimer

from pipeline.orchestrator import run_pipeline, run_video_pipeline
from schemas.contracts import PipelineInput, PipelineOutput

router = APIRouter(prefix="/api/v1/pipeline", tags=["pipeline"])


@router.post("/run", response_model=PipelineOutput)
def pipeline_run(body: PipelineInput, user_id: str = Depends(get_current_user_id)) -> PipelineOutput:
    """纯文本端到端管线。"""
    body.user_id = user_id
    output = run_pipeline(body)
    output.intervention["reply"] = apply_disclaimer(output.intervention.get("reply", ""))
    return output


@router.post("/run-with-audio", response_model=PipelineOutput)
async def pipeline_run_with_audio(
    audio: UploadFile = File(..., description="音频文件"),
    text: Optional[str] = Form(default=None, description="可选补充文本"),
    session_id: Optional[str] = Form(default=None, description="会话 ID"),
    user_id: str = Depends(get_current_user_id),
) -> PipelineOutput:
    """音频端到端管线：一次 SenseVoice 调用 → 文本 + 情绪 → 管线。"""
    await validate_upload(audio, category="audio")
    audio_bytes = await audio.read()
    suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = os.path.abspath(tmp.name)
    try:
        tmp.write(audio_bytes)
        tmp.close()

        # 一次 SenseVoice 调用，同时拿文本和情绪
        transcribed_text = ""
        pre_extracted_emotion = None
        try:
            from funasr import AutoModel
            from config.settings import settings

            model = AutoModel(
                model=settings.SENSEVOICE_ASR_MODEL,
                trust_remote_code=True,
                device=settings.SENSEVOICE_DEVICE,
            )
            raw_output = model.generate(input=temp_path, use_itn=True)
            raw_text = str(raw_output[0] if isinstance(raw_output, list) and raw_output else raw_output)

            # 提取文本
            from multimodal.asr import _clean_sensevoice_text
            transcribed_text = _clean_sensevoice_text(raw_text) or text or ""

            # 提取情绪标签
            from multimodal.audio_emotion import AudioEmotionRecognizer
            recognizer = AudioEmotionRecognizer(backend="sensevoice")
            emotion_tag = recognizer._extract_tagged_emotion(raw_text)
            if emotion_tag:
                pre_extracted_emotion = {
                    "primary_emotion": emotion_tag,
                    "confidence": 0.72 if emotion_tag != "neutral" else 0.6,
                    "all_emotions": recognizer._distribution_from_primary(emotion_tag, 0.72),
                    "model_name": "SenseVoiceSmall",
                    "backend": "sensevoice",
                }
        except Exception:
            # SenseVoice 不可用 → 回退纯文本
            transcribed_text = text or ""

        inp = PipelineInput(
            text=transcribed_text,
            user_id=user_id,
            pre_extracted_audio_emotion=pre_extracted_emotion,
            session_id=session_id,
        )
        result = run_pipeline(inp)
        result.intervention["reply"] = apply_disclaimer(result.intervention.get("reply", ""))
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"音频管线失败: {exc}")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@router.post("/run-with-video", response_model=PipelineOutput)
async def pipeline_run_with_video(
    video: UploadFile = File(..., description="视频文件（mp4/avi/mov 等）"),
    audio: Optional[UploadFile] = File(default=None, description="可选独立音频文件"),
    safety_text: Optional[str] = Form(default="", description="可选安全检测文本（为空则用 ASR 产出）"),
    session_id: Optional[str] = Form(default=None, description="会话 ID"),
    user_id: str = Depends(get_current_user_id),
) -> PipelineOutput:
    """视频端到端管线：预处理 → 安全 → 三模态情绪融合 → 路由 → 干预。

    与 /run（纯文本）和 /run-with-audio（音频）的区别：
    - 上传视频文件，自动完成 ASR + 音频情绪 + 视觉情绪三路信号提取
    - 可选的独立音频文件（分离上传场景）
    - 三模态信号冲突时自动仲裁，mixed_signals 触发风险加分
    """
    # 保存视频临时文件
    await validate_upload(video, category="video")
    video_bytes = await video.read()
    video_suffix = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    video_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=video_suffix)
    video_path = os.path.abspath(video_tmp.name)

    # 可选：保存独立音频临时文件
    audio_path = None
    audio_tmp_path = None
    if audio:
        await validate_upload(audio, category="audio")
        audio_bytes = await audio.read()
        audio_suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
        audio_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=audio_suffix)
        audio_tmp_path = os.path.abspath(audio_tmp.name)
        audio_tmp.write(audio_bytes)
        audio_tmp.close()
        audio_path = audio_tmp_path

    try:
        video_tmp.write(video_bytes)
        video_tmp.close()

        result = run_video_pipeline(
            video_path=video_path,
            audio_path=audio_path,
            safety_text=safety_text or "",
            session_id=session_id,
            user_id=user_id,
        )
        result.intervention["reply"] = apply_disclaimer(result.intervention.get("reply", ""))
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"视频管线失败: {exc}")
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)
        if audio_path and audio_tmp_path and os.path.exists(audio_tmp_path):
            os.unlink(audio_tmp_path)
