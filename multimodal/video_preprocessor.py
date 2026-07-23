"""
Video Preprocessor — 视频管线前置处理模块。

将视频文件解构为管线可用的三路信号：ASR 文本、音频情绪、视觉情绪。
重构自 pipeline_emotion_llama.py，升级为多帧分析 + 时序加权聚合。
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from multimodal.emotion_label_mapping import map_ekman_to_contract

logger = logging.getLogger(__name__)

# 契约 8 类情绪标签
_SUPPORTED_EMOTIONS = (
    "neutral", "anxiety", "sadness", "anger",
    "fear", "stress", "happiness", "confusion",
)

# 抽帧参数
_MAX_FRAMES = 20
_FACE_DETECTION_MIN_RATE = 0.3  # 人脸检测率低于此阈值整段降级
_MIN_VALID_FRAMES = 3


@dataclass
class PreprocessResult:
    """视频预处理的输出：三路信号 + 警告信息。"""
    text: str
    audio_emotion: Optional[Dict[str, Any]]
    visual_emotion: Optional[Dict[str, Any]]
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ffmpeg utilities (from pipeline_emotion_llama.py)
# ---------------------------------------------------------------------------

def _which_ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def _extract_wav_from_video(video_path: str) -> str:
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    ffmpeg = _which_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found; cannot extract audio from video")
    cmd = [
        ffmpeg, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if os.path.exists(out):
            os.unlink(out)
        raise RuntimeError(proc.stderr[:500] if proc.stderr else "ffmpeg failed")
    return out


def _sample_frames(video_path: str, max_frames: int = _MAX_FRAMES) -> List[np.ndarray]:
    """从视频中自适应抽帧。返回 BGR ndarray 列表。

    采用顺序读取而非 seek（某些 codec 不支持 CAP_PROP_POS_FRAMES）。
    最短采样 3 帧，确保降级逻辑有足够数据判断。
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    duration = total_frames / max(fps, 1)
    target_count = max(3, min(max_frames, int(duration)))

    all_frames: List[np.ndarray] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        all_frames.append(frame)
        frame_idx += 1
    cap.release()

    if not all_frames:
        return []

    # 均匀抽样
    n = len(all_frames)
    if n <= target_count:
        return all_frames

    indices = [int(i * (n - 1) / (target_count - 1)) for i in range(target_count)]
    return [all_frames[i] for i in indices]


# ---------------------------------------------------------------------------
# 时序加权聚合
# ---------------------------------------------------------------------------

def _temporal_weighted_aggregate(
    frame_results: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """对多帧情绪结果做时序加权聚合（靠后帧权重更高）。"""
    valid = [r for r in frame_results if r is not None]
    if not valid:
        return None

    n = len(valid)
    totals: Dict[str, float] = {e: 0.0 for e in _SUPPORTED_EMOTIONS}
    total_weight = 0.0

    for idx, result in enumerate(valid):
        # 越靠后的帧权重越高（线性递增）
        weight = (idx + 1) / n
        total_weight += weight
        all_emotions = result.get("all_emotions", {})
        for emotion in _SUPPORTED_EMOTIONS:
            totals[emotion] += all_emotions.get(emotion, 0.0) * weight

    if total_weight > 0:
        totals = {e: s / total_weight for e, s in totals.items()}

    primary, confidence = max(totals.items(), key=lambda item: item[1])

    return {
        "primary_emotion": primary,
        "confidence": round(confidence, 4),
        "all_emotions": {e: round(s, 4) for e, s in totals.items()},
    }


# ---------------------------------------------------------------------------
# VideoPreprocessor
# ---------------------------------------------------------------------------

class VideoPreprocessor:
    """将视频解构为 text + audio_emotion + visual_emotion 三路信号。

    通过依赖注入实现可测试性：asr_fn, audio_emotion_fn, face_detector,
    emotion_model 均可被 mock，不传则使用生产实现。
    """

    def __init__(
        self,
        asr_fn: Optional[Callable[[str], str]] = None,
        audio_emotion_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        face_detector: Optional[Callable[[np.ndarray], list]] = None,
        emotion_model: Optional[Callable[[np.ndarray], Dict[str, Any]]] = None,
    ):
        self._asr_fn = asr_fn
        self._audio_emotion_fn = audio_emotion_fn
        self._face_detector = face_detector
        self._emotion_model = emotion_model

    def process(self, video_path: str, audio_path: Optional[str] = None) -> PreprocessResult:
        """主入口：输入视频（及可选独立音频），输出 PreprocessResult。"""
        warnings: List[str] = []

        # 1. 准备音频
        wav_path: Optional[str] = None
        own_wav = False
        if audio_path and os.path.exists(audio_path):
            wav_path = audio_path
        else:
            try:
                wav_path = _extract_wav_from_video(video_path)
                own_wav = True
            except RuntimeError as exc:
                warnings.append(f"audio extraction failed: {exc}")
                wav_path = None

        # 2. ASR → text
        text = self._run_asr(wav_path) if wav_path else ""
        if not text:
            warnings.append("ASR produced empty text; pipeline may degrade.")

        # 3. 音频情绪
        audio_emotion = None
        if wav_path:
            try:
                audio_emotion = self._run_audio_emotion(wav_path)
            except Exception as exc:
                warnings.append(f"audio emotion failed: {exc}")

        # 4. 视觉情绪（多帧抽帧 + 推理 + 聚合）
        visual_emotion = self._process_visual(video_path, warnings)

        # 清理临时文件
        if own_wav and wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except OSError:
                pass

        return PreprocessResult(
            text=text,
            audio_emotion=audio_emotion,
            visual_emotion=visual_emotion,
            warnings=warnings,
        )

    # ---- 内部方法 ---------------------------------------------------------

    def _run_asr(self, wav_path: str) -> str:
        if self._asr_fn:
            return self._asr_fn(wav_path)
        return _transcribe_path_sync(wav_path)

    def _run_audio_emotion(self, wav_path: str) -> Dict[str, Any]:
        if self._audio_emotion_fn:
            return self._audio_emotion_fn(wav_path)
        return _audio_emotion_sync(wav_path)

    def _process_visual(
        self, video_path: str, warnings: List[str],
    ) -> Optional[Dict[str, Any]]:
        """抽样多帧 → 逐帧推理 → 时序加权聚合。"""
        try:
            frames = _sample_frames(video_path)
        except Exception as exc:
            warnings.append(f"frame sampling failed: {exc}")
            return None

        if not frames:
            warnings.append("no frames extracted from video; visual signal unavailable.")
            return None

        total_frames = len(frames)
        frame_results: List[Optional[Dict[str, Any]]] = []
        face_detected_count = 0

        for frame in frames:
            result = self._process_single_frame(frame)
            if result:
                frame_results.append(result)
                face_detected_count += 1
            else:
                frame_results.append(None)

        # 降级判断
        face_rate = face_detected_count / total_frames if total_frames > 0 else 0
        if face_detected_count < _MIN_VALID_FRAMES or face_rate < _FACE_DETECTION_MIN_RATE:
            warnings.append(
                f"visual signal degraded: only {face_detected_count}/{total_frames} "
                f"frames ({face_rate:.0%}) had detectable faces; required >={_FACE_DETECTION_MIN_RATE:.0%}"
            )
            return None

        aggregated = _temporal_weighted_aggregate(
            [r for r in frame_results if r is not None]
        )
        if aggregated is None:
            return None

        aggregated["face_detection_rate"] = round(face_rate, 2)
        aggregated["valid_frames"] = face_detected_count
        aggregated["total_frames"] = total_frames
        aggregated["model_name"] = "HSEmotion"
        return aggregated

    def _process_single_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """单帧处理：人脸检测 → 情绪推理 → Ekman→契约标签映射。"""
        try:
            faces = self._detect_faces(frame)
        except Exception:
            return None

        if not faces:
            return None

        # 取最大的人脸
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_img = frame[y:y + h, x:x + w]

        try:
            raw_emotion = self._run_emotion_model(face_img)
        except Exception:
            return None

        ekman_label = raw_emotion.get("primary_emotion", "neutral")
        ekman_confidence = float(raw_emotion.get("confidence", 0.5))
        ekman_all = raw_emotion.get("all_emotions", {})

        # 标签映射
        contract_label, adjusted_confidence = map_ekman_to_contract(ekman_label, ekman_confidence)

        # 将 Ekman 分布映射为契约 8 类分布
        contract_dist: Dict[str, float] = {e: 0.0 for e in _SUPPORTED_EMOTIONS}
        for ekman_e, prob in ekman_all.items():
            mapped_label, mapped_prob = map_ekman_to_contract(ekman_e, prob)
            contract_dist[mapped_label] = contract_dist.get(mapped_label, 0.0) + mapped_prob

        # 归一化
        total = sum(contract_dist.values())
        if total > 0:
            contract_dist = {e: v / total for e, v in contract_dist.items()}

        return {
            "primary_emotion": contract_label,
            "confidence": round(adjusted_confidence, 4),
            "all_emotions": contract_dist,
        }

    def _detect_faces(self, frame: np.ndarray) -> list:
        if self._face_detector:
            return self._face_detector(frame)
        return _detect_faces_with_mediapipe(frame)

    def _run_emotion_model(self, face_img: np.ndarray) -> Dict[str, Any]:
        if self._emotion_model:
            return self._emotion_model(face_img)
        return _run_hsemotion(face_img)


# ---------------------------------------------------------------------------
# 生产实现（当没有注入 mock 时使用）
# ---------------------------------------------------------------------------

def _transcribe_path_sync(wav_path: str) -> str:
    """使用 SenseVoice 做 ASR。"""
    from multimodal.asr import get_speech_recognizer
    recognizer = get_speech_recognizer()
    result = recognizer.transcribe(wav_path)
    return result.text if hasattr(result, "text") else str(result)


def _audio_emotion_sync(wav_path: str) -> Dict[str, Any]:
    """从音频提取情绪信号。"""
    from multimodal.audio_emotion import get_audio_emotion_recognizer
    recognizer = get_audio_emotion_recognizer()
    transcript = _transcribe_path_sync(wav_path)
    result = recognizer.recognize(audio_path=wav_path, transcript=transcript)
    return {
        "primary_emotion": result.primary_emotion,
        "confidence": result.confidence,
        "all_emotions": result.all_emotions,
        "model_name": result.model_name,
        "backend": result.backend,
    }


def _detect_faces_with_mediapipe(frame: np.ndarray) -> list:
    """使用 MediaPipe 做面部检测。"""
    import cv2
    try:
        import mediapipe as mp
        mp_face = mp.solutions.face_detection
        with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            if not results.detections:
                return []
            h, w = frame.shape[:2]
            boxes = []
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                boxes.append((x, y, bw, bh))
            return boxes
    except ImportError:
        # 回退到 OpenCV Haar
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return list(faces)
    except Exception:
        return []


def _run_hsemotion(face_img: np.ndarray) -> Dict[str, Any]:
    """使用 HSEmotion 做面部情绪推理。回退到 transformers ViT。"""
    import cv2

    # 尝试 HSEmotion
    try:
        import torch

        # PyTorch 2.6+ 默认 weights_only=True，但 HSEmotion 的模型文件
        # 包含大量 timm 类无法逐个白名单。临时恢复旧行为。
        _original_load = torch.load
        torch.load = lambda *a, **kw: _original_load(*a, **{**kw, "weights_only": False})

        from hsemotion.facial_emotions import HSEmotionRecognizer
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        recognizer = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")
        torch.load = _original_load
        emotion, scores = recognizer.predict_emotions(face_rgb, logits=False)
        # HSEmotion 返回的 emotions 顺序: ['Anger','Disgust','Fear','Happiness','Neutral','Sadness','Surprise']
        hse_labels = ["angry", "disgust", "fear", "happiness", "neutral", "sad", "surprise"]
        all_emotions = dict(zip(hse_labels, [float(s) for s in scores]))
        return {
            "primary_emotion": emotion.lower(),
            "confidence": float(max(scores)),
            "all_emotions": all_emotions,
        }
    except ImportError:
        logger.warning("HSEmotion not installed, trying ViT fallback")
    except Exception as exc:
        logger.warning("HSEmotion inference failed, trying fallback: %s", exc)

    # 回退到 transformers ViT（现有的 EmotionRecognizer）
    from multimodal.emotion import get_emotion_recognizer
    recognizer = get_emotion_recognizer(use_deep_model=True)
    result = recognizer._analyze_image(face_img)
    return {
        "primary_emotion": result.primary_emotion,
        "confidence": result.confidence,
        "all_emotions": result.all_emotions,
    }
