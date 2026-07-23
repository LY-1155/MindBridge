"""
画面违规检测模块 - CLIP 零样本分类
功能：检测视频帧中的违规内容（色情、暴力、血腥、自残等）
"""

import os
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV未安装，请运行: pip install opencv-python")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers未安装，请运行: pip install transformers")

VIOLATION_TYPES = {
    "nsfw":      {"name": "色情内容", "level": 1, "description": "裸露、性行为等内容"},
    "violence":  {"name": "暴力内容", "level": 1, "description": "打斗、攻击等暴力场景"},
    "gore":      {"name": "血腥内容", "level": 1, "description": "血腥、残肢等极端内容"},
    "self_harm": {"name": "自残内容", "level": 1, "description": "自残、自杀相关画面"},
    "drugs":     {"name": "毒品相关", "level": 2, "description": "毒品、吸毒工具等"},
}

CLIP_CANDIDATE_LABELS = {
    "violence":  "violence, fighting, physical attacks or weapons",
    "gore":      "blood, gore, dismemberment, severe injuries, dead bodies",
    "self_harm": "self-harm, suicide, self-inflicted wounds, cutting scars",
    "drugs":     "drugs, drug paraphernalia, injecting, smoking pipes",
    "normal":    "a normal photo without any harmful, violent, or disturbing content",
}

CLIP_LABEL_TO_TYPE = {v: k for k, v in CLIP_CANDIDATE_LABELS.items()}


class FrameDetector:
    """
    画面违规检测器
    基于 CLIP 零样本分类，覆盖色情/暴力/血腥/自残/毒品
    """

    def __init__(
        self,
        device: str = "cuda",
        confidence_threshold: float = 0.75,
        local_model_path: str = None,
        nsfw_model_path: str = None,
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.local_model_path = local_model_path
        self.nsfw_model_path = nsfw_model_path
        self._classifier = None
        self._cv2_available = CV2_AVAILABLE

        self.evidence_dir = Path(_PROJECT_ROOT) / "evidence"
        self.evidence_dir.mkdir(exist_ok=True)

        self.detection_mode = None  # "local" 或 "online"

        logger.info("画面检测器初始化完成, 置信度阈值: %s", confidence_threshold)

    def _load_classifier(self):
        """延迟加载 CLIP 零样本分类器"""
        resolved_path = None
        if self.local_model_path:
            resolved_path = self.local_model_path
            if not os.path.isabs(resolved_path):
                resolved_path = os.path.join(_PROJECT_ROOT, resolved_path)

        if resolved_path and os.path.exists(resolved_path):
            model_source = resolved_path
            self.detection_mode = "local"
            logger.info("使用本地CLIP模型: %s", resolved_path)
        elif TRANSFORMERS_AVAILABLE:
            model_source = "openai/clip-vit-base-patch16"
            self.detection_mode = "online"
            logger.info("使用在线CLIP模型: %s", model_source)
        else:
            raise RuntimeError("无法加载CLIP模型：本地模型不存在且未安装Transformers")

        try:
            import torch
            device_index = 0 if self.device == "cuda" and self._is_cuda_available() else -1

            self._classifier = pipeline(
                "zero-shot-image-classification",
                model=model_source,
                device=device_index
            )
            logger.info("CLIP模型加载完成 (模式: %s, 设备: %s)", self.detection_mode, self.device)
        except Exception as e:
            logger.error("CLIP模型加载失败: %s", e)
            raise

    def _is_cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    @property
    def classifier(self):
        if self._classifier is None:
            self._load_classifier()
        return self._classifier

    def detect_frame(self, frame) -> Dict:
        """
        检测单帧画面的违规内容
        使用 SigLIP2 专用模型检测 NSFW + CLIP 零样本检测暴力/血腥/自残/毒品
        """
        if not TRANSFORMERS_AVAILABLE:
            return {"error": "Transformers未安装", "violations": [], "level": 0}

        if not CV2_AVAILABLE:
            return {"error": "OpenCV未安装", "violations": [], "level": 0}

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = np_to_pil(frame_rgb)

            # Tier 1: SigLIP2 专用 NSFW 检测
            nsfw_result = self._detect_nsfw(pil_image)

            # Tier 2: CLIP 零样本检测 violence/gore/self_harm/drugs
            clip_result = self._detect_with_clip(pil_image)

            # 合并结果
            violations = nsfw_result.get("violations", []) + clip_result.get("violations", [])
            max_level = max(nsfw_result.get("level", 0), clip_result.get("level", 0))
            combined_probs = {}
            combined_probs.update(nsfw_result.get("all_probs", {}))
            combined_probs.update(clip_result.get("all_probs", {}))

            return {
                "violations": violations,
                "level": max_level,
                "all_probs": combined_probs,
                "detection_mode": "siglip2+clip",
            }
        except Exception as e:
            logger.error("帧检测失败: %s", e)
            return {"error": str(e), "violations": [], "level": 0}

    def _detect_nsfw(self, pil_image) -> Dict:
        """使用 SigLIP2 专用模型检测 NSFW"""
        try:
            from modules.safety.multimodal.nsfw_detector_siglip import SiglipNsfwDetector
        except ImportError:
            try:
                from nsfw_detector_siglip import SiglipNsfwDetector
            except ImportError:
                logger.warning("SigLIP2 NSFW 检测器不可用，跳过 nsfw 检测")
                return {"violations": [], "level": 0, "all_probs": {}}

        if not hasattr(self, "_nsfw_detector"):
            self._nsfw_detector = SiglipNsfwDetector(
                model_path=self.nsfw_model_path,
                device=self.device,
            )
        return self._nsfw_detector.detect(pil_image)

    def _detect_with_clip(self, pil_image) -> Dict:
        """CLIP 零样本分类检测 violence/gore/self_harm/drugs（nsfw 已交由 SigLIP2）"""
        candidate_labels = list(CLIP_CANDIDATE_LABELS.values())
        results = self.classifier(pil_image, candidate_labels=candidate_labels)

        violations = []
        max_level = 0
        all_probs = {}

        # 第一遍：收集所有概率
        for result in results:
            label_text = result.get("label", "")
            score = result.get("score", 0)
            violation_type = CLIP_LABEL_TO_TYPE.get(label_text, "unknown")
            all_probs[violation_type] = round(score, 4)

        normal_score = all_probs.get("normal", 0)

        # 第二遍：判定有害标签
        for result in results:
            label_text = result.get("label", "")
            score = result.get("score", 0)
            violation_type = CLIP_LABEL_TO_TYPE.get(label_text, "unknown")

            if violation_type == "normal":
                continue

            # 双重条件：超过阈值 且 超过 normal 得分
            if score >= self.confidence_threshold and score > normal_score:
                vtype_info = VIOLATION_TYPES.get(violation_type, {})
                violations.append({
                    "type": violation_type,
                    "name": vtype_info.get("name", violation_type),
                    "confidence": round(score, 4),
                    "level": vtype_info.get("level", 1)
                })
                max_level = max(max_level, vtype_info.get("level", 1))

        return {
            "violations": violations,
            "level": max_level,
            "all_probs": all_probs,
            "detection_mode": self.detection_mode,
        }

    def detect_frame_file(self, image_path: str) -> Dict:
        """检测图片文件的违规内容"""
        if not CV2_AVAILABLE:
            return {"error": "OpenCV未安装", "violations": [], "level": 0}

        frame = cv2.imread(image_path)
        if frame is None:
            return {"error": f"无法读取图片: {image_path}", "violations": [], "level": 0}

        return self.detect_frame(frame)

    def save_frame_evidence(
        self,
        frame,
        violation_type: str,
        timestamp: float = None
    ) -> str:
        """保存违规帧作为证据"""
        if not CV2_AVAILABLE:
            logger.error("OpenCV未安装，无法保存帧证据")
            return None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if timestamp is not None:
            ts = f"{ts}_{timestamp:.2f}s"

        evidence_path = self.evidence_dir / f"{ts}_{violation_type}.jpg"
        _, buf = cv2.imencode(os.path.splitext(str(evidence_path))[1], frame)
        buf.tofile(str(evidence_path))
        logger.info("保存违规帧证据: %s", evidence_path)

        return str(evidence_path)


def np_to_pil(np_image):
    """将numpy数组转换为PIL图像"""
    from PIL import Image
    return Image.fromarray(np_image)


class VideoFrameExtractor:
    """视频帧提取器"""

    def __init__(self, fps: float = 1.0):
        self.fps = fps
        self._cv2_available = CV2_AVAILABLE

    def extract_frames(
        self,
        video_path: str,
        max_frames: int = None
    ):
        """从视频中均匀抽帧"""
        frames = []
        timestamps = []

        if not self._cv2_available:
            logger.error("OpenCV未安装，无法提取视频帧")
            return frames, timestamps

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("无法打开视频: %s", video_path)
            return frames, timestamps

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_interval = int(video_fps / self.fps)
        if frame_interval < 1:
            frame_interval = 1

        logger.info("视频FPS: %s, 总帧数: %s, 抽帧间隔: %s", video_fps, total_frames, frame_interval)

        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                timestamp = frame_count / video_fps
                frames.append(frame)
                timestamps.append(timestamp)
                extracted_count += 1

                if max_frames and extracted_count >= max_frames:
                    break

            frame_count += 1

        cap.release()
        logger.info("抽取了 %d 帧", len(frames))

        return frames, timestamps

    def extract_frames_generator(
        self,
        video_path: str,
        max_frames: int = None
    ):
        """生成器模式抽帧（用于实时流处理）"""
        if not self._cv2_available:
            logger.error("OpenCV未安装，无法提取视频帧")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("无法打开视频流: %s", video_path)
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 25

        frame_interval = int(video_fps / self.fps)
        if frame_interval < 1:
            frame_interval = 1

        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                timestamp = frame_count / video_fps
                yield frame, timestamp
                extracted_count += 1

                if max_frames and extracted_count >= max_frames:
                    break

            frame_count += 1

        cap.release()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("画面违规检测器测试 (CLIP)")
    print("=" * 60)

    print("\n依赖检查:")
    print("  OpenCV可用: %s" % ('是' if CV2_AVAILABLE else '否'))
    print("  Transformers可用: %s" % ('是' if TRANSFORMERS_AVAILABLE else '否'))

    if not TRANSFORMERS_AVAILABLE:
        print("\n请先安装Transformers: pip install transformers")
        sys.exit(1)

    local_model_path = input("\n输入本地CLIP模型路径（回车使用在线 openai/clip-vit-base-patch16）: ").strip()

    try:
        if local_model_path and os.path.exists(local_model_path):
            print("\n使用本地CLIP模型: %s" % local_model_path)
            detector = FrameDetector(
                device="cpu",
                confidence_threshold=0.75,
                local_model_path=local_model_path,
            )
        elif local_model_path:
            print("\n路径不存在，退回在线模式: openai/clip-vit-base-patch16")
            detector = FrameDetector(device="cpu", confidence_threshold=0.5)
        else:
            print("\n使用在线模型: openai/clip-vit-base-patch16")
            detector = FrameDetector(device="cpu", confidence_threshold=0.5)

        test_image = input("\n输入测试图片路径（回车跳过）: ").strip()
        if test_image and os.path.exists(test_image):
            print("\n正在检测...")
            result = detector.detect_frame_file(test_image)
            print("\n结果:")
            print("  等级: %s" % result.get('level', 0))
            print("  违规类型: %s" % result.get('violations', []))
            if 'all_probs' in result:
                print("  检测概率: %s" % result['all_probs'])
            if 'error' in result:
                print("  错误: %s" % result['error'])
        else:
            print("跳过图片测试")

    except Exception as e:
        print("\n初始化失败: %s" % e)
        print("\n可能的解决方案：")
        print("1. 确保网络可以访问Hugging Face")
        print("2. 手动下载模型到本地并指定路径")
        print("3. 检查是否正确安装了transformers库")
