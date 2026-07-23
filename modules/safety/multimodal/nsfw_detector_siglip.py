"""
SigLIP2 NSFW 专用检测器
基于 prithivMLmods/siglip2-x256-explicit-content 微调模型
专用于色情/擦边内容检测，替代 CLIP 零样本的 nsfw 判定
"""

import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 模型 5 类标签
SIGLIP_LABELS = {
    0: "Anime Picture",
    1: "Hentai",
    2: "Normal",
    3: "Pornography",
    4: "Enticing or Sensual",
}

# 需判为 NSFW 的类别及各自的阈值
NSFW_CLASSES = {
    "Pornography": 0.5,
    "Hentai": 0.5,
    "Enticing or Sensual": 0.7,  # 擦边内容提高阈值，减少误判
}


class SiglipNsfwDetector:
    """基于 SigLIP2 的 NSFW 内容检测器"""

    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda",
        nsfw_thresholds: Dict[str, float] = None,
    ):
        self.model_path = model_path
        self.device = device
        self.nsfw_thresholds = nsfw_thresholds or NSFW_CLASSES
        self._model = None
        self._processor = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self):
        """延迟加载模型"""
        if self._model is not None:
            return

        from transformers import SiglipForImageClassification, AutoImageProcessor
        import torch

        resolved_path = None
        if self.model_path:
            if not os.path.isabs(self.model_path):
                resolved_path = os.path.join(_PROJECT_ROOT, self.model_path)
            else:
                resolved_path = self.model_path

        if resolved_path and os.path.isdir(resolved_path):
            model_source = resolved_path
            logger.info("加载本地 SigLIP2 NSFW 模型: %s", resolved_path)
        else:
            model_source = "prithivMLmods/siglip2-x256-explicit-content"
            logger.info("加载在线 SigLIP2 NSFW 模型: %s", model_source)

        device_map = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
        self._model = SiglipForImageClassification.from_pretrained(
            model_source,
        ).to(device_map)
        self._model.eval()

        self._processor = AutoImageProcessor.from_pretrained(model_source)
        logger.info("SigLIP2 NSFW 检测器加载完成 (device=%s)", device_map)

    def detect(self, pil_image) -> Dict:
        """
        检测图片中的 NSFW 内容

        Args:
            pil_image: PIL Image (RGB)

        Returns:
            {violations, level, all_probs, detection_mode} 格式
            兼容 FrameDetector._detect_with_clip 的返回格式
        """
        self._load_model()

        import torch
        import torch.nn.functional as F

        inputs = self._processor(images=pil_image, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).squeeze().cpu().tolist()

        # 构建 all_probs
        all_probs = {SIGLIP_LABELS[i]: round(p, 4) for i, p in enumerate(probs)}

        violations = []
        max_level = 0

        for label, threshold in self.nsfw_thresholds.items():
            label_prob = all_probs.get(label, 0.0)
            if label_prob >= threshold:
                violations.append({
                    "type": "nsfw",
                    "name": "色情内容",
                    "confidence": round(label_prob, 4),
                    "level": 1,
                    "subtype": label,
                })
                max_level = 1

        return {
            "violations": violations,
            "level": max_level,
            "all_probs": all_probs,
            "detection_mode": "siglip2",
        }

    def detect_from_frame(self, frame) -> Dict:
        """从 OpenCV BGR frame 检测"""
        import cv2
        from PIL import Image

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        return self.detect(pil_image)
