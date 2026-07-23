"""
情绪识别模块
===========

从图像或视频中识别面部情绪。
支持多种情绪类型识别，用于心理咨询中的情绪分析。

主要功能：
1. 面部检测
2. 情绪分类
3. 表情强度分析
4. 视频帧提取和分析

情绪类型：
- happy: 开心
- sad: 悲伤
- angry: 愤怒
- fearful: 恐惧
- disgusted: 厌恶
- surprised: 惊讶
- neutral: 平静
"""

import os
import base64
import tempfile
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmotionResult:
    """
    情绪识别结果
    
    Attributes:
        primary_emotion: 主要情绪
        confidence: 置信度（0-1）
        all_emotions: 所有情绪的概率分布
        face_detected: 是否检测到人脸
        face_count: 检测到的人脸数量
    """
    primary_emotion: str
    confidence: float
    all_emotions: Dict[str, float]
    face_detected: bool
    face_count: int


class EmotionRecognizer:
    """
    情绪识别器
    
    使用深度学习模型进行面部情绪识别。
    支持图像和视频输入。
    
    使用示例：
        recognizer = EmotionRecognizer()
        result = recognizer.recognize_from_image("face.jpg")
        print(f"主要情绪: {result.primary_emotion}")
    """
    
    # 支持的情绪类型
    EMOTIONS = [
        "angry",      # 愤怒
        "disgust",    # 厌恶
        "fear",       # 恐惧
        "happy",      # 开心
        "sad",        # 悲伤
        "surprise",   # 惊讶
        "neutral"     # 平静
    ]
    
    # 情绪中文映射
    EMOTION_CN = {
        "angry": "愤怒",
        "disgust": "厌恶",
        "fear": "恐惧",
        "happy": "开心",
        "sad": "悲伤",
        "surprise": "惊讶",
        "neutral": "平静"
    }
    
    def __init__(self, use_deep_model: bool = False):
        """
        初始化情绪识别器
        
        Args:
            use_deep_model: 是否使用深度学习模型
                          True: 使用transformers模型（更准确，需要GPU）
                          False: 使用OpenCV Haar级联（快速，CPU友好）
        """
        self.use_deep_model = use_deep_model
        self._face_detector = None
        self._emotion_model = None
        
        logger.info(f"初始化情绪识别器: deep_model={use_deep_model}")
    
    @property
    def face_detector(self):
        """延迟加载人脸检测器"""
        if self._face_detector is None:
            import cv2
            # 使用OpenCV的Haar级联分类器
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._face_detector = cv2.CascadeClassifier(cascade_path)
            logger.info("人脸检测器加载完成")
        return self._face_detector
    
    @property
    def emotion_model(self):
        """延迟加载情绪识别模型"""
        if self._emotion_model is None and self.use_deep_model:
            try:
                from transformers import pipeline
                logger.info("加载情绪识别模型...")
                self._emotion_model = pipeline(
                    "image-classification",
                    model="trpakov/vit-facial-expression-recognition",
                    top_k=7
                )
                logger.info("情绪识别模型加载完成")
            except Exception as e:
                logger.warning(f"加载深度模型失败，使用简化模式: {e}")
                self.use_deep_model = False
        return self._emotion_model
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        检测图像中的人脸
        
        Args:
            image: OpenCV图像数组（BGR格式）
        
        Returns:
            List[Tuple]: 人脸边界框列表 (x, y, width, height)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return list(faces)
    
    def recognize_from_image(
        self,
        image_path: str,
        return_face_image: bool = False
    ) -> EmotionResult:
        """
        从图像文件识别情绪
        
        Args:
            image_path: 图像文件路径
            return_face_image: 是否返回人脸图像
        
        Returns:
            EmotionResult: 情绪识别结果
        """
        import cv2
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        return self._analyze_image(image, return_face_image)
    
    def recognize_from_bytes(
        self,
        image_bytes: bytes,
        return_face_image: bool = False
    ) -> EmotionResult:
        """
        从字节流识别情绪
        
        Args:
            image_bytes: 图像字节流
            return_face_image: 是否返回人脸图像
        
        Returns:
            EmotionResult: 情绪识别结果
        """
        import cv2
        
        # 解码图像
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("无法解码图像数据")
        
        return self._analyze_image(image, return_face_image)
    
    def recognize_from_base64(
        self,
        base64_string: str,
        return_face_image: bool = False
    ) -> EmotionResult:
        """
        从Base64字符串识别情绪
        
        Args:
            base64_string: Base64编码的图像字符串
            return_face_image: 是否返回人脸图像
        
        Returns:
            EmotionResult: 情绪识别结果
        """
        # 移除可能的data URL前缀
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_bytes = base64.b64decode(base64_string)
        return self.recognize_from_bytes(image_bytes, return_face_image)
    
    def _analyze_image(
        self,
        image: np.ndarray,
        return_face_image: bool = False
    ) -> EmotionResult:
        """
        分析图像中的情绪
        
        内部方法，处理实际的图像分析。
        """
        import cv2
        
        # 检测人脸
        faces = self.detect_faces(image)
        
        if len(faces) == 0:
            # 未检测到人脸
            return EmotionResult(
                primary_emotion="unknown",
                confidence=0.0,
                all_emotions={},
                face_detected=False,
                face_count=0
            )
        
        # 使用深度学习模型
        if self.use_deep_model and self.emotion_model is not None:
            return self._analyze_with_deep_model(image, faces)
        else:
            return self._analyze_with_heuristic(image, faces)
    
    def _analyze_with_deep_model(
        self,
        image: np.ndarray,
        faces: List[Tuple[int, int, int, int]]
    ) -> EmotionResult:
        """使用深度学习模型分析情绪"""
        import cv2
        from PIL import Image
        
        # 取最大的人脸
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_img = image[y:y+h, x:x+w]
        
        # 转换为PIL图像
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)
        
        # 情绪识别
        results = self.emotion_model(pil_image)
        
        # 整理结果
        emotions = {}
        for r in results:
            label = r['label'].lower()
            if label in self.EMOTIONS:
                emotions[label] = r['score']
            elif label in self.EMOTION_CN.values():
                # 中文标签转英文
                for en, cn in self.EMOTION_CN.items():
                    if cn == label:
                        emotions[en] = r['score']
        
        # 找出主要情绪
        if emotions:
            primary = max(emotions.items(), key=lambda x: x[1])
            primary_emotion = primary[0]
            confidence = primary[1]
        else:
            primary_emotion = "neutral"
            confidence = 0.5
            emotions = {"neutral": 0.5}
        
        return EmotionResult(
            primary_emotion=primary_emotion,
            confidence=confidence,
            all_emotions=emotions,
            face_detected=True,
            face_count=len(faces)
        )
    
    def _analyze_with_heuristic(
        self,
        image: np.ndarray,
        faces: List[Tuple[int, int, int, int]]
    ) -> EmotionResult:
        """
        使用启发式方法分析情绪
        
        当没有深度学习模型时，基于面部特征进行简单分析。
        这是一个简化版本，实际应用建议使用深度学习模型。
        """
        import cv2
        
        # 取最大的人脸
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_gray = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        
        # 简单的特征分析（这里使用简化的方法）
        # 实际应用中应该使用训练好的模型
        
        # 计算图像的一些统计特征
        mean_val = np.mean(face_gray)
        std_val = np.std(face_gray)
        
        # 基于简单规则的情绪估计（非常简化）
        # 这只是演示，实际需要训练好的模型
        emotions = {
            "neutral": 0.4,
            "happy": 0.2,
            "sad": 0.15,
            "angry": 0.1,
            "fear": 0.05,
            "surprise": 0.05,
            "disgust": 0.05
        }
        
        # 根据亮度调整（简化示例）
        if mean_val > 130:
            emotions["happy"] += 0.2
            emotions["neutral"] -= 0.1
        elif mean_val < 100:
            emotions["sad"] += 0.15
            emotions["neutral"] -= 0.1
        
        # 归一化
        total = sum(emotions.values())
        emotions = {k: v/total for k, v in emotions.items()}
        
        primary = max(emotions.items(), key=lambda x: x[1])
        
        return EmotionResult(
            primary_emotion=primary[0],
            confidence=primary[1],
            all_emotions=emotions,
            face_detected=True,
            face_count=len(faces)
        )
    
    def analyze_video(
        self,
        video_path: str,
        sample_interval: int = 30
    ) -> List[EmotionResult]:
        """
        分析视频中的情绪
        
        从视频中提取帧并分析情绪变化。
        
        Args:
            video_path: 视频文件路径
            sample_interval: 采样间隔（每N帧采样一次）
        
        Returns:
            List[EmotionResult]: 情绪识别结果列表
        """
        import cv2
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        results = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 按间隔采样
            if frame_count % sample_interval == 0:
                result = self._analyze_image(frame)
                results.append(result)
            
            frame_count += 1
        
        cap.release()
        logger.info(f"视频分析完成: {len(results)}帧")
        
        return results
    
    def get_emotion_trend(
        self,
        results: List[EmotionResult]
    ) -> Dict[str, Any]:
        """
        分析情绪趋势
        
        从多个情绪识别结果中分析情绪变化趋势。
        
        Args:
            results: 情绪识别结果列表
        
        Returns:
            Dict: 包含情绪趋势分析结果
        """
        if not results:
            return {"trend": "unknown", "dominant_emotion": "unknown"}
        
        # 统计各情绪出现次数
        emotion_counts = {}
        for r in results:
            if r.face_detected:
                e = r.primary_emotion
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
        
        if not emotion_counts:
            return {"trend": "unknown", "dominant_emotion": "unknown"}
        
        # 找出主导情绪
        dominant = max(emotion_counts.items(), key=lambda x: x[1])
        
        # 分析趋势
        if len(results) >= 2:
            first_half = results[:len(results)//2]
            second_half = results[len(results)//2:]
            
            first_emotions = [r.primary_emotion for r in first_half if r.face_detected]
            second_emotions = [r.primary_emotion for r in second_half if r.face_detected]
            
            if first_emotions and second_emotions:
                first_counts = {}
                second_counts = {}
                
                for e in first_emotions:
                    first_counts[e] = first_counts.get(e, 0) + 1
                for e in second_emotions:
                    second_counts[e] = second_counts.get(e, 0) + 1
                
                first_dominant = max(first_counts.items(), key=lambda x: x[1])[0]
                second_dominant = max(second_counts.items(), key=lambda x: x[1])[0]
                
                if first_dominant != second_dominant:
                    trend = f"{first_dominant} -> {second_dominant}"
                else:
                    trend = "stable"
            else:
                trend = "unknown"
        else:
            trend = "insufficient_data"
        
        return {
            "trend": trend,
            "dominant_emotion": dominant[0],
            "emotion_distribution": emotion_counts,
            "total_frames": len(results),
            "frames_with_face": sum(1 for r in results if r.face_detected)
        }


# 全局实例
_emotion_recognizer: Optional[EmotionRecognizer] = None


def get_emotion_recognizer(use_deep_model: bool = False) -> EmotionRecognizer:
    """
    获取情绪识别器实例
    
    Args:
        use_deep_model: 是否使用深度学习模型
    
    Returns:
        EmotionRecognizer: 情绪识别器实例
    """
    global _emotion_recognizer
    
    if _emotion_recognizer is None:
        _emotion_recognizer = EmotionRecognizer(use_deep_model=use_deep_model)
    
    return _emotion_recognizer


# 导入cv2（用于类型提示）
try:
    import cv2
except ImportError:
    cv2 = None
