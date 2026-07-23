"""
视频安全过滤主模块
功能：整合音频处理、画面检测、证据管理，提供统一的视频安全检测接口
"""

import os
import json
import logging
import time
from typing import Dict, List, Callable, Optional, Tuple
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# 项目根目录（用于将相对路径转换为绝对路径，兼容 API 服务和直接运行）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入子模块（使用全路径，兼容 API 服务加载与命令行直接运行）
try:
    from modules.safety.multimodal.audio_processor import AudioProcessor
    AUDIO_AVAILABLE = True
except ImportError:
    try:
        from audio_processor import AudioProcessor
        AUDIO_AVAILABLE = True
    except ImportError as e:
        AUDIO_AVAILABLE = False
        logger.warning(f"音频模块导入失败: {e}")

try:
    from modules.safety.multimodal.frame_detector_local import FrameDetector, VideoFrameExtractor
    FRAME_AVAILABLE = True
except ImportError:
    try:
        from frame_detector_local import FrameDetector, VideoFrameExtractor
        FRAME_AVAILABLE = True
    except ImportError as e:
        FRAME_AVAILABLE = False
        logger.warning(f"画面模块导入失败: {e}")

try:
    from modules.safety.multimodal.evidence_manager import EvidenceManager
    EVIDENCE_AVAILABLE = True
except ImportError:
    try:
        from evidence_manager import EvidenceManager
        EVIDENCE_AVAILABLE = True
    except ImportError as e:
        EVIDENCE_AVAILABLE = False
        logger.warning(f"证据模块导入失败: {e}")

MODULES_AVAILABLE = AUDIO_AVAILABLE and FRAME_AVAILABLE and EVIDENCE_AVAILABLE


class VideoSafetyFilter:
    """
    视频安全过滤器
    支持视频文件和实时流的安全检测
    """

    def __init__(
        self,
        device: str = "cuda",
        frame_fps: float = 1.0,
        confidence_threshold: float = 0.75,
        save_evidence: bool = True,
        evidence_dir: str = "evidence",
        filter_config_path: str = None,
        local_model_path: str = None,
        nsfw_model_path: str = None,
    ):
        """
        初始化视频安全过滤器

        Args:
            device: 推理设备 (cuda/cpu)
            frame_fps: 抽帧频率
            confidence_threshold: 画面检测置信度阈值
            save_evidence: 是否保存证据
            evidence_dir: 证据存储目录
            filter_config_path: 敏感词库配置路径
            local_model_path: 本地 CLIP 模型路径（默认 None 使用在线）
            nsfw_model_path: SigLIP2 NSFW 专用模型路径（默认 None 不启用）
        """
        self.save_evidence = save_evidence
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.frame_fps = frame_fps
        self.filter_config_path = filter_config_path
        self.local_model_path = local_model_path
        self.nsfw_model_path = nsfw_model_path

        # 将相对路径转为基于项目根的绝对路径，避免 CWD 不同导致路径错乱
        if evidence_dir and not os.path.isabs(evidence_dir):
            evidence_dir = os.path.join(_PROJECT_ROOT, evidence_dir)

        # 初始化各子模块
        logger.info("初始化视频安全过滤器...")

        # 音频处理器（ASR 后端由 .env ASR_BACKEND 控制，SenseVoice / faster-whisper）
        if AUDIO_AVAILABLE:
            self.audio_processor = AudioProcessor(
                device=device,
                filter_config_path=filter_config_path
            )
        else:
            self.audio_processor = None
            logger.warning("音频处理器不可用")

        # 画面检测器
        if FRAME_AVAILABLE:
            detector_kwargs = dict(
                device=device,
                confidence_threshold=confidence_threshold,
            )
            if local_model_path:
                detector_kwargs["local_model_path"] = local_model_path
            if nsfw_model_path:
                detector_kwargs["nsfw_model_path"] = nsfw_model_path
            self.frame_detector = FrameDetector(**detector_kwargs)
            self.frame_extractor = VideoFrameExtractor(fps=frame_fps)
        else:
            self.frame_detector = None
            self.frame_extractor = None
            logger.warning("画面检测器不可用")

        # 证据管理器
        if EVIDENCE_AVAILABLE:
            self.evidence_manager = EvidenceManager(evidence_dir=evidence_dir)
        else:
            self.evidence_manager = None
            logger.warning("证据管理器不可用")

        # 预热 ASR 模型（首次加载 SenseVoice/Whisper 可能耗时较长）
        if self.audio_processor:
            try:
                from multimodal.asr import get_speech_recognizer
                asr = get_speech_recognizer()
                logger.info("ASR 模型预热完成 (backend=%s)", asr.backend)
            except Exception as e:
                logger.warning("ASR 模型预热失败（将在首次检测时加载）: %s", e)

        logger.info("视频安全过滤器初始化完成")

    def check_video_file(
        self,
        video_path: str,
        user_id: str = None,
        max_frames: int = None
    ) -> Dict:
        """
        检测视频文件的安全性

        Args:
            video_path: 视频文件路径
            user_id: 用户ID
            max_frames: 最大检测帧数

        Returns:
            检测结果字典
        """
        start_time = time.time()

        result = {
            "blocked": False,
            "video_path": video_path,
            "user_id": user_id,
            "audio_result": {
                "level": 0,
                "keywords": [],
                "transcript": "",
                "evidence_path": None
            },
            "video_result": {
                "level": 0,
                "violations": [],
                "timestamps": [],
                "evidence_paths": []
            },
            "processing_time": 0,
            "error": None
        }

        if not os.path.exists(video_path):
            result["error"] = f"视频文件不存在: {video_path}"
            return result

        try:
            # 并行处理音频和视频
            with ThreadPoolExecutor(max_workers=2) as executor:
                # 音频检测任务
                if self.audio_processor:
                    audio_future = executor.submit(
                        self.audio_processor.check_video_audio,
                        video_path,
                        self.save_evidence
                    )
                else:
                    audio_future = None

                # 视频帧检测任务
                if self.frame_extractor and self.frame_detector:
                    video_future = executor.submit(
                        self._check_video_frames,
                        video_path,
                        max_frames
                    )
                else:
                    video_future = None

                # 获取音频结果
                if audio_future:
                    try:
                        audio_result = audio_future.result(timeout=300)
                        result["audio_result"] = audio_result
                    except Exception as e:
                        logger.error(f"音频检测失败: {e}")
                        result["audio_result"]["error"] = str(e)
                else:
                    result["audio_result"]["error"] = "音频处理器不可用"

                # 获取视频结果
                if video_future:
                    try:
                        video_result = video_future.result(timeout=120)
                        result["video_result"] = video_result
                    except Exception as e:
                        logger.error(f"视频检测失败: {e}")
                        result["video_result"]["error"] = str(e)
                else:
                    result["video_result"]["error"] = "画面检测器不可用"

            # 确定是否拦截
            max_level = max(
                result["audio_result"].get("level", 0),
                result["video_result"].get("level", 0)
            )
            result["blocked"] = max_level >= 1

            # 保存完整报告
            if self.save_evidence and self.evidence_manager and (result["blocked"] or max_level > 0):
                self.evidence_manager.save_full_report(
                    result=result,
                    video_path=video_path,
                    user_id=user_id
                )

        except Exception as e:
            logger.error(f"视频检测异常: {e}")
            result["error"] = str(e)

        result["processing_time"] = round(time.time() - start_time, 2)
        return result

    def _check_video_frames(
        self,
        video_path: str,
        max_frames: int = None
    ) -> Dict:
        """
        检测视频帧的安全性

        Args:
            video_path: 视频文件路径
            max_frames: 最大检测帧数

        Returns:
            检测结果
        """
        result = {
            "level": 0,
            "violations": [],
            "timestamps": [],
            "evidence_paths": [],
            "frame_count": 0
        }

        if not self.frame_extractor or not self.frame_detector:
            result["error"] = "画面检测模块不可用"
            return result

        # 提取帧
        frames, timestamps = self.frame_extractor.extract_frames(
            video_path,
            max_frames=max_frames
        )
        result["frame_count"] = len(frames)

        if not frames:
            return result

        # 检测每一帧
        seen_types = set()
        for frame, timestamp in zip(frames, timestamps):
            frame_result = self.frame_detector.detect_frame(frame)

            if frame_result.get("violations"):
                for violation in frame_result["violations"]:
                    vtype = violation["type"]
                    result["timestamps"].append(round(timestamp, 2))
                    result["level"] = max(result["level"], violation.get("level", 1))

                    # 按类型去重，保留首次出现的完整违规信息
                    if vtype not in seen_types:
                        seen_types.add(vtype)
                        result["violations"].append({
                            "type": violation["type"],
                            "name": violation.get("name", ""),
                            "confidence": violation.get("confidence", 0),
                            "level": violation.get("level", 1),
                        })

                    # 保存证据
                    if self.save_evidence and self.evidence_manager:
                        evidence = self.evidence_manager.save_frame_evidence(
                            frame=frame,
                            violation_type=violation["type"],
                            timestamp=timestamp,
                            metadata={"confidence": violation.get("confidence")}
                        )
                        if evidence.get("success"):
                            result["evidence_paths"].append(evidence["path"])

        return result

    def check_video_stream(
        self,
        stream_url: str,
        callback: Callable[[Dict], None],
        user_id: str = None,
        duration: float = None
    ):
        """
        实时检测视频流

        Args:
            stream_url: 流地址 (rtmp/rtsp/http)
            callback: 检测结果回调函数
            user_id: 用户ID
            duration: 检测时长（秒），None表示持续检测
        """
        if not self.frame_extractor or not self.frame_detector:
            callback({
                "blocked": False,
                "type": "error",
                "error": "画面检测模块不可用"
            })
            return

        logger.info(f"开始实时流检测: {stream_url}")

        start_time = time.time()
        frame_count = 0
        audio_buffer = []
        last_audio_check = time.time()
        audio_check_interval = 5.0  # 每5秒检测一次音频

        try:
            for frame, timestamp in self.frame_extractor.extract_frames_generator(stream_url):
                # 检查时长限制
                if duration and (time.time() - start_time) > duration:
                    logger.info("达到指定检测时长，停止检测")
                    break

                frame_count += 1

                # 检测当前帧
                frame_result = self.frame_detector.detect_frame(frame)

                if frame_result.get("violations"):
                    result = {
                        "blocked": True,
                        "type": "video",
                        "timestamp": timestamp,
                        "violations": frame_result["violations"],
                        "user_id": user_id,
                        "frame_count": frame_count
                    }

                    # 保存证据
                    if self.save_evidence and self.evidence_manager:
                        for violation in frame_result["violations"]:
                            evidence = self.evidence_manager.save_frame_evidence(
                                frame=frame,
                                violation_type=violation["type"],
                                timestamp=timestamp,
                                metadata={"confidence": violation.get("confidence")}
                            )
                            if evidence.get("success"):
                                result["evidence_path"] = evidence["path"]

                    callback(result)

                # 定期检测音频
                if time.time() - last_audio_check > audio_check_interval:
                    # TODO: 实现实时流的音频检测
                    # 需要维护一个音频缓冲区并定期检测
                    last_audio_check = time.time()

        except Exception as e:
            logger.error(f"实时流检测异常: {e}")
            callback({
                "blocked": False,
                "type": "error",
                "error": str(e)
            })

    def get_stats(self) -> Dict:
        """
        获取过滤器统计信息

        Returns:
            统计信息字典
        """
        storage = self.evidence_manager.get_storage_usage() if self.evidence_manager else {}

        return {
            "device": self.device,
            "asr_backend": getattr(self.audio_processor, "_asr_backend", None) if self.audio_processor else None,
            "frame_detector": "clip",
            "confidence_threshold": self.confidence_threshold,
            "save_evidence": self.save_evidence,
            "storage": storage
        }


class StreamSafetyMonitor:
    """
    实时流安全监控器
    提供更便捷的实时流监控接口
    """

    def __init__(
        self,
        stream_url: str,
        on_violation: Callable[[Dict], None],
        user_id: str = None,
        **filter_kwargs
    ):
        """
        初始化监控器

        Args:
            stream_url: 流地址
            on_violation: 违规回调函数
            user_id: 用户ID
            **filter_kwargs: 传递给VideoSafetyFilter的参数
        """
        self.stream_url = stream_url
        self.on_violation = on_violation
        self.user_id = user_id

        self.filter = VideoSafetyFilter(**filter_kwargs)
        self._running = False

    def start(self, duration: float = None):
        """
        开始监控

        Args:
            duration: 监控时长（秒）
        """
        self._running = True
        logger.info(f"启动流监控: {self.stream_url}")

        def callback(result):
            if result.get("blocked"):
                self.on_violation(result)

        self.filter.check_video_stream(
            stream_url=self.stream_url,
            callback=callback,
            user_id=self.user_id,
            duration=duration
        )

    def stop(self):
        """停止监控"""
        self._running = False
        logger.info("停止流监控")


# ============ 简单测试 ============
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # 把项目根目录加到 path（解决从子目录运行时找不到 modules 包的问题）
    sys.path.insert(0, _PROJECT_ROOT)

    print("=" * 60)
    print("视频安全过滤器测试")
    print("=" * 60)

    # 初始化过滤器：优先读取 .env / 环境变量，否则交互式输入
    _env_model_path = os.getenv("SAFETY_MODEL_PATH", "")

    if _env_model_path:
        local_model = _env_model_path
        print("\n从环境变量读取: model_path=%s" % local_model)
    else:
        local_model = input("\n输入本地 CLIP 模型路径（回车使用在线模型）: ").strip() or None

    video_filter = VideoSafetyFilter(
        device="cuda",
        frame_fps=1.0,
        save_evidence=True,
        local_model_path=local_model,
    )

    # 打印统计信息
    stats = video_filter.get_stats()
    print(f"\n过滤器配置:")
    print(f"  设备: {stats['device']}")
    print(f"  ASR后端: {stats['asr_backend']}")
    print(f"  存储使用: {stats['storage']['total_size_mb']} MB")

    # 循环测试视频文件（输入 q 退出）
    print("\n输入 q 退出测试\n")
    while True:
        test_video = input("输入测试视频路径: ").strip()
        if test_video.lower() == "q":
            print("退出测试")
            break
        if not test_video:
            continue
        if not os.path.exists(test_video):
            print(f"文件不存在: {test_video}")
            continue

        print("正在检测视频...")
        result = video_filter.check_video_file(test_video, user_id="test_user")

        print(f"\n检测结果:")
        print(f"  是否拦截: {result['blocked']}")
        print(f"  处理时间: {result['processing_time']}秒")
        print(f"\n音频结果:")
        print(f"  等级: {result['audio_result']['level']}")
        print(f"  敏感词: {result['audio_result']['keywords']}")
        print(f"  转录: {result['audio_result'].get('transcript', '')[:100]}...")
        print(f"\n视频结果:")
        print(f"  等级: {result['video_result']['level']}")
        print(f"  违规类型: {result['video_result']['violations']}")
        print(f"  时间点: {result['video_result']['timestamps']}")
        print(f"  证据文件: {result['video_result']['evidence_paths']}")
        print()
