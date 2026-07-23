"""
音频处理模块
功能：从视频/流中提取音频，使用统一 ASR 接口（SenseVoice / faster-whisper）进行语音识别，
      调用敏感词过滤器
"""

import os
import subprocess
import tempfile
import logging
from typing import Optional, Tuple, Dict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# 项目根目录（用于将相对路径转换为绝对路径，兼容 API 服务和直接运行）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入敏感词过滤器（兼容直接运行和项目级导入）
try:
    from modules.safety.keyword_filter import SensitivityFilter
    FILTER_AVAILABLE = True
except ImportError:
    try:
        # 当直接运行此文件时，手动将上级目录加入 path 后重试
        _parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        import sys as _sys
        if _parent_dir not in _sys.path:
            _sys.path.insert(0, _parent_dir)
        from keyword_filter import SensitivityFilter
        FILTER_AVAILABLE = True
    except ImportError:
        FILTER_AVAILABLE = False
        logger.warning("敏感词过滤器未找到")


class AudioProcessor:
    """音频处理器：提取音频、语音识别（统一ASR接口）、敏感词检测"""

    def __init__(
        self,
        whisper_model: str = None,
        device: str = "cuda",
        filter_config_path: str = None
    ):
        """
        初始化音频处理器

        Args:
            whisper_model: 已废弃，保留兼容性（ASR 后端由 .env ASR_BACKEND 控制）
            device: 推理设备 (cuda/cpu)
            filter_config_path: 敏感词库配置文件路径
        """
        self.device = device
        self._whisper_model_deprecated = whisper_model  # 兼容旧参数，实际不使用

        # 初始化敏感词过滤器
        if FILTER_AVAILABLE:
            self.filter = SensitivityFilter(filter_config_path)
        else:
            self.filter = None

        # 证据保存目录（固定位于项目根目录，避免 CWD 不同导致路径错乱）
        self.evidence_dir = Path(_PROJECT_ROOT) / "evidence"
        self.evidence_dir.mkdir(exist_ok=True)

    def extract_audio(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        audio_format: str = "wav"
    ) -> Optional[str]:
        """
        从视频中提取音频

        Args:
            video_path: 视频文件路径或流URL
            output_path: 输出音频路径，None则使用临时文件
            audio_format: 音频格式 (wav/mp3)

        Returns:
            音频文件路径，失败返回None
        """
        if output_path is None:
            # 创建临时文件
            temp_dir = tempfile.gettempdir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(temp_dir, f"audio_{timestamp}.{audio_format}")

        try:
            # 使用ffmpeg提取音频
            cmd = [
                "ffmpeg", "-y",  # 覆盖输出文件
                "-i", video_path,
                "-vn",  # 不包含视频
                "-acodec", "pcm_s16le" if audio_format == "wav" else "libmp3lame",
                "-ar", "16000",  # 采样率16kHz（Whisper推荐）
                "-ac", "1",  # 单声道
                output_path
            ]

            # 对于实时流，添加超时和读取选项
            if video_path.startswith(("rtmp://", "rtsp://", "http://")):
                cmd.insert(1, "-rtsp_transport")
                cmd.insert(2, "tcp")
                cmd.insert(3, "-timeout")
                cmd.insert(4, "10")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30秒超时
            )

            if result.returncode != 0:
                logger.error(f"音频提取失败: {result.stderr}")
                return None

            logger.info(f"音频提取成功: {output_path}")
            return output_path

        except subprocess.TimeoutExpired:
            logger.error("音频提取超时")
            return None
        except FileNotFoundError:
            logger.error("ffmpeg未安装，请先安装ffmpeg")
            return None
        except Exception as e:
            logger.error(f"音频提取异常: {e}")
            return None

    def extract_audio_segment(
        self,
        video_path: str,
        start_time: float,
        duration: float,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        从视频中提取指定时间段的音频

        Args:
            video_path: 视频文件路径
            start_time: 开始时间（秒）
            duration: 持续时间（秒）
            output_path: 输出路径

        Returns:
            音频文件路径
        """
        if output_path is None:
            temp_dir = tempfile.gettempdir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(temp_dir, f"audio_segment_{timestamp}.wav")

        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-ss", str(start_time),
                "-t", str(duration),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                logger.error(f"音频片段提取失败: {result.stderr}")
                return None

            return output_path

        except Exception as e:
            logger.error(f"音频片段提取异常: {e}")
            return None

    def transcribe(self, audio_path: str) -> Tuple[str, Dict]:
        """
        将音频转换为文字（使用项目统一 ASR 接口）

        Args:
            audio_path: 音频文件路径

        Returns:
            (转录文本, 详细信息字典)
        """
        try:
            # 使用项目统一 ASR 接口：.env 里 ASR_BACKEND 控制用 SenseVoice 还是 faster-whisper
            from multimodal.asr import get_speech_recognizer

            recognizer = get_speech_recognizer()
            result = recognizer.transcribe(audio_path, language="zh")
            transcript = result.text.strip()

            details = {
                "language": result.language or "zh",
                "asr_backend": result.asr_backend,
                "segments": result.segments if result.segments else [],
            }

            logger.info("转录完成 (%s): %d 字符", result.asr_backend, len(transcript))
            return transcript, details

        except Exception as e:
            logger.error(f"转录失败: {e}")
            return "", {"error": str(e)}

    def check_audio(
        self,
        audio_path: str,
        save_evidence: bool = False
    ) -> Dict:
        """
        检测音频中的敏感内容

        Args:
            audio_path: 音频文件路径
            save_evidence: 是否保存证据

        Returns:
            检测结果字典
        """
        result = {
            "level": 0,
            "keywords": [],
            "transcript": "",
            "evidence_path": None,
            "error": None
        }

        # 转录音频
        transcript, transcribe_details = self.transcribe(audio_path)
        result["transcript"] = transcript

        if not transcript:
            result["error"] = transcribe_details.get("error", "转录结果为空")
            return result

        # 敏感词检测
        if self.filter:
            blocked, level, keywords = self.filter.check(transcript)
            result["level"] = level
            result["keywords"] = keywords

            # 保存证据
            if save_evidence and level > 0:
                evidence_path = self._save_audio_evidence(audio_path, keywords)
                result["evidence_path"] = evidence_path

        return result

    def check_video_audio(
        self,
        video_path: str,
        save_evidence: bool = True
    ) -> Dict:
        """
        从视频提取音频并检测敏感内容

        Args:
            video_path: 视频文件路径
            save_evidence: 是否保存证据

        Returns:
            检测结果字典
        """
        # 提取音频
        audio_path = self.extract_audio(video_path)

        if audio_path is None:
            return {
                "level": 0,
                "keywords": [],
                "transcript": "",
                "evidence_path": None,
                "error": "音频提取失败"
            }

        try:
            # 检测音频
            result = self.check_audio(audio_path, save_evidence=save_evidence)
            return result
        finally:
            # 清理临时音频文件
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass

    def _save_audio_evidence(self, audio_path: str, keywords: list) -> str:
        """保存音频证据"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        evidence_path = self.evidence_dir / f"{timestamp}_audio.wav"

        try:
            import shutil
            shutil.copy(audio_path, evidence_path)
            logger.info(f"音频证据已保存: {evidence_path}")
            return str(evidence_path)
        except Exception as e:
            logger.error(f"保存音频证据失败: {e}")
            return None


# ============ 简单测试 ============
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("音频处理器测试")
    print("=" * 60)

    # 检查依赖
    print(f"\n依赖检查:")
    print(f"  敏感词过滤器可用: {FILTER_AVAILABLE}")
    ffmpeg_ok = subprocess.run(['ffmpeg', '-version'], capture_output=True).returncode == 0
    print(f"  ffmpeg可用: {ffmpeg_ok}")
    from multimodal.asr import get_speech_recognizer
    try:
        rec = get_speech_recognizer()
        print(f"  ASR后端: {rec.backend}")
    except Exception as exc:
        print(f"  ASR不可用: {exc}")

    # 初始化处理器
    processor = AudioProcessor(device="cuda")

    # 测试音频检测（如果有测试文件）
    test_video = input("\n输入测试视频路径（回车跳过）: ").strip()
    if test_video and os.path.exists(test_video):
        print("\n正在处理...")
        result = processor.check_video_audio(test_video, save_evidence=True)
        print(f"\n结果:")
        print(f"  等级: {result['level']}")
        print(f"  敏感词: {result['keywords']}")
        print(f"  转录: {result['transcript'][:100]}..." if len(result.get('transcript', '')) > 100 else f"  转录: {result.get('transcript', '')}")
        if result.get('evidence_path'):
            print(f"  证据: {result['evidence_path']}")
    else:
        print("跳过文件测试")
