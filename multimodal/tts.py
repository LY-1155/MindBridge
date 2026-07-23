"""
语音合成模块（TTS）
=================

使用Edge-TTS实现文本转语音功能。
Edge-TTS是微软Edge浏览器的TTS引擎，免费且效果好。

主要功能：
1. 文本转语音
2. 支持多种语言和声音
3. 可调节语速和音调
4. 支持SSML标记

使用方式：
    tts = TextToSpeech()
    await tts.synthesize("你好，世界", "output.mp3")
"""

import os
import asyncio
import tempfile
from typing import Optional, List
from dataclasses import dataclass
import logging
import base64

logger = logging.getLogger(__name__)


@dataclass
class VoiceInfo:
    """
    声音信息
    
    Attributes:
        name: 声音名称
        short_name: 简短名称
        gender: 性别（Male/Female）
        language: 语言代码
        locale: 区域设置
    """
    name: str
    short_name: str
    gender: str
    language: str
    locale: str


class TextToSpeech:
    """
    语音合成器
    
    使用Edge-TTS进行文本转语音。
    支持多种语言和声音选择。
    
    使用示例：
        tts = TextToSpeech()
        
        # 同步使用
        await tts.synthesize("你好，世界", "output.mp3")
        
        # 获取音频字节
        audio_bytes = await tts.synthesize_to_bytes("你好，世界")
    """
    
    # 中文声音推荐
    CHINESE_VOICES = {
        "female": "zh-CN-XiaoxiaoNeural",      # 晓晓（女声，温柔）
        "male": "zh-CN-YunxiNeural",           # 云希（男声，年轻）
        "female_news": "zh-CN-XiaoyiNeural",   # 晓伊（女声，新闻播报风格）
        "male_news": "zh-CN-YunjianNeural",    # 云健（男声，新闻播报风格）
    }
    
    # 英文声音推荐
    ENGLISH_VOICES = {
        "female": "en-US-JennyNeural",         # Jenny（女声，自然）
        "male": "en-US-GuyNeural",             # Guy（男声，成熟）
    }
    
    # 默认声音
    DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
    
    def __init__(self, voice: str = None):
        """
        初始化语音合成器
        
        Args:
            voice: 声音名称，不指定则使用默认声音
        """
        self.voice = voice or self.DEFAULT_VOICE
        self._communicate = None
        
        logger.info(f"初始化语音合成器: voice={self.voice}")
    
    async def synthesize(
        self,
        text: str,
        output_path: str,
        rate: str = "+0%",
        pitch: str = "+0Hz"
    ) -> str:
        """
        将文本合成为语音文件
        
        Args:
            text: 要合成的文本
            output_path: 输出文件路径（MP3格式）
            rate: 语速调整（如 "+50%" 加快，"-50%" 减慢）
            pitch: 音调调整（如 "+50Hz" 升高，"-50Hz" 降低）
        
        Returns:
            str: 输出文件路径
        
        示例：
            await tts.synthesize("你好", "hello.mp3", rate="+20%")
        """
        try:
            import edge_tts
        except ImportError:
            raise ImportError("请安装edge-tts: pip install edge-tts")
        
        logger.info(f"开始语音合成: text_length={len(text)}")
        
        # 创建通信对象
        communicate = edge_tts.Communicate(
            text,
            self.voice,
            rate=rate,
            pitch=pitch
        )
        
        # 保存音频文件
        await communicate.save(output_path)
        
        logger.info(f"语音合成完成: {output_path}")
        
        return output_path
    
    async def synthesize_to_bytes(
        self,
        text: str,
        rate: str = "+0%",
        pitch: str = "+0Hz"
    ) -> bytes:
        """
        将文本合成为音频字节流
        
        Args:
            text: 要合成的文本
            rate: 语速调整
            pitch: 音调调整
        
        Returns:
            bytes: MP3格式的音频数据
        
        示例：
            audio_bytes = await tts.synthesize_to_bytes("你好")
            with open("output.mp3", "wb") as f:
                f.write(audio_bytes)
        """
        try:
            import edge_tts
        except ImportError:
            raise ImportError("请安装edge-tts: pip install edge-tts")
        
        logger.info(f"开始语音合成（字节流）: text_length={len(text)}")
        
        communicate = edge_tts.Communicate(
            text,
            self.voice,
            rate=rate,
            pitch=pitch
        )
        
        # 收集音频数据
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        
        logger.info(f"语音合成完成: size={len(audio_data)} bytes")
        
        return audio_data
    
    async def synthesize_to_base64(
        self,
        text: str,
        rate: str = "+0%",
        pitch: str = "+0Hz"
    ) -> str:
        """
        将文本合成为Base64编码的音频
        
        Args:
            text: 要合成的文本
            rate: 语速调整
            pitch: 音调调整
        
        Returns:
            str: Base64编码的MP3音频数据
        
        示例：
            base64_audio = await tts.synthesize_to_base64("你好")
            # 在HTML中使用: <audio src="data:audio/mp3;base64,{base64_audio}">
        """
        audio_bytes = await self.synthesize_to_bytes(text, rate, pitch)
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    def synthesize_sync(
        self,
        text: str,
        output_path: str,
        rate: str = "+0%",
        pitch: str = "+0Hz"
    ) -> str:
        """
        同步版本的语音合成
        
        用于非异步环境中调用。
        
        Args:
            text: 要合成的文本
            output_path: 输出文件路径
            rate: 语速调整
            pitch: 音调调整
        
        Returns:
            str: 输出文件路径
        """
        return asyncio.run(self.synthesize(text, output_path, rate, pitch))
    
    def synthesize_to_bytes_sync(
        self,
        text: str,
        rate: str = "+0%",
        pitch: str = "+0Hz"
    ) -> bytes:
        """
        同步版本的语音合成（返回字节流）
        """
        return asyncio.run(self.synthesize_to_bytes(text, rate, pitch))
    
    @staticmethod
    async def list_voices(language: str = None) -> List[VoiceInfo]:
        """
        列出可用的声音
        
        Args:
            language: 筛选语言（如"zh"表示中文）
        
        Returns:
            List[VoiceInfo]: 声音信息列表
        
        示例：
            voices = await TextToSpeech.list_voices("zh")
            for v in voices:
                print(f"{v.short_name}: {v.gender}, {v.locale}")
        """
        try:
            import edge_tts
        except ImportError:
            raise ImportError("请安装edge-tts: pip install edge-tts")
        
        voices = await edge_tts.list_voices()
        
        result = []
        for v in voices:
            locale = v.get("Locale", "")
            voice_language = locale.split("-")[0] if locale else ""
            
            voice_info = VoiceInfo(
                name=v.get("Name", ""),
                short_name=v.get("ShortName", ""),
                gender=v.get("Gender", ""),
                language=voice_language,
                locale=locale
            )
            
            # 按语言筛选
            if language is None or (voice_language and voice_language.lower().startswith(language.lower())):
                result.append(voice_info)
        
        return result
    
    def set_voice(self, voice: str) -> None:
        """
        设置声音
        
        Args:
            voice: 声音名称
        """
        self.voice = voice
        logger.info(f"切换声音: {voice}")
    
    def set_chinese_voice(self, voice_type: str = "female") -> None:
        """
        设置中文声音
        
        Args:
            voice_type: 声音类型（female/male/female_news/male_news）
        """
        if voice_type in self.CHINESE_VOICES:
            self.set_voice(self.CHINESE_VOICES[voice_type])
        else:
            raise ValueError(f"未知的声音类型: {voice_type}")
    
    def set_english_voice(self, voice_type: str = "female") -> None:
        """
        设置英文声音
        
        Args:
            voice_type: 声音类型（female/male）
        """
        if voice_type in self.ENGLISH_VOICES:
            self.set_voice(self.ENGLISH_VOICES[voice_type])
        else:
            raise ValueError(f"未知的声音类型: {voice_type}")


# 全局实例
_tts_instances: dict[str, TextToSpeech] = {}


def get_tts(voice: str = None) -> TextToSpeech:
    """
    获取语音合成器实例
    
    Args:
        voice: 声音名称
    
    Returns:
        TextToSpeech: 语音合成器实例
    """
    selected_voice = voice or TextToSpeech.DEFAULT_VOICE
    instance = _tts_instances.get(selected_voice)
    if instance is None:
        instance = TextToSpeech(voice=selected_voice)
        _tts_instances[selected_voice] = instance
    return instance
