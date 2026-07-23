"""
多模态处理模块
=============

包含语音识别、情绪识别、语音合成等多模态功能。

模块说明：
- asr.py: 语音识别（Faster-Whisper）
- emotion.py: 情绪识别（面部表情分析）
- tts.py: 语音合成（Edge-TTS）
"""

from .asr import SpeechRecognizer
from .emotion import EmotionRecognizer
from .tts import TextToSpeech
