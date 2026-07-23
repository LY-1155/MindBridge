"""
测试语音识别功能
"""

import asyncio
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from multimodal.asr import get_speech_recognizer


def test_asr():
    print("测试语音识别模块...")
    
    recognizer = get_speech_recognizer()
    print(f"模型大小: {recognizer.model_size}")
    print(f"模型路径: {recognizer.model_path}")
    print(f"设备: {recognizer.device}")
    
    print("\n尝试加载模型...")
    try:
        model = recognizer.model
        print("✓ 模型加载成功!")
        return True
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return False


if __name__ == "__main__":
    test_asr()
