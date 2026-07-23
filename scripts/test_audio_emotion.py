"""
手动测试脚本：用真实音频跑情绪分析模块。

用法：
    source activate qwen
    cd c:/Users/xt/Desktop/Project/mental-intervene-master
    python scripts/test_audio_emotion.py <音频文件路径.wav>

如果没有音频文件，可以先录一段话（比如"我最近压力好大，快崩溃了"），
保存为 wav 格式，然后用这个脚本测试。
"""
from __future__ import annotations

import os
import sys
import json

# 加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.emotion.stub import EmotionService
from schemas.contracts.v1 import EmotionAnalyzeRequest


def test_with_audio(audio_path: str):
    """用真实音频跑情绪分析。"""
    if not os.path.exists(audio_path):
        print(f"错误：文件不存在 {audio_path}")
        return

    print(f"音频文件: {audio_path}")
    print(f"文件大小: {os.path.getsize(audio_path) / 1024:.1f} KB")
    print("-" * 50)

    svc = EmotionService()
    sample_safety = {
        "level": 0,
        "blocked": False,
        "matched_terms": [],
        "meta": {},
        "contract_version": "1.1",
    }

    # 方式 1: 传 audio_path，走完整语音情绪流程
    print("【路径 1】传入 audio_path，调用 SenseVoice 语音情绪分析...")
    req = EmotionAnalyzeRequest(
        text="",
        audio_path=audio_path,
        safety=sample_safety,
    )
    result = svc.analyze(req)

    print(f"  主情绪:     {result.primary_emotion}")
    print(f"  情绪强度:   {result.intensity}")
    print(f"  风险值:     {result.risk}")
    print(f"  模态信息:   {json.dumps(result.modality_notes, ensure_ascii=False, indent=2)}")
    print()

    # 方式 2: 传 text + audio_path，文本和语音情绪融合
    print("【路径 2】传入 text + audio_path，融合文本和语音信号...")
    req2 = EmotionAnalyzeRequest(
        text="我很难过，压力很大",
        audio_path=audio_path,
        safety=sample_safety,
    )
    result2 = svc.analyze(req2)

    print(f"  主情绪:     {result2.primary_emotion}")
    print(f"  情绪强度:   {result2.intensity}")
    print(f"  风险值:     {result2.risk}")
    print(f"  模态信息:   {json.dumps(result2.modality_notes, ensure_ascii=False, indent=2)}")


def test_without_audio():
    """纯文本对比：确认降级路径工作。"""
    print("-" * 50)
    print("【对比】纯文本路径（无音频）")
    svc = EmotionService()
    sample_safety = {
        "level": 0,
        "blocked": False,
        "matched_terms": [],
        "meta": {},
        "contract_version": "1.1",
    }

    for text in [
        "我最近焦虑得晚上都睡不着",
        "太开心了今天！",
        "我感觉压力好大，快撑不住了",
        "今天天气还行吧",
    ]:
        req = EmotionAnalyzeRequest(text=text, safety=sample_safety)
        result = svc.analyze(req)
        print(f"  \"{text}\"")
        print(f"    → {result.primary_emotion} (intensity={result.intensity}, risk={result.risk})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_with_audio(sys.argv[1])
    else:
        print("没有音频文件，只测纯文本路径。")
        print("用法: python scripts/test_audio_emotion.py <音频.wav>")
        print()
        test_without_audio()
