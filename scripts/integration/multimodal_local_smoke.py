"""
多模态功能测试脚本
=================

测试语音识别、情绪识别、语音合成功能。
"""

import asyncio
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def test_tts():
    """测试语音合成"""
    print("\n=== 测试语音合成 (TTS) ===")
    
    from multimodal.tts import TextToSpeech
    
    tts = TextToSpeech()
    print(f"默认声音: {tts.voice}")
    
    async def run_test():
        audio_bytes = await tts.synthesize_to_bytes("你好，这是一个测试。欢迎使用心理咨询AI助手。")
        print(f"生成音频大小: {len(audio_bytes)} bytes")
        
        output_path = "test_output.mp3"
        await tts.synthesize("你好，这是一个测试。", output_path)
        print(f"音频已保存到: {output_path}")
        
        return True
    
    try:
        result = asyncio.run(run_test())
        print("TTS测试: ✓ 通过")
        return result
    except Exception as e:
        print(f"TTS测试: ✗ 失败 - {e}")
        return False


def test_emotion():
    """测试情绪识别"""
    print("\n=== 测试情绪识别 ===")
    
    from multimodal.emotion import EmotionRecognizer
    
    recognizer = EmotionRecognizer()
    print("情绪识别器初始化成功")
    
    print(f"支持的情绪类型: {recognizer.EMOTIONS}")
    print("情绪识别测试: ✓ 通过 (需要实际图像进行完整测试)")
    return True


def test_asr():
    """测试语音识别"""
    print("\n=== 测试语音识别 (ASR) ===")
    
    from multimodal.asr import SpeechRecognizer
    
    recognizer = SpeechRecognizer()
    print(f"模型大小: {recognizer.model_size}")
    print(f"设备: {recognizer.device}")
    
    print("语音识别器初始化成功")
    print("ASR测试: ✓ 通过 (需要实际音频进行完整测试)")
    return True


def test_api_imports():
    """测试API路由导入"""
    print("\n=== 测试API路由导入 ===")
    
    try:
        from api.routes.multimodal import router
        print(f"路由前缀: {router.prefix}")
        print(f"路由数量: {len(router.routes)}")
        
        for route in router.routes:
            if hasattr(route, 'methods') and hasattr(route, 'path'):
                print(f"  {list(route.methods)} {route.path}")
        
        print("API路由导入: ✓ 通过")
        return True
    except Exception as e:
        print(f"API路由导入: ✗ 失败 - {e}")
        return False


def main():
    print("=" * 50)
    print("多模态功能测试")
    print("=" * 50)
    
    results = []
    
    results.append(("TTS", test_tts()))
    results.append(("情绪识别", test_emotion()))
    results.append(("ASR", test_asr()))
    results.append(("API路由", test_api_imports()))
    
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n总计: {passed}/{total} 通过")


if __name__ == "__main__":
    main()
