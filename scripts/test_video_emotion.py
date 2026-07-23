"""
手动测试脚本：用真实视频跑视频情绪分析模块（含全管线）。

用法：
    source activate qwen
    cd c:/Users/xt/Desktop/Project/mental-intervene-master
    python scripts/test_video_emotion.py <视频文件路径.mp4>

也可以只测预处理，不跑全管线：
    python scripts/test_video_emotion.py <视频文件路径.mp4> --preprocess-only
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal.video_preprocessor import VideoPreprocessor
from pipeline.orchestrator import run_video_pipeline


def test_preprocessor(video_path: str):
    """只测 VideoPreprocessor：抽帧 + ASR + 音频情绪 + 视觉情绪。"""
    if not os.path.exists(video_path):
        print(f"错误：文件不存在 {video_path}")
        return

    print(f"视频文件: {video_path}")
    print(f"文件大小: {os.path.getsize(video_path) / 1024 / 1024:.1f} MB")
    print("-" * 50)

    p = VideoPreprocessor()
    result = p.process(video_path)

    print("【1】ASR 文本")
    print(f"  {result.text or '(空 — 可能视频无语音或 ASR 未安装)'}")
    print()

    print("【2】视觉情绪（多帧 HSEmotion → Ekman→契约 映射 → 时序加权聚合）")
    if result.visual_emotion:
        print(f"  主情绪:     {result.visual_emotion.get('primary_emotion')}")
        print(f"  置信度:     {result.visual_emotion.get('confidence')}")
        print(f"  人脸检测率: {result.visual_emotion.get('face_detection_rate')}")
        print(f"  有效帧:     {result.visual_emotion.get('valid_frames')} / {result.visual_emotion.get('total_frames')}")
        print(f"  模型:       {result.visual_emotion.get('model_name')}")
        dist = result.visual_emotion.get('all_emotions', {})
        if dist:
            print(f"  情绪分布:   {json.dumps({k: round(v, 3) for k, v in dist.items() if v > 0.01}, ensure_ascii=False)}")
    else:
        print("  (无 — 未检测到人脸或模型未安装)")
    print()

    print("【3】音频情绪（SenseVoice）")
    if result.audio_emotion:
        print(f"  主情绪: {result.audio_emotion.get('primary_emotion')}")
        print(f"  置信度: {result.audio_emotion.get('confidence')}")
    else:
        print("  (无 — 视频无音轨或 SenseVoice 未安装)")
    print()

    print("【4】三模态融合结果")
    try:
        from modules.runtime import get_pipeline_services
        from config.settings import Settings
        from schemas.contracts import EmotionAnalyzeRequest

        # MOCK_EMOTION 默认 True，这里关掉用真实三模态融合
        cfg = Settings(MOCK_EMOTION=False)

        sample_safety = {
            "level": 0, "blocked": False, "matched_terms": [], "meta": {},
            "contract_version": "1.2",
        }
        req = EmotionAnalyzeRequest(
            text=result.text,
            pre_extracted_audio_emotion=result.audio_emotion,
            pre_extracted_visual_emotion=result.visual_emotion,
            safety=sample_safety,
        )
        fused = get_pipeline_services(cfg).emotion.analyze(req)
        print(f"  主情绪:     {fused.primary_emotion}")
        print(f"  强度:       {fused.intensity}")
        print(f"  风险值:     {fused.risk}")
        print(f"  模态信息:   {json.dumps(fused.modality_notes, ensure_ascii=False, indent=2)}")
    except Exception as exc:
        print(f"  (融合失败: {exc})")
    print()

    print("【5】警告信息")
    if result.warnings:
        for w in result.warnings:
            print(f"  ⚠ {w}")
    else:
        print("  (无警告)")
    print("-" * 50)


def test_full_pipeline(video_path: str):
    """跑完整视频管线：预处理 → 安全 → 三模态情绪融合 → 路由 → 干预。"""
    if not os.path.exists(video_path):
        print(f"错误：文件不存在 {video_path}")
        return

    print(f"视频文件: {video_path}")
    print("-" * 50)

    print("运行完整管线 (run_video_pipeline)...")
    out = run_video_pipeline(video_path=video_path)

    print()
    print("【安全检测】")
    safety = out.safety
    print(f"  level:   {safety.get('level')}")
    print(f"  blocked: {safety.get('blocked')}")
    print(f"  匹配词:   {safety.get('matched_terms')}")
    print()

    print("【情绪分析（三模态融合）】")
    emotion = out.emotion
    print(f"  主情绪:     {emotion.get('primary_emotion')}")
    print(f"  强度:       {emotion.get('intensity')}")
    print(f"  风险值:     {emotion.get('risk')}")
    print(f"  模态信息:   {json.dumps(emotion.get('modality_notes'), ensure_ascii=False, indent=2)}")
    print()

    print("【路由决策】")
    route = out.route
    print(f"  路由:       {route.get('route')}")
    print(f"  原因:       {route.get('reason')}")
    print(f"  置信度:     {route.get('confidence')}")
    print()

    print("【干预回复】")
    intervention = out.intervention
    reply = intervention.get('reply', '')
    print(f"  回复: {reply[:200]}{'...' if len(reply) > 200 else ''}")
    print()

    print(f"安全短路: {out.stopped_after_safety}")
    print(f"契约版本: {out.contract_version}")
    print("-" * 50)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/test_video_emotion.py <视频文件路径.mp4> [--preprocess-only]")
        print()
        print("示例:")
        print('  python scripts/test_video_emotion.py "C:\\Users\\xt\\Videos\\my_video.mp4"')
        print('  python scripts/test_video_emotion.py video.mp4 --preprocess-only')
        sys.exit(1)

    video_path = sys.argv[1]
    preprocess_only = "--preprocess-only" in sys.argv

    if preprocess_only:
        test_preprocessor(video_path)
    else:
        test_preprocessor(video_path)
        print()
        test_full_pipeline(video_path)
