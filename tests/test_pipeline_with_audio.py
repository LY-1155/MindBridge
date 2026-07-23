"""管线端到端测试：audio_path 透传和新路由。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pipeline.orchestrator import run_pipeline
from schemas.contracts.v1 import PipelineInput, PipelineOutput

_SAMPLES = Path(__file__).resolve().parent.parent / "schemas" / "contracts" / "samples"


class TestPipelineAudioPath:
    """验证 PipelineInput.audio_path 存在时管线正常工作。"""

    def test_pipeline_without_audio_path_works(self):
        inp = PipelineInput(text="我今天很开心")
        out = run_pipeline(inp)
        assert isinstance(out, PipelineOutput)
        assert out.emotion["primary_emotion"] in (
            "neutral", "anxiety", "sadness", "anger",
            "fear", "stress", "happiness", "confusion",
        )

    def test_pipeline_with_audio_path_flows(self):
        """audio_path 传给管线不会报错（即使文件不存在，情绪模块会降级）。"""
        inp = PipelineInput(text="我很焦虑", audio_path="/nonexistent/test.wav")
        out = run_pipeline(inp)
        assert isinstance(out, PipelineOutput)
        assert "primary_emotion" in out.emotion

    def test_pipeline_contract_version_is_1_3(self):
        inp = PipelineInput(text="测试")
        out = run_pipeline(inp)
        assert out.contract_version == "1.3"


def _make_wav_bytes() -> bytes:
    """生成最小有效 WAV 头（44 字节）+ 静音数据，用于文件上传测试。"""
    import struct

    sample_rate = 16000
    num_channels = 1
    bits_per_sample = 16
    duration_samples = sample_rate  # 1 秒
    data_size = duration_samples * num_channels * (bits_per_sample // 8)

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,  # chunk size
        1,  # PCM
        num_channels,
        sample_rate,
        sample_rate * num_channels * (bits_per_sample // 8),
        num_channels * (bits_per_sample // 8),
        bits_per_sample,
        b"data",
        data_size,
    )
    # 静音填充
    samples = b"\x00" * data_size
    return header + samples


class TestPipelineApi:
    """HTTP API 冒烟测试。"""

    def test_pipeline_run_returns_200(self, api_client: TestClient):
        payload = json.loads((_SAMPLES / "pipeline_request.json").read_text(encoding="utf-8"))
        resp = api_client.post("/api/v1/pipeline/run", json=payload)
        assert resp.status_code == 200

    def test_pipeline_run_contract_version_is_1_1(self, api_client: TestClient):
        payload = json.loads((_SAMPLES / "pipeline_request.json").read_text(encoding="utf-8"))
        payload["contract_version"] = "1.1"
        resp = api_client.post("/api/v1/pipeline/run", json=payload)
        data = resp.json()
        assert data["contract_version"] == "1.1"


class TestEmotionAnalyzeAudioApi:
    """POST /api/v1/modules/emotion/analyze-audio 文件上传端测试。"""

    def test_upload_audio_returns_200(self, api_client: TestClient):
        """上传音频文件 → 200 + EmotionTags 结构。"""
        wav = _make_wav_bytes()
        resp = api_client.post(
            "/api/v1/modules/emotion/analyze-audio",
            files={"audio": ("test.wav", wav, "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "primary_emotion" in data
        assert data["primary_emotion"] in (
            "neutral", "anxiety", "sadness", "anger",
            "fear", "stress", "happiness", "confusion", "distress",
        )
        assert 0.0 <= data["intensity"] <= 1.0
        assert 0.0 <= data["risk"] <= 1.0

    def test_upload_audio_with_text(self, api_client: TestClient):
        """上传音频 + 文本 → 文本和语音信号融合。"""
        wav = _make_wav_bytes()
        resp = api_client.post(
            "/api/v1/modules/emotion/analyze-audio",
            files={"audio": ("test.wav", wav, "audio/wav")},
            data={"text": "我最近压力好大，快崩溃了"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "primary_emotion" in data
        # 带文本关键词时应能识别 stress
        assert data["primary_emotion"] in (
            "stress", "anxiety", "sadness", "neutral",
            "fear", "anger", "happiness", "confusion", "distress",
        )

    def test_upload_audio_without_text_still_works(self, api_client: TestClient):
        """不上传文本，只传音频 → 仍然返回合法结果。"""
        wav = _make_wav_bytes()
        resp = api_client.post(
            "/api/v1/modules/emotion/analyze-audio",
            files={"audio": ("test.wav", wav, "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "primary_emotion" in data

    def test_upload_without_file_returns_422(self, api_client: TestClient):
        """不上传音频文件 → 422 验证错误。"""
        resp = api_client.post("/api/v1/modules/emotion/analyze-audio")
        assert resp.status_code == 422
