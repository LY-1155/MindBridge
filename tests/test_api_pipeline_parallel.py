"""
流水线与其它并行模块：HTTP 冒烟测试。

运行：pytest tests/test_api_pipeline_parallel.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_SAMPLES = Path(__file__).resolve().parent.parent / "schemas" / "contracts" / "samples"


def test_pipeline_run_returns_four_stages(api_client: TestClient):
    payload = json.loads((_SAMPLES / "pipeline_request.json").read_text(encoding="utf-8"))
    resp = api_client.post("/api/v1/pipeline/run", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["contract_version"] in ("1.0", "1.1", "1.2", "1.3")
    for key in ("safety", "emotion", "route", "intervention"):
        assert key in data
        assert isinstance(data[key], dict)
    assert data["stopped_after_safety"] is False
    assert "[mock-" in data["intervention"].get("reply", "")


def test_module_safety_check(api_client: TestClient):
    payload = json.loads((_SAMPLES / "safety_request.json").read_text(encoding="utf-8"))
    resp = api_client.post("/api/v1/modules/safety/check", json=payload)
    assert resp.status_code == 200
    assert resp.json()["contract_version"] in ("1.0", "1.1", "1.2", "1.3")


def test_module_emotion_analyze(api_client: TestClient):
    payload = json.loads((_SAMPLES / "emotion_request.json").read_text(encoding="utf-8"))
    resp = api_client.post("/api/v1/modules/emotion/analyze", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "primary_emotion" in body
    assert "risk" in body


def test_module_router_route(api_client: TestClient):
    payload = json.loads((_SAMPLES / "router_request.json").read_text(encoding="utf-8"))
    resp = api_client.post("/api/v1/modules/router/route", json=payload)
    assert resp.status_code == 200
    assert resp.json().get("route") in ("general", "comfort", "knowledge", "crisis")


def test_health_ping(api_client: TestClient):
    r = api_client.get("/ping")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"
