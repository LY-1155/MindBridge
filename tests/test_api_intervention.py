"""
干预闭环：HTTP 层自动化测试（TestClient），数据来自 schemas/contracts/samples。

运行：python -m pytest tests/test_api_intervention.py -v

为什么「功能未全实现」仍可能 5 passed？
---------------------------------------
这些用例测的是 **API 契约 + 当前启用的实现（默认 Mock）** 是否按约定返回：
- 未实现的真实业务（LLM 共情、RAG、危机脚本等）被 **MockInterventionService** 用固定字符串替代；
- 断言只检查 HTTP 200、JSON 字段存在、以及 mock 回复里是否含 ``[mock-...]`` 等**形状/标志**；
- 因此 **通过 ≠ 产品级功能已做完**，只说明 **接口接上了、契约没破、Mock 行为稳定**。

要查看每个用例的**输入/输出 JSON**（推荐，不区分 CMD / PowerShell）::

    python -m pytest tests/test_api_intervention.py -v -s --print-io

也可用环境变量（必须加 ``-s`` 否则 print 被吞掉）::

  PowerShell:
    $env:PYTEST_PRINT_IO='1'; python -m pytest tests/test_api_intervention.py -v -s
  CMD（注意不要用 PowerShell 跑 ``set``）::
    set PYTEST_PRINT_IO=1&& python -m pytest tests/test_api_intervention.py -v -s

说明：若在 **PowerShell** 里使用 CMD 的 ``set VAR=1 && ...``，常会报错；请用上一行的 ``$env:`` 或直接用 ``--print-io``。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_SAMPLES = Path(__file__).resolve().parent.parent / "schemas" / "contracts" / "samples"


def _print_io_if_enabled(label: str, request: dict, response: dict) -> None:
    if os.environ.get("PYTEST_PRINT_IO", "").lower() not in ("1", "true", "yes"):
        return
    print(f"\n{'='*20} {label} {'='*20}")
    print("【请求 body】\n" + json.dumps(request, ensure_ascii=False, indent=2))
    print("【响应 body】\n" + json.dumps(response, ensure_ascii=False, indent=2))


def _post_intervention(client: TestClient, filename: str) -> dict:
    payload = json.loads((_SAMPLES / filename).read_text(encoding="utf-8"))
    resp = client.post("/api/v1/modules/intervention/run", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    _print_io_if_enabled(filename, payload, data)
    return data


@pytest.mark.parametrize(
    "filename,route_key,expect_emergency",
    [
        ("intervention_case_comfort.json", "comfort", False),
        ("intervention_case_knowledge.json", "knowledge", False),
        ("intervention_case_crisis.json", "crisis", True),
    ],
)
def test_intervention_module_returns_contract_shape(
    api_client: TestClient,
    filename: str,
    route_key: str,
    expect_emergency: bool,
):
    """Mock 干预：reply 带 [mock-{route}]，字段符合 InterventionResult。"""
    data = _post_intervention(api_client, filename)
    assert data["contract_version"] in ("1.0", "1.1", "1.2", "1.3")
    assert f"[mock-{route_key}]" in data["reply"]
    assert isinstance(data["empathy"], str)
    assert isinstance(data["suggestion"], str)
    assert isinstance(data["action_items"], list)
    assert data["emergency_triggered"] is expect_emergency


def test_intervention_minimal_fixture(api_client: TestClient):
    """最少字段冒烟。"""
    data = _post_intervention(api_client, "intervention_case_minimal.json")
    assert "[mock-comfort]" in data["reply"]
    assert data["emergency_triggered"] is False


def test_intervention_invalid_body_returns_422(api_client: TestClient):
    """缺少必填字段时应校验失败。"""
    resp = api_client.post("/api/v1/modules/intervention/run", json={"user_text": "only"})
    assert resp.status_code == 422
