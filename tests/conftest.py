"""
Pytest 公共配置：把项目根目录加入 PYTHONPATH，并提供 FastAPI TestClient。
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest
from fastapi.testclient import TestClient

# 测试套件使用高限额，避免干扰非限流测试；速率限制专项测试会自行降限
os.environ.setdefault("RATE_LIMIT_DEFAULT", "1000/minute")

from api.main import app


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--print-io",
        action="store_true",
        default=False,
        help="打印支持该开关的用例的请求/响应 JSON（等价于设置环境变量 PYTEST_PRINT_IO=1）。",
    )


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("--print-io", default=False):
        os.environ["PYTEST_PRINT_IO"] = "1"


@pytest.fixture(autouse=True)
def _reset_rate_limiter_storage():
    """每个测试前重置速率限制存储，避免跨测试泄漏。"""
    from modules.rate_limit import limiter
    from limits.storage import MemoryStorage
    from limits.strategies import FixedWindowRateLimiter

    saved_storage = limiter._storage
    saved_limiter = limiter._limiter

    fresh = MemoryStorage()
    limiter._storage = fresh
    limiter._limiter = FixedWindowRateLimiter(fresh)

    yield

    limiter._storage = saved_storage
    limiter._limiter = saved_limiter


@pytest.fixture
def api_client() -> TestClient:
    """内存中调用 ASGI，无需启动 uvicorn（类比不启 Tomcat 跑 Spring 测试）。"""
    return TestClient(app)
