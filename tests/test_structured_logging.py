"""
Gap #20 结构化日志测试
===================

验证：
- configure_structured_logging 初始化格式
- RequestContextFilter 注入 ContextVar
- set_request_context / get_request_id 正确性
- get_logger 携带 filter
- 第三方库噪音被抑制
- 生产代码无残留 print()
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import logging
import pytest

from config.logging_config import (
    configure_structured_logging,
    set_request_context,
    get_request_id,
    get_logger,
    RequestContextFilter,
    _request_id_ctx,
    _user_id_ctx,
    _session_id_ctx,
)


@pytest.fixture(autouse=True)
def _reset_logging():
    """每个测试前清理 logging 状态，避免互相污染。"""
    # 重置 ContextVar
    _request_id_ctx.set("-")
    _user_id_ctx.set("-")
    _session_id_ctx.set("-")
    # 清理根 logger
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    yield
    # 恢复默认
    root.setLevel(logging.WARNING)
    for h in list(root.handlers):
        root.removeHandler(h)


class TestRequestContextFilter:
    """ContextVar → LogRecord 注入"""

    def test_default_context_is_dash(self):
        f = RequestContextFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="", args=(), exc_info=None,
        )
        f.filter(record)
        assert record.request_id == "-"
        assert record.user_id == "-"
        assert record.session_id == "-"

    def test_set_context_injects_to_logrecord(self):
        set_request_context(
            request_id="req-abc",
            user_id="user-1",
            session_id="sess-xyz",
        )
        f = RequestContextFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="test", args=(), exc_info=None,
        )
        f.filter(record)
        assert record.request_id == "req-abc"
        assert record.user_id == "user-1"
        assert record.session_id == "sess-xyz"

    def test_set_context_partial(self):
        """只设置 request_id 时 user/session 保持默认。"""
        set_request_context(request_id="r1")
        f = RequestContextFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="x", args=(), exc_info=None,
        )
        f.filter(record)
        assert record.request_id == "r1"
        assert record.user_id == "-"
        assert record.session_id == "-"


class TestGetRequestId:
    def test_returns_dash_by_default(self):
        assert get_request_id() == "-"

    def test_returns_set_value(self):
        set_request_context(request_id="id-42")
        assert get_request_id() == "id-42"


class TestGetLogger:
    def test_returns_logger_with_filter(self):
        logger = get_logger("test.module.xyz")
        assert any(isinstance(f, RequestContextFilter) for f in logger.filters)

    def test_does_not_duplicate_filter(self):
        logger = get_logger("test.module.dup")
        count_before = sum(1 for f in logger.filters if isinstance(f, RequestContextFilter))
        get_logger("test.module.dup")
        count_after = sum(1 for f in logger.filters if isinstance(f, RequestContextFilter))
        assert count_before == count_after


class TestConfigureStructuredLogging:
    def test_handler_has_correct_format(self):
        configure_structured_logging(debug=False)
        root = logging.getLogger()
        assert len(root.handlers) == 1
        handler = root.handlers[0]
        fmt = handler.formatter._fmt
        assert "request_id" in fmt
        assert "user_id" in fmt
        assert "session_id" in fmt
        assert "asctime" in fmt

    def test_log_output_contains_request_id(self):
        configure_structured_logging(debug=False)
        set_request_context(request_id="out-test-1")
        logger = get_logger("test.output")

        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.getLogger().handlers[0].formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        logger.info("hello world")
        output = buf.getvalue()
        assert "out-test-1" in output
        assert "hello world" in output

    def test_noisy_libs_suppressed(self):
        configure_structured_logging(debug=False)
        for name in ("sqlalchemy.engine", "httpx", "openai._base_client"):
            assert logging.getLogger(name).level >= logging.WARNING


class TestNoResidualPrintInProduction:
    """验证生产模块无残留 print() 调用。"""

    _PROD_DIRS = ["pipeline", "api", "core", "config", "schemas"]
    _PROD_MODULE_FILES = [
        "modules/factory.py",
        "modules/runtime.py",
        "modules/ports.py",
        "modules/intervention/service.py",
        "modules/intervention/crisis_handler.py",
        "modules/intervention/generator.py",
        "modules/intervention/scale/orchestrator.py",
        "modules/intervention/scale/scorer.py",
        "modules/intervention/scale/models.py",
        "modules/emotion/keyword_engine.py",
        "modules/emotion/onnx_engine.py",
        "modules/emotion/service.py",  # stub
        "modules/router/router_service.py",
        "modules/safety/flag_recorder.py",
        "modules/user_service.py",
        "modules/auth_service.py",
        "modules/token_service.py",
        "modules/encryption.py",
        "modules/prompt_guard.py",
        "modules/auth_deps.py",
        "modules/rate_limit.py",
    ]

    @pytest.mark.parametrize("dir_name", _PROD_DIRS)
    def test_dir_has_no_print(self, dir_name):
        import subprocess
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        target = os.path.join(base, dir_name)
        if not os.path.isdir(target):
            pytest.skip(f"{dir_name} not found")
        result = subprocess.run(
            ["grep", "-rn", r"print(", target],
            capture_output=True, text=True,
            shell=True,
        )
        # grep returns 1 if no match — that's passing
        if result.returncode == 0:
            # Strip out comment-only occurrences
            lines = [l for l in result.stdout.splitlines() if not l.strip().startswith("#")]
            assert not lines, f"residual print() in {dir_name}:\n" + "\n".join(lines)

    @pytest.mark.parametrize("rel_path", _PROD_MODULE_FILES)
    def test_module_file_has_no_print(self, rel_path):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(base, rel_path)
        if not os.path.isfile(full_path):
            pytest.skip(f"{rel_path} not found")
        with open(full_path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                # Allow print() in `if __name__ == "__main__":` blocks and comments
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                    continue
                if "print(" in stripped:
                    raise AssertionError(
                        f"{rel_path}:{lineno} residual print(): {stripped[:80]}"
                    )
