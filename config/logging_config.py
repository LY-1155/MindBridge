"""
结构化日志配置
=============

Gap #20：统一日志格式，注入 request_id / user_id / session_id 上下文。

使用方式：
  from config.logging_config import get_logger
  logger = get_logger(__name__)
  logger.info("message", extra={"user_id": "...", "session_id": "..."})

请求级别的 request_id 由中间件自动注入到 logging.LogRecord 中。
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from typing import Optional

# ContextVar：跨异步任务传递请求上下文
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
_user_id_ctx: ContextVar[str] = ContextVar("user_id", default="-")
_session_id_ctx: ContextVar[str] = ContextVar("session_id", default="-")


def set_request_context(
    request_id: str = "",
    user_id: str = "",
    session_id: str = "",
) -> None:
    """设置当前请求的上下文变量。"""
    if request_id:
        _request_id_ctx.set(request_id)
    if user_id:
        _user_id_ctx.set(user_id)
    if session_id:
        _session_id_ctx.set(session_id)


def get_request_id() -> str:
    """获取当前请求 ID。"""
    return _request_id_ctx.get()


class RequestContextFilter(logging.Filter):
    """将 ContextVar 中的 request/user/session ID 注入到 LogRecord。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_ctx.get("-")
        record.user_id = _user_id_ctx.get("-")
        record.session_id = _session_id_ctx.get("-")
        return True


def get_logger(name: str) -> logging.Logger:
    """获取配置了 request_id 过滤器的 logger。"""
    logger = logging.getLogger(name)
    if not any(isinstance(f, RequestContextFilter) for f in logger.filters):
        logger.addFilter(RequestContextFilter())
    return logger


def configure_structured_logging(debug: bool = False) -> None:
    """配置根 logger 的结构化格式。

    格式：timestamp LEVEL [request_id] [user_id] [session_id] name: message

    在 api/main.py 的 startup 中调用一次即可。
    """
    level = logging.DEBUG if debug else logging.INFO
    fmt = (
        "%(asctime)s %(levelname)-8s "
        "[%(request_id)s] [%(user_id)s] [%(session_id)s] "
        "%(name)s: %(message)s"
    )
    root = logging.getLogger()
    root.setLevel(level)

    # 清除已有 handler，避免重复
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    handler.addFilter(RequestContextFilter())
    root.addHandler(handler)

    # 挂载 CRITICAL 告警 webhook handler
    from modules.alert_webhook import WebhookLogHandler
    from config.settings import settings as _s

    webhook_handler = WebhookLogHandler(
        dingtalk_url=_s.ALERT_WEBHOOK_DINGTALK_URL,
        feishu_url=_s.ALERT_WEBHOOK_FEISHU_URL,
        enabled=_s.ALERT_WEBHOOK_ENABLED,
    )
    webhook_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s\n%(message)s"
    ))
    root.addHandler(webhook_handler)

    # 抑制第三方库噪音
    _suppress_noisy()


def _suppress_noisy() -> None:
    """抑制第三方库 DEBUG 日志噪音。"""
    for name in (
        "sqlalchemy.engine",
        "sqlalchemy.engine.Engine",
        "sqlalchemy.pool",
        "watchfiles",
        "jieba",
        "chromadb",
        "httpcore",
        "httpx",
        "openai._base_client",
        "urllib3",
        "asyncio",
        "PIL",
        "matplotlib",
        "fsspec",
        "filelock",
        "numexpr",
        "redis.connection",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
