"""
Gap #21 CRITICAL 告警 Webhook 测试
=================================

验证：
- WebhookLogHandler 仅处理 CRITICAL 级别
- 钉钉/飞书 payload 格式正确
- 去重逻辑（5 分钟窗口）
- 禁用时不推送
- AlertWebhookService.push 返回结构
- settings 新增字段
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import time
import pytest

from modules.alert_webhook import (
    WebhookLogHandler,
    AlertWebhookService,
    _build_dingtalk_payload,
    _build_feishu_payload,
)


class TestDingtalkPayload:
    """钉钉 markdown 消息格式"""

    def test_structure(self):
        payload = _build_dingtalk_payload("紧急告警", "详细内容", "abc12345")
        assert payload["msgtype"] == "markdown"
        assert "紧急告警" in payload["markdown"]["title"]
        assert "详细内容" in payload["markdown"]["text"]
        assert "abc12345" in payload["markdown"]["text"]


class TestFeishuPayload:
    """飞书 interactive card 消息格式"""

    def test_structure(self):
        payload = _build_feishu_payload("紧急告警", "详细内容", "abc12345")
        assert payload["msg_type"] == "interactive"
        assert payload["card"]["header"]["template"] == "red"
        assert payload["card"]["header"]["title"]["content"] == "紧急告警"
        # 至少有一个 markdown element 和 一个 note
        elements = payload["card"]["elements"]
        assert len(elements) >= 1
        assert elements[0]["tag"] == "markdown"
        assert "详细内容" in elements[0]["content"]


class TestWebhookLogHandler:
    """CRITICAL 级别拦截 + 去重"""

    def test_level_critical_only(self):
        """handler level 设为 CRITICAL"""
        handler = WebhookLogHandler(enabled=True)
        assert handler.level == logging.CRITICAL

    def test_disabled_does_nothing(self):
        """enabled=False 时什么都不做"""
        handler = WebhookLogHandler(
            dingtalk_url="http://example.com/webhook",
            feishu_url="http://example.com/feishu",
            enabled=False,
        )
        record = logging.LogRecord(
            name="test", level=logging.CRITICAL, pathname="", lineno=0,
            msg="critical msg", args=(), exc_info=None,
        )
        # Should not raise
        handler.emit(record)
        handler.close()

    def test_no_urls_does_nothing(self):
        """没有 URL 配置时跳过"""
        handler = WebhookLogHandler(enabled=True)
        record = logging.LogRecord(
            name="test", level=logging.CRITICAL, pathname="", lineno=0,
            msg="critical msg", args=(), exc_info=None,
        )
        handler.emit(record)
        handler.close()

    def test_dedup_same_title(self):
        """5 分钟内相同标题去重"""
        handler = WebhookLogHandler(
            dingtalk_url="http://example.com/webhook",
            enabled=True,
            dedup_window_seconds=300,
        )
        # 第一次：不应被去重
        key = handler._dedup_key("dt:Same Alert")
        assert key is None

        # 第二次：应被去重
        key2 = handler._dedup_key("dt:Same Alert")
        assert key2 == "dt:Same Alert"

        handler.close()

    def test_dedup_expires_after_window(self):
        """超过去重窗口后可再次推送"""
        handler = WebhookLogHandler(
            dingtalk_url="http://example.com/webhook",
            enabled=True,
            dedup_window_seconds=0,  # immediate expiry
        )
        handler._dedup_key("dt:Old Alert")
        # 窗口为 0，立即过期
        time.sleep(0.01)
        key = handler._dedup_key("dt:Old Alert")
        assert key is None  # no longer dedup'd

        handler.close()


class TestAlertWebhookService:
    """独立推送服务"""

    def test_disabled_returns_skipped(self):
        svc = AlertWebhookService(
            dingtalk_url="http://example.com/webhook",
            enabled=False,
        )
        result = svc.push("Test", "Body")
        assert result["dingtalk"] == "disabled"
        assert result["feishu"] == "disabled"

    def test_enabled_with_urls_attempts_push(self):
        """enabled + URL 时尝试推送（因为是测试环境无真实 webhook，会返回 error）"""
        svc = AlertWebhookService(
            dingtalk_url="http://127.0.0.1:1/webhook",  # unreachable
            feishu_url="http://127.0.0.1:1/feishu",
            enabled=True,
        )
        result = svc.push("Test", "Body", level="CRITICAL")
        assert "alert_id" in result
        assert len(result["alert_id"]) == 8
        # unreachable URL → dingtalk/feishu should contain error
        assert "error" in result["dingtalk"] or "skipped" in result["dingtalk"]


class TestSettingsExist:
    """确认 settings 中新增了告警 webhook 字段"""

    def test_alert_webhook_settings(self):
        from config.settings import Settings
        s = Settings()
        assert hasattr(s, "ALERT_WEBHOOK_ENABLED")
        assert hasattr(s, "ALERT_WEBHOOK_DINGTALK_URL")
        assert hasattr(s, "ALERT_WEBHOOK_FEISHU_URL")
        assert s.ALERT_WEBHOOK_ENABLED is False  # default off
