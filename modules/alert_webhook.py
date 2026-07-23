"""
CRITICAL 告警 Webhook 模块
=========================

Gap #21：将 CRITICAL 级别日志事件推送到钉钉/飞书机器人 webhook。

设计：
- WebhookLogHandler 作为 logging.Handler 挂载到 root logger
- 仅 CRITICAL (level=50) 事件触发推送
- 支持钉钉 markdown 和飞书 interactive card 两种格式
- 去重：5 分钟内相同消息不重复推送
- 非阻塞：通过线程池异步发送，不阻塞日志 emit
- 降级：webhook 请求失败时仅记录 warning，不影响主流程
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Set

import httpx

logger = logging.getLogger(__name__)

# ── 格式模板 ──────────────────────────────────────────────


def _build_dingtalk_payload(title: str, text: str, alert_id: str) -> dict:
    """构建钉钉 markdown 消息体。"""
    return {
        "msgtype": "markdown",
        "markdown": {
            "title": title,
            "text": (
                f"## {title}\n\n"
                f"{text}\n\n"
                f"---\n"
                f"*alert_id: {alert_id}*"
            ),
        },
    }


def _build_feishu_payload(title: str, text: str, alert_id: str) -> dict:
    """构建飞书 interactive card 消息体。"""
    return {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": "red",
            },
            "elements": [
                {
                    "tag": "markdown",
                    "content": text,
                },
                {
                    "tag": "note",
                    "elements": [
                        {"tag": "plain_text", "content": f"alert_id: {alert_id}"}
                    ],
                },
            ],
        },
    }


# ── Webhook Logging Handler ───────────────────────────────


class WebhookLogHandler(logging.Handler):
    """CRITICAL 级日志 → Webhook 告警推送。

    用法：
        handler = WebhookLogHandler(
            dingtalk_url=os.environ.get("ALERT_WEBHOOK_DINGTALK_URL", ""),
            feishu_url=os.environ.get("ALERT_WEBHOOK_FEISHU_URL", ""),
            enabled=os.environ.get("ALERT_WEBHOOK_ENABLED", "false").lower() == "true",
        )
        logging.getLogger().addHandler(handler)
    """

    def __init__(
        self,
        dingtalk_url: str = "",
        feishu_url: str = "",
        enabled: bool = False,
        dedup_window_seconds: int = 300,
        timeout: float = 5.0,
    ):
        super().__init__(level=logging.CRITICAL)
        self.dingtalk_url = dingtalk_url
        self.feishu_url = feishu_url
        self.enabled = enabled
        self.dedup_window_seconds = dedup_window_seconds
        self.timeout = timeout
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="alert-webhook")
        # 去重：存储 (dingtalk_signature, feishu_signature) → timestamp
        self._dedup: Dict[str, float] = {}

    def _dedup_key(self, *signatures: str) -> Optional[str]:
        """生成去重 key，并清理过期条目。超出窗口则返回 None（不拦截）。"""
        now = time.time()
        key = "|".join(s for s in signatures if s)
        if not key:
            return None
        # 清理过期条目
        expired = [k for k, ts in self._dedup.items() if now - ts > self.dedup_window_seconds]
        for k in expired:
            del self._dedup[k]
        if key in self._dedup:
            return key
        self._dedup[key] = now
        return None

    def emit(self, record: logging.LogRecord) -> None:
        """仅 CRITICAL 级别触发（handler 已设 level=CRITICAL）。"""
        if not self.enabled:
            return
        if not (self.dingtalk_url or self.feishu_url):
            return

        # 提交到线程池，不阻塞
        self._executor.submit(self._emit_sync, record)

    def _emit_sync(self, record: logging.LogRecord) -> None:
        """同步推送（在线程池中执行）。"""
        try:
            alert_id = str(uuid.uuid4())[:8]
            msg_text = self.format(record)
            # 取 record 的第一行作为标题
            title = msg_text.split("\n")[0][:80] if msg_text else "CRITICAL Alert"

            # 去重检查
            ding_sig = f"dt:{title}" if self.dingtalk_url else ""
            feishu_sig = f"fe:{title}" if self.feishu_url else ""
            if self._dedup_key(ding_sig, feishu_sig):
                return  # 窗口内重复

            # 钉钉推送
            if self.dingtalk_url:
                self._post(self.dingtalk_url, _build_dingtalk_payload(title, msg_text, alert_id), "dingtalk")

            # 飞书推送
            if self.feishu_url:
                self._post(self.feishu_url, _build_feishu_payload(title, msg_text, alert_id), "feishu")

        except Exception:
            logger.warning("Alert webhook emit failed", exc_info=True)

    def _post(self, url: str, payload: dict, label: str) -> None:
        """HTTP POST 到 webhook URL。"""
        try:
            resp = httpx.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            logger.debug("Alert webhook [%s] sent: status=%d", label, resp.status_code)
        except httpx.HTTPError as exc:
            logger.warning("Alert webhook [%s] failed: %s", label, exc)

    def close(self) -> None:
        """关闭 handler 并等待线程池退出。"""
        self._executor.shutdown(wait=True)
        super().close()


# ── 辅助：HTTP 测试用简单推送 ─────────────────────────────


class AlertWebhookService:
    """独立推送服务（非 logging handler 路径，供业务代码直接调用）。

    用于 emergency_push 等需要主动触发告警的场景。
    """

    def __init__(
        self,
        dingtalk_url: str = "",
        feishu_url: str = "",
        enabled: bool = False,
    ):
        self.dingtalk_url = dingtalk_url
        self.feishu_url = feishu_url
        self.enabled = enabled
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="alert-svc")

    def push(self, title: str, body: str, level: str = "CRITICAL") -> dict:
        """推送一条告警到已配置的 webhook。

        返回 {"dingtalk": ..., "feishu": ...} 表示各平台推送结果。
        """
        if not self.enabled:
            return {"dingtalk": "disabled", "feishu": "disabled", "alert_id": ""}

        alert_id = str(uuid.uuid4())[:8]
        full_title = f"[{level}] {title}"
        result = {"alert_id": alert_id, "dingtalk": "skipped", "feishu": "skipped"}

        try:
            if self.dingtalk_url:
                resp = httpx.post(
                    self.dingtalk_url,
                    json=_build_dingtalk_payload(full_title, body, alert_id),
                    headers={"Content-Type": "application/json"},
                    timeout=5.0,
                )
                result["dingtalk"] = f"status={resp.status_code}"
                resp.raise_for_status()
        except Exception as exc:
            result["dingtalk"] = f"error: {exc}"

        try:
            if self.feishu_url:
                resp = httpx.post(
                    self.feishu_url,
                    json=_build_feishu_payload(full_title, body, alert_id),
                    headers={"Content-Type": "application/json"},
                    timeout=5.0,
                )
                result["feishu"] = f"status={resp.status_code}"
                resp.raise_for_status()
        except Exception as exc:
            result["feishu"] = f"error: {exc}"

        return result
