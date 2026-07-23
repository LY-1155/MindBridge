"""
AI 标注：每条 AI 回复底部追加 "AI 辅助回复，非医疗诊断"（gap #12）

用法（在 HTTP 响应构造处调用）：
    from modules.ai_disclaimer import apply_disclaimer
    response = apply_disclaimer(intervention.get("reply", ""))
"""

from __future__ import annotations

AI_DISCLAIMER = "\n\n---\n*AI 辅助回复，非医疗诊断*"


def apply_disclaimer(reply: str) -> str:
    """在非空 AI 回复末尾追加免责声明，不重复追加。"""
    if not reply:
        return reply
    if reply.endswith(AI_DISCLAIMER):
        return reply
    return reply + AI_DISCLAIMER
