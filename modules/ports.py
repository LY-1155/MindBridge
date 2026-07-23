"""
四模块边界接口（Protocol）
========================

具体实现可为 Mock / Stub / 日后真实实现；流水线只依赖 Protocol。
"""

from __future__ import annotations

from typing import Protocol

from schemas.contracts import (
    EmotionAnalyzeRequest,
    EmotionTags,
    InterventionRequest,
    InterventionResult,
    RouteDecision,
    RouteRequest,
    SafetyCheckRequest,
    SafetyCheckResult,
)


class SafetyPort(Protocol):
    def check(self, req: SafetyCheckRequest) -> SafetyCheckResult: ...


class EmotionPort(Protocol):
    def analyze(self, req: EmotionAnalyzeRequest) -> EmotionTags: ...


class RouterPort(Protocol):
    def route(self, req: RouteRequest) -> RouteDecision: ...


class InterventionPort(Protocol):
    """干预闭环对外唯一方法；输入输出即 schemas.contracts 中 Intervention* 模型。"""

    def intervene(self, req: InterventionRequest) -> InterventionResult: ...
