"""智能路由 · Stub（简单规则示例，可被 ML 替代）"""

from __future__ import annotations

from schemas.contracts import RouteDecision, RouteRequest


class StubRouterService:
    def route(self, req: RouteRequest) -> RouteDecision:
        emotion = req.emotion or {}
        risk = float(emotion.get("risk") or 0.0)
        if risk >= 0.7:
            return RouteDecision(route="crisis", reason="risk>=0.7", confidence=0.9)
        if 0.5 <= risk < 0.7:
            return RouteDecision(route="comfort", reason="risk in [0.5,0.7)", confidence=0.85)
        if 0.1 <= risk < 0.5:
            return RouteDecision(route="knowledge", reason="risk in [0.1,0.5)", confidence=0.85)
        return RouteDecision(route="general", reason="default", confidence=0.85)
