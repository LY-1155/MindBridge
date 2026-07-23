"""智能路由 · Mock"""

from __future__ import annotations

from schemas.contracts import RouteDecision, RouteRequest


class MockRouterService:
    def route(self, req: RouteRequest) -> RouteDecision:
        return RouteDecision(
            route="comfort",
            reason="mock_default",
            confidence=1.0,
        )
