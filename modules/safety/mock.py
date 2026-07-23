"""输入与安全过滤 · Mock（未实现时可联调下游）"""

from __future__ import annotations

from schemas.contracts import SafetyCheckRequest, SafetyCheckResult


class MockSafetyService:
    def check(self, req: SafetyCheckRequest) -> SafetyCheckResult:
        return SafetyCheckResult(
            level=0,
            blocked=False,
            matched_terms=[],
            meta={"implementation": "mock"},
        )
