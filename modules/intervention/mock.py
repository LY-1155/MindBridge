"""干预闭环 · Mock（并行开发占位；不接 LLM/RAG）。"""

from __future__ import annotations

from schemas.contracts import InterventionRequest, InterventionResult


class MockInterventionService:
    """供 MOCK_INTERVENTION=true 时使用；契约测试与 UI 联调不应依赖真实生成逻辑。"""

    def intervene(self, req: InterventionRequest) -> InterventionResult:
        route = (req.route or {}).get("route") or "comfort"
        return InterventionResult(
            reply=f"[mock-{route}] 收到，我们会陪伴你一步步处理。",
            empathy="我理解你现在不容易。",
            suggestion="可以先试着做一次缓慢深呼吸。",
            action_items=["深呼吸 3 次", "记录一件今天的小事"],
            chain_of_thought="mock → 固定模板",
            emergency_triggered=route == "crisis",
            meta={"implementation": "mock"},
        )
