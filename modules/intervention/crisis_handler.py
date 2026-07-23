"""CrisisHandler：危机路由的话术模板分发，封装 EmergencyPushService"""

from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)

from schemas.contracts import InterventionRequest, InterventionResult
from modules.safety.emergency_push import (
    EmergencyPushService,
    EmergencyPushResult,
    get_emergency_push_service,
)


class CrisisHandler:
    """危机路由：调用 EmergencyPushService 产出一段确定性话术模板回复"""

    def __init__(self, push_service: Optional[EmergencyPushService] = None):
        self._push = push_service or get_emergency_push_service()

    def handle(self, req: InterventionRequest) -> InterventionResult:
        push_result: EmergencyPushResult = self._push.trigger(
            session_id=req.session_id,
            matched_terms=(req.safety or {}).get("matched_terms", []),
            user_text=req.user_text,
        )

        hotline = "拨打 400-161-9995"
        action_items = [hotline]

        # 冷却期内：使用简短话术而非空模板
        if not push_result.triggered:
            reason = getattr(push_result, "reason", "")
            logger.info("危机干预冷却期内: %s", reason)
            return InterventionResult(
                reply="⚠️ 系统检测到高风险内容。如果您正处于危机中，请立即拨打心理援助热线：400-161-9995 或 120。",
                empathy="",
                suggestion=hotline,
                action_items=action_items,
                chain_of_thought=None,
                emergency_triggered=False,
                meta={
                    "implementation": "crisis_cooldown",
                    "crisis_type": push_result.crisis_type,
                    "reason": reason,
                },
            )

        if push_result.triggered and push_result.crisis_type == "suicide":
            action_items.append("告诉身边信任的人")

        return InterventionResult(
            reply=push_result.template,
            empathy="",
            suggestion=hotline,
            action_items=action_items,
            chain_of_thought=None,
            emergency_triggered=True,
            meta={
                "implementation": "crisis_template",
                "crisis_type": push_result.crisis_type,
                "rescue_api_called": push_result.rescue_api_called,
            },
        )
