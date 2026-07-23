"""
紧急推送服务 (Emergency Push Service)

当触发高危拦截（contract level >= 2，即 filter 一级拦截）时：
1. 返回预定义的紧急响应话术模板
2. 模拟调用 "120 救助 API"（结构化日志输出 / 通知管理员）
3. 防重复触发：同一会话在冷却期内不重复推送

设计原则：
- 冷却机制按 session_id 维度，避免同一会话短时间内重复报警
- 话术按危机类型（suicide / violence / self_harm / crisis）分类
- 救援 API 以结构化日志输出，预留真实 HTTP 推送扩展点
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 危机话术模板
# ---------------------------------------------------------------------------

EMERGENCY_TEMPLATES: Dict[str, Dict[str, str]] = {
    "suicide": {
        "title": "生命安全紧急干预",
        "template": (
            "【紧急心理危机干预】\n\n"
            "您好，我们非常重视您当前的状态。您刚才提到了一些让我们非常担心的话语，"
            "这表明您现在可能正处于极大的痛苦之中。\n\n"
            "请记住：\n"
            "1. 您不是一个人在面对这一切，有很多人愿意帮助您。\n"
            "2. 此刻的痛苦是真实的，但它不会永远持续下去。\n"
            "3. 您值得被关心、被帮助。\n\n"
            "如果您有立即伤害自己的冲动，请马上拨打以下电话：\n"
            "   - 全国24小时心理危机干预热线：400-161-9995\n"
            "   - 北京心理危机研究与干预中心：010-82951332\n"
            "   - 希望24热线：400-161-9995\n"
            "   - 急救电话：120\n\n"
            "我们的系统已自动记录此情况，请您保持在线，"
            "您也可以直接告诉身边的人您的感受。您并不孤单。"
        ),
    },
    "violence": {
        "title": "暴力风险紧急干预",
        "template": (
            "【紧急安全干预】\n\n"
            "您好，我们注意到您当前表达的内容涉及对他人安全的潜在威胁。"
            "我们理解您可能正经历强烈的情绪波动，但请先暂停一下，深呼吸。\n\n"
            "请记住：\n"
            "1. 强烈的愤怒是正常的情绪，但伤害他人会造成不可挽回的后果。\n"
            "2. 先停下来，给自己一点时间冷静。\n"
            "3. 有很多方式可以表达情绪而不伤害任何人。\n\n"
            "如果您感觉无法控制自己的愤怒，请拨打：\n"
            "   - 全国24小时心理援助热线：400-161-9995\n"
            "   - 急救电话：120（如涉及人身伤害）\n"
            "   - 报警电话：110（如涉及紧急威胁）\n\n"
            "我们的系统已记录此情况。请您保持在线，我们愿意倾听您现在的感受。"
        ),
    },
    "self_harm": {
        "title": "自伤行为紧急干预",
        "template": (
            "【紧急心理危机干预】\n\n"
            "您好，我们注意到您当前表达的内容涉及自我伤害的风险。"
            "我们非常关心您的安全和健康。\n\n"
            "请先尝试以下方法：\n"
            "1. 深呼吸 5 次，慢慢吸气、慢慢呼气。\n"
            "2. 将注意力转移到周围环境中：说出您能看到的 5 样东西。\n"
            "3. 握住一块冰块或洗一把冷水脸，帮助转移感官注意力。\n"
            "4. 打电话给一位你信任的朋友或家人。\n\n"
            "如果需要立即帮助，请拨打：\n"
            "   - 全国24小时心理危机干预热线：400-161-9995\n"
            "   - 急救电话：120\n\n"
            "我们的系统已自动记录此情况，请保持在线，我们在这里陪着你。"
        ),
    },
    "crisis": {
        "title": "心理危机紧急干预",
        "template": (
            "【紧急心理支持】\n\n"
            "您好，我们注意到您正处于一个非常艰难的时刻。"
            "您所感受到的痛苦是真实且有分量的，我们不会轻视您的任何感受。\n\n"
            "请尝试：\n"
            "1. 慢慢地、深深地呼吸 5 次。\n"
            "2. 感受您的双脚踩在地面上的感觉。\n"
            "3. 如果可能，请告诉您身边的某个人您现在的感受。\n\n"
            "专业帮助热线：\n"
            "   - 全国24小时心理危机干预热线：400-161-9995\n"
            "   - 中国心理学会心理咨询热线：400-100-1995\n"
            "   - 急救电话：120\n\n"
            "我们的系统已记录此情况，我们会尽力为您提供支持。"
        ),
    },
}


# ---------------------------------------------------------------------------
# 危机分类
# ---------------------------------------------------------------------------

class CrisisType(str, Enum):
    SUICIDE = "suicide"
    VIOLENCE = "violence"
    SELF_HARM = "self_harm"
    CRISIS = "crisis"
    UNKNOWN = "unknown"


# 关键词 → 危机类型映射
_KEYWORD_TO_CRISIS: Dict[str, List[str]] = {
    CrisisType.SUICIDE: [
        "自杀", "自尽", "寻死", "想死", "不想活", "去死", "轻生",
        "结束生命", "活不下去", "kill myself", "suicide",
        "siwang", "紫砂",
    ],
    CrisisType.VIOLENCE: [
        "杀人", "暴力", "伤害", "攻击", "报复", "毁灭",
        "kill", "murder", "attack", "hurt others",
    ],
    CrisisType.SELF_HARM: [
        "自残", "割腕", "自伤", "自毁", "伤害自己",
        "self harm", "cut myself", "hurt myself",
    ],
}


def _classify_crisis(matched_terms: List[str]) -> str:
    """根据命中的敏感词判定危机类型"""
    terms_lower = [t.lower() for t in matched_terms]
    for crisis_type, keywords in _KEYWORD_TO_CRISIS.items():
        for kw in keywords:
            for term in terms_lower:
                if kw in term:
                    return crisis_type
    return CrisisType.CRISIS


# ---------------------------------------------------------------------------
# 推送结果
# ---------------------------------------------------------------------------

@dataclass
class EmergencyPushResult:
    triggered: bool
    session_id: str
    crisis_type: str
    matched_terms: List[str]
    user_text: str
    template: str = ""
    template_title: str = ""
    rescue_api_called: bool = False
    rescue_api_result: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""          # 未触发时的原因（如 cooldown）
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "session_id": self.session_id,
            "crisis_type": self.crisis_type,
            "matched_terms": self.matched_terms,
            "user_text": self.user_text,
            "template_title": self.template_title,
            "template": self.template,
            "rescue_api_called": self.rescue_api_called,
            "rescue_api_result": self.rescue_api_result,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# 紧急推送服务
# ---------------------------------------------------------------------------

class EmergencyPushService:
    """
    紧急推送服务

    - 维护按 session_id 维度的触发历史，冷却期内不重复推送
    - 根据命中的关键词自动分类危机类型
    - 返回对应话术模板
    - 模拟 "120 救助 API" 调用（结构化日志输出）
    """

    def __init__(self, cooldown_seconds: int = 300, enabled: bool = False,
                 rescue_api_url: str = "", rescue_api_key: str = ""):
        self.cooldown_seconds = cooldown_seconds
        self.enabled = enabled
        self.rescue_api_url = rescue_api_url
        self.rescue_api_key = rescue_api_key
        self.trigger_history: Dict[str, datetime] = {}
        self._push_count: int = 0

    # ── 防重复 ──────────────────────────────────────────────

    def _is_in_cooldown(self, session_id: str) -> bool:
        """检查是否在冷却期内"""
        if session_id not in self.trigger_history:
            return False
        elapsed = (datetime.now() - self.trigger_history[session_id]).total_seconds()
        return elapsed < self.cooldown_seconds

    def _mark_triggered(self, session_id: str) -> None:
        """记录触发时间"""
        self.trigger_history[session_id] = datetime.now()

    def _cooldown_remaining(self, session_id: str) -> float:
        """剩余冷却时间（秒）"""
        if session_id not in self.trigger_history:
            return 0.0
        elapsed = (datetime.now() - self.trigger_history[session_id]).total_seconds()
        return max(0.0, self.cooldown_seconds - elapsed)

    # ── 救助 API 模拟 ─────────────────────────────────────

    def _call_rescue_api(
        self,
        session_id: str,
        crisis_type: str,
        matched_terms: List[str],
        user_text: str,
    ) -> Dict[str, Any]:
        """调用救助 API（或 dry-run 日志输出）。

        由 EMERGENCY_PUSH_ENABLED 控制：
        - False（默认）：仅结构化日志，不发起真实 HTTP 请求
        - True：POST 到 EMERGENCY_RESCUE_API_URL，携带 API key
        """
        self._push_count += 1
        alert_id = f"EMG-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._push_count:04d}"
        payload = {
            "alert_id": alert_id,
            "alert_type": "emergency_push",
            "crisis_type": crisis_type,
            "session_id": session_id,
            "matched_terms": matched_terms,
            "user_text_snippet": user_text[:200],
            "timestamp": datetime.now().isoformat(),
            "hotlines": [
                "全国24小时心理危机干预热线：400-161-9995",
                "急救电话：120",
            ],
        }

        # ── 结构化日志（始终输出） ──
        logger.warning(
            "========== 紧急推送触发 ==========\n"
            "  告警ID:     %(alert_id)s\n"
            "  危机类型:   %(crisis)s\n"
            "  会话ID:     %(session)s\n"
            "  命中词:     %(terms)s\n"
            "  用户文本:   %(text)s\n"
            "  模式:       %(mode)s\n"
            "=======================================",
            {
                "alert_id": alert_id,
                "crisis": crisis_type,
                "session": session_id,
                "terms": matched_terms,
                "text": user_text[:100],
                "mode": "production" if self.enabled else "dry-run",
            },
        )

        if not self.enabled or not self.rescue_api_url:
            return {
                "status": "dry_run_success",
                "alert_id": alert_id,
                "message": "dry-run 模式：已记录日志，未发起真实 API 调用",
            }

        # ── 生产模式：真实 HTTP POST ──
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.rescue_api_key}",
            }
            resp = httpx.post(
                self.rescue_api_url,
                json=payload,
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
            logger.info("救助 API 调用成功: status=%s alert_id=%s", resp.status_code, alert_id)
            return {
                "status": "success",
                "http_status": resp.status_code,
                "alert_id": alert_id,
                "message": "已通知救助系统",
                "response": resp.json() if resp.text else {},
            }
        except httpx.HTTPError as exc:
            logger.error("救助 API 调用失败: alert_id=%s error=%s", alert_id, exc)
            return {
                "status": "error",
                "alert_id": alert_id,
                "message": f"救助 API 调用失败: {exc}",
            }

    # ── 主触发入口 ─────────────────────────────────────────

    def trigger(
        self,
        session_id: str,
        matched_terms: List[str],
        user_text: str = "",
        crisis_type: Optional[str] = None,
    ) -> EmergencyPushResult:
        """
        触发紧急推送。

        参数：
            session_id: 会话标识，用于防重复
            matched_terms: 命中的敏感词列表
            user_text: 用户原始输入
            crisis_type: 危机类型（可选，自动分类）

        返回：
            EmergencyPushResult：包含是否触发、话术模板、API 调用结果
        """
        timestamp = datetime.now().isoformat()

        # 自动分类
        if crisis_type is None:
            crisis_type = _classify_crisis(matched_terms)

        # 检查冷却期
        if self._is_in_cooldown(session_id):
            remaining = self._cooldown_remaining(session_id)
            logger.info(
                "紧急推送冷却中，session=%s，剩余%.0f秒",
                session_id, remaining,
            )
            return EmergencyPushResult(
                triggered=False,
                session_id=session_id,
                crisis_type=crisis_type,
                matched_terms=matched_terms,
                user_text=user_text,
                reason=f"冷却期内，剩余 {remaining:.0f} 秒",
                timestamp=timestamp,
            )

        # 记录触发
        self._mark_triggered(session_id)

        # 获取话术模板
        template_data = EMERGENCY_TEMPLATES.get(
            crisis_type, EMERGENCY_TEMPLATES[CrisisType.CRISIS]
        )

        # 调用模拟救助 API
        api_result = self._call_rescue_api(
            session_id=session_id,
            crisis_type=crisis_type,
            matched_terms=matched_terms,
            user_text=user_text,
        )

        result = EmergencyPushResult(
            triggered=True,
            session_id=session_id,
            crisis_type=crisis_type,
            matched_terms=matched_terms,
            user_text=user_text,
            template=template_data["template"],
            template_title=template_data["title"],
            rescue_api_called=True,
            rescue_api_result=api_result,
            timestamp=timestamp,
        )

        logger.info(
            "紧急推送已触发: session=%s, crisis=%s, terms=%s",
            session_id, crisis_type, matched_terms,
        )

        return result

    # ── 管理接口 ───────────────────────────────────────────

    def get_history(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """获取触发历史"""
        if session_id:
            last = self.trigger_history.get(session_id)
            return {
                session_id: last.isoformat() if last else None,
                "in_cooldown": self._is_in_cooldown(session_id),
                "cooldown_remaining": self._cooldown_remaining(session_id),
            }
        return {
            session_id: dt.isoformat()
            for session_id, dt in self.trigger_history.items()
        }

    def clear_history(self, session_id: Optional[str] = None) -> int:
        """清除触发历史"""
        if session_id:
            removed = 1 if session_id in self.trigger_history else 0
            self.trigger_history.pop(session_id, None)
            return removed
        count = len(self.trigger_history)
        self.trigger_history.clear()
        return count

    def reset_cooldown(self, session_id: str) -> bool:
        """手动重置某会话的冷却期，允许其立即再次触发"""
        if session_id in self.trigger_history:
            del self.trigger_history[session_id]
            logger.info("已重置会话 %s 的紧急推送冷却期", session_id)
            return True
        return False


# ---------------------------------------------------------------------------
# 模块级单例
# ---------------------------------------------------------------------------

_push_service: Optional[EmergencyPushService] = None


def get_emergency_push_service() -> EmergencyPushService:
    """获取 EmergencyPushService 模块级单例（从 settings 读取配置）"""
    global _push_service
    if _push_service is None:
        _push_service = EmergencyPushService(
            cooldown_seconds=settings.EMERGENCY_PUSH_COOLDOWN_SECONDS,
            enabled=settings.EMERGENCY_PUSH_ENABLED,
            rescue_api_url=settings.EMERGENCY_RESCUE_API_URL,
            rescue_api_key=settings.EMERGENCY_RESCUE_API_KEY,
        )
        mode = "production" if settings.EMERGENCY_PUSH_ENABLED else "dry-run"
        logger.info(
            "EmergencyPushService 已初始化（冷却期=%ds, 模式=%s）",
            settings.EMERGENCY_PUSH_COOLDOWN_SECONDS, mode,
        )
    return _push_service


# ---------------------------------------------------------------------------
# 独立测试入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("紧急推送服务 - 功能测试")
    print("=" * 60)

    eps = EmergencyPushService(cooldown_seconds=10)  # 10秒冷却方便测试

    # ── 测试 1：触发推送 ──
    print("\n[测试 1] 首次触发（自杀关键词）")
    r1 = eps.trigger(
        session_id="test-session-001",
        matched_terms=["自杀", "不想活"],
        user_text="我真的不想活了，想去自杀",
    )
    print(f"  triggered: {r1.triggered}")
    print(f"  crisis_type: {r1.crisis_type}")
    print(f"  rescue_api_called: {r1.rescue_api_called}")
    print(f"  template_title: {r1.template_title}")
    assert r1.triggered is True
    assert r1.crisis_type == "suicide"
    assert r1.rescue_api_called is True
    assert len(r1.template) > 100
    print("  ✓ 通过")

    # ── 测试 2：冷却期内不重复触发 ──
    print("\n[测试 2] 冷却期内重复触发（应被阻止）")
    r2 = eps.trigger(
        session_id="test-session-001",
        matched_terms=["自杀"],
        user_text="我还是想死",
    )
    print(f"  triggered: {r2.triggered}")
    print(f"  reason: {r2.reason}")
    assert r2.triggered is False
    assert "冷却期" in r2.reason
    print("  ✓ 通过")

    # ── 测试 3：不同会话独立触发 ──
    print("\n[测试 3] 不同会话应独立触发")
    r3 = eps.trigger(
        session_id="test-session-002",
        matched_terms=["暴力", "杀人"],
        user_text="我想杀人",
    )
    print(f"  triggered: {r3.triggered}")
    print(f"  crisis_type: {r3.crisis_type}")
    assert r3.triggered is True
    assert r3.crisis_type == "violence"
    print("  ✓ 通过")

    # ── 测试 4：未知危机类型回退 ──
    print("\n[测试 4] 未知关键词回退到 crisis 类型")
    r4 = eps.trigger(
        session_id="test-session-003",
        matched_terms=["紧急情况"],
        user_text="有紧急情况",
    )
    print(f"  triggered: {r4.triggered}")
    print(f"  crisis_type: {r4.crisis_type}")
    assert r4.triggered is True
    assert r4.crisis_type == "crisis"
    print("  ✓ 通过")

    # ── 测试 5：self_harm 分类 ──
    print("\n[测试 5] 自伤关键词分类")
    r5 = eps.trigger(
        session_id="test-session-004",
        matched_terms=["自残", "割腕"],
        user_text="我想自残",
    )
    print(f"  triggered: {r5.triggered}")
    print(f"  crisis_type: {r5.crisis_type}")
    assert r5.triggered is True
    assert r5.crisis_type == "self_harm"
    print("  ✓ 通过")

    # ── 测试 6：reset_cooldown ──
    print("\n[测试 6] 重置冷却期后再触发")
    eps.reset_cooldown("test-session-001")
    r6 = eps.trigger(
        session_id="test-session-001",
        matched_terms=["自杀"],
        user_text="还是想死",
    )
    print(f"  triggered: {r6.triggered}")
    assert r6.triggered is True
    print("  ✓ 通过")

    # ── 测试 7：to_dict ──
    print("\n[测试 7] to_dict 序列化")
    d = r1.to_dict()
    assert d["triggered"] is True
    assert d["crisis_type"] == "suicide"
    assert "template" in d
    assert "rescue_api_result" in d
    print(f"  keys: {list(d.keys())}")
    print("  ✓ 通过")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
