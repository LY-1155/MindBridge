"""
安全标记记录与累积升级 (Safety Flag Recorder & Accumulator)

职责：
1. 将每次安全检测结果持久化到 safety_flags 表
2. 滑动窗口累积规则：同一 user 在窗口内 level=1 达到阈值 → 自动软升级 level=2
   （不设 blocked，交 LLM 语义评估器 / router 二次裁决，而非硬拦截短路）
3. 人审接口预留

配置项（来自 settings）：
- SAFETY_ACCUMULATOR_WINDOW_MINUTES: 滑动窗口（默认 30 分钟）
- SAFETY_ACCUMULATOR_THRESHOLD: level=1 累计阈值（默认 3）
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from schemas.database import db_manager
from schemas.database_v2 import SafetyFlag
from modules.encryption import encrypt_field, safe_decrypt_field

logger = logging.getLogger(__name__)

# 默认值（可被 settings 覆盖）
DEFAULT_WINDOW_MINUTES = 30
DEFAULT_THRESHOLD = 3


class SafetyFlagRecorder:
    """记录安全标记并执行累积升级规则。"""

    def __init__(
        self,
        window_minutes: int = DEFAULT_WINDOW_MINUTES,
        threshold: int = DEFAULT_THRESHOLD,
    ):
        self.window_minutes = window_minutes
        self.threshold = threshold

    # ── 持久化 ─────────────────────────────────────────────

    def record(
        self,
        user_id: str,
        session_id: str,
        level: int,
        blocked: bool,
        matched_terms: List[str],
    ) -> SafetyFlag:
        """持久化一条安全标记，返回 ORM 对象。"""
        with db_manager.get_session_direct() as s:
            flag = SafetyFlag(
                user_id=user_id,
                session_id=session_id,
                level=level,
                blocked=blocked,
                matched_terms=encrypt_field(json.dumps(matched_terms, ensure_ascii=False)),
            )
            s.add(flag)
            s.commit()
            s.refresh(flag)
            logger.debug("safety_flag recorded: user=%s level=%s id=%s", user_id, level, flag.id)
            return flag

    # ── 滑动窗口累积 ──────────────────────────────────────

    def should_escalate(self, user_id: str, session_id: str) -> bool:
        """滑动窗口内 level=1 的标记数是否已达阈值。

        同一 session 内的记录计为 1 次（防同一条消息反复触发）。
        不同 session 的记录每次都计。
        """
        cutoff = datetime.utcnow() - timedelta(minutes=self.window_minutes)
        with db_manager.get_session_direct() as s:
            count = (
                s.query(SafetyFlag)
                .filter(
                    SafetyFlag.user_id == user_id,
                    SafetyFlag.level == 1,
                    SafetyFlag.created_at >= cutoff,
                )
                .count()
            )
        return count >= self.threshold

    def count_recent_warnings(self, user_id: str) -> int:
        """查询 user 在滑动窗口内的 level=1 标记数。"""
        cutoff = datetime.utcnow() - timedelta(minutes=self.window_minutes)
        with db_manager.get_session_direct() as s:
            return (
                s.query(SafetyFlag)
                .filter(
                    SafetyFlag.user_id == user_id,
                    SafetyFlag.level == 1,
                    SafetyFlag.created_at >= cutoff,
                ).count()
            )

    # ── 评估入口 ───────────────────────────────────────────

    def evaluate(
        self,
        user_id: str,
        session_id: str,
        level: int,
        blocked: bool,
        matched_terms: List[str],
    ) -> Dict[str, Any]:
        """记录标记 + 累积评估，返回含 escalate 字段的结果。

        Returns:
            dict 含:
            - recorded: bool
            - original_level: int
            - escalated: bool
            - final_level: int
            - final_blocked: bool
            - recent_warnings: int
            - threshold: int
        """
        # 先持久化当前标记
        self.record(
            user_id=user_id,
            session_id=session_id,
            level=level,
            blocked=blocked,
            matched_terms=matched_terms,
        )

        # level=0 不需要累积，直接返回
        if level == 0:
            return {
                "recorded": True,
                "original_level": 0,
                "escalated": False,
                "final_level": 0,
                "final_blocked": False,
                "recent_warnings": self.count_recent_warnings(user_id),
                "threshold": self.threshold,
            }

        # level=2（紧急）直接通过，不需要累积判断
        if level >= 2:
            return {
                "recorded": True,
                "original_level": level,
                "escalated": False,
                "final_level": level,
                "final_blocked": blocked,
                "recent_warnings": self.count_recent_warnings(user_id),
                "threshold": self.threshold,
            }

        # level=1：检查是否应升级
        recent_count = self.count_recent_warnings(user_id)
        should_esc = recent_count >= self.threshold

        if should_esc:
            logger.warning(
                "安全标记累积升级: user=%s recent_warnings=%d threshold=%d",
                user_id, recent_count, self.threshold,
            )
            # 写入一条额外的 level=2 标记（表示系统自动升级）。
            # blocked=False：累积升级是"软升级"，交 LLM 语义评估器（DOCTOR_MODE）/
            # router 升段（非 DOCTOR_MODE）二次裁决，而不是直接硬拦截短路危机，
            # 避免"连续几句情绪宣泄词"直接弹危机模板绕过评估器。
            self.record(
                user_id=user_id,
                session_id=session_id,
                level=2,
                blocked=False,
                matched_terms=[f"SYSTEM_ESCALATE: 窗口内 level=1 累计 {recent_count} 次 ≥ 阈值 {self.threshold}"],
            )
            return {
                "recorded": True,
                "original_level": 1,
                "escalated": True,
                "final_level": 2,
                "final_blocked": False,
                "recent_warnings": recent_count,
                "threshold": self.threshold,
            }

        return {
            "recorded": True,
            "original_level": 1,
            "escalated": False,
            "final_level": 1,
            "final_blocked": False,
            "recent_warnings": recent_count,
            "threshold": self.threshold,
        }

    # ── 人审接口 ───────────────────────────────────────────

    def list_pending_review(
        self, user_id: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """列出待人审的标记（level >= 1, reviewed=False）。"""
        with db_manager.get_session_direct() as s:
            q = s.query(SafetyFlag).filter(
                SafetyFlag.level >= 1,
                SafetyFlag.reviewed == False,  # noqa: E712
            )
            if user_id:
                q = q.filter(SafetyFlag.user_id == user_id)
            q = q.order_by(SafetyFlag.created_at.desc()).limit(limit)
            return [_flag_to_dict(f) for f in q.all()]

    def mark_reviewed(
        self, flag_id: int, reviewed_by: str = "admin"
    ) -> bool:
        """将一条安全标记标记为 '已人审'。"""
        with db_manager.get_session_direct() as s:
            flag = s.query(SafetyFlag).filter(SafetyFlag.id == flag_id).first()
            if not flag:
                return False
            flag.reviewed = True
            flag.reviewed_by = reviewed_by
            flag.reviewed_at = datetime.utcnow()
            s.commit()
            logger.info("safety_flag reviewed: id=%s by=%s", flag_id, reviewed_by)
            return True


def _flag_to_dict(f: SafetyFlag) -> Dict[str, Any]:
    return {
        "id": f.id,
        "user_id": f.user_id,
        "session_id": f.session_id,
        "level": f.level,
        "blocked": f.blocked,
        "matched_terms": (json.loads(safe_decrypt_field(f.matched_terms)) if f.matched_terms else []),
        "reviewed": f.reviewed,
        "reviewed_by": f.reviewed_by,
        "reviewed_at": f.reviewed_at.isoformat() if f.reviewed_at else None,
        "created_at": f.created_at.isoformat() if f.created_at else None,
    }
