"""
数据库模型 v2 — 生产就绪 Schema
=============================

替代 schemas/database.py（旧 demo schema），新增：
  - users：自然人主体
  - credentials：可插拔登录凭证
  - safety_flags：安全标记落库
  - scale_screenings：量表筛查记录

修改：
  - sessions：user_id NOT NULL
  - messages：content 支持加密长密文
  - emotion_records：intensity/risk 改为 FLOAT，新增 intent/modality_notes

使用 Alembic 管理迁移（migrations/），不使用 create_all/drop_all。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta

# 北京时间（UTC+8）
_BEIJING_TZ = timezone(timedelta(hours=8))


def _beijing_now() -> datetime:
    """返回当前北京时间（带时区）。"""
    return datetime.now(_BEIJING_TZ)

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# ---------------------------------------------------------------------------
# v2 新表
# ---------------------------------------------------------------------------

class User(Base):
    """自然人主体 — 系统的数据归属根节点。"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        String(36), unique=True, nullable=False, index=True,
        default=lambda: str(uuid.uuid4()),
        comment="永久业务标识 UUID",
    )
    display_name = Column(String(128), nullable=True, comment="展示名/昵称")
    status = Column(
        String(16), nullable=False, default="active",
        comment="active | disabled | deleted",
    )
    created_at = Column(DateTime, default=_beijing_now, comment="注册时间")
    updated_at = Column(DateTime, default=_beijing_now, onupdate=_beijing_now, comment="更新时间")
    deleted_at = Column(DateTime, nullable=True, comment="软删除时间")
    consent_at = Column(DateTime, nullable=True, comment="知情同意签署时间")
    consent_version = Column(String(16), nullable=True, comment="同意的协议版本号")

    # relationships
    credentials = relationship("Credential", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("SessionV2", back_populates="user", cascade="all, delete-orphan")
    safety_flags = relationship("SafetyFlag", back_populates="user", cascade="all, delete-orphan")
    scale_screenings = relationship("ScaleScreening", back_populates="user", cascade="all, delete-orphan")


class Credential(Base):
    """可插拔登录凭证 — 与 User 解耦，支持账密/手机/微信等。"""
    __tablename__ = "credentials"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        String(36), ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False, index=True, comment="用户 ID",
    )
    type = Column(
        String(16), nullable=False,
        comment="password | phone | wechat",
    )
    identifier = Column(
        String(128), nullable=False,
        comment="账号名 | 手机号 | openid",
    )
    secret = Column(
        String(256), nullable=True,
        comment="密码 bcrypt hash；OAuth 可为空",
    )
    verified_at = Column(DateTime, nullable=True, comment="验证时间")
    created_at = Column(DateTime, default=_beijing_now, comment="绑定时间")

    __table_args__ = (
        UniqueConstraint("type", "identifier", name="uq_credential_type_identifier"),
    )

    user = relationship("User", back_populates="credentials")


class SafetyFlag(Base):
    """安全标记落库 — 每条安全过滤结果持久化。"""
    __tablename__ = "safety_flags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(36), nullable=False, index=True,
        comment="关联会话 ID",
    )
    user_id = Column(
        String(36), ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False, index=True, comment="用户 ID",
    )
    level = Column(Integer, nullable=False, comment="0 通过 / 1 记录 / 2 紧急")
    blocked = Column(Boolean, nullable=False, default=False, comment="是否触发安全短路")
    matched_terms = Column(Text, nullable=True, comment="命中敏感词（AES 加密后明文存此）")
    reviewed = Column(Boolean, nullable=False, default=False, comment="是否人审")
    reviewed_by = Column(String(64), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=_beijing_now)

    user = relationship("User", back_populates="safety_flags")


class ScaleScreening(Base):
    """量表筛查记录 — 跨轮次对话式量表评估。"""
    __tablename__ = "scale_screenings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(36), nullable=False, index=True,
        comment="关联会话 ID",
    )
    user_id = Column(
        String(36), ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False, index=True, comment="用户 ID",
    )
    scale_type = Column(
        String(32), nullable=False,
        comment="PHQ-9 | GAD-7 | ...",
    )
    state = Column(
        String(16), nullable=False, default="in_progress",
        comment="in_progress | completed | abandoned",
    )
    responses = Column(Text, nullable=True, comment="逐题作答 JSON")
    scores = Column(Text, nullable=True, comment="各维度得分 JSON")
    total_score = Column(Float, nullable=True)
    interpretation = Column(Text, nullable=True, comment="锚点评级 JSON")
    created_at = Column(DateTime, default=_beijing_now)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="scale_screenings")


# ---------------------------------------------------------------------------
# v2 修改表（替代旧 Schema）
# ---------------------------------------------------------------------------

class SessionV2(Base):
    """会话表 v2 — user_id NOT NULL。"""
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(36), unique=True, nullable=False, index=True,
        default=lambda: str(uuid.uuid4()),
        comment="会话 ID（完整 UUID）",
    )
    user_id = Column(
        String(36), ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False, index=True, comment="用户 ID",
    )
    message_count = Column(Integer, default=0, comment="消息数量")
    key_topics = Column(Text, nullable=True, comment="关键话题 JSON")
    scale_state = Column(Text, nullable=True, comment="量表进行中状态 JSON")
    scale_history = Column(Text, nullable=True, comment="量表历史 JSON")
    created_at = Column(DateTime, default=_beijing_now, comment="创建时间")
    last_active = Column(DateTime, default=_beijing_now, onupdate=_beijing_now, comment="最后活跃")

    user = relationship("User", back_populates="sessions")
    messages = relationship("MessageV2", back_populates="session", cascade="all, delete-orphan")
    emotion_records = relationship("EmotionRecordV2", back_populates="session", cascade="all, delete-orphan")


class MessageV2(Base):
    """消息表 v2。"""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(36), ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False, index=True, comment="会话 ID",
    )
    role = Column(String(16), nullable=False, comment="user | assistant | system")
    content = Column(Text, nullable=False, comment="消息内容（存储层为 AES 密文）")
    created_at = Column(DateTime, default=_beijing_now)

    session = relationship("SessionV2", back_populates="messages")


class EmotionRecordV2(Base):
    """情绪记录表 v2 — 修正 intensity/risk 类型，新增 intent/modality_notes/user_id。"""
    __tablename__ = "emotion_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(36), ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False, index=True, comment="会话 ID",
    )
    user_id = Column(
        String(36), ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False, index=True, comment="用户 ID（跨会话趋势查询）",
    )
    primary_emotion = Column(String(32), nullable=False, comment="主情绪")
    intensity = Column(Float, nullable=False, comment="情绪强度 0~1")
    risk = Column(Float, nullable=True, default=0.0, comment="风险值 0~1")
    triggers = Column(Text, nullable=True, comment="触发因素 JSON")
    context = Column(Text, nullable=True, comment="情境上下文（存储层为 AES 密文）")
    intent = Column(String(32), nullable=True, comment="intent 标签")
    modality_notes = Column(Text, nullable=True, comment="模态附加信息 JSON")
    created_at = Column(DateTime, default=_beijing_now)

    session = relationship("SessionV2", back_populates="emotion_records")
