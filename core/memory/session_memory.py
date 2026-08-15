"""
会话记忆模块
===========

Gap #19 重构：去除内存 _sessions dict 和 _messages list。
Redis 做热数据（会话目录、元数据缓存、最近消息），MySQL 做持久化。

核心概念：
- SessionMetadata：会话基本信息的 Pydantic 模型
- EmotionRecord：情绪记录
- TherapySessionMemory：单个会话的读写接口
- SessionManager：会话生命周期管理（创建、查找、删除）
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

import logging

from pydantic import BaseModel, ConfigDict, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from config.settings import settings

logger = logging.getLogger(__name__)


# ── 数据模型 ──────────────────────────────────────────────────────

class SessionMetadata(BaseModel):
    """会话元数据"""
    model_config = ConfigDict(extra="ignore")

    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)
    message_count: int = 0
    emotion_history: List[Dict[str, Any]] = Field(default_factory=list)
    key_topics: List[str] = Field(default_factory=list)
    probed_dimensions: List[str] = Field(default_factory=list)  # 已探测的临床维度
    scale_state: Optional[Dict[str, Any]] = None
    scale_history: List[Dict[str, Any]] = Field(default_factory=list)

    # ── 医生模式字段（DOCTOR_MODE=true 时使用）──────────────────
    phase: str = "check_in"  # 当前 session 阶段：check_in / explore / interpret / intervene
    family_members: List[Dict[str, Any]] = Field(default_factory=list)
    #   family_members 条目示例：{"role": "妈妈", "label": "焦虑型", "noted_at": "..."}
    working_hypothesis: Optional[str] = None
    #   如 "孩子的不上学可能承担了转移父母冲突的功能"
    scid_flags: Dict[str, Any] = Field(default_factory=dict)
    #   静默 SCID 追踪：{"MDD": {"criteria_met": ["sleep", "anhedonia"], "count": 2}, ...}
    scid_interview_state: Optional[Dict[str, Any]] = None
    #   主动式 SCID 访谈状态机（模块/步骤/已确认条目等，见 scid_interview.py）
    safety_state: Optional[Dict[str, Any]] = None
    #   危机状态机（ADR-0013）：{"status": "NONE"|"PROBING"|"CRISIS", "probe_count": N, "denial_mark": bool}

    # ── 统一序列化（整份状态的唯一权威来源）──────────────────
    # 历史教训：曾有四份手写字段清单（session_memory 写、db_storage 写/读、
    # redis_storage 写/读）字段集合不一致，导致蒸馏临床状态
    # （phase/假设/家庭/SCID/危机状态机）从不落库。现在存储层只认
    # to_state()/from_state() 这一个序列化器，杜绝字段清单漂移。

    def to_state(self) -> Dict[str, Any]:
        """整份状态 → JSON 安全 dict（datetime 自动 isoformat，可直接 json.dumps）。"""
        return self.model_dump(mode="json")

    @classmethod
    def from_state(cls, data: Dict[str, Any]) -> "SessionMetadata":
        """JSON 安全 dict → SessionMetadata。

        model_validate 对缺失 key 自动填字段默认值（旧行/不完整 dict 也能安全重建），
        ConfigDict(extra="ignore") 保证新字段向前兼容。
        """
        return cls.model_validate(data)


class EmotionRecord(BaseModel):
    """情绪记录"""
    timestamp: datetime = Field(default_factory=datetime.now)
    primary_emotion: str
    intensity: int
    triggers: List[str] = Field(default_factory=list)
    context: Optional[str] = None


class SessionOwnershipError(Exception):
    """会话归属校验失败。"""


def _verify_session_ownership(session, user_id: str) -> None:
    """校验 session 是否属于该 user。空 user_id 则跳过。"""
    if not user_id:
        return
    owner = session.metadata.user_id
    if owner and owner != user_id:
        raise SessionOwnershipError(
            f"会话 {session.session_id} 不属于用户 {user_id}"
        )


# ── TherapySessionMemory ──────────────────────────────────────────

class TherapySessionMemory:
    """
    心理咨询会话记忆类

    Gap #19 重构要点：
    - 去除 _messages 列表：消息从 Redis 缓存/MySQL 按需读取
    - add_message 同步写 Redis + MySQL
    - scale_state 通过 save_scale_state() 显式持久化
    """

    def __init__(
        self,
        session_id: str,
        max_history_turns: int = None,
        use_database: bool = None,
        user_id: Optional[str] = None,
    ):
        self.session_id = session_id
        self.max_history_turns = max_history_turns or settings.MAX_HISTORY_TURNS
        self.use_database = (
            use_database if use_database is not None else settings.USE_DATABASE
        )

        self.metadata = SessionMetadata(
            session_id=session_id,
            user_id=user_id,
        )
        # 情绪记录保留内存列表（非热路径，量小）
        self._emotion_records: List[EmotionRecord] = []

        if self.use_database:
            self._load_from_database()

    # ── 加载 ────────────────────────────────────────────────────

    def _load_from_database(self) -> None:
        """从 Redis 缓存（优先）或 MySQL 加载会话数据。"""
        # 1) 尝试 Redis 缓存
        try:
            from core.memory.redis_storage import (
                load_session_meta,
                get_cached_messages,
                refresh_session_ttl,
            )
            meta = load_session_meta(self.session_id)
            if meta:
                self.metadata = SessionMetadata.from_state(
                    {**meta, "session_id": self.session_id}
                )
                logger.debug("[SCALE:LOAD] Redis HIT session=%s scale_state=%s",
                             self.session_id,
                             "present" if meta.get("scale_state") else "None")
                # 热数据命中：消息也从 Redis 获取
                cached_msgs = get_cached_messages(
                    self.session_id, self.max_history_turns * 2
                )
                if cached_msgs:
                    self._cached_messages = [
                        HumanMessage(content=m["content"])
                        if m["role"] == "user"
                        else AIMessage(content=m["content"])
                        for m in cached_msgs
                    ]
                refresh_session_ttl(self.session_id)
                # 情绪记录仍从 DB 加载（非热路径）
                self._load_emotions_from_db()
                return
        except Exception:
            pass

        # 2) 回退 MySQL
        try:
            from core.memory.db_storage import DatabaseStorage

            session_data = DatabaseStorage.load_session(self.session_id)
            if session_data:
                self.metadata = SessionMetadata.from_state(
                    {**session_data, "session_id": self.session_id}
                )
                # 回填 Redis 缓存
                self._sync_meta_to_redis()

                msgs = DatabaseStorage.get_messages(
                    self.session_id, self.max_history_turns * 2
                )
                if msgs:
                    self._cached_messages = list(msgs)
                    # 回填 Redis 消息缓存
                    self._sync_messages_to_redis(msgs)

            self._load_emotions_from_db()
        except Exception as e:
            logger.warning("从数据库加载会话失败: %s", e)

    def _load_emotions_from_db(self) -> None:
        """从 MySQL 加载情绪记录。"""
        try:
            from core.memory.db_storage import DatabaseStorage

            emotion_data = DatabaseStorage.get_emotion_records(self.session_id)
            for ed in emotion_data:
                record = EmotionRecord(
                    timestamp=ed["timestamp"],
                    primary_emotion=ed["primary_emotion"],
                    intensity=ed["intensity"],
                    triggers=ed.get("triggers", []),
                    context=ed.get("context"),
                )
                self._emotion_records.append(record)
        except Exception:
            pass

    # ── 持久化辅助 ──────────────────────────────────────────────

    def _sync_meta_to_redis(self) -> None:
        """将当前整份 metadata 状态写回 Redis（state 字段）。"""
        try:
            from core.memory.redis_storage import save_session_meta

            save_session_meta(self.session_id, self.metadata.to_state())
        except Exception:
            pass

    def _sync_messages_to_redis(self, messages: List[BaseMessage]) -> None:
        """将消息列表写回 Redis 缓存。"""
        try:
            from core.memory.redis_storage import cache_message

            # 先删再写（简单重建缓存）
            from core.memory.redis_storage import delete_cached_messages
            delete_cached_messages(self.session_id)

            for msg in messages:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                cache_message(self.session_id, role, msg.content)
        except Exception:
            pass

    def _save_to_database(self) -> None:
        """保存整份会话状态到 MySQL（state_json）+ 同步 Redis。"""
        if not self.use_database:
            return
        try:
            from core.memory.db_storage import DatabaseStorage

            DatabaseStorage.save_session(self.session_id, self.metadata.to_state())
            # 同步 Redis
            self._sync_meta_to_redis()
        except Exception as e:
            logger.warning("保存会话到数据库失败: %s", e)

    # ── scale_state 持久化 ──────────────────────────────────────

    def save_scale_state(self) -> None:
        """显式持久化 scale_state 到 Redis + MySQL。

        ScaleOrchestrator 修改 metadata.scale_state 后必须调用此方法，
        否则 process restart 后 scale 状态丢失。

        先写 Redis（热数据，量表流程依赖），再写 MySQL（冷持久化）。
        Redis 写入失败会重抛；MySQL 写入失败只记 warning，不阻断量表流程。
        """
        if not self.use_database:
            return

        # 1. Redis 优先 —— 量表流程的实时状态依赖它
        try:
            from core.memory.redis_storage import save_session_meta

            save_session_meta(self.session_id, self.metadata.to_state())
            logger.debug("[SCALE:SAVE] Redis OK session=%s scale=%s status=%s",
                         self.session_id,
                         (self.metadata.scale_state or {}).get("scale_name", "-"),
                         (self.metadata.scale_state or {}).get("status", "-"))
        except Exception as e:
            logger.error("save_scale_state Redis 写入失败: %s", e)
            raise  # Redis 是量表流程的关键依赖，失败必须暴露

        # 2. MySQL 兜底 —— 失败不阻断
        try:
            from core.memory.db_storage import DatabaseStorage

            DatabaseStorage.save_session(self.session_id, self.metadata.to_state())
        except Exception as e:
            logger.warning("save_scale_state MySQL 写入失败（Redis 已成功）: %s", e)

    # ── 消息操作 ────────────────────────────────────────────────

    def add_message(self, message: BaseMessage) -> None:
        """添加消息到内存；持久化模式下同步 MySQL + Redis 热缓存。

        use_database=false 时严格纯内存（消息不跨实例、不写 Redis），
        与 get_messages 的读取 gate 对称。历史问题：Redis 写不 gate 在
        use_database 上，导致测试/开发模式消息泄漏到 Redis，跨实例读到
        别的会话（或本会话上一轮）的消息。
        """
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        content = message.content

        # MySQL（持久化）
        if self.use_database:
            try:
                from core.memory.db_storage import DatabaseStorage

                DatabaseStorage.add_message(self.session_id, role, content)
            except Exception as e:
                logger.warning("保存消息到数据库失败: %s", e)

            # Redis（热缓存）
            try:
                from core.memory.redis_storage import cache_message

                cache_message(self.session_id, role, content)
            except Exception:
                pass

        # 纯内存回退（无 Redis / 无 DB 场景）
        if not hasattr(self, "_messages"):
            self._messages: List[BaseMessage] = []
        self._messages.append(message)

        self.metadata.message_count += 1
        self.metadata.last_active = datetime.now()
        # 保存元数据（含 Redis TTL 刷新）
        self._save_to_database()

    def add_user_message(self, content: str) -> None:
        self.add_message(HumanMessage(content=content))

    def add_ai_message(self, content: str) -> None:
        self.add_message(AIMessage(content=content))

    def get_messages(self) -> List[BaseMessage]:
        """获取会话消息（持久化模式：Redis 缓存优先 → MySQL 回退；否则纯内存）。"""
        # 1) Redis 缓存（仅持久化模式读；use_database=false 时纯内存，不读残留 key）
        if self.use_database:
            try:
                from core.memory.redis_storage import get_cached_messages

                cached = get_cached_messages(
                    self.session_id, self.max_history_turns * 2
                )
                if cached:
                    return [
                        HumanMessage(content=m["content"])
                        if m["role"] == "user"
                        else AIMessage(content=m["content"])
                        for m in cached
                    ]
            except Exception:
                pass

        # 2) MySQL 回退
        if self.use_database:
            try:
                from core.memory.db_storage import DatabaseStorage

                return DatabaseStorage.get_messages(
                    self.session_id, self.max_history_turns * 2
                )
            except Exception:
                pass

        # 3) 纯内存回退（无 Redis / 无 DB 场景，如单元测试）
        msgs = getattr(self, "_messages", [])
        limit = self.max_history_turns * 2
        if len(msgs) > limit:
            return msgs[-limit:]
        return msgs

    def get_history_for_prompt(self) -> List[Dict[str, str]]:
        """获取用于 prompt 的对话历史。"""
        messages = self.get_messages()
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history

    # ── 情绪记录 ────────────────────────────────────────────────

    def add_emotion_record(self, record: EmotionRecord) -> None:
        """添加情绪记录（内存 + MySQL）。"""
        self._emotion_records.append(record)

        if self.use_database:
            try:
                from core.memory.db_storage import DatabaseStorage

                DatabaseStorage.add_emotion_record(
                    self.session_id,
                    record.primary_emotion,
                    float(record.intensity),
                    record.triggers,
                    record.context,
                    user_id=self.metadata.user_id or "",
                )
            except Exception as e:
                logger.warning("保存情绪记录到数据库失败: %s", e)

        self.metadata.emotion_history.append(
            {
                "timestamp": record.timestamp.isoformat(),
                "emotion": record.primary_emotion,
                "intensity": record.intensity,
            }
        )

    def get_emotion_trend(self, last_n: int = 5) -> Dict[str, Any]:
        """获取情绪趋势。"""
        records = self._emotion_records[-last_n:] if self._emotion_records else []
        if not records:
            return {"trend": "stable", "average_intensity": 0}

        intensities = [r.intensity for r in records]
        avg_intensity = sum(intensities) / len(intensities)

        if len(intensities) >= 2:
            if intensities[-1] > intensities[0]:
                trend = "increasing"
            elif intensities[-1] < intensities[0]:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "average_intensity": avg_intensity,
            "recent_emotions": [r.primary_emotion for r in records],
        }

    def add_key_topic(self, topic: str) -> None:
        """添加关键话题。"""
        if topic not in self.metadata.key_topics:
            self.metadata.key_topics.append(topic)
            self._save_to_database()

    def add_probed_dimension(self, dimension: str) -> None:
        """记录已探测的临床维度（用于防止重复提问）。

        维度名如 "时间线"、"频率"、"严重度"、"睡眠"、"精力"、"身体"。
        """
        if dimension not in self.metadata.probed_dimensions:
            self.metadata.probed_dimensions.append(dimension)
            self._save_to_database()

    def get_probed_dimensions(self) -> List[str]:
        """获取已探测的临床维度列表。"""
        return list(self.metadata.probed_dimensions)

    # ── 医生模式便利方法 ──────────────────────────────────────

    def update_phase(self, phase: str) -> None:
        """更新 session 阶段并持久化。"""
        valid_phases = {"check_in", "explore", "interpret", "intervene"}
        if phase not in valid_phases:
            logger.warning("无效的 phase: %s，使用 check_in", phase)
            phase = "check_in"
        self.metadata.phase = phase
        self._save_to_database()

    def update_hypothesis(self, hypothesis: Optional[str]) -> None:
        """更新工作假设并持久化。传 None 表示清除。"""
        self.metadata.working_hypothesis = hypothesis
        self._save_to_database()

    def add_family_member(self, role: str, label: str = "") -> None:
        """记录一个家庭成员。role 如 '妈妈'、'爸爸'、'孩子'。"""
        for m in self.metadata.family_members:
            if m.get("role") == role:
                if label:
                    m["label"] = label
                return
        self.metadata.family_members.append(
            {"role": role, "label": label, "noted_at": datetime.now().isoformat()}
        )
        self._save_to_database()

    def update_scid_flags(self, flags: Dict[str, Any]) -> None:
        """更新 SCID 追踪数据并持久化（合并模式）。"""
        existing = dict(self.metadata.scid_flags)
        for disorder, data in flags.items():
            if disorder in existing:
                existing_criteria = set(existing[disorder].get("criteria_met", []))
                new_criteria = set(data.get("criteria_met", []))
                merged = sorted(existing_criteria | new_criteria)
                existing[disorder]["criteria_met"] = merged
                existing[disorder]["count"] = len(merged)
            else:
                criteria = data.get("criteria_met", [])
                existing[disorder] = {
                    "criteria_met": criteria,
                    "count": len(criteria),
                }
        self.metadata.scid_flags = existing
        self._save_to_database()

    def update_scid_interview_state(self, state: Optional[Dict[str, Any]]) -> None:
        """更新主动式 SCID 访谈状态机并持久化。

        state 形如 {"module": "MDD", "status": "active", "step": "gate", ...}
        （见 modules/assessment/scid_interview.py 的 _new_state）。
        传 None 表示清除（会话重建/重置）。
        """
        self.metadata.scid_interview_state = state
        self._save_to_database()

    def update_safety_state(self, state: Optional[Dict[str, Any]]) -> None:
        """更新危机状态机（ADR-0013）并持久化。

        state 形如 {"status": "NONE"|"PROBING"|"CRISIS", "probe_count": N, "denial_mark": bool}。
        """
        self.metadata.safety_state = state
        self._save_to_database()

    def get_assessor_context(self) -> str:
        """构建注入 prompt 的评估上下文文本。"""
        parts = [f"**当前会话阶段**：{self.metadata.phase}"]
        if self.metadata.working_hypothesis:
            parts.append(f"**当前工作假设**：{self.metadata.working_hypothesis}")
        if self.metadata.family_members:
            members_text = "、".join(
                f"{m['role']}" + (f"（{m['label']}）" if m.get("label") else "")
                for m in self.metadata.family_members
            )
            parts.append(f"**已识别家庭成员**：{members_text}")
        return "\n".join(parts)

    def get_context_summary(self) -> str:
        """获取上下文摘要。"""
        parts = []
        if self.metadata.key_topics:
            parts.append(f"讨论的主要话题：{', '.join(self.metadata.key_topics)}")
        if self._emotion_records:
            latest = self._emotion_records[-1]
            parts.append(
                f"最近情绪状态：{latest.primary_emotion}"
                f"（强度：{latest.intensity}/10）"
            )
        parts.append(f"对话轮数：{self.metadata.message_count // 2}")
        return "\n".join(parts)

    def clear(self) -> None:
        """清空会话（Redis + MySQL）。"""
        # Redis 缓存
        try:
            from core.memory.redis_storage import delete_session as redis_delete
            redis_delete(self.session_id)
        except Exception:
            pass

        # MySQL
        if self.use_database:
            try:
                from core.memory.db_storage import DatabaseStorage
                DatabaseStorage.delete_session(self.session_id)
            except Exception as e:
                logger.warning("从数据库删除会话失败: %s", e)

        self._messages = []
        self._emotion_records = []
        self.metadata = SessionMetadata(session_id=self.session_id)

    # 兼容旧调用：emotion_records 仍可从内存列表读取
    @property
    def emotion_records(self) -> List[EmotionRecord]:
        return self._emotion_records


# ── SessionManager ────────────────────────────────────────────────

# 进程内内存 session 缓存（无 Redis/MySQL 环境下的跨轮状态 fallback）。
# Gap #19 移除的 _sessions dict 在此以"仅兜底"身份回归：
# 有可用存储时走 Redis/MySQL，存储不可用时才复用内存实例，
# 保证 USE_DATABASE=false 或存储故障场景下多轮状态不丢失。
_memory_sessions: Dict[str, "TherapySessionMemory"] = {}


class SessionManager:
    """
    会话管理器

    Gap #19 重构要点：
    - create_session() → 写 MySQL + Redis，只返回 session_id
    - get_session() → 优先 Redis/MySQL；存储不可用时回退进程内内存缓存
    - 会话发现走 Redis SET 索引 + MySQL 回退
    - 会话过期由 Redis TTL 管理，不再有 cleanup_inactive_sessions()
    """

    @classmethod
    def create_session(cls, user_id: str = "", **kwargs) -> str:
        """创建新会话，返回 session_id。"""
        session_id = str(uuid.uuid4())[:8]

        use_db = kwargs.get("use_database", settings.USE_DATABASE)

        # 统一状态：整份 SessionMetadata 默认值 → to_state()，MySQL/Redis 同一序列化器
        state = SessionMetadata(session_id=session_id, user_id=user_id).to_state()

        # 写 MySQL
        if use_db:
            try:
                from core.memory.db_storage import DatabaseStorage
                DatabaseStorage.save_session(session_id, state)
            except Exception as e:
                logger.warning("create_session DB error: %s", e)

        # 写 Redis
        try:
            from core.memory.redis_storage import (
                save_session_meta,
                add_user_session,
            )
            save_session_meta(session_id, state)
            if user_id:
                add_user_session(user_id, session_id)
        except Exception:
            pass

        return session_id

    @classmethod
    def get_session(
        cls, session_id: str, user_id: str = "", **kwargs
    ) -> TherapySessionMemory:
        """获取会话。每次创建新实例，数据来自 Redis/MySQL。"""
        use_db = kwargs.get("use_database", settings.USE_DATABASE)

        # 如果使用数据库，检查会话是否存在
        if use_db:
            try:
                from core.memory.db_storage import DatabaseStorage
                if DatabaseStorage.session_exists(session_id):
                    session = TherapySessionMemory(
                        session_id=session_id,
                        use_database=True,
                        **{k: v for k, v in kwargs.items() if k != "use_database"},
                    )
                    _verify_session_ownership(session, user_id)
                    return session
            except SessionOwnershipError:
                raise
            except Exception as e:
                logger.warning("get_session DB error: %s", e)

        # 会话不存在或未使用 DB：回退进程内内存缓存
        # （保证无 Redis/MySQL 环境下多轮状态跨轮共享）
        cached = _memory_sessions.get(session_id)
        if cached is not None:
            _verify_session_ownership(cached, user_id)
            return cached

        session = TherapySessionMemory(
            session_id=session_id,
            user_id=user_id,
            use_database=use_db,
            **{k: v for k, v in kwargs.items() if k != "use_database"},
        )
        _memory_sessions[session_id] = session
        return session

    @classmethod
    def remove_session(cls, session_id: str) -> None:
        """删除会话（Redis + MySQL + 内存缓存）。"""
        # 内存缓存
        _memory_sessions.pop(session_id, None)

        # Redis
        try:
            from core.memory.redis_storage import delete_session as redis_delete
            redis_delete(session_id)
        except Exception:
            pass

        # MySQL
        if settings.USE_DATABASE:
            try:
                from core.memory.db_storage import DatabaseStorage
                DatabaseStorage.delete_session(session_id)
            except Exception as e:
                logger.warning("remove_session DB error: %s", e)

    @classmethod
    def delete_session(cls, session_id: str) -> None:
        """删除会话（remove_session 的别名）。"""
        cls.remove_session(session_id)

    @classmethod
    def get_active_sessions(cls) -> List[str]:
        """获取所有活跃会话 ID。优先 Redis → 回退 MySQL。"""
        # Redis 没有全局索引（设计取舍），直接从 MySQL 查
        if settings.USE_DATABASE:
            try:
                from core.memory.db_storage import DatabaseStorage
                return DatabaseStorage.list_sessions()
            except Exception:
                pass
        return []

    @classmethod
    def get_active_sessions_by_user(cls, user_id: str) -> List[str]:
        """获取指定用户的活跃会话 ID 列表。优先 Redis → 回退 MySQL。"""
        if not user_id:
            return []

        # 1) Redis 用户索引
        try:
            from core.memory.redis_storage import get_user_sessions
            sessions = get_user_sessions(user_id)
            if sessions:
                return sessions
        except Exception:
            pass

        # 2) MySQL 回退
        if settings.USE_DATABASE:
            try:
                from core.memory.db_storage import DatabaseStorage
                return DatabaseStorage.list_sessions_by_user(user_id)
            except Exception:
                pass

        return []
