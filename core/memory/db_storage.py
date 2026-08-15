"""
数据库存储后端模块
=================

实现基于MySQL的会话存储功能。
将TherapySessionMemory的数据持久化到数据库。

主要功能：
1. 保存和加载会话
2. 保存和加载消息
3. 保存和加载情绪记录
"""

import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import contextmanager

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from schemas.database import db_manager
from schemas.database_v2 import SessionV2, MessageV2, EmotionRecordV2, ScaleScreening
from config.settings import settings
from modules.encryption import encrypt_field, safe_decrypt_field


class DatabaseStorage:
    """
    数据库存储后端
    
    提供会话数据的数据库持久化功能。
    所有方法都是类方法，不需要实例化。
    
    使用方式：
        DatabaseStorage.save_session(session_data)
        session = DatabaseStorage.load_session(session_id)
    """
    
    @classmethod
    @contextmanager
    def _get_db(cls):
        """
        获取数据库会话的上下文管理器
        
        自动处理提交和回滚。
        
        Yields:
            Session: SQLAlchemy会话对象
        """
        db = db_manager.get_session_direct()
        try:
            yield db
            db.commit()
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    @classmethod
    def save_session(cls, session_id: str, metadata: Dict[str, Any]) -> bool:
        """
        保存或更新会话信息

        如果会话已存在则更新，不存在则创建。

        metadata 为整份 SessionMetadata 状态（来自 SessionMetadata.to_state()）。
        state_json 列（AES 加密）是蒸馏状态的唯一权威来源；其余列
        （key_topics/scale_state/scale_history/message_count 等）为兼容列，
        保留写入以兼容旧读者（CLI、integration 脚本、chat 路由）。

        Args:
            session_id: 会话ID
            metadata: 整份会话状态字典

        Returns:
            bool: 保存是否成功
        """
        state_json = (
            encrypt_field(json.dumps(metadata, ensure_ascii=False))
            if metadata else None
        )
        with cls._get_db() as db:
            # 查找现有会话
            existing = db.query(SessionV2).filter(
                SessionV2.session_id == session_id
            ).first()

            if existing:
                # 更新现有会话
                existing.message_count = metadata.get("message_count", existing.message_count)
                existing.last_active = datetime.now()
                existing.state_json = state_json
                if metadata.get("key_topics"):
                    existing.key_topics = json.dumps(metadata["key_topics"], ensure_ascii=False)
                if metadata.get("scale_state"):
                    existing.scale_state = json.dumps(metadata["scale_state"], ensure_ascii=False)
                if metadata.get("scale_history"):
                    existing.scale_history = json.dumps(metadata["scale_history"], ensure_ascii=False)
                if metadata.get("user_id"):
                    existing.user_id = metadata["user_id"]
            else:
                # 创建新会话
                new_session = SessionV2(
                    session_id=session_id,
                    user_id=metadata.get("user_id"),
                    message_count=metadata.get("message_count", 0),
                    key_topics=json.dumps(metadata.get("key_topics", []), ensure_ascii=False),
                    scale_state=json.dumps(metadata.get("scale_state"), ensure_ascii=False) if metadata.get("scale_state") else None,
                    scale_history=json.dumps(metadata.get("scale_history", []), ensure_ascii=False),
                    state_json=state_json,
                    created_at=datetime.now(),
                    last_active=datetime.now()
                )
                db.add(new_session)

            return True
    
    @classmethod
    def load_session(cls, session_id: str) -> Optional[Dict[str, Any]]:
        """
        加载会话信息

        state_json（加密）存在时返回整份蒸馏状态（权威来源）；
        否则回退到兼容列拼出的部分 dict（旧行，缺失字段由 SessionMetadata
        默认值补齐）。

        Args:
            session_id: 会话ID

        Returns:
            Dict: 会话数据字典，如果不存在返回None
        """
        with cls._get_db() as db:
            session = db.query(SessionV2).filter(
                SessionV2.session_id == session_id
            ).first()

            if not session:
                return None

            if session.state_json:
                state = json.loads(safe_decrypt_field(session.state_json))
                if isinstance(state, dict):
                    return state

            # 兼容列回退（state_json 缺失的存量行）
            return {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "message_count": session.message_count,
                "key_topics": json.loads(session.key_topics) if session.key_topics else [],
                "scale_state": json.loads(session.scale_state) if session.scale_state else None,
                "scale_history": json.loads(session.scale_history) if session.scale_history else [],
                "created_at": session.created_at,
                "last_active": session.last_active
            }
    
    @classmethod
    def delete_session(cls, session_id: str) -> bool:
        """
        删除会话及其所有关联数据
        
        由于设置了级联删除，删除会话时会自动删除相关的消息和情绪记录。
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 删除是否成功
        """
        with cls._get_db() as db:
            session = db.query(SessionV2).filter(
                SessionV2.session_id == session_id
            ).first()
            
            if session:
                db.delete(session)
                return True
            return False
    
    @classmethod
    def add_message(cls, session_id: str, role: str, content: str) -> bool:
        """
        添加消息到数据库
        
        Args:
            session_id: 会话ID
            role: 消息角色（user/assistant/system）
            content: 消息内容
            
        Returns:
            bool: 添加是否成功
        """
        with cls._get_db() as db:
            # 检查会话是否存在
            session = db.query(SessionV2).filter(
                SessionV2.session_id == session_id
            ).first()
            
            if not session:
                return False
            
            # 创建消息记录（content 加密存储）
            message = MessageV2(
                session_id=session_id,
                role=role,
                content=encrypt_field(content),
                created_at=datetime.now()
            )
            db.add(message)
            
            # 更新会话的消息计数
            session.message_count += 1
            session.last_active = datetime.now()
            
            return True
    
    @classmethod
    def get_messages(cls, session_id: str, limit: int = None) -> List[BaseMessage]:
        """
        获取会话的所有消息
        
        Args:
            session_id: 会话ID
            limit: 最大返回数量（可选）
            
        Returns:
            List[BaseMessage]: LangChain消息对象列表
        """
        with cls._get_db() as db:
            query = db.query(MessageV2).filter(
                MessageV2.session_id == session_id
            ).order_by(MessageV2.created_at.asc())
            
            if limit:
                query = query.limit(limit)
            
            messages = query.all()
            
            # 转换为LangChain消息对象（content 解密）
            result = []
            for msg in messages:
                content = safe_decrypt_field(msg.content)
                if msg.role == "user":
                    result.append(HumanMessage(content=content))
                elif msg.role == "assistant":
                    result.append(AIMessage(content=content))
                elif msg.role == "system":
                    result.append(SystemMessage(content=content))
            
            return result
    
    @classmethod
    def add_emotion_record(
        cls,
        session_id: str,
        primary_emotion: str,
        intensity: float,
        triggers: List[str] = None,
        context: str = None,
        user_id: str = "",
        risk: float = 0.0,
    ) -> bool:
        """
        添加情绪记录到数据库（v2：intensity 为 float 0~1，user_id NOT NULL）。

        Args:
            session_id: 会话ID
            primary_emotion: 主要情绪
            intensity: 情绪强度（0~1）
            triggers: 触发因素列表
            context: 情境上下文
            user_id: 用户ID
            risk: 风险评分

        Returns:
            bool: 添加是否成功
        """
        with cls._get_db() as db:
            # 检查会话是否存在
            session = db.query(SessionV2).filter(
                SessionV2.session_id == session_id
            ).first()

            if not session:
                return False

            # 创建情绪记录（context 加密存储）
            record = EmotionRecordV2(
                session_id=session_id,
                user_id=user_id,
                primary_emotion=primary_emotion,
                intensity=intensity,
                risk=risk,
                triggers=json.dumps(triggers or [], ensure_ascii=False),
                context=encrypt_field(context),
                created_at=datetime.now()
            )
            db.add(record)
            
            return True
    
    @classmethod
    def get_emotion_records(cls, session_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        获取会话的情绪记录
        
        Args:
            session_id: 会话ID
            limit: 最大返回数量（可选）
            
        Returns:
            List[Dict]: 情绪记录字典列表
        """
        with cls._get_db() as db:
            query = db.query(EmotionRecordV2).filter(
                EmotionRecordV2.session_id == session_id
            ).order_by(EmotionRecordV2.created_at.asc())
            
            if limit:
                query = query.limit(limit)
            
            records = query.all()
            
            return [
                {
                    "primary_emotion": r.primary_emotion,
                    "intensity": r.intensity,
                    "triggers": json.loads(r.triggers) if r.triggers else [],
                    "context": safe_decrypt_field(r.context),
                    "timestamp": r.created_at
                }
                for r in records
            ]
    
    @classmethod
    def save_scale_screening(
        cls,
        session_id: str,
        scale_type: str,
        state: str,
        scores: list = None,
        total_score: float = None,
        responses: str = None,
        interpretation: str = None,
        user_id: str = "",
    ) -> bool:
        """写入 scale_screenings 表。

        state: in_progress | completed | abandoned
        """
        from datetime import datetime
        with cls._get_db() as db:
            screening = ScaleScreening(
                session_id=session_id,
                user_id=user_id,
                scale_type=scale_type,
                state=state,
                scores=json.dumps(scores) if scores else None,
                total_score=total_score,
                responses=responses,
                interpretation=interpretation,
                created_at=datetime.now(),
                completed_at=datetime.now() if state in ("completed", "abandoned") else None,
            )
            db.add(screening)
            return True

    @classmethod
    def list_sessions(cls, limit: int = 100) -> List[str]:
        """
        列出所有会话ID
        
        Args:
            limit: 最大返回数量
            
        Returns:
            List[str]: 会话ID列表
        """
        with cls._get_db() as db:
            sessions = db.query(SessionV2.session_id).order_by(
                SessionV2.last_active.desc()
            ).limit(limit).all()
            
            return [s.session_id for s in sessions]
    
    @classmethod
    def list_sessions_by_user(cls, user_id: str, limit: int = 100) -> List[str]:
        """
        列出指定用户的所有会话ID
        """
        if not user_id:
            return cls.list_sessions(limit)
        with cls._get_db() as db:
            sessions = db.query(SessionV2.session_id).filter(
                SessionV2.user_id == user_id
            ).order_by(
                SessionV2.last_active.desc()
            ).limit(limit).all()
            return [s.session_id for s in sessions]

    @classmethod
    def session_exists(cls, session_id: str) -> bool:
        """
        检查会话是否存在
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 是否存在
        """
        with cls._get_db() as db:
            count = db.query(SessionV2).filter(
                SessionV2.session_id == session_id
            ).count()
            return count > 0
