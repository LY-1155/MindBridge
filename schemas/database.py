"""
数据库模型模块
=============

定义MySQL数据库的表结构。
使用SQLAlchemy ORM进行数据库操作。

表结构说明：
- sessions: 会话表，存储会话基本信息
- messages: 消息表，存储对话历史
- emotion_records: 情绪记录表，存储情绪分析结果
"""

from datetime import datetime
import logging
from sqlalchemy import create_engine, Column, String, Text, Integer, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

from config.settings import settings

logger = logging.getLogger(__name__)


# SQLAlchemy基类
# 所有模型类都继承自这个基类
Base = declarative_base()


class SessionModel(Base):
    """
    会话表模型
    
    存储每个心理咨询会话的基本信息。
    
    字段说明：
        id: 自增主键
        session_id: 会话唯一标识符（业务ID）
        user_id: 用户ID（可选）
        message_count: 消息总数
        key_topics: 关键话题（JSON格式）
        created_at: 创建时间
        last_active: 最后活跃时间
    """
    __tablename__ = "sessions"
    
    # 主键
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 业务字段
    session_id = Column(String(64), unique=True, nullable=False, index=True, comment="会话ID")
    user_id = Column(String(64), nullable=True, index=True, comment="用户ID")
    message_count = Column(Integer, default=0, comment="消息数量")
    key_topics = Column(Text, nullable=True, comment="关键话题JSON")
    scale_history = Column(Text, nullable=True, comment="量表历史记录JSON")
    
    # 时间字段
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    last_active = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="最后活跃时间")
    
    # 关联关系：一个会话有多条消息和多条情绪记录
    messages = relationship("MessageModel", back_populates="session", cascade="all, delete-orphan")
    emotion_records = relationship("EmotionRecordModel", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Session(session_id={self.session_id})>"


class MessageModel(Base):
    """
    消息表模型

    存储对话中的每一条消息。

    字段说明：
        id: 自增主键
        session_id: 关联的会话ID（外键）
        user_id: 归属用户ID（v2 新增，用于所有权校验）
        role: 消息角色（user/assistant/system）
        content: 消息内容（AES 加密存储）
        created_at: 创建时间
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 外键关联到sessions表
    session_id = Column(String(64), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True, comment="会话ID")
    user_id = Column(String(64), nullable=True, default=None, comment="归属用户ID")

    # 消息内容
    role = Column(String(16), nullable=False, comment="角色: user/assistant/system")
    content = Column(Text, nullable=False, comment="消息内容（AES 加密）")

    # 时间字段
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")

    # 反向关联
    session = relationship("SessionModel", back_populates="messages")
    
    def __repr__(self):
        return f"<Message(session_id={self.session_id}, role={self.role})>"


class EmotionRecordModel(Base):
    """
    情绪记录表模型

    存储每次情绪分析的结果。

    字段说明：
        id: 自增主键
        session_id: 关联的会话ID（外键）
        user_id: 归属用户ID（v2 新增）
        primary_emotion: 主要情绪类型
        intensity: 情绪强度（1-10）
        triggers: 触发因素（JSON格式）
        context: 情境上下文（AES 加密存储）
        created_at: 创建时间
    """
    __tablename__ = "emotion_records"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 外键关联到sessions表
    session_id = Column(String(64), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True, comment="会话ID")
    user_id = Column(String(64), nullable=True, default=None, comment="归属用户ID")

    # 情绪数据
    primary_emotion = Column(String(32), nullable=False, comment="主要情绪")
    intensity = Column(Integer, nullable=False, comment="情绪强度1-10")
    triggers = Column(Text, nullable=True, comment="触发因素JSON")
    context = Column(Text, nullable=True, comment="情境上下文（AES加密）")

    # 时间字段
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    
    # 反向关联
    session = relationship("SessionModel", back_populates="emotion_records")
    
    def __repr__(self):
        return f"<EmotionRecord(session_id={self.session_id}, emotion={self.primary_emotion})>"


class DatabaseManager:
    """
    数据库管理器
    
    负责数据库连接和会话管理。
    使用单例模式，确保全局只有一个数据库连接池。
    
    使用方式：
        db = DatabaseManager()
        db.create_tables()  # 创建表
        session = db.get_session()  # 获取数据库会话
    """
    
    _instance = None  # 单例实例
    
    def __new__(cls):
        """单例模式：确保只有一个数据库管理器实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化数据库连接"""
        if self._initialized:
            return
            
        # 创建数据库引擎
        # echo=False 关闭 SQLAlchemy 自带的 SQL 日志（由应用层日志统一管理）
        self.engine = create_engine(
            settings.DATABASE_URL,
            echo=False,
            pool_pre_ping=True,  # 每次连接前检查连接是否有效
            pool_recycle=3600,   # 连接回收时间（秒）
        )
        
        # 创建会话工厂
        # autocommit=False: 需要手动提交
        # autoflush=False: 需要手动刷新
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        self._initialized = True
    
    def create_tables(self):
        """
        创建所有表
        
        如果表不存在则创建，存在则忽略。
        """
        Base.metadata.create_all(self.engine)
        logger.info("数据库表创建完成")
    
    def drop_tables(self):
        """
        删除所有表
        
        警告：这会删除所有数据！仅用于开发测试。
        """
        Base.metadata.drop_all(self.engine)
        logger.info("数据库表已删除")
    
    def get_session(self):
        """
        获取数据库会话
        
        使用上下文管理器确保会话正确关闭。
        
        Yields:
            Session: SQLAlchemy会话对象
        
        使用示例：
            with db.get_session() as session:
                session.add(new_record)
                session.commit()
        """
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()
    
    def get_session_direct(self):
        """
        直接获取数据库会话
        
        返回一个会话对象，需要手动关闭。
        
        Returns:
            Session: SQLAlchemy会话对象
        """
        return self.SessionLocal()


# 全局数据库管理器实例
db_manager = DatabaseManager()
