"""
Redis 会话热缓存层
=================

Gap #19：Redis 做会话热数据（目录、元数据、最近消息），MySQL 做持久化。

Key 设计：
  psy:session:{id}        HASH   — 会话元数据 (state: 整份 SessionMetadata JSON)
  psy:session:{id}:msgs   LIST   — 最近 N 条消息 (JSON, LPUSH / LRANGE)
  psy:user:{uid}:sessions SET    — 用户拥有的会话 ID 列表

每个 session key 设置 TTL，每次访问刷新 TTL。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import redis

from config.settings import settings

logger = logging.getLogger(__name__)

# ── Key 前缀 ──────────────────────────────────────────────────────
PREFIX_SESSION = "psy:session:"
PREFIX_MESSAGES = "psy:session:msgs:"
PREFIX_USER_SESSIONS = "psy:user:sessions:"
DEFAULT_TTL = getattr(settings, "REDIS_SESSION_TTL", 3600)  # 1 小时


def _get_redis() -> redis.Redis:
    """获取 Redis 客户端。"""
    return redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)


def _msg_key(session_id: str) -> str:
    return f"{PREFIX_MESSAGES}{session_id}"


def _session_key(session_id: str) -> str:
    return f"{PREFIX_SESSION}{session_id}"


def _user_key(user_id: str) -> str:
    return f"{PREFIX_USER_SESSIONS}{user_id}"


# ── 会话元数据 ────────────────────────────────────────────────────

def save_session_meta(session_id: str, meta: Dict[str, Any]) -> None:
    """将会话元数据写入 Redis HASH 的 state 字段并设置 TTL。

    meta 为整份 SessionMetadata 状态 dict（来自 SessionMetadata.to_state()）。
    单一 "state" 字段存整份蒸馏状态，与 MySQL state_json 共用同一序列化器，
    避免字段清单漂移（历史上曾因四份手写清单不一致导致蒸馏状态从不落库）。
    """
    try:
        r = _get_redis()
        key = _session_key(session_id)
        r.hset(key, mapping={
            "state": json.dumps(meta, ensure_ascii=False),
        })
        r.expire(key, DEFAULT_TTL)
    except redis.RedisError as e:
        logger.warning("Redis save_session_meta 失败: %s", e)


def load_session_meta(session_id: str) -> Optional[Dict[str, Any]]:
    """从 Redis HASH 的 state 字段加载整份会话状态。

    不存在或旧格式（无 state 字段）时返回 None，调用方回退 MySQL。
    """
    try:
        r = _get_redis()
        key = _session_key(session_id)
        raw = r.hget(key, "state")
        if not raw:
            return None
        state = json.loads(raw)
        if not isinstance(state, dict):
            return None
        return state
    except redis.RedisError as e:
        logger.warning("Redis load_session_meta 失败: %s", e)
        return None


def refresh_session_ttl(session_id: str) -> None:
    """刷新会话 TTL。每次访问时调用。"""
    try:
        r = _get_redis()
        r.expire(_session_key(session_id), DEFAULT_TTL)
        r.expire(_msg_key(session_id), DEFAULT_TTL)
    except redis.RedisError as e:
        logger.warning("Redis refresh_ttl 失败: %s", e)


def delete_session(session_id: str) -> None:
    """从 Redis 删除会话所有 key。"""
    try:
        r = _get_redis()
        r.delete(_session_key(session_id), _msg_key(session_id))
    except redis.RedisError as e:
        logger.warning("Redis delete_session 失败: %s", e)


# ── 消息缓存 ──────────────────────────────────────────────────────

def cache_message(session_id: str, role: str, content: str, cap: int = 40) -> None:
    """将一条消息 JSON 写入 Redis LIST（LPUSH），超出 cap 则 RTRIM。"""
    try:
        r = _get_redis()
        payload = json.dumps({"role": role, "content": content}, ensure_ascii=False)
        key = _msg_key(session_id)
        r.lpush(key, payload)
        r.ltrim(key, 0, cap - 1)
        r.expire(key, DEFAULT_TTL)
    except redis.RedisError as e:
        logger.warning("Redis cache_message 失败: %s", e)


def get_cached_messages(session_id: str, limit: int = 20) -> List[Dict[str, str]]:
    """从 Redis LIST 获取最近 limit 条消息（最早→最新）。"""
    try:
        r = _get_redis()
        key = _msg_key(session_id)
        raw = r.lrange(key, 0, limit - 1)
        if not raw:
            return []
        # LPUSH 把最新的放在最前面，所以需要反转
        messages = [json.loads(m) for m in reversed(raw)]
        return messages
    except redis.RedisError as e:
        logger.warning("Redis get_cached_messages 失败: %s", e)
        return []


def delete_cached_messages(session_id: str) -> None:
    """删除消息缓存 key。"""
    try:
        _get_redis().delete(_msg_key(session_id))
    except redis.RedisError as e:
        logger.warning("Redis delete_cached_messages 失败: %s", e)


# ── 用户-会话索引 ─────────────────────────────────────────────────

def add_user_session(user_id: str, session_id: str) -> None:
    """将 session_id 加入用户的会话索引 SET。"""
    if not user_id:
        return
    try:
        r = _get_redis()
        key = _user_key(user_id)
        r.sadd(key, session_id)
        r.expire(key, DEFAULT_TTL * 24)  # 用户索引保留更久
    except redis.RedisError as e:
        logger.warning("Redis add_user_session 失败: %s", e)


def get_user_sessions(user_id: str) -> List[str]:
    """获取用户拥有的会话 ID 列表。"""
    if not user_id:
        return []
    try:
        r = _get_redis()
        return list(r.smembers(_user_key(user_id)))
    except redis.RedisError as e:
        logger.warning("Redis get_user_sessions 失败: %s", e)
        return []


def remove_user_session(user_id: str, session_id: str) -> None:
    """从用户会话索引中移除。"""
    if not user_id:
        return
    try:
        r = _get_redis()
        r.srem(_user_key(user_id), session_id)
    except redis.RedisError as e:
        logger.warning("Redis remove_user_session 失败: %s", e)
