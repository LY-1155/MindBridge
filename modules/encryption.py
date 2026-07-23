"""
字段级 AES 加密模块
==================

对数据库中敏感字段做 AES-256-GCM 加密，密钥来自环境变量 ENCRYPTION_KEY。
每次加密使用随机 IV + GCM 认证标签，同一原文每次产出不同密文。
密文格式：base64(iv + ciphertext + tag)

加密范围（见 CONTEXT.md）：
  - messages.content
  - emotion_records.context
  - safety_flags.matched_terms

元数据字段不加密（session_id、user_id、emotion 标签名、risk 值、时间戳）。
"""

from __future__ import annotations

import base64
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def _get_key() -> bytes:
    """从 settings 对象获取 32 字节 AES-256 密钥。"""
    from config.settings import settings

    raw = settings.ENCRYPTION_KEY
    if not raw or raw == "change-me-to-a-32-byte-random-key":
        raise RuntimeError("ENCRYPTION_KEY 环境变量未设置，无法加解密敏感字段")
    key = raw.encode("utf-8")
    if len(key) < 32:
        key = key.ljust(32, b"\x00")
    return key[:32]


def encrypt_field(plaintext: str | None) -> str | None:
    """加密明文字段，返回 base64 密文。"""
    if plaintext is None:
        return None
    if plaintext == "":
        return ""
    aesgcm = AESGCM(_get_key())
    nonce = os.urandom(12)  # GCM 推荐 12 字节 nonce
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    # 将 nonce + ciphertext 打包后 base64 编码
    return base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii")


def safe_decrypt_field(value: str | None) -> str | None:
    """尝试解密；若失败则返回原值（兼容 DB 中已有明文数据）。

    在加密层上线后，遗留的明文数据不能直接通过 decrypt_field 解密，
    此函数兜底处理：能解密则解密，不能解密则原样返回。
    """
    if value is None:
        return None
    if value == "":
        return ""
    try:
        return decrypt_field(value)
    except ValueError:
        return value  # 已明文，直接返回


def decrypt_field(ciphertext: str | None) -> str | None:
    """解密 base64 密文，恢复原文。"""
    if ciphertext is None:
        return None
    if ciphertext == "":
        return ""
    try:
        raw = base64.urlsafe_b64decode(ciphertext)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError(f"解密失败：无效的密文格式 ({exc})") from exc
    if len(raw) < 13:  # 至少 12 字节 nonce + 1 字节 ciphertext
        raise ValueError("解密失败：密文长度不足")
    nonce = raw[:12]
    encrypted = raw[12:]
    try:
        aesgcm = AESGCM(_get_key())
        plain = aesgcm.decrypt(nonce, encrypted, None)
        return plain.decode("utf-8")
    except Exception as exc:
        raise ValueError(f"解密失败：密钥不匹配或密文损坏 ({exc})") from exc
