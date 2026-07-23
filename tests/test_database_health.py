"""
预上线数据库健康检查
执行：pytest tests/test_database_health.py -v
用途：上线前验证数据库连接、表结构、加密密钥等关键配置。
"""
import os
import sys
import base64
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from config.settings import settings
from schemas.database import db_manager


def test_database_url_configured():
    """验证 DATABASE_URL 不是默认占位值"""
    url = settings.DATABASE_URL
    assert url, "DATABASE_URL 为空"
    assert "changeme" not in url.lower(), "DATABASE_URL 仍为占位值"
    assert "mysql+pymysql" in url, "不是 MySQL 连接"
    assert "charset=utf8mb4" in url, "缺少 utf8mb4 charset"


def test_mysql_host_reachable():
    """验证 MySQL 端口可达（仅 TCP 握手）"""
    import socket
    host = settings.MYSQL_HOST
    port = int(settings.MYSQL_PORT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    try:
        result = sock.connect_ex((host, port))
        assert result == 0, f"MySQL {host}:{port} 不可达 (error={result})"
    finally:
        sock.close()


def _execute_raw(sql: str) -> list:
    """在数据库上执行原始 SQL 查询"""
    db = db_manager.get_session_direct()
    try:
        result = db.execute(text(sql))
        rows = result.fetchall()
        result.close()
        return rows
    finally:
        db.close()


@pytest.mark.skipif(
    not settings.USE_DATABASE,
    reason="USE_DATABASE=False，跳过数据库实际连接测试",
)
class TestDatabaseConnection:
    """需要实际连接 MySQL 的测试"""

    def test_can_connect_and_query(self):
        """验证数据库可连接并执行查询"""
        rows = _execute_raw("SELECT 1 AS health_check")
        assert rows is not None
        assert len(rows) == 1

    def test_all_v2_tables_exist(self):
        """验证 7 张 V2 表全部存在"""
        expected_tables = [
            "users",
            "credentials",
            "sessions",
            "messages",
            "emotion_records",
            "safety_flags",
            "scale_screenings",
        ]

        rows = _execute_raw("SHOW TABLES")
        actual_tables = {row[0] for row in rows}

        missing = [t for t in expected_tables if t not in actual_tables]
        assert not missing, f"缺少 V2 表: {missing}"

    def test_alembic_head_applied(self):
        """验证 alembic 迁移已执行"""
        try:
            rows = _execute_raw("SELECT version_num FROM alembic_version")
            assert rows, "alembic_version 表为空，迁移未执行"
        except Exception as e:
            pytest.fail(f"alembic_version 表不存在或无法读取: {e}")

    def test_database_charset(self):
        """验证数据库字符集为 utf8mb4"""
        rows = _execute_raw(
            "SELECT DEFAULT_CHARACTER_SET_NAME "
            "FROM information_schema.SCHEMATA "
            f"WHERE SCHEMA_NAME = '{settings.MYSQL_DATABASE}'"
        )
        assert rows, "无法查询数据库字符集"
        charset = rows[0][0]
        assert charset == "utf8mb4", f"数据库字符集为 {charset}，需要 utf8mb4"


@pytest.mark.skipif(
    not os.environ.get("ENCRYPTION_KEY"),
    reason="ENCRYPTION_KEY 未设置（本地开发可跳过，生产部署前必须验证）",
)
class TestEncryptionKey:
    """验证字段级加密密钥（ENCRYPTION_KEY 环境变量）

    所有测试在生产环境必须全部通过。开发环境跳过。
    """

    def test_encryption_key_configured(self):
        """验证 ENCRYPTION_KEY 已设置且不是默认值"""
        key = os.environ.get("ENCRYPTION_KEY", "")
        assert key, "ENCRYPTION_KEY 未设置"
        assert "change-me" not in key.lower(), "ENCRYPTION_KEY 仍为示例值"
        # AES-256-GCM 需要 32 字节密钥
        try:
            decoded = base64.b64decode(key)
            assert len(decoded) == 32, (
                f"ENCRYPTION_KEY 解码后 {len(decoded)} 字节，需要 32 字节"
            )
        except Exception:
            pytest.fail("ENCRYPTION_KEY 不是有效 base64 编码")

    def test_encrypt_decrypt_roundtrip(self):
        """验证加解密往返正常"""
        from modules.encryption import encrypt_field, decrypt_field

        original = "这是一条包含敏感信息的测试消息"
        encrypted = encrypt_field(original)
        assert encrypted != original
        decrypted = decrypt_field(encrypted)
        assert decrypted == original

    def test_encryption_key_stability(self):
        """验证同一密钥对同一明文的每次加密产生不同密文（GCM nonce 随机）"""
        from modules.encryption import encrypt_field, decrypt_field

        plaintext = "重复加密测试"
        c1 = encrypt_field(plaintext)
        c2 = encrypt_field(plaintext)
        assert c1 != c2, "两次加密应产生不同密文（GCM nonce 去重）"
        assert decrypt_field(c1) == plaintext
        assert decrypt_field(c2) == plaintext
