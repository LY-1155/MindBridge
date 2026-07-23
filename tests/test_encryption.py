"""
TDD Slice 5: AES 字段级加密层测试

验证 encrypt/decrypt 在写入-读取路径中透明加解密。
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(_ROOT))

import pytest


@pytest.fixture(autouse=True)
def _ensure_key():
    """确保测试环境有加密密钥。"""
    if not os.environ.get("ENCRYPTION_KEY"):
        os.environ["ENCRYPTION_KEY"] = "test-key-32-bytes-long!!!"  # 恰好 32 字节用于 AES-256
    # 重新导入以获取新的环境变量
    import importlib
    import modules.encryption as enc_mod
    importlib.reload(enc_mod)


@pytest.fixture
def encrypt():
    from modules.encryption import encrypt_field
    return encrypt_field


@pytest.fixture
def decrypt():
    from modules.encryption import decrypt_field
    return decrypt_field


class TestEncryptDecryptRoundtrip:
    """核心行为：加密后解密应还原原文。"""

    def test_roundtrip_simple(self, encrypt, decrypt):
        original = "我在表达我的焦虑"
        ciphertext = encrypt(original)
        assert ciphertext != original
        assert decrypt(ciphertext) == original

    def test_roundtrip_long_text(self, encrypt, decrypt):
        original = "长文本" * 500
        assert decrypt(encrypt(original)) == original

    def test_roundtrip_chinese_emoji(self, encrypt, decrypt):
        original = "用户输入 😢 包含 emoji 和中文，感觉很难过"
        assert decrypt(encrypt(original)) == original

    def test_roundtrip_empty_string(self, encrypt, decrypt):
        assert decrypt(encrypt("")) == ""

    def test_roundtrip_special_characters(self, encrypt, decrypt):
        original = "SELECT * FROM users; DROP TABLE users; -- <script>alert('xss')</script>"
        assert decrypt(encrypt(original)) == original


class TestEncryptDeterminism:
    """每次加密同一原文应产出不同的密文（随机 IV）。"""

    def test_same_input_different_ciphertext(self, encrypt):
        original = "hello world"
        c1 = encrypt(original)
        c2 = encrypt(original)
        assert c1 != c2


class TestDecryptInvalidInput:
    """解密无效输入应优雅失败。"""

    def test_decrypt_corrupted_ciphertext(self, decrypt):
        with pytest.raises(ValueError, match="解密失败"):
            decrypt("not-valid-base64!!!")

    def test_decrypt_wrong_key_ciphertext(self, encrypt, decrypt):
        """用不同密钥加密的密文应无法解密。"""
        original = "sensitive data"
        ciphertext = encrypt(original)
        # 临时改密钥
        os.environ["ENCRYPTION_KEY"] = "another-key-32-bytes-long!!!"
        import importlib
        import modules.encryption as enc_mod
        importlib.reload(enc_mod)
        from modules.encryption import decrypt_field as decrypt2
        with pytest.raises(ValueError, match="解密失败"):
            decrypt2(ciphertext)


class TestEncryptNoneHandling:
    """None 值应安全处理（用于可选字段）。"""

    def test_encrypt_none_returns_none(self, encrypt):
        assert encrypt(None) is None

    def test_decrypt_none_returns_none(self, decrypt):
        assert decrypt(None) is None


class TestEncryptDatabaseRoundtrip:
    """加密后的值存入数据库再取出应正常解密（集成测试）。"""

    @pytest.fixture(autouse=True)
    def _migrate(self):
        from alembic.config import Config
        from alembic import command
        from pathlib import Path
        ini = Path(__file__).resolve().parent.parent / "alembic.ini"
        cfg = Config(str(ini))
        try:
            command.upgrade(cfg, "head")
        except Exception:
            pass  # 表可能已存在
        yield
        command.downgrade(cfg, "base")

    @pytest.mark.integration
    def test_save_encrypted_message_read_back(self, encrypt, decrypt):
        from schemas.database_v2 import SessionV2, MessageV2, User
        from schemas.database import db_manager

        user_id = "test-enc-user-001"
        session_id = "test-enc-session-001"

        with db_manager.get_session_direct() as s:
            # 创建用户和会话
            user = User(user_id=user_id, status="active")
            sess = SessionV2(session_id=session_id, user_id=user_id)
            s.add(user)
            s.add(sess)
            s.commit()

            # 写入加密消息
            original_content = "用户说：我感到很焦虑，想咨询一些应对方法"
            encrypted = encrypt(original_content)
            msg = MessageV2(session_id=session_id, role="user", content=encrypted)
            s.add(msg)
            s.commit()

            # 读取并解密
            read_msg = s.query(MessageV2).filter_by(session_id=session_id).first()
            assert read_msg.content == encrypted
            assert read_msg.content != original_content  # 数据库里存的是密文
            assert decrypt(read_msg.content) == original_content  # 解密后恢复原文

            # 清理
            s.delete(msg)
            s.delete(sess)
            s.delete(user)
            s.commit()
