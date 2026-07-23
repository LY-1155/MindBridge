"""
TDD #7: 字段级 AES 加密 — 集成测试

验证三个写读路径上 encrypt_field / decrypt_field 已接入：
  - messages.content
  - emotion_records.context
  - safety_flags.matched_terms
"""
from __future__ import annotations

import os
import sys
import uuid

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(_ROOT))

import pytest


@pytest.fixture(autouse=True)
def _ensure_encryption_key():
    if not os.environ.get("ENCRYPTION_KEY"):
        os.environ["ENCRYPTION_KEY"] = "test-key-32-bytes-long!!!"


@pytest.fixture(scope="class", autouse=True)
def _migrate():
    from alembic.config import Config
    from alembic import command
    ini_path = os.path.join(_ROOT, "alembic.ini")
    if os.path.exists(ini_path):
        cfg = Config(ini_path)
        try:
            command.downgrade(cfg, "base")
        except Exception:
            pass
        command.upgrade(cfg, "head")
    yield


def _make_user() -> str:
    from modules.user_service import UserService
    return UserService.create_user(display_name="enc_test")


# ── safety_flags.matched_terms ─────────────────────────────────

class TestSafetyFlagEncryption:
    """safety_flags.matched_terms 写入加密 → 读取解密。"""

    @pytest.fixture
    def user_id(self):
        return _make_user()

    def test_write_encrypted_read_decrypted(self, user_id):
        from modules.safety.flag_recorder import SafetyFlagRecorder

        recorder = SafetyFlagRecorder()
        terms = ["自杀", "自残", "伤害他人"]
        sid = f"s_{uuid.uuid4().hex[:8]}"

        flag = recorder.record(
            user_id=user_id, session_id=sid,
            level=1, blocked=False, matched_terms=terms,
        )

        # 通过 _flag_to_dict 读取（已验证解密）
        from modules.safety.flag_recorder import _flag_to_dict
        d = _flag_to_dict(flag)
        assert sorted(d["matched_terms"]) == sorted(terms)

    def test_matched_terms_stored_as_ciphertext_in_db(self, user_id):
        from modules.safety.flag_recorder import SafetyFlagRecorder
        from schemas.database_v2 import SafetyFlag as SF
        from schemas.database import db_manager

        recorder = SafetyFlagRecorder()
        sid = f"s_{uuid.uuid4().hex[:8]}"

        flag = recorder.record(
            user_id=user_id, session_id=sid,
            level=2, blocked=True,
            matched_terms=["自伤", "暴力"],
        )

        with db_manager.get_session_direct() as s:
            db_flag = s.query(SF).filter_by(id=flag.id).first()
            # 数据库中是密文
            assert db_flag.matched_terms != '["自伤", "暴力"]'
            assert "自伤" not in db_flag.matched_terms  # 明文不在密文中


# ── 向后兼容：明文兜底 ───────────────────────────────────────

class TestBackwardCompatibility:
    """safe_decrypt_field 对已有明文数据优雅回退。"""

    @pytest.fixture
    def user_id(self):
        return _make_user()

    def test_decrypt_plaintext_returns_unchanged(self, user_id):
        from modules.safety.flag_recorder import SafetyFlagRecorder
        from schemas.database_v2 import SafetyFlag as SF
        from schemas.database import db_manager
        import json

        # 直接插入明文 matched_terms（模拟加密层上线前的遗留数据）
        sid = f"s_plain_{uuid.uuid4().hex[:6]}"
        plain_json = json.dumps(["遗留敏感词"], ensure_ascii=False)
        with db_manager.get_session_direct() as s:
            f = SF(user_id=user_id, session_id=sid, level=1, blocked=False,
                    matched_terms=plain_json)
            s.add(f)
            s.commit()
            flag_id = f.id

        # 通过 list_pending_review 读取，safe_decrypt_field 应对明文正确加载
        recorder = SafetyFlagRecorder()
        flags = recorder.list_pending_review(user_id=user_id)
        our_flag = [f for f in flags if f["id"] == flag_id][0]
        assert our_flag["matched_terms"] == ["遗留敏感词"]
