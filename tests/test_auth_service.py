"""
TDD #2/#3: Auth Service 测试

验证密码哈希、credential 管理、注册/登录流程。
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(_ROOT))

import pytest
from alembic.config import Config
from alembic import command


@pytest.fixture(scope="module")
def _migrate_module():
    ini_path = os.path.join(_ROOT, "alembic.ini")
    if os.path.exists(ini_path):
        cfg = Config(ini_path)
        try:
            command.downgrade(cfg, "base")
        except Exception:
            pass
        command.upgrade(cfg, "head")
    yield
    if os.path.exists(ini_path):
        cfg = Config(ini_path)
        command.downgrade(cfg, "base")


class TestPasswordHash:
    """密码哈希/验证。"""

    def test_hash_produces_different_from_plaintext(self, _migrate_module):
        from modules.auth_service import AuthService
        pw = "mypassword123"
        hashed = AuthService.hash_password(pw)
        assert hashed != pw
        assert len(hashed) > 20

    def test_verify_correct_password(self, _migrate_module):
        from modules.auth_service import AuthService
        pw = "correct_horse_battery_staple"
        hashed = AuthService.hash_password(pw)
        assert AuthService.verify_password(pw, hashed) is True

    def test_verify_wrong_password(self, _migrate_module):
        from modules.auth_service import AuthService
        hashed = AuthService.hash_password("right_password")
        assert AuthService.verify_password("wrong_password", hashed) is False

    def test_same_password_different_hash(self, _migrate_module):
        from modules.auth_service import AuthService
        h1 = AuthService.hash_password("same")
        h2 = AuthService.hash_password("same")
        assert h1 != h2  # bcrypt salt 导致每次不同


class TestCredentialBinding:
    """credential 与 user 绑定。"""

    def test_bind_password_credential(self, _migrate_module):
        from modules.auth_service import AuthService
        from modules.user_service import UserService
        uid = UserService.create_user(display_name="test")
        result = AuthService.bind_credential(uid, "password", "testuser", "secret123")
        assert result is True

    def test_duplicate_identifier_rejected(self, _migrate_module):
        from modules.auth_service import AuthService
        from modules.user_service import UserService
        uid = UserService.create_user()
        AuthService.bind_credential(uid, "password", "dup_user", "pw1")
        with pytest.raises(ValueError, match="已存在"):
            AuthService.bind_credential(uid, "password", "dup_user", "pw2")

    def test_same_identifier_different_type_allowed(self, _migrate_module):
        from modules.auth_service import AuthService
        from modules.user_service import UserService
        uid = UserService.create_user()
        AuthService.bind_credential(uid, "password", "multi_user", "pw")
        # 同一标识符不同 type 可以共存（如后续绑定手机号）
        AuthService.bind_credential(uid, "wechat", "multi_user", None)


class TestRegisterAndLogin:
    """注册/登录全流程。"""

    def test_register_creates_user_and_credential(self, _migrate_module):
        from modules.auth_service import AuthService
        result = AuthService.register("newuser", "password123", display_name="新用户")
        assert "user_id" in result
        assert len(result["user_id"]) == 36

    def test_register_duplicate_username_fails(self, _migrate_module):
        from modules.auth_service import AuthService
        AuthService.register("dupname", "pw1")
        with pytest.raises(ValueError, match="已存在"):
            AuthService.register("dupname", "pw2")

    def test_login_with_correct_credentials(self, _migrate_module):
        from modules.auth_service import AuthService
        AuthService.register("loginuser", "correct_pw")
        result = AuthService.login("loginuser", "correct_pw")
        assert "user_id" in result
        assert result["user_id"] is not None

    def test_login_with_wrong_password(self, _migrate_module):
        from modules.auth_service import AuthService
        AuthService.register("loginuser2", "right_pw")
        with pytest.raises(ValueError, match="密码错误"):
            AuthService.login("loginuser2", "wrong_pw")

    def test_login_nonexistent_user(self, _migrate_module):
        from modules.auth_service import AuthService
        with pytest.raises(ValueError, match="用户不存在"):
            AuthService.login("no_such_user", "pw")

    def test_login_deleted_user_fails(self, _migrate_module):
        from modules.auth_service import AuthService
        from modules.user_service import UserService
        result = AuthService.register("deleted_user", "pw")
        UserService.soft_delete(result["user_id"])
        with pytest.raises(ValueError, match="已注销"):
            AuthService.login("deleted_user", "pw")


class TestChangePassword:
    """密码变更。"""

    def test_change_password_success(self, _migrate_module):
        from modules.auth_service import AuthService
        result = AuthService.register("pwchange", "old_pw")
        AuthService.change_password(result["user_id"], "old_pw", "new_pw")
        # 旧密码登录失败
        with pytest.raises(ValueError, match="密码错误"):
            AuthService.login("pwchange", "old_pw")
        # 新密码登录成功
        login_result = AuthService.login("pwchange", "new_pw")
        assert login_result["user_id"] == result["user_id"]

    def test_change_password_wrong_old(self, _migrate_module):
        from modules.auth_service import AuthService
        result = AuthService.register("pwchange2", "correct_old")
        with pytest.raises(ValueError, match="原密码错误"):
            AuthService.change_password(result["user_id"], "wrong_old", "new_pw")
