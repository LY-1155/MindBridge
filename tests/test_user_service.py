"""
TDD #2: User Service 层测试

验证 User 的创建、查询、状态管理和软删除。
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
    """模块级别：跑一次迁移建表，所有测试共用。"""
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


class TestCreateUser:
    """创建用户的基本行为。"""

    def test_create_user_returns_user_id(self, _migrate_module):
        from modules.user_service import UserService
        user_id = UserService.create_user(display_name="测试用户")
        assert user_id is not None
        assert len(user_id) == 36  # UUID

    def test_create_user_stores_in_db(self, _migrate_module):
        from modules.user_service import UserService
        user_id = UserService.create_user(display_name="张三")
        user = UserService.get_user(user_id)
        assert user is not None
        assert user["display_name"] == "张三"
        assert user["status"] == "active"

    def test_create_user_without_display_name(self, _migrate_module):
        from modules.user_service import UserService
        user_id = UserService.create_user()
        user = UserService.get_user(user_id)
        assert user["display_name"] is None
        assert user["status"] == "active"


class TestGetUser:
    """查询用户。"""

    def test_get_existing_user(self, _migrate_module):
        from modules.user_service import UserService
        uid = UserService.create_user(display_name="李四")
        user = UserService.get_user(uid)
        assert user["user_id"] == uid

    def test_get_nonexistent_user_returns_none(self, _migrate_module):
        from modules.user_service import UserService
        assert UserService.get_user("nonexistent-id") is None


class TestUpdateUserStatus:
    """用户状态管理。"""

    def test_disable_user(self, _migrate_module):
        from modules.user_service import UserService
        uid = UserService.create_user()
        result = UserService.set_status(uid, "disabled")
        assert result is True
        assert UserService.get_user(uid)["status"] == "disabled"

    def test_reactivate_user(self, _migrate_module):
        from modules.user_service import UserService
        uid = UserService.create_user()
        UserService.set_status(uid, "disabled")
        UserService.set_status(uid, "active")
        assert UserService.get_user(uid)["status"] == "active"

    def test_invalid_status_raises(self, _migrate_module):
        from modules.user_service import UserService
        uid = UserService.create_user()
        with pytest.raises(ValueError, match="无效状态"):
            UserService.set_status(uid, "invalid_status")

    def test_set_status_nonexistent_user(self, _migrate_module):
        from modules.user_service import UserService
        assert UserService.set_status("nonexistent", "disabled") is False


class TestSoftDelete:
    """软删除流程。"""

    def test_soft_delete_sets_status_and_deleted_at(self, _migrate_module):
        from modules.user_service import UserService
        uid = UserService.create_user()
        UserService.soft_delete(uid)
        user = UserService.get_user(uid)
        assert user["status"] == "deleted"
        assert user["deleted_at"] is not None

    def test_soft_deleted_user_can_be_restored(self, _migrate_module):
        from modules.user_service import UserService
        uid = UserService.create_user()
        UserService.soft_delete(uid)
        UserService.restore(uid)
        user = UserService.get_user(uid)
        assert user["status"] == "active"
        assert user["deleted_at"] is None

    def test_soft_delete_nonexistent_user(self, _migrate_module):
        from modules.user_service import UserService
        assert UserService.soft_delete("nonexistent") is False


class TestListUsers:
    """列出用户。"""

    def test_list_users_returns_all(self, _migrate_module):
        from modules.user_service import UserService
        uid1 = UserService.create_user(display_name="A")
        uid2 = UserService.create_user(display_name="B")
        users = UserService.list_users()
        uids = {u["user_id"] for u in users}
        assert uid1 in uids
        assert uid2 in uids

    def test_list_users_excludes_deleted(self, _migrate_module):
        from modules.user_service import UserService
        uid1 = UserService.create_user(display_name="active")
        uid2 = UserService.create_user(display_name="deleted")
        UserService.soft_delete(uid2)
        users = UserService.list_users(include_deleted=False)
        uids = {u["user_id"] for u in users}
        assert uid1 in uids
        assert uid2 not in uids
