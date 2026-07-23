"""
TDD Slice 1: Alembic 迁移骨架测试

验证 alembic upgrade head → downgrade base 全链路可用。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pytest
from alembic.config import Config
from alembic import command
from sqlalchemy import text

from config.settings import settings


@pytest.fixture(scope="module")
def alembic_cfg():
    """创建指向项目 alembic.ini 的配置对象（通过 env.py 使用 settings.DATABASE_URL）。"""
    ini_path = _ROOT / "alembic.ini"
    if not ini_path.exists():
        pytest.skip("alembic.ini 尚未初始化（Slice 1 实现阶段创建）")
    cfg = Config(str(ini_path))
    return cfg


@pytest.mark.integration
class TestAlembicMigrationSkeleton:
    """验证 alembic upgrade/downgrade 基础流程。"""

    def test_upgrade_head_success(self, alembic_cfg):
        """alembic upgrade head 应成功执行到最新版本。"""
        command.upgrade(alembic_cfg, "head")

    def test_downgrade_base_success(self, alembic_cfg):
        """alembic downgrade base 应成功回退到空库。"""
        # 先确保在最新版
        command.upgrade(alembic_cfg, "head")
        # 再回退
        command.downgrade(alembic_cfg, "base")

    def test_upgrade_creates_alembic_version_table(self, alembic_cfg):
        """upgrade head 后应存在 alembic_version 表，表明迁移已记录。"""
        command.upgrade(alembic_cfg, "head")

        from schemas.database import db_manager
        with db_manager.get_session_direct() as session:
            result = session.execute(
                text("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=:db AND table_name='alembic_version'"),
                {"db": settings.MYSQL_DATABASE}
            ).scalar()
            assert result == 1


@pytest.mark.integration
class TestNewTablesExist:
    """验证新 schema 下所有表的形态（以下测试在 Slice 2~4 实现后通过）。"""

    @pytest.fixture(autouse=True)
    def _migrate_up(self, alembic_cfg):
        # 先回退到干净状态（上一个 test class 可能遗留数据）
        try:
            command.downgrade(alembic_cfg, "base")
        except Exception:
            pass  # base 已经是干净的
        command.upgrade(alembic_cfg, "head")
        yield
        # teardown: 回退，保持干净
        command.downgrade(alembic_cfg, "base")

    def _get_tables(self):
        from schemas.database import db_manager
        with db_manager.get_session_direct() as session:
            rows = session.execute(
                text("SELECT TABLE_NAME FROM information_schema.tables WHERE table_schema=:db"),
                {"db": settings.MYSQL_DATABASE}
            ).fetchall()
            return {r[0] for r in rows}

    # ---- Slice 2: users + credentials ----

    def test_users_table_exists(self, alembic_cfg):
        tables = self._get_tables()
        assert "users" in tables, f"users 表不存在，当前表: {tables}"

    def test_credentials_table_exists(self, alembic_cfg):
        tables = self._get_tables()
        assert "credentials" in tables, f"credentials 表不存在"

    def test_credentials_user_id_fk(self, alembic_cfg):
        """credentials.user_id 应为外键指向 users.user_id。"""
        from schemas.database import db_manager
        with db_manager.get_session_direct() as session:
            fks = session.execute(text("""
                SELECT COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                FROM information_schema.KEY_COLUMN_USAGE
                WHERE TABLE_SCHEMA=:db AND TABLE_NAME='credentials'
                  AND REFERENCED_TABLE_NAME IS NOT NULL
            """), {"db": settings.MYSQL_DATABASE}).fetchall()
            fk_dict = {r[0]: (r[1], r[2]) for r in fks}
            assert "user_id" in fk_dict, f"credentials 缺少 user_id 外键，现有外键: {list(fk_dict.keys())}"
            assert fk_dict["user_id"] == ("users", "user_id"), f"外键指向错误: {fk_dict['user_id']}"

    # ---- Slice 3: sessions + messages ----

    def test_sessions_requires_user_id(self, alembic_cfg):
        """sessions.user_id 应为 NOT NULL（与旧 schema 的 nullable=True 不同）。"""
        from schemas.database import db_manager
        with db_manager.get_session_direct() as session:
            nullable = session.execute(text("""
                SELECT IS_NULLABLE FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA=:db AND TABLE_NAME='sessions' AND COLUMN_NAME='user_id'
            """), {"db": settings.MYSQL_DATABASE}).scalar()
            assert nullable == "NO", f"user_id 应为 NOT NULL，实际: {nullable}"

    def test_messages_content_is_text_type(self, alembic_cfg):
        """messages.content 应为 TEXT 类型（支持加密后的长密文）。"""
        from schemas.database import db_manager
        with db_manager.get_session_direct() as session:
            dtype = session.execute(text("""
                SELECT DATA_TYPE FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA=:db AND TABLE_NAME='messages' AND COLUMN_NAME='content'
            """), {"db": settings.MYSQL_DATABASE}).scalar()
            assert dtype == "text"

    # ---- Slice 4: emotion_records, safety_flags, scale_screenings ----

    def test_emotion_records_has_risk_column(self, alembic_cfg):
        """emotion_records 应有 FLOAT risk 列（旧 schema 缺失）。"""
        from schemas.database import db_manager
        with db_manager.get_session_direct() as session:
            cols = session.execute(text("""
                SELECT COLUMN_NAME, DATA_TYPE FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA=:db AND TABLE_NAME='emotion_records'
            """), {"db": settings.MYSQL_DATABASE}).fetchall()
            col_map = {r[0]: r[1] for r in cols}
            assert "risk" in col_map, f"缺少 risk 列，现有列: {list(col_map.keys())}"
            assert col_map["risk"] == "float"

    def test_emotion_records_intensity_is_float(self, alembic_cfg):
        """emotion_records.intensity 应为 FLOAT（旧 schema 是 Integer）。"""
        from schemas.database import db_manager
        with db_manager.get_session_direct() as session:
            dtype = session.execute(text("""
                SELECT DATA_TYPE FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA=:db AND TABLE_NAME='emotion_records' AND COLUMN_NAME='intensity'
            """), {"db": settings.MYSQL_DATABASE}).scalar()
            assert dtype == "float", f"intensity 应为 float，实际: {dtype}"

    def test_safety_flags_table_exists(self, alembic_cfg):
        tables = self._get_tables()
        assert "safety_flags" in tables, f"safety_flags 表不存在，当前表: {tables}"

    def test_safety_flags_has_reviewed_column(self, alembic_cfg):
        """safety_flags 应有 reviewed 布尔列，默认 false。"""
        from schemas.database import db_manager
        with db_manager.get_session_direct() as session:
            cols = session.execute(text("""
                SELECT COLUMN_NAME, COLUMN_DEFAULT FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA=:db AND TABLE_NAME='safety_flags' AND COLUMN_NAME='reviewed'
            """), {"db": settings.MYSQL_DATABASE}).fetchall()
            assert len(cols) == 1

    def test_scale_screenings_table_exists(self, alembic_cfg):
        tables = self._get_tables()
        assert "scale_screenings" in tables, f"scale_screenings 表不存在"
