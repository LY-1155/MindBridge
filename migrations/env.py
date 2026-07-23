import os
import sys
from logging.config import fileConfig

# 确保项目根在 PYTHONPATH（migrations/env.py 在子目录中）
_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _here not in sys.path:
    sys.path.insert(0, _here)

from sqlalchemy import engine_from_config, pool
from alembic import context

from config.settings import settings

# this is the Alembic Config object
config = context.config

# 从项目配置读取数据库 URL，覆盖 alembic.ini 中的占位值
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------------
# target_metadata:
#   导入 schemas.database_v2.Base 后启用 autogenerate
#   当前阶段（Slice 1）为 None，Slice 2 引入 ORM 后替换
# ---------------------------------------------------------------------------
try:
    from schemas.database_v2 import Base as DatabaseV2Base
    target_metadata = DatabaseV2Base.metadata
except ImportError:
    target_metadata = None


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
