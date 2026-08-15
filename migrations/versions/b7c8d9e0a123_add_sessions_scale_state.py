"""add_sessions_scale_state

补上 sessions.scale_state 列：ORM SessionV2 自始至终声明了该列，
但 v2_initial_schema 迁移漏建，导致任何经 ORM 全列查询 sessions 的操作报
"Unknown column 'sessions.scale_state'"。此迁移补齐，与 ORM 对齐。

Revision ID: b7c8d9e0a123
Revises: a1b2c3d4e5f6
Create Date: 2026-08-10 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b7c8d9e0a123'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('sessions', sa.Column(
        'scale_state', sa.Text(), nullable=True, comment='量表进行中状态 JSON'))


def downgrade() -> None:
    op.drop_column('sessions', 'scale_state')
