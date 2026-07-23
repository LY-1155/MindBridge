"""add_consent_fields

Revision ID: a1b2c3d4e5f6
Revises: e85bf13dd867
Create Date: 2026-07-04 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'e85bf13dd867'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('users', sa.Column(
        'consent_at', sa.DateTime(), nullable=True, comment='知情同意签署时间'))
    op.add_column('users', sa.Column(
        'consent_version', sa.String(16), nullable=True, comment='同意的协议版本号'))


def downgrade() -> None:
    op.drop_column('users', 'consent_version')
    op.drop_column('users', 'consent_at')
