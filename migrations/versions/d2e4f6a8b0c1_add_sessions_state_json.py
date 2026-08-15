"""add_sessions_state_json

给 sessions 表加 state_json 列：整份 SessionMetadata 蒸馏状态
（phase / probed_dimensions / family_members / working_hypothesis /
scid_flags / scid_interview_state / safety_state 等）以加密 JSON 落库。
此前这些字段从未持久化（Redis/MySQL 只读写 key_topics/scale_state 等
少数字段），跨实例加载时全部归默认值——SCID 跨轮累积、危机状态机、
医生模式 phase 在 USE_DATABASE=true 的多请求场景下丢失。
此列是蒸馏状态的唯一权威来源，与 ORM SessionV2.state_json 对齐。

Revision ID: d2e4f6a8b0c1
Revises: b7c8d9e0a123
Create Date: 2026-08-15 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'd2e4f6a8b0c1'
down_revision: Union[str, Sequence[str], None] = 'b7c8d9e0a123'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('sessions', sa.Column(
        'state_json', sa.Text(), nullable=True,
        comment='会话蒸馏状态 JSON（AES 加密存储）'))


def downgrade() -> None:
    op.drop_column('sessions', 'state_json')
