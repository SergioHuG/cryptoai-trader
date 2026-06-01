"""add reviewer_sessions and discretionary_signals

Revision ID: b3e9f1a72c08
Revises: 9f7cc2db4f7c
Create Date: 2026-05-31 00:00:00.000000
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


revision: str = 'b3e9f1a72c08'
down_revision: Union[str, None] = '9f7cc2db4f7c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'reviewer_sessions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('signal_id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('direction', sa.String(length=10), nullable=False),
        sa.Column('confidence', sa.Numeric(precision=6, scale=4), nullable=False),
        sa.Column('recommended_size', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('sent_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('decided_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('decision', sa.String(length=10), nullable=False),
        sa.Column('decision_latency_seconds', sa.Numeric(precision=8, scale=3), nullable=True),
        sa.Column('hour_of_day', sa.Integer(), nullable=False),
        sa.Column('is_override', sa.Boolean(), nullable=False),
        sa.Column('historical_override_rate', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('deviation_z_score', sa.Numeric(precision=6, scale=3), nullable=True),
        sa.Column('escalation_flag', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_reviewer_sessions_decided_at', 'reviewer_sessions', ['decided_at'], unique=False)
    op.create_index('ix_reviewer_sessions_signal_id', 'reviewer_sessions', ['signal_id'], unique=False)

    op.create_table(
        'discretionary_signals',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('signal_ref', sa.String(length=36), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('direction', sa.String(length=10), nullable=False),
        sa.Column('conviction', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('provenance', sa.String(length=10), nullable=False),
        sa.Column('submitted_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('meta_label_consumed', sa.Boolean(), nullable=False),
        sa.Column('consumed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('signal_ref', name='uq_discretionary_signal_ref'),
    )
    op.create_index('ix_discretionary_signals_consumed', 'discretionary_signals', ['meta_label_consumed'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_discretionary_signals_consumed', table_name='discretionary_signals')
    op.drop_table('discretionary_signals')
    op.drop_index('ix_reviewer_sessions_signal_id', table_name='reviewer_sessions')
    op.drop_index('ix_reviewer_sessions_decided_at', table_name='reviewer_sessions')
    op.drop_table('reviewer_sessions')