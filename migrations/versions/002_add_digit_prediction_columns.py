"""Add digit prediction columns to engine_predictions table.

Revision ID: 002
Revises: 001
Create Date: 2026-08-12
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("engine_predictions", sa.Column("predicted_digit", sa.Integer(), nullable=True))
    op.add_column("engine_predictions", sa.Column("digit_confidence", sa.Float(), nullable=True))
    op.add_column("engine_predictions", sa.Column("digit_top_3", sa.Text(), nullable=True))
    op.add_column("engine_predictions", sa.Column("digit_top_4", sa.Text(), nullable=True))
    op.add_column("engine_predictions", sa.Column("digit_probabilities", sa.Text(), nullable=True))
    op.add_column("engine_predictions", sa.Column("digit_method", sa.String(length=64), nullable=True))
    op.add_column("engine_predictions", sa.Column("digit_abstained", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("engine_predictions", "digit_abstained")
    op.drop_column("engine_predictions", "digit_method")
    op.drop_column("engine_predictions", "digit_probabilities")
    op.drop_column("engine_predictions", "digit_top_4")
    op.drop_column("engine_predictions", "digit_top_3")
    op.drop_column("engine_predictions", "digit_confidence")
    op.drop_column("engine_predictions", "predicted_digit")
