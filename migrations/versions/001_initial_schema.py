"""Initial schema — all tables for WinGo 30S platform.

Revision ID: 001
Revises: None
Create Date: 2026-08-09
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # === source_requests ===
    op.create_table(
        "source_requests",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("request_timestamp_ms", sa.BigInteger(), nullable=False),
        sa.Column("http_status", sa.Integer(), nullable=True),
        sa.Column("response_time_ms", sa.Integer(), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("error_type", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("records_received", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_index("idx_source_requests_requested_at", "source_requests", ["requested_at"])

    # === raw_responses ===
    op.create_table(
        "raw_responses",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("source_request_id", sa.BigInteger(),
                  sa.ForeignKey("source_requests.id"), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", JSONB(), nullable=False),
        sa.Column("payload_hash", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_index("idx_raw_responses_received_at", "raw_responses", ["received_at"])

    # === game_results ===
    op.create_table(
        "game_results",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("issue_id", sa.Text(), nullable=False, unique=True),
        sa.Column("result_number", sa.Integer(), nullable=False),
        sa.Column("source_color", sa.Text(), nullable=False),
        sa.Column("premium", sa.Text(), nullable=True),
        sa.Column("sum_value", sa.Integer(), nullable=True),
        sa.Column("calculated_size", sa.Text(), nullable=False),
        sa.Column("source_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("first_observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("raw_response_id", sa.BigInteger(),
                  sa.ForeignKey("raw_responses.id"), nullable=True),
        sa.Column("data_hash", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_index("idx_game_results_issue_id", "game_results", ["issue_id"], unique=True)
    op.create_index("idx_game_results_first_observed_at", "game_results", ["first_observed_at"])
    op.create_index("idx_game_results_source_created_at", "game_results", ["source_created_at"])
    op.create_index("idx_game_results_size", "game_results", ["calculated_size"])
    op.create_index("idx_game_results_number", "game_results", ["result_number"])

    # === data_quality ===
    op.create_table(
        "data_quality",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("severity", sa.Text(), nullable=False),
        sa.Column("issue_id", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("details", sa.Text(), nullable=True),
        sa.Column("resolved", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_index("idx_data_quality_created_at", "data_quality", ["created_at"])
    op.create_index("idx_data_quality_event_type", "data_quality", ["event_type"])

    # === system_heartbeat ===
    op.create_table(
        "system_heartbeat",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("service_name", sa.Text(), nullable=False, unique=True),
        sa.Column("last_heartbeat", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_successful_fetch", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_new_record", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="STARTING"),
        sa.Column("total_records", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_errors", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_duplicates", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("uptime_seconds", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_index("idx_heartbeat_service_name", "system_heartbeat", ["service_name"], unique=True)

    # === analytics_snapshots ===
    op.create_table(
        "analytics_snapshots",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("window_size", sa.Integer(), nullable=False),
        sa.Column("total_records", sa.Integer(), nullable=False),
        sa.Column("small_count", sa.Integer(), nullable=False),
        sa.Column("big_count", sa.Integer(), nullable=False),
        sa.Column("small_percentage", sa.Float(), nullable=False),
        sa.Column("big_percentage", sa.Float(), nullable=False),
        sa.Column("current_streak_size", sa.Text(), nullable=True),
        sa.Column("current_streak_length", sa.Integer(), nullable=True),
        sa.Column("longest_small_streak", sa.Integer(), nullable=True),
        sa.Column("longest_big_streak", sa.Integer(), nullable=True),
        sa.Column("transition_statistics", JSONB(), nullable=True),
        sa.Column("anomaly_statistics", JSONB(), nullable=True),
        sa.Column("prediction_data", JSONB(), nullable=True),
        sa.Column("analytics_version", sa.Text(), nullable=False, server_default="1.0.0"),
    )


def downgrade() -> None:
    op.drop_table("analytics_snapshots")
    op.drop_table("system_heartbeat")
    op.drop_table("data_quality")
    op.drop_table("game_results")
    op.drop_table("raw_responses")
    op.drop_table("source_requests")
