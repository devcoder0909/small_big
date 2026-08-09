"""Game result model — core historical data table."""

from datetime import datetime, timezone
from sqlalchemy import BigInteger, Boolean, Integer, Text, DateTime, Index, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base


class GameResult(Base):
    """
    Normalized game results table — the core historical dataset.

    NEVER delete or truncate this table.
    Each issue_id represents one unique game round from the source.
    """

    __tablename__ = "game_results"
    __table_args__ = (
        Index("idx_game_results_issue_id", "issue_id", unique=True),
        Index("idx_game_results_first_observed_at", "first_observed_at"),
        Index("idx_game_results_source_created_at", "source_created_at"),
        Index("idx_game_results_size", "calculated_size"),
        Index("idx_game_results_number", "result_number"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Unique game identifier from source — PRIMARY LOGICAL KEY
    issue_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)

    # Result value (0-9)
    result_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Color from source (can be "red", "green", "red,violet", "green,violet")
    source_color: Mapped[str] = mapped_column(Text, nullable=False)

    # Premium value from source
    premium: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Sum value from source
    sum_value: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Size classification
    calculated_size: Mapped[str] = mapped_column(Text, nullable=False)  # "BIG" or "SMALL"

    # Timestamps
    source_created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    first_observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    last_observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # Source tracking
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    raw_response_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("raw_responses.id"), nullable=True
    )

    # Data integrity
    data_hash: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
