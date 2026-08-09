"""Analytics snapshot model for periodic analytics captures."""

from datetime import datetime, timezone
from sqlalchemy import BigInteger, Integer, Float, Text, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base


class AnalyticsSnapshot(Base):
    """Periodic snapshots of analytics calculations for historical tracking."""

    __tablename__ = "analytics_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    window_size: Mapped[int] = mapped_column(Integer, nullable=False)
    total_records: Mapped[int] = mapped_column(Integer, nullable=False)
    small_count: Mapped[int] = mapped_column(Integer, nullable=False)
    big_count: Mapped[int] = mapped_column(Integer, nullable=False)
    small_percentage: Mapped[float] = mapped_column(Float, nullable=False)
    big_percentage: Mapped[float] = mapped_column(Float, nullable=False)
    current_streak_size: Mapped[str | None] = mapped_column(Text, nullable=True)
    current_streak_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    longest_small_streak: Mapped[int | None] = mapped_column(Integer, nullable=True)
    longest_big_streak: Mapped[int | None] = mapped_column(Integer, nullable=True)
    transition_statistics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    anomaly_statistics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    prediction_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    analytics_version: Mapped[str] = mapped_column(
        Text, nullable=False, default="1.0.0"
    )
