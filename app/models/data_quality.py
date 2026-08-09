"""Data quality event tracking model."""

from datetime import datetime, timezone
from sqlalchemy import BigInteger, Text, DateTime, Boolean, Index
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base, BIGINT_PK


class DataQualityEvent(Base):
    """Tracks data quality anomalies, mismatches, and issues."""

    __tablename__ = "data_quality"
    __table_args__ = (
        Index("idx_data_quality_created_at", "created_at"),
        Index("idx_data_quality_event_type", "event_type"),
    )

    id: Mapped[int] = mapped_column(BIGINT_PK, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    # Types: missing_issue, duplicate_issue, invalid_result,
    #        timestamp_anomaly, api_failure, parse_failure,
    #        size_mismatch, gap_detected
    severity: Mapped[str] = mapped_column(Text, nullable=False)
    # Severities: INFO, WARNING, ERROR, CRITICAL
    issue_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
