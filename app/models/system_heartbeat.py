"""System heartbeat model for collector liveness monitoring."""

from datetime import datetime, timezone
from sqlalchemy import BigInteger, Text, DateTime, Integer, Index
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base, BIGINT_PK


class SystemHeartbeat(Base):
    """Tracks collector and service heartbeats for monitoring."""

    __tablename__ = "system_heartbeat"
    __table_args__ = (
        Index("idx_heartbeat_service_name", "service_name", unique=True),
    )

    id: Mapped[int] = mapped_column(BIGINT_PK, primary_key=True, autoincrement=True)
    service_name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    last_heartbeat: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    last_successful_fetch: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_new_record: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, default="STARTING")
    # Statuses: STARTING, HEALTHY, DEGRADED, ERROR
    total_records: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_errors: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_duplicates: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    uptime_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
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
