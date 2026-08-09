"""Raw response storage model."""

from datetime import datetime, timezone
from sqlalchemy import BigInteger, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base


class RawResponse(Base):
    """Stores raw JSON payloads for audit, debugging, and parser recovery."""

    __tablename__ = "raw_responses"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source_request_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("source_requests.id"), nullable=False
    )
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    payload_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
