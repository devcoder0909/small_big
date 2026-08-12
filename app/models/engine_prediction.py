"""EnginePrediction model — immutable audit log of original predictions."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, BigInteger, String, Float, DateTime, Index
from app.models.base import Base


class EnginePrediction(Base):
    """
    Immutable Engine Prediction Audit Trail Table.

    Stores the ORIGINAL predicted value (SMALL or BIG) for every period ID
    at the exact moment it is generated BEFORE the draw occurs.
    Once stored, records in this table are permanently locked and NEVER updated or modified.
    """
    __tablename__ = "engine_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    issue_id = Column(String(32), nullable=False, unique=True, index=True)
    predicted_size = Column(String(10), nullable=False)
    confidence = Column(Float, nullable=False)
    confluence_level = Column(String(32), nullable=True)
    agreeing_indicators = Column(Integer, nullable=True)
    active_indicators = Column(Integer, nullable=True)
    created_at_ms = Column(BigInteger, nullable=True)
    expires_at_ms = Column(BigInteger, nullable=True)
    regime_at_prediction = Column(String(64), nullable=True)
    champion_at_prediction = Column(String(64), nullable=True)
    analysis_window_at_prediction = Column(Integer, nullable=True)
    predicted_digit = Column(Integer, nullable=True)
    digit_confidence = Column(Float, nullable=True)
    digit_top_3 = Column(String(64), nullable=True)
    digit_top_4 = Column(String(64), nullable=True)
    digit_probabilities = Column(String(512), nullable=True)
    digit_method = Column(String(64), nullable=True)
    digit_abstained = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_engine_predictions_issue_id", "issue_id", unique=True),
    )
