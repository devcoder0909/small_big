"""Measure database row sizes for all models."""
from sqlalchemy import create_engine, inspect
from app.models.base import Base
from app.models.game_result import GameResult
from app.models.engine_prediction import EnginePrediction
from app.models.raw_response import RawResponse
from app.models.data_quality import DataQualityEvent
from app.models.source_request import SourceRequest
from app.models.system_heartbeat import SystemHeartbeat

engine = create_engine("sqlite:///:memory:")
Base.metadata.create_all(engine)

insp = inspect(engine)
for tname in insp.get_table_names():
    cols = insp.get_columns(tname)
    indexes = insp.get_indexes(tname)
    print(f"TABLE: {tname}")
    total_est = 0
    for c in cols:
        ctype = str(c["type"])
        name = c["name"]
        if "TEXT" in ctype or "VARCHAR" in ctype:
            est = 50
        elif "BIGINT" in ctype or "INTEGER" in ctype:
            est = 8
        elif "FLOAT" in ctype or "REAL" in ctype:
            est = 8
        elif "DATETIME" in ctype or "TIMESTAMP" in ctype:
            est = 8
        elif "BOOLEAN" in ctype:
            est = 1
        elif "JSON" in ctype:
            est = 2000
        else:
            est = 20
        total_est += est
        print(f"  {name:35s} {ctype:20s} ~{est} bytes")
    print(f"  TOTAL ESTIMATED ROW SIZE: ~{total_est} bytes")
    print(f"  INDEXES: {len(indexes)}")
    for idx in indexes:
        print(f"    {idx['name']}: {idx['column_names']}")
    print()
