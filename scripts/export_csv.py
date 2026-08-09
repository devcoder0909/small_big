"""Export game results to CSV."""

import asyncio
import csv
import sys
from datetime import datetime
from sqlalchemy import select, desc

from app.core import get_settings
from app.core.database import async_session_factory
from app.models.game_result import GameResult


async def export_csv(
    output_file: str = "export_results.csv",
    limit: int | None = None,
):
    """Export game results to CSV file."""
    query = select(GameResult).order_by(desc(GameResult.issue_id))
    if limit:
        query = query.limit(limit)

    async with async_session_factory() as session:
        result = await session.execute(query)
        rows = result.scalars().all()

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "issue_id", "result_number", "calculated_size",
            "source_color", "first_observed_at", "last_observed_at",
        ])
        for r in rows:
            writer.writerow([
                r.issue_id, r.result_number, r.calculated_size,
                r.source_color,
                r.first_observed_at.isoformat() if r.first_observed_at else "",
                r.last_observed_at.isoformat() if r.last_observed_at else "",
            ])

    print(f"Exported {len(rows)} records to {output_file}")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "export_results.csv"
    limit_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    asyncio.run(export_csv(output, limit_arg))
