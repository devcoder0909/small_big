"""Export game results to JSON."""

import asyncio
import json
import sys
from sqlalchemy import select, desc

from app.core.database import async_session_factory
from app.models.game_result import GameResult


async def export_json(
    output_file: str = "export_results.json",
    limit: int | None = None,
):
    """Export game results to JSON file."""
    query = select(GameResult).order_by(desc(GameResult.issue_id))
    if limit:
        query = query.limit(limit)

    async with async_session_factory() as session:
        result = await session.execute(query)
        rows = result.scalars().all()

    data = {
        "exported_at": __import__("datetime").datetime.now().isoformat(),
        "total_records": len(rows),
        "results": [
            {
                "issue_id": r.issue_id,
                "result_number": r.result_number,
                "calculated_size": r.calculated_size,
                "source_color": r.source_color,
                "first_observed_at": r.first_observed_at.isoformat() if r.first_observed_at else None,
                "last_observed_at": r.last_observed_at.isoformat() if r.last_observed_at else None,
            }
            for r in rows
        ],
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Exported {len(rows)} records to {output_file}")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "export_results.json"
    limit_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    asyncio.run(export_json(output, limit_arg))
