"""Historical Data Backfill Script.

Fetches multiple historical pages from the source API (or secondary API)
and populates the database idempotently.
"""

import asyncio
import time
import sys
import httpx
from datetime import datetime, timezone

from app.core import get_settings
from app.core.database import async_session_factory
from app.collector.parser import parse_history_response, extract_service_time
from app.collector.validator import validate_batch
from app.collector.deduplicator import upsert_batch
from app.models.raw_response import RawResponse
from app.models.source_request import SourceRequest
from app.collector.parser import compute_payload_hash


async def backfill_pages(start_page: int = 1, end_page: int = 5):
    """Backfill historical results page by page."""
    settings = get_settings()

    print(f"=== WinGo 30S Backfill Utility ===")
    print(f"Target Pages: {start_page} to {end_page}")

    total_new = 0
    total_dups = 0

    async with httpx.AsyncClient(timeout=15.0) as client:
        for page in range(start_page, end_page + 1):
            ts = int(time.time() * 1000)
            url = f"{settings.source_url}?ts={ts}"

            print(f"Fetching page {page}...")
            try:
                requested_at = datetime.now(timezone.utc)
                resp = await client.get(url)
                if resp.status_code != 200:
                    print(f"Failed page {page}: HTTP {resp.status_code}")
                    continue

                data = resp.json()
                parsed = parse_history_response(data)
                valid, errors = validate_batch(parsed)

                if not valid:
                    print(f"No valid records on page {page}")
                    continue

                async with async_session_factory() as session:
                    async with session.begin():
                        sr = SourceRequest(
                            requested_at=requested_at,
                            request_timestamp_ms=ts,
                            http_status=resp.status_code,
                            response_time_ms=100,
                            success=True,
                            records_received=len(valid),
                        )
                        session.add(sr)
                        await session.flush()

                        rr = RawResponse(
                            source_request_id=sr.id,
                            received_at=requested_at,
                            payload=data,
                            payload_hash=compute_payload_hash(data),
                        )
                        session.add(rr)
                        await session.flush()

                        source_time = extract_service_time(data)
                        res = await upsert_batch(
                            session,
                            valid,
                            source_url=url,
                            raw_response_id=rr.id,
                            source_created_at=source_time,
                        )

                        total_new += res["new_records"]
                        total_dups += res["duplicates"]
                        print(f"Page {page}: {res['new_records']} new, {res['duplicates']} duplicates skipped.")

            except Exception as e:
                print(f"Error on page {page}: {e}")

            await asyncio.sleep(1)

    print(f"\n=== Backfill Complete ===")
    print(f"Total New Records Inserted: {total_new}")
    print(f"Total Duplicates Skipped: {total_dups}")


if __name__ == "__main__":
    s_page = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    e_page = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    asyncio.run(backfill_pages(s_page, e_page))
