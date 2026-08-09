"""Verification script — tests live WinGo 30S data fetch, parsing, validation, and DB persistence."""

import asyncio
import os
import sys

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath("."))

from app.core.database import async_session_factory, engine
from app.models.base import Base
from app.collector.client import SourceClient
from app.collector.parser import parse_history_response
from app.collector.validator import validate_batch
from app.collector.runner import CollectorRunner
from app.services.result_service import get_results


async def run_verification():
    print("=" * 65)
    print(" END-TO-END DATA FETCH & STORAGE VERIFICATION TEST")
    print("=" * 65)

    # Step 1: Create DB tables
    print("\n[Step 1] Initializing database schema...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("  -> Schema ready.")

    # Step 2: Fetch from Live WinGo Source API
    print("\n[Step 2] Fetching live draws from WinGo source API...")
    client = SourceClient()
    try:
        fetch_res = await client.fetch_history()
        print(f"  -> HTTP Status: {fetch_res.status_code}")
        print(f"  -> Response Time: {fetch_res.response_time_ms} ms")
        print(f"  -> Endpoint Used: {fetch_res.used_endpoint}")
        assert fetch_res.success, f"Fetch failed: {fetch_res.error_message}"
    finally:
        await client.close()

    # Step 3: Parse and Validate
    print("\n[Step 3] Parsing and validating payload...")
    parsed = parse_history_response(fetch_res.data)
    print(f"  -> Parsed Records Count: {len(parsed)}")
    assert len(parsed) > 0, "No records parsed from payload!"

    valid, errors = validate_batch(parsed)
    print(f"  -> Valid Records: {len(valid)} / {len(parsed)}")
    print(f"  -> Validation Errors: {len(errors)}")
    assert len(valid) > 0, "No valid records after validation!"
    print(f"  -> Latest Draw: Issue #{valid[0].issue_id} | Num: {valid[0].result_number} | Size: {valid[0].calculated_size}")

    # Step 4: Execute Full Collector Runner Cycle
    print("\n[Step 4] Executing CollectorRunner.run_single_cycle()...")
    runner = CollectorRunner()
    try:
        cycle_res = await runner.run_single_cycle()
        print(f"  -> Cycle Result: {cycle_res}")
        assert cycle_res["success"], "Collector cycle failed!"
    finally:
        await runner.shutdown()

    # Step 5: Verify Persistence in Database
    print("\n[Step 5] Verifying database records...")
    async with async_session_factory() as session:
        results_data = await get_results(session, limit=5)
        total_in_db = results_data["total"]
        results_list = results_data["results"]

        print(f"  -> Total Game Results stored in DB: {total_in_db}")
        assert total_in_db > 0, "Database game_results table is empty!"

        print("\n  -> Top 5 Stored Draws:")
        for r in results_list:
            print(f"     - Issue #{r['issue_id']} | Result Num: {r['result']} | Size: {r['size']} | Color: {r['color']}")

    print("\n" + "=" * 65)
    print(" SUCCESS: Live fetch & storage working 100%!")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(run_verification())
