"""
1000 Sequential Period Pipeline Integrity Test.

Verifies:
1. 1000 sequential periods parsed, validated, and stored with 100% fidelity.
2. Zero missing, zero wrong periods, zero wrong BIG/SMALL classifications.
3. Duplicate feeding does not increase DB row count.
4. Conflicting duplicate feeding preserves original authoritative result.
"""

import pytest
from datetime import datetime, timezone
from sqlalchemy import select
from app.collector.parser import ParsedGameResult, classify_size, compute_data_hash
from app.collector.validator import validate_batch
from app.collector.deduplicator import upsert_batch, get_total_record_count
from app.models.game_result import GameResult


@pytest.mark.asyncio
async def test_1000_sequential_periods_pipeline(db_session):
    # 1. Generate 1000 sequential game raw structures
    base_issue = 202608100000
    raw_items = []
    expected_sizes = {}
    
    for i in range(1000):
        issue_id = str(base_issue + i)
        number = i % 10  # 0 to 9
        color = "green" if number % 2 != 0 else "red"
        calc_size = "SMALL" if number <= 4 else "BIG"
        expected_sizes[issue_id] = calc_size

        raw_items.append({
            "issueNumber": issue_id,
            "number": str(number),
            "color": color,
            "premium": str(number),
            "sum": 0
        })

    # 2. Simulate Parser & Validation
    parsed_results = []
    for item in raw_items:
        issue_id = item["issueNumber"]
        num = int(item["number"])
        c_size = classify_size(num)
        d_hash = compute_data_hash(issue_id, num, item["color"])
        parsed_results.append(
            ParsedGameResult(
                issue_id=issue_id,
                result_number=num,
                source_color=item["color"],
                premium=item["premium"],
                sum_value=0,
                calculated_size=c_size,
                data_hash=d_hash,
            )
        )

    assert len(parsed_results) == 1000

    valid_results, validation_errors = validate_batch(parsed_results)
    assert len(valid_results) == 1000
    assert len(validation_errors) == 0

    # 3. Store Batch
    now = datetime.now(timezone.utc)
    batch_res = await upsert_batch(
        db_session,
        valid_results,
        source_url="http://test.source",
        raw_response_id=None,
        source_created_at=now,
    )

    await db_session.commit()

    assert batch_res["new_records"] == 1000
    assert batch_res["duplicates"] == 0
    assert batch_res["errors"] == 0

    total_stored = await get_total_record_count(db_session)
    assert total_stored == 1000

    # Verify every record stored matches issue_id and BIG/SMALL classification exactly
    db_rows = await db_session.execute(select(GameResult))
    all_rows = db_rows.scalars().all()
    assert len(all_rows) == 1000

    wrong_period = 0
    wrong_size = 0
    for r in all_rows:
        if r.issue_id not in expected_sizes:
            wrong_period += 1
        elif r.calculated_size != expected_sizes[r.issue_id]:
            wrong_size += 1

    assert wrong_period == 0
    assert wrong_size == 0

    # 4. Test Duplicate Input (same period fed twice)
    dup_batch_res = await upsert_batch(
        db_session,
        valid_results,
        source_url="http://test.source",
        raw_response_id=None,
        source_created_at=now,
    )
    await db_session.commit()

    assert dup_batch_res["new_records"] == 0
    assert dup_batch_res["duplicates"] == 1000

    total_after_dup = await get_total_record_count(db_session)
    assert total_after_dup == 1000  # Database count MUST remain 1000!

    # 5. Test Conflicting Duplicate Values
    # Try inserting issue_id 202608100000 with opposite size ("BIG" instead of "SMALL")
    conflicting_parsed = ParsedGameResult(
        issue_id=str(base_issue),
        result_number=9,
        source_color="red",
        premium="9",
        sum_value=0,
        calculated_size="BIG",
        data_hash="conflicting_hash",
    )

    conflict_res = await upsert_batch(
        db_session,
        [conflicting_parsed],
        source_url="http://test.source",
        raw_response_id=None,
        source_created_at=now,
    )
    await db_session.commit()

    assert conflict_res["new_records"] == 0
    assert conflict_res["duplicates"] == 1

    # Verify original record unchanged
    row_0 = await db_session.execute(
        select(GameResult).where(GameResult.issue_id == str(base_issue))
    )
    r0 = row_0.scalar_one()
    assert r0.calculated_size == "SMALL"  # Authoritative original size remains SMALL!
    assert r0.result_number == 0          # Authoritative original number remains 0!
