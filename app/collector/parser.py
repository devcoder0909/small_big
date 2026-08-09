"""Parser module — converts source JSON into normalized Python objects."""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedGameResult:
    """A single parsed game result from the source API."""

    issue_id: str
    result_number: int
    source_color: str
    premium: str | None
    sum_value: int | None
    calculated_size: str  # "BIG" or "SMALL"
    data_hash: str


def classify_size(number: int) -> str:
    """
    Classify a result number as BIG or SMALL.

    WinGo 30S rules (verified from source):
        0-4 → SMALL
        5-9 → BIG

    Args:
        number: Result value (0-9)

    Returns:
        "BIG" or "SMALL"
    """
    if 0 <= number <= 4:
        return "SMALL"
    elif 5 <= number <= 9:
        return "BIG"
    else:
        raise ValueError(f"Invalid number for size classification: {number}")


def compute_data_hash(issue_id: str, number: int, color: str) -> str:
    """Compute SHA-256 hash of core result fields for integrity checking."""
    content = f"{issue_id}:{number}:{color}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def parse_history_response(raw_data: dict) -> list[ParsedGameResult]:
    """
    Parse the raw API JSON response into a list of ParsedGameResult objects.

    Expected structure:
    {
        "data": {
            "list": [
                {
                    "issueNumber": "20260809100051292",
                    "number": "6",
                    "color": "red",
                    "premium": "6",
                    "sum": 0
                }
            ],
            "pageNo": 1,
            "totalPage": 50,
            "totalCount": 500
        },
        "code": 0,
        "msg": "Succeed",
        "msgCode": 0,
        "serviceTime": 1786272357313
    }

    The color field can contain:
        "red", "green", "red,violet", "green,violet"

    Args:
        raw_data: Complete JSON response dict from the source API.

    Returns:
        List of ParsedGameResult objects, ordered as received (newest first).

    Raises:
        ValueError: If the response structure is unexpected.
    """
    # Validate response status
    code = raw_data.get("code")
    if code != 0:
        msg = raw_data.get("msg", "Unknown error")
        raise ValueError(f"API returned error code {code}: {msg}")

    # Extract data container
    data = raw_data.get("data")
    if not data or not isinstance(data, dict):
        raise ValueError("Missing or invalid 'data' field in response")

    # Extract history list
    history_list = data.get("list")
    if history_list is None:
        raise ValueError("Missing 'list' field in data")

    if not isinstance(history_list, list):
        raise ValueError(f"'list' field is not an array: {type(history_list)}")

    if len(history_list) == 0:
        logger.warning("empty_history_response")
        return []

    results = []
    parse_errors = 0

    for idx, item in enumerate(history_list):
        try:
            # Extract issue number (unique identifier)
            issue_id = item.get("issueNumber")
            if not issue_id:
                logger.error("parse_error", reason="missing issueNumber", index=idx)
                parse_errors += 1
                continue

            issue_id = str(issue_id).strip()

            # Extract and parse number (result value)
            number_str = item.get("number")
            if number_str is None:
                logger.error("parse_error", reason="missing number", issue_id=issue_id)
                parse_errors += 1
                continue

            try:
                result_number = int(number_str)
            except (ValueError, TypeError):
                logger.error(
                    "parse_error",
                    reason=f"invalid number: {number_str}",
                    issue_id=issue_id,
                )
                parse_errors += 1
                continue

            # Validate number range
            if not (0 <= result_number <= 9):
                logger.error(
                    "parse_error",
                    reason=f"number out of range: {result_number}",
                    issue_id=issue_id,
                )
                parse_errors += 1
                continue

            # Extract color
            source_color = str(item.get("color", "")).strip()

            # Extract optional fields
            premium = item.get("premium")
            if premium is not None:
                premium = str(premium).strip()

            sum_value = item.get("sum")
            if sum_value is not None:
                try:
                    sum_value = int(sum_value)
                except (ValueError, TypeError):
                    sum_value = None

            # Classify size
            calculated_size = classify_size(result_number)

            # Compute integrity hash
            data_hash = compute_data_hash(issue_id, result_number, source_color)

            results.append(ParsedGameResult(
                issue_id=issue_id,
                result_number=result_number,
                source_color=source_color,
                premium=premium,
                sum_value=sum_value,
                calculated_size=calculated_size,
                data_hash=data_hash,
            ))

        except Exception as e:
            logger.error("parse_error", reason=str(e), index=idx)
            parse_errors += 1

    if parse_errors > 0:
        logger.warning(
            "parse_completed_with_errors",
            total_items=len(history_list),
            successful=len(results),
            errors=parse_errors,
        )

    logger.info(
        "parse_complete",
        total_parsed=len(results),
        errors=parse_errors,
    )

    return results


def compute_payload_hash(payload: dict) -> str:
    """Compute SHA-256 hash of the entire payload for deduplication."""
    payload_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload_str.encode()).hexdigest()


def extract_pagination_info(raw_data: dict) -> dict:
    """Extract pagination info from the API response."""
    data = raw_data.get("data", {})
    return {
        "page_no": data.get("pageNo"),
        "total_page": data.get("totalPage"),
        "total_count": data.get("totalCount"),
    }


def extract_service_time(raw_data: dict) -> datetime | None:
    """Extract service timestamp from the API response."""
    service_time = raw_data.get("serviceTime")
    if service_time and isinstance(service_time, (int, float)):
        try:
            return datetime.fromtimestamp(service_time / 1000, tz=timezone.utc)
        except (ValueError, OSError):
            return None
    return None
