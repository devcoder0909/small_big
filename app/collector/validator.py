"""Validator module — validates parsed game results."""

import re
from datetime import datetime
from app.collector.parser import ParsedGameResult
from app.core.logging import get_logger

logger = get_logger(__name__)

# Issue ID pattern: YYYYMMDD + game code + sequence number
ISSUE_ID_PATTERN = re.compile(r"^\d{17,}$")


def validate_issue_id(issue_id: str) -> bool:
    """Validate the issue identifier format."""
    if not issue_id:
        return False
    return bool(ISSUE_ID_PATTERN.match(issue_id))


def validate_result_number(number: int) -> bool:
    """Validate the result number is in range 0-9."""
    return isinstance(number, int) and 0 <= number <= 9


def validate_size(size: str) -> bool:
    """Validate the calculated size value."""
    return size in ("BIG", "SMALL")


def validate_color(color: str) -> bool:
    """Validate color field from source."""
    if not color:
        return False
    valid_colors = {"red", "green", "violet"}
    parts = [c.strip() for c in color.split(",")]
    return all(p in valid_colors for p in parts)


def validate_parsed_result(result: ParsedGameResult) -> tuple[bool, list[str]]:
    """
    Validate a single parsed game result.

    Args:
        result: ParsedGameResult to validate.

    Returns:
        Tuple of (is_valid, list_of_error_messages).
    """
    errors = []

    if not validate_issue_id(result.issue_id):
        errors.append(f"Invalid issue_id format: {result.issue_id}")

    if not validate_result_number(result.result_number):
        errors.append(f"Invalid result number: {result.result_number}")

    if not validate_size(result.calculated_size):
        errors.append(f"Invalid calculated size: {result.calculated_size}")

    if not validate_color(result.source_color):
        errors.append(f"Invalid color: {result.source_color}")

    # Cross-validate size classification
    if result.result_number is not None and 0 <= result.result_number <= 9:
        expected_size = "SMALL" if result.result_number <= 4 else "BIG"
        if result.calculated_size != expected_size:
            errors.append(
                f"Size mismatch: number={result.result_number}, "
                f"calculated={result.calculated_size}, expected={expected_size}"
            )

    is_valid = len(errors) == 0

    if not is_valid:
        logger.warning(
            "validation_failed",
            issue_id=result.issue_id,
            errors=errors,
        )

    return is_valid, errors


def validate_batch(results: list[ParsedGameResult]) -> tuple[list[ParsedGameResult], list[dict]]:
    """
    Validate a batch of parsed results.

    Args:
        results: List of ParsedGameResult objects.

    Returns:
        Tuple of (valid_results, validation_errors).
    """
    valid = []
    errors = []

    for result in results:
        is_valid, error_messages = validate_parsed_result(result)
        if is_valid:
            valid.append(result)
        else:
            errors.append({
                "issue_id": result.issue_id,
                "errors": error_messages,
            })

    if errors:
        logger.warning(
            "batch_validation",
            total=len(results),
            valid=len(valid),
            invalid=len(errors),
        )

    return valid, errors
