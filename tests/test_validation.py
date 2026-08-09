"""Tests for validator module."""

from app.collector.validator import (
    validate_issue_id,
    validate_result_number,
    validate_size,
    validate_color,
    validate_parsed_result,
)
from app.collector.parser import ParsedGameResult


def test_validation_functions():
    assert validate_issue_id("20260809100051300") is True
    assert validate_issue_id("short") is False

    assert validate_result_number(0) is True
    assert validate_result_number(9) is True
    assert validate_result_number(10) is False

    assert validate_size("SMALL") is True
    assert validate_size("BIG") is True
    assert validate_size("MEDIUM") is False

    assert validate_color("green") is True
    assert validate_color("red,violet") is True
    assert validate_color("invalid") is False


def test_validate_parsed_result_mismatch():
    """Test size mismatch detection."""
    parsed = ParsedGameResult(
        issue_id="20260809100051300",
        result_number=2,  # 2 should be SMALL
        source_color="red",
        premium="2",
        sum_value=0,
        calculated_size="BIG",  # Intentional mismatch for test
        data_hash="abc",
    )

    is_valid, errors = validate_parsed_result(parsed)
    assert is_valid is False
    assert any("Size mismatch" in err for err in errors)
