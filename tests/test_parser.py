"""Tests for JSON response parser."""

import pytest
from app.collector.parser import parse_history_response, classify_size


def test_classify_size():
    """Test Small/Big size classification rules (0-4=SMALL, 5-9=BIG)."""
    assert classify_size(0) == "SMALL"
    assert classify_size(1) == "SMALL"
    assert classify_size(4) == "SMALL"
    assert classify_size(5) == "BIG"
    assert classify_size(9) == "BIG"

    with pytest.raises(ValueError):
        classify_size(10)


def test_parse_valid_response():
    """Test parsing a valid source response."""
    payload = {
        "data": {
            "list": [
                {
                    "issueNumber": "20260809100051300",
                    "number": "3",
                    "color": "green",
                    "premium": "3",
                    "sum": 0,
                },
                {
                    "issueNumber": "20260809100051299",
                    "number": "0",
                    "color": "red,violet",
                    "premium": "0",
                    "sum": 0,
                },
                {
                    "issueNumber": "20260809100051296",
                    "number": "6",
                    "color": "red",
                    "premium": "6",
                    "sum": 0,
                },
            ]
        },
        "code": 0,
        "msg": "Succeed",
    }

    results = parse_history_response(payload)
    assert len(results) == 3

    assert results[0].issue_id == "20260809100051300"
    assert results[0].result_number == 3
    assert results[0].calculated_size == "SMALL"
    assert results[0].source_color == "green"

    assert results[1].issue_id == "20260809100051299"
    assert results[1].calculated_size == "SMALL"
    assert results[1].source_color == "red,violet"

    assert results[2].calculated_size == "BIG"


def test_parse_empty_response():
    """Test parsing an empty list response."""
    payload = {"data": {"list": []}, "code": 0}
    results = parse_history_response(payload)
    assert results == []


def test_parse_malformed_response():
    """Test error on invalid API structure."""
    with pytest.raises(ValueError):
        parse_history_response({"code": 500, "msg": "Server Error"})

    with pytest.raises(ValueError):
        parse_history_response({})
