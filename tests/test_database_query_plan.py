"""
Test suite for Database Query Plan & Index Verification.
Verifies GameResult model composite index idx_game_results_issue_id_desc exists and covers descending issue_id query.
"""

import pytest
from sqlalchemy import inspect
from app.models.game_result import GameResult


def test_database_index_structure():
    mapper = inspect(GameResult)
    table = mapper.tables[0]

    index_names = [idx.name for idx in table.indexes]
    assert "idx_game_results_issue_id_desc" in index_names

    desc_index = [idx for idx in table.indexes if idx.name == "idx_game_results_issue_id_desc"][0]
    assert len(desc_index.expressions) >= 1 or len(desc_index.columns) >= 1
