"""
Tests for the local submit_optimization tool in optimizer_tools.py.
"""

import json
from tools.optimizer_tools import run_optimizer_tool


_VALID_INPUT = {
    "fixes": [
        {
            "finding_rule":   "B608",
            "finding_line":   12,
            "suggested_code": "cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))",
            "explanation":    "Parameterized query eliminates SQL injection risk.",
            "grounded_in":    ["pyguide §3.10", "company_rules §1.3"],
        }
    ],
    "summary": "One SQL injection fix generated.",
}


def test_submit_optimization_valid_input():
    """Returns success status and correct metadata for a well-formed submission."""
    result = json.loads(run_optimizer_tool("submit_optimization", _VALID_INPUT))

    assert result["status"] == "success"
    assert "optimization_results" in result
    assert result["metadata"]["total_fixes"] == 1


def test_submit_optimization_missing_fixes():
    """Returns error and names the missing field when fixes is absent."""
    incomplete = {k: v for k, v in _VALID_INPUT.items() if k != "fixes"}
    result = json.loads(run_optimizer_tool("submit_optimization", incomplete))

    assert result["status"] == "error"
    assert "fixes" in result["message"]


def test_submit_optimization_missing_summary():
    """Returns error and names the missing field when summary is absent."""
    incomplete = {k: v for k, v in _VALID_INPUT.items() if k != "summary"}
    result = json.loads(run_optimizer_tool("submit_optimization", incomplete))

    assert result["status"] == "error"
    assert "summary" in result["message"]


def test_submit_optimization_fixes_not_a_list():
    """Returns error when fixes is a dict instead of a list."""
    bad_input = {**_VALID_INPUT, "fixes": {"finding_rule": "B608"}}
    result = json.loads(run_optimizer_tool("submit_optimization", bad_input))

    assert result["status"] == "error"
    assert "list" in result["message"]


def test_submit_optimization_empty_fixes_is_valid():
    """Empty fixes list is valid — clean code produces no fixes."""
    empty_input = {**_VALID_INPUT, "fixes": []}
    result = json.loads(run_optimizer_tool("submit_optimization", empty_input))

    assert result["status"] == "success"
    assert result["metadata"]["total_fixes"] == 0


def test_submit_optimization_multiple_fixes():
    """total_fixes count matches the number of entries submitted."""
    multi_input = {
        **_VALID_INPUT,
        "fixes": [_VALID_INPUT["fixes"][0], _VALID_INPUT["fixes"][0]],
    }
    result = json.loads(run_optimizer_tool("submit_optimization", multi_input))

    assert result["status"] == "success"
    assert result["metadata"]["total_fixes"] == 2


def test_submit_optimization_unknown_tool_name():
    """Returns error JSON for an unrecognised tool name."""
    result = json.loads(run_optimizer_tool("nonexistent_tool", {}))

    assert "error" in result