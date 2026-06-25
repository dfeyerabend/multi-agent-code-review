"""
Tests for the local submit_optimization tool in optimizer_tools.py.
"""

import json
from tools.optimizer_tools import run_optimizer_tool


_VALID_INPUT = {
    "fixes": [
        {
            "index":          0,
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
    assert "fixes" in result
    assert "summary" in result
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
    bad_input = {**_VALID_INPUT, "fixes": {"index": 0}}
    result = json.loads(run_optimizer_tool("submit_optimization", bad_input))

    assert result["status"] == "error"
    assert "list" in result["message"]


def test_submit_optimization_summary_not_a_string():
    """Returns error when summary is not a string."""
    bad_input = {**_VALID_INPUT, "summary": 123}
    result = json.loads(run_optimizer_tool("submit_optimization", bad_input))

    assert result["status"] == "error"
    assert "summary" in result["message"]


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
        "fixes": [
            _VALID_INPUT["fixes"][0],
            {**_VALID_INPUT["fixes"][0], "index": 1},
        ],
    }
    result = json.loads(run_optimizer_tool("submit_optimization", multi_input))

    assert result["status"] == "success"
    assert result["metadata"]["total_fixes"] == 2


def test_submit_optimization_missing_index():
    """Returns error naming the entry when index is missing."""
    bad_input = {**_VALID_INPUT, "fixes": [{k: v for k, v in _VALID_INPUT["fixes"][0].items() if k != "index"}]}
    result = json.loads(run_optimizer_tool("submit_optimization", bad_input))

    assert result["status"] == "error"
    assert "fixes[0]" in result["errors"][0]
    assert "index" in result["errors"][0]


def test_submit_optimization_index_not_an_int():
    """Returns error naming the entry when index is not an integer."""
    bad_input = {**_VALID_INPUT, "fixes": [{**_VALID_INPUT["fixes"][0], "index": "0"}]}
    result = json.loads(run_optimizer_tool("submit_optimization", bad_input))

    assert result["status"] == "error"
    assert "fixes[0]" in result["errors"][0]
    assert "index" in result["errors"][0]


def test_submit_optimization_index_bool_rejected():
    """Booleans are ints in Python — must be explicitly rejected as a valid index."""
    bad_input = {**_VALID_INPUT, "fixes": [{**_VALID_INPUT["fixes"][0], "index": True}]}
    result = json.loads(run_optimizer_tool("submit_optimization", bad_input))

    assert result["status"] == "error"
    assert "index" in result["errors"][0]


def test_submit_optimization_suggested_code_null_is_valid():
    """suggested_code may be null when the model genuinely cannot produce a fix."""
    null_input = {**_VALID_INPUT, "fixes": [{**_VALID_INPUT["fixes"][0], "suggested_code": None}]}
    result = json.loads(run_optimizer_tool("submit_optimization", null_input))

    assert result["status"] == "success"


def test_submit_optimization_grounded_in_not_list_of_strings():
    """Returns error when grounded_in contains a non-string entry."""
    bad_input = {**_VALID_INPUT, "fixes": [{**_VALID_INPUT["fixes"][0], "grounded_in": ["ok", 5]}]}
    result = json.loads(run_optimizer_tool("submit_optimization", bad_input))

    assert result["status"] == "error"
    assert "fixes[0]" in result["errors"][0]
    assert "grounded_in" in result["errors"][0]


def test_submit_optimization_no_longer_accepts_finding_rule_or_line():
    """finding_rule/finding_line are no longer validated or required — identity is attached in Python."""
    legacy_input = {
        **_VALID_INPUT,
        "fixes": [{"finding_rule": "B608", "finding_line": 12, "index": 0,
                   "suggested_code": "x", "explanation": "y", "grounded_in": []}],
    }
    result = json.loads(run_optimizer_tool("submit_optimization", legacy_input))

    assert result["status"] == "success"     # extra legacy fields are simply ignored, not rejected


def test_submit_optimization_unknown_tool_name():
    """Returns error JSON for an unrecognised tool name."""
    result = json.loads(run_optimizer_tool("nonexistent_tool", {}))

    assert result["status"] == "error"
    assert "nonexistent_tool" in result["message"]